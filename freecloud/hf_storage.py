"""hf_storage — HF Hub 저장소 용량 위생(자가청소)의 단일 창구.

왜 한 곳에 모으나:
  · HF 로의 저장소 관련 호출(사용량 조회·고아 LFS 삭제·쓰기 프로브·히스토리 축약)을
    전부 여기로 모아, 테스트가 `_api` **한 곳만** 가짜로 바꾸면 네트워크 0 으로 전 경로를
    돌릴 수 있게 한다.

근본 배경(2026-10-31 실측, memory v8a-stuck-hf-storage-full):
  running 인데 진행 0 → ckpt push 가 403(private storage limit reached)으로 **저장만**
  실패하고 커널이 그 예외를 삼켰다. 토큰 문제가 아니었는데 403 이 AUTH 로 분류돼
  "토큰 확인" 이라는 틀린 조언이 나갔다. 여기서 403+용량 문구를 **auth 보다 먼저** 가려낸다.

핵심 사실(huggingface_hub 0.36.2):
  · 트리 메타데이터의 보존 식별자 = `BlobLfsInfo.sha256` (repo_info files_metadata=True).
  · 삭제 대상 식별자 = `LFSFileInfo.file_oid` (list_lfs_files). **oid 가 아니라 file_oid 가
    트리의 sha256 과 같은 값**이다 — oid 로 비교하면 보존집합과 영영 안 맞아 전부 삭제된다.
  · permanently_delete_lfs_files(rewrite_history=False) 로 과거 커밋 블롭을 물리 삭제.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

#: 프로브/고아 청소에 쓰는 전용 브랜치·경로(main 을 절대 건드리지 않기 위함).
PROBE_BRANCH = "fcc-probe"
PROBE_DIR = ".fcc-probe/"
PROBE_PATH = ".fcc-probe/probe.bin"
PROBE_SIZE = 11 * 1024 * 1024  # 11MB — HF LFS 임계(10MB) 위라 반드시 LFS 로 저장된다.

#: 403 응답에서 '계정 저장소 한도'를 뜻하는 문구들. 'storage limit' 이 셋을 다 포함한다.
_STORAGE_PHRASES = ("storage limit reached", "private storage limit", "storage limit")


def _api(tok: str | None):
    """HfApi 생성 — **모든 HF 호출의 단일 관문**. 테스트는 이 함수만 가짜로 바꾼다."""
    try:
        from huggingface_hub import HfApi
    except ImportError as e:  # pragma: no cover - 설치 환경에선 안 탐
        raise RuntimeError("pip install 'freecloud[hub]' 필요(huggingface_hub).") from e
    return HfApi(token=tok)


def is_storage_full(e: Exception) -> bool:
    """예외가 'HF 계정 저장소 한도 초과(403)' 인가 — **auth 판정보다 먼저** 물어야 한다.

    403 은 '토큰 권한 없음' 과 '저장소 가득참' 둘 다에 쓰인다. 둘을 구분하려면 상태코드
    403 **그리고** 저장소 한도 문구가 함께 있어야 한다. 문구가 없으면 일반 403(→ AUTH).
    """
    s = str(e).lower()
    code = getattr(getattr(e, "response", None), "status_code", None)
    is_403 = code == 403 or "403" in s
    return is_403 and any(p in s for p in _STORAGE_PHRASES)


def human(n: int) -> str:
    """바이트를 사람이 읽는 단위로(로그/CLI 출력용)."""
    f = float(n or 0)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if f < 1024 or unit == "TiB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}TiB"


# ── 보존집합(fail-closed 의 핵심) ─────────────────────────────────────────────
def _lfs_sha_of_revision(api, repo: str, revision: str, tok: str | None) -> set[str]:
    """한 revision(브랜치 head·태그·커밋 sha)의 트리에 실린 LFS 파일 sha256 집합."""
    info = api.repo_info(repo, repo_type="model", revision=revision,
                         files_metadata=True, token=tok)
    out: set[str] = set()
    for s in getattr(info, "siblings", None) or []:
        lfs = getattr(s, "lfs", None)
        if not lfs:
            continue
        sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        if sha:
            out.add(sha)
    return out


def _keep_shas(api, repo: str, tok: str | None, keep_revisions) -> set[str]:
    """보존할 LFS sha256 = (모든 브랜치 head + 모든 태그 + keep_revisions) 트리의 합집합.

    조회가 하나라도 실패하면 예외가 그대로 올라간다 — 호출부(prune)가 그걸 잡아
    **fail-closed(삭제 0)** 로 처리한다. 반쪽 보존집합으로 삭제하면 산 체크포인트가 날아간다.
    """
    refs = api.list_repo_refs(repo, repo_type="model", token=tok)
    revs: list[str] = []
    for r in list(getattr(refs, "branches", None) or []) + list(getattr(refs, "tags", None) or []):
        rev = getattr(r, "target_commit", None) or getattr(r, "name", None)
        if rev:
            revs.append(rev)
    for rev in (keep_revisions or ()):
        if rev:
            revs.append(rev)
    keep: set[str] = set()
    for rev in revs:
        keep |= _lfs_sha_of_revision(api, repo, rev, tok)
    return keep


# ── 사용량 리포트(읽기전용) ───────────────────────────────────────────────────
@dataclass
class StorageReport:
    repo: str
    used_bytes: int = 0        # 전체 LFS 합(사용량 근사)
    lfs_count: int = 0
    keep_count: int = 0        # 보존(산 트리가 참조) 블롭 수
    keep_bytes: int = 0
    orphan_count: int = 0      # 어느 산 트리도 참조하지 않는 블롭 수
    orphan_bytes: int = 0


def storage_report(repo: str, tok: str | None, keep_revisions=()) -> StorageReport:
    """저장소의 LFS 사용량·보존/고아 현황을 읽어 리포트(쓰기·삭제 없음)."""
    api = _api(tok)
    keep = _keep_shas(api, repo, tok, keep_revisions)
    rep = StorageReport(repo=repo, keep_count=0)
    for b in api.list_lfs_files(repo, repo_type="model", token=tok):
        size = int(getattr(b, "size", 0) or 0)
        rep.used_bytes += size
        rep.lfs_count += 1
        if getattr(b, "file_oid", None) in keep:
            rep.keep_count += 1
            rep.keep_bytes += size
        else:
            rep.orphan_count += 1
            rep.orphan_bytes += size
    return rep


# ── 고아 LFS 청소(fail-closed) ────────────────────────────────────────────────
@dataclass
class PruneResult:
    repo: str
    dry_run: bool = False
    deleted_count: int = 0
    deleted_bytes: int = 0
    kept_count: int = 0
    skipped_recent: int = 0
    aborted: bool = False
    detail: str = ""


def _age_s(pushed_at, now: datetime) -> float:
    """블롭 push 이후 경과 초. tz 없는 값은 UTC 로 간주. 모르면 매우 큰 값(=오래됨)."""
    if not isinstance(pushed_at, datetime):
        return float("inf")
    pa = pushed_at if pushed_at.tzinfo else pushed_at.replace(tzinfo=timezone.utc)
    return (now - pa).total_seconds()


def prune_orphans(repo: str, tok: str | None, *, min_age_s: int = 3600,
                  batch: int = 1000, dry_run: bool = False, keep_revisions=()) -> PruneResult:
    """보존집합에 없는 고아 LFS 블롭을 영구삭제. **fail-closed**.

    보존집합 = 모든 브랜치·태그(+keep_revisions) 트리의 sha256. 이게 비었거나 트리 조회가
    실패하면 **삭제 0** — 반쪽 집합으로 삭제하면 산 체크포인트가 날아가기 때문.
    `min_age_s` 이내에 push 된 블롭은 보존(지금 돌고 있는 커널이 막 올린 것일 수 있다).
    단 `.fcc-probe/` 프로브 블롭은 나이와 무관하게 청소 대상(우리가 만든 쓰레기).

    ★비교 키는 `LFSFileInfo.file_oid` (= 트리 sha256). oid 로 비교하면 전부 삭제된다.
    """
    api = _api(tok)
    try:
        keep = _keep_shas(api, repo, tok, keep_revisions)
    except Exception as e:
        return PruneResult(repo, dry_run, aborted=True,
                           detail=f"보존집합 조회 실패 → fail-closed(삭제 0): {str(e)[:80]}")
    if not keep:
        return PruneResult(repo, dry_run, aborted=True,
                           detail="보존집합이 비었음 → fail-closed(삭제 0). "
                                  "브랜치/태그/트리를 못 읽었거나 LFS 가 하나도 안 걸림.")

    now = datetime.now(timezone.utc)
    candidates = []
    skipped_recent = 0
    for b in api.list_lfs_files(repo, repo_type="model", token=tok):
        if getattr(b, "file_oid", None) in keep:
            continue
        is_probe = (getattr(b, "filename", "") or "").startswith(PROBE_DIR)
        if not is_probe and _age_s(getattr(b, "pushed_at", None), now) < min_age_s:
            skipped_recent += 1
            continue
        candidates.append(b)

    deleted_bytes = sum(int(getattr(b, "size", 0) or 0) for b in candidates)
    if dry_run:
        return PruneResult(repo, True, len(candidates), deleted_bytes,
                           kept_count=len(keep), skipped_recent=skipped_recent,
                           detail="dry-run(삭제 없음)")

    for i in range(0, len(candidates), batch):
        chunk = candidates[i:i + batch]
        api.permanently_delete_lfs_files(repo, chunk, rewrite_history=False,
                                         repo_type="model", token=tok)

    # 삭제 후 보존 sha 가 전부 남아있는지 재확인 — 하나라도 사라졌으면 비상.
    remaining = {getattr(b, "file_oid", None)
                 for b in api.list_lfs_files(repo, repo_type="model", token=tok)}
    missing = keep - remaining
    if missing:
        return PruneResult(repo, False, len(candidates), deleted_bytes,
                           kept_count=len(keep), skipped_recent=skipped_recent,
                           aborted=True,
                           detail=f"⚠ 삭제 후 보존 sha {len(missing)}개가 사라졌다 — 검증 실패!")
    return PruneResult(repo, False, len(candidates), deleted_bytes,
                       kept_count=len(keep), skipped_recent=skipped_recent,
                       detail="완료")


# ── 쓰기 프로브(용량/권한 실측) ───────────────────────────────────────────────
def _cleanup_probe(api, repo: str, tok: str | None, sha: str) -> None:
    """프로브 브랜치 + 로컬 sha 와 일치하는 블롭을 영구삭제. 실패는 경고만."""
    try:
        api.delete_branch(repo, branch=PROBE_BRANCH, repo_type="model", token=tok)
    except Exception as e:
        print(f"(경고) 프로브 브랜치 정리 실패: {str(e)[:80]}", flush=True)
    try:
        junk = [b for b in api.list_lfs_files(repo, repo_type="model", token=tok)
                if getattr(b, "file_oid", None) == sha]
        if junk:
            api.permanently_delete_lfs_files(repo, junk, rewrite_history=False,
                                             repo_type="model", token=tok)
    except Exception as e:
        print(f"(경고) 프로브 블롭 정리 실패: {str(e)[:80]}", flush=True)


def write_probe(repo: str, tok: str | None, size: int = PROBE_SIZE) -> tuple[bool, str]:
    """실제 쓰기로 용량/권한을 실측. **main 이 아니라 임시 브랜치 fcc-probe 에 커밋**한다.

    main 에 커밋하면 무진행 워치독이 그 커밋을 '진행 신호'로 오인한다. 그래서 전용 브랜치에
    쓰고, finally 에서 브랜치 삭제 + 로컬 sha256 과 같은 블롭을 영구삭제해 흔적을 없앤다.
    업로드가 403(용량/권한)으로 터지면 예외가 그대로 올라간다 — 호출부가 is_storage_full /
    auth 로 분류한다. 정리 실패는 경고만(쓰기 성공 판정을 뒤집지 않는다).
    """
    api = _api(tok)
    content = os.urandom(size)
    sha = hashlib.sha256(content).hexdigest()
    try:
        api.create_branch(repo, branch=PROBE_BRANCH, repo_type="model",
                          token=tok, exist_ok=True)
        api.upload_file(path_or_fileobj=content, path_in_repo=PROBE_PATH, repo_id=repo,
                        repo_type="model", revision=PROBE_BRANCH, token=tok,
                        commit_message="fcc write probe (temp branch)")
    finally:
        _cleanup_probe(api, repo, tok, sha)
    return True, f"쓰기 가능 확인(프로브 {human(size)}, 임시 브랜치, {repo})"


# ── 히스토리 축약(수동 CLI 전용) ──────────────────────────────────────────────
def super_squash(repo: str, tok: str | None, message: str | None = None) -> None:
    """main 히스토리를 1커밋으로 축약(과거 커밋의 블롭을 고아로 만들어 prune 가 치우게)."""
    api = _api(tok)
    api.super_squash_history(repo, branch="main", repo_type="model",
                             commit_message=message or "fcc squash", token=tok)


# ── 정책별 위생 1줄 ───────────────────────────────────────────────────────────
def hygiene(repo: str, tok: str | None, *, policy: str = "warn", keep_revisions=()) -> str:
    """프리플라이트에서 정책(auto|warn|off)에 따라 위생을 수행하고 **한 줄 요약**을 돌려준다.

    · off  — 아무것도 안 함.
    · warn — 고아만 세어 경고(삭제 안 함). 기본값(사용자 결정: 삭제는 사람/일임).
    · auto — 고아를 prune(삭제). **auto-squash 는 하지 않는다**(squash 는 수동 CLI 전용).
    """
    if policy == "off":
        return "위생 off(점검 생략)"
    try:
        rep = storage_report(repo, tok, keep_revisions=keep_revisions)
    except Exception as e:
        return f"위생 점검 실패(무시): {str(e)[:60]}"
    if rep.orphan_count == 0:
        return f"고아 LFS 0 (LFS 사용 {human(rep.used_bytes)})"
    if policy == "auto":
        pr = prune_orphans(repo, tok, keep_revisions=keep_revisions)
        if pr.aborted:
            return f"auto 청소 중단(fail-closed): {pr.detail}"
        return f"auto 청소: 고아 {pr.deleted_count}개/{human(pr.deleted_bytes)} 삭제"
    # warn
    return (f"⚠ 고아 LFS {rep.orphan_count}개/{human(rep.orphan_bytes)} 누적 — 삭제 안 함(warn). "
            f"`fcc ckpt squash {repo}` 로 청소하세요.")
