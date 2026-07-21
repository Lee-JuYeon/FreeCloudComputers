"""checkpoint — 재개의 린치핀. 공유 스토리지에 상태를 저장/복원.

기본 백엔드 = Hugging Face Hub(git-lfs, 무료, 비공개 가능, rate-limit 관대).
이 모듈은 두 곳에서 쓰인다:
  · 원격 노드의 entrypoint 안 — 시작 시 pull(), N스텝마다 push().
  · 오케스트레이터 로컬 — 제출 전 latest 확인, 완료 후 산출물 회수 보조.

의존: huggingface_hub (pip install 'freecloud[hub]'). 토큰=env HF_TOKEN.

원격 entrypoint에서 쓰는 최소 예:
    from freecloud.checkpoint import Checkpoint
    ck = Checkpoint.from_env()          # FREECLOUD_CKPT_REPO + HF_TOKEN
    ck.pull("./ckpt")                   # 이전 노드가 남긴 상태 복원(없으면 no-op)
    ...training...
    ck.push("./ckpt", step=step)        # 매 N스텝
"""
from __future__ import annotations

import os
from typing import Optional


def _is_auth_error(e: Exception) -> bool:
    """HF 인증/권한 오류인지(토큰 만료·무효·gated). 상태코드(401/403)도 확인."""
    s = str(e).lower()
    code = getattr(getattr(e, "response", None), "status_code", None)
    if code in (401, 403):
        return True
    return any(k in s for k in ("401", "403", "unauthorized", "authentication",
                                "gated repo", "invalid user token", "invalid credentials",
                                "invalid token", "permission denied"))


def _has_payload(d: str) -> bool:
    """dest 에 '진짜 체크포인트'가 있는가.

    ⚠️ `any(os.scandir(d))` 로는 안 된다. HF 는 저장소를 만들 때 `.gitattributes` 를 자동
    생성하고, snapshot_download 는 `.cache/` 를 남긴다. 그래서 **한 번도 push 한 적 없는
    빈 저장소도 항상 resume=True 로 보고**됐다(실측 2026-07-21). 학습 스크립트가 이 값으로
    scratch/resume 을 분기하면 처음 실행인데 재개 경로를 타게 된다.
    """
    for e in os.scandir(d):
        if e.name.startswith(".") or e.name == "README.md":
            continue  # HF 메타데이터(.gitattributes/.cache/…)는 페이로드가 아니다
        return True
    return False


class Checkpoint:
    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = True):
        self.repo_id = repo_id
        # env 우선(원격 노드는 resolved_env 로 주입받는다) → 없으면 로컬 시크릿 저장소.
        # ⚠️ 저장소 폴백이 없으면, `fcc login huggingface` 로 저장만 하고 env 를 안 쓰는
        #    사용자는 로컬에서 토큰을 못 찾는다(HF 캐시 미러에 우연히 기대게 됨).
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not self.token:
            try:
                from . import secrets
                self.token = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_TOKEN")
            except Exception:
                pass
        self.private = private

    @classmethod
    def from_env(cls) -> Optional["Checkpoint"]:
        repo = os.environ.get("FREECLOUD_CKPT_REPO")
        if not repo:
            return None
        return cls(repo)

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RuntimeError("pip install 'freecloud[hub]' 필요(huggingface_hub).") from e
        return HfApi(token=self.token)

    def ensure_repo(self) -> None:
        api = self._api()
        api.create_repo(self.repo_id, repo_type="model", private=self.private, exist_ok=True)

    def pull(self, dest: str) -> bool:
        """최신 체크포인트를 dest로 복원. 저장소가 비었으면 False(=처음 실행)."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
        os.makedirs(dest, exist_ok=True)
        try:
            snapshot_download(self.repo_id, repo_type="model", local_dir=dest,
                              token=self.token)
            has = _has_payload(dest)
            print(f"[checkpoint] pull {self.repo_id} → {dest} (resume={has})", flush=True)
            return has
        except RepositoryNotFoundError:
            print(f"[checkpoint] {self.repo_id} 없음 → 처음 실행(scratch).", flush=True)
            return False

    def push(self, src: str, step: Optional[int] = None) -> None:
        """src 폴더를 저장소로 업로드(덮어쓰기). N스텝마다 호출.

        토큰이 런 도중 만료/무효가 되면(401) 진행분을 잃지 않도록 명확히 실패시킨다:
        체크포인트는 이미 로컬 src에 있으므로 보존되며, 토큰 갱신 후 재개하면 이어서 push된다.
        메시지에 '401/토큰' 키워드를 담아 errors 플레이북이 AUTH로 분류(→ 사람 개입)하게 한다."""
        msg = f"checkpoint step={step}" if step is not None else "checkpoint"
        try:
            self.ensure_repo()
            api = self._api()
            api.upload_folder(folder_path=src, repo_id=self.repo_id, repo_type="model",
                              commit_message=msg)
        except Exception as e:
            if _is_auth_error(e):
                raise RuntimeError(
                    f"[checkpoint] HF 토큰 만료/무효(401 unauthorized) — 체크포인트는 로컬 "
                    f"'{src}'에 보존됨(유실 없음). `freecloud login huggingface`로 갱신 후 "
                    f"재개하면 이어서 push됨. 원인: {str(e)[:120]}") from e
            raise
        print(f"[checkpoint] push {src} → {self.repo_id} ({msg})", flush=True)

    def last_step(self) -> Optional[int]:
        """저장소 최신 커밋 메시지에서 step 파싱(진행 판정용). 없으면 None."""
        import re
        try:
            api = self._api()
            commits = api.list_repo_commits(self.repo_id, repo_type="model")
            for c in commits:
                m = re.search(r"step=(\d+)", c.title or "")
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        return None
