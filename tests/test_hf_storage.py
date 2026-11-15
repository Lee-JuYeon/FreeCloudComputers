"""HF 저장소 용량 자가청소(E) 회귀 테스트 — 네트워크 0, 가짜 HfApi 한 곳만.

전부 `hf_storage._api` 를 FakeApi 로 바꿔(=`@pytest.mark.allow_hf_storage`) 돌린다.
conftest 의 autouse 블록이 그 표식이 없으면 `_api` 를 터뜨려 네트워크 접근 0 을 보장한다.
"""
import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from freecloud import hf_storage, errors, auth, orchestrator, registry, state, checkpoint
from freecloud.job import Job
from freecloud.providers.base import Provider, Probe, RunResult


# ── 가짜 HF 데이터 모델 ───────────────────────────────────────────────────────
class _Blob:
    """LFSFileInfo 흉내. ★oid 와 file_oid 를 일부러 다르게 둬 '잘못된 키 비교' 변이를 잡는다."""
    def __init__(self, file_oid, *, size=1000, filename="model.safetensors",
                 pushed_at=None, ref=None):
        self.file_oid = file_oid
        self.oid = "OID_" + file_oid           # 트리 sha 와 절대 안 맞는 값(변이 킬러)
        self.filename = filename
        self.size = size
        self.pushed_at = pushed_at or datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.ref = ref


class _Ref:
    def __init__(self, name, target_commit):
        self.name = name
        self.target_commit = target_commit


class _Refs:
    def __init__(self, branches, tags, pull_requests=None):
        self.branches = branches
        self.tags = tags
        self.pull_requests = pull_requests or []


class _Sibling:
    def __init__(self, sha256):
        self.lfs = {"sha256": sha256} if sha256 else None


class _Info:
    def __init__(self, siblings):
        self.siblings = siblings


class FakeApi:
    def __init__(self, *, tree=None, lfs=None, refs=None):
        self.tree = tree or {"C_MAIN": []}            # revision -> [sha256,...]
        self.lfs = list(lfs or [])
        self.refs = refs or _Refs([_Ref("main", "C_MAIN")], [])
        self.deleted = []            # permanently_delete_lfs_files 호출별 chunk
        self.branches_created = []
        self.branches_deleted = []
        self.uploads = []            # (revision, path)
        self.commits = {}            # branch -> commit count
        self.squashed = []
        self.events = []             # 순서 검증용('delete-orphan'/'upload-probe')
        self.fail_delete_branch = False

    def repo_info(self, repo, *, repo_type=None, revision=None, files_metadata=False, token=None):
        return _Info([_Sibling(s) for s in self.tree.get(revision, [])])

    def list_repo_refs(self, repo, *, repo_type=None, token=None, include_pull_requests=False):
        self.refs_pr_flag = include_pull_requests
        return self.refs

    def list_lfs_files(self, repo, *, repo_type=None, token=None):
        return list(self.lfs)

    def permanently_delete_lfs_files(self, repo, lfs_files, *, rewrite_history=True,
                                     repo_type=None, token=None):
        assert rewrite_history is False, "rewrite_history=False 이어야 한다(히스토리 보존)"
        chunk = list(lfs_files)
        self.deleted.append(chunk)
        for b in chunk:
            tag = "probe" if (b.filename or "").startswith(hf_storage.PROBE_DIR) else "orphan"
            self.events.append(f"delete-{tag}")
            if b in self.lfs:
                self.lfs.remove(b)

    def create_branch(self, repo, *, branch, revision=None, token=None, repo_type=None, exist_ok=False):
        self.branches_created.append(branch)

    def delete_branch(self, repo, *, branch, token=None, repo_type=None):
        if self.fail_delete_branch:
            raise RuntimeError("delete_branch 실패(시뮬)")
        self.branches_deleted.append(branch)

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, token=None,
                    repo_type=None, revision=None, commit_message=None, **kw):
        self.uploads.append((revision, path_in_repo))
        self.commits[revision] = self.commits.get(revision, 0) + 1
        self.events.append("upload-probe")
        if isinstance(path_or_fileobj, (bytes, bytearray)):
            sha = hashlib.sha256(bytes(path_or_fileobj)).hexdigest()
            self.lfs.append(_Blob(sha, filename=path_in_repo, size=len(path_or_fileobj),
                                  pushed_at=datetime.now(timezone.utc)))

    def super_squash_history(self, repo, *, branch=None, commit_message=None, repo_type=None, token=None):
        self.squashed.append(branch)


def _wire_api(monkeypatch, api):
    monkeypatch.setattr(hf_storage, "_api", lambda tok=None: api)
    return api


class _Resp:
    def __init__(self, code):
        self.status_code = code


def _storage_full_exc():
    e = RuntimeError("403 Client Error: Forbidden — private storage limit reached")
    e.response = _Resp(403)
    return e


def _auth_exc():
    e = RuntimeError("401 Client Error: Unauthorized: invalid token")
    e.response = _Resp(401)
    return e


# ── is_storage_full / classify ───────────────────────────────────────────────
def test_is_storage_full_true_only_with_403_and_phrase():
    assert hf_storage.is_storage_full(_storage_full_exc())
    assert hf_storage.is_storage_full(RuntimeError("403: storage limit reached"))
    # 403 이지만 용량 문구 없음 → auth 계열(여기선 False)
    assert not hf_storage.is_storage_full(RuntimeError("403 permission denied"))
    # 용량 문구는 있으나 403 아님 → False
    assert not hf_storage.is_storage_full(RuntimeError("500 storage limit reached"))
    assert not hf_storage.is_storage_full(_auth_exc())


def test_classify_storage_full_is_abort_not_auth():
    d = errors.classify("403 Client Error: private storage limit reached")
    assert d.error_class == "HF_STORAGE_FULL" and d.action == errors.ABORT and d.cooldown_s == 0


def test_classify_plain_403_is_auth():
    d = errors.classify("403 Forbidden: permission denied")
    assert d.error_class == "AUTH"


def test_classify_hf_storage_full_token():
    assert errors.classify("[checkpoint] hf_storage_full ...").error_class == "HF_STORAGE_FULL"


# ── write_probe ──────────────────────────────────────────────────────────────
@pytest.mark.allow_hf_storage
def test_write_probe_success_cleans_up_and_no_main_commit(monkeypatch):
    api = _wire_api(monkeypatch, FakeApi())
    ok, detail = hf_storage.write_probe("user/ckpt", "hf_x", size=1024)
    assert ok
    assert hf_storage.PROBE_BRANCH in api.branches_created
    # 모든 업로드는 임시 브랜치에만 — main 커밋 0회
    assert all(rev == hf_storage.PROBE_BRANCH for rev, _ in api.uploads)
    assert api.commits.get("main", 0) == 0
    # 정리: 브랜치 삭제 + 프로브 블롭 영구삭제 → lfs 비어있어야
    assert api.branches_deleted == [hf_storage.PROBE_BRANCH]
    assert api.lfs == []
    assert any(chunk for chunk in api.deleted)


@pytest.mark.allow_hf_storage
def test_write_probe_cleanup_failure_still_ok(monkeypatch):
    api = FakeApi()
    api.fail_delete_branch = True       # 정리(브랜치 삭제) 실패해도 쓰기 성공 판정은 유지
    _wire_api(monkeypatch, api)
    ok, _ = hf_storage.write_probe("user/ckpt", "hf_x", size=1024)
    assert ok


@pytest.mark.allow_hf_storage
def test_write_probe_storage_full_propagates(monkeypatch):
    api = FakeApi()
    def _boom(**kw):
        raise _storage_full_exc()
    api.upload_file = _boom
    _wire_api(monkeypatch, api)
    with pytest.raises(Exception) as ei:
        hf_storage.write_probe("user/ckpt", "hf_x", size=1024)
    assert hf_storage.is_storage_full(ei.value)
    assert api.branches_deleted == [hf_storage.PROBE_BRANCH]   # finally 정리 돎


# ── verify_hf_write 분류(용량 vs 토큰) ────────────────────────────────────────
def _noop_ensure(monkeypatch):
    monkeypatch.setattr(checkpoint.Checkpoint, "ensure_repo", lambda self: None)


def test_verify_write_storage_full_prefix_not_auth(monkeypatch):
    _noop_ensure(monkeypatch)
    monkeypatch.setattr(hf_storage, "write_probe",
                        lambda r, t, size=hf_storage.PROBE_SIZE: (_ for _ in ()).throw(_storage_full_exc()))
    ok, detail = auth.verify_hf_write("user/ckpt", "hf_x", hygiene="off")
    assert not ok and detail.startswith("HF_STORAGE_FULL")
    # 오케스트레이터가 접두사로 가르는 error_class 가 AUTH 가 아니어야
    assert errors.classify(detail).error_class == "HF_STORAGE_FULL"


def test_verify_write_401_token_advice(monkeypatch):
    _noop_ensure(monkeypatch)
    monkeypatch.setattr(hf_storage, "write_probe",
                        lambda r, t, size=hf_storage.PROBE_SIZE: (_ for _ in ()).throw(_auth_exc()))
    ok, detail = auth.verify_hf_write("user/ckpt", "hf_x", hygiene="off")
    assert not ok and "쓰기 권한" in detail and not detail.startswith("HF_STORAGE_FULL")


def test_verify_write_probe_off_skips_probe(monkeypatch):
    _noop_ensure(monkeypatch)
    monkeypatch.setenv("FCC_HF_PROBE", "0")
    called = {"probe": False}
    monkeypatch.setattr(hf_storage, "write_probe",
                        lambda *a, **k: called.__setitem__("probe", True) or (True, "x"))
    ok, detail = auth.verify_hf_write("user/ckpt", "hf_x", hygiene="off")
    assert ok and not called["probe"] and "프로브 생략" in detail


# ── hygiene: warn 은 삭제0+경고 / auto 는 프로브보다 먼저 삭제 ────────────────
@pytest.mark.allow_hf_storage
def test_verify_warn_does_not_delete_orphans(monkeypatch):
    _noop_ensure(monkeypatch)
    old = datetime.now(timezone.utc) - timedelta(days=5)
    orphan = _Blob("ORPH", filename="old.bin", pushed_at=old)
    api = FakeApi(tree={"C_MAIN": ["KEEP"]},
                  lfs=[_Blob("KEEP"), orphan])
    _wire_api(monkeypatch, api)
    ok, detail = auth.verify_hf_write("user/ckpt", "hf_x", hygiene="warn")
    assert ok
    assert "고아" in detail and "warn" in detail
    assert orphan in api.lfs                      # 삭제 안 됨
    assert all(e != "delete-orphan" for e in api.events)


@pytest.mark.allow_hf_storage
def test_verify_auto_prunes_before_probe(monkeypatch):
    _noop_ensure(monkeypatch)
    old = datetime.now(timezone.utc) - timedelta(days=5)
    orphan = _Blob("ORPH", filename="old.bin", pushed_at=old)
    api = FakeApi(tree={"C_MAIN": ["KEEP"]},
                  lfs=[_Blob("KEEP"), orphan])
    _wire_api(monkeypatch, api)
    ok, detail = auth.verify_hf_write("user/ckpt", "hf_x", hygiene="auto", keep_revisions=())
    assert ok
    assert "delete-orphan" in api.events and "upload-probe" in api.events
    assert api.events.index("delete-orphan") < api.events.index("upload-probe")


# ── prune_orphans ────────────────────────────────────────────────────────────
@pytest.mark.allow_hf_storage
def test_prune_uses_file_oid_not_oid(monkeypatch):
    """보존 비교는 file_oid(=트리 sha). oid 로 비교하면 산 블롭까지 다 지운다(변이 킬러)."""
    old = datetime.now(timezone.utc) - timedelta(days=5)
    keep = _Blob("KEEP", pushed_at=old)
    orphan = _Blob("ORPH", filename="o.bin", pushed_at=old)
    api = FakeApi(tree={"C_MAIN": ["KEEP"]}, lfs=[keep, orphan])
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x")
    assert not res.aborted and res.deleted_count == 1
    assert keep in api.lfs and orphan not in api.lfs


@pytest.mark.allow_hf_storage
def test_prune_preserves_tags_keeprevs_and_recent(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=5)
    now = datetime.now(timezone.utc)
    km, kt, kr = _Blob("KM", pushed_at=old), _Blob("KT", pushed_at=old), _Blob("KR", pushed_at=old)
    orph_old = _Blob("OLD", filename="old.bin", pushed_at=old)
    orph_recent = _Blob("NEW", filename="new.bin", pushed_at=now)
    api = FakeApi(
        tree={"C_MAIN": ["KM"], "C_V1": ["KT"], "C_KR": ["KR"]},
        refs=_Refs([_Ref("main", "C_MAIN")], [_Ref("v1", "C_V1")]),
        lfs=[km, kt, kr, orph_old, orph_recent])
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x", keep_revisions=["C_KR"])
    assert res.deleted_count == 1 and res.skipped_recent == 1
    for b in (km, kt, kr, orph_recent):
        assert b in api.lfs
    assert orph_old not in api.lfs


@pytest.mark.allow_hf_storage
def test_prune_probe_blob_deleted_even_if_recent(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=5)
    now = datetime.now(timezone.utc)
    probe = _Blob("PRB", filename=hf_storage.PROBE_PATH, pushed_at=now)   # 최근이지만 우리 쓰레기
    api = FakeApi(tree={"C_MAIN": ["KEEP"]}, lfs=[_Blob("KEEP", pushed_at=old), probe])
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x")
    assert probe not in api.lfs and res.deleted_count == 1


@pytest.mark.allow_hf_storage
def test_prune_fail_closed_on_empty_keepset(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=5)
    orphan = _Blob("ORPH", pushed_at=old)
    api = FakeApi(tree={"C_MAIN": []}, lfs=[orphan])   # 보존집합 비어있음
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x")
    assert res.aborted and res.deleted_count == 0
    assert api.deleted == [] and orphan in api.lfs


@pytest.mark.allow_hf_storage
def test_prune_fail_closed_on_tree_lookup_error(monkeypatch):
    api = FakeApi(lfs=[_Blob("ORPH")])
    def _boom(*a, **k):
        raise RuntimeError("refs 조회 실패")
    api.list_repo_refs = _boom
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x")
    assert res.aborted and res.deleted_count == 0 and api.deleted == []


@pytest.mark.allow_hf_storage
def test_prune_batches(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=5)
    orphans = [_Blob(f"O{i}", filename=f"o{i}.bin", pushed_at=old) for i in range(5)]
    api = FakeApi(tree={"C_MAIN": ["KEEP"]}, lfs=[_Blob("KEEP", pushed_at=old), *orphans])
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x", batch=2)
    assert res.deleted_count == 5
    assert [len(c) for c in api.deleted] == [2, 2, 1]


@pytest.mark.allow_hf_storage
def test_prune_dry_run_writes_nothing(monkeypatch):
    old = datetime.now(timezone.utc) - timedelta(days=5)
    orphan = _Blob("ORPH", pushed_at=old)
    api = FakeApi(tree={"C_MAIN": ["KEEP"]}, lfs=[_Blob("KEEP", pushed_at=old), orphan])
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x", dry_run=True)
    assert res.dry_run and res.deleted_count == 1
    assert api.deleted == [] and orphan in api.lfs   # 실제 삭제 0


# ── item1: PR ref 보존(refs/pr/* 에만 실린 LFS 는 고아가 아니다) ────────────────
@pytest.mark.allow_hf_storage
def test_prune_keeps_pr_only_blob(monkeypatch):
    """PR head 트리에만 있는 블롭은 어느 브랜치/태그에도 없어도 보존돼야 한다(고아 아님)."""
    old = datetime.now(timezone.utc) - timedelta(days=5)
    pr_blob = _Blob("PRB", filename="pr.safetensors", pushed_at=old)
    orphan = _Blob("ORPH", filename="o.bin", pushed_at=old)
    api = FakeApi(
        tree={"C_MAIN": ["KEEP"], "C_PR1": ["PRB"]},
        refs=_Refs([_Ref("main", "C_MAIN")], [],
                   pull_requests=[_Ref("refs/pr/1", "C_PR1")]),
        lfs=[_Blob("KEEP", pushed_at=old), pr_blob, orphan])
    _wire_api(monkeypatch, api)
    res = hf_storage.prune_orphans("user/ckpt", "hf_x")
    assert api.refs_pr_flag is True            # include_pull_requests 를 실제로 넘겼다
    assert res.deleted_count == 1              # orphan 만 삭제
    assert pr_blob in api.lfs and orphan not in api.lfs


# ── orchestrator 프리플라이트 ─────────────────────────────────────────────────
class Fake(Provider):
    def __init__(self, name="fake"):
        self.name = name
        self.capabilities = {"gpu": "T4", "vram_gb": 16, "headless": True}
        self.calls = 0

    def probe(self):
        return Probe(True, "")

    def run(self, job):
        self.calls += 1
        return RunResult(True, "done", "ok")


def _wire_prov(monkeypatch, fake):
    monkeypatch.setattr(registry, "get", lambda n: fake)
    monkeypatch.setattr(registry, "DEFAULT_ORDER", [fake.name])
    state.clear_cooldown(fake.name)


def test_orch_storage_full_run_0_and_error_class(monkeypatch):
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (True, "ok"))
    monkeypatch.setattr(auth, "verify_hf_write",
                        lambda r, t=None, **kw: (False, "HF_STORAGE_FULL: 계정 저장소 한도 초과"))
    fake = Fake()
    _wire_prov(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", checkpoint_repo="user/ckpt")
    res = orchestrator.run(job, max_rounds=1)
    assert not res["ok"] and fake.calls == 0
    ec = [a.get("error_class") for a in res["attempts"]]
    assert "HF_STORAGE_FULL" in ec and "AUTH" not in ec


# ── CLI: ckpt squash ──────────────────────────────────────────────────────────
import argparse  # noqa: E402

from freecloud import cli  # noqa: E402


def _squash_args(**kw):
    base = dict(repo="user/ckpt", dry_run=False, yes=False, no_squash=False,
                min_age_min=60, keep_revision=None)
    base.update(kw)
    return argparse.Namespace(**base)


def _fake_prune(monkeypatch, **res_kw):
    calls = []
    def _p(repo, tok, *, min_age_s=3600, batch=1000, dry_run=False, keep_revisions=()):
        calls.append(dict(dry_run=dry_run))
        return hf_storage.PruneResult(repo, dry_run=dry_run, deleted_count=3,
                                      deleted_bytes=30, kept_count=2, skipped_recent=0, **res_kw)
    monkeypatch.setattr(hf_storage, "prune_orphans", _p)
    squashed = []
    monkeypatch.setattr(hf_storage, "super_squash", lambda r, t, *a, **k: squashed.append(r))
    monkeypatch.setattr(cli, "_hf_token", lambda: "hf_x")
    return calls, squashed


def test_cli_squash_refuses_noninteractive_without_yes(monkeypatch):
    calls, squashed = _fake_prune(monkeypatch)
    monkeypatch.setattr(cli.sys, "stdin",
                        type("S", (), {"isatty": staticmethod(lambda: False)})())
    rc = cli._cmd_ckpt_squash(_squash_args(yes=False))
    assert rc == 1
    assert squashed == []                       # squash 안 함
    assert all(c["dry_run"] for c in calls)     # 실제 삭제(dry_run=False) 호출 없음


def test_cli_squash_dry_run_writes_nothing(monkeypatch):
    calls, squashed = _fake_prune(monkeypatch)
    rc = cli._cmd_ckpt_squash(_squash_args(dry_run=True))
    assert rc == 0
    assert squashed == [] and calls == [dict(dry_run=True)]   # 계획(dry)만, 실삭제·squash 0


# ── item2: --keep-revision 가 prune 의 보존집합까지 전달된다 ───────────────────
@pytest.mark.allow_hf_storage
def test_cli_squash_keep_revision_passthrough(monkeypatch):
    """--keep-revision 로 지정한 리비전 트리의 블롭은 삭제되지 않고 보존돼야 한다."""
    old = datetime.now(timezone.utc) - timedelta(days=5)
    kr_blob = _Blob("KR", filename="kr.bin", pushed_at=old)    # main 트리엔 없음
    orphan = _Blob("ORPH", filename="o.bin", pushed_at=old)
    api = FakeApi(tree={"C_MAIN": ["KEEP"], "C_KR": ["KR"]},
                  lfs=[_Blob("KEEP", pushed_at=old), kr_blob, orphan])
    _wire_api(monkeypatch, api)
    monkeypatch.setattr(cli, "_hf_token", lambda: "x")
    rc = cli._cmd_ckpt_squash(_squash_args(yes=True, no_squash=True,
                                           keep_revision=["C_KR"]))
    assert rc == 0
    assert kr_blob in api.lfs and orphan not in api.lfs
    # keep_revision 없으면 kr_blob 은 고아로 삭제됨(대조군)
    kr2 = _Blob("KR", filename="kr.bin", pushed_at=old)
    api2 = FakeApi(tree={"C_MAIN": ["KEEP"], "C_KR": ["KR"]},
                   lfs=[_Blob("KEEP", pushed_at=old), kr2])
    _wire_api(monkeypatch, api2)
    cli._cmd_ckpt_squash(_squash_args(yes=True, no_squash=True))
    assert kr2 not in api2.lfs


# ── item3: squash 경고(무진행 워치독 오인) ────────────────────────────────────
@pytest.mark.allow_hf_storage
def test_cli_squash_warns_when_squashing(monkeypatch, capsys):
    calls, squashed = _fake_prune(monkeypatch)
    cli._cmd_ckpt_squash(_squash_args(dry_run=True, no_squash=False))
    assert "워치독" in capsys.readouterr().out
    # --no-squash 면 커밋을 안 만드니 경고도 없다
    _fake_prune(monkeypatch)
    cli._cmd_ckpt_squash(_squash_args(dry_run=True, no_squash=True))
    assert "워치독" not in capsys.readouterr().out


# ── item5: ckpt status 예외 시 트레이스백 대신 깔끔한 에러 ─────────────────────
@pytest.mark.allow_hf_storage
def test_cli_status_exception_clean_message(monkeypatch, capsys):
    def _boom(tok=None):
        raise RuntimeError("조회 폭발(시뮬)")
    monkeypatch.setattr(hf_storage, "_api", _boom)
    monkeypatch.setattr(cli, "_hf_token", lambda: "x")
    rc = cli._cmd_ckpt_status(argparse.Namespace(repo="user/ckpt"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "[FAIL]" in out and "조회 폭발" in out
