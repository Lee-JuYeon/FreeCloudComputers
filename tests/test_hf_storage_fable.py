"""Fable 검증용 추가 테스트(스크래치 사본에만 존재)."""
from datetime import datetime, timedelta, timezone
import pytest
from freecloud import hf_storage, auth, orchestrator, registry, state, checkpoint, errors, cli
from freecloud.job import Job
from tests.test_hf_storage import FakeApi, _Blob, _Refs, _Ref, _wire_api, _storage_full_exc, Fake, _wire_prov, _squash_args

OLD = datetime.now(timezone.utc) - timedelta(days=5)

def _api_with_orphan():
    orphan = _Blob("ORPH", filename="old.bin", pushed_at=OLD)
    return FakeApi(tree={"C_MAIN": ["KEEP"]}, lfs=[_Blob("KEEP", pushed_at=OLD), orphan]), orphan

@pytest.mark.allow_hf_storage
def test_default_job_end_to_end_never_deletes(monkeypatch):
    """Job 기본값 → orchestrator → 실 verify_hf_write → 고아 삭제 0."""
    api, orphan = _api_with_orphan(); _wire_api(monkeypatch, api)
    monkeypatch.setattr(checkpoint.Checkpoint, "ensure_repo", lambda self: None)
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (True, "ok"))
    fake = Fake(); _wire_prov(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", checkpoint_repo="user/ckpt")
    assert job.checkpoint_hygiene == "warn"
    res = orchestrator.run(job, max_rounds=1)
    assert res["ok"] and fake.calls == 1
    assert orphan in api.lfs and "delete-orphan" not in api.events

@pytest.mark.allow_hf_storage
def test_verify_default_arg_never_deletes(monkeypatch):
    api, orphan = _api_with_orphan(); _wire_api(monkeypatch, api)
    monkeypatch.setattr(checkpoint.Checkpoint, "ensure_repo", lambda self: None)
    ok, d = auth.verify_hf_write("user/ckpt", "hf_x")
    assert ok and orphan in api.lfs
    assert "삭제 안 함" in hf_storage.hygiene("user/ckpt", "hf_x") and orphan in api.lfs

@pytest.mark.allow_hf_storage
@pytest.mark.parametrize("pol", ["Auto", "AUTO", "true", True, None, "", "prune", "on", 1])
def test_unknown_policy_never_deletes(monkeypatch, pol):
    api, orphan = _api_with_orphan(); _wire_api(monkeypatch, api)
    hf_storage.hygiene("user/ckpt", "hf_x", policy=pol)
    assert orphan in api.lfs

@pytest.mark.allow_hf_storage
def test_orch_passes_job_policy_and_keeprevs(monkeypatch):
    seen = {}
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (True, "ok"))
    monkeypatch.setattr(auth, "verify_hf_write", lambda r, t=None, **kw: seen.update(kw) or (True, "ok"))
    fake = Fake(); _wire_prov(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", checkpoint_repo="user/ckpt", checkpoint_hygiene="auto", checkpoint_keep_revisions=["abc"])
    orchestrator.run(job, max_rounds=1)
    assert seen == dict(hygiene="auto", keep_revisions=["abc"])

@pytest.mark.allow_hf_storage
def test_auto_honours_keep_revisions(monkeypatch):
    kr = _Blob("KR", pushed_at=OLD)
    api = FakeApi(tree={"C_MAIN": ["KEEP"], "C_KR": ["KR"]}, lfs=[_Blob("KEEP", pushed_at=OLD), kr, _Blob("O", filename="o", pushed_at=OLD)])
    _wire_api(monkeypatch, api)
    hf_storage.hygiene("user/ckpt", "hf_x", policy="auto", keep_revisions=["C_KR"])
    assert kr in api.lfs

@pytest.mark.allow_hf_storage
def test_checkpoint_push_storage_full_classified(monkeypatch):
    class A:
        def create_repo(self, *a, **k): pass
        def upload_folder(self, **k): raise _storage_full_exc()
    ck = checkpoint.Checkpoint("user/ckpt", token="x")
    monkeypatch.setattr(ck, "_api", lambda: A())
    with pytest.raises(RuntimeError) as ei:
        ck.push("/tmp/x", step=1)
    assert errors.classify(str(ei.value)).error_class == "HF_STORAGE_FULL"

@pytest.mark.allow_hf_storage
def test_report_uses_file_oid(monkeypatch):
    api, _ = _api_with_orphan(); _wire_api(monkeypatch, api)
    r = hf_storage.storage_report("user/ckpt", "x")
    assert (r.keep_count, r.orphan_count) == (1, 1)

@pytest.mark.allow_hf_storage
def test_postdelete_recheck_flags_missing_keep(monkeypatch):
    api, _ = _api_with_orphan(); _wire_api(monkeypatch, api)
    orig = api.permanently_delete_lfs_files
    def evil(repo, files, **k):
        orig(repo, files, **k); api.lfs[:] = []      # 서버가 보존 블롭까지 날린 상황
    api.permanently_delete_lfs_files = evil
    assert hf_storage.prune_orphans("user/ckpt", "x").aborted

@pytest.mark.allow_hf_storage
def test_cli_real_squash_flow(monkeypatch, capsys):
    api, orphan = _api_with_orphan(); _wire_api(monkeypatch, api)
    monkeypatch.setattr(cli, "_hf_token", lambda: "x")
    assert cli._cmd_ckpt_squash(_squash_args(yes=True, no_squash=True)) == 0
    assert api.squashed == [] and orphan not in api.lfs
    # 보존집합 빈 저장소: squash 도 삭제도 0
    api2 = FakeApi(tree={"C_MAIN": []}, lfs=[_Blob("X", pushed_at=OLD)]); _wire_api(monkeypatch, api2)
    assert cli._cmd_ckpt_squash(_squash_args(yes=True)) == 1
    assert api2.squashed == [] and api2.deleted == []

@pytest.mark.allow_hf_storage
def test_probe_does_not_touch_other_blobs(monkeypatch):
    api, orphan = _api_with_orphan(); _wire_api(monkeypatch, api)
    hf_storage.write_probe("user/ckpt", "x", size=64)
    assert orphan in api.lfs and len(api.lfs) == 2

def test_cli_parser_wires():
    import argparse
    for argv in (["ckpt", "status", "u/r"], ["ckpt", "squash", "u/r", "--dry-run"]):
        try:
            # main() 은 fn 을 실행 → _api 차단으로 RuntimeError 가 나야 정상(네트워크0)
            cli.main(argv)
        except RuntimeError as e:
            assert "차단" in str(e)
