"""HF 토큰 처리 개선 회귀 테스트: verify/미러/프리플라이트/체크포인트 만료."""
import os

import pytest

from freecloud import orchestrator, registry, state, checkpoint, auth
from freecloud.job import Job
from freecloud.providers.base import Provider, Probe, RunResult


# ── verify_hf_token ──────────────────────────────────────────────────────────
def test_verify_hf_token_none():
    ok, detail = auth.verify_hf_token(None)
    assert not ok and "HF_TOKEN" in detail


def test_verify_hf_token_invalid(monkeypatch):
    # 라이브러리가 있어야 의미 있는 테스트 — 없는 환경에선 skip(에러 아님).
    huggingface_hub = pytest.importorskip("huggingface_hub")
    def _boom(self, token=None):
        raise RuntimeError("Invalid user token")
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami", _boom)
    ok, detail = auth.verify_hf_token("hf_bad")
    assert not ok


def test_verify_hf_token_valid(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    monkeypatch.setattr(huggingface_hub.HfApi, "whoami",
                        lambda self, token=None: {"name": "tester"})
    ok, detail = auth.verify_hf_token("hf_good")
    assert ok and "tester" in detail


def test_verify_hf_token_hub_미설치면_무효로_보지_않는다(monkeypatch):
    """CI 가 드러낸 분기 — huggingface_hub 이 없다고 토큰을 '무효'로 단정하면 안 된다.

    로컬 검증이 불가할 뿐이고 원격 노드에는 설치돼 있을 수 있다. 여기서 False 를 주면
    멀쩡한 토큰으로도 프리플라이트가 런을 막아버린다.
    """
    import builtins
    real_import = builtins.__import__

    def _no_hub(name, *a, **k):
        if name == "huggingface_hub":
            raise ImportError("no huggingface_hub")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", _no_hub)
    ok, detail = auth.verify_hf_token("hf_whatever")
    assert ok and "미설치" in detail


# ── ~/.cache 미러 ────────────────────────────────────────────────────────────
def test_mirror_hf_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    p = auth._mirror_hf_cache("hf_secret123\n")   # 개행 포함 → strip 확인
    assert p and os.path.exists(p)
    with open(p, encoding="utf-8") as f:
        assert f.read() == "hf_secret123"


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


def _wire(monkeypatch, fake):
    monkeypatch.setattr(registry, "get", lambda n: fake)
    monkeypatch.setattr(registry, "DEFAULT_ORDER", [fake.name])
    state.clear_cooldown(fake.name)


def test_preflight_blocks_before_dispatch(monkeypatch):
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (False, "토큰 무효 test"))
    fake = Fake()
    _wire(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", checkpoint_repo="user/ckpt")
    res = orchestrator.run(job, max_rounds=1)
    assert not res["ok"]
    assert fake.calls == 0                       # dispatch 前 차단 → 업로드/쿼터 낭비 없음
    assert "프리플라이트" in res["reason"]


def test_preflight_passes_with_valid_token(monkeypatch):
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (True, "ok"))
    monkeypatch.setattr(auth, "verify_hf_write", lambda r, t=None: (True, "쓰기 OK"))
    fake = Fake()
    _wire(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", checkpoint_repo="user/ckpt")
    res = orchestrator.run(job, max_rounds=1)
    assert res["ok"] and fake.calls == 1


def test_preflight_blocks_when_token_cannot_write(monkeypatch):
    """whoami 통과 ≠ 쓰기 가능. 읽기 전용 토큰은 dispatch 前에 걸러야 한다 —
    안 그러면 런이 끝나갈 무렵 체크포인트 push 에서 401 이 나고 쿼터를 날린다."""
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (True, "ok"))
    monkeypatch.setattr(auth, "verify_hf_write",
                        lambda r, t=None: (False, "토큰에 쓰기 권한이 없습니다"))
    fake = Fake()
    _wire(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", checkpoint_repo="user/ckpt")
    res = orchestrator.run(job, max_rounds=1)
    assert not res["ok"]
    assert fake.calls == 0
    assert "쓰기 권한" in res["reason"]


def test_preflight_write검사는_체크포인트_없으면_안한다(monkeypatch):
    """checkpoint_repo 가 없으면 쓸 저장소가 없으니 write 검사도 없어야 한다."""
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (True, "ok"))
    called = {"w": False}
    def _w(r, t=None):
        called["w"] = True
        return (True, "")
    monkeypatch.setattr(auth, "verify_hf_write", _w)
    fake = Fake()
    _wire(monkeypatch, fake)
    job = Job(name="t", entrypoint="x", secrets=["HF_TOKEN"])
    orchestrator.run(job, max_rounds=1)
    assert not called["w"]


def test_preflight_skipped_when_hf_not_needed(monkeypatch):
    called = {"v": False}
    def _v(t):
        called["v"] = True
        return (False, "should-not-run")
    monkeypatch.setattr(auth, "verify_hf_token", _v)
    fake = Fake()
    _wire(monkeypatch, fake)
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=1)   # checkpoint 없음
    assert res["ok"] and not called["v"]         # HF 불필요 → 검증 자체를 안 함


# ── 체크포인트 만료 우아처리 ──────────────────────────────────────────────────
def test_is_auth_error():
    assert checkpoint._is_auth_error(RuntimeError("401 Client Error: Unauthorized"))
    assert checkpoint._is_auth_error(RuntimeError("Invalid user token"))
    assert not checkpoint._is_auth_error(RuntimeError("connection reset by peer"))


def test_push_preserves_local_on_auth_error(monkeypatch, tmp_path):
    from freecloud.errors import classify, ABORT
    ck = checkpoint.Checkpoint("user/ckpt", token="hf_x")
    monkeypatch.setattr(ck, "ensure_repo", lambda: None)

    class FakeApi:
        def upload_folder(self, **k):
            raise RuntimeError("401 Unauthorized: invalid token")

    monkeypatch.setattr(ck, "_api", lambda: FakeApi())
    src = str(tmp_path)
    with pytest.raises(RuntimeError) as ei:
        ck.push(src, step=5)
    msg = str(ei.value)
    assert "보존" in msg and src in msg           # 로컬 유실 없음 안내
    assert classify(msg).action == ABORT          # 플레이북이 AUTH(사람개입)로 분류
