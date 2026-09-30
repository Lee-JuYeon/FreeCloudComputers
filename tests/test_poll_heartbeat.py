"""폴링 heartbeat — HEARTBEAT_S 마다 한 줄(경과·status·최신 checkpoint 커밋).

2026-09-30 실측: 3시간 폴링 동안 한 줄도 안 찍혀 살았는지 죽었는지 알 수 없었다.
네트워크 0 — `_sh`·`time`·`latest_checkpoint_title` 만 몽키패치한다.
"""
from __future__ import annotations

from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.kaggle import KaggleProvider

RUNNING = 'someone/freecloud-hb has status "KernelWorkerStatus.RUNNING"'


def _provider(monkeypatch, title_fn=lambda repo: "checkpoint step=200"):
    clock = {"t": 0.0}
    monkeypatch.setattr(K.time, "time", lambda: clock["t"])
    monkeypatch.setattr(K.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(K, "HEARTBEAT_S", 600)
    monkeypatch.setattr(K, "latest_checkpoint_title", title_fn)
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    monkeypatch.setattr(p, "_sh", lambda cmd, timeout, env=None, cwd=None:
                        (0, RUNNING) if cmd[:3] == ["kaggle", "kernels", "status"] else (0, "ok"))
    return p


def _beats(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if "♥" in ln]


def test_heartbeat_every_interval_with_checkpoint_title(monkeypatch, capsys):
    p = _provider(monkeypatch)
    job = Job(name="hb", entrypoint="x", max_runtime_s=1800, checkpoint_repo="me/ckpt")
    p._poll_and_fetch(job, work="/unused")
    beats = _beats(capsys.readouterr().out)
    assert len(beats) == 3                                   # 600·1200·1800s
    assert "10분 경과" in beats[0] and "30분 경과" in beats[-1]
    assert "status=running" in beats[0]
    assert "ckpt=checkpoint step=200" in beats[0]


def test_heartbeat_without_checkpoint_repo_has_no_ckpt(monkeypatch, capsys):
    called: list[str] = []
    p = _provider(monkeypatch, title_fn=lambda repo: called.append(repo) or "x")
    p._poll_and_fetch(Job(name="hb", entrypoint="x", max_runtime_s=1200), work="/unused")
    beats = _beats(capsys.readouterr().out)
    assert len(beats) == 2 and all("ckpt=" not in b for b in beats)
    assert called == []                                      # HF 조회 자체를 안 한다


def test_no_heartbeat_before_interval(monkeypatch, capsys):
    p = _provider(monkeypatch)
    p._poll_and_fetch(Job(name="hb", entrypoint="x", max_runtime_s=540), work="/unused")
    assert _beats(capsys.readouterr().out) == []


def test_hf_hiccup_never_breaks_polling(monkeypatch, capsys):
    import sys
    import types

    class Boom:
        def __init__(self, *a, **k):
            pass

        def list_repo_commits(self, repo):
            raise RuntimeError("HF 503")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Boom))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert K.latest_checkpoint_title("me/ckpt") == ""        # 예외를 삼킨다
    p = _provider(monkeypatch, title_fn=K.latest_checkpoint_title)
    res = p._poll_and_fetch(Job(name="hb", entrypoint="x", max_runtime_s=600,
                                checkpoint_repo="me/ckpt"), work="/unused")
    assert res.status == "detached"                          # 폴링은 끝까지 갔다
    assert "ckpt=(조회 실패/없음)" in _beats(capsys.readouterr().out)[0]


def test_latest_checkpoint_title_reads_first_commit(monkeypatch):
    import sys
    import types

    class Api:
        def __init__(self, token=None):
            self.token = token

        def list_repo_commits(self, repo):
            return [types.SimpleNamespace(title="checkpoint step=300"),
                    types.SimpleNamespace(title="checkpoint step=200")]

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api))
    monkeypatch.setenv("HF_TOKEN", "hf_x")
    assert K.latest_checkpoint_title("me/ckpt") == "checkpoint step=300"
