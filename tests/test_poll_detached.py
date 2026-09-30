"""폴링 상한 초과가 살아 있는 커널을 고아로 만들지 않는다(detached).

2026-09-30 실측: T4×2 커널은 ~4시간짜리인데 5400s 에서 TIMEOUT → kaggle-ui 30분 쿨다운
→ P100(UNFIT) 으로 넘어가 "실패" 보고. 원격 커널은 checkpoint 를 계속 push 하고 있었다.
kaggle CLI·네트워크 0 — `_sh`·`time` 만 몽키패치한다.
"""
from __future__ import annotations


from freecloud import orchestrator, registry, state
from freecloud.errors import DETACH, classify
from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.base import Provider, Probe, RunResult
from freecloud.providers.kaggle import KaggleProvider

RUNNING = 'someone/freecloud-det has status "KernelWorkerStatus.RUNNING"'


def _provider(monkeypatch, status_text=RUNNING):
    """status 는 늘 running, 시계는 sleep 할 때만 흐르는 provider."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(K.time, "time", lambda: clock["t"])
    monkeypatch.setattr(K.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(K, "latest_checkpoint_title", lambda repo: "checkpoint step=200")
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    calls: list[list[str]] = []

    def fake_sh(cmd, timeout, env=None, cwd=None):
        calls.append(list(cmd))
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 0, status_text
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    return p, calls


def _job(**kw):
    d = dict(name="det", entrypoint="python x.py", max_runtime_s=300)
    d.update(kw)
    return Job(**d)


def test_running_with_checkpoint_repo_is_detached(monkeypatch):
    p, calls = _provider(monkeypatch)
    res = p._poll_and_fetch(_job(checkpoint_repo="me/ckpt"), work="/unused")
    assert not res.ok
    assert res.status == "detached" and res.error_class == "DETACHED"
    assert "someone/freecloud-det" in res.reason and "checkpoint_repo" in res.reason
    assert not any(c[:3] == ["kaggle", "kernels", "output"] for c in calls)


def test_running_without_checkpoint_repo_keeps_timeout_and_tries_stop(monkeypatch, capsys):
    p, _ = _provider(monkeypatch)
    stops: list[str] = []
    orig = p.stop_kernel
    monkeypatch.setattr(p, "stop_kernel", lambda kid: (stops.append(kid), orig(kid))[1])
    res = p._poll_and_fetch(_job(), work="/unused")
    assert (res.ok, res.status, res.error_class) == (False, "timeout", "TIMEOUT")
    assert res.reason == "Kaggle 폴링 300s 초과."          # 기존 문구 그대로
    assert stops == ["someone/freecloud-det"]
    assert "커널 중지 실패" in capsys.readouterr().out     # API 없음 → 말은 한다


def test_timeout_when_kernel_not_alive_even_with_checkpoint(monkeypatch):
    # status 조회가 실패해 살아 있는지 모르면 detached 로 단정하지 않는다.
    p, _ = _provider(monkeypatch, status_text="Authentication required")
    res = p._poll_and_fetch(_job(checkpoint_repo="me/ckpt"), work="/unused")
    assert res.status == "timeout" and res.error_class == "TIMEOUT"


def test_detached_classifies_as_detach_without_cooldown():
    d = classify("DETACHED 폴링 상한 5400s 초과 — 커널은 계속 실행 중 timeout")
    assert d.action == DETACH and d.cooldown_s == 0      # "timeout" 이 RETRY 로 새지 않는다


class _Fake(Provider):
    def __init__(self, name, results):
        self.name = name
        self.capabilities = {"gpu": "T4", "vram_gb": 16, "headless": True}
        self._results = list(results)
        self.calls = 0

    def probe(self):
        return Probe(True, "")

    def run(self, job):
        self.calls += 1
        return self._results.pop(0) if self._results else RunResult(True, "done", "ok")


def test_orchestrator_stops_on_detached(monkeypatch, capsys):
    det = RunResult(False, "detached", "폴링 상한 5400s 초과 — 커널은 계속 실행 중(kernel=u/k).",
                    RUNNING, "DETACHED")
    a = _Fake("a", [det])
    b = _Fake("b", [RunResult(True, "done", "ok")])
    monkeypatch.setattr(registry, "get", lambda n: {"a": a, "b": b}[n])
    monkeypatch.setattr(registry, "DEFAULT_ORDER", ["a", "b"])
    for n in ("a", "b"):
        state.clear_cooldown(n)
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=3)
    assert res["ok"] is False and res.get("detached") is True
    assert res["provider"] == "a"
    assert a.calls == 1                                  # 재시도(RETRY) 없음
    assert b.calls == 0                                  # 다른 provider 로 이중 학습 없음
    assert state.cooldown_remaining("a") == 0            # 쿨다운 없음
    assert "⏳ [detached]" in capsys.readouterr().out
