"""UNFIT provider 는 라운드를 받지 않는다 + 적합 provider 전부 쿨다운이면 라운드 대신 대기.

2026-09-30 실측: needs.gpu=T4 인데 kaggle(P100)이 라운드 2·3 을 차지해 "3라운드 내 성공
provider 없음"으로 끝났다. 네트워크 0 — registry·time.sleep 만 몽키패치한다.
"""
from __future__ import annotations

from freecloud import orchestrator, registry, state
from freecloud.job import Job
from freecloud.providers.base import Provider, Probe, RunResult


class Fake(Provider):
    def __init__(self, name, results, gpu="T4"):
        self.name = name
        self.capabilities = {"gpu": gpu, "vram_gb": 16, "headless": True}
        self._results = list(results)
        self.calls = 0
        self.probes = 0

    def probe(self):
        self.probes += 1
        return Probe(True, "")

    def run(self, job):
        self.calls += 1
        return self._results.pop(0) if self._results else RunResult(False, "error", "no-more")


def _wire(monkeypatch, mapping, order):
    monkeypatch.setattr(registry, "get", lambda n: mapping[n])
    monkeypatch.setattr(registry, "DEFAULT_ORDER", order)
    for n in order:
        state.clear_cooldown(n)


def test_unfit_provider_never_gets_a_round(monkeypatch, capsys):
    t4 = Fake("kaggle-ui", [RunResult(False, "error", "boom-1"),
                            RunResult(False, "error", "boom-2"),
                            RunResult(True, "done", "ok")], gpu="T4x2")
    p100 = Fake("kaggle", [RunResult(True, "done", "ok")], gpu="P100")
    _wire(monkeypatch, {"kaggle-ui": t4, "kaggle": p100}, ["kaggle-ui", "kaggle"])
    # 실패 뒤 쿨다운이 걸려도 테스트가 실제로 기다리지 않게: sleep 이 쿨다운을 풀어준다.
    monkeypatch.setattr(orchestrator.time, "sleep",
                        lambda s: state.clear_cooldown("kaggle-ui"))
    res = orchestrator.run(Job(name="t", entrypoint="x", needs={"gpu": "T4"}), max_rounds=3)
    assert res["ok"] and res["provider"] == "kaggle-ui"
    assert p100.calls == 0 and p100.probes == 0          # UNFIT → probe 조차 안 함
    unfit = [a for a in res["attempts"] if a.get("error_class") == "UNFIT"]
    assert len(unfit) == 1                               # 라운드마다가 아니라 한 번만
    out = capsys.readouterr().out
    assert out.count("건너뜀:") == 1 and "kaggle" in out


def test_no_fit_provider_exits_without_rounds(monkeypatch):
    p100 = Fake("kaggle", [RunResult(True, "done", "ok")], gpu="P100")
    _wire(monkeypatch, {"kaggle": p100}, ["kaggle"])
    slept: list[float] = []
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: slept.append(s))
    res = orchestrator.run(Job(name="t", entrypoint="x", needs={"gpu": "T4"}), max_rounds=3)
    assert not res["ok"]
    assert "적합한 provider 없음" in res["reason"]
    assert p100.calls == 0 and slept == []               # 라운드도 대기도 없음


def test_all_fit_in_cooldown_waits_instead_of_burning_rounds(monkeypatch, capsys):
    a = Fake("a", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"a": a}, ["a"])
    state.set_cooldown("a", 5000, "test")                # 상한(1800)보다 길다
    slept: list[float] = []

    def fake_sleep(s):
        slept.append(s)
        state.clear_cooldown("a")                        # 기다리는 사이 쿨다운 종료

    monkeypatch.setattr(orchestrator.time, "sleep", fake_sleep)
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=1)
    assert res["ok"] and a.calls == 1                    # 라운드 1개로도 성공(소모 안 됨)
    assert len(slept) == 1
    assert slept[0] <= orchestrator.COOLDOWN_WAIT_CAP_S + 1   # 1회 대기 상한
    assert "쿨다운" in capsys.readouterr().out


def test_wait_uses_shortest_cooldown(monkeypatch):
    a = Fake("a", [RunResult(True, "done", "ok")])
    b = Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"a": a, "b": b}, ["a", "b"])
    state.set_cooldown("a", 900, "t")
    state.set_cooldown("b", 120, "t")
    slept: list[float] = []

    def fake_sleep(s):
        slept.append(s)
        state.clear_cooldown("b")

    monkeypatch.setattr(orchestrator.time, "sleep", fake_sleep)
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=1)
    assert res["ok"] and res["provider"] == "b"
    assert 100 < slept[0] <= 121                         # 가장 짧은 쿨다운(b) 만큼
    state.clear_cooldown("a")


def test_once_still_exits_on_first_failure(monkeypatch):
    a = Fake("a", [RunResult(False, "error", "boom")])
    b = Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"a": a, "b": b}, ["a", "b"])
    res = orchestrator.run(Job(name="t", entrypoint="x"), once=True)
    assert not res["ok"] and "once" in res["reason"]
    assert b.calls == 0
    state.clear_cooldown("a")
