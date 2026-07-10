from freecloud import orchestrator, registry, state
from freecloud.job import Job
from freecloud.providers.base import Provider, Probe, RunResult


class Fake(Provider):
    def __init__(self, name, results, available=True):
        self.name = name
        self.capabilities = {"gpu": "T4", "vram_gb": 16, "headless": True}
        self._results = list(results)
        self._available = available
        self.calls = 0

    def probe(self):
        return Probe(self._available, "" if self._available else "down")

    def run(self, job):
        self.calls += 1
        return self._results.pop(0) if self._results else RunResult(False, "error", "no-more")


def _wire(monkeypatch, mapping, order):
    monkeypatch.setattr(registry, "get", lambda n: mapping[n])
    monkeypatch.setattr(registry, "DEFAULT_ORDER", order)
    for n in order:
        state.clear_cooldown(n)


def _job():
    return Job(name="t", entrypoint="python x.py")


def test_failover_quota_then_success(monkeypatch):
    a = Fake("a", [RunResult(False, "quota", "TooManyAssignments", error_class="QUOTA_COOLDOWN")])
    b = Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"a": a, "b": b}, ["a", "b"])
    res = orchestrator.run(_job(), max_rounds=1)
    assert res["ok"] and res["provider"] == "b"
    assert a.calls == 1 and b.calls == 1
    assert state.cooldown_remaining("a") > 0   # a는 쿨다운 걸림


def test_abort_stops_immediately(monkeypatch):
    a = Fake("a", [RunResult(False, "error", "401 gated repo", error_class="AUTH")])
    b = Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"a": a, "b": b}, ["a", "b"])
    res = orchestrator.run(_job(), max_rounds=1)
    assert not res["ok"]
    assert b.calls == 0                        # ABORT → 다음 provider 시도 안 함


def test_skip_when_unavailable(monkeypatch):
    a = Fake("a", [], available=False)
    b = Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"a": a, "b": b}, ["a", "b"])
    res = orchestrator.run(_job(), max_rounds=1)
    assert res["ok"] and res["provider"] == "b"
    assert a.calls == 0                        # probe 실패 → run 안 함


def test_fits_filters_vram(monkeypatch):
    small = Fake("small", [RunResult(True, "done", "ok")])
    small.capabilities = {"gpu": "T4", "vram_gb": 8, "headless": True}
    big = Fake("big", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"small": small, "big": big}, ["small", "big"])
    job = Job(name="t", entrypoint="x", needs={"min_vram_gb": 16})
    res = orchestrator.run(job, max_rounds=1)
    assert res["ok"] and res["provider"] == "big"
    assert small.calls == 0                    # VRAM 부족 → 스킵
