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


def test_probe_login_required_aborts_without_cooldown(monkeypatch):
    """probe 가 LOGIN_REQUIRED(ABORT)를 알리면 쿨다운·대기 없이 즉시 종료(설계 §3·T15).

    L0 를 넣으면 단일 provider(kaggle-ui)가 만료됐을 때 '전부 쿨다운 → 1800s 대기'를 라운드마다
    반복해 약 1시간 헛대기가 되던 경로가 새로 열린다 — 그 경로를 막는다."""
    class LoginExpired(Fake):
        def probe(self):
            self.probes = getattr(self, "probes", 0) + 1
            return Probe(False, "브라우저 로그인 만료 → fcc kaggle-login", error_class="LOGIN_REQUIRED")

    a = LoginExpired("kaggle-ui", [])
    _wire(monkeypatch, {"kaggle-ui": a}, ["kaggle-ui"])
    slept: list = []
    monkeypatch.setattr(orchestrator.time, "sleep", lambda s: slept.append(s))
    res = orchestrator.run(_job(), max_rounds=3)
    assert not res["ok"] and res["provider"] is None
    assert slept == []                                     # 헛대기 없음
    assert a.calls == 0                                    # run 은 안 불린다
    assert state.cooldown_remaining("kaggle-ui") == 0      # 쿨다운도 안 건다
    assert res["attempts"][0]["error_class"] == "LOGIN_REQUIRED"


def test_fits_filters_vram(monkeypatch):
    small = Fake("small", [RunResult(True, "done", "ok")])
    small.capabilities = {"gpu": "T4", "vram_gb": 8, "headless": True}
    big = Fake("big", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, {"small": small, "big": big}, ["small", "big"])
    job = Job(name="t", entrypoint="x", needs={"min_vram_gb": 16})
    res = orchestrator.run(job, max_rounds=1)
    assert res["ok"] and res["provider"] == "big"
    assert small.calls == 0                    # VRAM 부족 → 스킵
