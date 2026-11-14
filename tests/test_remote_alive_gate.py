"""remote_alive 게이트(#4 밑바탕) — 원격 세션이 살아 있을 수 있으면 즉시 멈춘다.

detached 와 같은 원칙: KERNEL_BUSY·POLL_BLIND·stop 실패 STALLED 처럼 '원격 커널이 아직
돌 수 있는' 결과(res.remote_alive=True)는 쿨다운·재시도·페일오버 없이 바로 끝낸다 —
다른 provider 를 띄우면 이중 세션이 된다. 네트워크 0(가짜 provider).
"""
from __future__ import annotations

from freecloud import orchestrator, registry, state
from freecloud.errors import ABORT, FAILOVER, RETRY, classify
from freecloud.job import Job
from freecloud.providers.base import Provider, Probe, RunResult


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


def _wire(monkeypatch, a, b):
    monkeypatch.setattr(registry, "get", lambda n: {"a": a, "b": b}[n])
    monkeypatch.setattr(registry, "DEFAULT_ORDER", ["a", "b"])
    for n in ("a", "b"):
        state.clear_cooldown(n)


def test_remote_alive_stops_without_retry_cooldown_or_failover(monkeypatch):
    alive = RunResult(False, "error", "KERNEL_BUSY: kernel=u/k status=running — push 하면 이중 세션",
                      "", "KERNEL_BUSY", remote_alive=True)
    a = _Fake("a", [alive])
    b = _Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, a, b)
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=3)
    assert res["ok"] is False and res.get("remote_alive") is True
    assert res["provider"] == "a"
    assert a.calls == 1          # RETRY 없음
    assert b.calls == 0          # 페일오버로 이중 세션 없음
    assert state.cooldown_remaining("a") == 0   # 쿨다운 0


def test_remote_alive_beats_abort_classification(monkeypatch):
    # STALLED(ABORT) 이지만 stop 실패로 remote_alive=True → ABORT 보다 먼저 걸러 멈춘다.
    alive = RunResult(False, "error", "STALLED: 50분째 진행신호 없음", "", "STALLED",
                      remote_alive=True)
    a = _Fake("a", [alive])
    b = _Fake("b", [RunResult(True, "done", "ok")])
    _wire(monkeypatch, a, b)
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=3)
    assert res.get("remote_alive") is True and b.calls == 0


# ── 새 error_class 5개의 action 계약 ─────────────────────────────────────────
def test_stalled_aborts():
    d = classify("STALLED: 진행신호 없음")
    assert d.error_class == "STALLED" and d.action == ABORT and d.cooldown_s == 0


def test_kernel_busy_aborts():
    d = classify("KERNEL_BUSY: kernel=u/k status=running")
    assert d.error_class == "KERNEL_BUSY" and d.action == ABORT and d.cooldown_s == 0


def test_poll_blind_aborts():
    d = classify("POLL_BLIND: 상태조회 20회 연속 실패")
    assert d.error_class == "POLL_BLIND" and d.action == ABORT and d.cooldown_s == 0


def test_cancelled_aborts():
    d = classify("CANCELLED: 커널이 취소됨(cancel_acknowledged)")
    assert d.error_class == "CANCELLED" and d.action == ABORT and d.cooldown_s == 0


def test_ui_automation_failovers():
    d = classify("UI_AUTOMATION: 셀렉터 타임아웃")
    assert d.error_class == "UI_AUTOMATION" and d.action == FAILOVER and d.cooldown_s == 900


# ── 순서 계약: 새 reason 안의 403·timeout 이 AUTH/TRANSIENT 로 새지 않는다 ──────
def test_stalled_with_403_and_timeout_stays_abort():
    d = classify("STALLED: HF 업로드 403 실패로 커밋 안 됨, status timeout")
    assert d.error_class == "STALLED" and d.action == ABORT


def test_ui_automation_with_timeout_is_no_longer_retry():
    # 예전엔 'timeout' 이 TRANSIENT(RETRY)로 새 같은 provider 를 재시도했다.
    d = classify("UI_AUTOMATION 셀렉터 타임아웃: selector timeout")
    assert d.action == FAILOVER and d.action != RETRY
