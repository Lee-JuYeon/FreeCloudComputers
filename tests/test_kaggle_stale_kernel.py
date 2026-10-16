"""stale-kernel 가드 — push(rc=0) 가 새 세션을 못 띄우면 STALE_KERNEL 로 실패시킨다.

실측 2026-10-08·10-16: 주간 GPU 쿼터 소진 상태에서 `kaggle kernels push` 가 rc=0 이지만
새 버전/세션을 안 띄운다. 폴링이 옛 COMPLETE 를 읽어 거짓 성공을 보고했다.
push 전후로 `kaggle kernels list -m -v`(CSV)의 lastRunTime + status 변화를 폴링해
변화가 없고 status 가 여전히 complete 면 STALE_KERNEL(→ FAILOVER).

kaggle CLI 는 실제로 부르지 않는다(_sh 몽키패치). 폴링 상한은 0 으로 눌러 즉시 판정.
"""
from __future__ import annotations

import pytest

from freecloud.errors import FAILOVER, classify
from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.kaggle import KaggleProvider, _parse_last_run_time


# ── CSV 파서 ─────────────────────────────────────────────────────────────────
_CSV = ("ref,title,author,lastRunTime,totalVotes\n"
        "someone/freecloud-stale,My Kernel,someone,2026-10-16 09:00:00,0\n"
        "someone/other-kernel,Other,someone,2026-10-01 00:00:00,3\n")


def test_parse_last_run_time_matches_ref():
    assert _parse_last_run_time(_CSV, "someone/freecloud-stale") == "2026-10-16 09:00:00"


def test_parse_last_run_time_matches_by_slug():
    assert _parse_last_run_time(_CSV, "other-user/freecloud-stale") == "2026-10-16 09:00:00"


def test_parse_last_run_time_absent_returns_empty():
    assert _parse_last_run_time(_CSV, "someone/nope") == ""
    assert _parse_last_run_time("", "someone/freecloud-stale") == ""


# ── _push: stale vs healthy ──────────────────────────────────────────────────
def _provider(monkeypatch):
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    monkeypatch.setattr(K, "STALE_KERNEL_POLL_S", 0)
    monkeypatch.setattr(K, "STALE_KERNEL_POLL_INTERVAL_S", 0)
    return p


def _job():
    return Job(name="stale", entrypoint="python train.py", max_runtime_s=60)


def _csv_for(kid, lrt):
    return (f"ref,title,author,lastRunTime,totalVotes\n{kid},T,someone,{lrt},0\n")


def test_push_stale_when_lastruntime_unchanged(monkeypatch, capsys):
    p = _provider(monkeypatch)
    kid = p.kernel_id(_job())            # someone/freecloud-stale
    calls = []

    def fake_sh(cmd, timeout, env=None, cwd=None):
        calls.append(" ".join(cmd))
        j = " ".join(cmd)
        if "kernels push" in j:
            return 0, "pushed ok"
        if "kernels list" in j:
            return 0, _csv_for(kid, "2026-10-16 09:00:00")   # 항상 동일 → 변화 없음
        if "kernels status" in j:
            return 0, 'status "complete"'                    # 그대로 complete
        return 0, "ready"
    monkeypatch.setattr(p, "_sh", fake_sh)

    rc, log = p._push("work", _job())
    assert rc == 1
    assert "STALE_KERNEL" in log
    d = classify(log)
    assert d.error_class == "STALE_KERNEL" and d.action == FAILOVER
    assert any("kernels push" in c for c in calls)


def test_push_healthy_when_new_run_started(monkeypatch):
    p = _provider(monkeypatch)
    kid = p.kernel_id(_job())
    st = {"pushed": False}

    def fake_sh(cmd, timeout, env=None, cwd=None):
        j = " ".join(cmd)
        if "kernels push" in j:
            st["pushed"] = True
            return 0, "pushed ok"
        if "kernels list" in j:
            lrt = "2026-10-16 10:00:00" if st["pushed"] else "2026-10-16 09:00:00"
            return 0, _csv_for(kid, lrt)                     # push 후 바뀜 → 새 run
        if "kernels status" in j:
            return 0, 'status "complete"'
        return 0, "ready"
    monkeypatch.setattr(p, "_sh", fake_sh)

    rc, log = p._push("work", _job())
    assert rc == 0 and "STALE_KERNEL" not in log


def test_push_healthy_when_status_running(monkeypatch):
    # lastRunTime 은 그대로여도 status 가 running 으로 전환되면 새 세션이 떴다 → stale 아님.
    p = _provider(monkeypatch)
    kid = p.kernel_id(_job())
    st = {"pushed": False}

    def fake_sh(cmd, timeout, env=None, cwd=None):
        j = " ".join(cmd)
        if "kernels push" in j:
            st["pushed"] = True
            return 0, "pushed ok"
        if "kernels list" in j:
            return 0, _csv_for(kid, "2026-10-16 09:00:00")
        if "kernels status" in j:
            return 0, 'status "running"' if st["pushed"] else 'status "complete"'
        return 0, "ready"
    monkeypatch.setattr(p, "_sh", fake_sh)

    rc, log = p._push("work", _job())
    assert rc == 0 and "STALE_KERNEL" not in log


def test_push_passes_when_no_baseline(monkeypatch):
    # push 전 지문을 못 읽으면(새 커널 등) 비교 기준이 없다 → 막지 않는다(거짓양성 금지).
    p = _provider(monkeypatch)

    def fake_sh(cmd, timeout, env=None, cwd=None):
        j = " ".join(cmd)
        if "kernels push" in j:
            return 0, "pushed ok"
        if "kernels list" in j:
            return 0, "ref,title,author,lastRunTime,totalVotes\n"   # 내 커널 없음
        if "kernels status" in j:
            return 0, ""                                             # status 불명
        return 0, "ready"
    monkeypatch.setattr(p, "_sh", fake_sh)

    rc, log = p._push("work", _job())
    assert rc == 0 and "STALE_KERNEL" not in log


def test_push_returns_push_failure_unchanged(monkeypatch):
    # push 자체가 실패(rc!=0)하면 가드는 건드리지 않고 그대로 돌려준다.
    p = _provider(monkeypatch)

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if "kernels push" in " ".join(cmd):
            return 1, "403 Forbidden"
        return 0, "ready"
    monkeypatch.setattr(p, "_sh", fake_sh)

    rc, log = p._push("work", _job())
    assert rc == 1 and "403" in log and "STALE_KERNEL" not in log


# ── 플레이북 등록 확인 ───────────────────────────────────────────────────────
def test_stale_kernel_classifies_as_failover():
    d = classify("STALE_KERNEL: push 가 새 세션을 못 띄움")
    assert d.error_class == "STALE_KERNEL"
    assert d.action == FAILOVER
    assert d.cooldown_s > 0
