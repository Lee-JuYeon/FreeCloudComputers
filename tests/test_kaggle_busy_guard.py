"""push 앞 관문(#8) — 같은 커널이 이미 running/queued 면 push 하지 않는다(이중 세션 방지).

_busy_check 가 _kernel_run_signature 의 상태어를 보고 KERNEL_BUSY(remote_alive=True)로 막는다.
못 읽으면('') 통과(거짓 양성 금지). run() 은 busy 면 _stage_and_write·_push 에 **진입하지 않는다**.
kaggle CLI·네트워크 0 — 지문/스테이징/푸시를 전부 몽키패치한다.
"""
from __future__ import annotations

from freecloud.job import Job
from freecloud.providers.base import RunResult
from freecloud.providers.kaggle import KaggleProvider


def _job(**kw):
    d = dict(name="busy", entrypoint="python x.py", max_runtime_s=60)
    d.update(kw)
    return Job(**d)


# ── _busy_check 순수 판정 ────────────────────────────────────────────────────
def test_busy_check_blocks_running_and_queued(monkeypatch):
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    for word in ("running", "queued"):
        res = p._busy_check(_job(), before=("lrt", word))
        assert res is not None
        assert res.error_class == "KERNEL_BUSY" and res.remote_alive is True
        assert word in res.reason


def test_busy_check_passes_on_complete_or_unreadable(monkeypatch):
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    assert p._busy_check(_job(), before=("lrt", "complete")) is None
    assert p._busy_check(_job(), before=("", "")) is None       # 못 읽음 → 통과


# ── run(): busy 면 stage/push 0, 아니면 진행 ────────────────────────────────
def _instrument(monkeypatch, p, status_word):
    monkeypatch.setattr(p, "_user", lambda: "someone")
    monkeypatch.setattr(p, "_kernel_run_signature", lambda kid: ("lrt", status_word))
    n = {"stage": 0, "push": 0}
    monkeypatch.setattr(p, "_stage_and_write",
                        lambda job: (n.__setitem__("stage", n["stage"] + 1), ("work", None))[1])
    monkeypatch.setattr(p, "_push",
                        lambda work, job, before=None: (n.__setitem__("push", n["push"] + 1),
                                                        (0, "pushed"))[1])
    monkeypatch.setattr(p, "_poll_and_fetch", lambda job, work: RunResult(True, "done", "ok"))
    return n


def test_run_busy_skips_stage_and_push(monkeypatch):
    p = KaggleProvider()
    n = _instrument(monkeypatch, p, "running")
    res = p.run(_job())
    assert res.error_class == "KERNEL_BUSY" and res.remote_alive is True
    assert n == {"stage": 0, "push": 0}            # 업로드·push 0


def test_run_not_busy_proceeds_and_marks_pushed(monkeypatch):
    p = KaggleProvider()
    n = _instrument(monkeypatch, p, "complete")
    res = p.run(_job())
    assert res.ok is True and res.pushed is True
    assert n == {"stage": 1, "push": 1}


def test_run_unreadable_signature_proceeds(monkeypatch):
    p = KaggleProvider()
    n = _instrument(monkeypatch, p, "")
    res = p.run(_job())
    assert res.ok is True and n == {"stage": 1, "push": 1}
