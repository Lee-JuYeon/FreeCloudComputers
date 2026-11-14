"""폴링 진실(#1) — 상태어 완전일치로만 종결, 못 읽으면 blind 를 센다.

예전엔 rc 를 버리고 'complete'/'error' 를 **부분문자열**로 찾아, 조회 실패(ConnectionError)나
슬러그('freecloud-error-x')의 'error' 를 종결로 오독했다. 이제 _parse_status 엄격파서로만
판정하고, 연속 POLL_BLIND_MAX 회 못 읽으면 생사 판정 불가로 멈춘다(원격 보호 remote_alive).

kaggle CLI·네트워크 0 — `_sh`·`time`·`latest_checkpoint_title` 만 몽키패치한다.
"""
from __future__ import annotations

from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.kaggle import KaggleProvider


def _prov(monkeypatch):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(K.time, "time", lambda: clock["t"])
    monkeypatch.setattr(K.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(K, "latest_checkpoint_title", lambda repo: "ckpt")
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    return p, clock


def _running(kid):
    return 0, f'{kid} has status "KernelWorkerStatus.RUNNING"'


def _job(name="t", **kw):
    d = dict(name=name, entrypoint="python x.py", max_runtime_s=300)
    d.update(kw)
    return Job(**d)


# ── #1a: 일시적 조회 실패(ConnectionError) 는 종결시키지 않는다 ────────────────
def test_transient_status_failure_does_not_terminate(monkeypatch):
    p, _ = _prov(monkeypatch)
    job = _job(checkpoint_repo="me/ckpt")
    kid = p.kernel_id(job)
    n = {"c": 0}

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            n["c"] += 1
            if n["c"] == 1:
                return 1, "ConnectionError: failed to reach kaggle"
            return _running(kid)
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.status == "detached"          # 끝까지 폴링(종결 안 함), blind 리셋됨
    assert res.error_class != "POLL_BLIND"


# ── #1b: 슬러그에 'error' 가 들어 있어도 running 을 error 로 오독하지 않는다 ───────
def test_error_in_slug_not_misread(monkeypatch):
    p, _ = _prov(monkeypatch)
    job = _job(name="error-x", checkpoint_repo="me/ckpt")   # slug=freecloud-error-x
    kid = p.kernel_id(job)
    assert "error" in kid

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return _running(kid)
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.status == "detached"          # error 로 종결되지 않음


# ── #1c: 20 연속 blind → 체크포인트 유무별 ───────────────────────────────────
def _blind_provider(monkeypatch, job):
    p, _ = _prov(monkeypatch)

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 1, "boom: cannot reach"       # 늘 못 읽음
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    return p


def test_poll_blind_with_checkpoint_is_detached(monkeypatch):
    job = _job(checkpoint_repo="me/ckpt", max_runtime_s=2000)
    p = _blind_provider(monkeypatch, job)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.status == "detached" and res.remote_alive is True


def test_poll_blind_without_checkpoint_is_poll_blind(monkeypatch, capsys):
    job = _job(max_runtime_s=2000)
    p = _blind_provider(monkeypatch, job)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.error_class == "POLL_BLIND" and res.remote_alive is True
    assert "상태조회 실패" in capsys.readouterr().out     # heartbeat 가 blind 를 알린다


# ── #1d: CANCEL_ACKNOWLEDGED → CANCELLED ─────────────────────────────────────
def test_cancel_acknowledged_is_cancelled(monkeypatch):
    p, _ = _prov(monkeypatch)
    job = _job()
    monkeypatch.setattr(p, "fetch_outputs", lambda job, kid: ("/tmp/adir", "diag"))

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 0, f'{p.kernel_id(job)} has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.ok is False and res.error_class == "CANCELLED"


# ── #1e: 마지막 폴만 blind·그전 running → detached(생사판정=마지막 읽힌 상태어) ──
def test_last_poll_blind_but_earlier_running_is_detached(monkeypatch):
    p, _ = _prov(monkeypatch)
    job = _job(checkpoint_repo="me/ckpt", max_runtime_s=180)   # 3 폴
    kid = p.kernel_id(job)
    n = {"c": 0}

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            n["c"] += 1
            if n["c"] >= 3:
                return 1, "boom"
            return _running(kid)
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.status == "detached"      # 마지막이 blind 여도 그전 running 으로 alive 판정


# ── Fable #2: 앵커·화이트리스트 (_parse_status 직접) ─────────────────────────
def test_parse_status_slug_ending_status_reads_complete():
    # ref 가 '-status' 로 끝나도 맨 낱말('has')이 아니라 따옴표 안을 읽어 complete.
    st = 'someone/freecloud-check-status has status "KernelWorkerStatus.COMPLETE"'
    assert KaggleProvider._parse_status(0, st) == "complete"


def test_parse_status_rc_nonzero_is_blank_even_if_complete():
    assert KaggleProvider._parse_status(1, 'x has status "KernelWorkerStatus.COMPLETE"') == ""


def test_parse_status_rejects_unknown_word_and_no_anchor():
    assert KaggleProvider._parse_status(0, 'x has status "weird_thing"') == ""
    assert KaggleProvider._parse_status(0, 'no anchor here complete') == ""


# ── Fable #4①: rc=1 인데 출력에 COMPLETE → rc 무시 변이를 잡는다(종결 안 함) ────
def test_rc_nonzero_complete_output_does_not_terminate(monkeypatch):
    p, _ = _prov(monkeypatch)
    job = _job(checkpoint_repo="me/ckpt")
    kid = p.kernel_id(job)

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 1, f'{kid} has status "KernelWorkerStatus.COMPLETE"'   # rc=1 → 무시해야
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.status != "done"          # rc 를 무시하면 done 으로 오종결 → 이 단언이 잡는다


# ── Fable #2(폴링 레벨): 슬러그가 '-status' 로 끝나도 complete 를 읽는다 ────────
def test_poll_slug_ending_status_reads_complete(monkeypatch):
    p, _ = _prov(monkeypatch)
    job = _job(name="check-status")
    kid = p.kernel_id(job)
    assert kid.endswith("-status")
    monkeypatch.setattr(p, "fetch_outputs",
                        lambda job, kid: ("/tmp/a", "boot\nGPU 0: Tesla P100-PCIE-16GB\n"))

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 0, f'{kid} has status "KernelWorkerStatus.COMPLETE"'
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    res = p._poll_and_fetch(job, work="/unused")
    assert res.ok is True and res.status == "done"
