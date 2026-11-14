"""무진행 워치독(A) — 'running 인데 진행신호(HF 커밋)가 끊긴' 좀비 커널을 잡는다.

checkpoint_repo 가 있을 때만 켜고(결과를 받을 곳이 있을 때만), 느린 job 은 관측된 최대
커밋간격의 2배까지 기다려 안 죽인다. HF None(조회 실패)은 '모름=정지 아님'. env 로 끈다.

kaggle CLI·HF·네트워크 0 — time·latest_checkpoint_commit·stop_kernel·_sh 만 몽키패치.
"""
from __future__ import annotations

from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.kaggle import KaggleProvider

RUN_STATUS = 'has status "KernelWorkerStatus.RUNNING"'


def _prov(monkeypatch, commit_fn, stop_result=(True, "stopped"), status_fn=None):
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(K.time, "time", lambda: clock["t"])
    monkeypatch.setattr(K.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(K, "latest_checkpoint_commit", commit_fn)
    monkeypatch.setattr(K, "latest_checkpoint_title", lambda repo: "")
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    stops: list[str] = []
    monkeypatch.setattr(p, "stop_kernel", lambda kid: (stops.append(kid), stop_result)[1])

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            word = status_fn(clock["t"]) if status_fn else "RUNNING"
            return 0, f'{cmd[3]} has status "KernelWorkerStatus.{word}"'
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    return p, clock, stops


def _schedule(clock, commits):
    """commits=[(created_epoch, title)] → fn(repo) 로 now 이하 최신 커밋((title,created)) 또는 None."""
    def fn(repo):
        now = clock["t"]
        latest = None
        for created, title in commits:
            if created <= now:
                latest = (title, float(created))
        return latest
    return fn


def _job(name="np", **kw):
    d = dict(name=name, entrypoint="python x.py", max_runtime_s=4000,
             checkpoint_repo="me/ckpt")
    d.update(kw)
    return Job(**d)


# ── 1) 첫 진행신호 없이 45분+ → STALLED + stop 1회 ───────────────────────────
def test_no_first_progress_stalls_after_grace(monkeypatch):
    # run 이전의 옛 커밋 하나만 있고(this_run 아님) 새 커밋은 영영 없음.
    commit = lambda repo: ("step=0", 990_000.0)
    p, _, stops = _prov(monkeypatch, commit)
    res = p._poll_and_fetch(_job(max_runtime_s=4000), work="/u")
    assert res.error_class == "STALLED" and res.remote_alive is False
    assert res.pushed is True
    assert len(stops) == 1


# ── 1b) Fable #1: 재개 시나리오 — 옛 커밋이 한도를 부풀리면 안 된다 ──────────
def test_old_commit_does_not_inflate_limit(monkeypatch):
    # repo 에 3일 전 옛 커밋 + 이번 run 의 첫 커밋(20분)뒤 정지 → 약 66분에 STALLED.
    # 버그(옛 커밋을 간격에 넣음)면 한도가 수천 분으로 부풀어 영영 미발동.
    p, clock, stops = _prov(monkeypatch, lambda repo: None)
    old = 1_000_000.0 - 3 * 86400          # 3일 전(이번 run 아님)
    first_run_commit = 1_001_260.0          # running 시작(≈+60s) 뒤 20분

    def commit(repo):
        return ("step=1", first_run_commit) if clock["t"] >= first_run_commit else ("old", old)
    monkeypatch.setattr(K, "latest_checkpoint_commit", commit)
    res = p._poll_and_fetch(_job(max_runtime_s=5000), work="/u")
    assert res.error_class == "STALLED" and len(stops) == 1
    assert "한도 40분" in res.reason          # 적응 한도=max(40분, 2×20분)=40분(부풀지 않음)


# ── 2) 44분(grace 미만) → 미발동 ─────────────────────────────────────────────
def test_under_grace_does_not_stall(monkeypatch):
    commit = lambda repo: ("step=0", 990_000.0)
    p, _, stops = _prov(monkeypatch, commit)
    res = p._poll_and_fetch(_job(max_runtime_s=2640), work="/u")   # 44분
    assert res.error_class != "STALLED" and res.status == "detached"
    assert stops == []


# ── 3) 17분 간격 커밋 오다 끊김 → 초과 시 발동 ───────────────────────────────
def test_regular_commits_then_gap_stalls(monkeypatch):
    # _schedule 은 _prov 내부 clock 을 참조해야 하므로 provider 를 먼저 만든 뒤 재패치한다.
    p, clock, stops = _prov(monkeypatch, lambda repo: None)
    monkeypatch.setattr(K, "latest_checkpoint_commit",
                        _schedule(clock, [(1_001_020, "s1"), (1_002_040, "s2"), (1_003_060, "s3")]))
    res = p._poll_and_fetch(_job(max_runtime_s=8000), work="/u")
    assert res.error_class == "STALLED" and len(stops) == 1


# ── 4) 60분 간격 job 은 40분 공백에 안 죽는다(적응) ─────────────────────────
def test_slow_job_survives_short_gap(monkeypatch):
    p, clock, stops = _prov(monkeypatch, lambda repo: None)
    monkeypatch.setattr(K, "latest_checkpoint_commit",
                        _schedule(clock, [(1_003_600, "s1"), (1_007_200, "s2")]))
    res = p._poll_and_fetch(_job(max_runtime_s=9600), work="/u")   # 마지막 커밋 뒤 40분
    assert res.error_class != "STALLED" and res.status == "detached"
    assert stops == []


# ── 5) HF None(조회 실패) → 미발동(모름=정지 아님) ──────────────────────────
def test_hf_none_never_stalls(monkeypatch):
    p, _, stops = _prov(monkeypatch, lambda repo: None)
    res = p._poll_and_fetch(_job(max_runtime_s=5000), work="/u")
    assert res.error_class != "STALLED" and res.status == "detached"
    assert stops == []


# ── 6) queued 동안은 시계가 안 간다(grace 는 첫 running 부터) ────────────────
def test_queued_time_does_not_count(monkeypatch):
    # 50분 queued → running. poll_start 부터 세면 즉시 stall 이지만, 첫 running 부터 세야 한다.
    # HF 는 **옛 커밋**을 돌려줘 grace 경로를 실제로 태운다(None 이면 strike 리셋으로 빠져 못 봄).
    status_fn = lambda t: "RUNNING" if t >= 1_003_000 else "QUEUED"
    old_commit = lambda repo: ("old", 1_000_000.0 - 86400)   # 하루 전(이번 run 아님)
    p, _, stops = _prov(monkeypatch, old_commit, status_fn=status_fn)
    # running 시작(≈1_003_060) 뒤 15분만 돌고 종료 → grace(45분) 안쪽 → 미발동.
    res = p._poll_and_fetch(_job(max_runtime_s=4000), work="/u")
    assert res.error_class != "STALLED" and stops == []


# ── 7) 1회 초과 뒤 새 커밋 → strike 리셋(미발동) ────────────────────────────
def test_new_commit_resets_strike(monkeypatch):
    p, clock, stops = _prov(monkeypatch, lambda repo: None)

    def commit(repo):
        now = clock["t"]
        if now >= 1_003_900:
            return ("s2", 1_003_900.0)
        if now >= 1_001_000:
            return ("s1", 1_001_000.0)
        return None
    monkeypatch.setattr(K, "latest_checkpoint_commit", commit)
    # s1 뒤 공백으로 strike 1 까지 갔다가 s2 가 와서 리셋 → deadline 전에 발동 안 함.
    res = p._poll_and_fetch(_job(max_runtime_s=4200), work="/u")
    assert res.error_class != "STALLED" and stops == []


# ── 8) env=0 → 비활성 ────────────────────────────────────────────────────────
def test_env_zero_disables(monkeypatch):
    monkeypatch.setenv("FCC_NOPROGRESS_MIN", "0")
    commit = lambda repo: ("step=0", 990_000.0)
    p, _, stops = _prov(monkeypatch, commit)
    res = p._poll_and_fetch(_job(max_runtime_s=5000), work="/u")
    assert res.error_class != "STALLED" and stops == []


# ── 9) checkpoint_repo 없음 → 비활성 ─────────────────────────────────────────
def test_no_checkpoint_repo_disables(monkeypatch):
    commit = lambda repo: ("step=0", 990_000.0)
    p, _, stops = _prov(monkeypatch, commit)
    res = p._poll_and_fetch(Job(name="np", entrypoint="x", max_runtime_s=5000), work="/u")
    assert res.error_class != "STALLED"      # 워치독 미발동(상한 도달 = TIMEOUT)
    assert res.error_class == "TIMEOUT"


# ── 10) stop 실패 → remote_alive True ────────────────────────────────────────
def test_stop_failure_sets_remote_alive(monkeypatch):
    commit = lambda repo: ("step=0", 990_000.0)
    p, _, stops = _prov(monkeypatch, commit, stop_result=(False, "못 멈춤"))
    res = p._poll_and_fetch(_job(max_runtime_s=4000), work="/u")
    assert res.error_class == "STALLED"
    assert res.remote_alive is True and res.pushed is True
    assert len(stops) == 1
