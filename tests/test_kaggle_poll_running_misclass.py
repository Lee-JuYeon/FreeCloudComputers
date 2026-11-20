"""실행 중/commit된 kaggle-ui 커널을 '종단 미분류 실패'로 오판하지 않는다.

재현(2026-11-21): kaggle-ui 로 학습을 돌릴 때, `kaggle kernels push` 가 커널의 저장된
가속기(T4x2)로 run 을 이미 띄워 UI 커밋이 skip 된다(running-but-uncommitted). 이후 폴링에서
`kaggle kernels status` 가 (1) 일시적으로 실패(rc!=0, 5xx/네트워크 — 출력에 'error' 포함)
하거나 (2) 실행 중 커널의 출력에 'error' 문자열이 우연히 섞이면, 예전 `_poll_and_fetch` 는
raw 출력을 `"error" in low` 로 부분문자열 스캔해 **살아 있는 커널을 종단 UNKNOWN 실패로**
떨어뜨렸다("미분류 오류 [ui_commit: skipped]"). 오케스트레이터만 포기(exit 1)하고 정작
커널은 Kaggle 에서 끝까지 돌아 HF 에 push 됐다.

근본 수정: 종단은 **파싱된 상태어**(KernelWorkerStatus)로만 판정. rc!=0 또는 running/queued/
미상 상태어는 종단이 아니다 → 계속 폴링. 실제 KernelWorkerStatus.ERROR 만 종단 실패.

kaggle CLI·네트워크 0 — `_sh`·`time` 만 몽키패치한다(test_poll_detached.py 와 동일 하네스).
"""
from __future__ import annotations

from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.kaggle import KaggleProvider

RUNNING = 'someone/freecloud-run has status "KernelWorkerStatus.RUNNING"'
# 실행 중인데 출력 문자열에 'error' 가 우연히 섞인 경우(이전 버전 메모 등).
RUNNING_WITH_NOISE = RUNNING + "  (note: previous version ended in error)"
# status 조회 자체가 일시적으로 실패(rc!=0). 출력에 'error' 가 있지만 커널 상태는 아니다.
TRANSIENT_FAIL = "500 Server Error: unexpected response from kaggle api (status check error)"
REAL_ERROR = 'someone/freecloud-run has status "KernelWorkerStatus.ERROR"'


def _provider(monkeypatch, status_fn):
    """status 조회를 status_fn(call_idx)->(rc, text) 로 제어. 시계는 sleep 할 때만 흐른다."""
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(K.time, "time", lambda: clock["t"])
    monkeypatch.setattr(K.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(K, "latest_checkpoint_title", lambda repo: "checkpoint step=200")
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    state = {"n": 0}

    def fake_sh(cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            state["n"] += 1
            return status_fn(state["n"])
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    return p


def _job(**kw):
    d = dict(name="run", entrypoint="python x.py", max_runtime_s=300,
             checkpoint_repo="me/ckpt")
    d.update(kw)
    return Job(**d)


def test_transient_status_query_failure_does_not_abandon_running_kernel(monkeypatch):
    # 첫 폴링에서 status 조회가 rc!=0('error' 포함)로 실패 → 예전엔 종단 UNKNOWN 으로 포기.
    # 이후엔 정상 RUNNING. 살아 있는 커널이므로 폴링 상한에서 detached 여야 한다.
    def status(n):
        return (1, TRANSIENT_FAIL) if n == 1 else (0, RUNNING)
    p = _provider(monkeypatch, status)
    res = p._poll_and_fetch(_job(), work="/unused")
    assert res.status == "detached" and res.error_class == "DETACHED"
    assert res.error_class != "UNKNOWN"


def test_incidental_error_substring_in_running_status_is_not_terminal(monkeypatch):
    # rc==0 이지만 실행 중 커널 출력에 'error' 문자열이 섞임 → 종단으로 보면 안 된다.
    p = _provider(monkeypatch, lambda n: (0, RUNNING_WITH_NOISE))
    res = p._poll_and_fetch(_job(), work="/unused")
    assert res.status == "detached" and res.error_class == "DETACHED"


def test_lifecycle_classifies_states():
    # 분류 체계 단위 검증: 종단은 complete/error 상태어뿐, 나머지는 종단 아님.
    f = KaggleProvider._kernel_lifecycle
    assert f(0, RUNNING) == "alive"
    assert f(0, RUNNING_WITH_NOISE) == "alive"          # 우연한 'error' 무시
    assert f(0, 'k has status "KernelWorkerStatus.QUEUED"') == "alive"
    assert f(0, 'k has status "KernelWorkerStatus.COMPLETE"') == "complete"
    assert f(0, REAL_ERROR) == "error"
    assert f(1, TRANSIENT_FAIL) == "unknown"            # 조회 실패 = 종단 아님
    assert f(1, 'k has status "KernelWorkerStatus.COMPLETE"') == "unknown"  # rc!=0 이면 complete 도 불신


def test_real_error_status_still_fails_terminally(monkeypatch):
    # 과잉교정 방지: 실제 KernelWorkerStatus.ERROR 는 여전히 종단 실패로 처리된다.
    p = _provider(monkeypatch, lambda n: (0, REAL_ERROR))
    monkeypatch.setattr(p, "fetch_outputs", lambda job, kid: ("/unused", REAL_ERROR))
    res = p._poll_and_fetch(_job(), work="/unused")
    assert res.ok is False and res.status == "error"
