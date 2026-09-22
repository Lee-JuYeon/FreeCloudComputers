from freecloud.errors import classify, RETRY, FAILOVER, ABORT


def test_quota_cooldown_failover():
    d = classify("Service Unavailable: TooManyAssignments")
    assert d.error_class == "QUOTA_COOLDOWN"
    assert d.action == FAILOVER
    assert d.cooldown_s > 0


def test_session_lost_failover():
    assert classify("Connection was lost to the runtime").action == FAILOVER


def test_auth_aborts():
    d = classify("401 Unauthorized: gated repo")
    assert d.error_class == "AUTH"
    assert d.action == ABORT


def test_oom_failover():
    assert classify("CUDA out of memory").error_class == "OOM"


def test_arch_mismatch():
    assert classify("segmentation fault (core dumped)").error_class == "ARCH_MISMATCH"


def test_transient_retry():
    assert classify("connection reset, please try again").action == RETRY


def test_unknown_defaults_to_failover():
    d = classify("some totally novel error text zzz")
    assert d.error_class == "UNKNOWN"
    assert d.action == FAILOVER


# ── ENTRYPOINT_ERROR / UNKNOWN 쿨다운 (2026-09-23, 불변조건 8) ───────────────
# 결함 E: entrypoint 가 TypeError 로 죽었는데 UNKNOWN→FAILOVER 로 분류돼
#         kaggle-ui 에 30분 쿨다운이 걸리고 다음 라운드를 60초씩 3번 돌았다.
#         코드 버그는 어느 provider 로 가도 똑같이 죽는다 — 쿼터·쿨다운만 태운다.
_TRACEBACK = ('Traceback (most recent call last):\n'
              '  File "/kaggle/working/train.py", line 42, in <module>\n'
              '    main()\n'
              '  File "/kaggle/working/train.py", line 30, in main\n'
              '    x = a + b\n'
              'TypeError: unsupported operand type(s) for +: \'int\' and \'str\'\n')


def test_entrypoint_traceback_aborts():
    d = classify(_TRACEBACK)
    assert d.error_class == "ENTRYPOINT_ERROR"
    assert d.action == ABORT
    assert d.cooldown_s == 0          # ABORT 는 쿨다운을 걸지 않는다


import pytest  # noqa: E402


@pytest.mark.parametrize("exc", [
    "TypeError", "ValueError", "KeyError", "AttributeError", "AssertionError",
    "IndexError", "NameError", "ZeroDivisionError", "RuntimeError", "SystemExit",
])
def test_entrypoint_exception_names_all_abort(exc):
    log = f"Traceback (most recent call last):\n  File \"x.py\", line 1\n{exc}: boom\n"
    assert classify(log).action == ABORT


def test_oom_wins_over_entrypoint_error():
    """같은 로그에 CUDA OOM 이 있으면 OOM 우선 — VRAM 문제는 더 큰 GPU 에서 산다."""
    log = "CUDA out of memory. Tried to allocate 2.00 GiB\n" + _TRACEBACK
    d = classify(log)
    assert d.error_class == "OOM"
    assert d.action == FAILOVER


def test_quota_wins_over_entrypoint_error():
    log = "Service Unavailable: TooManyAssignments\n" + _TRACEBACK
    assert classify(log).error_class == "QUOTA_COOLDOWN"


def test_traceback_without_known_exception_stays_unknown():
    log = ("Traceback (most recent call last):\n  File \"x.py\", line 1\n"
           "SomeVendorSpecificError: weird\n")
    assert classify(log).error_class == "UNKNOWN"


def test_exception_line_without_traceback_is_not_entrypoint_error():
    """'TypeError' 라는 낱말이 지나가기만 한 로그를 ABORT 로 잡으면 안 된다."""
    assert classify("note: a TypeError can happen here").error_class == "UNKNOWN"


def test_unknown_cooldown_is_300s():
    """미분류를 30분(1800s) 잠그는 건 과하다 — 300초로 줄였다(설계 §1.E)."""
    assert classify("some totally novel error text zzz").cooldown_s == 300


def test_no_artifact_aborts():
    """커널은 끝났는데 산출물 0 → 어느 provider 로 가도 빈손이다(설계 §1.B)."""
    d = classify("NO_ARTIFACT: success_glob 매치 0")
    assert d.error_class == "NO_ARTIFACT"
    assert d.action == ABORT


# ── NO_SLOT vs OOM 오분류 회귀 (2026-09-23 Kaggle 라이브 실측) ───────────────
# 커널이 torch OOM 으로 죽었는데 fcc 가 NO_SLOT 으로 분류했다 — NO_SLOT 정규식의
# 맨몸 `capacity` 가 OOM 메시지의 "total capacity of" 에 먼저 매치된 탓.
# OOM 은 더 큰 VRAM 의 provider 에서 살 수 있으므로 action 은 FAILOVER 유지.
_LIVE_TORCH_OOM = (
    "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 510.00 MiB. "
    "GPU 0 has a total capacity of 14.56 GiB of which 12.00 MiB is free.")


def test_live_torch_oom_is_oom_not_no_slot():
    d = classify(_LIVE_TORCH_OOM)
    assert d.error_class == "OOM"
    assert d.action == FAILOVER


def test_live_torch_oom_inside_traceback_is_still_oom():
    assert classify(_TRACEBACK + _LIVE_TORCH_OOM).error_class == "OOM"


@pytest.mark.parametrize("log", [
    "no slot available right now",
    "all gpus busy",
    "queue too long",
    "cluster is at capacity",
    "slot request wait 20m",
])
def test_no_slot_cases_still_classify_as_no_slot(log):
    d = classify(log)
    assert d.error_class == "NO_SLOT"
    assert d.action == FAILOVER and d.cooldown_s == 900


@pytest.mark.parametrize("log", ["TypeError: boom", "zoom call failed", "loom render"])
def test_oom_needs_word_boundary(log):
    """'boom'·'zoom'·'loom' 안의 oom 을 OOM 으로 잡으면 안 된다(맨몸 `oom` 회귀)."""
    assert classify(log).error_class != "OOM"
