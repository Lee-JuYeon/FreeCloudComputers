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
