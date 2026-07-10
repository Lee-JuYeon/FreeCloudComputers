from freecloud import state


def test_cooldown_roundtrip():
    state.clear_cooldown("prov-x")
    assert state.cooldown_remaining("prov-x") == 0
    state.set_cooldown("prov-x", 3600, "quota")
    rem = state.cooldown_remaining("prov-x")
    assert 3500 < rem <= 3600
    snap = state.snapshot()
    assert "prov-x" in snap and snap["prov-x"]["reason"] == "quota"
    state.clear_cooldown("prov-x")
    assert state.cooldown_remaining("prov-x") == 0


def test_expired_cooldown_is_zero():
    state.set_cooldown("prov-y", 0, "already-elapsed")
    assert state.cooldown_remaining("prov-y") == 0
    state.clear_cooldown("prov-y")
