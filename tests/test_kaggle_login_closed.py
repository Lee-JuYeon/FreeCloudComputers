"""kaggle-login 견고화(결함 F) — 불변조건 9.

docs/IMPROVEMENTS_2026-09-23.md §2.9:
  `save_login_state` 가 TargetClosedError 에서 **트레이스백 없이** SystemExit(1) +
  안내 문자열로 끝난다.

playwright 는 설치돼 있을 수도, 아닐 수도 있다 → 어느 쪽이든 같은 테스트가 돌도록
sys.modules 에 가짜 모듈을 심는다(실제 브라우저는 절대 뜨지 않는다).
"""
from __future__ import annotations

import sys
import types

import pytest

from freecloud.providers import kaggle_ui as UI


class _FakeTargetClosedError(Exception):
    """이름이 판정 기준이다 — Playwright 의 것과 같은 이름을 쓴다."""


_FakeTargetClosedError.__name__ = "TargetClosedError"


class _FakePage:
    def __init__(self, ctx):
        self.ctx = ctx

    def goto(self, *a, **k):
        return None

    def wait_for_timeout(self, ms):
        self.ctx.ticks += 1
        if self.ctx.close_at_tick and self.ctx.ticks >= self.ctx.close_at_tick:
            raise self.ctx.closed_exc


class _FakeCtx:
    def __init__(self, closed_exc, close_at_tick=1, anon=True):
        self.pages: list = []
        self.ticks = 0
        self.closed_exc = closed_exc
        self.close_at_tick = close_at_tick
        self.anon = anon
        self.saved = False

    def new_page(self):
        return _FakePage(self)

    def cookies(self):
        # CLIENT-TOKEN(JWT) 의 anon 클레임으로 로그인 여부를 본다 — 실물과 같은 모양
        return [] if self.anon else [{"name": "CLIENT-TOKEN", "value": 'h.eyJhbm9uIjogZmFsc2V9.s'}]

    def storage_state(self, path=None):
        self.saved = True

    def close(self):
        return None


class _FakeChromium:
    def __init__(self, ctx):
        self.ctx = ctx

    def launch_persistent_context(self, *a, **k):
        return self.ctx


class _FakePW:
    def __init__(self, ctx):
        self.chromium = _FakeChromium(ctx)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install(monkeypatch, ctx):
    mod = types.ModuleType("playwright.sync_api")
    mod.sync_playwright = lambda: _FakePW(ctx)
    mod.TimeoutError = TimeoutError
    mod.Error = Exception
    pkg = types.ModuleType("playwright")
    pkg.sync_api = mod
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", mod)


def test_browser_closed_exits_with_hint_not_traceback(monkeypatch, capsys):
    ctx = _FakeCtx(_FakeTargetClosedError("Target page, context or browser has been closed"))
    _install(monkeypatch, ctx)
    with pytest.raises(SystemExit) as e:
        UI.save_login_state(timeout_s=30)
    msg = str(e.value)
    assert "브라우저가 로그인 감지 전에 닫혔습니다" in msg
    assert "창을 닫지 마세요" in msg
    assert msg == UI.BROWSER_CLOSED_MSG
    # SystemExit("문자열") → rc 1 (0 이 아니어야 한다)
    assert e.value.code != 0


def test_closed_during_cookie_poll_also_exits(monkeypatch):
    """쿠키 폴링의 except 가 닫힘을 삼켜서 타임아웃까지 헛도는 일이 없어야 한다."""
    class _Ctx(_FakeCtx):
        def cookies(self):
            raise _FakeTargetClosedError("target closed")

    ctx = _Ctx(RuntimeError("unused"), close_at_tick=0)
    _install(monkeypatch, ctx)
    with pytest.raises(SystemExit) as e:
        UI.save_login_state(timeout_s=30)
    assert str(e.value) == UI.BROWSER_CLOSED_MSG


def test_unrelated_exception_is_not_swallowed(monkeypatch):
    """브라우저 종료가 아닌 진짜 버그는 삼키지 않는다(디버깅 가능성 유지)."""
    ctx = _FakeCtx(ValueError("something else entirely"))
    _install(monkeypatch, ctx)
    with pytest.raises(ValueError):
        UI.save_login_state(timeout_s=30)


def test_successful_login_still_saves_and_reports_ok(monkeypatch, capsys):
    ctx = _FakeCtx(RuntimeError("unused"), close_at_tick=0, anon=False)
    _install(monkeypatch, ctx)
    path = UI.save_login_state(timeout_s=30)
    assert ctx.saved is True and path
    assert "[OK]" in capsys.readouterr().out


@pytest.mark.parametrize("exc,expect", [
    (_FakeTargetClosedError("x"), True),
    (Exception("Target page, context or browser has been closed"), True),
    (Exception("Browser has been closed"), True),
    (ValueError("nope"), False),
    (TimeoutError("selector timeout"), False),
])
def test_is_browser_closed_matrix(exc, expect):
    assert UI._is_browser_closed(exc) is expect


# ── L0 파일검사 browser_login_status (2026-10-24, 설계 §2·T1–T7) ──────────────
# 근본원인: __Host-KAGGLEID(14일) vs 기존 CLIENT-TOKEN JWT(30일) 검사. 파일만 보고(브라우저 0)
# 만료/부재를 LOGIN_REQUIRED 로 조기 감지한다.
import base64 as _b64         # noqa: E402
import json as _json         # noqa: E402
import time as _time         # noqa: E402

from freecloud import state as _state  # noqa: E402


def _ct(payload: dict) -> str:
    """CLIENT-TOKEN(JWT) 값 — 서명은 더미, 페이로드만 유효(우리 쿠키 상태 확인용)."""
    p = _b64.urlsafe_b64encode(_json.dumps(payload).encode()).decode().rstrip("=")
    return f"h.{p}.s"


def _write_state(tmp_path, cookies) -> str:
    p = tmp_path / "kaggle_state.json"
    p.write_text(_json.dumps({"cookies": cookies}), encoding="utf-8")
    return str(p)


def test_t1_kaggleid_expired_is_stale(tmp_path):
    now = _time.time()
    exp = now - 86400
    path = _write_state(tmp_path, [{"name": UI.LOGIN_COOKIE, "expires": exp},
                                   {"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}])
    st = UI.browser_login_status(path)
    assert st.ok is False and st.stale is True
    assert UI._fmt_day(exp) in st.detail and "fcc kaggle-login" in st.detail


def test_t2_kaggleid_13days_is_ok(tmp_path):
    now = _time.time()
    exp = now + 13 * 86400
    path = _write_state(tmp_path, [{"name": UI.LOGIN_COOKIE, "expires": exp},
                                   {"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}])
    st = UI.browser_login_status(path)
    assert st.ok is True and st.stale is False
    assert st.days_left is not None and 12.8 < st.days_left < 13.1
    assert "만료" in st.detail and "남음" in st.detail


def test_t3_no_cookie_markers_pass_with_warning(tmp_path, capsys):
    # __Host-KAGGLEID·JWT exp 둘 다 없음·non-anon → 통과 + 경고(쿠키 이름 변경 함정).
    path = _write_state(tmp_path, [{"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}])
    st = UI.browser_login_status(path)
    assert st.ok is True and st.stale is False and st.expires_at is None
    assert "둘 다 없음" in capsys.readouterr().out


def test_t4_client_token_exp_past_is_stale(tmp_path):
    now = _time.time()
    path = _write_state(tmp_path, [{"name": "CLIENT-TOKEN",
                                    "value": _ct({"anon": False, "exp": now - 100})}])
    st = UI.browser_login_status(path)
    assert st.ok is False and st.stale is True
    assert "만료" in st.detail and "fcc kaggle-login" in st.detail


def test_t5_kaggleid_within_margin_is_stale(tmp_path):
    now = _time.time()
    exp = now + 600  # 10분 — LOGIN_EXPIRY_MARGIN_S(1800) 안쪽
    path = _write_state(tmp_path, [{"name": UI.LOGIN_COOKIE, "expires": exp},
                                   {"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}])
    st = UI.browser_login_status(path)
    assert st.ok is False and st.stale is True


def test_t6_missing_file_is_stale(tmp_path):
    st = UI.browser_login_status(str(tmp_path / "does_not_exist.json"))
    assert st.ok is False and st.stale is True
    assert "fcc kaggle-login" in st.detail


def test_t7_save_login_clears_cooldown(monkeypatch):
    ctx = _FakeCtx(RuntimeError("unused"), close_at_tick=0, anon=False)
    _install(monkeypatch, ctx)
    _state.set_cooldown("kaggle-ui", 1800, "예전 LOGIN_REQUIRED")
    assert _state.cooldown_remaining("kaggle-ui") > 0
    try:
        UI.save_login_state(timeout_s=30)
        assert _state.cooldown_remaining("kaggle-ui") == 0
    finally:
        _state.clear_cooldown("kaggle-ui")
