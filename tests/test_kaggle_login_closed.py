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
