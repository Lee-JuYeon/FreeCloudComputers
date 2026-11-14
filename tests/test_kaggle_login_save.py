"""로그인 저장 견고화(#10b·c + #13) — 실패해도 기존 세션을 덮지 않고, 쿠키 연장을 관측한다.

· save_login_state: 로그인 성공일 때만 원자적으로(tmp→chmod 0600→replace) 저장. 실패면
  기존 파일 바이트 불변. 성공 메시지에 만료일.
· _measure_cookie_extension: 라이브 __Host-KAGGLEID 가 파일보다 1h 넘게 늦고 anon 아닐 때만
  되저장. 측정 예외는 삼켜 preflight 결과를 바꾸지 않는다.
브라우저/네트워크 0 — 가짜 playwright.
"""
from __future__ import annotations

import base64
import json
import sys
import time
import types

import pytest

from freecloud.job import Job
from freecloud.providers import kaggle_ui as UI


def _ct(payload: dict) -> str:
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"h.{p}.s"


def _ui():
    return UI.KaggleUIProvider.__new__(UI.KaggleUIProvider)


# ══ save_login_state 가짜 브라우저(launch_persistent_context) ══════════════════
class _LoginPage:
    def goto(self, *a, **k):
        return None

    def wait_for_timeout(self, ms):
        return None


class _LoginCtx:
    def __init__(self, anon, cookies=None):
        self.pages: list = []
        self._anon = anon
        self._cookies = cookies

    def new_page(self):
        return _LoginPage()

    def cookies(self):
        if self._cookies is not None:
            return self._cookies
        return [{"name": "CLIENT-TOKEN", "value": _ct({"anon": self._anon})}]

    def storage_state(self, path=None):
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cookies": self.cookies()}, f)

    def close(self):
        return None


class _LoginChromium:
    def __init__(self, ctx):
        self.ctx = ctx

    def launch_persistent_context(self, *a, **k):
        return self.ctx


class _LoginPW:
    def __init__(self, ctx):
        self.chromium = _LoginChromium(ctx)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_login(monkeypatch, ctx):
    mod = types.ModuleType("playwright.sync_api")
    mod.sync_playwright = lambda: _LoginPW(ctx)
    mod.TimeoutError = TimeoutError
    mod.Error = Exception
    pkg = types.ModuleType("playwright")
    pkg.sync_api = mod
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", mod)


def test_detect_failure_preserves_existing_file(monkeypatch, tmp_path):
    state = tmp_path / "kaggle_state.json"
    original = '{"cookies": [{"name": "keep-this"}]}'
    state.write_text(original, encoding="utf-8")
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", str(state))
    _install_login(monkeypatch, _LoginCtx(anon=True))      # 영영 로그인 안 됨
    UI.save_login_state(timeout_s=6)
    assert state.read_text(encoding="utf-8") == original   # 바이트 불변 → 기존 세션 유지


def test_success_saves_0600_and_prints_expiry(monkeypatch, tmp_path, capsys):
    state = tmp_path / "kaggle_state.json"
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", str(state))
    exp = time.time() + 13 * 86400
    ctx = _LoginCtx(anon=False, cookies=[
        {"name": UI.LOGIN_COOKIE, "expires": exp},
        {"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}])
    _install_login(monkeypatch, ctx)
    UI.save_login_state(timeout_s=30)
    assert state.exists()
    assert (state.stat().st_mode & 0o777) == 0o600        # 0600 권한
    out = capsys.readouterr().out
    assert "[OK]" in out and UI._fmt_day(exp) in out       # 만료일 출력


# ══ _measure_cookie_extension ════════════════════════════════════════════════
class _MeasureCtx:
    def __init__(self, cookies):
        self._cookies = cookies

    def cookies(self):
        return self._cookies

    def storage_state(self, path=None):
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cookies": self._cookies}, f)


def _write_state(tmp_path, kaggleid_exp):
    state = tmp_path / "kaggle_state.json"
    state.write_text(json.dumps({"cookies": [
        {"name": UI.LOGIN_COOKIE, "expires": kaggleid_exp}]}), encoding="utf-8")
    return state


def test_measure_resaves_on_extension(monkeypatch, tmp_path):
    file_exp = time.time() + 5 * 86400
    state = _write_state(tmp_path, file_exp)
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", str(state))
    live_exp = file_exp + 2 * 3600        # 2h 늦음(>1h)
    ctx = _MeasureCtx([{"name": UI.LOGIN_COOKIE, "expires": live_exp},
                       {"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}])
    _ui()._measure_cookie_extension(ctx)
    saved = json.loads(state.read_text(encoding="utf-8"))
    assert UI._kaggleid_expiry(saved["cookies"]) == live_exp   # 되저장됨


@pytest.mark.parametrize("delta, anon", [
    (0, False),        # 동일 → 미저장
    (-7200, False),    # 더 이름 → 미저장
    (7200, True),      # 늦지만 anon → 미저장
])
def test_measure_no_resave(monkeypatch, tmp_path, delta, anon):
    file_exp = time.time() + 5 * 86400
    state = _write_state(tmp_path, file_exp)
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", str(state))
    original = state.read_text(encoding="utf-8")
    live_exp = file_exp + delta
    ctx = _MeasureCtx([{"name": UI.LOGIN_COOKIE, "expires": live_exp},
                       {"name": "CLIENT-TOKEN", "value": _ct({"anon": anon})}])
    _ui()._measure_cookie_extension(ctx)
    assert state.read_text(encoding="utf-8") == original       # 미저장


def test_measure_swallows_internal_errors(monkeypatch, tmp_path):
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", str(tmp_path / "kaggle_state.json"))

    class _Boom:
        def cookies(self):
            raise RuntimeError("boom")
    _ui()._measure_cookie_extension(_Boom())   # 예외를 삼킨다(raise 하지 않음)


# ══ 측정 예외 → preflight 결과 불변 ═══════════════════════════════════════════
class _PWTimeout(Exception):
    pass


class _PFPage:
    def __init__(self, url):
        self._url = url

    @property
    def url(self):
        return self._url

    def goto(self, url, **k):
        self._url = url

    def wait_for_selector(self, *a, **k):
        return None


class _PFCtx:
    def __init__(self):
        pass

    def new_page(self):
        return _PFPage("https://www.kaggle.com/code/u/slug/edit")

    def cookies(self):
        return [{"name": "CLIENT-TOKEN", "value": _ct({"anon": False})}]

    def close(self):
        return None


class _PFBrowser:
    def new_context(self, **k):
        return _PFCtx()

    def close(self):
        return None


class _PFChromium:
    def launch(self, **k):
        return _PFBrowser()


class _PFPW:
    chromium = _PFChromium()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_measure_exception_does_not_change_preflight(monkeypatch):
    mod = types.ModuleType("playwright.sync_api")
    mod.sync_playwright = lambda: _PFPW()
    mod.TimeoutError = _PWTimeout
    mod.Error = Exception
    pkg = types.ModuleType("playwright")
    pkg.sync_api = mod
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", mod)

    p = _ui()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")

    def boom(ctx):
        raise RuntimeError("measure blew up")
    monkeypatch.setattr(p, "_measure_cookie_extension", boom)
    monkeypatch.setattr(p, "_page_text", lambda page: "")        # 쿼터 없음
    monkeypatch.setattr(p, "_read_accelerator", lambda page: None)
    ok, why, accel = p._browser_preflight(Job(name="pf", entrypoint="y", max_runtime_s=60),
                                          kernel_exists=True)
    assert ok is True and accel is None                          # 측정 예외가 결과를 안 바꾼다
