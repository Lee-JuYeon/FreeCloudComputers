"""kaggle-ui 브라우저 로그인 만료 조기 감지(LOGIN_REQUIRED) — provider·L1/L2(설계 §2·§3·T8–T13).

근본원인: __Host-KAGGLEID(14일) vs 기존 CLIENT-TOKEN JWT(30일) 검사. L0(파일), L1(goto 직후 쿠키/URL),
L2(타임아웃 재판정) 세 층이 '로그인 만료'를 UI 변경(UI_AUTOMATION)과 분리해 LOGIN_REQUIRED(ABORT)로 낸다.

브라우저·GPU 0 — 가짜 playwright 를 sys.modules 에 심는다(test_kaggle_login_closed 패턴 재사용).
"""
from __future__ import annotations

import base64
import json
import sys
import types

from freecloud.job import Job
from freecloud.providers import kaggle_ui as UI
from freecloud.providers.base import RunResult


class _PWTimeout(Exception):
    pass


def _ct(payload: dict) -> str:
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"h.{p}.s"


def _stale_state_file(tmp_path) -> str:
    import time
    p = tmp_path / "kaggle_state.json"
    p.write_text(json.dumps({"cookies": [
        {"name": UI.LOGIN_COOKIE, "expires": time.time() - 86400},
        {"name": "CLIENT-TOKEN", "value": _ct({"anon": False})},
    ]}), encoding="utf-8")
    return str(p)


def _prov():
    return UI.KaggleUIProvider.__new__(UI.KaggleUIProvider)


# ── 가짜 playwright(_set_t4x2_and_commit / stop_running_session 용) ────────────
class _FakePage:
    def __init__(self, url, raise_timeout=False, url_after_timeout=None):
        self._url = url
        self.raise_timeout = raise_timeout
        self._url_after = url_after_timeout
        self.wait_selector_called = False

    @property
    def url(self):
        return self._url

    def goto(self, url, **k):
        return None

    def wait_for_selector(self, *a, **k):
        self.wait_selector_called = True
        if self.raise_timeout:
            # SPA 가 로그인 만료로 404 로 리다이렉트된 뒤에야 셀렉터 대기가 타임아웃난다 —
            # goto 직후(L1)엔 URL 이 아직 /edit 였다가 타임아웃 시점(L2)에 바뀌는 상황.
            if self._url_after is not None:
                self._url = self._url_after
            raise _PWTimeout("selector timeout")
        return None


class _FakeCtx:
    def __init__(self, page, anon):
        self._page = page
        self._anon = anon

    def new_page(self):
        return self._page

    def cookies(self):
        # anon=True 는 실물처럼 CLIENT-TOKEN 의 anon 클레임으로 표현한다(_cookie_says_anon 가 봄).
        return [{"name": "CLIENT-TOKEN", "value": _ct({"anon": self._anon})}]

    def close(self):
        return None


class _FakeBrowser:
    def __init__(self, ctx):
        self._ctx = ctx

    def new_context(self, **k):
        return self._ctx

    def close(self):
        return None


class _FakeChromium:
    def __init__(self, browser):
        self._b = browser

    def launch(self, **k):
        return self._b


class _FakePW:
    def __init__(self, chromium):
        self.chromium = chromium

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_pw(monkeypatch, page, anon):
    ctx = _FakeCtx(page, anon)
    pw = _FakePW(_FakeChromium(_FakeBrowser(ctx)))
    mod = types.ModuleType("playwright.sync_api")
    mod.sync_playwright = lambda: pw
    mod.TimeoutError = _PWTimeout
    mod.Error = Exception
    pkg = types.ModuleType("playwright")
    pkg.sync_api = mod
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", mod)
    return ctx


def _job():
    return Job(name="x", entrypoint="y", max_runtime_s=60)


# ── T8: run() L0 stale → LOGIN_REQUIRED, 비싼 경로 미진입 ─────────────────────
def test_t8_run_stale_file_is_login_required(monkeypatch, tmp_path):
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", _stale_state_file(tmp_path))
    p = _prov()
    monkeypatch.setattr(p, "_pw", lambda: object())

    def boom(*a, **k):
        raise AssertionError("L0 실패인데 비싼 경로(_stage_and_write/_push)로 진입했다")

    monkeypatch.setattr(p, "_stage_and_write", boom)
    monkeypatch.setattr(p, "_push", boom)
    res = p.run(_job())
    assert res.ok is False and res.status == "needs_setup"
    assert res.error_class == "LOGIN_REQUIRED"
    assert "fcc kaggle-login" in res.reason


# ── T9: probe() stale → available False, error_class LOGIN_REQUIRED ───────────
def test_t9_probe_stale_is_login_required(monkeypatch, tmp_path):
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", _stale_state_file(tmp_path))
    # 상위(KaggleProvider) probe 는 kaggle CLI 를 보지만, 여기선 로그인 게이트만 검증한다.
    monkeypatch.setattr(UI.KaggleProvider, "probe",
                        lambda self: __import__("freecloud.providers.base", fromlist=["Probe"]).Probe(True, "base ok"))
    p = _prov()
    monkeypatch.setattr(p, "_pw", lambda: object())
    pr = p.probe()
    assert pr.available is False and pr.error_class == "LOGIN_REQUIRED"
    assert "fcc kaggle-login" in pr.reason


# ── T10: L1 goto 후 /edit 탈락 + anon → LOGIN_REQUIRED, wait_for_selector 미호출 ─
def test_t10_l1_redirect_and_anon(monkeypatch):
    page = _FakePage(url="https://www.kaggle.com/")  # /edit 없음(404 리다이렉트)
    _install_pw(monkeypatch, page, anon=True)
    p = _prov()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")
    ok, why = p._set_t4x2_and_commit(_job())
    assert ok is False and why.startswith("LOGIN_REQUIRED")
    assert page.wait_selector_called is False   # 비싼 60s 대기 전에 끊었다


# ── T11: url 정상·non-anon·PWTimeout → 셀렉터 타임아웃(로그인 아님) → run 매핑 UI_AUTOMATION ─
def test_t11_timeout_without_login_stays_ui_automation(monkeypatch):
    page = _FakePage(url="https://www.kaggle.com/code/u/slug/edit", raise_timeout=True)
    _install_pw(monkeypatch, page, anon=False)
    p = _prov()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")
    ok, why = p._set_t4x2_and_commit(_job())
    assert ok is False and "셀렉터 타임아웃" in why and not why.startswith("LOGIN_REQUIRED")
    # run 매핑: 로그인 아닌 타임아웃은 UI_AUTOMATION 으로 남는다(분리 검증).
    monkeypatch.setattr(p, "_pw", lambda: object())
    monkeypatch.setattr(UI, "session_is_logged_in", lambda path=None: (True, "ok"))
    monkeypatch.setattr(p, "_stage_and_write", lambda job: ("work", None))
    monkeypatch.setattr(p, "_push", lambda work, job: (0, "pushed"))
    monkeypatch.setattr(p, "_set_t4x2_and_commit", lambda job: (False, why))
    res = p.run(_job())
    assert res.ok is False and res.error_class == "UI_AUTOMATION"


# ── T12: PWTimeout + anon → L2 재판정 → LOGIN_REQUIRED ────────────────────────
def test_t12_timeout_with_anon_is_login_required(monkeypatch):
    # L1 통과(goto 직후엔 /edit·non-anon)했다가 타임아웃 시점에 404 로 리다이렉트 → L2 가 잡는다.
    page = _FakePage(url="https://www.kaggle.com/code/u/slug/edit", raise_timeout=True,
                     url_after_timeout="https://www.kaggle.com/")
    _install_pw(monkeypatch, page, anon=False)
    p = _prov()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")
    ok, why = p._set_t4x2_and_commit(_job())
    assert ok is False and why.startswith("LOGIN_REQUIRED")


# ── T13: 유효 상태 → run 이 _poll_and_fetch 까지 진행(해피패스 불변) ──────────
def test_t13_valid_session_reaches_poll(monkeypatch):
    p = _prov()
    monkeypatch.setattr(p, "_pw", lambda: object())
    monkeypatch.setattr(UI, "session_is_logged_in", lambda path=None: (True, "ok"))
    monkeypatch.setattr(p, "_stage_and_write", lambda job: ("work", None))
    monkeypatch.setattr(p, "_push", lambda work, job: (0, "pushed"))
    monkeypatch.setattr(p, "_set_t4x2_and_commit", lambda job: (True, "skipped: already T4x2"))
    reached = {}

    def fake_poll(job, work):
        reached["poll"] = (job, work)
        return RunResult(True, "done", "ok")

    monkeypatch.setattr(p, "_poll_and_fetch", fake_poll)
    res = p.run(_job())
    assert reached.get("poll") is not None    # _poll_and_fetch 까지 진행
    assert res.ok is True and "ui_commit: skipped" in res.reason
