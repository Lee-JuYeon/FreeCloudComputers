"""브라우저 프리플라이트(#8+#2+D) — push 앞에서 로그인/쿼터/가속기를 확인한다.

run() 매핑(_browser_preflight 반환 why 접두사별): LOGIN_REQUIRED→needs_setup·push0 /
QUOTA_COOLDOWN→status quota·push0 / 그 외→UI_AUTOMATION·push0. push 뒤 가속기=T4x2 면
브라우저 커밋 생략, 그 외는 _set_t4x2_and_commit 호출. 커밋 실패(비쿼터)면 stop 시도 +
remote_alive. 브라우저/네트워크 0 — 가짜 playwright 또는 _browser_preflight 몽키패치.
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


def _ui():
    return UI.KaggleUIProvider.__new__(UI.KaggleUIProvider)


def _job():
    return Job(name="pf", entrypoint="y", max_runtime_s=60)


# ══ run()-레벨: 프리플라이트 결과 매핑 + push 뒤 분기 ════════════════════════
def _base(monkeypatch, p, accel):
    monkeypatch.setattr(p, "_pw", lambda: object())
    monkeypatch.setattr(UI, "session_is_logged_in", lambda path=None: (True, "ok"))
    monkeypatch.setattr(p, "_kernel_run_signature", lambda kid: ("", ""))
    monkeypatch.setattr(p, "_browser_preflight", lambda job, ke: (True, "ok", accel))
    n = {"stage": 0, "push": 0}
    monkeypatch.setattr(p, "_stage_and_write",
                        lambda job: (n.__setitem__("stage", n["stage"] + 1), ("work", None))[1])
    monkeypatch.setattr(p, "_push",
                        lambda work, job, before=None: (n.__setitem__("push", n["push"] + 1),
                                                        (0, "pushed"))[1])
    monkeypatch.setattr(p, "_poll_and_fetch", lambda job, work: RunResult(True, "done", "ok"))
    return n


def _preflight_fail(monkeypatch, p, why):
    monkeypatch.setattr(p, "_pw", lambda: object())
    monkeypatch.setattr(UI, "session_is_logged_in", lambda path=None: (True, "ok"))
    monkeypatch.setattr(p, "_kernel_run_signature", lambda kid: ("", ""))
    monkeypatch.setattr(p, "_browser_preflight", lambda job, ke: (False, why, None))
    n = {"stage": 0, "push": 0}
    monkeypatch.setattr(p, "_stage_and_write",
                        lambda job: (n.__setitem__("stage", n["stage"] + 1), ("work", None))[1])
    monkeypatch.setattr(p, "_push",
                        lambda work, job, before=None: (n.__setitem__("push", n["push"] + 1),
                                                        (0, "pushed"))[1])
    return n


def test_preflight_login_required_no_push(monkeypatch):
    p = _ui()
    n = _preflight_fail(monkeypatch, p, "LOGIN_REQUIRED: 익명 세션 쿠키(로그인 만료)")
    res = p.run(_job())
    assert res.error_class == "LOGIN_REQUIRED" and res.status == "needs_setup"
    assert n == {"stage": 0, "push": 0}


def test_preflight_quota_no_push_status_quota(monkeypatch):
    p = _ui()
    n = _preflight_fail(monkeypatch, p, "QUOTA_COOLDOWN: GPU 쿼터 소진(31.6h)")
    res = p.run(_job())
    assert res.status == "quota" and res.error_class == "QUOTA_COOLDOWN"
    assert n == {"stage": 0, "push": 0}


def test_preflight_selector_timeout_is_ui_automation_no_push(monkeypatch):
    p = _ui()
    n = _preflight_fail(monkeypatch, p, "셀렉터 타임아웃(UI 변경 가능): selector timeout")
    res = p.run(_job())
    assert res.error_class == "UI_AUTOMATION"
    assert n == {"stage": 0, "push": 0}


def test_preflight_t4x2_skips_ui_commit(monkeypatch):
    p = _ui()
    _base(monkeypatch, p, UI.T4X2_LABEL)
    c = {"n": 0}
    monkeypatch.setattr(p, "_set_t4x2_and_commit",
                        lambda job: (c.__setitem__("n", c["n"] + 1), (True, "x"))[1])
    res = p.run(_job())
    assert res.ok is True and c["n"] == 0                    # 커밋 호출 0
    assert "ui_commit: skipped(preflight)" in res.reason and res.pushed is True


def test_preflight_p100_calls_ui_commit(monkeypatch):
    p = _ui()
    _base(monkeypatch, p, "GPU P100")
    c = {"n": 0}
    monkeypatch.setattr(p, "_set_t4x2_and_commit",
                        lambda job: (c.__setitem__("n", c["n"] + 1),
                                     (True, "committed (prev=GPU P100)"))[1])
    res = p.run(_job())
    assert res.ok is True and c["n"] == 1                    # 커밋 호출 1
    assert res.pushed is True


def test_ui_commit_failure_after_push_stops_and_remote_alive(monkeypatch):
    p = _ui()
    _base(monkeypatch, p, "GPU P100")
    monkeypatch.setattr(p, "_set_t4x2_and_commit", lambda job: (False, "셀렉터 타임아웃"))
    stops: list[str] = []
    monkeypatch.setattr(p, "stop_kernel", lambda kid: (stops.append(kid), (False, "못 멈춤"))[1])
    monkeypatch.setattr(p, "_user", lambda: "someone")
    res = p.run(_job())
    assert res.error_class == "UI_AUTOMATION"
    assert len(stops) == 1                                   # push 된 세션 중지 시도
    assert res.remote_alive is True and res.pushed is True   # 못 멈춤 → remote_alive


# ══ _browser_preflight 직접(가짜 playwright) ═══════════════════════════════
class _Page:
    def __init__(self, url, selector_raises=0):
        self._url = url
        self.selector_raises = selector_raises
        self.selector_calls = 0

    @property
    def url(self):
        return self._url

    def goto(self, url, **k):
        self._url = url

    def wait_for_selector(self, *a, **k):
        if self.selector_calls < self.selector_raises:
            self.selector_calls += 1
            raise _PWTimeout("selector timeout")
        return None


class _Ctx:
    def __init__(self, page, anon):
        self._page = page
        self._anon = anon

    def new_page(self):
        return self._page

    def cookies(self):
        return [{"name": "CLIENT-TOKEN", "value": _ct({"anon": self._anon})}]

    def close(self):
        return None


class _Browser:
    def __init__(self, ctx):
        self._ctx = ctx

    def new_context(self, **k):
        return self._ctx

    def close(self):
        return None


class _Chromium:
    def __init__(self, browser):
        self._b = browser

    def launch(self, **k):
        return self._b


class _PW:
    def __init__(self, chromium):
        self.chromium = chromium

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _install_pw(monkeypatch, page, anon):
    pw = _PW(_Chromium(_Browser(_Ctx(page, anon))))
    mod = types.ModuleType("playwright.sync_api")
    mod.sync_playwright = lambda: pw
    mod.TimeoutError = _PWTimeout
    mod.Error = Exception
    pkg = types.ModuleType("playwright")
    pkg.sync_api = mod
    monkeypatch.setitem(sys.modules, "playwright", pkg)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", mod)


def test_browser_preflight_first_run_anon_false_passes(monkeypatch):
    # 첫 실행(kernel_exists=False): /code 에서 쿠키 non-anon → 통과(에디터 404 오판 금지).
    page = _Page(url="https://www.kaggle.com/")
    _install_pw(monkeypatch, page, anon=False)
    p = _ui()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")
    ok, why, accel = p._browser_preflight(_job(), kernel_exists=False)
    assert ok is True and accel is None
    assert page.selector_calls == 0                          # 에디터 셀렉터 대기 없음


def test_browser_preflight_first_run_anon_true_login_required(monkeypatch):
    page = _Page(url="https://www.kaggle.com/")
    _install_pw(monkeypatch, page, anon=True)
    p = _ui()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")
    ok, why, accel = p._browser_preflight(_job(), kernel_exists=False)
    assert ok is False and why.startswith("LOGIN_REQUIRED")


def test_browser_preflight_selector_timeout_twice_is_ui_automation(monkeypatch):
    # kernel_exists=True·/edit·non-anon → 로그인 아님. wait_for_selector 2회 타임아웃 → UI 변경.
    page = _Page(url="https://www.kaggle.com/code/u/slug/edit", selector_raises=2)
    _install_pw(monkeypatch, page, anon=False)
    p = _ui()
    monkeypatch.setattr(p, "_user", lambda: "u")
    monkeypatch.setattr(p, "kernel_slug", lambda job: "slug")
    ok, why, accel = p._browser_preflight(_job(), kernel_exists=True)
    assert ok is False and "셀렉터 타임아웃" in why and not why.startswith("LOGIN_REQUIRED")
    assert page.selector_calls == 2                          # 1회 재시도까지 함
