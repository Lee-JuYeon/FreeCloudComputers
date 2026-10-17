"""실행 중 Kaggle 세션 Stop — Playwright 없이 가짜 page 로 로직 단위검증.

Playwright 는 실페이지 의존이라 셀렉터 자체는 여기서 검증하지 못한다. 대신
`_stop_on_page(page)` 의 **판정 로직**을 검증한다:
  · Stop 버튼을 찾아 눌러 중지되면 (True, ...)
  · Stop 컨트롤을 하나도 못 찾으면 (False, ...)  ← 안전 degrade
  · 눌렀는데 중지 신호가 없으면 (False, ...)      ← 거짓 성공 방지
그리고 kaggle.stop_kernel 의 kid 파싱/위임을 검증한다.
"""
from __future__ import annotations

from freecloud.providers import kaggle_ui as UI
from freecloud.providers.kaggle import KaggleProvider


class _Loc:
    def __init__(self, page, role, name, exact):
        self.page, self.role, self.name, self.exact = page, role, name, exact

    @property
    def first(self):
        return self

    def _is_stop(self):
        return self.role == "button" and self.name in self.page.stop_names

    def count(self):
        if self._is_stop():
            if self.page.clicked_stop and self.page.vanishes:
                return 0
            return 1
        return 0

    def click(self, timeout=None):
        self.page.clicks.append((self.role, self.name))
        if self._is_stop():
            if self.page.clicked_stop and self.page.vanishes:
                raise RuntimeError("element hidden")
            self.page.clicked_stop = True
            return
        # Expand/confirm 등 존재하지 않는 컨트롤 → 호출부가 try/except 로 무시.
        raise RuntimeError(f"no element {self.role}:{self.name}")


class _Page:
    """가짜 에디터 page.

    stop_names: 존재하는 Stop 류 버튼 이름 집합.
    vanishes:   클릭 후 Stop 버튼이 사라지는가(=중지 성공 신호).
    body:       inner_text('body') 가 돌려줄 텍스트(중지 마커 검증용).
    """

    def __init__(self, stop_names=(), vanishes=True, body=""):
        self.stop_names = set(stop_names)
        self.vanishes = vanishes
        self.body = body
        self.clicked_stop = False
        self.clicks: list = []

    def get_by_role(self, role, name=None, exact=False):
        return _Loc(self, role, name, exact)

    def wait_for_timeout(self, ms):
        return None

    def inner_text(self, selector, timeout=None):
        return self.body


def test_finds_and_stops_when_button_vanishes():
    page = _Page(stop_names={"Stop Session"}, vanishes=True)
    ok, why = UI._stop_on_page(page)
    assert ok is True
    assert "사라짐" in why and "Stop Session" in why
    assert page.clicked_stop is True


def test_stops_via_stopped_marker_when_button_stays():
    # 버튼이 안 사라져도 페이지 텍스트의 중지 마커로 성공 판정.
    page = _Page(stop_names={"Stop"}, vanishes=False, body="Session stopped. Idle.")
    ok, why = UI._stop_on_page(page)
    assert ok is True and "마커" in why


def test_no_stop_control_returns_false():
    page = _Page(stop_names=set())
    ok, why = UI._stop_on_page(page)
    assert ok is False
    assert "Stop 컨트롤 못 찾음" in why
    # 모든 후보를 시도했는지(=무한 탐색 아님) 확인.
    for name in UI._STOP_BUTTON_NAMES:
        assert name in why


def test_clicked_but_no_stop_signal_is_failure():
    # 버튼은 있어 눌렀지만 사라지지도, 마커도 없으면 '거짓 성공' 금지 → False.
    page = _Page(stop_names={"Stop Session"}, vanishes=False, body="Running...")
    ok, why = UI._stop_on_page(page)
    assert ok is False
    assert "중지" in why
    assert page.clicked_stop is True  # 클릭은 했다


def test_candidate_order_specific_first():
    # 'Stop' 만 존재해도(가장 포괄적 후보) 찾아서 누른다.
    page = _Page(stop_names={"Stop"}, vanishes=True)
    ok, _ = UI._stop_on_page(page)
    assert ok is True


def test_stop_kernel_parses_kid_and_delegates(monkeypatch):
    calls = {}

    def fake_stop(user, slug, state_path=None):
        calls["user"], calls["slug"] = user, slug
        return True, "stopped"

    monkeypatch.setattr(UI, "stop_running_session", fake_stop)
    prov = KaggleProvider()
    ok, why = prov.stop_kernel("benedictarthur/freecloud-laibrio-gemma4")
    assert ok is True
    assert calls == {"user": "benedictarthur", "slug": "freecloud-laibrio-gemma4"}
    assert "성공" in why


def test_stop_kernel_bad_kid_fails_gracefully():
    prov = KaggleProvider()
    for bad in ("noslash", "user/", "/slug", ""):
        ok, why = prov.stop_kernel(bad)
        assert ok is False
        assert "수동 Stop" in why


def test_stop_kernel_delegation_failure_degrades(monkeypatch):
    monkeypatch.setattr(UI, "stop_running_session",
                        lambda u, s, state_path=None: (False, "Stop 컨트롤 못 찾음"))
    prov = KaggleProvider()
    ok, why = prov.stop_kernel("u/s")
    assert ok is False
    assert "수동 Stop 필요" in why and "Stop 컨트롤 못 찾음" in why
