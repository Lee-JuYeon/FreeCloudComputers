"""kaggle-ui 중복 세션 방지 — 저장된 가속기가 이미 T4 x2 면 UI 커밋을 생략한다.

실측 2026-09-30: `kaggle kernels push` 가 커널의 저장된 가속기(T4×2)로 run 을 이미 시작했는데
UI 가 또 Save & Run All → 동일 세션 2개(체크포인트 step 이 HF 에 두 번씩 push).

브라우저 없이 `_commit_on_page(page)` 에 가짜 page 를 넣어 검증한다.
"""
from __future__ import annotations

import re

from freecloud.providers import kaggle_ui as UI


class _Locator:
    def __init__(self, page, role, name):
        self.page, self.role, self.name = page, role, name

    @property
    def first(self):
        return self

    @property
    def last(self):
        return self

    def _name_str(self):
        return self.name.pattern if isinstance(self.name, re.Pattern) else str(self.name)

    def click(self, timeout=None):
        n = self._name_str()
        self.page.clicks.append((self.role, n))
        if self.role == "combobox" and "Select Accelerator" in n:
            if self.page.combo_missing:
                raise RuntimeError("combobox not found")
            return
        if self.role == "option" and n in ("None", "GPU T4 x2", "GPU P100"):
            self.page.selected = n
            return
        if self.role == "button" and n == "Save":
            self.page.saved = True
            return
        if self.role == "button" and n == "Save Version":
            return
        raise RuntimeError(f"no element {self.role}:{n}")  # Expand/confirm 없음 → 무시 경로

    def get_attribute(self, attr, timeout=None):
        if self.page.combo_missing:
            raise RuntimeError("combobox not found")
        if self.page.aria_override is not None:
            return self.page.aria_override
        return f"Select Accelerator. {self.page.selected} currently selected."

    def count(self):
        return 1

    def wait_for(self, state=None, timeout=None):
        return None


class _Page:
    def __init__(self, selected="GPU T4 x2", aria_override=None, combo_missing=False):
        self.selected = selected
        self.aria_override = aria_override
        self.combo_missing = combo_missing
        self.clicks: list = []
        self.saved = False

    def get_by_role(self, role, name=None, exact=False):
        return _Locator(self, role, name)

    def get_by_text(self, text, exact=False):
        return _Locator(self, "text", text)

    def wait_for_timeout(self, ms):
        return None

    def wait_for_selector(self, *a, **k):
        return None

    def goto(self, *a, **k):
        return None

    # 테스트 편의
    def option_clicks(self):
        return [c for c in self.clicks if c[0] == "option"]

    def save_version_clicked(self):
        return ("button", "Save Version") in self.clicks


def _prov():
    return UI.KaggleUIProvider.__new__(UI.KaggleUIProvider)


def test_already_t4x2_skips_select_and_commit(capsys):
    page = _Page(selected=UI.T4X2_LABEL)
    ok, why = _prov()._commit_on_page(page)
    assert ok is True and why.startswith("skipped")
    assert page.option_clicks() == []
    assert not page.save_version_clicked() and not page.saved
    assert "UI 커밋 생략(중복 세션 방지)" in capsys.readouterr().out


def test_other_accelerator_selects_and_commits(capsys):
    for prev in ("None", "GPU P100"):
        page = _Page(selected=prev)
        ok, why = _prov()._commit_on_page(page)
        assert ok is True and why.startswith("committed") and prev in why
        assert page.selected == UI.T4X2_LABEL
        assert page.save_version_clicked() and page.saved
        out = capsys.readouterr().out
        assert f"이전 가속기({prev})" in out


def test_unreadable_falls_back_to_commit(capsys):
    # aria-label 형식이 바뀐 경우 → 읽기 실패, 예전 동작으로
    page = _Page(selected="GPU P100", aria_override="Accelerator picker")
    # 선택 후 검증 단계는 aria-label 을 다시 읽으므로, 선택 시점에 형식을 정상화한다.
    orig = _Locator.click

    def click(self, timeout=None):
        orig(self, timeout)
        if self.role == "option":
            self.page.aria_override = None
    _Locator.click = click
    try:
        ok, why = _prov()._commit_on_page(page)
    finally:
        _Locator.click = orig
    assert ok is True and why.startswith("committed")
    assert page.saved and page.selected == UI.T4X2_LABEL
    assert "현재 가속기 읽기 실패" in capsys.readouterr().out


def test_combobox_missing_read_does_not_skip_and_select_error_propagates(capsys):
    # 읽기 예외는 삼키고(=fallback), 선택 단계의 실패는 기존대로 예외로 표면화.
    page = _Page(combo_missing=True)
    try:
        _prov()._commit_on_page(page)
    except RuntimeError:
        pass
    else:
        raise AssertionError("선택 실패가 표면화되지 않음")
    assert "현재 가속기 읽기 실패" in capsys.readouterr().out
    assert not page.saved
