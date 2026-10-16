"""kaggle-ui 쿼터 감지 — 소진 상태에서 거짓 성공 대신 QUOTA_COOLDOWN 으로 페일오버.

실측 2026-10-08·10-16: Kaggle 주간 GPU 쿼터 소진(에디터에 'Quota: 31:38 / 30 hrs' +
"You're almost out of quota")에서 `kaggle kernels push` 가 rc=0 이어도 새 세션을 안 띄운다.
그런데 '가속기 이미 T4x2 → UI 커밋 생략' 분기가 옛 COMPLETE 를 성공으로 읽어 거짓 성공을
보고했다. 커밋/진행 전에 페이지의 쿼터 배너를 읽어 소진이면 실패시킨다.

브라우저 없이 가짜 page(inner_text 주입)로 검증한다. 실페이지 DOM 셀렉터는 미검증.
"""
from __future__ import annotations

import re

from freecloud.errors import FAILOVER, classify
from freecloud.job import Job
from freecloud.providers import kaggle_ui as UI


# ── _quota_exhausted / _parse_quota_hours (순수 함수) ────────────────────────
def test_parse_quota_hours():
    assert UI.KaggleUIProvider._parse_quota_hours("Quota: 31:38 / 30 hrs") == 31 + 38 / 60.0
    assert UI.KaggleUIProvider._parse_quota_hours("Quota: 12:00 / 30 hrs") == 12.0
    assert UI.KaggleUIProvider._parse_quota_hours("가속기만 보임") is None


def test_quota_exhausted_over_30h():
    ex, detail = UI.KaggleUIProvider._quota_exhausted("blah Quota: 31:38 / 30 hrs blah")
    assert ex is True and "소진" in detail


def test_quota_exhausted_warning_string():
    ex, detail = UI.KaggleUIProvider._quota_exhausted("You're almost out of quota, friend")
    assert ex is True and "almost out of quota" in detail


def test_quota_not_exhausted_under_30h():
    ex, detail = UI.KaggleUIProvider._quota_exhausted("Quota: 29:59 / 30 hrs")
    assert ex is False and detail == ""


def test_quota_not_exhausted_no_banner():
    assert UI.KaggleUIProvider._quota_exhausted("ordinary editor text") == (False, "")


# ── _commit_on_page: 소진이면 커밋/스킵 없이 QUOTA_COOLDOWN 반환 ─────────────
class _QuotaPage:
    """inner_text 로 쿼터 배너를 돌려주는 최소 가짜 page. 커밋 경로를 타면 테스트 실패."""

    def __init__(self, body):
        self.body = body

    def inner_text(self, sel, timeout=None):
        return self.body

    # 아래가 호출되면 '소진인데도 진행'한 것 → 즉시 터뜨린다.
    def get_by_role(self, *a, **k):
        raise AssertionError("쿼터 소진 시 커밋/가속기 경로로 진행하면 안 된다")

    def wait_for_timeout(self, ms):
        raise AssertionError("쿼터 소진 시 진행하면 안 된다")


def _prov():
    return UI.KaggleUIProvider.__new__(UI.KaggleUIProvider)


def test_commit_on_page_aborts_when_quota_exhausted(capsys):
    page = _QuotaPage("Quota: 31:38 / 30 hrs\nYou're almost out of quota")
    ok, why = _prov()._commit_on_page(page)
    assert ok is False
    assert why.startswith("QUOTA_COOLDOWN")
    assert "쿼터 소진 감지" in capsys.readouterr().out


def test_commit_on_page_proceeds_when_quota_ok():
    # 쿼터가 넉넉하면 쿼터 게이트를 지나 기존 경로(가속기 읽기)로 간다 → get_by_role 호출됨.
    page = _QuotaPage("Quota: 10:00 / 30 hrs")
    try:
        _prov()._commit_on_page(page)
    except AssertionError as e:
        assert "진행하면 안 된다" in str(e)   # 기존 경로로 들어갔다는 증거
    else:
        raise AssertionError("쿼터 OK 인데 기존 경로로 진입하지 않았다")


# ── run(): QUOTA_COOLDOWN why → status='quota', error_class='QUOTA_COOLDOWN' ──
def test_run_maps_quota_cooldown(monkeypatch):
    p = _prov()
    monkeypatch.setattr(p, "_pw", lambda: object())
    monkeypatch.setattr(UI, "session_is_logged_in", lambda path=None: (True, "ok"))
    monkeypatch.setattr(p, "_stage_and_write", lambda job: ("work", None))
    monkeypatch.setattr(p, "_push", lambda work, job: (0, "pushed"))
    monkeypatch.setattr(p, "_set_t4x2_and_commit",
                        lambda job: (False, "QUOTA_COOLDOWN: GPU 쿼터 소진(31.63h ≥ 30h)"))
    res = p.run(Job(name="q", entrypoint="x", max_runtime_s=60))
    assert res.ok is False
    assert res.status == "quota"
    assert res.error_class == "QUOTA_COOLDOWN"
    # 오케스트레이터 분류도 FAILOVER 여야 다음 provider 로 넘어간다.
    assert classify(res.error_class + " " + res.reason).action == FAILOVER


def test_run_non_quota_ui_failure_stays_ui_automation(monkeypatch):
    p = _prov()
    monkeypatch.setattr(p, "_pw", lambda: object())
    monkeypatch.setattr(UI, "session_is_logged_in", lambda path=None: (True, "ok"))
    monkeypatch.setattr(p, "_stage_and_write", lambda job: ("work", None))
    monkeypatch.setattr(p, "_push", lambda work, job: (0, "pushed"))
    monkeypatch.setattr(p, "_set_t4x2_and_commit", lambda job: (False, "셀렉터 타임아웃"))
    res = p.run(Job(name="q", entrypoint="x", max_runtime_s=60))
    assert res.ok is False and res.error_class == "UI_AUTOMATION"
