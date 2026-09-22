"""Kaggle OAuth access token 자동 갱신(결함 D) — 불변조건 7.

docs/IMPROVEMENTS_2026-09-23.md §2.7:
  (a) 유효하면 그대로  (b) 만료+refresh_token → 갱신 함수 호출 → 새 토큰 + credentials.json
  갱신(mode 0600, expiration 갱신)  (c) 갱신 실패 → "" + 안내  (d) kagglesdk 없음 → ""

kagglesdk 는 **한 번도 부르지 않는다** — `_refresh_via_sdk` 하나만 몽키패치한다.
그 간접층이 있는 이유이기도 하다(엔드포인트는 SDK 가 알고, 테스트는 SDK 를 모른다).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import stat

from freecloud.providers import kaggle as K


def _creds(tmp_path, monkeypatch, *, exp_delta_s: int, token="old-tok",
           refresh="refresh-me", **extra) -> str:
    exp = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=exp_delta_s)).isoformat()
    d = tmp_path / ".kaggle"
    d.mkdir(parents=True, exist_ok=True)
    body = {"refresh_token": refresh, "access_token": token,
            "access_token_expiration": exp, "username": "someone",
            "scopes": ["kaggle"]}
    body.update(extra)
    (d / "credentials.json").write_text(json.dumps(body), encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))
    return str(d / "credentials.json")


# ── (a) 유효하면 갱신하지 않는다 ────────────────────────────────────────────
def test_valid_token_is_returned_untouched(tmp_path, monkeypatch):
    fp = _creds(tmp_path, monkeypatch, exp_delta_s=3600, token="still-good")
    called = []
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: called.append(rt) or ("x", "y"))
    assert K.oauth_access_token() == "still-good"
    assert called == []                                   # 갱신 호출 0
    assert json.load(open(fp))["access_token"] == "still-good"


# ── (b) 만료(또는 5분 이내) + refresh_token → 갱신 ──────────────────────────
def test_expired_token_is_refreshed_and_persisted(tmp_path, monkeypatch, capsys):
    fp = _creds(tmp_path, monkeypatch, exp_delta_s=-3600)
    new_exp = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)).isoformat()
    seen = {}

    def fake(rt):
        seen["rt"] = rt
        return "brand-new", new_exp
    monkeypatch.setattr(K, "_refresh_via_sdk", fake)

    assert K.oauth_access_token() == "brand-new"
    assert seen["rt"] == "refresh-me"
    saved = json.load(open(fp))
    assert saved["access_token"] == "brand-new"
    assert saved["access_token_expiration"] == new_exp
    assert saved["username"] == "someone"                 # 다른 필드를 지우면 안 된다
    assert saved["scopes"] == ["kaggle"]
    assert "갱신" in capsys.readouterr().out               # 갱신 성공 시 한 줄 로그


def test_token_expiring_within_5min_is_refreshed(tmp_path, monkeypatch):
    """아직 안 만료여도 5분 이내면 미리 받는다 — 긴 push 중에 만료되면 그게 더 비싸다."""
    _creds(tmp_path, monkeypatch, exp_delta_s=120)
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("fresh", ""))
    assert K.oauth_access_token() == "fresh"


def test_refreshed_credentials_file_is_0600(tmp_path, monkeypatch):
    fp = _creds(tmp_path, monkeypatch, exp_delta_s=-10)
    os.chmod(fp, 0o644)
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("t", ""))
    K.oauth_access_token()
    assert stat.S_IMODE(os.stat(fp).st_mode) == 0o600


def test_refresh_is_atomic_no_tmp_left_behind(tmp_path, monkeypatch):
    fp = _creds(tmp_path, monkeypatch, exp_delta_s=-10)
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("t", ""))
    K.oauth_access_token()
    assert not os.path.exists(fp + ".tmp")
    json.load(open(fp))                                   # 반쪽 JSON 이면 여기서 터진다


# ── (c) 갱신 실패 → "" + 안내 ───────────────────────────────────────────────
def test_refresh_exception_returns_empty_with_hint(tmp_path, monkeypatch, capsys):
    _creds(tmp_path, monkeypatch, exp_delta_s=-10, token="stale")
    def boom(rt):
        raise RuntimeError("network down")
    monkeypatch.setattr(K, "_refresh_via_sdk", boom)
    assert K.oauth_access_token() == ""
    assert "auth login --force" in capsys.readouterr().out


def test_refresh_empty_token_returns_empty(tmp_path, monkeypatch, capsys):
    _creds(tmp_path, monkeypatch, exp_delta_s=-10)
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("", ""))
    assert K.oauth_access_token() == ""
    assert "auth login --force" in capsys.readouterr().out


# ── (d) kagglesdk 없음 → "" (옛 동작) ───────────────────────────────────────
def test_missing_kagglesdk_falls_back_to_empty(tmp_path, monkeypatch, capsys):
    _creds(tmp_path, monkeypatch, exp_delta_s=-10)

    def no_sdk(rt):
        raise ImportError("No module named 'kagglesdk'")
    monkeypatch.setattr(K, "_refresh_via_sdk", no_sdk)
    assert K.oauth_access_token() == ""
    assert "ImportError" in capsys.readouterr().out


def test_expired_without_refresh_token_returns_empty(tmp_path, monkeypatch):
    """refresh_token 이 없으면 갱신 시도조차 하지 않는다(옛 동작 그대로)."""
    _creds(tmp_path, monkeypatch, exp_delta_s=-10, refresh="")
    called = []
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: called.append(rt) or ("x", ""))
    assert K.oauth_access_token() == ""
    assert called == []


# ── 경계: 만료시각이 없거나 깨졌으면 옛 동작(그대로 반환) ──────────────────
def test_missing_expiration_keeps_legacy_behavior(tmp_path, monkeypatch):
    _creds(tmp_path, monkeypatch, exp_delta_s=0, access_token_expiration="")
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("nope", ""))
    assert K.oauth_access_token() == "old-tok"


def test_unparsable_expiration_keeps_legacy_behavior(tmp_path, monkeypatch):
    _creds(tmp_path, monkeypatch, exp_delta_s=0, access_token_expiration="어제쯤")
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("nope", ""))
    assert K.oauth_access_token() == "old-tok"


def test_no_credentials_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert K.oauth_access_token() == ""


# ── 간접층 계약: SDK 를 부르는 것은 _refresh_via_sdk 하나뿐 ────────────────
def test_refresh_indirection_uses_sdk_not_hardcoded_url():
    import inspect
    src = inspect.getsource(K._refresh_via_sdk)
    body = src.replace(K._refresh_via_sdk.__doc__ or "", "")   # 독스트링 제외(설명은 자유)
    assert "kagglesdk" in body
    assert "http://" not in body and "https://" not in body    # 엔드포인트 하드코딩 금지
    # 갱신 경로가 credentials.json 을 SDK 의 save() 로 덮어쓰지 않는다(username 보존)
    assert "refresh_access_token" not in body
