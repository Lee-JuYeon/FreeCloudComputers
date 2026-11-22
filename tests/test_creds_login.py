"""creds(AuthSpec 선언 + 단일 로그인 엔진) 테스트 — 네트워크·브라우저 0, 전부 몽키패치.

덮는 축: kaggle(CLI 로그인 판정·legacy 파일 리네임·_sh 주입 순서·kaggle.json 안 씀) ·
lightning(3종 저장 → env 없이 resolve/export) · modal(`token info`) · 인자 없는 `fcc login` ·
requires 연쇄(kaggle-ui → kaggle) · 검증 실패 시 저장 안 함.
"""
from __future__ import annotations

import datetime as dt
import inspect
import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

from freecloud import auth, cli, creds, registry, secrets
from freecloud.providers import base
from freecloud.providers.base import Probe
from freecloud.providers.kaggle import KaggleProvider
from freecloud.providers.modal import ModalProvider

_ALL_ENV = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "KAGGLE_API_TOKEN", "LIGHTNING_API_KEY",
            "LIGHTNING_USER_ID", "LIGHTNING_TEAMSPACE", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
            "SATURN_TOKEN", "SSH_CONNECTION", "SSH_TTY")


@pytest.fixture(autouse=True)
def iso(monkeypatch, tmp_path):
    """테스트별 격리: 임시 HOME·임시 secrets.json·자격증명 env 0·진짜 프롬프트 봉쇄."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("FREECLOUD_HOME", str(tmp_path / ".freecloud"))
    for n in _ALL_ENV:
        monkeypatch.delenv(n, raising=False)     # export_env 가 올린 값도 teardown 에서 지워진다

    def boom(*a, **k):
        raise AssertionError("프롬프트가 호출되면 안 된다")
    monkeypatch.setattr("builtins.input", boom)
    import getpass
    monkeypatch.setattr(getpass, "getpass", boom)
    return tmp_path


def _tty(monkeypatch, on: bool):
    monkeypatch.setattr(auth, "interactive_available", lambda: on)


def _probes(monkeypatch, table: dict):
    """creds.probe_full 을 표로 대체. 값: True/False/"NOT_INSTALLED", callable 이면 호출 시점 평가."""
    def fake(name):
        v = table[name]
        v = v() if callable(v) else v
        if v == creds.NOT_INSTALLED:
            return (False, f"{name} 미설치(pip install {name})", creds.NOT_INSTALLED)
        return (v, "ok" if v else f"{name} 미인증", "")
    monkeypatch.setattr(creds, "probe_full", fake)


def _fake_run(monkeypatch, rc=0, on_run=None):
    calls = []

    def run(cmd, *a, **k):
        calls.append(list(cmd))
        if on_run:
            on_run(cmd)
        return SimpleNamespace(returncode=rc)
    monkeypatch.setattr(creds.subprocess, "run", run)
    return calls


def _answers(monkeypatch, *ans):
    q, asked = list(ans), []

    def fake(prompt=""):
        asked.append(prompt)
        return q.pop(0)
    monkeypatch.setattr("builtins.input", fake)
    return asked


def _kaggle_files(home, *names, body="old"):
    d = home / ".kaggle"
    d.mkdir(exist_ok=True)
    for n in names:
        (d / n).write_text(body, encoding="utf-8")
    return d


# ── Spec 등록표 ──────────────────────────────────────────────────────────────
def test_등록된_provider는_전부_Spec이_있다():
    for name in registry.available_names():
        assert creds.get_spec(name) is not None, name
    assert creds.get_spec("hf") is creds.get_spec("huggingface")
    assert creds.get_spec("nope") is None
    assert not hasattr(auth, "_HANDLERS")


def test_알수없는_provider는_실패를_돌려준다():
    ok, detail = creds.login("nope")
    assert ok is False and "알 수 없는" in detail


def test_시크릿_이름은_Spec에서_유도된다(monkeypatch):
    names = creds.known_secret_names()
    for n in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "KAGGLE_API_TOKEN", "LIGHTNING_API_KEY",
              "MODAL_TOKEN_SECRET", "SATURN_TOKEN"):
        assert n in names
    assert "LIGHTNING_USER_ID" not in names and "MODAL_TOKEN_ID" not in names   # 비밀 아님
    monkeypatch.setenv("KAGGLE_API_TOKEN", "x")
    assert "KAGGLE_API_TOKEN" in secrets.names()
    monkeypatch.setitem(creds.SPECS, "newcloud",
                        creds.AuthSpec("newcloud", items=[creds.CredItem("NEWCLOUD_KEY")]))
    monkeypatch.setenv("NEWCLOUD_KEY", "y")
    assert "NEWCLOUD_KEY" in secrets.names()          # 하드코딩 목록이 아니다


def test_status_table과_status_all(monkeypatch):
    _probes(monkeypatch, {s: s != "modal" for s in creds.SPECS})
    rows = creds.status_table()
    assert [r[0] for r in rows] == list(creds.SPECS)
    assert ("modal", False, "modal 미인증", "") in rows
    st = auth.status_all()
    assert [(s.provider, s.ok) for s in st] == [(r[0], r[1]) for r in rows]


def test_resolve는_env가_저장값보다_앞이고_별칭도_본다(monkeypatch):
    secrets.set("HF_TOKEN", "stored")
    assert creds.resolve("huggingface") == {"HF_TOKEN": "stored"}
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "alias-env")
    assert creds.resolve("hf") == {"HF_TOKEN": "alias-env"}
    monkeypatch.setenv("HF_TOKEN", "env")
    assert creds.resolve("huggingface") == {"HF_TOKEN": "env"}


# ── kaggle: CLI 로그인 + probe 판정 ──────────────────────────────────────────
def test_kaggle_TTY_로그인은_CLI를_돌리고_probe로_판정한다(monkeypatch):
    _tty(monkeypatch, True)
    calls = _fake_run(monkeypatch, rc=0)
    monkeypatch.setattr(KaggleProvider, "probe", lambda self, token=None: Probe(True, "kaggle 인증 OK."))
    ok, detail = creds.login("kaggle")
    assert calls == [["kaggle", "auth", "login", "--force"]]
    assert (ok, detail) == (True, "kaggle 인증 OK.")


def test_kaggle_CLI가_rc0이어도_probe가_실패면_실패다(monkeypatch):
    _tty(monkeypatch, True)
    _fake_run(monkeypatch, rc=0)
    monkeypatch.setattr(KaggleProvider, "probe", lambda self, token=None: Probe(False, "여전히 거절"))
    assert creds.login("kaggle") == (False, "여전히 거절")


def test_kaggle_브라우저_못_여는_환경이면_no_launch_browser(monkeypatch):
    _tty(monkeypatch, True)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 1 5.6.7.8 22")
    calls = _fake_run(monkeypatch)
    _probes(monkeypatch, {"kaggle": True})
    creds.login("kaggle")
    assert calls == [["kaggle", "auth", "login", "--force", "--no-launch-browser"]]


def test_kaggle_CLI_미설치(monkeypatch):
    _tty(monkeypatch, True)

    def run(cmd, *a, **k):
        raise FileNotFoundError(cmd[0])
    monkeypatch.setattr(creds.subprocess, "run", run)
    _probes(monkeypatch, {"kaggle": False})
    ok, detail = creds.login("kaggle")
    assert ok is False and "미설치" in detail


# ── kaggle: legacy 파일 리네임(결정1) ────────────────────────────────────────
def test_리네임_TTY_y면_bak으로_바꾼다(monkeypatch, iso):
    _tty(monkeypatch, True)
    d = _kaggle_files(iso, "kaggle.json", "access_token", "access_token.txt")
    (d / "access_token.bak").write_text("older", encoding="utf-8")     # 이미 .bak 있음
    state = {"logged_in": False}
    _probes(monkeypatch, {"kaggle": lambda: state["logged_in"]})
    calls = _fake_run(monkeypatch, on_run=lambda c: state.update(logged_in=True))
    asked = _answers(monkeypatch, "y", "y", "y")

    ok, _ = creds.login("kaggle")

    assert ok and len(asked) == 3 and all("OAuth를 가립니다" in a for a in asked)
    assert not (d / "kaggle.json").exists() and (d / "kaggle.json.bak").read_text() == "old"
    assert not (d / "access_token").exists()
    assert (d / "access_token.bak").read_text() == "older"              # 기존 .bak 보존
    stamped = [p.name for p in d.iterdir() if p.name.startswith("access_token.bak.")]
    assert len(stamped) == 1 and stamped[0].split(".")[-1].isdigit()
    assert (d / "access_token.txt.bak").exists()
    assert calls == [["kaggle", "auth", "login", "--force"]]            # 리네임 뒤에 로그인


def test_리네임_TTY_n이면_그대로_둔다(monkeypatch, iso):
    _tty(monkeypatch, True)
    d = _kaggle_files(iso, "kaggle.json")
    _probes(monkeypatch, {"kaggle": False})
    _fake_run(monkeypatch)
    asked = _answers(monkeypatch, "")                                    # Enter = N
    creds.login("kaggle")
    assert len(asked) == 1
    assert (d / "kaggle.json").exists() and not (d / "kaggle.json.bak").exists()


def test_리네임_probe가_통과중이면_access_token은_묻지_않는다(monkeypatch, iso):
    _tty(monkeypatch, True)
    d = _kaggle_files(iso, "kaggle.json", "access_token", "access_token.txt")
    _probes(monkeypatch, {"kaggle": True})
    _fake_run(monkeypatch)
    asked = _answers(monkeypatch, "n")
    creds.login("kaggle")
    assert len(asked) == 1 and "kaggle.json" in asked[0]
    assert (d / "access_token").exists() and (d / "access_token.txt").exists()


def test_리네임_비TTY는_안_하고_경고만_한다(monkeypatch, iso, capsys):
    _tty(monkeypatch, False)
    d = _kaggle_files(iso, "kaggle.json", "access_token")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_new")
    seen = []
    monkeypatch.setattr(KaggleProvider, "probe",
                        lambda self, token=None: (seen.append(token), Probe(False, "거절"))[1])
    calls = _fake_run(monkeypatch)

    ok, detail = creds.login("kaggle")                                   # input 은 boom — 0회여야 함

    out = capsys.readouterr().out
    assert sorted(p.name for p in d.iterdir()) == ["access_token", "kaggle.json"]
    assert out.count("OAuth 로그인을 가릴 수 있습니다") == 2
    assert ok is False and "거절" in detail                               # 최종 판정 = probe
    assert calls == []                                                   # 브라우저 로그인 안 돌림
    assert "KAGGLE_API_TOKEN" not in secrets._load()                     # 검증 실패 → 저장 안 함


# ── kaggle: 비TTY 토큰 경로 ──────────────────────────────────────────────────
def test_kaggle_비TTY_env토큰은_주입_probe로_검증후_저장(monkeypatch):
    _tty(monkeypatch, False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "KGAT_good")
    seen = []
    monkeypatch.setattr(KaggleProvider, "probe",
                        lambda self, token=None: (seen.append(token), Probe(True, "kaggle 인증 OK."))[1])
    calls = _fake_run(monkeypatch)
    ok, detail = creds.login("kaggle")
    assert ok and detail == "kaggle 인증 OK."
    assert seen[0] == "KGAT_good"                                        # 검증은 그 값을 주입
    assert secrets._load()["KAGGLE_API_TOKEN"] == "KGAT_good"
    assert calls == []


def test_kaggle_비TTY_env토큰_없으면_즉시_SystemExit(monkeypatch):
    _tty(monkeypatch, False)
    calls = _fake_run(monkeypatch)
    with pytest.raises(SystemExit) as e:
        creds.login("kaggle")
    assert "터미널" in str(e.value) and "KAGGLE_API_TOKEN=..." in str(e.value)
    assert calls == []


def test_kaggle_probe는_token을_env로_주입한다(monkeypatch):
    seen = {}
    p = KaggleProvider()
    monkeypatch.setattr(p, "_sh", lambda cmd, t, env=None, cwd=None: (seen.update(env or {}), (0, ""))[1])
    assert p.probe(token="T").available
    assert seen["KAGGLE_API_TOKEN"] == "T"


# ── kaggle: _sh 주입 순서(env > OAuth > 저장토큰) ────────────────────────────
def _oauth_file(home, token="OAUTH"):
    exp = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).isoformat()
    d = _kaggle_files(home)
    (d / "credentials.json").write_text(json.dumps(
        {"refresh_token": "r", "access_token": token, "access_token_expiration": exp,
         "username": "someone", "scopes": []}), encoding="utf-8")


def _sh_env(monkeypatch, env=None):
    seen = {}
    monkeypatch.setattr(base.Provider, "_sh", staticmethod(
        lambda cmd, timeout, env=None, cwd=None: (seen.update(env or {}), (0, "ok"))[1]))
    KaggleProvider()._sh(["kaggle", "kernels", "list"], 10, env)
    return seen


def test_sh_순서1_env가_있으면_아무것도_주입하지_않는다(monkeypatch, iso):
    _oauth_file(iso)
    secrets.set("KAGGLE_API_TOKEN", "STORED")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "ENV")
    assert "KAGGLE_API_TOKEN" not in _sh_env(monkeypatch)                # 프로세스 env 가 그대로 쓰인다


def test_sh_순서2_OAuth가_저장토큰보다_앞이다(monkeypatch, iso):
    """거꾸로면 낡은 저장 토큰이 방금 한 OAuth 로그인을 가린다."""
    _oauth_file(iso, "OAUTH")
    secrets.set("KAGGLE_API_TOKEN", "STALE_STORED")
    assert _sh_env(monkeypatch)["KAGGLE_API_TOKEN"] == "OAUTH"


def test_sh_순서3_OAuth_없으면_저장토큰(monkeypatch, iso):
    secrets.set("KAGGLE_API_TOKEN", "STORED")
    assert _sh_env(monkeypatch)["KAGGLE_API_TOKEN"] == "STORED"
    assert _sh_env(monkeypatch, {"KAGGLE_API_TOKEN": "EXPLICIT"})["KAGGLE_API_TOKEN"] == "EXPLICIT"


def test_sh_아무것도_없으면_주입_없음(monkeypatch, iso):
    assert "KAGGLE_API_TOKEN" not in _sh_env(monkeypatch)


# ── kaggle.json 을 쓰지 않는다 ───────────────────────────────────────────────
def test_kaggle_json을_쓰는_코드가_없다(monkeypatch, iso):
    assert not hasattr(auth, "login_kaggle")
    for mod in (auth, creds):
        src = inspect.getsource(mod)
        assert "json.dump" not in src and "KAGGLE_KEY" not in src
    # 두 경로(TTY·비TTY) 다 돌려도 파일이 생기지 않는다
    _probes(monkeypatch, {"kaggle": True})
    _fake_run(monkeypatch)
    _tty(monkeypatch, True)
    creds.login("kaggle")
    _tty(monkeypatch, False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t")
    monkeypatch.setattr(creds.SPECS["kaggle"], "verify", lambda v: (True, "ok"))
    creds.login("kaggle")
    assert not (iso / ".kaggle" / "kaggle.json").exists()


# ── lightning ────────────────────────────────────────────────────────────────
_L3 = {"LIGHTNING_API_KEY": "key", "LIGHTNING_USER_ID": "uid", "LIGHTNING_TEAMSPACE": "me/ts"}


def _fake_lightning(monkeypatch, teamspace=None):
    fake = types.ModuleType("lightning_sdk")
    if teamspace is not None:
        fake.Teamspace = teamspace
    monkeypatch.setitem(sys.modules, "lightning_sdk", fake)


def test_lightning_3종_저장후_env없이_resolve_export(monkeypatch):
    _tty(monkeypatch, False)
    seen = {}

    def Teamspace(name=None):
        seen.update(name=name, env={k: os.environ.get(k) for k in _L3})
    _fake_lightning(monkeypatch, Teamspace)
    for k, v in _L3.items():
        monkeypatch.setenv(k, v)

    ok, detail = auth.login("lightning")

    assert ok, detail
    assert seen == {"name": "me/ts", "env": _L3}                         # verify = Teamspace(name=…)
    assert {k: secrets._load()[k] for k in _L3} == _L3                   # 3종 다 저장

    for k in _L3:
        monkeypatch.delenv(k)
    assert creds.resolve("lightning") == _L3                             # env 없이 저장소에서
    from freecloud.providers.lightning import LightningProvider
    assert LightningProvider().probe().available                         # probe 가 export_env 를 부른다
    assert {k: os.environ.get(k) for k in _L3} == _L3


def test_export_env는_기존_env를_덮지_않는다(monkeypatch):
    for k, v in _L3.items():
        secrets.set(k, v)
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "other/ts")
    creds.export_env("lightning")
    assert os.environ["LIGHTNING_TEAMSPACE"] == "other/ts"
    assert os.environ["LIGHTNING_API_KEY"] == "key"


def test_lightning_run도_저장값을_env로_올린다(monkeypatch):
    for k, v in _L3.items():
        secrets.set(k, v)
    _fake_lightning(monkeypatch)                                         # Job/Machine 없음 → import 실패
    from freecloud.job import Job
    from freecloud.providers.lightning import LightningProvider
    res = LightningProvider().run(Job(name="x", entrypoint="true"))
    assert res.error_class == "JOB_CONFIG"
    assert os.environ.get("LIGHTNING_API_KEY") == "key"


def test_lightning_verify_실패와_env_복원(monkeypatch):
    def Teamspace(name=None):
        raise RuntimeError("401 unauthorized")
    _fake_lightning(monkeypatch, Teamspace)
    ok, detail = creds._lightning_verify(dict(_L3))
    assert ok is False and "401" in detail
    assert all(k not in os.environ for k in _L3)                          # 임시 env 는 되돌린다
    assert creds._lightning_verify({"LIGHTNING_API_KEY": "k"})[0] is False
    monkeypatch.setitem(sys.modules, "lightning_sdk", None)               # SDK 미설치
    ok, detail = creds._lightning_verify(dict(_L3))
    assert ok is True and "3종 존재만" in detail


def test_lightning_TTY_프롬프트로_3종을_받는다(monkeypatch):
    _tty(monkeypatch, True)
    monkeypatch.setattr(auth, "_prompt_secret", lambda p: "key")
    asked = _answers(monkeypatch, "uid", "me/ts")
    monkeypatch.setattr(creds.SPECS["lightning"], "verify", lambda v: (True, "ok"))
    _probes(monkeypatch, {"lightning": True})
    assert creds.login("lightning")[0]
    assert len(asked) == 2
    assert {k: secrets._load()[k] for k in _L3} == _L3


# ── 검증 실패 시 저장 안 함 ──────────────────────────────────────────────────
def test_검증_실패면_저장하지_않는다_프롬프트경로(monkeypatch, capsys):
    _tty(monkeypatch, True)
    monkeypatch.setattr(auth, "_prompt_secret", lambda p: "bad")
    _answers(monkeypatch, "uid", "me/ts")
    monkeypatch.setattr(creds.SPECS["lightning"], "verify", lambda v: (False, "키가 틀림"))
    called = []
    monkeypatch.setattr(creds, "probe_full", lambda n: called.append(n) or (True, "ok", ""))
    ok, detail = creds.login("lightning")
    assert ok is False and "저장 안 함" in detail and "키가 틀림" in detail
    assert "저장 안 함" not in capsys.readouterr().out                    # 사유는 반환만(이중 출력 금지)
    assert secrets._load() == {}
    assert called == []                                                   # 재-probe 까지 가지 않는다


def test_검증_실패면_저장하지_않는다_HF(monkeypatch):
    _tty(monkeypatch, True)
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (False, "무효"))
    monkeypatch.setattr(auth, "_prompt_secret", lambda p: "hf_bad")
    mirrored = []
    monkeypatch.setattr(auth, "_mirror_hf_cache", lambda t: mirrored.append(t))
    ok, _ = auth.login("hf")
    assert ok is False and secrets._load() == {} and mirrored == []


def test_검증_예외도_실패로_다룬다(monkeypatch):
    _tty(monkeypatch, False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "t")

    def boom(v):
        raise RuntimeError("네트워크")
    monkeypatch.setattr(creds.SPECS["kaggle"], "verify", boom)
    ok, detail = creds.login("kaggle")
    assert ok is False and "네트워크" in detail and secrets._load() == {}


# ── modal ────────────────────────────────────────────────────────────────────
def test_modal_probe는_token_info를_부른다(monkeypatch):
    seen = []
    p = ModalProvider()
    monkeypatch.setattr(p, "_sh", lambda cmd, t, env=None, cwd=None: (seen.append(cmd), (0, ""))[1])
    assert p.probe().available
    assert seen == [["modal", "token", "info"]]
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (2, "No such command"))
    assert not p.probe().available
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (127, ""))
    assert "미설치" in p.probe().reason


def test_modal_TTY_로그인은_token_new후_재probe(monkeypatch):
    _tty(monkeypatch, True)
    calls = _fake_run(monkeypatch)
    probed = []
    monkeypatch.setattr(ModalProvider, "probe", lambda self: (probed.append(1), Probe(True, "modal 인증 OK."))[1])
    assert creds.login("modal") == (True, "modal 인증 OK.")
    assert calls == [["modal", "token", "new"]] and probed == [1]


def test_modal_비TTY는_저장하지_않고_probe로만_확인한다(monkeypatch):
    """시크릿을 `modal token set` 의 argv 로 넘기지 않는다 — modal 이 env 를 직접 읽는다."""
    _tty(monkeypatch, False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-2")
    calls = _fake_run(monkeypatch)
    _probes(monkeypatch, {"modal": True})
    ok, detail = creds.login("modal")
    assert ok and "env 로 인증됨(저장 안 함)" in detail
    assert calls == []                                                    # 어떤 명령도 안 돌린다
    assert secrets._load() == {}
    assert not hasattr(creds, "_modal_store_cmd")
    assert "native_store_cmd" not in creds.AuthSpec.__dataclass_fields__


def test_modal_비TTY_env가_틀리면_probe_결과대로_실패(monkeypatch):
    _tty(monkeypatch, False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "a")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "b")
    _fake_run(monkeypatch)
    _probes(monkeypatch, {"modal": False})
    assert creds.login("modal") == (False, "modal 미인증")


def test_modal_비TTY_env_없으면_SystemExit(monkeypatch):
    _tty(monkeypatch, False)
    monkeypatch.setenv("MODAL_TOKEN_ID", "ak-1")                          # 하나만 있음
    calls = _fake_run(monkeypatch)
    with pytest.raises(SystemExit):
        creds.login("modal")
    assert calls == []


# ── requires 연쇄 ────────────────────────────────────────────────────────────
def test_requires_kaggle_ui는_kaggle을_먼저_로그인한다(monkeypatch):
    _tty(monkeypatch, True)
    order, state = [], {"kaggle": False, "kaggle-ui": False}
    _probes(monkeypatch, {"kaggle": lambda: state["kaggle"], "kaggle-ui": lambda: state["kaggle-ui"]})
    _fake_run(monkeypatch, on_run=lambda c: (order.append("kaggle-cli"), state.update(kaggle=True)))
    monkeypatch.setattr(creds.SPECS["kaggle-ui"], "browser",
                        lambda: (order.append("browser"), state.update({"kaggle-ui": True})))
    assert creds.login("kaggle-ui")[0]
    assert order == ["kaggle-cli", "browser"]


def test_requires_선행이_실패하면_본_로그인을_하지_않는다(monkeypatch):
    _tty(monkeypatch, True)
    order = []
    _probes(monkeypatch, {"kaggle": False, "kaggle-ui": False})
    _fake_run(monkeypatch, on_run=lambda c: order.append("kaggle-cli"))
    monkeypatch.setattr(creds.SPECS["kaggle-ui"], "browser", lambda: order.append("browser"))
    ok, detail = creds.login("kaggle-ui")
    assert ok is False and "선행 'kaggle'" in detail
    assert order == ["kaggle-cli"]


def test_requires_선행이_이미_통과면_건너뛴다(monkeypatch):
    _tty(monkeypatch, True)
    order = []
    _probes(monkeypatch, {"kaggle": True, "kaggle-ui": True})
    calls = _fake_run(monkeypatch)
    monkeypatch.setattr(creds.SPECS["kaggle-ui"], "browser", lambda: order.append("browser"))
    assert creds.login("kaggle-ui")[0]
    assert calls == [] and order == ["browser"]


def test_ensure_for_job은_creds_login으로_연쇄한다(monkeypatch):
    _tty(monkeypatch, True)
    monkeypatch.setattr(registry, "get",
                        lambda name: SimpleNamespace(probe=lambda: Probe(False, "미인증")))
    logged = []
    monkeypatch.setattr(creds, "login", lambda name, interactive=None: logged.append(name) or (True, "ok"))
    _answers(monkeypatch, "y", "n")
    auth.ensure_for_job(SimpleNamespace(checkpoint_repo=""), ["kaggle-ui", "modal"])
    assert logged == ["kaggle-ui"]


# ── fcc login (인자 없음) / rc ───────────────────────────────────────────────
def _status(monkeypatch, table: dict):
    _probes(monkeypatch, table)
    monkeypatch.setattr(creds, "SPECS", {k: creds.SPECS[k] for k in table})


def test_fcc_login_인자없음_비TTY는_표만_찍고_묻지_않는다(monkeypatch, capsys):
    _tty(monkeypatch, False)
    _status(monkeypatch, {"kaggle": True, "kaggle-ui": False, "colab": False})
    logged = []
    monkeypatch.setattr(creds, "login", lambda *a, **k: logged.append(a) or (True, ""))
    rc = cli.main(["login"])                                              # input 은 boom — 0회
    out = capsys.readouterr().out
    assert rc == 1 and logged == []
    assert "OK kaggle" in out and "-- kaggle-ui" in out and "-- colab" in out


def test_fcc_login_인자없음_전부_OK면_rc0(monkeypatch):
    _tty(monkeypatch, False)
    _status(monkeypatch, {"kaggle": True, "modal": True})
    assert cli.main(["login"]) == 0


def test_fcc_login_인자없음_TTY는_실패한_것만_차례로_묻는다(monkeypatch):
    _tty(monkeypatch, True)
    state = {"huggingface": True, "kaggle": False, "kaggle-ui": False, "colab": False}
    _status(monkeypatch, {k: (lambda k=k: state[k]) for k in state})
    logged = []

    def fake_login(name, interactive=None):
        logged.append(name)
        state[name] = True
        return True, "ok"
    monkeypatch.setattr(creds, "login", fake_login)
    asked = _answers(monkeypatch, "y", "y", "n")
    rc = cli.main(["login"])
    assert logged == ["kaggle", "kaggle-ui"]                               # kaggle-ui·colab 도 묻는다(결정2)
    assert len(asked) == 3 and "[colab]" in asked[2]
    assert rc == 1                                                         # colab 은 거절 → 아직 실패


def test_fcc_login_인자없음_TTY_전부_해결되면_rc0(monkeypatch):
    _tty(monkeypatch, True)
    state = {"kaggle": False, "kaggle-ui": False}
    _status(monkeypatch, {k: (lambda k=k: state[k]) for k in state})

    def fake_login(name, interactive=None):
        state.update({"kaggle": True, "kaggle-ui": True})                  # 한 번에 둘 다 풀림
        return True, "ok"
    monkeypatch.setattr(creds, "login", fake_login)
    asked = _answers(monkeypatch, "y")
    assert cli.main(["login"]) == 0
    assert len(asked) == 1                                                 # 이미 풀린 건 다시 묻지 않는다


@pytest.mark.parametrize("ok, rc", [(True, 0), (False, 1)])
def test_fcc_login_provider_rc(monkeypatch, ok, rc):
    monkeypatch.setattr(creds, "login", lambda name, interactive=None: (ok, "detail"))
    assert cli.main(["login", "saturn"]) == rc


def test_fcc_login_알수없는_provider는_rc1(capsys):
    assert cli.main(["login", "nope"]) == 1
    assert "알 수 없는" in capsys.readouterr().out


def test_fcc_logs_안내는_fcc_login_kaggle(monkeypatch, capsys):
    monkeypatch.setattr(KaggleProvider, "_user", lambda self: "")
    assert cli.main(["logs", "somejob"]) == 1
    assert "fcc login kaggle" in capsys.readouterr().out


# ── 미설치는 미인증이 아니다 ─────────────────────────────────────────────────
def test_미설치_probe는_NOT_INSTALLED를_단다(monkeypatch):
    from freecloud.providers.colab import ColabProvider
    from freecloud.providers.lightning import LightningProvider
    from freecloud.providers.saturn import SaturnProvider
    for cls in (KaggleProvider, ModalProvider, ColabProvider):
        p = cls()
        monkeypatch.setattr(p, "_sh", lambda *a, **k: (127, "명령 없음"))
        assert p.probe().error_class == "NOT_INSTALLED", cls.__name__
    monkeypatch.setitem(sys.modules, "lightning_sdk", None)
    monkeypatch.setitem(sys.modules, "saturn_client", None)
    assert LightningProvider().probe().error_class == "NOT_INSTALLED"
    assert SaturnProvider().probe().error_class == "NOT_INSTALLED"
    # 인증 실패는 미설치가 아니다
    p = ModalProvider()
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (1, ""))
    assert p.probe().error_class == ""
    # creds 가 그대로 전달한다
    monkeypatch.setattr(ModalProvider, "_sh", staticmethod(lambda *a, **k: (127, "")))
    ok, detail, ec = creds.probe_full("modal")
    assert (ok, ec) == (False, "NOT_INSTALLED") and "미설치" in detail
    assert creds.probe("modal") == (ok, detail)


def test_미설치는_오케스트레이터에서_ABORT가_아니다():
    from freecloud.errors import classify, ABORT
    assert classify("NOT_INSTALLED").action != ABORT                     # 페일오버는 계속된다


@pytest.mark.parametrize("tty", [False, True])
def test_fcc_login_미설치만_남으면_rc0_프롬프트0(monkeypatch, capsys, tty):
    _tty(monkeypatch, tty)
    _status(monkeypatch, {"kaggle": True, "colab": "NOT_INSTALLED", "saturn": "NOT_INSTALLED"})
    logged = []
    monkeypatch.setattr(creds, "login", lambda *a, **k: logged.append(a) or (True, ""))
    rc = cli.main(["login"])                                              # input 은 boom — 0회
    out = capsys.readouterr().out
    assert rc == 0 and logged == []
    assert "·· colab" in out and "미설치(쓰려면 위 설치 명령 먼저): colab, saturn" in out
    assert "미인증" not in out


def test_fcc_login_미설치와_미인증이_섞이면_미인증만_묻는다(monkeypatch):
    _tty(monkeypatch, True)
    _status(monkeypatch, {"colab": "NOT_INSTALLED", "lightning": False})
    monkeypatch.setattr(creds, "login", lambda *a, **k: (False, "x"))
    asked = _answers(monkeypatch, "n")
    assert cli.main(["login"]) == 1
    assert len(asked) == 1 and "[lightning]" in asked[0]


# ── 저장 토큰만 있어도 username 을 안다 ──────────────────────────────────────
_CONFIG_VIEW = "Configuration values:\n- path: /x\n- username: tokuser\n- auth_method: ACCESS_TOKEN\n"


def test_user는_토큰만_있으면_config_view에서_유도하고_캐시한다(monkeypatch, iso):
    secrets.set("KAGGLE_API_TOKEN", "STORED")
    calls = []

    def fake(cmd, timeout, env=None, cwd=None):
        calls.append((list(cmd), dict(env or {})))
        return 0, _CONFIG_VIEW
    monkeypatch.setattr(base.Provider, "_sh", staticmethod(fake))
    p = KaggleProvider()
    assert p._user() == "tokuser" and p._user() == "tokuser"
    assert [c[0] for c in calls] == [["kaggle", "config", "view"]]        # 인스턴스당 1회
    assert calls[0][1]["KAGGLE_API_TOKEN"] == "STORED"                    # 저장 토큰이 주입된 채로


def test_user_순서_env_credentials_cli_kaggle_json(monkeypatch, iso):
    monkeypatch.setattr(base.Provider, "_sh", staticmethod(lambda *a, **k: (0, _CONFIG_VIEW)))
    _kaggle_files(iso, "kaggle.json", body=json.dumps({"username": "legacy", "key": "k"}))
    assert KaggleProvider()._user() == "tokuser"                           # CLI 가 kaggle.json 보다 앞
    _oauth_file(iso)
    assert KaggleProvider()._user() == "someone"                           # credentials.json 이 CLI 보다 앞
    monkeypatch.setenv("KAGGLE_USERNAME", "envuser")
    assert KaggleProvider()._user() == "envuser"
    monkeypatch.delenv("KAGGLE_USERNAME")
    (iso / ".kaggle" / "credentials.json").unlink()
    monkeypatch.setattr(base.Provider, "_sh", staticmethod(lambda *a, **k: (1, "Authentication required")))
    assert KaggleProvider()._user() == "legacy"                            # CLI 가 모르면 kaggle.json


def test_저장토큰만으로_run이_AUTH로_죽지_않는다(monkeypatch, iso):
    from freecloud.job import Job
    from freecloud.providers.base import RunResult
    secrets.set("KAGGLE_API_TOKEN", "STORED")
    monkeypatch.setattr(base.Provider, "_sh", staticmethod(lambda *a, **k: (0, _CONFIG_VIEW)))
    p = KaggleProvider()
    seen = {}
    monkeypatch.setattr(p, "_kernel_run_signature", lambda kid: (seen.update(kid=kid), ("", ""))[1])
    monkeypatch.setattr(p, "_busy_check", lambda job, before=None: RunResult(False, "error", "stop", error_class="KERNEL_BUSY"))
    res = p.run(Job(name="j", entrypoint="true"))
    assert res.error_class == "KERNEL_BUSY"                                # 'Kaggle 인증 없음'(AUTH) 관문을 넘었다
    assert seen["kid"] == "tokuser/freecloud-j"


# ── 실패 안내는 fcc login kaggle ─────────────────────────────────────────────
def test_kaggle_안내는_외부_CLI가_아니라_fcc_login(monkeypatch):
    from freecloud.providers import kaggle as K
    h = K.kaggle_auth_hint("Authentication required")
    assert "fcc login kaggle" in h and "kaggle auth login" not in h and "unset" not in h
    assert "kaggle.json 이 OAuth 인증을 가린다" in h and "통과하지 않는다" not in h      # 새 진단
    monkeypatch.setenv("KAGGLE_API_TOKEN", "old")
    h = K.kaggle_auth_hint("401 Unauthorized")
    assert "KAGGLE_API_TOKEN 값이 우선" in h and "unset" in h
    assert "kaggle auth login" not in inspect.getsource(K.oauth_access_token).split('"""')[2]


# ── 실패 메시지는 한 번만 ────────────────────────────────────────────────────
def test_fcc_login_실패_사유는_한_번만_찍힌다(monkeypatch, capsys):
    _tty(monkeypatch, False)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "bad")
    monkeypatch.setattr(creds.SPECS["kaggle"], "verify", lambda v: (False, "토큰 거절"))
    assert cli.main(["login", "kaggle"]) == 1
    assert capsys.readouterr().out.count("검증 실패(저장 안 함)") == 1


def test_래퍼는_실패_사유를_한_번_찍는다(monkeypatch, capsys):
    _tty(monkeypatch, True)
    monkeypatch.setattr(auth, "verify_hf_token", lambda t: (False, "무효"))
    monkeypatch.setattr(auth, "_prompt_secret", lambda p: "hf_bad")
    auth.login_huggingface()
    assert capsys.readouterr().out.count("검증 실패(저장 안 함)") == 1
