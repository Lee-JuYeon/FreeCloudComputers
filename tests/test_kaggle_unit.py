from freecloud.providers.kaggle import KaggleProvider, _build_kernel_script


def test_assigned_gpu_p100():
    # 실측(라이브 스모크): nvidia-smi가 'Tesla P100-PCIE-16GB'로 찍힘 → 'P100'로 정규화
    diag = "boot\nGPU 0: Tesla P100-PCIE-16GB (UUID: GPU-xxx)\n"
    assert KaggleProvider.assigned_gpu(diag) == "P100"


def test_assigned_gpu_t4x2():
    diag = ("GPU 0: Tesla T4 (UUID: GPU-a)\n"
            "GPU 1: Tesla T4 (UUID: GPU-b)\n")
    assert KaggleProvider.assigned_gpu(diag) == "T4x2"


def test_assigned_gpu_empty():
    assert KaggleProvider.assigned_gpu("no gpu here") == ""


def test_gpu_matches_short_vs_full():
    # 회귀: 요청 'P100' 은 실제 'Tesla P100-PCIE-16GB'(정규화 'P100')과 일치해야 함
    assert KaggleProvider._gpu_matches("P100", KaggleProvider.assigned_gpu(
        "GPU 0: Tesla P100-PCIE-16GB\n"))
    assert KaggleProvider._gpu_matches("T4", "T4x2")      # T4 요청 → T4x2 허용
    assert not KaggleProvider._gpu_matches("T4x2", "P100")


def test_kernel_script_injects_env_and_is_valid_python():
    env = {"HF_TOKEN": "hf_x", "FREECLOUD_CKPT_REPO": "u/r"}
    src = _build_kernel_script("python train.py", env)
    # env가 들어갔고, 중괄호 깨짐 없이 컴파일 가능해야 함
    assert "hf_x" in src and "u/r" in src
    assert "python train.py" in src
    compile(src, "<kernel>", "exec")   # SyntaxError면 실패


def test_kernel_script_handles_braces_in_values():
    # 값에 중괄호가 있어도 안전(과거 str.format 버그 회귀 방지)
    src = _build_kernel_script("echo {weird}", {"X": "a{b}c"})
    compile(src, "<kernel>", "exec")


# ── probe 거짓 양성 회귀 (2026-08-21) ────────────────────────────────────────
# 버그: probe 가 rc==127(CLI 부재)만 검사하고 rc!=0(인증 실패)은 통과시켜
#       `fcc auth` 는 "인증 OK", `fcc run` 은 AUTH 실패로 죽었다.
#       probe 는 '붙는다'를 보장해야 의미가 있다.
def test_probe_rejects_auth_failure(monkeypatch):
    from freecloud.providers.kaggle import KaggleProvider
    p = KaggleProvider()
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (1, "Authentication required to call the Kaggle API."))
    pr = p.probe()
    assert pr.available is False
    assert "kaggle auth login" in pr.reason


def test_probe_ok_only_when_rc_zero(monkeypatch):
    from freecloud.providers.kaggle import KaggleProvider
    p = KaggleProvider()
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (0, "ref  title\n---  -----"))
    assert p.probe().available is True


def test_probe_missing_cli(monkeypatch):
    from freecloud.providers.kaggle import KaggleProvider
    p = KaggleProvider()
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (127, "command not found"))
    pr = p.probe()
    assert pr.available is False and "미설치" in pr.reason


def test_auth_hint_classifies():
    from freecloud.providers.kaggle import kaggle_auth_hint
    assert "auth login" in kaggle_auth_hint("Authentication required")
    assert "403" in kaggle_auth_hint("HTTP 403 Forbidden")
    assert "429" in kaggle_auth_hint("429 Too Many Requests")
    assert "원인 불명" in kaggle_auth_hint("")


# ── OAuth 토큰 자동 주입 (2026-08-21) ────────────────────────────────────────
# `kaggle auth login` 은 ~/.kaggle/credentials.json 을 만들지만 CLI 2.2.x 가
# kernels 계열에서 이 파일을 자동으로 읽지 않는다(실측: 로그인 직후에도 거절).
# fcc 가 access_token 을 KAGGLE_API_TOKEN 으로 넣어 메운다.
def _write_creds(tmp_path, monkeypatch, token="tok-abc", exp=None):
    import json as _j, datetime as _dt
    if exp is None:
        exp = (_dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(hours=1)).isoformat()
    d = tmp_path / ".kaggle"
    d.mkdir(parents=True, exist_ok=True)
    (d / "credentials.json").write_text(_j.dumps(
        {"refresh_token": "r", "access_token": token,
         "access_token_expiration": exp, "username": "someone", "scopes": []}))
    monkeypatch.setenv("HOME", str(tmp_path))
    return d


def test_oauth_token_read(tmp_path, monkeypatch):
    from freecloud.providers import kaggle as K
    _write_creds(tmp_path, monkeypatch)
    assert K.oauth_access_token() == "tok-abc"


def test_oauth_token_expired_returns_empty(tmp_path, monkeypatch):
    # 2026-09-23부터 만료 토큰은 refresh_token 으로 자동 갱신을 시도한다(결함 D).
    # 여기서는 **갱신이 실패하는 경우**의 옛 동작(빈 문자열)을 고정한다.
    # `_refresh_via_sdk` 를 반드시 몽키패치할 것 — 안 그러면 테스트가 실제 kagglesdk
    # 네트워크 호출을 한다. 갱신 성공 경로는 tests/test_oauth_refresh.py 가 덮는다.
    import datetime as dt
    from freecloud.providers import kaggle as K
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
    _write_creds(tmp_path, monkeypatch, exp=past)
    monkeypatch.setattr(K, "_refresh_via_sdk", lambda rt: ("", ""))
    assert K.oauth_access_token() == ""


def test_sh_injects_token(tmp_path, monkeypatch):
    from freecloud.providers.kaggle import KaggleProvider
    from freecloud.providers import base
    _write_creds(tmp_path, monkeypatch, token="INJECTED")
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    seen = {}

    def fake(cmd, timeout, env=None, cwd=None):
        seen.update(env or {})
        return 0, "ok"
    monkeypatch.setattr(base.Provider, "_sh", staticmethod(fake))
    KaggleProvider()._sh(["kaggle", "kernels", "list"], 10, {"PYTHONUTF8": "1"})
    assert seen.get("KAGGLE_API_TOKEN") == "INJECTED"
    assert seen.get("PYTHONUTF8") == "1"      # 기존 env 를 지우면 안 된다


def test_sh_does_not_override_existing_env_token(tmp_path, monkeypatch):
    from freecloud.providers.kaggle import KaggleProvider
    from freecloud.providers import base
    _write_creds(tmp_path, monkeypatch, token="FROM_FILE")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "FROM_ENV")
    seen = {}
    monkeypatch.setattr(base.Provider, "_sh",
                        staticmethod(lambda c, t, env=None, cwd=None: (seen.update(env or {}), (0, "ok"))[1]))
    KaggleProvider()._sh(["kaggle", "x"], 10)
    assert "KAGGLE_API_TOKEN" not in seen     # 이미 있으면 건드리지 않는다
