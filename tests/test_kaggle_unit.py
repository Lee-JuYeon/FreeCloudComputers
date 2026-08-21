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
