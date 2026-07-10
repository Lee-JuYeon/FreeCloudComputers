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
