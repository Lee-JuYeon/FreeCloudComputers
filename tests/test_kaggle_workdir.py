"""workdir 업로드(결함 A) — 불변조건 1·2·3.

docs/IMPROVEMENTS_2026-09-23.md §2:
  1. Job 스키마 키는 늘리지 않는다(12개 그대로).
  2. `.git`·`__pycache__`·`*.pyc`·`.env` 는 **절대** 안 올라간다. 상한 초과 →
     RunResult(False,"error",…,"JOB_CONFIG") 이고 kaggle CLI 호출 0회.
  3. 커널 스크립트는 workdir 이 있을 때만 복사+cd 가 들어가고, 없을 때는 **바이트 동일**.

kaggle CLI 는 단 한 번도 실제로 부르지 않는다(_sh 몽키패치).
"""
from __future__ import annotations

import json
import os

import pytest

from freecloud.errors import ABORT, classify
from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.kaggle import KaggleProvider, _build_kernel_script


def _job(tmp_path, **kw):
    """workdir 를 **명시한** job(=업로드 대상). 미지정 job 은 _job_no_workdir()."""
    d = dict(name="wd", entrypoint="python train.py", workdir=str(tmp_path),
             max_runtime_s=60)
    d.update(kw)
    return Job(**d)


def _job_no_workdir(**kw):
    """workdir 키가 없는 job — Laibrio run_fcc 가 만드는 모양(2026-09-23 회귀 재현)."""
    d = dict(name="wd", entrypoint="python train.py", max_runtime_s=60)
    d.update(kw)
    return Job(**d)


def _plant(root, **files):
    """root 아래에 파일을 심는다. 키의 '/' 는 하위 디렉터리."""
    for rel, body in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)


def _provider(monkeypatch, calls, rcmap=None):
    """_sh 를 가로채 (cmd 기록, 정해둔 rc/출력) 만 돌려주는 provider."""
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")

    def fake_sh(cmd, timeout, env=None, cwd=None):
        calls.append(list(cmd))
        for key, val in (rcmap or {}).items():
            if key in " ".join(cmd):
                return val
        return 0, "ready"
    monkeypatch.setattr(p, "_sh", fake_sh)
    return p


# ── 불변조건 1: Job 스키마는 12개 키 그대로 ─────────────────────────────────
def test_job_schema_keys_unchanged():
    keys = sorted(Job.__dataclass_fields__)
    assert keys == sorted([
        "name", "entrypoint", "repo", "workdir", "checkpoint_repo", "artifacts",
        "needs", "env", "secrets", "max_runtime_s", "providers", "success_glob",
    ])
    assert len(keys) == 12


# ── 불변조건 3: workdir 없을 때 커널 스크립트 바이트 동일 ───────────────────
def test_kernel_script_without_workdir_is_byte_identical():
    """옛 사용자 회귀 0 — 2026-09-23 이전 _build_kernel_script 의 출력과 1바이트도 다르면 안 된다."""
    entrypoint, env = "python train.py", {"HF_TOKEN": "hf_x"}
    legacy = (
        "import os, json, subprocess, traceback\n"
        "os.environ.update(json.loads(%r))\n" % json.dumps(env) +
        "open('/kaggle/working/diag.txt','w').write('boot\\n')\n"
        "try:\n"
        "    subprocess.run('nvidia-smi -L >> /kaggle/working/diag.txt 2>&1', shell=True)\n"
        "    subprocess.run(%r, shell=True, check=True)\n" % entrypoint +
        "except Exception:\n"
        "    open('/kaggle/working/diag.txt','a').write(traceback.format_exc())\n"
        "    raise\n"
    )
    assert _build_kernel_script(entrypoint, env) == legacy
    assert _build_kernel_script(entrypoint, env, None) == legacy


def _legacy_kernel_script(entrypoint: str, env: dict) -> str:
    """2026-09-23 이전 `_build_kernel_script(entrypoint, env)` 의 출력 그대로."""
    return (
        "import os, json, subprocess, traceback\n"
        "os.environ.update(json.loads(%r))\n" % json.dumps(env) +
        "open('/kaggle/working/diag.txt','w').write('boot\\n')\n"
        "try:\n"
        "    subprocess.run('nvidia-smi -L >> /kaggle/working/diag.txt 2>&1', shell=True)\n"
        "    subprocess.run(%r, shell=True, check=True)\n" % entrypoint +
        "except Exception:\n"
        "    open('/kaggle/working/diag.txt','a').write(traceback.format_exc())\n"
        "    raise\n"
    )


def test_job_without_workdir_produces_byte_identical_kernel(tmp_path, monkeypatch):
    """불변조건 3(실물판): workdir 키가 **없는** job 이 만든 kernel.py 가 옛 출력과 바이트 동일.

    2026-09-23 라이브 회귀: Job.workdir 기본값이 "." 이라 '명시 안 함'과 'cwd 를 올려라'를
    구분하지 못했고, workdir 키 없는 Laibrio job 의 커널에까지 복사 코드가 들어가
    `shutil.copytree('/kaggle/input/freecloud-...-workdir', ...)` → FileNotFoundError 로
    죽었다(JOB_CONFIG ABORT). 함수 단위가 아니라 **생성된 파일**로 고정한다.
    """
    monkeypatch.chdir(tmp_path)                 # cwd 에 파일이 있어도 올라가면 안 된다
    _plant(str(tmp_path), **{"train.py": "print(1)", "data.bin": "x" * 1000})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls)
    job = _job_no_workdir()
    work, err = p._stage_and_write(job)
    assert err is None
    assert calls == []                          # 데이터셋 CLI 호출 0회

    produced = open(os.path.join(work, "kernel.py"), encoding="utf-8").read()
    assert produced == _legacy_kernel_script(job.entrypoint, job.resolved_env())
    assert "/kaggle/input/" not in produced
    assert "copytree" not in produced
    kmeta = json.load(open(os.path.join(work, "kernel-metadata.json")))
    assert kmeta["dataset_sources"] == []


def test_job_load_leaves_unset_workdir_empty(tmp_path):
    """Job.load 가 빈 workdir 를 abspath 해서 cwd 로 둔갑시키면 안 된다."""
    y = tmp_path / "job.yaml"
    y.write_text("name: demo\nentrypoint: \"python x.py\"\n", encoding="utf-8")
    assert Job.load(str(y)).workdir == ""
    assert Job.__dataclass_fields__["workdir"].default == ""    # 기본값도 "" 여야 한다


def test_job_load_abspaths_explicit_workdir(tmp_path):
    (tmp_path / "src").mkdir()
    y = tmp_path / "job.yaml"
    y.write_text("name: demo\nentrypoint: \"python x.py\"\nworkdir: \"src\"\n",
                 encoding="utf-8")
    loaded = Job.load(str(y)).workdir
    assert os.path.isabs(loaded) and loaded.endswith("src")


def test_kernel_script_with_workdir_copies_and_cds():
    src = _build_kernel_script("python train.py", {}, "freecloud-wd-workdir")
    compile(src, "<kernel>", "exec")                 # SyntaxError 면 실패
    assert "/kaggle/input/freecloud-wd-workdir" in src
    assert "_dst = '/kaggle/working/workdir'" in src
    assert "os.chdir(_dst)" in src
    assert "os.environ['FREECLOUD_WORKDIR'] = _dst" in src
    # 복사·cd 는 entrypoint **앞**에 와야 한다(그래야 상대경로 entrypoint 가 산다)
    assert src.index("os.chdir") < src.index("python train.py")


# ── 불변조건 2: 제외 규칙 ───────────────────────────────────────────────────
def test_secrets_and_junk_never_staged(tmp_path):
    _plant(str(tmp_path), **{
        "train.py": "print(1)", ".env": "HF_TOKEN=secret",
        "pkg/mod.py": "x=1", "pkg/mod.pyc": "junk",
        "__pycache__/mod.cpython-311.pyc": "junk", ".git/config": "[core]",
        ".git/objects/ab/cdef": "blob",
    })
    rels = [rel for _, rel in K.collect_workdir_files(str(tmp_path))]
    assert sorted(rels) == sorted(["train.py", os.path.join("pkg", "mod.py")])
    assert not any(".env" in r or ".git" in r or r.endswith(".pyc") or
                   "__pycache__" in r for r in rels)


def test_extra_ignore_globs(tmp_path, monkeypatch):
    _plant(str(tmp_path), **{"a.py": "1", "big.bin": "x", "data/train.csv": "y"})
    monkeypatch.setenv("FREECLOUD_WORKDIR_IGNORE", "*.bin, data")
    rels = [rel for _, rel in K.collect_workdir_files(str(tmp_path))]
    assert rels == ["a.py"]


def test_dotenv_excluded_even_if_user_ignores_nothing(tmp_path, monkeypatch):
    """`.env` 는 사용자가 뭘 하든 항상 빠진다 — 시크릿 유출 방지(설계 §1.A)."""
    monkeypatch.setenv("FREECLOUD_WORKDIR_IGNORE", "")
    _plant(str(tmp_path), **{".env": "K=V", "ok.txt": "hi"})
    assert [rel for _, rel in K.collect_workdir_files(str(tmp_path))] == ["ok.txt"]


# ── 불변조건 2: 상한 초과 → JOB_CONFIG, CLI 호출 0 ─────────────────────────
def test_oversize_workdir_fails_before_any_cli_call(tmp_path, monkeypatch):
    _plant(str(tmp_path), **{"big.bin": "x" * 200_000})
    monkeypatch.setenv("FREECLOUD_WORKDIR_MAX_MB", "0.1")     # 100KB 상한
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls)
    res = p.run(_job(tmp_path))
    assert res.ok is False and res.status == "error"
    assert res.error_class == "JOB_CONFIG"
    assert "too large" in res.reason
    assert calls == []                                        # kaggle CLI 0회


def test_zero_max_mb_disables_upload(tmp_path, monkeypatch):
    _plant(str(tmp_path), **{"a.py": "1"})
    monkeypatch.setenv("FREECLOUD_WORKDIR_MAX_MB", "0")
    p = _provider(monkeypatch, [])
    ref, note = p._stage_workdir(_job(tmp_path))
    assert ref is None and "끔" in note


# ── 업로드 경로: create → (있으면) version, 그리고 dataset_sources 배선 ─────
def test_stage_creates_private_dataset_and_wires_sources(tmp_path, monkeypatch):
    _plant(str(tmp_path), **{"train.py": "print(1)", "pkg/mod.py": "x=1"})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls)
    work, err = p._stage_and_write(_job(tmp_path))
    assert err is None

    create = [c for c in calls if c[:3] == ["kaggle", "datasets", "create"]]
    assert len(create) == 1
    assert "-r" in create[0] and create[0][create[0].index("-r") + 1] == "zip"
    assert "-u" not in create[0] and "--public" not in create[0]   # private 기본 유지
    stage_dir = create[0][create[0].index("-p") + 1]
    meta = json.load(open(os.path.join(stage_dir, "dataset-metadata.json")))
    assert meta["id"] == "someone/freecloud-wd-workdir"
    assert meta["licenses"] == [{"name": "other"}]
    assert not os.path.exists(os.path.join(stage_dir, ".env"))

    # 커널 메타에 데이터셋이 붙고, 커널 스크립트가 그걸 workdir 로 복사한다
    kmeta = json.load(open(os.path.join(work, "kernel-metadata.json")))
    assert kmeta["dataset_sources"] == ["someone/freecloud-wd-workdir"]
    assert kmeta["is_private"] is True
    src = open(os.path.join(work, "kernel.py"), encoding="utf-8").read()
    assert "/kaggle/input/freecloud-wd-workdir" in src


def test_stage_falls_back_to_version_when_dataset_exists(tmp_path, monkeypatch):
    _plant(str(tmp_path), **{"train.py": "1"})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls,
                  rcmap={"datasets create": (1, "ERROR: dataset already exists")})
    ref, _note = p._stage_workdir(_job(tmp_path))
    assert ref == "someone/freecloud-wd-workdir"
    verbs = [c[2] for c in calls if c[:2] == ["kaggle", "datasets"]]
    assert verbs[:3] == ["create", "version", "status"]      # 순서가 계약


def test_stage_upload_failure_aborts_before_push(tmp_path, monkeypatch):
    """업로드 실패 → JOB_CONFIG(ABORT), **커널 push 는 하지 않는다**.

    push 해버리면 코드 없는 노드에서 돌거나, 페일오버해서 조용히 옛 코드로 학습한다.
    """
    _plant(str(tmp_path), **{"train.py": "1"})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls,
                  rcmap={"datasets create": (1, "500 internal"),
                         "datasets version": (1, "401 Unauthorized")})
    res = p.run(_job(tmp_path))
    assert res.ok is False and res.status == "error"
    assert res.error_class == "JOB_CONFIG"
    assert classify(res.error_class + " " + res.reason).action == ABORT
    assert not any(c[:3] == ["kaggle", "kernels", "push"] for c in calls)


def test_stage_not_ready_aborts_before_push(tmp_path, monkeypatch):
    """데이터셋이 ready 가 아니면 붙여봐야 커널이 빈손이다 → push 전에 끝낸다."""
    _plant(str(tmp_path), **{"train.py": "1"})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls,
                  rcmap={"datasets status": (0, "error: processing failed")})
    res = p.run(_job(tmp_path))
    assert res.ok is False and res.error_class == "JOB_CONFIG"
    assert not any(c[:3] == ["kaggle", "kernels", "push"] for c in calls)


def test_copy_code_and_dataset_sources_never_diverge(tmp_path, monkeypatch):
    """커널의 '복사 코드' 유무와 dataset_sources 는 **항상 같이** 간다(회귀의 핵심)."""
    p = _provider(monkeypatch, [])
    for ref in ("", "someone/freecloud-wd-workdir"):
        work = p._write_kernel_dir(_job(tmp_path), dataset_ref=ref)
        src = open(os.path.join(work, "kernel.py"), encoding="utf-8").read()
        meta = json.load(open(os.path.join(work, "kernel-metadata.json")))
        assert ("/kaggle/input/" in src) is bool(ref)
        assert bool(meta["dataset_sources"]) is bool(ref)


def test_repo_job_skips_workdir_upload(tmp_path, monkeypatch):
    _plant(str(tmp_path), **{"train.py": "1"})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls)
    ref, note = p._stage_workdir(_job(tmp_path, repo="https://example.com/x.git"))
    assert ref is None and "repo=" in note
    assert calls == []


def test_unset_workdir_skips_upload(tmp_path, monkeypatch):
    """workdir 키가 없으면 cwd 에 파일이 있어도 아무것도 올리지 않는다."""
    monkeypatch.chdir(tmp_path)
    _plant(str(tmp_path), **{"train.py": "1"})
    calls: list[list[str]] = []
    p = _provider(monkeypatch, calls)
    ref, note = p._stage_workdir(_job_no_workdir())
    assert ref is None and "미지정" in note
    assert calls == []


def test_empty_workdir_skips_upload(tmp_path, monkeypatch):
    p = _provider(monkeypatch, [])
    ref, note = p._stage_workdir(_job(tmp_path))
    assert ref is None and "없음" in note


@pytest.mark.parametrize("bad", ["/nope/not/here"])
def test_missing_workdir_skips_upload(bad, monkeypatch, tmp_path):
    p = _provider(monkeypatch, [])
    ref, _ = p._stage_workdir(_job(tmp_path, workdir=bad))
    assert ref is None


# ── kaggle-ui 도 같은 경로를 탄다(설계 §3: 부모 헬퍼 재사용) ────────────────
def test_kaggle_ui_reuses_parent_staging(tmp_path, monkeypatch):
    from freecloud.providers.kaggle_ui import KaggleUIProvider
    _plant(str(tmp_path), **{"train.py": "1"})
    calls: list[list[str]] = []
    p = KaggleUIProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")
    monkeypatch.setattr(p, "_sh", lambda cmd, t, env=None, cwd=None:
                        (calls.append(list(cmd)), (0, "ready"))[1])
    work, err = p._stage_and_write(_job(tmp_path))
    assert err is None
    kmeta = json.load(open(os.path.join(work, "kernel-metadata.json")))
    assert kmeta["dataset_sources"] == ["someone/freecloud-wd-workdir"]
