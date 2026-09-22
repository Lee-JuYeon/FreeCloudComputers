"""산출물 회수(결함 B) — 불변조건 4·5.

docs/IMPROVEMENTS_2026-09-23.md §2:
  4. `_poll_and_fetch` 는 complete/error **양쪽에서** 산출물 디렉터리를 만들고
     `RunResult.artifacts_dir` 를 채운다. success_glob 매치 0 → NO_ARTIFACT ABORT.
  5. `RunResult` 에 `artifacts_dir: str = ""` 추가(기존 위치인자 생성자 호환).

kaggle CLI·네트워크 0 — `_sh` 와 `time.sleep` 만 몽키패치한다.
"""
from __future__ import annotations

import os

from freecloud.errors import ABORT, classify
from freecloud.job import Job
from freecloud.providers import kaggle as K
from freecloud.providers.base import RunResult
from freecloud.providers.kaggle import KaggleProvider


def _job(**kw):
    d = dict(name="art", entrypoint="python x.py", max_runtime_s=600)
    d.update(kw)
    return Job(**d)


def _wire(monkeypatch, tmp_path, status_text, outputs=None):
    """status 는 정해둔 값을, `kernels output` 은 tmp 파일을 심어 돌려주는 provider."""
    monkeypatch.setenv("FREECLOUD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(K.time, "sleep", lambda s: None)      # 60초 폴링 제거
    calls: list[list[str]] = []
    p = KaggleProvider()
    monkeypatch.setattr(p, "_user", lambda: "someone")

    def fake_sh(cmd, timeout, env=None, cwd=None):
        calls.append(list(cmd))
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 0, status_text
        if cmd[:3] == ["kaggle", "kernels", "output"]:
            dest = cmd[cmd.index("-p") + 1]
            os.makedirs(dest, exist_ok=True)
            for name, body in (outputs or {}).items():
                fp = os.path.join(dest, name)
                os.makedirs(os.path.dirname(fp), exist_ok=True)
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(body)
            return 0, "downloaded"
        return 0, "ok"
    monkeypatch.setattr(p, "_sh", fake_sh)
    return p, calls


# ── 불변조건 5: 기존 위치인자 생성자 호환 ───────────────────────────────────
def test_runresult_positional_construction_still_works():
    r = RunResult(False, "error", "reason", "log", "CLASS")
    assert (r.ok, r.status, r.reason, r.log_tail, r.error_class) == (
        False, "error", "reason", "log", "CLASS")
    assert r.artifacts_dir == ""                       # 새 필드는 기본값
    assert RunResult(True, "done").artifacts_dir == ""
    assert RunResult(True, "done", "r", "l", "", "/tmp/a").artifacts_dir == "/tmp/a"


def test_runresult_lands_in_orchestrator_attempts(monkeypatch):
    """불변조건 5: orchestrator.run 반환 dict 의 attempts 에 그대로 실린다."""
    from dataclasses import asdict
    rec = dict(provider="kaggle", **asdict(RunResult(True, "done", "r", "l", "", "/a/b")))
    assert rec["artifacts_dir"] == "/a/b"


# ── 불변조건 4: complete 경로 ───────────────────────────────────────────────
def test_complete_fetches_outputs_and_sets_dir(tmp_path, monkeypatch):
    p, calls = _wire(monkeypatch, tmp_path, "complete",
                     outputs={"diag.txt": "boot\nGPU 0: Tesla P100-PCIE-16GB\n",
                              "model.bin": "w"})
    res = p._poll_and_fetch(_job(needs={"gpu": "P100"}), work="/unused")
    assert res.ok is True
    expected = os.path.join(str(tmp_path / "artifacts"), "art")
    assert res.artifacts_dir == expected
    assert os.path.isfile(os.path.join(expected, "model.bin"))
    assert "P100" in res.reason and expected in res.reason

    # CLI 호출 순서·인자 검증(불변조건 4)
    assert calls[0][:3] == ["kaggle", "kernels", "status"]
    assert calls[0][3] == "someone/freecloud-art"
    assert calls[1][:3] == ["kaggle", "kernels", "output"]
    assert calls[1][3] == "someone/freecloud-art"
    assert calls[1][calls[1].index("-p") + 1] == expected


# ── 불변조건 4: error 경로도 회수한다 ───────────────────────────────────────
def test_error_also_fetches_outputs(tmp_path, monkeypatch):
    diag = ("boot\nGPU 0: Tesla P100\nTraceback (most recent call last):\n"
            "  File \"train.py\", line 3, in <module>\n"
            "TypeError: unsupported operand type(s)\n")
    p, calls = _wire(monkeypatch, tmp_path, "error", outputs={"diag.txt": diag})
    res = p._poll_and_fetch(_job(), work="/unused")
    assert res.ok is False and res.status == "error"
    assert res.artifacts_dir == os.path.join(str(tmp_path / "artifacts"), "art")
    assert os.path.isfile(os.path.join(res.artifacts_dir, "diag.txt"))
    assert "TypeError" in res.log_tail             # 원인이 log_tail 에 실린다(결함 C)
    assert res.error_class == "ENTRYPOINT_ERROR"   # 코드 버그는 ABORT(결함 E)
    assert any(c[:3] == ["kaggle", "kernels", "output"] for c in calls)


# ── 불변조건 4: success_glob 매치 0 → NO_ARTIFACT ABORT ─────────────────────
def test_success_glob_miss_is_no_artifact_abort(tmp_path, monkeypatch):
    p, _ = _wire(monkeypatch, tmp_path, "complete", outputs={"diag.txt": "boot\n"})
    res = p._poll_and_fetch(_job(success_glob="**/model.safetensors"), work="/unused")
    assert res.ok is False and res.status == "error"
    assert res.error_class == "NO_ARTIFACT"
    assert res.artifacts_dir                       # 회수 경로는 그래도 알려준다
    # 페일오버가 아니라 ABORT 여야 한다 — 어느 provider 로 가도 똑같이 빈손이다
    assert classify(res.error_class + " " + res.reason).action == ABORT


def test_success_glob_hit_passes(tmp_path, monkeypatch):
    p, _ = _wire(monkeypatch, tmp_path, "complete",
                 outputs={"diag.txt": "boot\n", "hello.txt": "hi"})
    res = p._poll_and_fetch(_job(success_glob="**/hello.txt"), work="/unused")
    assert res.ok is True


def test_success_glob_relative_to_cwd_also_hits(tmp_path, monkeypatch):
    """smoke_t4.yaml 의 `artifacts/**/hello.txt` 처럼 cwd 기준 패턴도 맞아야 한다."""
    monkeypatch.chdir(tmp_path)
    p, _ = _wire(monkeypatch, tmp_path, "complete",
                 outputs={"diag.txt": "boot\n", "hello.txt": "hi"})
    res = p._poll_and_fetch(_job(success_glob="artifacts/**/hello.txt"), work="/unused")
    assert res.ok is True


def test_no_success_glob_means_no_gate(tmp_path, monkeypatch):
    p, _ = _wire(monkeypatch, tmp_path, "complete", outputs={"diag.txt": "boot\n"})
    assert p._poll_and_fetch(_job(), work="/unused").ok is True


# ── job.artifacts = '회수본에 있어야 할 파일명' (없으면 경고) ───────────────
def test_declared_artifacts_missing_warns(tmp_path, monkeypatch, capsys):
    p, _ = _wire(monkeypatch, tmp_path, "complete", outputs={"diag.txt": "boot\n"})
    res = p._poll_and_fetch(_job(artifacts=["/kaggle/working/out.glb"]), work="/unused")
    assert res.ok is True                          # 경고일 뿐 실패는 아니다
    assert "out.glb" in capsys.readouterr().out


def test_declared_artifacts_present_is_quiet(tmp_path, monkeypatch, capsys):
    p, _ = _wire(monkeypatch, tmp_path, "complete",
                 outputs={"diag.txt": "boot\n", "out.glb": "mesh"})
    p._poll_and_fetch(_job(artifacts=["/kaggle/working/out.glb"]), work="/unused")
    assert "미확인" not in capsys.readouterr().out


# ── 회수 디렉터리 기본값(cwd 기준 ./artifacts/<job.name>/) ──────────────────
def test_default_artifacts_dir_is_cwd_relative(tmp_path, monkeypatch):
    monkeypatch.delenv("FREECLOUD_ARTIFACTS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert K.artifacts_dir_for(_job()) == os.path.join(str(tmp_path), "artifacts", "art")


def test_gpu_mismatch_still_reports_artifacts_dir(tmp_path, monkeypatch):
    p, _ = _wire(monkeypatch, tmp_path, "complete",
                 outputs={"diag.txt": "GPU 0: Tesla P100-PCIE-16GB\n"})
    res = p._poll_and_fetch(_job(needs={"gpu": "T4x2"}), work="/unused")
    assert res.ok is False and res.error_class == "GPU_MISMATCH"
    assert res.artifacts_dir
