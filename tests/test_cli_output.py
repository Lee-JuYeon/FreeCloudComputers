"""실패 상세 출력 + `fcc logs`(결함 C) — 불변조건 6.

docs/IMPROVEMENTS_2026-09-23.md §2.6:
  CLI 실패 출력에 error_class·reason·log_tail 마지막 40줄이 나온다. `--quiet` 면 안 나온다.

예전엔 `→ ok=False UNKNOWN 미분류 오류 — …` 한 줄이 전부였고, 진짜 원인(커널 Traceback)은
반환 dict 안에서 잠자고 있었다. 오케스트레이터/네트워크는 전부 몽키패치.
"""
from __future__ import annotations

import os

import pytest

from freecloud import cli, orchestrator


LONG_TAIL = "\n".join(f"line{i}" for i in range(1, 121)) + (
    "\nTraceback (most recent call last):\n  File \"train.py\", line 9\n"
    "TypeError: unsupported operand\n")


def _result(ok=False, **over):
    att = dict(provider="kaggle", ok=False, status="error",
               reason="entrypoint 코드 예외(Traceback).", log_tail=LONG_TAIL,
               error_class="ENTRYPOINT_ERROR", artifacts_dir="/tmp/artifacts/demo")
    att.update(over)
    return dict(ok=ok, provider=None, reason="실패", attempts=[att])


def _run(monkeypatch, tmp_path, result, argv_extra=()):
    yml = tmp_path / "job.yaml"
    yml.write_text("name: demo\nentrypoint: \"python x.py\"\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "run", lambda *a, **k: result)
    return cli.main(["run", str(yml), "--no-interactive", *argv_extra])


def test_failure_prints_class_reason_and_log_tail(monkeypatch, tmp_path, capsys):
    rc = _run(monkeypatch, tmp_path, _result())
    out = capsys.readouterr().out
    assert rc == 1
    assert "error_class=ENTRYPOINT_ERROR" in out
    assert "entrypoint 코드 예외" in out
    assert "TypeError: unsupported operand" in out
    assert "artifacts: /tmp/artifacts/demo" in out


def test_log_tail_is_capped_at_40_lines(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, _result())
    out = capsys.readouterr().out
    piped = [ln for ln in out.splitlines() if ln.startswith("  | ")]
    assert len(piped) == cli.LOG_TAIL_LINES == 40
    # 마지막 40줄이므로 꼬리는 들어오고 머리는 빠진다
    assert any("TypeError" in ln for ln in piped)
    assert not any(ln.endswith("line1") for ln in piped)


def test_quiet_suppresses_details(monkeypatch, tmp_path, capsys):
    _run(monkeypatch, tmp_path, _result(), argv_extra=("--quiet",))
    out = capsys.readouterr().out
    assert "error_class=" not in out
    assert "  | " not in out
    assert "===== RESULT =====" in out          # 요약 JSON 은 그대로 남는다


def test_success_prints_artifacts_dir(monkeypatch, tmp_path, capsys):
    ok = dict(ok=True, provider="kaggle", reason="done",
              attempts=[dict(provider="kaggle", ok=True, status="done", reason="done",
                             log_tail="", error_class="", artifacts_dir="/tmp/a/demo")])
    rc = _run(monkeypatch, tmp_path, ok)
    assert rc == 0
    assert "산출물: /tmp/a/demo" in capsys.readouterr().out


def test_successful_attempt_is_not_dumped(monkeypatch, tmp_path, capsys):
    """성공한 attempt 의 로그까지 쏟아내면 실패 원인이 묻힌다."""
    res = _result(ok=True)
    res["attempts"].append(dict(provider="colab", ok=True, status="done", reason="ok",
                                log_tail="SHOULD-NOT-APPEAR", error_class=""))
    _run(monkeypatch, tmp_path, res)
    out = capsys.readouterr().out
    # 요약 JSON 에는 남지만(그게 계약), 상세 덤프(`  | `)에는 안 나와야 한다
    assert "  | SHOULD-NOT-APPEAR" not in out
    assert "[colab]" not in out


def test_dry_run_still_works(monkeypatch, tmp_path, capsys):
    yml = tmp_path / "job.yaml"
    yml.write_text("name: demo\nentrypoint: \"python x.py\"\n", encoding="utf-8")
    assert cli.main(["run", str(yml), "--dry-run"]) == 0
    assert '"job": "demo"' in capsys.readouterr().out


# ── fcc logs ───────────────────────────────────────────────────────────────
def _fake_kaggle(monkeypatch, tmp_path, diag="boot\nGPU 0: Tesla P100\nTypeError: x\n"):
    from freecloud.providers.kaggle import KaggleProvider
    monkeypatch.setenv("FREECLOUD_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setattr(KaggleProvider, "_user", lambda self: "someone")

    def fake_sh(self, cmd, timeout, env=None, cwd=None):
        if cmd[:3] == ["kaggle", "kernels", "status"]:
            return 0, "kernel has status \"error\""
        if cmd[:3] == ["kaggle", "kernels", "output"]:
            dest = cmd[cmd.index("-p") + 1]
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(dest, "diag.txt"), "w", encoding="utf-8") as f:
                f.write(diag)
            return 0, "downloaded"
        return 0, "ok"
    monkeypatch.setattr(KaggleProvider, "_sh", fake_sh)


def test_logs_from_job_yaml(monkeypatch, tmp_path, capsys):
    _fake_kaggle(monkeypatch, tmp_path)
    yml = tmp_path / "job.yaml"
    yml.write_text("name: demo\nentrypoint: \"python x.py\"\n", encoding="utf-8")
    assert cli.main(["logs", str(yml)]) == 0
    out = capsys.readouterr().out
    assert "kernel: someone/freecloud-demo" in out
    assert "diag.txt" in out
    assert "TypeError: x" in out


def test_logs_from_bare_name(monkeypatch, tmp_path, capsys):
    """job.yaml 이 손에 없어도 이름만으로 커널 로그를 볼 수 있어야 한다."""
    _fake_kaggle(monkeypatch, tmp_path)
    assert cli.main(["logs", "demo", "--lines", "5"]) == 0
    assert "kernel: someone/freecloud-demo" in capsys.readouterr().out


def test_logs_without_auth_returns_1(monkeypatch, tmp_path, capsys):
    from freecloud.providers.kaggle import KaggleProvider
    monkeypatch.setattr(KaggleProvider, "_user", lambda self: "")
    assert cli.main(["logs", "demo"]) == 1
    assert "Kaggle 인증 없음" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["run", "--help"], ["logs", "--help"], ["--help"]])
def test_help_surfaces_do_not_crash(argv, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 0
