"""probe 거짓 양성 / 시크릿 인용 / artifacts 침묵 — 어댑터 공통 위생.

kaggle 에서 잡은 버그(probe 가 인증 실패를 OK 로 통과)가 colab 에도 똑같이 있었다.
같은 실수가 다음 어댑터에서 되풀이되지 않도록 **계약을 테스트로 고정**한다.
"""
from __future__ import annotations

import pytest

from freecloud.job import Job
from freecloud.providers.base import Provider
from freecloud.providers.colab import ColabProvider


def _job(**kw):
    d = dict(name="t", entrypoint="echo hi", max_runtime_s=60)
    d.update(kw)
    return Job(**d)


# ── export_prefix: 시크릿 값이 셸을 깨거나 주입되면 안 된다 ────────────────
@pytest.mark.parametrize("val", [
    "plain", "with space", "semi;rm -rf /", "quote'x", 'dq"x', "sub$(id)", "back`id`",
])
def test_export_prefix_quotes_every_value(val):
    import shlex
    out = Provider.export_prefix({"HF_TOKEN": val})
    assert out.startswith("export HF_TOKEN=")
    # 셸이 다시 쪼갰을 때 원래 값 하나로 복원되어야 한다
    toks = shlex.split(out.rstrip("; ").replace("export HF_TOKEN=", "", 1))
    assert toks == [val]


def test_export_prefix_rejects_bogus_keys():
    out = Provider.export_prefix({"OK_1": "a", "bad key": "b", "x;y": "c"})
    assert "OK_1" in out and "bad key" not in out and "x;y" not in out


# ── probe: 실패한 명령을 available=True 로 통과시키면 안 된다 ──────────────
@pytest.mark.parametrize("rc,expect_available", [(0, True), (1, False), (124, False), (127, False)])
def test_colab_probe_no_false_positive(monkeypatch, rc, expect_available):
    p = ColabProvider()
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (rc, "Hardware: T4" if rc == 0 else "err"))
    assert p.probe().available is expect_available


def test_colab_probe_names_auth_expiry(monkeypatch):
    p = ColabProvider()
    monkeypatch.setattr(p, "_sh", lambda *a, **k: (1, "401 Unauthorized: login required"))
    r = p.probe()
    assert not r.available and "인증" in r.reason


# ── artifacts: 회수 못 하면 최소한 말은 해야 한다 ─────────────────────────
def test_warn_unfetched_speaks_up(capsys):
    msg = Provider.warn_unfetched(_job(artifacts=["/kaggle/working/out.glb"]), "modal")
    assert "out.glb" in msg and "out.glb" in capsys.readouterr().out


def test_warn_unfetched_silent_when_nothing_declared(capsys):
    assert Provider.warn_unfetched(_job(), "modal") == ""
    assert capsys.readouterr().out == ""
