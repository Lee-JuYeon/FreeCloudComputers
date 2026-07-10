import os

from freecloud.job import Job
from freecloud import secrets


def _job(**kw):
    base = dict(name="t", entrypoint="python x.py")
    base.update(kw)
    return Job(**base)


def test_ckpt_env_injected():
    j = _job(checkpoint_repo="u/r")
    assert j.resolved_env().get("FREECLOUD_CKPT_REPO") == "u/r"


def test_hf_secret_auto_injected_when_ckpt(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    j = _job(checkpoint_repo="u/r")
    env = j.resolved_env()
    assert env.get("HF_TOKEN") == "hf_abc"


def test_no_secret_without_ckpt(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_abc")
    j = _job()  # checkpoint_repo 없음 → 시크릿 주입 안 함
    assert "HF_TOKEN" not in j.resolved_env()


def test_named_secret_injected(monkeypatch):
    monkeypatch.setenv("MY_TOKEN", "v1")
    j = _job(secrets=["MY_TOKEN"])
    assert j.resolved_env().get("MY_TOKEN") == "v1"


def test_collect_for_job_empty_when_nothing():
    j = _job()
    assert secrets.collect_for_job(j) == {}
