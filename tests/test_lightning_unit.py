"""lightning 어댑터 단위 테스트 — 가짜 lightning_sdk 를 sys.modules 에 꽂는다(네트워크 0).

검증 축: teamspace 슬러그(user/org 없음)·이미지 기본 경로·스튜디오 경로(start 금지)·
머신 이름 오류·폴링 상한=detached(stop 금지)+heartbeat·상태 매핑·크레딧 부족=quota.
"""
from __future__ import annotations

import enum
import sys
import types

import pytest

from freecloud.job import Job
from freecloud.providers import lightning as L
from freecloud.providers.lightning import LightningProvider


class FakeStatus(enum.Enum):
    Completed = "Completed"
    Failed = "Failed"
    NotCreated = "NotCreated"
    Pending = "Pending"
    Running = "Running"
    Stopped = "Stopped"
    Stopping = "Stopping"

    def __str__(self):
        return self.value


class FakeLJob:
    def __init__(self, statuses, logs=""):
        self._statuses = list(statuses)
        self._logs = logs
        self.stopped = False
        self.name = "fc-x-1"
        self.link = "https://lightning.ai/fake/jobs/fc-x-1"

    @property
    def status(self):
        return self._statuses.pop(0) if len(self._statuses) > 1 else self._statuses[0]

    def logs(self):
        return self._logs

    def stop(self):
        self.stopped = True


def _fake_sdk(monkeypatch, statuses=(FakeStatus.Completed,), logs="", run_exc=None):
    rec: dict = {"run": [], "studio": []}
    holder: dict = {}

    class FakeJobCls:
        @staticmethod
        def run(**kw):
            rec["run"].append(kw)
            if run_exc:
                raise run_exc
            holder["job"] = FakeLJob(statuses, logs)
            return holder["job"]

    class FakeStudio:
        def __init__(self, **kw):
            rec["studio"].append(kw)

        def start(self):
            raise AssertionError("Studio.start must never be called")

    class FakeMachine:
        T4 = "MACHINE_T4"
        T4_X_2 = "MACHINE_T4_X_2"
        L4 = "MACHINE_L4"

    fake = types.ModuleType("lightning_sdk")
    fake.Job, fake.Studio, fake.Machine, fake.Status = FakeJobCls, FakeStudio, FakeMachine, FakeStatus
    monkeypatch.setitem(sys.modules, "lightning_sdk", fake)
    clock = {"t": 1_000_000.0}
    monkeypatch.setattr(L.time, "time", lambda: clock["t"])
    monkeypatch.setattr(L.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    for k in ("LIGHTNING_STUDIO", "FREECLOUD_LIGHTNING_MACHINE", "FREECLOUD_LIGHTNING_IMAGE",
              "FREECLOUD_LIGHTNING_ATTEMPTS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "lee-juyeon/default-project")
    monkeypatch.setenv("LIGHTNING_USER_ID", "0" * 36)
    monkeypatch.setenv("LIGHTNING_API_KEY", "k")
    return rec, holder


def _job(**kw):
    d = dict(name="x", entrypoint="python train.py", max_runtime_s=300, env={"FOO": "bar baz"})
    d.update(kw)
    return Job(**d)


def test_image_path_default_with_slug_env_and_attempts(monkeypatch):
    rec, _ = _fake_sdk(monkeypatch)
    res = LightningProvider().run(_job())
    assert (res.ok, res.status) == (True, "done")
    kw = rec["run"][0]
    assert kw["teamspace"] == "lee-juyeon/default-project"
    assert "user" not in kw and "org" not in kw
    assert kw["image"] == L.DEFAULT_IMAGE and "studio" not in kw
    assert kw["command"].endswith("python train.py")
    assert "export FOO='bar baz'; " in kw["command"]          # 접두사 유지(이중 안전장치)
    assert kw["env"]["FOO"] == "bar baz"
    assert kw["interruptible"] is True and kw["max_run_attempts"] == 5
    assert kw["machine"] == "MACHINE_T4"
    assert kw["name"].startswith("fc-x-") and len(kw["name"]) <= 40
    assert rec["studio"] == []


def test_image_and_machine_overrides(monkeypatch):
    rec, _ = _fake_sdk(monkeypatch)
    monkeypatch.setenv("FREECLOUD_LIGHTNING_IMAGE", "my/img:1")
    monkeypatch.setenv("FREECLOUD_LIGHTNING_MACHINE", "L4")
    monkeypatch.setenv("FREECLOUD_LIGHTNING_ATTEMPTS", "2")
    LightningProvider().run(_job())
    kw = rec["run"][0]
    assert (kw["image"], kw["machine"], kw["max_run_attempts"]) == ("my/img:1", "MACHINE_L4", 2)


def test_studio_path_never_starts(monkeypatch):
    rec, _ = _fake_sdk(monkeypatch)
    monkeypatch.setenv("LIGHTNING_STUDIO", "my-studio")
    res = LightningProvider().run(_job())
    assert res.ok
    assert rec["studio"] == [dict(name="my-studio", teamspace="lee-juyeon/default-project",
                                  create_ok=False)]
    kw = rec["run"][0]
    assert "studio" in kw and "image" not in kw and kw["command"].endswith("python train.py")
    assert "teamspace" not in kw            # SDK 가 슬러그 vs studio.teamspace.name 비교로 터진다


def test_unknown_machine_is_job_config(monkeypatch):
    rec, _ = _fake_sdk(monkeypatch)
    monkeypatch.setenv("FREECLOUD_LIGHTNING_MACHINE", "H200_BOGUS")
    res = LightningProvider().run(_job())
    assert (res.ok, res.status, res.error_class) == (False, "error", "JOB_CONFIG")
    assert "H200_BOGUS" in res.reason
    assert rec["run"] == []


def test_running_forever_detaches_without_stop(monkeypatch, capsys):
    rec, holder = _fake_sdk(monkeypatch, statuses=(FakeStatus.Running,))
    res = LightningProvider().run(_job(max_runtime_s=1800))
    assert (res.ok, res.status, res.error_class) == (False, "detached", "DETACHED")
    assert holder["job"].link in res.reason
    assert holder["job"].stopped is False
    out = capsys.readouterr().out
    assert "[lightning] ♥ 10분 경과 · job=fc-x-1 · status=running" in out


def test_string_status_tolerated(monkeypatch):
    _fake_sdk(monkeypatch, statuses=("Status.Pending", "Completed"))
    assert LightningProvider().run(_job()).ok


@pytest.mark.parametrize("statuses,logs,expect", [
    ((FakeStatus.Pending, FakeStatus.Running, FakeStatus.Completed), "", (True, "done", "")),
    ((FakeStatus.Running, FakeStatus.Stopped), "preempted", (False, "error", "INTERRUPTED")),
    ((FakeStatus.Stopping,), "", (False, "error", "INTERRUPTED")),
])
def test_status_mapping(monkeypatch, statuses, logs, expect):
    _fake_sdk(monkeypatch, statuses=statuses, logs=logs)
    res = LightningProvider().run(_job())
    assert (res.ok, res.status, res.error_class) == expect
    if expect[2] == "INTERRUPTED":
        assert "체크포인트 재개" in res.reason


def test_failed_returns_log_tail(monkeypatch):
    logs = "x" * 3000 + "\nTraceback ...\nModuleNotFoundError: No module named 'peft'"
    _fake_sdk(monkeypatch, statuses=(FakeStatus.Failed,), logs=logs)
    res = LightningProvider().run(_job())
    assert (res.ok, res.status) == (False, "error")
    assert len(res.log_tail) == 1500 and res.log_tail.endswith("'peft'")
    assert res.error_class == "JOB_CONFIG"


def test_failed_with_credit_logs_is_quota(monkeypatch):
    _fake_sdk(monkeypatch, statuses=(FakeStatus.Failed,), logs="Job failed: insufficient balance")
    res = LightningProvider().run(_job())
    assert (res.status, res.error_class) == ("quota", "QUOTA_COOLDOWN")


def test_run_raising_insufficient_credits_is_quota(monkeypatch):
    _fake_sdk(monkeypatch, run_exc=RuntimeError("Insufficient credits to start job"))
    res = LightningProvider().run(_job())
    assert (res.ok, res.status, res.error_class) == (False, "quota", "QUOTA_COOLDOWN")


def test_blind_reads_become_session_lost(monkeypatch):
    _fake_sdk(monkeypatch, statuses=("",))
    res = LightningProvider().run(_job(max_runtime_s=3600))
    assert res.error_class == "SESSION_LOST"


def test_probe_messages(monkeypatch):
    _fake_sdk(monkeypatch)
    p = LightningProvider()
    assert p.probe().available
    monkeypatch.setenv("LIGHTNING_TEAMSPACE", "default-project")      # 슬래시 없어도 통과
    assert p.probe().available
    monkeypatch.delenv("LIGHTNING_TEAMSPACE")
    assert p.probe().reason == "LIGHTNING_TEAMSPACE 미설정(예: myuser/teamspace)."
    monkeypatch.delenv("LIGHTNING_API_KEY")
    assert p.probe().reason == "LIGHTNING_API_KEY/LIGHTNING_USER_ID 미설정."
    monkeypatch.setitem(sys.modules, "lightning_sdk", None)
    assert p.probe().reason == "lightning-sdk 미설치(pip install lightning-sdk)."
