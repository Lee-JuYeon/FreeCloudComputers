"""lightning 어댑터 — Lightning AI(lightning-sdk)로 Studio에서 실행.

Lightning 무료: 15크레딧/월 ≈ 80 GPU-h(interruptible). 지속 스토리지 → 재개 친화적이라
페일오버 풀에서 물량이 가장 큰 축. lightning-sdk의 Job/Studio API를 subprocess 대신 직접 쓴다.

의존: pip install lightning-sdk. 인증: env LIGHTNING_USER_ID + LIGHTNING_API_KEY,
teamspace: env LIGHTNING_TEAMSPACE(예: "user/vision"), 머신: FREECLOUD_LIGHTNING_MACHINE(기본 T4).
"""
from __future__ import annotations

import os

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult


class LightningProvider(Provider):
    name = "lightning"
    capabilities = {"gpu": "T4|L4", "vram_gb": 16, "headless": True,
                    "note": "15크레딧/월 ≈ 80 GPU-h, interruptible."}

    def _sdk(self):
        try:
            import lightning_sdk  # noqa: F401
            return lightning_sdk
        except ImportError:
            return None

    def probe(self) -> Probe:
        if self._sdk() is None:
            return Probe(False, "lightning-sdk 미설치(pip install lightning-sdk).")
        if not (os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID")):
            return Probe(False, "LIGHTNING_API_KEY/USER_ID 미설정.")
        if not os.environ.get("LIGHTNING_TEAMSPACE"):
            return Probe(False, "LIGHTNING_TEAMSPACE 미설정(예: user/teamspace).")
        return Probe(True, "lightning 인증 OK.")

    def run(self, job: Job) -> RunResult:
        sdk = self._sdk()
        if sdk is None:
            return RunResult(False, "error", "lightning-sdk 미설치.", error_class="JOB_CONFIG")
        machine = os.environ.get("FREECLOUD_LIGHTNING_MACHINE", "T4")
        try:
            from lightning_sdk import Job as LJob, Machine  # type: ignore
            ts = os.environ.get("LIGHTNING_TEAMSPACE")
            mach = getattr(Machine, machine, machine)
            envs = job.resolved_env()
            cmd = " ".join(f"export {k}={v};" for k, v in envs.items()) + " " + job.entrypoint
            j = LJob.run(name=f"fc-{job.name}"[:40], command=cmd, machine=mach,
                         teamspace=ts, interruptible=True)
            # lightning-sdk Job.run 은 완료까지 블록(구현마다 다르면 j.wait()).
            status = getattr(j, "status", "completed")
            log = getattr(j, "logs", lambda: "")() if callable(getattr(j, "logs", None)) else ""
            if str(status).lower() in ("completed", "success", "finished"):
                return RunResult(True, "done", "Lightning Job 완료.", str(log)[-400:])
            d = classify(str(log) + " " + str(status))
            st = "quota" if d.error_class == "QUOTA_COOLDOWN" else "error"
            return RunResult(False, st, d.advice, str(log)[-1000:], d.error_class)
        except Exception as e:
            d = classify(str(e))
            st = "quota" if ("credit" in str(e).lower() or d.error_class == "QUOTA_COOLDOWN") \
                else "error"
            return RunResult(False, st, f"Lightning 예외: {e}", str(e)[-800:], d.error_class)
