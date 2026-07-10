"""lightning 어댑터 — Lightning AI(lightning-sdk)로 Studio Job 실행.

Lightning 무료: 15크레딧/월 ≈ 80 GPU-h(interruptible). 지속 스토리지 → 재개 친화적이라
페일오버 풀에서 물량이 가장 큰 축.

인증(토큰 방식, 비대화형): env LIGHTNING_USER_ID + LIGHTNING_API_KEY.
teamspace: env LIGHTNING_TEAMSPACE(예: "myuser/vision"), 스튜디오: LIGHTNING_STUDIO(선택),
머신: FREECLOUD_LIGHTNING_MACHINE(기본 T4).

문서 확인된 시그니처:
    from lightning_sdk import Studio, Job, Machine
    studio = Studio(name=..., teamspace=..., user=...); studio.start()
    job = Job.run(command=cmd, name=..., machine=Machine.T4, studio=studio, interruptible=True)
    job.status  # Pending/Running/Completed/Failed
"""
from __future__ import annotations

import os
import time

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult


class LightningProvider(Provider):
    name = "lightning"
    capabilities = {"gpu": "T4|L4|A10G", "vram_gb": 16, "headless": True,
                    "note": "15크레딧/월 ≈ 80 GPU-h, interruptible."}

    def _sdk(self):
        try:
            import lightning_sdk
            return lightning_sdk
        except ImportError:
            return None

    def probe(self) -> Probe:
        if self._sdk() is None:
            return Probe(False, "lightning-sdk 미설치(pip install lightning-sdk).")
        if not (os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID")):
            return Probe(False, "LIGHTNING_API_KEY/LIGHTNING_USER_ID 미설정.")
        if not os.environ.get("LIGHTNING_TEAMSPACE"):
            return Probe(False, "LIGHTNING_TEAMSPACE 미설정(예: myuser/teamspace).")
        return Probe(True, "lightning 인증 OK.")

    def run(self, job: Job) -> RunResult:
        sdk = self._sdk()
        if sdk is None:
            return RunResult(False, "error", "lightning-sdk 미설치.", error_class="JOB_CONFIG")
        try:
            from lightning_sdk import Studio, Job as LJob, Machine
        except Exception as e:
            return RunResult(False, "error", f"lightning-sdk import 실패: {e}",
                             error_class="JOB_CONFIG")

        ts = os.environ.get("LIGHTNING_TEAMSPACE")
        user = os.environ.get("LIGHTNING_USER_ID")
        mname = os.environ.get("FREECLOUD_LIGHTNING_MACHINE", "T4")
        machine = getattr(Machine, mname, mname)
        # env export 를 명령 앞에 붙여 원격에 주입(체크포인트 repo 등)
        cmd = " ".join(f"export {k}={v};" for k, v in job.resolved_env().items()) \
            + " " + job.entrypoint

        try:
            studio_name = os.environ.get("LIGHTNING_STUDIO", f"fc-{job.name}")[:40]
            studio = Studio(name=studio_name, teamspace=ts, user=user, create_ok=True)
            studio.start()
            lj = LJob.run(command=cmd, name=f"fc-{job.name}"[:40], machine=machine,
                          studio=studio, interruptible=True)
            return self._poll(lj, job)
        except Exception as e:
            d = classify(str(e))
            st = "quota" if ("credit" in str(e).lower() or d.error_class == "QUOTA_COOLDOWN") \
                else "error"
            return RunResult(False, st, f"Lightning 예외: {e}", str(e)[-800:], d.error_class)

    def _poll(self, lj, job: Job) -> RunResult:
        deadline = time.time() + job.max_runtime_s
        while time.time() < deadline:
            status = str(getattr(lj, "status", "")).lower()
            if any(s in status for s in ("completed", "success", "finished")):
                return RunResult(True, "done", "Lightning Job 완료.")
            if any(s in status for s in ("failed", "error", "stopped")):
                log = ""
                try:
                    log = str(lj.logs() if callable(getattr(lj, "logs", None)) else "")
                except Exception:
                    pass
                d = classify(log + " " + status)
                st = "quota" if d.error_class == "QUOTA_COOLDOWN" else "error"
                return RunResult(False, st, d.advice, log[-1000:], d.error_class)
            time.sleep(30)
        try:
            lj.stop()
        except Exception:
            pass
        return RunResult(False, "timeout", f"Lightning 폴링 {job.max_runtime_s}s 초과.",
                         error_class="TIMEOUT")
