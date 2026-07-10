"""colab 어댑터 — colab CLI 감싸기. colab_vm.sh의 강제 원칙을 파이썬으로 이식.

 [A] 고아 방지: 어떤 경로로 끝나도 finally에서 colab stop(슬롯 잡은 채 죽는 VM 0).
 [B] 세션 재사용: 같은 이름 세션 있으면 재사용(HF캐시/커널 상태 유지 → 재다운로드 없음).
 [C] 슬롯 폴링: colab sessions 읽기전용 확인으로 슬롯 빌 때까지 대기(throttle 악화 방지).
 [D] 무료=계정당 GPU 동시 1개. 쿨다운(503/TooManyAssignments)이면 페일오버.

콜랩 CLI는 Windows에선 보통 WSL 경유. FREECLOUD_COLAB_WSL=1 이면 wsl -d Ubuntu 로 감싼다.
"""
from __future__ import annotations

import os
import time

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult


class ColabProvider(Provider):
    name = "colab"
    capabilities = {"gpu": "T4", "vram_gb": 16, "headless": True}

    def __init__(self):
        self._bin = os.environ.get("COLAB_BIN", "colab")
        self._wsl = os.environ.get("FREECLOUD_COLAB_WSL") == "1"

    def _cmd(self, *args: str) -> list[str]:
        base = ["wsl", "-d", os.environ.get("FREECLOUD_WSL_DISTRO", "Ubuntu"), "--",
                self._bin] if self._wsl else [self._bin]
        return base + list(args)

    def probe(self) -> Probe:
        rc, log = self._sh(self._cmd("sessions"), 60)
        if rc == 127:
            return Probe(False, "colab CLI 미설치/미가용(WSL 필요할 수 있음).")
        # 슬롯 점유 여부는 run()의 대기 로직에서 처리. 여기선 CLI 생존만 확인.
        return Probe(True, "colab CLI OK.")

    def run(self, job: Job) -> RunResult:
        sess = f"fc-{job.name}".lower()[:40]
        gpu = job.needs.get("gpu", "T4")

        # [C] 슬롯 폴링(우리 세션 있으면 즉시 진행)
        for i in range(int(os.environ.get("COLAB_WAIT_TRIES", "40"))):
            rc, s = self._sh(self._cmd("sessions"), 30)
            if sess in s:
                break                                   # [B] 재사용
            if "Hardware:" not in s and "Variant:" not in s:
                break                                   # 슬롯 빔
            print(f"[colab] 슬롯 점유중 — 대기 ({i+1})", flush=True)
            time.sleep(int(os.environ.get("COLAB_WAIT_SEC", "150")))
        else:
            return RunResult(False, "quota", "Colab 슬롯 대기 초과 → 페일오버.",
                             error_class="NO_SLOT")

        started = False
        try:
            rc, s = self._sh(self._cmd("sessions"), 30)
            if sess not in s:
                rc, log = self._sh(self._cmd("new", "-s", sess, "--gpu", gpu), 300)
                if rc != 0:
                    d = classify(log + " too many assignments")  # NEW_FAIL≈쿨다운
                    return RunResult(False, "quota", "Colab new 실패(503/쿨다운) → 페일오버.",
                                     log[-800:], "QUOTA_COOLDOWN")
            started = True

            # entrypoint 실행. 체크포인트 pull/push는 entrypoint 내부 책임.
            envflags = " ".join(f"{k}={v}" for k, v in job.resolved_env().items())
            remote = f"{envflags} {job.entrypoint}".strip()
            rc, log = self._sh(self._cmd("exec", "-s", sess, "-c", remote), job.max_runtime_s)

            for path in job.artifacts:                  # 부분이라도 per-file 회수
                self._sh(self._cmd("download", "-s", sess, path, "./artifacts"), 300)

            if rc == 0:
                return RunResult(True, "done", "Colab 실행 완료.", log[-400:])
            d = classify(log)
            status = "quota" if d.error_class in ("QUOTA_COOLDOWN", "SESSION_LOST") else "error"
            return RunResult(False, status, d.advice, log[-1000:], d.error_class)
        finally:
            if started and os.environ.get("COLAB_VM_KEEP") != "1":
                self._sh(self._cmd("stop", "-s", sess), 60)   # [A] 고아 방지(핵심)
