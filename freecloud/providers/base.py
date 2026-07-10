"""base — Provider 어댑터 인터페이스(모든 무료 클라우드 공통 계약).

기존 검증 코드(Infopu train_failover.try_kaggle/try_colab, colab_vm.sh의 trap/슬롯폴링)를
이 인터페이스로 일반화한 것. 각 메서드는 부작용 없이 실패 시 예외 대신 결과 dataclass 반환.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..job import Job


@dataclass
class Probe:
    available: bool
    reason: str = ""
    cooldown_s: int = 0     # available=False일 때 권장 쿨다운


@dataclass
class RunResult:
    ok: bool
    status: str             # done | quota | error | timeout
    reason: str = ""
    log_tail: str = ""
    error_class: str = ""


class Provider(ABC):
    #: 표시 이름(registry 키). 서브클래스에서 지정.
    name: str = "base"
    #: 능력 선언 — job.needs 와 매칭해 못 맞으면 스킵. 예: {"gpu": "T4", "min_vram_gb": 16, "headless": True}
    capabilities: dict = {}

    # ── 라이프사이클 ─────────────────────────────────────────────────────────
    @abstractmethod
    def probe(self) -> Probe:
        """사전 가용성 확인(CLI 인증됨? 슬롯 있음? 쿼터 남음?). 값싸게."""

    @abstractmethod
    def run(self, job: Job) -> RunResult:
        """job을 이 provider에서 끝까지 실행(제출→폴링→산출물 회수→정리).

        blocking. 내부에서 max_runtime_s 존중, 실패 시 로그를 담아 RunResult 반환.
        고아 방지: 어떤 경로로 끝나도 원격 세션을 반드시 stop(colab_vm.sh [A] 원칙).
        """

    def fits(self, job: Job) -> tuple[bool, str]:
        """job.needs vs capabilities 정적 매칭. (가능?, 사유)."""
        need_vram = job.needs.get("min_vram_gb")
        cap_vram = self.capabilities.get("min_vram_gb") or self.capabilities.get("vram_gb")
        if need_vram and cap_vram and cap_vram < need_vram:
            return False, f"{self.name}: VRAM {cap_vram}<{need_vram}GB 요구"
        if job.needs.get("headless") and not self.capabilities.get("headless", True):
            return False, f"{self.name}: 헤드리스 미지원(UI 전용)"
        return True, ""

    # ── 공용 헬퍼 ────────────────────────────────────────────────────────────
    @staticmethod
    def _sh(cmd: list[str], timeout: int, env: Optional[dict] = None,
            cwd: Optional[str] = None) -> tuple[int, str]:
        """subprocess 실행 → (returncode, combined_log). 타임아웃/미설치 안전 처리."""
        import os
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                               env={**os.environ, **(env or {})}, cwd=cwd)
            return p.returncode, (p.stdout or "") + (p.stderr or "")
        except FileNotFoundError:
            return 127, f"명령 없음: {cmd[0]} (해당 provider CLI 미설치)"
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + (e.stderr or "") if hasattr(e, "stdout") else ""
            return 124, out + f"\n[timeout {timeout}s] {cmd[0]}"
