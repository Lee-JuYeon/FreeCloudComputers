"""base — Provider 어댑터 인터페이스(모든 무료 클라우드 공통 계약).

기존 검증 코드(Infopu train_failover.try_kaggle/try_colab, colab_vm.sh의 trap/슬롯폴링)를
이 인터페이스로 일반화한 것. 각 메서드는 부작용 없이 실패 시 예외 대신 결과 dataclass 반환.
"""
from __future__ import annotations

import re
import shlex
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..job import Job


def _gpu_match(need_gpu: str, cap_gpu: str) -> bool:
    """job이 요구한 GPU를 이 provider가 제공하는가.

    capabilities['gpu']는 여러 개를 '|'/','/'/'로 나열할 수 있고(예 'T4|L4|A10G'),
    'T4x2'처럼 접미사가 붙기도 한다. 요구 모델명이 제공 토큰의 부분문자열이면(양방향) 매치.
    이래야 needs:{gpu:'T4'} 가 kaggle(P100)을 제외하고 T4 제공자(kaggle-ui 'T4x2', colab 'T4' 등)를 고른다."""
    import re
    want = str(need_gpu).strip().upper()
    offered = [t.strip().upper() for t in re.split(r"[|,/]", str(cap_gpu)) if t.strip()]
    return any(want in tok or tok in want for tok in offered)


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
    # 회수한 산출물 디렉터리(있으면). **맨 뒤에 기본값과 함께** 둔다 — 기존 코드·테스트가
    # RunResult(False,"error","reason","log","CLASS") 처럼 위치인자로 만들기 때문에
    # 중간에 끼우면 조용히 어긋난다(불변조건 5, docs/IMPROVEMENTS_2026-09-23.md §2).
    artifacts_dir: str = ""


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
        need_gpu = job.needs.get("gpu")
        cap_gpu = self.capabilities.get("gpu")
        if need_gpu and cap_gpu and not _gpu_match(need_gpu, cap_gpu):
            return False, f"{self.name}: GPU '{cap_gpu}' — '{need_gpu}' 요구 불충족"
        return True, ""


    # ── 공용 헬퍼 ────────────────────────────────────────────────────────────
    @staticmethod
    def export_prefix(env: dict) -> str:
        """원격 셸에 env 를 주입하는 접두사. **반드시 인용한다.**

        과거: `" ".join(f"export {k}={v};")` — 값에 공백·$·;·따옴표가 있으면
        명령이 깨지거나 주입된다. resolved_env() 에는 HF_TOKEN 같은 시크릿이
        들어오므로 값 형태를 통제할 수 없다 → shlex.quote 필수.
        (kaggle 은 커널 스크립트에 넣고 is_private 로 막는다 — 그쪽이 더 안전한 축)
        """
        return "".join(f"export {k}={shlex.quote(str(v))}; "
                       for k, v in env.items() if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k))

    @staticmethod
    def warn_unfetched(job, provider: str) -> str:
        """artifacts 를 회수하지 못하는 provider 가 **조용히 넘어가지 않게** 한다.

        job.artifacts 는 '회수할 원격 경로들'로 선언돼 있는데 colab 외 어댑터는
        이를 아예 읽지 않았다. 사용자는 파일이 생긴 줄 알고 기다린다.
        회수 경로가 생기기 전까지는 최소한 **말은 해야** 한다.
        """
        if not getattr(job, "artifacts", None):
            return ""
        msg = (f"[{provider}] artifacts 미회수: {', '.join(job.artifacts)} — "
               f"이 어댑터는 원격 파일 회수를 아직 지원하지 않는다. "
               f"entrypoint 안에서 HF Hub 등으로 직접 업로드할 것.")
        print(msg, flush=True)
        return msg

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
