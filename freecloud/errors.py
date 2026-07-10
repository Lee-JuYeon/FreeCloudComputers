"""errors — 중앙 오류 플레이북(단일 SSOT).

모든 provider 어댑터가 실패 로그를 여기로 넘겨 분류한다. 오류 추가는 여기 한 곳만 고치면
오케스트레이터·전 어댑터에 즉시 적용된다(무인 자가복구 일관성).

분류 결과(action)가 오케스트레이터의 다음 행동을 결정한다:
  RETRY     — 같은 provider로 재시도(일시적 네트워크 등).
  FAILOVER  — 이 provider는 지금 못 씀 → 쿨다운 걸고 다음 provider로.
  ABORT     — job 자체 결함(코드/설정 오류) → 페일오버해도 어디서도 실패. 사람 개입 필요.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RETRY = "RETRY"
FAILOVER = "FAILOVER"
ABORT = "ABORT"


@dataclass
class Diagnosis:
    error_class: str
    action: str          # RETRY | FAILOVER | ABORT
    advice: str          # 사람이 읽을 조치
    cooldown_s: int = 0  # FAILOVER 시 이 provider를 얼마나 쉬게 할지(0=기본값 사용)


# (정규식, error_class, action, advice, cooldown_초). 위에서부터 첫 매치.
_PLAYBOOK: list[tuple[str, str, str, str, int]] = [
    # ── 쿼터/가용성: 다른 provider로 페일오버 ────────────────────────────────
    (r"too\s*many\s*assignments|service unavailable|503|no backend|"
     r"cannot connect to gpu|usage limit|quota (?:exceeded|exhausted)|"
     r"you have exhausted|weekly gpu|out of .*credits",
     "QUOTA_COOLDOWN", FAILOVER, "이 provider의 무료 GPU 쿼터/쿨다운. 다음 provider로 넘어감.", 3600),

    (r"connection was lost|websocket|session (?:expired|died|disconnected)|"
     r"runtime disconnected|kernel (?:died|restart)",
     "SESSION_LOST", FAILOVER, "원격 세션 끊김(웹소켓 수명/유휴). 체크포인트분 회수 후 다음 provider.", 0),

    (r"no slot|all .* busy|capacity|queue(?:d)? too long|slot .* wait",
     "NO_SLOT", FAILOVER, "무료 슬롯이 꽉 참. 다음 provider로.", 900),

    # ── 인증/설정: 페일오버해도 같은 결과 → 사람 개입 ────────────────────────
    (r"gated repo|401|403|unauthorized|invalid .*token|authentication|permission denied",
     "AUTH", ABORT, "인증 실패(토큰/gated repo). provider Secret·HF/Kaggle 토큰 확인 필요.", 0),
    (r"no such file|not found|modulenotfound|importerror|no module named",
     "JOB_CONFIG", ABORT, "job 코드/경로/의존성 오류. entrypoint·requirements 확인.", 0),

    # ── 리소스: 페일오버는 하되 job 조정 권고 ────────────────────────────────
    (r"out of memory|cuda .*out of memory|oom",
     "OOM", FAILOVER, "메모리 부족. batch/max_len 축소 권고 + 다른(더 큰 VRAM) provider로.", 0),
    (r"segmentation fault|illegal instruction|cc<7|compute capability|sm_\d+ .*not",
     "ARCH_MISMATCH", FAILOVER, "GPU 아키텍처 비호환(예: P100 sm60 4bit). fp16 커널/다른 GPU provider로.", 0),

    # ── 일시적: 같은 provider 재시도 ─────────────────────────────────────────
    (r"timeout|timed out|connection reset|temporarily|rate.?limit(?:ed)?|try again",
     "TRANSIENT", RETRY, "일시적 오류. 같은 provider로 재시도.", 0),
]


def classify(log: str) -> Diagnosis:
    """실패 로그 텍스트 → Diagnosis. 매치 없으면 보수적으로 FAILOVER(다음 provider 시도)."""
    text = (log or "").lower()
    for pat, klass, action, advice, cd in _PLAYBOOK:
        if re.search(pat, text):
            return Diagnosis(klass, action, advice, cd)
    return Diagnosis("UNKNOWN", FAILOVER,
                     "미분류 오류 — 다음 provider로 시도. 로그 꼬리를 errors.py 플레이북에 추가 권장.", 0)
