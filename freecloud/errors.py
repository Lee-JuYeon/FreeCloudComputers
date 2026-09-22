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

    # ⚠️ 2026-09-23 실측 회귀: 예전 패턴의 맨몸 `capacity` 가 torch OOM 메시지의
    #    "GPU 0 has a total capacity of 14.56 GiB" 에 **먼저** 매치돼, CUDA OOM 을
    #    NO_SLOT(900s 쿨다운)으로 오분류했다. '슬롯 없음'을 뜻하는 표현으로 좁힌다.
    (r"no slot|all .* busy|(?:at|over|no) capacity|queue(?:d)? too long|slot .* wait",
     "NO_SLOT", FAILOVER, "무료 슬롯이 꽉 참. 다음 provider로.", 900),

    # ── 인증/설정: 페일오버해도 같은 결과 → 사람 개입 ────────────────────────
    (r"gated repo|401|403|unauthorized|invalid .*token|authentication|permission denied",
     "AUTH", ABORT, "인증 실패(토큰/gated repo). provider Secret·HF/Kaggle 토큰 확인 필요.", 0),
    (r"no such file|not found|modulenotfound|importerror|no module named",
     "JOB_CONFIG", ABORT, "job 코드/경로/의존성 오류. entrypoint·requirements 확인.", 0),

    # 커널은 '완료'로 끝났는데 success_glob 산출물이 0개 — 어느 provider 로 가도 똑같이
    # 빈손이다(코드가 파일을 안 만든 것). 오케스트레이터가 error_class 로 다시 분류하므로
    # 문자열 자체도 여기서 잡는다. (설계 §1.B)
    (r"no_artifact",
     "NO_ARTIFACT", ABORT, "커널은 끝났는데 success_glob 산출물이 없다 — "
     "entrypoint 가 파일을 만들지 않았다. 코드/경로 확인.", 0),
    # entrypoint 코드 예외. 아래 _is_entrypoint_error() 가 로그에서 찾아내지만,
    # 재분류 때는 error_class 문자열만 오기도 한다 → 그 경우도 ABORT 로 붙잡는다.
    (r"entrypoint_error",
     "ENTRYPOINT_ERROR", ABORT, "entrypoint 코드 예외(Traceback). 페일오버해도 "
     "같은 자리에서 죽는다 — 쿼터를 태우지 말고 코드를 고쳐라.", 0),

    # ── 리소스: 페일오버는 하되 job 조정 권고 ────────────────────────────────
    # `oom` 은 낱말 경계로 묶는다 — 없으면 "boom"·"zoom"·"loom" 이 OOM 으로 잡힌다
    # (실제로 `TypeError: boom` 짜리 로그가 OOM 으로 분류됐다).
    (r"out of memory|cuda .*out of memory|\boom\b",
     "OOM", FAILOVER, "메모리 부족. batch/max_len 축소 권고 + 다른(더 큰 VRAM) provider로.", 0),
    (r"segmentation fault|illegal instruction|cc<7|compute capability|sm_\d+ .*not",
     "ARCH_MISMATCH", FAILOVER, "GPU 아키텍처 비호환(예: P100 sm60 4bit). fp16 커널/다른 GPU provider로.", 0),

    # ── 일시적: 같은 provider 재시도 ─────────────────────────────────────────
    (r"timeout|timed out|connection reset|temporarily|rate.?limit(?:ed)?|try again",
     "TRANSIENT", RETRY, "일시적 오류. 같은 provider로 재시도.", 0),
]


# entrypoint(사용자 코드) 예외로 볼 파이썬 내장 예외 이름.
# 이 목록에 "CUDA/OOM/쿼터" 계열은 없다 — 그런 건 환경 탓이라 다른 provider 에서 살 수
# 있지만, 아래 것들은 **어느 노드로 가도 똑같이 죽는다**.
_ENTRYPOINT_EXCEPTIONS = {
    "TypeError", "ValueError", "KeyError", "AttributeError", "AssertionError",
    "IndexError", "NameError", "ZeroDivisionError", "RuntimeError", "SystemExit",
}

#: `ExceptionName` 또는 `pkg.mod.ExceptionName: message` 꼴 한 줄.
_EXC_LINE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*)\s*(?::.*)?$")


def _is_entrypoint_error(log: str) -> bool:
    """로그에 커널 Traceback 이 있고, 그 뒤 **마지막 예외 줄**이 코드 버그 계열인가.

    결함 E(docs/IMPROVEMENTS_2026-09-23.md §0): entrypoint 가 TypeError 로 죽었는데
    UNKNOWN→FAILOVER 로 분류돼 kaggle-ui 에 30분 쿨다운이 걸리고 다음 라운드를 60초씩
    3번 돌았다. 코드 버그는 페일오버로 고쳐지지 않는다 — 쿼터·쿨다운만 태운다.
    """
    text = log or ""
    idx = text.lower().rfind("traceback (most recent call last)")
    if idx < 0:
        return False
    for line in reversed(text[idx:].splitlines()):
        line = line.strip()
        if not line or line.startswith(("File \"", "Traceback")):
            continue
        m = _EXC_LINE.match(line)
        if m:
            return m.group(1).rsplit(".", 1)[-1] in _ENTRYPOINT_EXCEPTIONS
    return False


def classify(log: str) -> Diagnosis:
    """실패 로그 텍스트 → Diagnosis. 매치 없으면 보수적으로 FAILOVER(다음 provider 시도).

    순서가 계약이다(불변조건 8): 기존 플레이북 → ENTRYPOINT_ERROR → UNKNOWN.
    같은 로그에 `CUDA out of memory` 와 Traceback 이 함께 있으면 **OOM 이 이긴다** —
    코드가 멀쩡해도 VRAM 이 작으면 죽으므로 더 큰 GPU 로 넘기는 게 맞다.
    """
    text = (log or "").lower()
    for pat, klass, action, advice, cd in _PLAYBOOK:
        if re.search(pat, text):
            return Diagnosis(klass, action, advice, cd)
    if _is_entrypoint_error(log or ""):
        return Diagnosis("ENTRYPOINT_ERROR", ABORT,
                         "entrypoint 코드 예외(Traceback). 페일오버해도 같은 자리에서 "
                         "죽는다 — 쿼터를 태우지 말고 코드를 고쳐라.", 0)
    # 미분류를 30분(1800s) 잠그는 건 과하다 — 그 사이 provider 가 멀쩡해져도 못 쓴다.
    # 300초면 '연달아 두드리기'는 막으면서 회복을 놓치지 않는다(설계 §1.E).
    return Diagnosis("UNKNOWN", FAILOVER,
                     "미분류 오류 — 다음 provider로 시도. 로그 꼬리를 errors.py 플레이북에 추가 권장.",
                     300)
