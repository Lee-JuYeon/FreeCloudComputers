"""catalog — 비중국 무료 컴퓨트 카탈로그(2026-07 조사 기준).

`freecloud clouds` 로 출력. adapter가 아직 없는 것도 후보로 실어 로드맵 역할.
값은 수시 변동 → 갱신 PR 환영. status: adapter(구현됨) / candidate(후보).

표시용 카탈로그일 뿐 어댑터 인터페이스는 아니다. 자원 종류별 섹션(SECTIONS)으로
묶어 보여주고, 시뮬 런타임은 별도 LABS 목록으로 둔다.
"""
from __future__ import annotations

from dataclasses import dataclass

#: 섹션 정본(출력 순서 = 이 순서). 키 = Entry.section, 값 = 헤더 문구.
SECTIONS = [
    ("gpu",      "GPU — 가속기 대여"),
    ("computer", "Computer — 범용 컴퓨트"),
    ("mobile",   "Mobile — 모바일 기기"),
]


@dataclass(frozen=True)
class Entry:
    section: str      # SECTIONS 키 중 하나
    name: str
    spec: str         # 핵심 스펙(가속기/CPU 등)
    free_limit: str
    credit_card: str  # no | yes | gcp
    status: str       # adapter | candidate
    note: str


@dataclass(frozen=True)
class Lab:
    name: str
    requires: str     # 요구 하드웨어
    free_pool: bool   # 무료 풀에서 구동 가능한가
    note: str


CATALOG = [
    # ── GPU — 가속기 대여 ───────────────────────────────────────────────
    Entry("gpu", "Kaggle",               "P100 / T4x2 / TPU v5e-8", "~30 GPU-h/주, 세션 12h", "no",  "adapter",
          "API=단일 P100. T4x2는 provider 'kaggle-ui'(Playwright 자동화)로. 출력 /kaggle/working 영속."),
    Entry("gpu", "Google Colab",         "T4 16GB",                 "~15-30 GPU-h/주(변동)",  "no",  "adapter",
          "무료=계정당 GPU 동시 1개. 90분 유휴 끊김, 세션 최대 12h."),
    Entry("gpu", "Lightning AI",         "L4 / T4",                 "15크레딧/월 ≈ 80 GPU-h", "no",  "adapter",
          "interruptible. 지속 스토리지 → 재개 최적. 무료 물량 최대."),
    Entry("gpu", "Modal",                "T4 / L4 / A10G",          "~$30 크레딧/월",         "no",  "adapter",
          "서버리스 GPU(슬롯 대기 없음). 크레딧 소진 시 페일오버."),
    Entry("gpu", "Saturn Cloud",         "T4 16GB",                 "반복 무료 티어",          "no",  "adapter",
          "saturn-client recipe. SATURN_TOKEN 필요."),
    Entry("gpu", "SageMaker Studio Lab", "T4",                      "4h/세션·4h/24h",         "no",  "candidate",
          "⚠️ 신규 가입 2026-07-30 마감 — 쓸 거면 지금 등록. 기존 계정은 유지."),
    Entry("gpu", "Intel Tiber AI Cloud", "Gaudi2 / GPU Max",        "무료 JupyterLab",         "no",  "candidate",
          "비엔비디아, 물량 큼(포팅 필요)."),
    Entry("gpu", "HF Spaces ZeroGPU",    "H200 (버스트)",           "~5분/일, 60초/호출",     "no",  "candidate",
          "데모/추론용. 학습엔 부적합."),
    Entry("gpu", "Paperspace (DO)",      "M4000 8GB",               "6h 자동종료, 재시작 무제한","no", "candidate",
          "약함. 큐 대기 잦음."),
    Entry("gpu", "Google TRC",           "TPU v3/v4/v5 1000+대",    "신청제, 임시 무료 쿼터",  "gcp", "candidate",
          "⭐ 오픈소스/논문 공유 조건 → 이 프로젝트가 자격. 학습이면 판을 바꿈."),

    # ── Computer — 범용 컴퓨트 ──────────────────────────────────────────
    Entry("computer", "Oracle Always Free", "ARM A1 4코어/24GB",    "영구 무료",               "yes", "candidate",
          "컨트롤 플레인/데이터 준비/디스패처 상주에 이상적."),
    Entry("computer", "GitHub Actions",     "CPU 러너",             "2000분/월",               "no",  "candidate",
          "무인 디스패처 상시 실행처."),

    # ── Mobile — 모바일 기기 ────────────────────────────────────────────
    # (등록 항목 없음 — 후보 PR 환영)
]


LABS = [
    Lab("MJX+Brax",  "비RTX GPU", True,
        "채택 런타임 — MuJoCo MJX 헤드리스라 비RTX 무료 풀에서 그대로 구동."),
    Lab("Isaac Lab", "RTX 4080+", False,
        "무료 풀 구동 불가 — 기존 탈락 결정."),
]


def render(rows) -> str:
    keys = {key for key, _ in SECTIONS}
    stray = [e.name for e in rows if e.section not in keys]
    if stray:
        raise ValueError(f"SECTIONS 밖 section 을 가진 Entry: {stray} — 누락 방지")

    tag = {"adapter": "[✓ 어댑터]", "candidate": "[· 후보]"}
    cc_s = {"no": "카드X", "yes": "카드필요", "gcp": "GCP연동"}
    lines = ["무료(비중국) 컴퓨트 카탈로그 — 2026-07 기준 (값 변동, 갱신 PR 환영)"]

    for key, header in SECTIONS:
        lines.append("")
        lines.append(f"── {header} ──")
        section_rows = [e for e in rows if e.section == key]
        if not section_rows:
            lines.append("  (등록 항목 없음 — 후보 PR 환영)")
            continue
        for e in section_rows:
            lines.append(f"{tag[e.status]:10} {e.name:22} {e.spec:26} {e.free_limit:22} {cc_s[e.credit_card]}")
            lines.append(f"           └ {e.note}")

    lines.append("")
    lines.append("── Lab — 시뮬 런타임 ──")
    lines.append(f"{'이름':14} {'요구 HW':12} {'무료풀 구동':10} 비고")
    for lab in LABS:
        pool = "가능" if lab.free_pool else "불가"
        lines.append(f"{lab.name:14} {lab.requires:12} {pool:10} {lab.note}")

    return "\n".join(lines)
