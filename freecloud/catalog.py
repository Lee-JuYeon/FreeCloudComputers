"""catalog — 비중국 무료 클라우드 컴퓨트 카탈로그(2026-07 조사 기준).

`freecloud clouds` 로 출력. adapter가 아직 없는 것도 후보로 실어 로드맵 역할.
값은 수시 변동 → 갱신 PR 환영. status: adapter(구현됨) / candidate(후보) / control(컨트롤플레인).
"""
from __future__ import annotations

CATALOG = [
    # name, gpu, free_limit, credit_card, status, note
    ("Kaggle",              "P100 / T4x2 / TPU v5e-8", "~30 GPU-h/주, 세션 12h", "no",  "adapter",
     "API=단일 P100. T4x2는 provider 'kaggle-ui'(Playwright 자동화)로. 출력 /kaggle/working 영속."),
    ("Google Colab",        "T4 16GB",                 "~15-30 GPU-h/주(변동)",  "no",  "adapter",
     "무료=계정당 GPU 동시 1개. 90분 유휴 끊김, 세션 최대 12h."),
    ("Lightning AI",        "L4 / T4",                 "15크레딧/월 ≈ 80 GPU-h", "no",  "adapter",
     "interruptible. 지속 스토리지 → 재개 최적. 무료 물량 최대."),
    ("Modal",               "T4 / L4 / A10G",          "~$30 크레딧/월",         "no",  "adapter",
     "서버리스 GPU(슬롯 대기 없음). 크레딧 소진 시 페일오버."),
    ("Saturn Cloud",        "T4 16GB",                 "반복 무료 티어",          "no",  "candidate",
     "Colab/Kaggle 대안. 어댑터 추가 후보."),
    ("SageMaker Studio Lab","T4",                      "4h/세션·4h/24h",         "no",  "candidate",
     "⚠️ 신규 가입 2026-07-30 마감 — 쓸 거면 지금 등록. 기존 계정은 유지."),
    ("Intel Tiber AI Cloud","Gaudi2 / GPU Max",        "무료 JupyterLab",         "no",  "candidate",
     "비엔비디아, 물량 큼(포팅 필요)."),
    ("HF Spaces ZeroGPU",   "H200 (버스트)",           "~5분/일, 60초/호출",     "no",  "candidate",
     "데모/추론용. 학습엔 부적합."),
    ("Paperspace (DO)",     "M4000 8GB",               "6h 자동종료, 재시작 무제한","no", "candidate",
     "약함. 큐 대기 잦음."),
    ("Google TRC",          "TPU v3/v4/v5 1000+대",    "신청제, 임시 무료 쿼터",  "gcp", "candidate",
     "⭐ 오픈소스/논문 공유 조건 → 이 프로젝트가 자격. 학습이면 판을 바꿈."),
    ("Oracle Always Free",  "없음(ARM A1 4코어/24GB)",  "영구 무료",               "yes", "control",
     "컨트롤 플레인/데이터 준비/디스패처 상주에 이상적."),
    ("GitHub Actions",      "없음(CPU)",               "2000분/월",               "no",  "control",
     "무인 디스패처 상시 실행처."),
]


def render(rows) -> str:
    lines = ["무료(비중국) 클라우드 카탈로그 — 2026-07 기준 (값 변동, 갱신 PR 환영)\n"]
    for name, gpu, lim, cc, status, note in rows:
        tag = {"adapter": "[✓ 어댑터]", "candidate": "[· 후보]", "control": "[⚙ 컨트롤]"}[status]
        cc_s = {"no": "카드X", "yes": "카드필요", "gcp": "GCP연동"}[cc]
        lines.append(f"{tag:10} {name:22} {gpu:26} {lim:22} {cc_s}")
        lines.append(f"           └ {note}")
    return "\n".join(lines)
