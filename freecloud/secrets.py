"""secrets — 로컬 시크릿 저장소 + 원격 주입 소스.

우선순위: 환경변수 > 저장소파일(.freecloud/secrets.json, gitignore).
env가 있으면 그걸 쓰고(저장 안 함), 없으면 저장소에서 읽는다.

원격 주입: 오케스트레이터가 job에 필요한 시크릿(특히 HF_TOKEN)을 각 provider의
원격 env로 실어보낸다 → job.yaml에 토큰을 안 적어도 됨. collect_for_job() 참고.

⚠️ 저장소는 평문 JSON이다(개발 도구 편의). 진짜 민감하면 env만 쓰고 파일 저장은 피할 것.
파일은 chmod 600 시도. gitignore로 커밋 차단됨.
"""
from __future__ import annotations

import json
import os
import stat

_DIR = os.environ.get("FREECLOUD_HOME", os.path.join(os.getcwd(), ".freecloud"))
_PATH = os.path.join(_DIR, "secrets.json")

# job에 checkpoint_repo가 있으면 원격에 자동 주입할 시크릿 이름들(HF 계열).
CKPT_SECRETS = ["HF_TOKEN", "HUGGINGFACE_TOKEN"]


def _load() -> dict:
    try:
        with open(_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(d: dict) -> None:
    os.makedirs(_DIR, exist_ok=True)
    tmp = _PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _PATH)
    try:
        os.chmod(_PATH, stat.S_IRUSR | stat.S_IWUSR)  # 0600 (posix)
    except Exception:
        pass


def get(name: str) -> str | None:
    """env 우선, 없으면 저장소."""
    return os.environ.get(name) or _load().get(name)


def set(name: str, value: str) -> None:
    d = _load()
    d[name] = value
    _save(d)


def delete(name: str) -> None:
    d = _load()
    if name in d:
        del d[name]
        _save(d)


def names() -> list[str]:
    """저장소 + 관련 env에 존재하는 시크릿 이름(값은 노출 안 함).

    주의: 이 모듈은 set()을 함수로 재정의하므로 빌트인 set을 쓰지 않는다(dict로 dedup)."""
    have: dict[str, bool] = {k: True for k in _load().keys()}   # 저장소
    for k in CKPT_SECRETS + ["KAGGLE_KEY", "LIGHTNING_API_KEY", "MODAL_TOKEN_SECRET",
                             "SATURN_TOKEN"]:
        if os.environ.get(k):
            have[k] = True
    return sorted(have.keys())


def collect_for_job(job) -> dict:
    """이 job을 원격에서 돌릴 때 노드에 주입해야 할 시크릿 env dict.

    - checkpoint_repo 있으면 HF 토큰 자동 포함(있는 것만).
    - job.secrets 로 추가 시크릿 이름 지정 가능.
    """
    out: dict[str, str] = {}
    wanted: list[str] = list(getattr(job, "secrets", []) or [])
    if getattr(job, "checkpoint_repo", ""):
        wanted += CKPT_SECRETS
    for name in wanted:
        v = get(name)
        if v:
            # HF는 표준 env 이름(HF_TOKEN)으로 통일해 주입
            out[name] = v
            if name in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
                out.setdefault("HF_TOKEN", v)
    return out
