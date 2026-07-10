"""state — provider 쿨다운 영속화.

'Kaggle 쿼터 소진 → 다음 토요일까지 쉬어라'를 프로세스 재시작에도 기억해야
무인 루프가 죽은 provider를 계속 두드리지 않는다. 간단한 JSON 파일 DB.

Date.now류 비결정 함수 회피와 무관 — 여긴 실제 런타임이므로 time.time() 사용.
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional

_DIR = os.environ.get("FREECLOUD_HOME", os.path.join(os.getcwd(), ".freecloud"))
_PATH = os.path.join(_DIR, "cooldown.json")


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


def cooldown_remaining(provider: str) -> float:
    """0이면 쓸 수 있음. >0이면 남은 초."""
    until = _load().get(provider, {}).get("until", 0)
    return max(0.0, until - time.time())


def set_cooldown(provider: str, seconds: int, reason: str = "") -> None:
    d = _load()
    d[provider] = {"until": time.time() + seconds, "reason": reason, "set_at": time.time()}
    _save(d)


def clear_cooldown(provider: str) -> None:
    d = _load()
    if provider in d:
        del d[provider]
        _save(d)


def snapshot() -> dict:
    """CLI status용 — provider별 남은 쿨다운."""
    out = {}
    d = _load()
    now = time.time()
    for prov, rec in d.items():
        rem = max(0.0, rec.get("until", 0) - now)
        out[prov] = {"cooldown_s": round(rem), "reason": rec.get("reason", "")}
    return out
