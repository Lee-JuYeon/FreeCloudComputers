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

from . import paths

# 쿨다운은 **사용자 전역**이다 — 쿼터 소진은 프로젝트가 아니라 계정의 속성이라, 실행
# 디렉토리마다 갈라지면 이미 죽은 provider 를 다시 두드리게 된다. 상세는 paths.py.


def _path() -> str:
    return paths.user_file("cooldown.json")


def _load() -> dict:
    try:
        with open(_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save(d: dict) -> None:
    path = paths.ensure_user_home() and _path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


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
