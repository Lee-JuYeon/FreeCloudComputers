"""registry — provider 이름 → 어댑터 인스턴스. 새 클라우드 추가는 여기 한 줄.

기본 시도 순서 = 물량/안정성 기준: lightning(80h/월) → kaggle(안정) → colab → modal(크레딧).
job.providers 로 덮어쓸 수 있음.
"""
from __future__ import annotations

from .providers.base import Provider
from .providers.kaggle import KaggleProvider
from .providers.colab import ColabProvider
from .providers.modal import ModalProvider
from .providers.lightning import LightningProvider

_FACTORIES = {
    "lightning": LightningProvider,
    "kaggle": KaggleProvider,
    "colab": ColabProvider,
    "modal": ModalProvider,
}

DEFAULT_ORDER = ["lightning", "kaggle", "colab", "modal"]


def get(name: str) -> Provider:
    if name not in _FACTORIES:
        raise KeyError(f"알 수 없는 provider '{name}'. 가능: {list(_FACTORIES)}")
    return _FACTORIES[name]()


def available_names() -> list[str]:
    return list(_FACTORIES)
