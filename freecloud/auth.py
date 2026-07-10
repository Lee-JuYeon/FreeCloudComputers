"""auth — 통합 로그인/인증 상태. `freecloud login <provider>` 와 `freecloud auth`.

두 종류만 있다:
  · 토큰 방식(huggingface/kaggle/lightning/modal/saturn) — 붙여넣기/CLI 1회 → 저장소.
  · 브라우저 방식(colab/kaggle-ui) — headful 로그인 1회 → 쿠키(storage_state) 저장 → 재사용.

각 provider의 로그인 핸들러 + 상태 체크를 여기 모아 CLI가 호출.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import secrets


@dataclass
class AuthStatus:
    provider: str
    ok: bool
    detail: str


# ── 상태 체크 ────────────────────────────────────────────────────────────────
def _hf_whoami() -> AuthStatus:
    tok = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_TOKEN")
    if not tok:
        return AuthStatus("huggingface", False, "HF_TOKEN 없음 → freecloud login huggingface")
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=tok)
        return AuthStatus("huggingface", True, f"로그인됨: {who.get('name','?')}")
    except ImportError:
        return AuthStatus("huggingface", False, "huggingface_hub 미설치(pip install 'freecloud[hub]').")
    except Exception as e:
        return AuthStatus("huggingface", False, f"토큰 무효/네트워크: {str(e)[:80]}")


def status_all() -> list[AuthStatus]:
    out = [_hf_whoami()]
    # provider들의 자체 probe를 인증 상태로 재사용
    from . import registry
    for name in registry.available_names():
        try:
            pr = registry.get(name).probe()
            out.append(AuthStatus(name, pr.available, pr.reason))
        except Exception as e:
            out.append(AuthStatus(name, False, str(e)[:80]))
    return out


# ── 로그인 핸들러 ────────────────────────────────────────────────────────────
def login_huggingface() -> None:
    """write 토큰을 붙여넣기 받아 whoami로 검증 후 저장."""
    print("HF write 토큰을 발급하세요: https://huggingface.co/settings/tokens (Type: Write)")
    tok = _prompt_secret("HF 토큰 붙여넣기: ")
    if not tok:
        print("취소됨."); return
    try:
        from huggingface_hub import HfApi
        who = HfApi().whoami(token=tok)
    except ImportError:
        print("huggingface_hub 미설치 → 검증 없이 저장(pip install 'freecloud[hub]' 권장).")
        secrets.set("HF_TOKEN", tok); return
    except Exception as e:
        print(f"토큰 검증 실패(저장 안 함): {e}"); return
    secrets.set("HF_TOKEN", tok)
    print(f"저장 완료 — 계정: {who.get('name','?')}")


def login_kaggle() -> None:
    """~/.kaggle/kaggle.json 안내 + KAGGLE_KEY 저장 옵션."""
    kj = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kj):
        print(f"이미 있음: {kj} — 그대로 사용됩니다.")
        return
    print("Kaggle 토큰: kaggle.com → Settings → API → 'Create New Token' → kaggle.json 다운로드")
    print(f"그 파일을 {kj} 에 두거나, 아래에 username/key를 입력하세요.")
    user = input("KAGGLE_USERNAME: ").strip()
    key = _prompt_secret("KAGGLE_KEY: ")
    if user and key:
        os.makedirs(os.path.dirname(kj), exist_ok=True)
        import json
        with open(kj, "w", encoding="utf-8") as f:
            json.dump({"username": user, "key": key}, f)
        try:
            os.chmod(kj, 0o600)
        except Exception:
            pass
        print(f"저장 완료 → {kj}")


def login_modal() -> None:
    print("Modal 인증: 브라우저가 열립니다(최초 1회).")
    os.system("modal token new")


def login_lightning() -> None:
    print("Lightning: lightning.ai → Settings 에서 User ID / API Key 발급 후 env로 설정:")
    print("  export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=... LIGHTNING_TEAMSPACE=you/space")
    key = _prompt_secret("LIGHTNING_API_KEY(저장하려면 붙여넣기, 건너뛰려면 Enter): ")
    if key:
        secrets.set("LIGHTNING_API_KEY", key)
        print("저장됨. USER_ID/TEAMSPACE는 env로 설정하세요.")


def login_saturn() -> None:
    print("Saturn Cloud: app.community.saturnenterprise.io → Settings 에서 User Token 발급.")
    tok = _prompt_secret("SATURN_TOKEN 붙여넣기: ")
    if tok:
        secrets.set("SATURN_TOKEN", tok)
        print("저장됨.")


def login_browser(provider: str) -> None:
    """colab/kaggle-ui — Playwright headful 로그인 → storage_state 저장."""
    if provider in ("kaggle-ui", "kaggle"):
        from .providers.kaggle_ui import save_login_state
        save_login_state()
        return
    if provider == "colab":
        from .providers.colab_login import save_login_state as colab_save
        colab_save()
        return
    print(f"'{provider}' 브라우저 로그인 핸들러 없음.")


_HANDLERS = {
    "huggingface": login_huggingface,
    "hf": login_huggingface,
    "kaggle": login_kaggle,
    "kaggle-ui": lambda: login_browser("kaggle-ui"),
    "colab": lambda: login_browser("colab"),
    "modal": login_modal,
    "lightning": login_lightning,
    "saturn": login_saturn,
}


def login(provider: str) -> None:
    fn = _HANDLERS.get(provider)
    if not fn:
        print(f"알 수 없는 provider '{provider}'. 가능: {sorted(_HANDLERS)}")
        return
    fn()


def _prompt_secret(prompt: str) -> str:
    """터미널에서 시크릿 입력(가능하면 에코 없이)."""
    try:
        import getpass
        return getpass.getpass(prompt).strip()
    except Exception:
        return input(prompt).strip()
