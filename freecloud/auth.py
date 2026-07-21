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
def verify_hf_token(tok: str | None) -> tuple[bool, str]:
    """HF 토큰 유효성 검증 → (ok, detail). run 프리플라이트와 상태표시가 공유.

    - 토큰 없음        → (False, ...)  (확실히 실패)
    - hub 미설치       → (True,  ...)  (로컬 검증 불가일 뿐 무효는 아님 → 원격서 검증)
    - whoami 성공/실패 → (True/False, ...)
    """
    if not tok:
        return False, "HF_TOKEN 없음 → freecloud login huggingface"
    try:
        from huggingface_hub import HfApi
    except ImportError:
        return True, "huggingface_hub 미설치 — 로컬 검증 생략(원격 노드서 설치·검증)."
    try:
        who = HfApi().whoami(token=tok)
        return True, f"로그인됨: {who.get('name','?')}"
    except Exception as e:
        return False, f"토큰 무효/네트워크: {str(e)[:80]}"


def _hf_whoami() -> AuthStatus:
    tok = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_TOKEN")
    ok, detail = verify_hf_token(tok)
    return AuthStatus("huggingface", ok, detail)


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


# ── 대화형 가드 ──────────────────────────────────────────────────────────────
def _require_interactive(what: str, env_hint: str = "") -> None:
    """비대화형이면 즉시 안내하고 끝낸다.

    ⚠️ 이게 없으면 `getpass`/`input`이 입력할 수 없는 프롬프트에서 **영원히 멈춘다**
    (실측 2026-07-21: 에이전트가 `fcc login huggingface`를 돌려 아무 출력 없이 행).
    조용히 멈추는 것보다 빨리 실패하고 방법을 알려주는 편이 낫다.
    """
    if interactive_available():
        return
    lines = [
        f"[freecloud] '{what}'에는 터미널 입력이 필요한데, 지금 stdin이 TTY가 아닙니다"
        f"(파이프·에이전트·CI).",
        "  이대로 두면 입력할 수 없는 프롬프트에서 멈춥니다 → 실제 터미널 창에서 실행하세요.",
    ]
    if env_hint:
        lines.append(f"  무인으로 넘기려면 환경변수를 쓰세요:  {env_hint}")
    raise SystemExit("\n".join(lines))


def _from_env(*names: str) -> str | None:
    for n in names:
        v = os.environ.get(n)
        if v:
            return v.strip()
    return None


# ── 로그인 핸들러 ────────────────────────────────────────────────────────────
def _mirror_hf_cache(tok: str) -> str | None:
    """표준 HF 캐시(~/.cache/huggingface/token)에도 미러.

    이렇게 하면 freecloud 밖 도구들(`hf`/`huggingface_hub` CLI, Kaggle 토큰 임베드 런처 등)이
    같은 토큰을 쓴다 → '한 곳 갱신 = 어디서든 반영'. HF_HOME 존중."""
    try:
        base = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        os.makedirs(base, exist_ok=True)
        path = os.path.join(base, "token")
        with open(path, "w", encoding="utf-8") as f:
            f.write(tok.strip())
        try:
            import stat as _stat
            os.chmod(path, _stat.S_IRUSR | _stat.S_IWUSR)  # 0600 (posix)
        except Exception:
            pass
        return path
    except Exception as e:
        print(f"(경고) HF 캐시 미러 실패: {str(e)[:80]}")
        return None


def login_huggingface() -> None:
    """write 토큰을 붙여넣기 받아 whoami로 검증 후 저장 + 표준 HF 캐시에 미러."""
    # env에 이미 있으면 프롬프트 없이 검증·저장 — 무인 경로.
    tok = _from_env("HF_TOKEN", "HUGGINGFACE_TOKEN")
    if not tok:
        _require_interactive("HF 토큰 입력", "HF_TOKEN=hf_xxx fcc login huggingface")
        print("HF write 토큰을 발급하세요: https://huggingface.co/settings/tokens (Type: Write)")
        tok = _prompt_secret("HF 토큰 붙여넣기: ")
    if not tok:
        print("취소됨."); return
    ok, detail = verify_hf_token(tok)
    if not ok:
        print(f"토큰 검증 실패(저장 안 함): {detail}"); return
    secrets.set("HF_TOKEN", tok)
    if _mirror_hf_cache(tok):
        print("표준 HF 캐시(~/.cache/huggingface/token)에도 미러 — 외부 런처도 이 토큰 사용.")
    print(f"저장 완료 — {detail}")


def login_kaggle() -> None:
    """~/.kaggle/kaggle.json 안내 + KAGGLE_KEY 저장 옵션."""
    kj = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kj):
        print(f"이미 있음: {kj} — 그대로 사용됩니다.")
        return
    user = _from_env("KAGGLE_USERNAME")
    key = _from_env("KAGGLE_KEY")
    if not (user and key):
        _require_interactive("Kaggle username/key 입력",
                             "KAGGLE_USERNAME=... KAGGLE_KEY=... fcc login kaggle")
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
    key = _from_env("LIGHTNING_API_KEY")
    if not key:
        _require_interactive("Lightning API key 입력",
                             "LIGHTNING_API_KEY=... fcc login lightning")
        print("Lightning: lightning.ai → Settings 에서 User ID / API Key 발급 후 env로 설정:")
        print("  export LIGHTNING_USER_ID=... LIGHTNING_API_KEY=... LIGHTNING_TEAMSPACE=you/space")
        key = _prompt_secret("LIGHTNING_API_KEY(저장하려면 붙여넣기, 건너뛰려면 Enter): ")
    if key:
        secrets.set("LIGHTNING_API_KEY", key)
        print("저장됨. USER_ID/TEAMSPACE는 env로 설정하세요.")


def login_saturn() -> None:
    tok = _from_env("SATURN_TOKEN")
    if not tok:
        _require_interactive("Saturn 토큰 입력", "SATURN_TOKEN=... fcc login saturn")
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


def interactive_available() -> bool:
    """TTY면 대화형 로그인 유도 가능(CI/파이프에선 False)."""
    import sys
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def ensure_for_job(job, order, interactive: bool = True) -> None:
    """Cloudflare CLI 스타일 온디맨드 인증 — 실행 전에 필요한 로그인을 그 자리에서 유도.

    - job.checkpoint_repo 있으면 HF 토큰 유효성 확인 → 없/무효면 즉석 로그인.
    - 시도할 provider들이 미인증이면 지금 로그인할지 물어봄.
    비대화형(CI)에선 경고만 출력하고 진행.
    """
    can = interactive and interactive_available()

    if getattr(job, "checkpoint_repo", ""):
        st = _hf_whoami()
        if not st.ok:
            if can:
                print(f"\n▶ 체크포인트 저장소({job.checkpoint_repo})에 HF 토큰이 필요합니다. ({st.detail})")
                login_huggingface()
            else:
                print(f"⚠ HF 토큰 없음/무효({st.detail}) — 체크포인트 재개가 동작하지 않을 수 있음.")

    from . import registry
    for name in order:
        try:
            prov = registry.get(name)
        except Exception:
            continue
        pr = prov.probe()
        if pr.available:
            continue
        if can and name in _HANDLERS:
            ans = input(f"\n▶ [{name}] 사용불가: {pr.reason}\n  지금 로그인/설정할까요? [y/N] ").strip().lower()
            if ans == "y":
                login(name)
        else:
            print(f"⚠ [{name}] {pr.reason} (스킵됨)")


def _prompt_secret(prompt: str) -> str:
    """터미널에서 시크릿 입력(가능하면 에코 없이).

    호출부가 _require_interactive 가드를 빠뜨려도 여기서 한 번 더 막는다 — 비대화형에서
    프롬프트에 걸려 멈추는 것이 이 CLI에서 제일 진단하기 어려운 실패 모드였다.
    """
    _require_interactive(prompt.strip().rstrip(":"))
    try:
        import getpass
        return getpass.getpass(prompt).strip()
    except Exception:
        return input(prompt).strip()
