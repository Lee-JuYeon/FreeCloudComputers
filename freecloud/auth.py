"""auth — 통합 로그인/인증 상태. `freecloud login <provider>` 와 `freecloud auth`.

두 종류만 있다:
  · 토큰 방식(huggingface/kaggle/lightning/modal/saturn) — 붙여넣기/CLI 1회 → 저장소.
  · 브라우저 방식(colab/kaggle-ui) — headful 로그인 1회 → 쿠키(storage_state) 저장 → 재사용.

provider 별 차이(항목·저장 위치·획득 방법·검증·선행 요구)는 creds.py 의 AuthSpec 선언에 있고,
로그인 절차는 creds.login 엔진 하나다. 여기는 CLI 가 부르는 얇은 표면 + 대화형 가드 + HF 검증.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from . import creds, secrets


@dataclass
class AuthStatus:
    provider: str
    ok: bool
    detail: str
    error_class: str = ""       # "NOT_INSTALLED" = 로그인으로 못 푼다(설치가 먼저)


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
    except Exception as e:
        return False, f"토큰 무효/네트워크: {str(e)[:80]}"
    name = who.get("name", "?")
    role = ((who.get("auth") or {}).get("accessToken") or {}).get("role")
    # role 은 참고용으로만 붙인다 — 'fineGrained' 는 권한 구성에 따라 쓰기가 될 수도, 안 될
    # 수도 있어 정적 판정이 불가능하다. 쓰기 가능 여부는 verify_hf_write() 로 실측한다.
    return True, f"로그인됨: {name}" + (f" (토큰 role={role})" if role else "")


def _write_fail_detail(repo_id: str, e: Exception) -> str | None:
    """쓰기 실패 예외 → 사람이 읽을 detail(용량 초과면 `HF_STORAGE_FULL:` 접두).

    반환 None = '쓰기 실패가 아님'(해당 안 됨). ★용량(403 storage limit)을 auth(401/403)보다
    **먼저** 가린다 — 순서를 바꾸면 403 이 '토큰 확인' 으로 오분류된다(2026-10-31 헛짚음).
    """
    from . import hf_storage
    if hf_storage.is_storage_full(e):
        return (f"HF_STORAGE_FULL: 계정 저장소 한도 초과({repo_id}) — 토큰 문제가 아니라 저장 "
                f"공간이 가득 참. `fcc ckpt squash {repo_id}` 로 고아 체크포인트를 청소하세요 "
                f"(같은 계정의 다른 비공개 저장소도 한도를 함께 먹습니다). 원인: {str(e)[:80]}")
    from .checkpoint import _is_auth_error
    if _is_auth_error(e):
        return (f"토큰에 쓰기 권한이 없습니다({repo_id}). HF 토큰을 **Write** 로 다시 발급하세요 "
                f"— https://huggingface.co/settings/tokens 에서 'Create new token' → Write. "
                f"(Fine-grained 를 쓰신다면 해당 저장소에 대한 write 권한 + 저장소 생성 권한이 "
                f"필요합니다.) 그 뒤 `fcc login huggingface`.")
    return f"저장소 확인 실패: {str(e)[:100]}"


def verify_hf_write(repo_id: str, tok: str | None = None, *,
                    hygiene: str = "warn", keep_revisions=()) -> tuple[bool, str]:
    """`repo_id` 에 실제로 쓸 수 있는지 실측 → (ok, detail). 순서: ensure_repo → 위생 → 프로브.

    whoami 는 **읽기 전용 토큰도 통과시킨다**. 또한 토큰이 멀쩡해도 **계정 저장소가 가득 차면**
    push 에서 403 이 난다(2026-10-31). 그래서 (1) 저장소 생성(멱등)으로 권한을, (2) 정책별
    위생으로 고아 누적을, (3) 임시 브랜치 쓰기 프로브로 '실제 쓰기 + 용량'을 미리 실측한다.
    셋 다 어차피 런이 곧 할 일이라 부작용이 없다.

    반환 detail 이 `HF_STORAGE_FULL:` 로 시작하면 용량 초과(호출부가 접두로 error_class 를 가른다).
    `FCC_HF_PROBE=0` 이면 프로브를 생략(권한만 확인). hub 미설치는 통과.
    """
    try:
        from huggingface_hub import HfApi  # noqa: F401
    except ImportError:
        return True, "huggingface_hub 미설치 — 로컬 검증 생략."
    from . import hf_storage

    # (1) 저장소 존재/쓰기 권한(멱등 create_repo)
    try:
        from .checkpoint import Checkpoint
        Checkpoint(repo_id, token=tok).ensure_repo()
    except Exception as e:
        return False, _write_fail_detail(repo_id, e)

    # (2) 정책별 위생(auto 면 여기서 고아 삭제 — 프로브보다 **먼저**)
    hyg = hf_storage.hygiene(repo_id, tok, policy=hygiene, keep_revisions=keep_revisions)

    # (3) 실제 쓰기 + 용량 프로브(임시 브랜치)
    if os.environ.get("FCC_HF_PROBE") == "0":
        return True, f"쓰기 가능 확인(프로브 생략, FCC_HF_PROBE=0): {repo_id} | {hyg}"
    try:
        ok, detail = hf_storage.write_probe(repo_id, tok)
    except Exception as e:
        return False, _write_fail_detail(repo_id, e)
    return True, f"{detail} | {hyg}"


def _hf_whoami() -> AuthStatus:
    tok = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_TOKEN")
    ok, detail = verify_hf_token(tok)
    return AuthStatus("huggingface", ok, detail)


def status_all() -> list[AuthStatus]:
    """등록된 Spec 전부의 인증 상태(creds.status_table)."""
    return [AuthStatus(*row) for row in creds.status_table()]


# ── 대화형 가드 ──────────────────────────────────────────────────────────────
def _require_interactive(what: str, env_hint: str = "") -> None:
    """비대화형이면 즉시 안내하고 끝낸다.

    ⚠️ 이게 없으면 `getpass`/`input`이 입력할 수 없는 프롬프트에서 **영원히 멈춘다**
    (실측 2026-07-21: 에이전트가 `fcc login huggingface`를 돌려 아무 출력 없이 행).
    조용히 멈추는 것보다 빨리 실패하고 방법을 알려주는 편이 낫다.
    """
    if interactive_available():
        return
    _noninteractive_exit(what, env_hint)


def _noninteractive_exit(what: str, env_hint: str = "") -> None:
    """입력이 필요한데 대화형이 아닐 때의 종료(안내 포함). 항상 SystemExit."""
    lines = [
        f"[freecloud] '{what}'에는 터미널 입력이 필요한데, 지금 stdin이 TTY가 아닙니다"
        f"(파이프·에이전트·CI).",
        "  이대로 두면 입력할 수 없는 프롬프트에서 멈춥니다 → 실제 터미널 창에서 실행하세요.",
    ]
    if env_hint:
        lines.append(f"  무인으로 넘기려면 환경변수를 쓰세요:  {env_hint}")
    raise SystemExit("\n".join(lines))


def _warn_if_env_shadows(name: str, stored: str) -> None:
    """저장한 값이 환경변수에 가려지면 크게 경고한다.

    secrets.get() 은 'env 우선, 없으면 저장소'다. 그래서 만료된 토큰이 env 에 영구 설정돼
    있으면, 새 토큰을 저장해도 계속 낡은 값이 쓰인다 — 로그인은 성공한 것처럼 보이는데
    실제 사용은 계속 실패하는, 알아채기 매우 어려운 상태가 된다(실측 2026-07-21).
    """
    env_val = os.environ.get(name)
    if not env_val or env_val.strip() == stored.strip():
        return
    print()
    print(f"⚠ 경고: 환경변수 {name} 이(가) 방금 저장한 값과 다릅니다.")
    print(f"  secrets 는 env 를 우선하므로, 이대로면 저장한 토큰이 아니라 env 값이 쓰입니다.")
    print(f"  env 값을 지우세요:")
    print(f"    PowerShell(현재 세션):  $env:{name}=$null")
    print(f"    PowerShell(영구):       [Environment]::SetEnvironmentVariable('{name}', $null, 'User')")
    print(f"    bash:                   unset {name}")
    print(f"  (지운 뒤 새 터미널에서 `fcc auth` 로 확인하세요.)")
    print()


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
    """write 토큰을 env/붙여넣기로 받아 whoami 로 검증 후 저장 + 표준 HF 캐시에 미러."""
    _report(creds.login("huggingface"))


def login_lightning() -> None:
    """LIGHTNING_API_KEY·LIGHTNING_USER_ID·LIGHTNING_TEAMSPACE 3종을 받아 저장."""
    _report(creds.login("lightning"))


def login_saturn() -> None:
    _report(creds.login("saturn"))


def _report(result: tuple[bool, str]) -> None:
    """반환값을 버리는 래퍼용 — 엔진은 실패 사유를 찍지 않으므로 여기서 한 번 찍는다."""
    ok, detail = result
    if not ok:
        print(detail)


def login(provider: str, interactive: bool | None = None) -> tuple[bool, str]:
    """`fcc login <provider>` — creds 엔진에 위임. 반환 (ok, detail)."""
    return creds.login(provider, interactive)


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
    - 시도할 provider들이 미인증이면 지금 로그인할지 물어봄(creds.login — 선행 요구 자동 연쇄).
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
        if can and creds.get_spec(name) is not None:
            ans = input(f"\n▶ [{name}] 사용불가: {pr.reason}\n  지금 로그인/설정할까요? [y/N] ").strip().lower()
            if ans == "y":
                # 선행 요구(예: kaggle-ui → kaggle)는 엔진이 알아서 먼저 처리한다.
                ok, detail = creds.login(name)
                print(f"  → [{name}] {'OK' if ok else '실패'}: {detail}")
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
