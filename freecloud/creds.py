"""creds — 자격증명 선언(AuthSpec) + 그걸 해석하는 단일 로그인 엔진.

provider 마다 다른 것(무슨 값이 필요한가·어디에 저장되나·어떻게 얻나·뭘로 검증하나·선행 요구)은
전부 아래 **선언부**(`SPECS`)에 있다. 엔진(`login`/`resolve`/`status_table`)에는 provider 이름
분기가 없다 — 새 provider 는 Spec 하나를 등록하면 끝난다.

  · store   = "fcc"    → 값이 ~/.freecloud/secrets.json 에 들어간다.
              "native" → 그 도구 자체 저장소(예: ~/.modal.toml). fcc 는 값을 보관하지 않는다.
  · acquire = "paste"  → env 에 있으면 그 값, 없으면 터미널에 붙여넣기.
              "cli"    → 그 도구의 로그인 명령(stdio 상속)을 그대로 돌린다.
              "browser"→ headful 브라우저 로그인(쿠키 저장).

우선순위는 어디서나 **env > secrets.json**.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import secrets


# ── 선언 타입 ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class CredItem:
    name: str                       # 정본 이름(저장 키 = env 이름)
    env_aliases: tuple = ()         # 같은 값으로 받아주는 다른 env 이름
    secret: bool = True             # True 면 에코 없이 입력받고 시크릿 목록에 낀다
    help_url: str = ""              # 발급 위치
    required: bool = True


@dataclass
class AuthSpec:
    provider: str
    items: list = field(default_factory=list)
    store: str = "fcc"              # "fcc" | "native"
    acquire: str = "paste"          # "paste" | "cli" | "browser"
    cli_cmd: list = field(default_factory=list)
    requires: list = field(default_factory=list)      # 먼저 통과해야 하는 provider
    verify: Optional[Callable[[dict], tuple]] = None  # (values) -> (ok, detail). 저장 **전** 검증.
    # ── 아래는 provider 차이를 엔진 밖(선언)에 두기 위한 선택 훅 ──
    aliases: tuple = ()                               # `fcc login hf` 같은 별칭
    probe: Optional[Callable[[], tuple]] = None       # () -> (ok, detail). 없으면 registry probe.
    prepare: Optional[Callable[[bool], None]] = None  # (interactive) 로그인 직전 정리
    cli_headless_args: list = field(default_factory=list)   # 브라우저 못 여는 환경에서 덧붙일 인자
    browser: Optional[Callable[[], object]] = None    # acquire="browser" 실행체
    after_store: Optional[Callable[[dict], None]] = None  # 저장 직후(예: 표준 캐시 미러)


# ── 엔진: 조회 ───────────────────────────────────────────────────────────────
def _ui():
    """대화형 가드·프롬프트는 auth 모듈 것을 **호출 시점에** 찾는다(테스트가 그쪽을 몽키패치)."""
    from . import auth
    return auth


def get_spec(provider: str) -> Optional[AuthSpec]:
    spec = SPECS.get(provider)
    if spec is not None:
        return spec
    for s in SPECS.values():
        if provider in s.aliases:
            return s
    return None


def provider_names() -> list:
    """`fcc login` 이 받는 이름(별칭 포함)."""
    out = []
    for s in SPECS.values():
        out.append(s.provider)
        out.extend(s.aliases)
    return out


def _env_value(item: CredItem) -> Optional[str]:
    for n in (item.name, *item.env_aliases):
        v = os.environ.get(n)
        if v and v.strip():
            return v.strip()
    return None


def _stored_value(item: CredItem) -> Optional[str]:
    stored = secrets._load()
    for n in (item.name, *item.env_aliases):
        v = stored.get(n)
        if v:
            return str(v).strip()
    return None


def resolve(provider: str) -> dict:
    """항목별 값. env(별칭 포함) > secrets.json. 없는 항목은 빠진다."""
    spec = get_spec(provider)
    out: dict = {}
    if spec is None:
        return out
    for it in spec.items:
        v = _env_value(it) or _stored_value(it)
        if v:
            out[it.name] = v
    return out


def export_env(provider: str) -> dict:
    """resolve 결과를 os.environ 에 setdefault 로 주입(env 를 읽는 SDK 용). 이미 있는 env 는 안 건드린다."""
    vals = resolve(provider)
    for k, v in vals.items():
        os.environ.setdefault(k, v)
    return vals


def known_secret_names() -> list:
    """Spec 에서 유도한 시크릿 이름(별칭 포함). 값은 다루지 않는다."""
    out: dict = {}
    for s in SPECS.values():
        for it in s.items:
            if it.secret:
                for n in (it.name, *it.env_aliases):
                    out[n] = True
    return list(out)


#: probe 가 "로그인으로는 못 푼다 — 설치가 먼저"라고 알리는 error_class.
NOT_INSTALLED = "NOT_INSTALLED"


def probe_full(provider: str) -> tuple:
    """(ok, detail, error_class). Spec 에 probe 가 있으면 그것, 없으면 registry provider 의 probe."""
    spec = get_spec(provider)
    try:
        if spec is not None and spec.probe is not None:
            ok, detail = spec.probe()
            return bool(ok), str(detail), ""
        from . import registry
        pr = registry.get(spec.provider if spec else provider).probe()
        return bool(pr.available), str(pr.reason), str(getattr(pr, "error_class", "") or "")
    except Exception as e:
        return False, str(e)[:80], ""


def probe(provider: str) -> tuple:
    """(ok, detail)."""
    return probe_full(provider)[:2]


def status_table() -> list:
    """등록된 Spec 전부 → [(provider, ok, detail, error_class)]."""
    return [(s.provider, *probe_full(s.provider)) for s in SPECS.values()]


# ── 엔진: 로그인 ─────────────────────────────────────────────────────────────
def _env_hint(spec: AuthSpec) -> str:
    names = [it.name for it in spec.items if it.required]
    if not names:
        return ""
    return " ".join(f"{n}=..." for n in names) + f" fcc login {spec.provider}"


def _verify(spec: AuthSpec, vals: dict) -> tuple:
    if spec.verify is None:
        return True, ""
    try:
        ok, detail = spec.verify(vals)
        return bool(ok), str(detail)
    except Exception as e:
        return False, f"검증 중 예외: {str(e)[:100]}"


def _can_open_browser() -> bool:
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux") and not (
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


def _run(cmd: list) -> int:
    """도구의 명령을 stdio 상속으로 실행 → rc. 미설치는 127."""
    try:
        return subprocess.run(cmd).returncode
    except FileNotFoundError:
        return 127


#: store="native" 인데 값이 env 로 왔을 때. 그 도구가 env 를 직접 읽으므로 fcc 는 아무 데도
#: 쓰지 않는다(시크릿을 다른 명령의 argv 로 넘기지 않기 위해) — 통과 여부는 재-probe 가 본다.
ENV_ONLY_NOTE = "env 로 인증됨(저장 안 함)"


def _store(spec: AuthSpec, vals: dict) -> tuple:
    """(ok, note). store="fcc" 만 secrets.json 에 쓴다."""
    if spec.store == "native":
        return True, ENV_ONLY_NOTE
    ui = _ui()
    for it in spec.items:
        if it.name in vals:
            secrets.set(it.name, vals[it.name])
            ui._warn_if_env_shadows(it.name, vals[it.name])
    if spec.after_store is not None:
        spec.after_store(vals)
    return True, ""


def _acquire_paste(spec: AuthSpec, interactive: bool) -> tuple:
    ui = _ui()
    vals = {it.name: v for it in spec.items if (v := _env_value(it))}
    need = [it for it in spec.items if it.required and it.name not in vals]
    if vals and not need:
        # 무인 경로. ⚠️ 무효면 여기서 끝내지 않는다 — 만료 값이 env 에 남은 사람은 새 값을
        # 넣을 기회조차 없어진다(실측). 프롬프트로 넘어간다.
        ok, detail = _verify(spec, vals)
        if ok:
            _store(spec, vals)
            print(f"env 값이 유효합니다 — 저장 완료. {detail}".rstrip())
            return True, ""
        print(f"env 의 {spec.provider} 값이 무효합니다: {detail}")
        print("→ 새 값을 입력받겠습니다.")
        vals, need = {}, [it for it in spec.items if it.required]

    what = f"{spec.provider} 자격증명 입력({', '.join(it.name for it in need)})"
    if not interactive:
        ui._noninteractive_exit(what, _env_hint(spec))
    ui._require_interactive(what, _env_hint(spec))
    for it in spec.items:
        if it.name in vals:
            continue
        if it.help_url:
            print(f"{it.name} 발급: {it.help_url}")
        kept = _stored_value(it)
        label = f"{it.name} 붙여넣기" + ("(Enter=저장된 값 유지)" if kept else "") + ": "
        v = ui._prompt_secret(label) if it.secret else input(label).strip()
        v = v or kept or ""
        if v:
            vals[it.name] = v
        elif it.required:
            return False, "취소됨"
    ok, detail = _verify(spec, vals)
    if not ok:
        return False, f"검증 실패(저장 안 함): {detail}"
    _store(spec, vals)
    print(f"저장 완료 — {detail}".rstrip(" —"))
    return True, ""


def _acquire_cli(spec: AuthSpec, interactive: bool) -> tuple:
    if interactive:
        cmd = list(spec.cli_cmd)
        if not _can_open_browser():
            cmd += list(spec.cli_headless_args)
        print(f"{spec.provider} 인증: `{' '.join(cmd)}` 를 실행합니다.")
        rc = _run(cmd)
        if rc == 127:
            return False, f"{cmd[0]} CLI 미설치."
        return True, ""                            # 통과 여부는 재-probe 가 판정한다
    # 비TTY: 브라우저 로그인은 못 한다 → env 로 받은 값을 검증해 저장(store="fcc" 일 때만).
    vals = {it.name: v for it in spec.items if (v := _env_value(it))}
    if not spec.items or any(it.required and it.name not in vals for it in spec.items):
        _ui()._noninteractive_exit(f"{spec.provider} 로그인", _env_hint(spec))
    ok, detail = _verify(spec, vals)
    if not ok:
        return False, f"검증 실패(저장 안 함): {detail}"
    return _store(spec, vals)


def _acquire_browser(spec: AuthSpec, interactive: bool) -> tuple:
    if spec.browser is None:
        return False, f"'{spec.provider}' 브라우저 로그인 핸들러 없음."
    spec.browser()
    return True, ""


_ACQUIRE = {"paste": _acquire_paste, "cli": _acquire_cli, "browser": _acquire_browser}


def login(provider: str, interactive: Optional[bool] = None) -> tuple:
    """선행 요구 → (정리) → 수집 → 검증 → 저장 → 재-probe. 반환 (ok, detail).

    실패 사유는 **반환만 한다**(여기서 또 찍으면 호출부 출력과 겹친다).

    검증에 실패하면 저장하지 않는다. 최종 판정은 항상 재-probe 결과다.
    """
    spec = get_spec(provider)
    if spec is None:
        return False, f"알 수 없는 provider '{provider}'. 가능: {sorted(provider_names())}"
    tty = _ui().interactive_available()
    interactive = tty if interactive is None else (bool(interactive) and tty)

    for req in spec.requires:
        ok, detail = probe(req)
        if ok:
            continue
        print(f"▶ [{spec.provider}] 선행 요구 '{req}' 가 아직 안 됩니다({detail}) — 먼저 로그인합니다.")
        ok, detail = login(req, interactive)
        if not ok:
            return False, f"선행 '{req}' 로그인 실패: {detail}"

    if spec.prepare is not None:
        spec.prepare(interactive)

    ok, note = _ACQUIRE[spec.acquire](spec, interactive)     # 성공이면 note, 실패면 사유
    if not ok:
        return False, note
    ok, detail = probe(spec.provider)
    if ok and note:
        detail = f"{detail} — {note}"
    return ok, detail


# ═════════════════════════════════════════════════════════════════════════════
# 선언부 — provider 별 차이는 여기에만 둔다.
# ═════════════════════════════════════════════════════════════════════════════
@contextlib.contextmanager
def _temp_env(vals: dict):
    old = {k: os.environ.get(k) for k in vals}
    os.environ.update(vals)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── huggingface ──
def _hf_verify(vals: dict) -> tuple:
    return _ui().verify_hf_token(vals.get("HF_TOKEN"))


def _hf_probe() -> tuple:
    st = _ui()._hf_whoami()
    return st.ok, st.detail


def _hf_after_store(vals: dict) -> None:
    if _ui()._mirror_hf_cache(vals["HF_TOKEN"]):
        print("표준 HF 캐시(~/.cache/huggingface/token)에도 미러 — 외부 런처도 이 토큰 사용.")


# ── kaggle ──
#: OAuth(credentials.json)보다 인증 순서상 앞이라, 낡아 있으면 OAuth 를 가리는 파일들.
KAGGLE_SHADOW_FILES = ("kaggle.json", "access_token", "access_token.txt")


def _kaggle_dir() -> str:
    return os.path.expanduser("~/.kaggle")


def _kaggle_prepare(interactive: bool) -> None:
    """낡은 legacy 자격증명이 OAuth 를 가리지 않게 한다(지우지 않고 .bak 으로).

    TTY 면 파일마다 묻고, 비TTY 면 건드리지 않고 경고만 한다. `access_token*` 은 지금 probe 가
    통과 중이면 유효한 토큰일 수 있으니 묻지 않는다.
    """
    present = [n for n in KAGGLE_SHADOW_FILES if os.path.exists(os.path.join(_kaggle_dir(), n))]
    probe_ok = None
    for name in present:
        fp = os.path.join(_kaggle_dir(), name)
        if name != "kaggle.json":
            if probe_ok is None:
                probe_ok = probe("kaggle")[0]
            if probe_ok:
                continue
        if not interactive:
            print(f"⚠ {fp} 가 있습니다 — OAuth 로그인을 가릴 수 있습니다. "
                  f"(비대화형이라 그대로 둡니다. 인증이 계속 실패하면 이 파일을 치우세요.)")
            continue
        ans = input(f"{fp} 가 OAuth를 가립니다. `.bak`으로 바꿀까요? [y/N] ").strip().lower()
        if ans != "y":
            continue
        dst = fp + ".bak"
        if os.path.exists(dst):
            dst = f"{fp}.bak.{int(time.time())}"
        os.replace(fp, dst)
        print(f"  → {dst}")


def _kaggle_verify(vals: dict) -> tuple:
    """받은 KAGGLE_API_TOKEN 을 주입한 probe 로 실제 API 가 붙는지 본다."""
    from .providers.kaggle import KaggleProvider
    pr = KaggleProvider().probe(token=vals.get("KAGGLE_API_TOKEN"))
    return pr.available, pr.reason


# ── lightning ──
def _lightning_verify(vals: dict) -> tuple:
    """3종을 env 로 잠깐 올리고 `lightning_sdk.Teamspace(name=…)` 한 번(GET teamspace).

    이 한 호출이 API key·user id·teamspace 셋을 한꺼번에 확인한다. SDK 가 없으면 로컬 검증이
    불가능할 뿐이므로 '3종 존재'만 확인하고 통과시킨다(probe 가 미설치를 따로 알린다).
    """
    missing = [n for n in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID", "LIGHTNING_TEAMSPACE")
               if not vals.get(n)]
    if missing:
        return False, f"{', '.join(missing)} 없음"
    try:
        from lightning_sdk import Teamspace
    except ImportError:
        return True, "lightning-sdk 미설치 — 3종 존재만 확인(원격 검증 생략)."
    with _temp_env({k: vals[k] for k in ("LIGHTNING_API_KEY", "LIGHTNING_USER_ID",
                                         "LIGHTNING_TEAMSPACE")}):
        try:
            Teamspace(name=vals["LIGHTNING_TEAMSPACE"])
        except Exception as e:
            return False, f"Lightning 확인 실패(키·user id·teamspace 중 하나가 틀림): {str(e)[:120]}"
    return True, f"teamspace 확인: {vals['LIGHTNING_TEAMSPACE']}"


# ── browser ──
def _kaggle_ui_browser():
    from .providers.kaggle_ui import save_login_state
    return save_login_state()


def _colab_browser():
    from .providers.colab_login import save_login_state
    return save_login_state()


def _register(*specs: AuthSpec) -> dict:
    return {s.provider: s for s in specs}


#: Spec 등록표. 순서 = `fcc auth`/`fcc login` 상태표 순서.
SPECS: dict = _register(
    AuthSpec("huggingface", aliases=("hf",),
             items=[CredItem("HF_TOKEN", env_aliases=("HUGGINGFACE_TOKEN",),
                             help_url="https://huggingface.co/settings/tokens (Type: Write)")],
             store="fcc", acquire="paste",
             verify=_hf_verify, probe=_hf_probe, after_store=_hf_after_store),
    AuthSpec("lightning",
             items=[CredItem("LIGHTNING_API_KEY", help_url="lightning.ai → Settings → Keys"),
                    CredItem("LIGHTNING_USER_ID", secret=False,
                             help_url="lightning.ai → Settings → Keys (User ID)"),
                    CredItem("LIGHTNING_TEAMSPACE", secret=False,
                             help_url="owner/teamspace 슬러그 (예: myuser/default-project)")],
             store="fcc", acquire="paste", verify=_lightning_verify),
    AuthSpec("kaggle",
             items=[CredItem("KAGGLE_API_TOKEN", help_url="https://www.kaggle.com/settings/api")],
             store="fcc", acquire="cli",
             cli_cmd=["kaggle", "auth", "login", "--force"],
             cli_headless_args=["--no-launch-browser"],
             prepare=_kaggle_prepare, verify=_kaggle_verify),
    AuthSpec("kaggle-ui", store="native", acquire="browser",
             requires=["kaggle"], browser=_kaggle_ui_browser),
    AuthSpec("colab", store="native", acquire="browser", browser=_colab_browser),
    AuthSpec("modal",
             items=[CredItem("MODAL_TOKEN_ID", secret=False,
                             help_url="https://modal.com/settings/tokens"),
                    CredItem("MODAL_TOKEN_SECRET", help_url="https://modal.com/settings/tokens")],
             store="native", acquire="cli", cli_cmd=["modal", "token", "new"]),
    AuthSpec("saturn",
             items=[CredItem("SATURN_TOKEN",
                             help_url="app.community.saturnenterprise.io → Settings → User Token")],
             store="fcc", acquire="paste"),
)
