"""colab_login — Colab(구글 계정) 브라우저 로그인 상태 저장.

구글 로그인은 자동화 불가/금지(계정 잠금·ToS) → 사람이 headful로 1회 로그인하면
세션(쿠키)을 저장해 이후 재사용한다. Kaggle-UI와 동일 패턴.

⚠️ Google은 Playwright **번들 Chromium 로그인을 차단**한다("브라우저 또는 앱이 안전하지
   않을 수 있습니다" — 패스키/PIN을 다 통과해도 마지막에 거부). kaggle-ui에서 실측으로
   확인한 사항이라 여기도 동일하게 **실제 Chrome + 영속 프로필**을 쓴다.

주의: 사용자가 쓰는 `colab` CLI가 자체 인증을 가진 경우엔 그 CLI 로그인을 그대로 쓰고,
이 storage_state는 브라우저 기반 접근을 할 때만 필요하다.
"""
from __future__ import annotations

import os

from .. import paths

# 구글 로그인 성공 시 심어지는 세션 쿠키(둘 중 하나면 로그인으로 본다).
_AUTH_COOKIES = ("__Secure-1PSID", "SID")


def _state_path() -> str:
    """로그인 세션 파일 — 사용자 전역(paths.py)."""
    return os.environ.get("FREECLOUD_COLAB_STATE") or paths.user_file("colab_state.json")


def _profile_dir() -> str:
    return os.environ.get("FREECLOUD_COLAB_PROFILE") or paths.user_file("chrome_profile_colab")


def _logged_in(cookies) -> bool:
    return any(c.get("name") in _AUTH_COOKIES and c.get("value")
               for c in cookies or [])


def save_login_state(timeout_s: int = 300) -> str:
    """`fcc login colab` 구현 — 사람이 직접 로그인하면 세션을 저장.

    Enter 입력에 의존하지 않는다(비대화형에서 즉시 EOF가 나 미로그인 세션이 저장되던
    문제) — 쿠키를 폴링해 실제 로그인 완료를 감지한다.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install 'freecloud[kaggle-ui]' && playwright install chromium")
    paths.ensure_user_home()
    os.makedirs(_profile_dir(), exist_ok=True)

    with sync_playwright() as p:
        ctx = None
        last_err = None
        for channel in ("chrome", "msedge", None):
            try:
                ctx = p.chromium.launch_persistent_context(
                    _profile_dir(),
                    headless=False,  # 구글 로그인은 반드시 headful
                    channel=channel,
                    args=["--disable-blink-features=AutomationControlled"],
                    ignore_default_args=["--enable-automation"],
                )
                if channel:
                    print(f"브라우저: 실제 {channel} + 영속 프로필({_profile_dir()})")
                else:
                    print("⚠ 실제 Chrome/Edge를 못 찾아 번들 Chromium 사용 — Google이 차단할 수 있음.")
                break
            except Exception as e:
                last_err = e
                continue
        if ctx is None:
            raise SystemExit(f"브라우저 실행 실패: {last_err}")

        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        # networkidle 금지 — Colab도 SPA라 '유휴'가 오지 않는다.
        page.goto("https://colab.research.google.com/", wait_until="domcontentloaded")
        print(f"브라우저에서 구글 로그인을 완료하세요(2FA 포함). 최대 {timeout_s}초 대기하며, "
              "로그인이 감지되면 자동으로 저장하고 닫힙니다.")

        ok = False
        waited = 0
        while waited < timeout_s:
            page.wait_for_timeout(3000)
            waited += 3
            try:
                if _logged_in(ctx.cookies()):
                    ok = True
                    break
            except Exception:
                pass

        ctx.storage_state(path=_state_path())
        ctx.close()

    # Windows 콘솔(cp949)에서 죽지 않도록 ASCII 출력.
    if ok:
        print(f"[OK] 로그인 확인 + 상태 저장됨 -> {_state_path()}")
    else:
        print(f"[FAIL] {timeout_s}초 내 로그인 감지 실패 - 저장은 했지만 미로그인 상태일 수 있음 "
              f"({_state_path()}).")
    return _state_path()
