"""colab_login — Colab(구글 계정) 브라우저 로그인 상태 저장.

구글 로그인은 자동화 불가/금지(계정 잠금·ToS) → 사람이 headful로 1회 로그인하면
세션(쿠키)을 저장해 이후 재사용한다. Kaggle-UI와 동일 패턴.

주의: 사용자가 쓰는 `colab` CLI가 자체 인증을 가진 경우엔 그 CLI 로그인을 그대로 쓰고,
이 storage_state는 브라우저 기반 접근을 할 때만 필요하다.
"""
from __future__ import annotations

import os

STATE_PATH = os.environ.get(
    "FREECLOUD_COLAB_STATE",
    os.path.join(os.environ.get("FREECLOUD_HOME", os.path.join(os.getcwd(), ".freecloud")),
                 "colab_state.json"))


def save_login_state() -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install 'freecloud[kaggle-ui]' && playwright install chromium")
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)   # 구글 로그인은 반드시 headful
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto("https://colab.research.google.com/", wait_until="networkidle")
        print("브라우저에서 구글 로그인을 완료하세요(2FA 포함). "
              "Colab이 열리면 이 터미널에서 Enter를 누르세요.")
        try:
            input()
        except EOFError:
            page.wait_for_timeout(60000)
        ctx.storage_state(path=STATE_PATH)
        browser.close()
    print(f"로그인 상태 저장됨 → {STATE_PATH}")
    return STATE_PATH
