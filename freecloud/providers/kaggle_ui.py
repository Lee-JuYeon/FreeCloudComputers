"""kaggle-ui 어댑터 — Playwright로 T4×2를 완전 무인 설정.

배경: Kaggle API/CLI는 단일 GPU(P100)만 노출. T4×2는 웹 UI 전용이라, 사람이 매번
Settings→Accelerator→GPU T4 x2→Save를 눌러야 한다. 이 어댑터가 그 클릭을 자동화한다.

흐름:
  1) KaggleProvider 헬퍼로 코드 push(커널 등록). ※push는 P100 run을 한 번 트리거하지만,
     job이 체크포인트 재개형이라 무해 — 이어서 T4×2 커밋이 최신 버전으로 덮어씀.
  2) Playwright: 저장된 로그인 상태로 에디터 열기 → 가속기 'GPU T4 x2' 선택 → Save & Run All.
  3) KaggleProvider 헬퍼로 상태 폴링 + 산출물 회수 + nvidia-smi로 T4×2 실제 확인.

인증(2FA 회피): 최초 1회 `freecloud kaggle-login` 으로 로그인 상태를 저장
  → .freecloud/kaggle_state.json (env FREECLOUD_KAGGLE_STATE 로 변경 가능).

⚠️ 셀렉터는 Kaggle 에디터 DOM(React SPA) 기준 best-effort. UI 변경 시 조정 필요.
   디버그: FREECLOUD_KAGGLE_HEADFUL=1 (브라우저 띄워서 눈으로 확인).
"""
from __future__ import annotations

import os

from ..job import Job
from .base import Probe, RunResult
from .kaggle import KaggleProvider

STATE_PATH = os.environ.get(
    "FREECLOUD_KAGGLE_STATE",
    os.path.join(os.environ.get("FREECLOUD_HOME", os.path.join(os.getcwd(), ".freecloud")),
                 "kaggle_state.json"))


class KaggleUIProvider(KaggleProvider):
    name = "kaggle-ui"
    capabilities = {"gpu": "T4x2", "vram_gb": 32, "headless": True,
                    "note": "Playwright로 UI 자동화. T4x2(2×16GB). kaggle-login 선행 필요."}

    def _pw(self):
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
            return sync_playwright
        except ImportError:
            return None

    def probe(self) -> Probe:
        base = super().probe()
        if not base.available:
            return base
        if self._pw() is None:
            return Probe(False, "playwright 미설치(pip install 'freecloud[kaggle-ui]' && playwright install chromium).")
        if not os.path.exists(STATE_PATH):
            return Probe(False, f"로그인 상태 없음 → 먼저 'freecloud kaggle-login' 실행({STATE_PATH}).")
        return Probe(True, "kaggle-ui 준비 OK(Playwright + 저장된 로그인).")

    def run(self, job: Job) -> RunResult:
        if self._pw() is None:
            return RunResult(False, "error", "playwright 미설치.", error_class="JOB_CONFIG")
        if not os.path.exists(STATE_PATH):
            return RunResult(False, "needs_setup",
                             "kaggle-login 선행 필요(로그인 상태 파일 없음).", error_class="AUTH")
        # 1) 코드 push(커널 등록)
        work = self._write_kernel_dir(job)
        rc, log = self._push(work, job)
        if rc != 0:
            from ..errors import classify
            d = classify(log)
            return RunResult(False, "error", d.advice, log[-1000:], d.error_class)
        # 2) UI로 T4×2 설정 + 커밋
        ok, why = self._set_t4x2_and_commit(job)
        if not ok:
            return RunResult(False, "needs_setup", f"UI 자동화 실패: {why}",
                             error_class="UI_AUTOMATION")
        # 3) 폴링 + GPU 검증(부모 헬퍼 재사용)
        return self._poll_and_fetch(job, work)

    # ── Playwright 자동화 ────────────────────────────────────────────────────
    def _set_t4x2_and_commit(self, job: Job) -> tuple[bool, str]:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        user = self._user()
        slug = self.kernel_slug(job)
        url = f"https://www.kaggle.com/code/{user}/{slug}/edit"
        headful = os.environ.get("FREECLOUD_KAGGLE_HEADFUL") == "1"

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headful)
            ctx = browser.new_context(storage_state=STATE_PATH)
            page = ctx.new_page()
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)

                # (a) 가속기 설정 열기 — Settings 사이드바의 'Accelerator' 섹션.
                #     ⚠️ Kaggle DOM 변동 → 여러 방식 fallback. 라이브 검증 시 조정.
                self._select_accelerator(page, "GPU T4 x2")

                # (b) Save Version → Save & Run All (Commit) → Save
                self._save_and_run_all(page)

                page.wait_for_timeout(3000)  # 커밋 제출 반영 대기
                return True, ""
            except PWTimeout as e:
                return False, f"셀렉터 타임아웃(UI 변경 가능): {e}"
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"
            finally:
                ctx.close()
                browser.close()

    @staticmethod
    def _select_accelerator(page, label: str) -> None:
        """가속기 드롭다운에서 label 선택. best-effort 다단 fallback."""
        # 설정 패널이 접혀있으면 'Accelerator' 헤더 클릭해 펼치기 시도
        for opener in ("Accelerator", "Settings", "Session options"):
            try:
                page.get_by_text(opener, exact=False).first.click(timeout=4000)
                break
            except Exception:
                continue
        # 현재 값(None/GPU P100 등) 버튼을 눌러 옵션 목록 열기
        for cur in ("GPU P100", "None", "GPU T4 x2", "Accelerator"):
            try:
                page.get_by_text(cur, exact=False).first.click(timeout=3000)
                break
            except Exception:
                continue
        # 원하는 옵션 클릭
        page.get_by_text(label, exact=False).first.click(timeout=8000)

    @staticmethod
    def _save_and_run_all(page) -> None:
        page.get_by_role("button", name="Save Version").click(timeout=10000)
        # 다이얼로그: 'Save & Run All (Commit)' 라디오 선택
        for opt in ("Save & Run All", "Save & Run All (Commit)", "Run All"):
            try:
                page.get_by_text(opt, exact=False).first.click(timeout=3000)
                break
            except Exception:
                continue
        page.get_by_role("button", name="Save").last.click(timeout=10000)


def save_login_state(headful: bool = True) -> str:
    """`freecloud kaggle-login` 구현 — 브라우저 띄워 사용자가 직접 로그인하면 상태 저장.

    2FA 포함 로그인은 사람이 하고, 그 세션(쿠키)을 파일로 저장해 이후 무인 자동화가 재사용.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install 'freecloud[kaggle-ui]' && playwright install chromium")
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # 로그인은 반드시 headful
        ctx = browser.new_context()
        page = ctx.new_page()
        page.goto("https://www.kaggle.com/account/login", wait_until="networkidle")
        print("브라우저에서 Kaggle 로그인을 완료하세요(2FA 포함). "
              "로그인 후 이 터미널에서 Enter를 누르면 세션을 저장합니다.")
        try:
            input()
        except EOFError:
            page.wait_for_timeout(60000)
        ctx.storage_state(path=STATE_PATH)
        browser.close()
    print(f"로그인 상태 저장됨 → {STATE_PATH}")
    return STATE_PATH
