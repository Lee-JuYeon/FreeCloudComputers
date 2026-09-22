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
import re

from ..job import Job
from .base import Probe, RunResult
from .kaggle import KaggleProvider

from .. import paths


def _state_path() -> str:
    """로그인 세션 파일 — 사용자 전역(paths.py). 실행 디렉토리와 무관하게 같은 로그인을 쓴다."""
    return os.environ.get("FREECLOUD_KAGGLE_STATE") or paths.user_file("kaggle_state.json")


def _profile_dir() -> str:
    """실제 Chrome 프로필 — Google이 번들 Chromium 로그인을 차단하므로 영속 프로필 사용."""
    return os.environ.get("FREECLOUD_KAGGLE_PROFILE") or paths.user_file("chrome_profile")


def _cookie_says_anon(cookies) -> bool | None:
    """Kaggle CLIENT-TOKEN(JWT)의 anon 클레임 → True=미로그인, False=로그인, None=판정불가.

    로그인 여부를 '파일 존재'가 아니라 실제 세션 내용으로 판정하기 위한 것.
    JWT는 서명 검증 없이 페이로드만 읽는다(우리 자신의 쿠키 상태 확인 용도).
    """
    import base64
    import json
    for c in cookies or []:
        if (c.get("name") if isinstance(c, dict) else None) != "CLIENT-TOKEN":
            continue
        try:
            payload = c["value"].split(".")[1]
            payload += "=" * (-len(payload) % 4)
            data = json.loads(base64.urlsafe_b64decode(payload))
            return bool(data.get("anon", False))
        except Exception:
            return None
    return None


#: 사용자가 로그인 감지 전에 창을 닫았을 때 보여줄 한 줄(설계 §1.F / 불변조건 9).
BROWSER_CLOSED_MSG = ("[FAIL] 브라우저가 로그인 감지 전에 닫혔습니다 — "
                      "다시 실행하고 창을 닫지 마세요(자동으로 닫힙니다)")


def _is_browser_closed(exc: BaseException) -> bool:
    """예외가 '사용자가 브라우저를 닫았다'인가.

    playwright 를 import 해서 isinstance 로 보지 않는다 — playwright 가 없는 환경
    (그리고 테스트의 가짜 컨텍스트)에서도 같은 판정이 나와야 한다. 클래스 이름과
    메시지로 본다. TargetClosedError 는 Playwright Error 의 하위 타입이라 이름이
    MRO 어딘가에 남는다.
    """
    names = {c.__name__ for c in type(exc).__mro__}
    if "TargetClosedError" in names:
        return True
    low = str(exc).lower()
    return ("target page, context or browser has been closed" in low
            or "browser has been closed" in low
            or "target closed" in low)


def session_is_logged_in(path: str | None = None) -> tuple[bool, str]:
    """저장된 storage_state가 '실제 로그인된' 세션인지 → (ok, detail)."""
    import json
    path = path or _state_path()
    if not os.path.exists(path):
        return False, f"로그인 상태 없음 → 'fcc kaggle-login' 실행({path})."
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        return False, f"상태 파일 손상({type(e).__name__}) → 'fcc kaggle-login' 재실행."
    anon = _cookie_says_anon(state.get("cookies"))
    if anon is True:
        return False, "저장된 세션이 익명(미로그인) → 'fcc kaggle-login' 재실행."
    if anon is None:
        return False, "세션에 CLIENT-TOKEN 없음(미로그인 추정) → 'fcc kaggle-login' 재실행."
    return True, "로그인된 세션."


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
        # ⚠️ 파일 존재만 보면 '익명 세션'이 통과해 실패가 AUTH가 아닌 UI_AUTOMATION으로
        #    오분류된다(실측 2026-07-20). 세션 내용까지 검사한다.
        ok, detail = session_is_logged_in(_state_path())
        if not ok:
            return Probe(False, detail)
        return Probe(True, "kaggle-ui 준비 OK(Playwright + 저장된 로그인).")

    def run(self, job: Job) -> RunResult:
        if self._pw() is None:
            return RunResult(False, "error", "playwright 미설치.", error_class="JOB_CONFIG")
        ok, detail = session_is_logged_in(_state_path())
        if not ok:
            return RunResult(False, "needs_setup", f"kaggle-login 선행 필요: {detail}",
                             error_class="AUTH")
        # 1) 코드 push(커널 등록) — workdir 업로드·dataset_sources 배선은 부모 헬퍼가
        #    한다(설계 §3: kaggle-ui 는 재사용만으로 결함 A 가 자동 적용돼야 한다).
        work, staged_err = self._stage_and_write(job)
        if staged_err is not None:
            return staged_err
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
            ctx = browser.new_context(storage_state=_state_path())
            page = ctx.new_page()
            try:
                # ⚠️ networkidle 금지 — Kaggle 에디터는 웹소켓/폴링이 상시 열려 있어
                #    '유휴'가 영원히 오지 않는다(실측: goto 60s 타임아웃 3연속).
                page.goto(url, wait_until="domcontentloaded", timeout=60000)
                # SPA가 에디터를 렌더할 때까지는 별도로 기다린다.
                page.wait_for_selector("text=Save Version", timeout=60000)

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
        """가속기 콤보박스에서 label 선택. 실패하면 조용히 넘어가지 않고 예외를 던진다.

        실측 DOM(2026-07-20 라이브 덤프):
          · 패널 토글  = button  "Expand Session options" / "Collapse Session options"
          · 가속기     = div[role=combobox] aria-label="Select Accelerator. <현재값> currently selected."
          · 옵션       = role=option  name ∈ {"None", "GPU T4 x2", "GPU P100"}
        ('Accelerator'/'T4' 텍스트로는 못 찾는다 — 콤보박스를 열기 전엔 DOM에 없음.)

        ⚠️ 구버전은 전 단계가 try/except라 아무것도 못 찾아도 통과했고, 그 과정에서 상단
           'Settings' 메뉴를 열어둔 채 빠져나와 다음 단계(Save Version) 클릭을 오버레이로
           막았다. 그래서 여기서는 실패를 반드시 표면화한다.
        """
        # 1) 패널 펼치기 — 이미 펼쳐져 있으면 이 버튼이 없으므로(=Collapse) 무시.
        try:
            page.get_by_role("button", name="Expand Session options").click(timeout=8000)
            page.wait_for_timeout(1500)
        except Exception:
            pass

        combo = page.get_by_role("combobox", name=re.compile("Select Accelerator"))
        current = combo.first.get_attribute("aria-label") or ""
        if f"{label} currently selected" in current:
            return  # 이미 원하는 가속기 — 건드리지 않는다.

        # 2) 열고 옵션 선택
        combo.first.click(timeout=10000)
        page.wait_for_timeout(1000)
        page.get_by_role("option", name=label, exact=True).click(timeout=10000)
        page.wait_for_timeout(2000)

        # 3) 전환 확인 다이얼로그가 뜨면 승인(세션 재시작 안내 등). 없으면 그냥 통과.
        for confirm in ("Turn on GPU", "Confirm", "OK", "Yes"):
            try:
                page.get_by_role("button", name=confirm, exact=False).first.click(timeout=2500)
                break
            except Exception:
                continue

        # 4) 실제로 바뀌었는지 검증 — '조용한 실패'를 여기서 차단.
        page.wait_for_timeout(1500)
        after = combo.first.get_attribute("aria-label") or ""
        if f"{label} currently selected" not in after:
            raise RuntimeError(f"가속기 전환 실패: 기대={label!r}, 현재 aria-label={after!r}")

    @staticmethod
    def _save_and_run_all(page) -> None:
        """'Save Version' → 'Save & Run All (Commit)' 커밋 제출.

        실측 다이얼로그(2026-07-20):
          · role=dialog 없음(Kaggle 모달은 그 role을 안 씀) — 존재 확인은 버튼으로 한다.
          · VERSION TYPE 콤보박스 기본값이 이미 'Save & Run All (Commit)' → 라디오 없음.
            (구코드가 찾던 라디오는 UI 개편으로 사라졌다.)
          · 하단 버튼 = 'Cancel' / 'Save'.
        """
        page.get_by_role("button", name="Save Version", exact=True).click(timeout=15000)
        page.wait_for_timeout(2500)

        # VERSION TYPE 확인 — 기본이 아니면 드롭다운에서 명시적으로 고른다.
        try:
            if page.get_by_text("Save & Run All", exact=False).count() == 0:
                page.get_by_role("combobox").last.click(timeout=5000)
                page.get_by_role("option", name=re.compile("Save & Run All")).first.click(timeout=5000)
                page.wait_for_timeout(1000)
        except Exception:
            pass  # 기본값이면 손댈 필요 없음

        # ⚠️ exact=True 필수. 부분일치면 name="Save"가 툴바 'Save Version'에도 매칭되고,
        #    구코드의 .last가 그 가려진 버튼을 집어 actionability 대기로 타임아웃났다(실측).
        page.get_by_role("button", name="Save", exact=True).click(timeout=15000)

        # 제출되면 모달이 닫힌다 — 안 닫히면 실패를 표면화.
        try:
            page.get_by_role("button", name="Save", exact=True).wait_for(
                state="hidden", timeout=20000)
        except Exception:
            raise RuntimeError("저장 다이얼로그가 닫히지 않음 — 커밋 제출 실패 가능.")


def save_login_state(headful: bool = True, timeout_s: int = 300) -> str:
    """`fcc kaggle-login` 구현 — 브라우저 띄워 사용자가 직접 로그인하면 상태 저장.

    2FA 포함 로그인은 사람이 하고, 그 세션(쿠키)을 파일로 저장해 이후 무인 자동화가 재사용.

    ⚠️ 실측(2026-07-20): Playwright 번들 Chromium으로는 Google이 로그인을 차단한다
      ("브라우저 또는 앱이 안전하지 않을 수 있습니다 … 로그인할 수 없음") — 패스키/PIN을
      다 통과해도 마지막에 막힘. 그래서 **시스템에 설치된 실제 Chrome + 영속 프로필**을 쓰고
      자동화 표식(--enable-automation, navigator.webdriver)을 끈다.

    또한 Enter 입력(input())에 의존하지 않는다 — 비대화형(에이전트/파이프)에서 즉시 EOF가 나
    미로그인 세션이 저장되던 문제가 있었다. 대신 쿠키를 폴링해 실제 로그인 완료를 감지한다.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        raise SystemExit("pip install 'freecloud[kaggle-ui]' && playwright install chromium")
    os.makedirs(os.path.dirname(_state_path()), exist_ok=True)
    os.makedirs(_profile_dir(), exist_ok=True)

    with sync_playwright() as p:
        ctx = None
        last_err = None
        # 실제 Chrome 우선 → Edge → 번들 Chromium(차단 가능성 높음, 최후수단).
        for channel in ("chrome", "msedge", None):
            try:
                ctx = p.chromium.launch_persistent_context(
                    _profile_dir(),
                    headless=False,  # 로그인은 반드시 headful
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
        page.goto("https://www.kaggle.com/account/login", wait_until="domcontentloaded")
        print(f"브라우저에서 Kaggle 로그인을 완료하세요(2FA 포함). 최대 {timeout_s}초 대기하며, "
              "로그인이 감지되면 자동으로 저장하고 닫힙니다.")

        ok = False
        waited = 0
        # ⚠️ 사용자가 로그인 감지 전에 창을 닫으면 Playwright 가 TargetClosedError 를
        #    던진다. 예전엔 그게 트레이스백으로 그대로 터져 "내가 뭘 잘못했나" 싶은
        #    화면만 남았다(결함 F). 안내 한 줄로 바꿔 rc=1 로 끝낸다.
        try:
            while waited < timeout_s:
                page.wait_for_timeout(3000)
                waited += 3
                try:
                    if _cookie_says_anon(ctx.cookies()) is False:
                        ok = True
                        break
                except Exception as e:
                    if _is_browser_closed(e):
                        raise       # 닫힌 건 '일시적 실패'가 아니다 — 폴링해봐야 헛돈다
                    pass  # 페이지 전환 중 일시적 실패는 무시하고 계속 폴링

            ctx.storage_state(path=_state_path())
            ctx.close()
        except Exception as e:
            if _is_browser_closed(e):
                raise SystemExit(BROWSER_CLOSED_MSG)
            raise

    # ⚠️ Windows 콘솔(cp949)은 이모지를 못 찍는다 — 여기서 UnicodeEncodeError로 죽으면
    #    로그인이 성공했는데도 실패처럼 보인다(실측 2026-07-20). CLI 출력은 ASCII로.
    if ok:
        print(f"[OK] 로그인 확인 + 상태 저장됨 -> {_state_path()}")
    else:
        print(f"[FAIL] {timeout_s}초 내 로그인 감지 실패 - 저장은 했지만 미로그인 상태일 수 있음 "
              f"({_state_path()}). 'fcc providers'로 확인하세요.")
    return _state_path()
