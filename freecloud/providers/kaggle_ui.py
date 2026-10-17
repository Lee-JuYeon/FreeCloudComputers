"""kaggle-ui 어댑터 — Playwright로 T4×2를 완전 무인 설정.

배경: Kaggle API/CLI는 단일 GPU(P100)만 노출. T4×2는 웹 UI 전용이라, 사람이 매번
Settings→Accelerator→GPU T4 x2→Save를 눌러야 한다. 이 어댑터가 그 클릭을 자동화한다.

흐름:
  1) KaggleProvider 헬퍼로 코드 push(커널 등록). ※push는 그 자체로 run 을 하나 시작한다 —
     가속기는 커널에 **저장된 값**(이전 UI 선택이 남아 있으면 T4×2)을 쓴다.
  2) Playwright: 저장된 로그인 상태로 에디터 열기 → 현재 가속기 읽기.
     · 이미 'GPU T4 x2' → UI 커밋 생략(push 가 이미 T4×2 로 돌고 있다 — 또 커밋하면 동일
       세션 2개가 GPU 쿼터를 2배로 쓰고 ckpt/ 를 두고 경합한다, 실측 2026-09-30).
     · 아니면 'GPU T4 x2' 선택 → Save & Run All(push 가 띄운 이전 가속기 세션은 멈출 API 가
       없어 그대로 둔다 — 쓸 GPU 가 없으면 스스로 abort).
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

#: 원하는 가속기 옵션 라벨(실측 DOM 의 role=option name).
T4X2_LABEL = "GPU T4 x2"

_ACCEL_ARIA_RE = re.compile(r"Select Accelerator\.\s*(.*?)\s*currently selected", re.S)

#: 쿼터 소진 감지(2026-10-16 수리). 에디터 페이지 텍스트에서 읽는다.
#: 실측: 쿼터 소진 상태의 에디터에 'Quota: 31:38 / 30 hrs' + "You're almost out of quota".
#: 이때 push 는 rc=0 이어도 새 세션을 못 띄우고, '이미 T4x2 → UI 커밋 생략' 경로가
#: 옛 COMPLETE 를 성공으로 읽어 거짓 성공을 보고했다(10-08·10-16 두 번 재발).
_QUOTA_RE = re.compile(r"Quota:\s*(\d+):(\d+)\s*/\s*30\s*hrs", re.I)
_QUOTA_WARN = "almost out of quota"
#: 주간 GPU 쿼터 상한(시간). 이 이상 썼으면 소진으로 본다.
QUOTA_LIMIT_H = 30.0

#: 실행 중 세션을 멈추는 Stop 컨트롤 후보(2026-10-17 추가).
#: ⚠️ 실측 DOM(2026-10-17 dogfood, benedictarthur/freecloud-laibrio-gemma4 RUNNING 중):
#:    에디터에 '레이블된 Stop 버튼'은 **없다**. 노출된 실행 관련 컨트롤은
#:      · button text='fast_forward Run All' (aria='Run All')
#:      · button text='circle Draft Session off (run a cell to start)'  ← 인터랙티브 세션 토글(현재 off)
#:      · button text='update 1  View Active Events' (aria-controls="active-events-panel")
#:    `kaggle kernels push` 가 띄우는 건 '커밋된 배치 런'이라 Draft Session(off)이 아니라
#:    'View Active Events'(배지 1) 뒤 패널에 뜬다. 그 패널 안에 Stop/Cancel 이 있을 것으로
#:    보이나, (a) Google 쿠키 동의 오버레이(div.sc-gtcBCx "Kaggle uses cookies from Google…")가
#:    pointer 이벤트를 가로채 View Active Events 클릭을 막고(닫기 버튼 미발견),
#:    (b) 패널이 포털 렌더라 headless 에서 열리지 않아 **내부 Stop 셀렉터를 실측 확정 못 함**.
#:    → 아래는 '레이블된 Stop 버튼' 전제의 best-effort(인터랙티브 세션엔 유효). 커밋 배치 런의
#:       확정 경로는 headful + 쿠키수락 세션에서 View Active Events 패널을 열어 재확인 필요.
#:    못 찾으면 안전하게 (False, 사유) — 거짓 성공·무한 탐색 금지.
_STOP_BUTTON_NAMES = (
    "Stop Session",
    "Stop session",
    "Stop running session",
    "Cancel Run",
    "Stop",
)

#: 중지/유휴로 바뀌었음을 알리는 페이지 텍스트 마커(best-effort, 소문자 비교).
_STOPPED_MARKERS = (
    "session stopped",
    "no active session",
    "session is not running",
    "stopped",
    "idle",
)


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
            # 쿼터 소진은 UI 자동화 실패가 아니라 provider 쿼터 문제 →
            # QUOTA_COOLDOWN(status="quota")로 다음 provider 로 페일오버시킨다.
            if why.startswith("QUOTA_COOLDOWN"):
                return RunResult(False, "quota", f"GPU 쿼터 소진: {why}",
                                 error_class="QUOTA_COOLDOWN")
            return RunResult(False, "needs_setup", f"UI 자동화 실패: {why}",
                             error_class="UI_AUTOMATION")
        ui_commit = "skipped" if why.startswith("skipped") else "committed"
        print(f"[kaggle-ui] ui_commit={ui_commit} ({why})", flush=True)
        # 3) 폴링 + GPU 검증(부모 헬퍼 재사용)
        res = self._poll_and_fetch(job, work)
        # 어느 경로(생략/커밋)로 돌았는지 사람이 결과만 보고 알 수 있게 남긴다.
        res.reason = f"{res.reason} [ui_commit: {ui_commit}]"
        return res

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
                return self._commit_on_page(page)
            except PWTimeout as e:
                return False, f"셀렉터 타임아웃(UI 변경 가능): {e}"
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"
            finally:
                ctx.close()
                browser.close()

    def _commit_on_page(self, page, label: str = T4X2_LABEL) -> tuple[bool, str]:
        """에디터가 렌더된 page 에서 '필요할 때만' 가속기 전환 + Save & Run All.

        `kaggle kernels push` 가 이미 커널의 **저장된 가속기**로 run 을 하나 띄웠다. 저장값이
        이미 label 이면 여기서 또 커밋하면 동일 세션이 2개가 된다(실측 2026-09-30: 매 체크포인트
        step 이 HF 에 두 번 push, GPU 쿼터 2배·ckpt/ 경합). 그래서 현재값을 먼저 읽는다.
        현재값을 못 읽으면(DOM 변경) 예전 동작(선택+커밋)으로 — 읽기 실패로 job 을 죽이지 않는다.
        선택/커밋 단계의 예외는 그대로 위로 던진다(_set_t4x2_and_commit 이 실패로 변환).
        """
        # ⚠️ 쿼터 소진 감지 먼저. 쿼터가 0 이면 push(rc=0) 가 새 세션을 못 띄우고,
        #    아래 'already T4x2 → skip' 경로가 옛 COMPLETE 를 성공으로 읽는다(거짓 성공).
        #    소진이면 커밋/진행하지 말고 QUOTA_COOLDOWN 으로 실패시켜 다음 provider 로 넘긴다.
        exhausted, qdetail = self._quota_exhausted(self._page_text(page))
        if exhausted:
            print(f"[kaggle-ui] GPU 쿼터 소진 감지 — UI 커밋/진행 중단(거짓 성공 방지): "
                  f"{qdetail}", flush=True)
            return False, f"QUOTA_COOLDOWN: {qdetail}"

        try:
            current = self._read_accelerator(page)
            read_err = "" if current is not None else "aria-label 형식 불일치"
        except Exception as e:  # 콤보박스 못 찾음 등
            current, read_err = None, f"{type(e).__name__}: {e}"

        if current == label:
            print("[kaggle-ui] 가속기 이미 T4 x2 — API push 가 이미 실행 중이므로 UI 커밋 생략"
                  "(중복 세션 방지)", flush=True)
            return True, "skipped: already T4x2"

        if current is None:
            print(f"[kaggle-ui] 현재 가속기 읽기 실패({read_err}) — 예전 동작(선택+커밋)으로 진행."
                  " push 가 띄운 세션과 중복될 수 있다.", flush=True)
        else:
            print(f"[kaggle-ui] ⚠️ API push 가 이전 가속기({current})로 세션을 이미 시작했다 — "
                  "멈출 API 가 없어 그대로 둔다(쓸 GPU 가 없으면 스스로 abort). "
                  f"{label} 로 전환 후 커밋.", flush=True)

        # (a) 가속기 선택 — 실패는 예외로 표면화.
        self._select_accelerator(page, label)
        # (b) Save Version → Save & Run All (Commit) → Save
        self._save_and_run_all(page)
        page.wait_for_timeout(3000)  # 커밋 제출 반영 대기
        prev = current if current is not None else "unknown"
        return True, f"committed (prev={prev})"

    @staticmethod
    def _read_accelerator(page) -> str | None:
        """현재 저장된 가속기 라벨. aria-label 형식이 다르면 None, 콤보박스가 없으면 예외.

        aria-label = "Select Accelerator. <현재값> currently selected." (_select_accelerator 참조)
        """
        try:
            page.get_by_role("button", name="Expand Session options").click(timeout=8000)
            page.wait_for_timeout(1500)
        except Exception:
            pass  # 이미 펼쳐져 있음
        combo = page.get_by_role("combobox", name=re.compile("Select Accelerator"))
        aria = combo.first.get_attribute("aria-label", timeout=10000) or ""
        m = _ACCEL_ARIA_RE.search(aria)
        return m.group(1).strip() if m and m.group(1).strip() else None

    @staticmethod
    def _page_text(page) -> str:
        """에디터 페이지 본문 텍스트. 쿼터 배너를 읽기 위한 것.

        ⚠️ DOM 의존: body inner_text 로 통째로 읽는다(쿼터 배너의 정확한 셀렉터는
           실페이지 없이 확정 못 함 — 실측 검증 필요). 읽기 실패는 '' 로 삼켜
           쿼터 미감지(=기존 동작)로 떨어뜨린다. 테스트는 이 메서드를 스텁한다.
        """
        try:
            return page.inner_text("body", timeout=5000) or ""
        except Exception:
            return ""

    @staticmethod
    def _parse_quota_hours(text: str) -> float | None:
        """페이지 텍스트에서 'Quota: HH:MM / 30 hrs' → 사용시간(h). 없으면 None."""
        m = _QUOTA_RE.search(text or "")
        if not m:
            return None
        return int(m.group(1)) + int(m.group(2)) / 60.0

    @classmethod
    def _quota_exhausted(cls, text: str) -> tuple[bool, str]:
        """쿼터 소진? → (exhausted, detail).

        처방(Fable): 사용시간 ≥30h 이거나 'almost out of quota' 문구가 있으면 소진.
        """
        hrs = cls._parse_quota_hours(text)
        if hrs is not None and hrs >= QUOTA_LIMIT_H:
            return True, f"GPU 쿼터 소진({hrs:.2f}h ≥ {QUOTA_LIMIT_H:.0f}h)"
        if _QUOTA_WARN in (text or "").lower():
            return True, "GPU 쿼터 소진 경고('almost out of quota')"
        return False, ""

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


def _body_text(page) -> str:
    """에디터 본문 텍스트(중지 확인용). 읽기 실패는 '' 로 삼킨다."""
    try:
        return page.inner_text("body", timeout=5000) or ""
    except Exception:
        return ""


def _confirm_stopped(page, clicked_name: str) -> tuple[bool, str]:
    """Stop 클릭 후 실제로 중지됐는지 확인 → (ok, detail).

    실페이지 DOM 을 확신할 수 없으므로 '거짓 성공'을 피하도록 보수적으로 판정한다.
    두 신호 중 하나라도 있어야 성공으로 본다:
      (1) 방금 누른 Stop 버튼이 사라짐(실행 중이 아니면 Stop 컨트롤이 없다).
      (2) 페이지 텍스트에 중지/유휴 마커.
    둘 다 없으면 '눌렀지만 확인 못 함'으로 실패 처리(수동 확인 유도).
    """
    try:
        if page.get_by_role("button", name=clicked_name, exact=False).count() == 0:
            return True, f"stopped (Stop 버튼 사라짐, via '{clicked_name}')"
    except Exception:
        pass
    text = _body_text(page).lower()
    for m in _STOPPED_MARKERS:
        if m in text:
            return True, f"stopped (마커 '{m}', via '{clicked_name}')"
    return False, "Stop 클릭 후에도 중지/유휴 신호를 못 읽음"


def _stop_on_page(page) -> tuple[bool, str]:
    """에디터가 렌더된 page 에서 실행 중 세션의 Stop 컨트롤을 눌러 중지 → (ok, detail).

    흐름:
      · (있으면) 세션 패널을 펼친다 — Stop 컨트롤이 접힌 패널 안에 있을 수 있다.
      · _STOP_BUTTON_NAMES 를 순서대로 찾아, 존재하는 첫 버튼을 클릭.
      · 확인 다이얼로그가 뜨면 승인(best-effort).
      · _confirm_stopped 로 실제 중지됐는지 검증.
    Stop 컨트롤을 하나도 못 찾으면 (False, 사유) — 안전 degrade(무한 탐색·거짓 성공 금지).
    """
    # 세션 옵션 패널이 접혀 있으면 펼친다(이미 펼쳐졌거나 버튼이 없으면 무시).
    try:
        page.get_by_role("button", name="Expand Session options").click(timeout=5000)
        page.wait_for_timeout(1000)
    except Exception:
        pass

    tried: list[str] = []
    for name in _STOP_BUTTON_NAMES:
        loc = page.get_by_role("button", name=name, exact=False)
        try:
            present = loc.count() > 0
        except Exception:
            present = False
        if not present:
            tried.append(name)
            continue
        try:
            loc.first.click(timeout=5000)
        except Exception as e:
            tried.append(f"{name}(click:{type(e).__name__})")
            continue
        page.wait_for_timeout(2000)
        # 확인 다이얼로그(세션 중지 안내 등)가 있으면 승인. 없으면 조용히 통과.
        for confirm in ("Confirm", "OK", "Yes", "Stop Session", "Stop"):
            try:
                page.get_by_role("button", name=confirm, exact=False).first.click(timeout=1500)
                break
            except Exception:
                continue
        page.wait_for_timeout(1500)
        return _confirm_stopped(page, name)

    return False, ("Stop 컨트롤 못 찾음(실행 중이 아니거나 UI 변경) — 시도: "
                   + ", ".join(tried))


def stop_running_session(user: str, slug: str,
                         state_path: str | None = None) -> tuple[bool, str]:
    """실행 중 Kaggle 커널(에디터 세션)을 UI Stop 버튼으로 중지 → (ok, detail).

    kaggle CLI/SDK 에 커널 중지 공식 경로가 없어, 저장된 Playwright 로그인 세션
    (~/.freecloud/kaggle_state.json)으로 에디터를 열고 Stop 컨트롤을 누른다.
    토큰(oauth)이 아니라 kaggle-login 세션을 쓴다 — _set_t4x2_and_commit 과 동일 방식.
    실패 시 (False, 사유) — 안전 degrade(호출부가 '수동 Stop 필요'로 안내).
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
    except ImportError:
        return False, "playwright 미설치(pip install 'freecloud[kaggle-ui]')."
    path = state_path or _state_path()
    ok, detail = session_is_logged_in(path)
    if not ok:
        return False, f"kaggle-login 선행 필요: {detail}"
    if not user or not slug:
        return False, f"user/slug 비어 있음(user={user!r}, slug={slug!r})."
    url = f"https://www.kaggle.com/code/{user}/{slug}/edit"
    headful = os.environ.get("FREECLOUD_KAGGLE_HEADFUL") == "1"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headful)
        ctx = browser.new_context(storage_state=path)
        page = ctx.new_page()
        try:
            # networkidle 금지(에디터는 상시 소켓/폴링). _set_t4x2_and_commit 과 동일.
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_selector("text=Save Version", timeout=60000)
            return _stop_on_page(page)
        except PWTimeout as e:
            return False, f"셀렉터 타임아웃(UI 변경 가능): {e}"
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        finally:
            ctx.close()
            browser.close()


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
