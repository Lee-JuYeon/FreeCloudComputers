"""lightning 어댑터 — Lightning AI(lightning-sdk 2026.10.x)의 Job 으로 실행.

Lightning 무료: 15크레딧/월 ≈ 80 GPU-h(interruptible). 페일오버 풀에서 물량이 가장 큰 축.

인증(토큰 방식, 비대화형): env LIGHTNING_USER_ID + LIGHTNING_API_KEY (SDK 가 env 에서 읽는다).
teamspace: env LIGHTNING_TEAMSPACE = "owner/teamspace" 슬러그(예: "lee-juyeon/default-project").
    → `teamspace=<슬러그>` 하나로 넘긴다. `user=`/`org=` 는 SDK 에서 deprecated 이고,
      과거 코드처럼 36자 user **id** 를 `user=` 에 넣으면 owner 해석이 틀어진다.
    슬래시 없는 이름만 주면 SDK 가 owner 를 기본값(로그인 유저)으로 채운다.

실행 경로 두 가지 (왜: Job 은 스냅샷/이미지로 돌아 스튜디오를 켤 필요가 없다 —
스튜디오 start 는 그 자체로 크레딧을 태운다):
  · 기본 = docker 이미지 Job: `image=FREECLOUD_LIGHTNING_IMAGE`
    (기본 "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"), `command=<entrypoint>`.
  · env LIGHTNING_STUDIO 가 있으면 그 **기존** 스튜디오의 스냅샷으로 Job:
    `Studio(name=..., teamspace=<슬러그>, create_ok=False)` — `.start()` 는 절대 부르지 않는다.
    (이 경로에선 Job.run 에 teamspace 를 넘기지 않는다 — SDK 가 슬러그와 studio.teamspace.name
     을 문자열 그대로 비교해 불일치로 터진다. 스튜디오가 teamspace 를 이미 안다.)

머신: FREECLOUD_LIGHTNING_MACHINE(기본 T4; `Machine` 속성명, 예 T4_X_2·L4).
interruptible=True(무료 티어). 선점되면 Lightning 이 max_run_attempts
(FREECLOUD_LIGHTNING_ATTEMPTS, 기본 5)까지 스스로 재실행 — 워크로드는 자기 체크포인트로 재개.
Job 이름은 teamspace 안에서 유일해야 하므로 `fc-<job>-<epoch>` 로 매번 새로 짓는다.

env 주입: `env=job.resolved_env()` + command 앞 export 접두사(이중 안전장치 — 일부 SDK 버전은
image Job 에서 env 를 무시한다).

폴링 상한(job.max_runtime_s)에 닿으면 Job 을 **멈추지 않고** detached 로 돌려준다
(kaggle 과 같은 계약, errors.DETACH) — 30시간짜리 학습을 6시간에 죽이지 않기 위해.

SDK 시그니처(2026.10.1 확인):
    from lightning_sdk import Job, Studio, Machine, Status
    Job.run(name, machine, command=None, studio=None, image=None, teamspace=None, env=None,
            interruptible=False, max_run_attempts=None, ...) -> Job
    job.status -> Status.{Completed, Failed, NotCreated, Pending, Running, Stopped, Stopping}
"""
from __future__ import annotations

import contextlib
import os
import re
import time

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult

DEFAULT_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
HEARTBEAT_S = 600
POLL_S = 30
_QUOTA_RE = re.compile(r"credit|quota|insufficient|billing", re.I)


def _skip_studio_setup():
    """lightning_sdk 의 skip_studio_setup 컨텍스트(없는 버전이면 아무것도 안 함)."""
    try:
        from lightning_sdk.utils.resolve import skip_studio_setup
        return skip_studio_setup()
    except Exception:
        return contextlib.nullcontext()


def _status_word(status) -> str:
    """Status enum / 문자열 / 'Status.Running' 무엇이 와도 소문자 단어 하나로."""
    if status is None:
        return ""
    s = getattr(status, "name", None) or getattr(status, "value", None) or str(status)
    return str(s).rsplit(".", 1)[-1].strip().lower()


class LightningProvider(Provider):
    name = "lightning"
    capabilities = {"gpu": "T4|L4|A10G", "vram_gb": 16, "headless": True,
                    "note": "15크레딧/월 ≈ 80 GPU-h, interruptible."}

    def _sdk(self):
        try:
            import lightning_sdk
            return lightning_sdk
        except ImportError:
            return None

    def probe(self) -> Probe:
        if self._sdk() is None:
            return Probe(False, "lightning-sdk 미설치(pip install lightning-sdk).")
        if not (os.environ.get("LIGHTNING_API_KEY") and os.environ.get("LIGHTNING_USER_ID")):
            return Probe(False, "LIGHTNING_API_KEY/LIGHTNING_USER_ID 미설정.")
        if not os.environ.get("LIGHTNING_TEAMSPACE"):
            return Probe(False, "LIGHTNING_TEAMSPACE 미설정(예: myuser/teamspace).")
        # 슬래시 없는 이름도 받는다 — owner 는 SDK 가 로그인 유저로 채운다.
        return Probe(True, "lightning 인증 OK.")

    def run(self, job: Job) -> RunResult:
        sdk = self._sdk()
        if sdk is None:
            return RunResult(False, "error", "lightning-sdk 미설치.", error_class="JOB_CONFIG")
        try:
            from lightning_sdk import Studio, Job as LJob, Machine
        except Exception as e:
            return RunResult(False, "error", f"lightning-sdk import 실패: {e}",
                             error_class="JOB_CONFIG")

        ts = (os.environ.get("LIGHTNING_TEAMSPACE") or "").strip() or None
        mname = os.environ.get("FREECLOUD_LIGHTNING_MACHINE", "T4").strip()
        machine = getattr(Machine, mname, None)
        if machine is None:
            return RunResult(False, "error",
                             f"FREECLOUD_LIGHTNING_MACHINE={mname!r} 는 lightning_sdk.Machine 에 "
                             f"없다(예: T4, T4_X_2, L4).", error_class="JOB_CONFIG")
        try:
            attempts = int(os.environ.get("FREECLOUD_LIGHTNING_ATTEMPTS", "5"))
        except ValueError:
            return RunResult(False, "error", "FREECLOUD_LIGHTNING_ATTEMPTS 는 정수여야 한다.",
                             error_class="JOB_CONFIG")

        env = job.resolved_env()
        cmd = self.export_prefix(env) + job.entrypoint
        self.warn_unfetched(job, self.name)

        kw = dict(name=f"fc-{job.name}-{int(time.time())}"[:40], machine=machine,
                  command=cmd, teamspace=ts, env=env, interruptible=True,
                  max_run_attempts=attempts)
        try:
            studio_name = (os.environ.get("LIGHTNING_STUDIO") or "").strip()
            if studio_name:
                # 기존 스튜디오의 스냅샷으로만 돈다 — 만들지도, 켜지도 않는다.
                # skip_studio_setup: 스튜디오가 Running 이면 SDK 생성자가 keep-alive 스레드를
                # 띄운다(크레딧 소모) — Job.run 내부와 똑같이 그걸 끈다.
                with _skip_studio_setup():
                    kw["studio"] = Studio(name=studio_name, teamspace=ts, create_ok=False)
                # SDK 2026.10.1 Job.run 은 studio.teamspace.name 과 teamspace 문자열을 그대로
                # 비교한다 → 슬러그("owner/ts")를 같이 주면 불일치 ValueError. 스튜디오가 이미
                # teamspace 를 갖고 있으니 Job.run 에는 넘기지 않는다.
                kw.pop("teamspace")
            else:
                kw["image"] = os.environ.get("FREECLOUD_LIGHTNING_IMAGE", DEFAULT_IMAGE)
            lj = LJob.run(**kw)
        except Exception as e:
            msg = str(e)
            d = classify(msg)
            if _QUOTA_RE.search(msg) or d.error_class == "QUOTA_COOLDOWN":
                return RunResult(False, "quota", f"Lightning 크레딧/쿼터 부족: {msg}",
                                 msg[-800:], "QUOTA_COOLDOWN")
            return RunResult(False, "error", f"Lightning 예외: {msg}", msg[-800:], d.error_class)
        return self._poll(lj, job)

    @staticmethod
    def _logs_tail(lj, n: int = 1500) -> str:
        try:
            fn = getattr(lj, "logs", None)
            return str(fn() if callable(fn) else (fn or ""))[-n:]
        except Exception:
            return ""

    @staticmethod
    def _link(lj) -> str:
        try:
            return str(getattr(lj, "link", "") or "")
        except Exception:
            return ""

    def _poll(self, lj, job: Job) -> RunResult:
        start = time.time()
        deadline = start + job.max_runtime_s
        last_hb = start
        jname = getattr(lj, "name", "") or job.name
        blind = 0                       # status 를 못 읽은 연속 횟수
        word = ""
        while time.time() < deadline:
            e_txt = ""
            try:
                word = _status_word(getattr(lj, "status", None))
            except Exception as e:      # SDK 세션 만료 등
                word, e_txt = "", str(e)
            now = time.time()
            if now - last_hb >= HEARTBEAT_S:
                print(f"[{self.name}] ♥ {int(now - start) // 60}분 경과 · job={jname} "
                      f"· status={word or '?'}", flush=True)
                last_hb = now
            if not word:
                # 20회(≈10분) 연속 상태를 못 읽으면 세션이 죽은 것으로 보고 페일오버.
                blind += 1
                if blind >= 20:
                    return RunResult(False, "error",
                                     "Lightning 상태 조회 불가(10분 연속) — 세션 유실로 판단.",
                                     e_txt[-400:], "SESSION_LOST")
                time.sleep(POLL_S); continue
            blind = 0
            if word == "completed":
                return RunResult(True, "done", "Lightning Job 완료.")
            if word == "failed":
                tail = self._logs_tail(lj)
                if _QUOTA_RE.search(tail):
                    return RunResult(False, "quota", "Lightning 크레딧/쿼터 부족으로 Job 실패.",
                                     tail, "QUOTA_COOLDOWN")
                d = classify(tail + " " + word)
                st = "quota" if d.error_class == "QUOTA_COOLDOWN" else "error"
                return RunResult(False, st, d.advice, tail, d.error_class)
            if word in ("stopped", "stopping"):
                return RunResult(False, "error",
                                 "Lightning Job 중단(선점/수동 정지) — 재시도 시 체크포인트 재개",
                                 self._logs_tail(lj), "INTERRUPTED")
            # pending / running / notcreated / 알 수 없는 값 → 계속 기다린다.
            time.sleep(POLL_S)
        # 폴링 상한 도달: 멈추지 않는다(30시간 학습을 6시간에 죽이지 않기). orchestrator 는
        # DETACHED 를 보고 재시도·페일오버 없이 멈춘다(이중 학습 방지).
        link = self._link(lj)
        return RunResult(False, "detached",
                         f"폴링 상한 {job.max_runtime_s}s 초과 — Lightning Job 은 계속 실행 중"
                         f"(job={jname}{', ' + link if link else ''}). 산출물은 job 이 직접 "
                         f"올리는 곳(checkpoint_repo 등)에서 회수.",
                         f"status={word}", "DETACHED")
