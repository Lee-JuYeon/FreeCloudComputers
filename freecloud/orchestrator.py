"""orchestrator — 페일오버 루프. train_failover.orchestrate 를 provider-무관으로 일반화.

절차:
  provider 순서대로:
    쿨다운 중이면 스킵 → job.needs 안 맞으면 스킵 → probe() 실패면 쿨다운 걸고 스킵
    run(job)  # 제출→폴링→산출물 회수→정리(고아 방지)
    성공 → 종료
    실패 → errors.action 에 따라: RETRY(같은 provider 1회) / FAILOVER(쿨다운+다음) / ABORT(중단)

체크포인트가 있으면 어느 provider에서 죽어도 다음이 이어받는다(상태는 원격 저장소에).
"""
from __future__ import annotations

import time
from dataclasses import asdict

from . import registry, state
from .errors import RETRY, FAILOVER, ABORT, classify
from .job import Job
from .providers.base import Provider


def _order(job: Job) -> list[str]:
    return job.providers or registry.DEFAULT_ORDER


def _preflight(job: Job) -> dict | None:
    """provider 시도 前 필수 시크릿을 검증. 문제 있으면 결과 dict(즉시 중단), 없으면 None.

    데이터셋 업로드·쿼터·원격 부팅을 낭비한 뒤 커널 깊숙이서 죽는 대신, dispatch 前에
    빠르게 실패시키고 사람이 읽을 조치를 준다. HF는 whoami로 실검증."""
    from . import secrets
    from .auth import verify_hf_token
    needs_hf = bool(job.checkpoint_repo) or any(
        s in ("HF_TOKEN", "HUGGINGFACE_TOKEN") for s in (job.secrets or []))
    if needs_hf:
        tok = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_TOKEN")
        ok, detail = verify_hf_token(tok)
        if not ok:
            reason = (f"프리플라이트 실패(HF): {detail} — `freecloud login huggingface`로 "
                      f"갱신 후 재실행. (체크포인트/게이티드 모델에 HF 토큰 필요.)")
            print(f"[orch] {reason}", flush=True)
            return dict(ok=False, provider=None, reason=reason,
                        attempts=[dict(provider="(preflight)", ok=False,
                                       error_class="AUTH", reason=detail)])
    return None


def run(job: Job, once: bool = False, max_rounds: int = 3) -> dict:
    """job을 성공할 때까지(또는 max_rounds 소진까지) provider들을 순회.

    반환: {ok, provider, reason, attempts:[...]}  ← CLI/MCP가 파싱.
    """
    pf = _preflight(job)
    if pf is not None:
        return pf

    attempts: list[dict] = []
    names = _order(job)

    for rnd in range(1, max_rounds + 1):
        progressed = False
        for name in names:
            try:
                prov = registry.get(name)
            except KeyError as e:
                attempts.append(dict(provider=name, ok=False, error_class="UNKNOWN_PROVIDER",
                                     reason=str(e)))
                continue

            cd = state.cooldown_remaining(name)
            if cd > 0:
                print(f"[orch] {name} 쿨다운 {int(cd)}s 남음 → 스킵", flush=True)
                continue

            fit_ok, why = prov.fits(job)
            if not fit_ok:
                attempts.append(dict(provider=name, ok=False, error_class="UNFIT", reason=why))
                continue

            pr = prov.probe()
            if not pr.available:
                secs = pr.cooldown_s or 1800
                state.set_cooldown(name, secs, pr.reason)
                attempts.append(dict(provider=name, ok=False, error_class="UNAVAILABLE",
                                     reason=pr.reason))
                print(f"[orch] {name} 사용불가: {pr.reason} → 쿨다운 {secs}s", flush=True)
                continue

            print(f"\n===== [round {rnd}] provider 시도: {name} (job={job.name}) =====", flush=True)
            res = _run_with_retry(prov, job)
            rec = dict(provider=name, **asdict(res))
            attempts.append(rec)
            print(f"  → ok={res.ok} {res.error_class} {res.reason}", flush=True)

            if res.ok:
                state.clear_cooldown(name)
                return dict(ok=True, provider=name, reason=res.reason, attempts=attempts)

            # 실패 분류 → 다음 행동
            d = classify((res.error_class + " " + res.reason + " " + res.log_tail))
            if d.action == ABORT:
                return dict(ok=False, provider=None,
                            reason=f"[{name}] job 결함으로 중단: {d.advice}", attempts=attempts)
            if res.status in ("quota", "timeout") or d.action == FAILOVER:
                state.set_cooldown(name, d.cooldown_s or 1800, res.reason)
                progressed = True
            if once:
                return dict(ok=False, provider=None, reason="once 모드 — 첫 실패로 종료.",
                            attempts=attempts)

        if not progressed:
            # 이번 라운드에 아무도 진전 없음(전부 쿨다운/불가) → 잠깐 쉬고 재시도
            wait = 60
            print(f"[orch] 라운드 {rnd}: 가용 provider 없음 → {wait}s 후 재시도", flush=True)
            time.sleep(wait)

    return dict(ok=False, provider=None,
                reason=f"{max_rounds}라운드 내 성공 provider 없음 — attempts 참고.",
                attempts=attempts)


def _run_with_retry(prov: Provider, job: Job) -> "":
    """RETRY 분류면 같은 provider로 1회 더."""
    res = prov.run(job)
    if res.ok:
        return res
    d = classify(res.error_class + " " + res.reason + " " + res.log_tail)
    if d.action == RETRY:
        print(f"  · {prov.name} 일시적 오류 → 1회 재시도", flush=True)
        res = prov.run(job)
    return res
