"""orchestrator — 페일오버 루프. train_failover.orchestrate 를 provider-무관으로 일반화.

절차:
  라운드 전에 1회: job.needs 안 맞는(UNFIT) provider 는 목록에서 빼고 한 번만 알린다.
    적합한 provider 가 하나도 없으면 라운드 없이 즉시 종료.
  라운드마다: 적합 provider 가 **전부** 쿨다운이면 라운드를 태우지 않고 가장 짧은 쿨다운만큼
    (1회 최대 COOLDOWN_WAIT_CAP_S) 기다렸다 다시 본다.
  provider 순서대로:
    쿨다운 중이면 스킵 → probe() 실패면 쿨다운 걸고 스킵
    run(job)  # 제출→폴링→산출물 회수→정리(고아 방지)
    성공 → 종료
    detached(폴링 상한 초과, 커널은 살아서 checkpoint push 중) → 즉시 종료. 쿨다운·다른 provider 없음.
    실패 → errors.action 에 따라: RETRY(같은 provider 1회) / FAILOVER(쿨다운+다음) / ABORT(중단)

체크포인트가 있으면 어느 provider에서 죽어도 다음이 이어받는다(상태는 원격 저장소에).
"""
from __future__ import annotations

import time
from dataclasses import asdict

from . import registry, state
from .errors import RETRY, FAILOVER, ABORT, DETACH, classify
from .job import Job
from .providers.base import Provider


#: 적합 provider 가 전부 쿨다운일 때 한 번에 기다리는 최대 시간(초).
COOLDOWN_WAIT_CAP_S = 1800


def _order(job: Job) -> list[str]:
    return job.providers or registry.DEFAULT_ORDER


def _preflight(job: Job) -> dict | None:
    """provider 시도 前 필수 시크릿을 검증. 문제 있으면 결과 dict(즉시 중단), 없으면 None.

    데이터셋 업로드·쿼터·원격 부팅을 낭비한 뒤 커널 깊숙이서 죽는 대신, dispatch 前에
    빠르게 실패시키고 사람이 읽을 조치를 준다. HF는 whoami로 실검증."""
    from . import secrets
    from .auth import verify_hf_token, verify_hf_write

    def _fail(detail: str, suffix: str = "") -> dict:
        reason = f"프리플라이트 실패(HF): {detail}{suffix}"
        print(f"[orch] {reason}", flush=True)
        return dict(ok=False, provider=None, reason=reason,
                    attempts=[dict(provider="(preflight)", ok=False,
                                   error_class="AUTH", reason=detail)])

    needs_hf = bool(job.checkpoint_repo) or any(
        s in ("HF_TOKEN", "HUGGINGFACE_TOKEN") for s in (job.secrets or []))
    if not needs_hf:
        return None

    tok = secrets.get("HF_TOKEN") or secrets.get("HUGGINGFACE_TOKEN")
    ok, detail = verify_hf_token(tok)
    if not ok:
        return _fail(detail, " — `fcc login huggingface`로 갱신 후 재실행."
                             " (체크포인트/게이티드 모델에 HF 토큰 필요.)")

    # ⚠️ whoami 통과 ≠ 쓰기 가능. 체크포인트를 쓸 job 이면 실제 쓰기까지 확인한다 —
    #    안 그러면 런이 다 끝나갈 무렵 push 에서 401 이 나고 쿼터를 날린다.
    if job.checkpoint_repo:
        ok, wdetail = verify_hf_write(job.checkpoint_repo, tok)
        if not ok:
            return _fail(wdetail)
        print(f"[orch] 프리플라이트 OK: {detail} / {wdetail}", flush=True)
    return None


def run(job: Job, once: bool = False, max_rounds: int = 3) -> dict:
    """job을 성공할 때까지(또는 max_rounds 소진까지) provider들을 순회.

    반환: {ok, provider, reason, attempts:[...]}  ← CLI/MCP가 파싱.
      detached 면 {ok: False, detached: True, provider: <그 provider>, ...} — 실패가 아니라
      '원격에서 계속 실행 중'. 결과는 checkpoint_repo 에서 회수한다.
    """
    pf = _preflight(job)
    if pf is not None:
        return pf

    attempts: list[dict] = []

    # ── 적합성은 정적(needs vs capabilities) → 라운드 전에 한 번만 거른다 ──────
    # 2026-09-30 실측: needs.gpu=T4 인데 kaggle(P100)이 라운드 2·3 을 차지했다 —
    # 어차피 못 쓰는 provider 가 라운드만 태우고 "3라운드 내 성공 없음"으로 끝났다.
    fit: list[tuple[str, Provider]] = []
    for name in _order(job):
        try:
            prov = registry.get(name)
        except KeyError as e:
            attempts.append(dict(provider=name, ok=False, error_class="UNKNOWN_PROVIDER",
                                 reason=str(e)))
            continue
        fit_ok, why = prov.fits(job)
        if not fit_ok:
            print(f"[orch] 건너뜀: {why}", flush=True)
            attempts.append(dict(provider=name, ok=False, error_class="UNFIT", reason=why))
            continue
        fit.append((name, prov))

    if not fit:
        return dict(ok=False, provider=None,
                    reason="적합한 provider 없음(needs 불충족/알 수 없는 이름) — "
                           "라운드를 돌지 않고 종료. attempts 참고.",
                    attempts=attempts)

    rnd = 0
    while rnd < max_rounds:
        # 적합 provider 가 전부 쿨다운 → 라운드를 태우지 말고 가장 짧은 쿨다운만큼 기다린다.
        cds = [state.cooldown_remaining(n) for n, _ in fit]
        if all(c > 0 for c in cds):
            wait = int(min(min(cds), COOLDOWN_WAIT_CAP_S)) + 1
            print(f"[orch] 적합 provider 전부 쿨다운 → {wait}s 대기 후 재확인 "
                  f"(라운드 소모 없음)", flush=True)
            time.sleep(wait)
            continue
        rnd += 1

        progressed = False
        for name, prov in fit:
            cd = state.cooldown_remaining(name)
            if cd > 0:
                print(f"[orch] {name} 쿨다운 {int(cd)}s 남음 → 스킵", flush=True)
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
            if res.status == "detached" or d.action == DETACH:
                # 커널은 살아서 checkpoint 를 push 중 — 쿨다운 없음, 다른 provider 없음.
                print(f"⏳ [detached] 커널 계속 실행 중 — {res.reason}", flush=True)
                return dict(ok=False, detached=True, provider=name,
                            reason=f"[detached] {res.reason}", attempts=attempts)
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
    """RETRY 분류면 같은 provider로 1회 더. detached 는 재시도하지 않는다(살아 있는 런)."""
    res = prov.run(job)
    if res.ok or res.status == "detached":
        return res
    d = classify(res.error_class + " " + res.reason + " " + res.log_tail)
    if d.action == RETRY:
        print(f"  · {prov.name} 일시적 오류 → 1회 재시도", flush=True)
        res = prov.run(job)
    return res
