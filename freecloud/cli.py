"""cli — `freecloud` 진입점.

  freecloud run job.yaml [--once] [--providers a,b] [--dry-run] [--quiet]
  freecloud logs job.yaml        # 커널 상태 + diag/로그 꼬리 + 산출물 회수
  freecloud providers            # 등록된 provider + probe 상태
  freecloud status               # 쿨다운 스냅샷
  freecloud clouds               # 지원/후보 무료 클라우드 카탈로그
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from . import registry, state
from .job import Job
from . import orchestrator


#: 실패 로그를 얼마나 보여줄지. 40줄이면 대개 Traceback 전체가 들어온다.
LOG_TAIL_LINES = 40


def print_attempt_details(result: dict, lines: int = LOG_TAIL_LINES) -> None:
    """실패한 attempt 마다 provider·status·error_class·reason + 로그 꼬리를 찍는다.

    결함 C(docs/IMPROVEMENTS_2026-09-23.md §0): 예전 CLI 는
    `→ ok=False UNKNOWN 미분류 오류 — …` 한 줄만 찍었고, 실제 원인(커널 Traceback)이
    담긴 `RunResult.log_tail` 은 반환 dict 안에서 잠자고 있었다. 사람이 원인을 보려면
    `kaggle kernels output` 을 손으로 받아야 했다 — 그건 도구가 할 일이다.
    """
    for a in result.get("attempts") or []:
        if a.get("ok"):
            continue
        print(f"\n--- [{a.get('provider')}] status={a.get('status', '?')} "
              f"error_class={a.get('error_class') or '-'} ---")
        print(f"  reason: {a.get('reason', '')}")
        if a.get("artifacts_dir"):
            print(f"  artifacts: {a['artifacts_dir']}")
        tail = (a.get("log_tail") or "").rstrip().splitlines()
        for ln in tail[-lines:]:
            print(f"  | {ln}")


def _cmd_run(args) -> int:
    job = Job.load(args.job)
    if args.providers:
        job.providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    order = job.providers or registry.DEFAULT_ORDER
    if args.dry_run:
        print(json.dumps({"job": job.name, "providers": order,
                          "checkpoint_repo": job.checkpoint_repo,
                          "max_runtime_s": job.max_runtime_s}, ensure_ascii=False, indent=2))
        return 0
    if not args.no_interactive:
        from . import auth
        auth.ensure_for_job(job, order, interactive=True)  # cloudflare 스타일 온디맨드 로그인
    result = orchestrator.run(job, once=args.once, max_rounds=args.rounds)
    print("\n===== RESULT =====")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.quiet:
        print_attempt_details(result)
    for a in result.get("attempts") or []:
        if a.get("ok") and a.get("artifacts_dir"):
            print(f"\n산출물: {a['artifacts_dir']}")
    return 0 if result["ok"] else 1


def _job_for_logs(target: str) -> Job:
    """`fcc logs <job.yaml|name>` 인자 해석.

    파일이면 그대로 로드하고, 아니면 이름만 받은 것으로 본다(커널 slug 계산에는
    job.name 하나면 충분하다 — 실행을 다시 하지 않으므로 entrypoint 는 비워둔다).
    """
    if os.path.isfile(target):
        return Job.load(target)
    return Job(name=target, entrypoint="")


def _cmd_logs(args) -> int:
    """Kaggle 커널의 상태 + 산출물(diag.txt 포함)을 받아 로그 꼬리를 보여준다(결함 C).

    run 이 이미 끝난 뒤에도, 혹은 다른 창에서 도는 중에도 원인을 볼 수 있어야 한다.
    """
    from .providers.kaggle import KaggleProvider, _UTF8

    job = _job_for_logs(args.job)
    prov = KaggleProvider()
    if not prov._user():
        print("Kaggle 인증 없음 — `kaggle auth login` 후 다시 시도하세요.")
        return 1
    kid = prov.kernel_id(job)
    print(f"kernel: {kid}")
    _, st = prov._sh(["kaggle", "kernels", "status", kid], 60, _UTF8)
    print(f"status: {(st or '').strip()}")

    adir, diag = prov.fetch_outputs(job, kid)
    print(f"artifacts: {adir}")
    files = sorted(os.path.relpath(p, adir)
                   for p in glob.glob(os.path.join(adir, "**", "*"), recursive=True)
                   if os.path.isfile(p))
    print(f"files({len(files)}): {', '.join(files[:20]) or '(없음)'}")
    if diag:
        print(f"\n--- diag.txt (마지막 {args.lines}줄) ---")
        for ln in diag.rstrip().splitlines()[-args.lines:]:
            print(f"  | {ln}")
    else:
        print("\n(diag.txt 없음 — 커널이 아직 안 돌았거나 부팅 전에 죽었다.)")
    return 0


def _cmd_providers(args) -> int:
    for name in registry.available_names():
        prov = registry.get(name)
        pr = prov.probe()
        mark = "OK " if pr.available else "-- "
        print(f"{mark}{name:10} {prov.capabilities.get('gpu','?'):12} {pr.reason}")
    return 0


def _cmd_status(args) -> int:
    snap = state.snapshot()
    if not snap:
        print("쿨다운 없음 — 모든 provider 시도 가능.")
    else:
        print(json.dumps(snap, ensure_ascii=False, indent=2))
    return 0


def _cmd_clouds(args) -> int:
    from .catalog import CATALOG, render
    print(render(CATALOG))
    return 0


def _cmd_kaggle_login(args) -> int:
    """Kaggle 로그인 상태 저장(kaggle-ui/T4x2 자동화 선행 1회)."""
    from .providers.kaggle_ui import save_login_state
    save_login_state()
    return 0


def _cmd_login(args) -> int:
    """freecloud login <provider> — 사이트별 최초 1회 로그인/토큰 등록."""
    from . import auth
    auth.login(args.provider)
    return 0


def _cmd_auth(args) -> int:
    """freecloud auth — 모든 인증 상태(계정/토큰/세션) 한눈에."""
    from . import auth, secrets
    for s in auth.status_all():
        mark = "OK " if s.ok else "-- "
        print(f"{mark}{s.provider:12} {s.detail}")
    stored = secrets.names()
    if stored:
        print("\n저장된 시크릿(이름만):", ", ".join(stored))
    return 0


def main(argv=None) -> int:
    # 구버전(cwd 상대)에 남은 로그인·시크릿·쿨다운을 사용자 전역으로 1회 승격.
    # 안 하면 예전 디렉토리에서 만든 로그인이 고아가 된다. 옮길 게 없으면 조용히 지나간다.
    try:
        from . import paths
        paths.migrate_legacy()
    except Exception:
        pass  # 마이그레이션 실패가 CLI 자체를 막지는 않게 한다.

    ap = argparse.ArgumentParser(prog="fcc",
                                 description="무료(비중국) 클라우드 GPU 페일오버 오케스트레이터")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="job.yaml 실행(페일오버)")
    r.add_argument("job")
    r.add_argument("--once", action="store_true", help="첫 실패로 종료(페일오버 안 함)")
    r.add_argument("--providers", help="시도 순서 덮어쓰기(쉼표)")
    r.add_argument("--rounds", type=int, default=3, help="전체 순회 최대 라운드")
    r.add_argument("--dry-run", action="store_true", help="계획만 출력")
    r.add_argument("--no-interactive", action="store_true",
                   help="온디맨드 로그인 프롬프트 끔(CI/무인)")
    r.add_argument("--quiet", action="store_true",
                   help="실패 상세(error_class/reason/로그 꼬리) 출력 끔")
    r.set_defaults(fn=_cmd_run)

    lo = sub.add_parser("logs", help="Kaggle 커널 상태 + 산출물/로그 꼬리")
    lo.add_argument("job", help="job.yaml 경로 또는 job 이름")
    lo.add_argument("--lines", type=int, default=LOG_TAIL_LINES,
                    help=f"보여줄 로그 줄 수(기본 {LOG_TAIL_LINES})")
    lo.set_defaults(fn=_cmd_logs)

    sub.add_parser("providers", help="provider probe 상태").set_defaults(fn=_cmd_providers)
    sub.add_parser("status", help="쿨다운 스냅샷").set_defaults(fn=_cmd_status)
    sub.add_parser("clouds", help="무료 클라우드 카탈로그").set_defaults(fn=_cmd_clouds)
    sub.add_parser("kaggle-login", help="Kaggle 로그인 저장(kaggle-ui/T4x2용, 1회)"
                   ).set_defaults(fn=_cmd_kaggle_login)

    lg = sub.add_parser("login", help="provider 로그인/토큰 등록(1회)")
    lg.add_argument("provider", help="huggingface|kaggle|colab|modal|lightning|saturn|kaggle-ui")
    lg.set_defaults(fn=_cmd_login)

    sub.add_parser("auth", help="인증 상태 한눈에").set_defaults(fn=_cmd_auth)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
