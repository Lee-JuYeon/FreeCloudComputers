"""cli — `freecloud` 진입점.

  freecloud run job.yaml [--once] [--providers a,b] [--dry-run]
  freecloud providers            # 등록된 provider + probe 상태
  freecloud status               # 쿨다운 스냅샷
  freecloud clouds               # 지원/후보 무료 클라우드 카탈로그
"""
from __future__ import annotations

import argparse
import json
import sys

from . import registry, state
from .job import Job
from . import orchestrator


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
    return 0 if result["ok"] else 1


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
    ap = argparse.ArgumentParser(prog="freecloud",
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
    r.set_defaults(fn=_cmd_run)

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
