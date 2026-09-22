"""job — 선언형 Job 스펙(YAML 한 장). provider-무관.

핵심: job은 '체크포인트에서 재개 가능'해야 한다. entrypoint는 원격 노드에서
  1) 시작 시 checkpoint repo에서 최신 상태를 pull 하고
  2) N스텝마다 push 한다.
그래야 어느 provider에서 죽어도 다음 provider가 이어받는다. freecloud.checkpoint 참고.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Job:
    name: str
    entrypoint: str                       # 원격에서 실행할 명령/스크립트 (예: "python train.py")
    repo: str = ""                        # 코드 소스(git url). 비면 workdir 업로드 방식.
    # 로컬 코드 루트. **명시했을 때만** 업로드한다 — "" = 업로드 없음(코드는 entrypoint 가
    # 알아서 받는다는 뜻). 기본값이 "." 이던 시절에는 "명시 안 함"과 "cwd 를 올려라"를
    # 구분할 수 없어서, workdir 키가 없는 job 의 커널에까지 '입력 데이터셋 복사' 코드가
    # 들어가 FileNotFoundError 로 죽었다(2026-09-23 라이브 회귀).
    workdir: str = ""
    checkpoint_repo: str = ""             # HF Hub repo id — 재개 상태 저장소(예: "user/myjob-ckpt")
    artifacts: list[str] = field(default_factory=list)  # 회수할 원격 경로들
    needs: dict[str, Any] = field(default_factory=dict) # {gpu, min_vram_gb, ...} — provider 필터
    env: dict[str, str] = field(default_factory=dict)   # 원격에 주입할 비밀 아닌 env
    secrets: list[str] = field(default_factory=list)    # 원격에 주입할 시크릿 이름(값은 저장소/env에서)
    max_runtime_s: int = 5400             # 단일 노드 최대 실행(무료 세션 한도 안쪽)
    providers: list[str] = field(default_factory=list)  # 시도 순서. 비면 registry 기본순.
    success_glob: str = ""                # 성공 판정용 산출물 패턴(mtime 갱신 = 성공)

    @staticmethod
    def load(path: str) -> "Job":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        known = {f_.name for f_ in Job.__dataclass_fields__.values()}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"job.yaml 알 수 없는 키: {sorted(unknown)}")
        job = Job(**d)
        if job.workdir:        # 빈 문자열을 abspath 하면 cwd 가 되어 '명시함'으로 둔갑한다
            job.workdir = os.path.abspath(os.path.expanduser(job.workdir))
        return job

    def resolved_env(self) -> dict[str, str]:
        """원격에 주입할 env = 비밀아닌 env + 필요한 시크릿(HF_TOKEN 등) 자동 주입.

        시크릿은 저장소/환경변수에서 값을 끌어와 원격 노드로 실어보낸다(secrets.collect_for_job).
        그래서 job.yaml에 토큰을 적지 않아도 체크포인트 push/pull이 원격에서 동작한다.
        """
        e = dict(self.env)
        if self.checkpoint_repo:
            e.setdefault("FREECLOUD_CKPT_REPO", self.checkpoint_repo)
        from . import secrets  # 지연 import(순환 회피)
        e.update(secrets.collect_for_job(self))
        return e
