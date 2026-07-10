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
    workdir: str = "."                    # 로컬 코드 루트(repo 없을 때 업로드 대상)
    checkpoint_repo: str = ""             # HF Hub repo id — 재개 상태 저장소(예: "user/myjob-ckpt")
    artifacts: list[str] = field(default_factory=list)  # 회수할 원격 경로들
    needs: dict[str, Any] = field(default_factory=dict) # {gpu, min_vram_gb, ...} — provider 필터
    env: dict[str, str] = field(default_factory=dict)   # 원격에 주입할 env
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
        job.workdir = os.path.abspath(os.path.expanduser(job.workdir))
        return job

    def resolved_env(self) -> dict[str, str]:
        """원격에 주입할 env. checkpoint 재개에 필요한 것들을 자동 포함."""
        e = dict(self.env)
        if self.checkpoint_repo:
            e.setdefault("FREECLOUD_CKPT_REPO", self.checkpoint_repo)
        return e
