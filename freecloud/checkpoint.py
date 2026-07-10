"""checkpoint — 재개의 린치핀. 공유 스토리지에 상태를 저장/복원.

기본 백엔드 = Hugging Face Hub(git-lfs, 무료, 비공개 가능, rate-limit 관대).
이 모듈은 두 곳에서 쓰인다:
  · 원격 노드의 entrypoint 안 — 시작 시 pull(), N스텝마다 push().
  · 오케스트레이터 로컬 — 제출 전 latest 확인, 완료 후 산출물 회수 보조.

의존: huggingface_hub (pip install 'freecloud[hub]'). 토큰=env HF_TOKEN.

원격 entrypoint에서 쓰는 최소 예:
    from freecloud.checkpoint import Checkpoint
    ck = Checkpoint.from_env()          # FREECLOUD_CKPT_REPO + HF_TOKEN
    ck.pull("./ckpt")                   # 이전 노드가 남긴 상태 복원(없으면 no-op)
    ...training...
    ck.push("./ckpt", step=step)        # 매 N스텝
"""
from __future__ import annotations

import os
from typing import Optional


class Checkpoint:
    def __init__(self, repo_id: str, token: Optional[str] = None, private: bool = True):
        self.repo_id = repo_id
        self.token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        self.private = private

    @classmethod
    def from_env(cls) -> Optional["Checkpoint"]:
        repo = os.environ.get("FREECLOUD_CKPT_REPO")
        if not repo:
            return None
        return cls(repo)

    def _api(self):
        try:
            from huggingface_hub import HfApi
        except ImportError as e:
            raise RuntimeError("pip install 'freecloud[hub]' 필요(huggingface_hub).") from e
        return HfApi(token=self.token)

    def ensure_repo(self) -> None:
        api = self._api()
        api.create_repo(self.repo_id, repo_type="model", private=self.private, exist_ok=True)

    def pull(self, dest: str) -> bool:
        """최신 체크포인트를 dest로 복원. 저장소가 비었으면 False(=처음 실행)."""
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import RepositoryNotFoundError
        os.makedirs(dest, exist_ok=True)
        try:
            snapshot_download(self.repo_id, repo_type="model", local_dir=dest,
                              token=self.token)
            has = any(os.scandir(dest))
            print(f"[checkpoint] pull {self.repo_id} → {dest} (resume={has})", flush=True)
            return has
        except RepositoryNotFoundError:
            print(f"[checkpoint] {self.repo_id} 없음 → 처음 실행(scratch).", flush=True)
            return False

    def push(self, src: str, step: Optional[int] = None) -> None:
        """src 폴더를 저장소로 업로드(덮어쓰기). N스텝마다 호출."""
        self.ensure_repo()
        api = self._api()
        msg = f"checkpoint step={step}" if step is not None else "checkpoint"
        api.upload_folder(folder_path=src, repo_id=self.repo_id, repo_type="model",
                          commit_message=msg)
        print(f"[checkpoint] push {src} → {self.repo_id} ({msg})", flush=True)

    def last_step(self) -> Optional[int]:
        """저장소 최신 커밋 메시지에서 step 파싱(진행 판정용). 없으면 None."""
        import re
        try:
            api = self._api()
            commits = api.list_repo_commits(self.repo_id, repo_type="model")
            for c in commits:
                m = re.search(r"step=(\d+)", c.title or "")
                if m:
                    return int(m.group(1))
        except Exception:
            pass
        return None
