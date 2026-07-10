"""saturn 어댑터 — Saturn Cloud(saturn-client)로 Job 실행.

무료: Hosted Free 티어 T4급 GPU(반복 무료). saturn-client의 recipe apply → start 패턴.
문서 확인: SaturnConnection().apply(recipe) → result["state"]["id"] → conn.start("job", id).

인증: env SATURN_TOKEN(+ SATURN_URL, 기본 community). 없으면 probe 실패.
recipe는 job 스펙에서 생성 — git repo/command/이미지/리소스.
"""
from __future__ import annotations

import os
import time

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult


class SaturnProvider(Provider):
    name = "saturn"
    capabilities = {"gpu": "T4", "vram_gb": 16, "headless": True,
                    "note": "Hosted Free 티어. saturn-client recipe."}

    def _conn(self):
        try:
            from saturn_client import SaturnConnection
        except ImportError:
            return None
        url = os.environ.get("SATURN_URL", "https://app.community.saturnenterprise.io")
        from . import _saturn_token
        tok = _saturn_token()
        if not tok:
            return None
        return SaturnConnection(url=url, api_token=tok)

    def probe(self) -> Probe:
        try:
            import saturn_client  # noqa: F401
        except ImportError:
            return Probe(False, "saturn-client 미설치(pip install saturn-client).")
        from . import _saturn_token
        if not _saturn_token():
            return Probe(False, "SATURN_TOKEN 미설정(freecloud login saturn).")
        return Probe(True, "saturn 인증 OK.")

    def run(self, job: Job) -> RunResult:
        conn = self._conn()
        if conn is None:
            return RunResult(False, "error", "saturn-client/토큰 없음.", error_class="AUTH")
        # env export 를 command 앞에 붙여 시크릿/체크포인트 주입
        cmd = " ".join(f"export {k}={v};" for k, v in job.resolved_env().items()) \
            + " " + job.entrypoint
        recipe = {
            "type": "job", "spec": {
                "name": f"fc-{job.name}"[:40],
                "command": cmd,
                "resource": {"instance_type": os.environ.get("SATURN_INSTANCE", "g4dnxlarge")},
                "git_repositories": ([{"url": job.repo}] if job.repo else []),
                "image": os.environ.get("SATURN_IMAGE", "saturncloud/saturn-python:latest"),
            }}
        try:
            result = conn.apply(recipe)
            rid = result["state"]["id"]
            conn.start("job", rid)
            return self._poll(conn, rid, job)
        except Exception as e:
            d = classify(str(e))
            st = "quota" if d.error_class == "QUOTA_COOLDOWN" else "error"
            return RunResult(False, st, f"Saturn 예외: {e}", str(e)[-800:], d.error_class)

    def _poll(self, conn, rid: str, job: Job) -> RunResult:
        deadline = time.time() + job.max_runtime_s
        while time.time() < deadline:
            try:
                st = str(conn.get_job(rid).get("status", "")).lower()
            except Exception:
                st = ""
            if any(s in st for s in ("completed", "success")):
                return RunResult(True, "done", "Saturn Job 완료.")
            if any(s in st for s in ("failed", "error")):
                return RunResult(False, "error", f"Saturn 실패: {st}", error_class="UNKNOWN")
            time.sleep(30)
        return RunResult(False, "timeout", f"Saturn 폴링 {job.max_runtime_s}s 초과.",
                         error_class="TIMEOUT")
