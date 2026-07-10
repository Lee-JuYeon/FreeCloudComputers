"""kaggle 어댑터 — kaggle CLI(kernels push/status/output)로 무인 실행.

중요 제약(설계): Kaggle **API/CLI는 단일 GPU만** 노출한다(보통 P100 또는 T4×1).
  T4×2 는 웹 UI 전용 → 헤드리스 오케스트레이터에선 못 씀. capabilities에 명시.
쿼터 ~30 GPU-h/주. 출력이 /kaggle/working 에 영속 → 웹소켓 끊김 레이스 없음(안정적).
Windows cp949 회피 위해 PYTHONUTF8=1 강제(Infopu E4 교훈).
"""
from __future__ import annotations

import json
import os
import tempfile

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult

_UTF8 = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


class KaggleProvider(Provider):
    name = "kaggle"
    capabilities = {"gpu": "P100|T4x1", "vram_gb": 16, "headless": True,
                    "note": "API는 단일 GPU만. T4x2는 UI 전용."}

    def _user(self) -> str:
        if os.environ.get("KAGGLE_USERNAME"):
            return os.environ["KAGGLE_USERNAME"]
        kj = os.path.expanduser("~/.kaggle/kaggle.json")
        if os.path.exists(kj):
            try:
                return json.load(open(kj)).get("username", "")
            except Exception:
                pass
        return ""

    def probe(self) -> Probe:
        if not self._user():
            return Probe(False, "~/.kaggle/kaggle.json 없음(KAGGLE_USERNAME 미설정).")
        rc, log = self._sh(["kaggle", "kernels", "list", "-m", "--page-size", "1"], 60, _UTF8)
        if rc == 127:
            return Probe(False, "kaggle CLI 미설치(pip install kaggle).")
        return Probe(True, "kaggle 인증 OK.")

    def run(self, job: Job) -> RunResult:
        user = self._user()
        if not user:
            return RunResult(False, "error", "Kaggle 인증 없음.", error_class="AUTH")

        slug = f"freecloud-{job.name}".lower().replace("_", "-")[:50]
        kernel_id = f"{user}/{slug}"
        work = tempfile.mkdtemp(prefix="fc-kaggle-")
        # entrypoint를 감싸는 커널 스크립트(체크포인트 pull/push는 entrypoint 책임).
        script = os.path.join(work, "kernel.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(_KERNEL_WRAPPER.format(entrypoint=job.entrypoint))
        meta = {
            "id": kernel_id, "title": slug, "code_file": "kernel.py",
            "language": "python", "kernel_type": "script",
            "enable_gpu": True, "enable_internet": True,   # 단일 GPU만 붙음(설계 제약)
            "dataset_sources": [], "kernel_sources": [],
        }
        with open(os.path.join(work, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)

        env = {**_UTF8, **job.resolved_env()}
        rc, log = self._sh(["kaggle", "kernels", "push", "-p", work], 300, env)
        if rc != 0:
            d = classify(log)
            return RunResult(False, "error", d.advice, log[-1200:], d.error_class)

        # 폴링: kaggle kernels status → complete/error/running.
        import time
        deadline = time.time() + job.max_runtime_s
        while time.time() < deadline:
            time.sleep(60)
            rc, st = self._sh(["kaggle", "kernels", "status", kernel_id], 60, _UTF8)
            low = st.lower()
            if "complete" in low:
                out = os.path.join(work, "out")
                self._sh(["kaggle", "kernels", "output", kernel_id, "-p", out], 180, _UTF8)
                return RunResult(True, "done", "Kaggle 커널 완료.", st[-400:])
            if "error" in low:
                # 커널 내부 diag 회수해 정밀 분류
                out = os.path.join(work, "out")
                self._sh(["kaggle", "kernels", "output", kernel_id, "-p", out], 120, _UTF8)
                diag = ""
                dp = os.path.join(out, "diag.txt")
                if os.path.exists(dp):
                    diag = open(dp, encoding="utf-8", errors="replace").read()
                d = classify(st + "\n" + diag)
                return RunResult(False, "error", d.advice, (diag or st)[-1200:], d.error_class)
        return RunResult(False, "timeout", f"Kaggle 폴링 {job.max_runtime_s}s 초과.",
                         error_class="TIMEOUT")


# 커널 안에서 실행되는 래퍼: 진단 파일(diag.txt) 남겨 실패 분류를 돕는다.
_KERNEL_WRAPPER = '''import subprocess, sys, traceback
try:
    subprocess.run({entrypoint!r}, shell=True, check=True)
except Exception as e:
    open("/kaggle/working/diag.txt","w").write("".join(traceback.format_exc()))
    raise
'''
