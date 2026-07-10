"""kaggle 어댑터 — kaggle CLI(kernels push/status/output)로 무인 실행.

중요 제약(설계): Kaggle **API/CLI는 단일 GPU만** 노출한다(기본 P100).
  T4×2 는 웹 UI 전용 → 헤드리스 API로는 못 고른다. 완전 무인 T4×2가 필요하면
  KaggleUIProvider(name="kaggle-ui", Playwright)를 쓴다 — 이 클래스의 헬퍼를 재사용.
쿼터 ~30 GPU-h/주. 출력이 /kaggle/working 에 영속 → 웹소켓 끊김 레이스 없음(안정적).
Windows cp949 회피 위해 PYTHONUTF8=1 강제(Infopu E4 교훈).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import time

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult

_UTF8 = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


class KaggleProvider(Provider):
    name = "kaggle"
    capabilities = {"gpu": "P100", "vram_gb": 16, "headless": True,
                    "note": "API는 단일 GPU(기본 P100). T4x2는 kaggle-ui(Playwright)로."}

    # ── 공용 헬퍼(kaggle-ui가 재사용) ────────────────────────────────────────
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

    def kernel_slug(self, job: Job) -> str:
        return f"freecloud-{job.name}".lower().replace("_", "-")[:50]

    def kernel_id(self, job: Job) -> str:
        return f"{self._user()}/{self.kernel_slug(job)}"

    def _write_kernel_dir(self, job: Job) -> str:
        """work 디렉터리에 kernel.py + kernel-metadata.json 생성 → 경로 반환.

        job.resolved_env(시크릿 HF_TOKEN 등 포함)를 커널 스크립트에 주입한다.
        토큰이 커널 소스에 들어가므로 반드시 is_private=True(아래)로 push한다.
        """
        work = tempfile.mkdtemp(prefix="fc-kaggle-")
        with open(os.path.join(work, "kernel.py"), "w", encoding="utf-8") as f:
            f.write(_build_kernel_script(job.entrypoint, job.resolved_env()))
        meta = {
            "id": self.kernel_id(job), "title": self.kernel_slug(job),
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True,                 # ★ env에 시크릿 주입되므로 비공개 강제
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
        }
        with open(os.path.join(work, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return work

    def _push(self, work: str, job: Job) -> tuple[int, str]:
        env = {**_UTF8, **job.resolved_env()}
        return self._sh(["kaggle", "kernels", "push", "-p", work], 300, env)

    def _read_diag(self, kernel_id: str, work: str) -> str:
        out = os.path.join(work, "out")
        self._sh(["kaggle", "kernels", "output", kernel_id, "-p", out], 180, _UTF8)
        dp = os.path.join(out, "diag.txt")
        return open(dp, encoding="utf-8", errors="replace").read() if os.path.exists(dp) else ""

    @staticmethod
    def assigned_gpu(diag: str) -> str:
        """nvidia-smi -L 파싱 → 짧은 정규화 코드('P100'/'T4'/'T4x2'/'A100'…). 없으면 ''."""
        models = re.findall(r"(P100|V100|A100|A10G|L40S|L40|L4|T4)", diag or "")
        if not models:
            return ""
        if models.count("T4") >= 2:      # T4가 2개 = T4x2
            return "T4x2"
        return models[0]

    @staticmethod
    def _gpu_matches(want: str, got: str) -> bool:
        """요청 GPU와 실제 GPU가 호환되는지(부분일치). 'P100'~'Tesla P100', 'T4'~'T4x2' 등."""
        w = (want or "").lower().replace(" ", "").replace("-", "")
        g = (got or "").lower().replace(" ", "").replace("-", "")
        return not w or not g or w in g or g in w

    def _poll_and_fetch(self, job: Job, work: str) -> RunResult:
        kid = self.kernel_id(job)
        deadline = time.time() + job.max_runtime_s
        while time.time() < deadline:
            time.sleep(60)
            rc, st = self._sh(["kaggle", "kernels", "status", kid], 60, _UTF8)
            low = st.lower()
            if "complete" in low:
                diag = self._read_diag(kid, work)
                got = self.assigned_gpu(diag)
                want = job.needs.get("gpu")
                if want and got and not self._gpu_matches(want, got):
                    return RunResult(False, "needs_setup",
                                     f"요청={want} 인데 실제={got}. UI에서 가속기 재설정 필요.",
                                     diag[-600:], "GPU_MISMATCH")
                return RunResult(True, "done", f"Kaggle 완료(GPU={got or '?'}).", st[-400:])
            if "error" in low:
                diag = self._read_diag(kid, work)
                d = classify(st + "\n" + diag)
                return RunResult(False, "error", d.advice, (diag or st)[-1200:], d.error_class)
        return RunResult(False, "timeout", f"Kaggle 폴링 {job.max_runtime_s}s 초과.",
                         error_class="TIMEOUT")

    # ── 인터페이스 ───────────────────────────────────────────────────────────
    def probe(self) -> Probe:
        if not self._user():
            return Probe(False, "~/.kaggle/kaggle.json 없음(KAGGLE_USERNAME 미설정).")
        rc, log = self._sh(["kaggle", "kernels", "list", "-m", "--page-size", "1"], 60, _UTF8)
        if rc == 127:
            return Probe(False, "kaggle CLI 미설치(pip install kaggle).")
        return Probe(True, "kaggle 인증 OK.")

    def run(self, job: Job) -> RunResult:
        if not self._user():
            return RunResult(False, "error", "Kaggle 인증 없음.", error_class="AUTH")
        work = self._write_kernel_dir(job)
        rc, log = self._push(work, job)
        if rc != 0:
            d = classify(log)
            return RunResult(False, "error", d.advice, log[-1200:], d.error_class)
        return self._poll_and_fetch(job, work)


# 커널 안에서 실행되는 래퍼: env(시크릿) 주입 + 실제 GPU(nvidia-smi)·예외를 diag.txt에 남긴다.
# str.format을 쓰지 않는다(env dict의 중괄호가 깨짐) — json.dumps + %r 로 안전하게 삽입.
def _build_kernel_script(entrypoint: str, env: dict) -> str:
    return (
        "import os, json, subprocess, traceback\n"
        "os.environ.update(json.loads(%r))\n" % json.dumps(env or {}) +
        "open('/kaggle/working/diag.txt','w').write('boot\\n')\n"
        "try:\n"
        "    subprocess.run('nvidia-smi -L >> /kaggle/working/diag.txt 2>&1', shell=True)\n"
        "    subprocess.run(%r, shell=True, check=True)\n" % entrypoint +
        "except Exception:\n"
        "    open('/kaggle/working/diag.txt','a').write(traceback.format_exc())\n"
        "    raise\n"
    )
