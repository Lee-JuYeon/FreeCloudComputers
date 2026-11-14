"""modal 어댑터 — modal CLI(`modal run`)로 실행.

Modal은 무료 크레딧(~$30/월)형. 진짜 서버리스 GPU라 세션/슬롯 개념이 없고 즉시 뜬다.
크레딧 소진 시 결제 오류 → 페일오버. entrypoint를 Modal 함수로 감싼 러너 스크립트를 생성해 실행.
사용자가 이미 쓰는 modal_worker.py 패턴을 일반화.
"""
from __future__ import annotations

import os
import tempfile

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult


class ModalProvider(Provider):
    name = "modal"
    capabilities = {"gpu": "T4|L4|A10G", "vram_gb": 16, "headless": True,
                    "note": "무료 크레딧 소진 시 페일오버."}

    def probe(self) -> Probe:
        # modal 1.5.x 엔 `token current` 가 없다(rc≠0) → `token info` 로 확인.
        rc, log = self._sh(["modal", "token", "info"], 30)
        if rc == 127:
            return Probe(False, "modal CLI 미설치(pip install modal).")
        if rc != 0:
            return Probe(False, "modal 토큰 미설정(modal token new).")
        return Probe(True, "modal 인증 OK.")

    def run(self, job: Job) -> RunResult:
        gpu = job.needs.get("gpu", "T4")
        self.warn_unfetched(job, self.name)
        work = tempfile.mkdtemp(prefix="fc-modal-")
        runner = os.path.join(work, "runner.py")
        envs = job.resolved_env()
        # 베이스 이미지: 기본은 torch 선탑재 레지스트리 이미지(debian_slim 에서 unsloth/
        # xformers 핀 설치가 torch 버전을 끌어올리는 사고 방지). env 로 교체 가능.
        base = os.environ.get("FREECLOUD_MODAL_IMAGE",
                              "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
        img_expr = (f'modal.Image.from_registry({base!r}).pip_install("huggingface_hub")'
                    if base else
                    'modal.Image.debian_slim().pip_install("huggingface_hub")')
        with open(runner, "w", encoding="utf-8") as f:
            f.write(_RUNNER.format(
                gpu=gpu, timeout=job.max_runtime_s, img_expr=img_expr,
                entrypoint=job.entrypoint, env=repr(envs)))

        # --detach: 로컬 프로세스가 끊겨도 Modal 앱은 계속 실행(caffeinate 와 함께).
        rc, log = self._sh(["modal", "run", "--detach", runner],
                           job.max_runtime_s + 300, env=envs)
        if rc == 0:
            return RunResult(True, "done", "Modal 실행 완료.", log[-400:])
        d = classify(log)
        # 결제/크레딧 문구는 QUOTA로 → 페일오버
        status = "quota" if d.error_class == "QUOTA_COOLDOWN" or "payment" in log.lower() \
            else ("timeout" if rc == 124 else "error")
        return RunResult(False, status, d.advice, log[-1200:], d.error_class)


_RUNNER = '''import modal, subprocess, os
app = modal.App("freecloud")
img = {img_expr}
@app.function(gpu={gpu!r}, timeout={timeout}, image=img, memory=32768, cpu=4)
def _job():
    env = dict(os.environ); env.update({env})
    subprocess.run({entrypoint!r}, shell=True, check=True, env=env)
@app.local_entrypoint()
def main():
    _job.remote()
'''
