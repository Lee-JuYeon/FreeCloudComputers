"""kaggle 어댑터 — kaggle CLI(kernels push/status/output)로 무인 실행.

중요 제약(설계): Kaggle **API/CLI는 단일 GPU만** 노출한다(기본 P100).
  T4×2 는 웹 UI 전용 → 헤드리스 API로는 못 고른다. 완전 무인 T4×2가 필요하면
  KaggleUIProvider(name="kaggle-ui", Playwright)를 쓴다 — 이 클래스의 헬퍼를 재사용.
쿼터 ~30 GPU-h/주. 출력이 /kaggle/working 에 영속 → 웹소켓 끊김 레이스 없음(안정적).
Windows cp949 회피 위해 PYTHONUTF8=1 강제(Infopu E4 교훈).

2026-09-23 개선(docs/IMPROVEMENTS_2026-09-23.md, 결함 A·B·D):
  · A. `workdir` 를 private 데이터셋으로 실어 보낸다 — 예전엔 생성된 kernel.py 한 장만
    올라가서 `python train_lora.py` 같은 entrypoint 가 원격에 코드가 없어 죽었다.
  · B. complete/error 양쪽에서 `kernels output` 을 **버리지 않고** ./artifacts/<job>/ 에
    남긴다 — 예전엔 임시 디렉터리에 받아 diag.txt 만 읽고 경로를 버려서, 실패 원인을
    보려면 사람이 손으로 다시 받아야 했다.
  · D. access token 수명이 3시간이라 학습 직전마다 재로그인이 필요했다 → refresh_token
    으로 자동 갱신한다.
"""
from __future__ import annotations

import fnmatch
import glob as _glob
import json
import os
import re
import shutil
import tempfile
import time

from ..job import Job
from ..errors import classify
from .base import Provider, Probe, RunResult

_UTF8 = {"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

# ── workdir 스테이징 상수(불변조건 2) ────────────────────────────────────────
# `.env` 는 **항상** 제외한다. 사용자가 시크릿을 거기 두는 것이 관례이고, 데이터셋은
# private 이라도 Kaggle 계정이 뚫리면 그대로 새어나간다 — 애초에 보내지 않는다.
_SKIP_DIRS = (".git", "__pycache__")
_SKIP_FILES = (".env",)
_SKIP_GLOBS = ("*.pyc",)

#: 원격 노드에서 workdir 이 풀리는 곳. 커널이 env 로도 주입한다(FREECLOUD_WORKDIR).
REMOTE_WORKDIR = "/kaggle/working/workdir"


class _WorkdirError(Exception):
    """workdir 스테이징 실패 — run() 이 RunResult 로 바꿔 돌려준다.

    예외를 쓰는 이유: `_stage_workdir` 는 (ds_ref, note) 를 돌려주는 계약이라
    실패를 같은 튜플에 섞으면 호출부가 매번 분기해야 한다. 실패는 한 곳에서 잡는다.
    """

    def __init__(self, msg: str, error_class: str = "JOB_CONFIG", log: str = ""):
        super().__init__(msg)
        self.error_class = error_class
        self.log = log


def workdir_max_bytes() -> int:
    """workdir 업로드 상한. env FREECLOUD_WORKDIR_MAX_MB(기본 500MB).

    0 이면 업로드 자체를 끈다(무료 쿼터를 태우기 전에 사용자가 막을 수 있는 탈출구).
    """
    try:
        mb = float(os.environ.get("FREECLOUD_WORKDIR_MAX_MB", "500"))
    except ValueError:
        mb = 500.0
    return int(max(0.0, mb) * 1024 * 1024)


def _extra_ignores() -> list[str]:
    """env FREECLOUD_WORKDIR_IGNORE — 쉼표로 나눈 glob 목록(추가 제외)."""
    return [p.strip() for p in os.environ.get("FREECLOUD_WORKDIR_IGNORE", "").split(",") if p.strip()]


def _ignored(rel: str, name: str, extra: list[str]) -> bool:
    if name in _SKIP_FILES:
        return True
    if any(fnmatch.fnmatch(name, g) for g in _SKIP_GLOBS):
        return True
    return any(fnmatch.fnmatch(name, g) or fnmatch.fnmatch(rel, g) for g in extra)


def collect_workdir_files(root: str) -> list[tuple[str, str]]:
    """업로드 대상 파일 목록 → [(절대경로, 상대경로)]. 제외 규칙 적용(불변조건 2).

    심볼릭 링크는 따라가지 않는다 — 링크 너머가 workdir 밖(홈·시크릿)일 수 있다.
    """
    extra = _extra_ignores()
    out: list[tuple[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS
            and not _ignored(os.path.relpath(os.path.join(dirpath, d), root), d, extra)
        ]
        for fn in sorted(filenames):
            ap = os.path.join(dirpath, fn)
            rel = os.path.relpath(ap, root)
            if os.path.islink(ap) or not os.path.isfile(ap):
                continue
            if _ignored(rel, fn, extra):
                continue
            out.append((ap, rel))
    return sorted(out, key=lambda t: t[1])


def artifacts_dir_for(job: Job) -> str:
    """산출물 회수 디렉터리 = `./artifacts/<job.name>/`.

    env FREECLOUD_ARTIFACTS_DIR 로 루트(`./artifacts`)를 바꾼다. cwd 기준인 이유:
    산출물은 '그 작업'에 딸린 것이라 사용자가 서 있는 곳에 남아야 찾는다
    (쿨다운/자격증명처럼 계정 전역인 것과 성격이 다르다 — paths.py 주석 참고).
    """
    root = os.environ.get("FREECLOUD_ARTIFACTS_DIR") or os.path.join(os.getcwd(), "artifacts")
    return os.path.join(os.path.abspath(os.path.expanduser(root)), job.name)


class KaggleProvider(Provider):
    name = "kaggle"
    capabilities = {"gpu": "P100", "vram_gb": 16, "headless": True,
                    "note": "API는 단일 GPU(기본 P100). T4x2는 kaggle-ui(Playwright)로."}

    # ── 공용 헬퍼(kaggle-ui가 재사용) ────────────────────────────────────────
    def _sh(self, cmd, timeout, env=None, cwd=None):
        """kaggle CLI 호출 전 OAuth 토큰을 환경변수로 주입한다.

        `kaggle auth login`(OAuth)은 ~/.kaggle/credentials.json 을 만들지만
        **CLI 2.2.x 가 이 파일을 kernels 계열 명령에서 자동으로 읽지 않는다**
        (2026-08-21 실측: 로그인 직후에도 "Authentication required").
        같은 파일의 access_token 을 KAGGLE_API_TOKEN 으로 넣어주면 즉시 통과한다.
        → 사용자에게 수동 export 를 요구하지 않고 여기서 메운다.
        """
        merged = dict(env or {})
        if cmd and cmd[0] == "kaggle" and not os.environ.get("KAGGLE_API_TOKEN"):
            tok = oauth_access_token()
            if tok:
                merged["KAGGLE_API_TOKEN"] = tok
        return super()._sh(cmd, timeout, merged, cwd)

    def _user(self) -> str:
        if os.environ.get("KAGGLE_USERNAME"):
            return os.environ["KAGGLE_USERNAME"]
        for path, key in ((("~/.kaggle/credentials.json"), "username"),
                          (("~/.kaggle/kaggle.json"), "username")):
            fp = os.path.expanduser(path)
            if os.path.exists(fp):
                try:
                    v = json.load(open(fp)).get(key, "")
                    if v:
                        return v
                except Exception:
                    pass
        return ""

    def kernel_slug(self, job: Job) -> str:
        return f"freecloud-{job.name}".lower().replace("_", "-")[:50]

    def kernel_id(self, job: Job) -> str:
        return f"{self._user()}/{self.kernel_slug(job)}"

    def workdir_dataset_slug(self, job: Job) -> str:
        """workdir 데이터셋 slug. Kaggle 은 소문자·하이픈만 받고 50자 제한."""
        return f"freecloud-{job.name}-workdir".lower().replace("_", "-")[:50]

    def workdir_dataset_id(self, job: Job) -> str:
        return f"{self._user()}/{self.workdir_dataset_slug(job)}"

    # ── A. workdir → private 데이터셋 ────────────────────────────────────────
    def _stage_workdir(self, job: Job) -> tuple[str | None, str]:
        """job.workdir 를 Kaggle private 데이터셋으로 올린다 → (dataset_ref|None, note).

        결함 A(docs/IMPROVEMENTS_2026-09-23.md §0): 예전엔 `workdir` 를 Job.load 가
        abspath 만 하고 **어느 provider 도 읽지 않았다**. push 되는 건 생성된 kernel.py
        한 장뿐이라 `python train_lora.py` 같은 entrypoint 는 원격에 코드가 없어 죽었다.

        None 을 돌려주는 경우(note 에 이유):
          · job.workdir 가 비어 있음 — 사용자가 **명시하지 않았다**(기본값). 이때는
            아무것도 올리지 않고 커널도 예전과 똑같이 만든다(불변조건 3).
          · repo 가 지정됨 — 코드 소스가 git 이므로 업로드 대상이 아니다
          · workdir 디렉터리가 없음 / 업로드할 파일이 하나도 없음
          · FREECLOUD_WORKDIR_MAX_MB=0 (업로드 끔)
        상한 초과·업로드 실패는 _WorkdirError 로 올린다(불변조건 2 — 이때 kaggle CLI
        호출은 0회여야 하므로 크기 검사를 CLI 호출보다 **먼저** 한다).
        """
        # ⚠️ 2026-09-23 라이브 회귀: 기본값이 "." 이던 시절에는 workdir 키가 **없는**
        #    job 도 '업로드 대상 있음'으로 보여, 붙지도 않은 입력 데이터셋을 복사하려다
        #    커널이 FileNotFoundError 로 죽었다. opt-in 으로 못 박는다(job.py 주석 참고).
        if not job.workdir:
            return None, "workdir 미지정 → 업로드 없음(코드는 entrypoint 가 직접 확보)."
        if job.repo:
            return None, f"repo={job.repo} 지정 → workdir 업로드 생략."
        root = os.path.abspath(os.path.expanduser(job.workdir))
        if not os.path.isdir(root):
            return None, f"workdir 없음({root}) → 업로드 생략."
        cap = workdir_max_bytes()
        if cap <= 0:
            return None, "FREECLOUD_WORKDIR_MAX_MB=0 → workdir 업로드 끔."

        files = collect_workdir_files(root)
        if not files:
            return None, f"workdir({root})에 올릴 파일 없음 → 업로드 생략."
        total = sum(os.path.getsize(ap) for ap, _ in files)
        if total > cap:
            raise _WorkdirError(
                f"workdir too large: {total/1048576:.1f}MB > {cap/1048576:.0f}MB "
                f"({root}, {len(files)}개). FREECLOUD_WORKDIR_IGNORE 로 제외하거나 "
                f"FREECLOUD_WORKDIR_MAX_MB 를 올려라.")

        ref = self.workdir_dataset_id(job)
        stage = tempfile.mkdtemp(prefix="fc-workdir-")
        for ap, rel in files:
            dst = os.path.join(stage, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(ap, dst)
        with open(os.path.join(stage, "dataset-metadata.json"), "w", encoding="utf-8") as f:
            json.dump({"title": self.workdir_dataset_slug(job), "id": ref,
                       "licenses": [{"name": "other"}]}, f)

        # -r zip: 하위 디렉터리를 zip 으로 묶어 올린다(기본 skip 은 **하위 폴더를 통째로
        # 버린다** — 패키지 구조가 있는 코드는 그대로면 반쪽만 올라간다).
        # --public 플래그를 주지 않으므로 private 이 기본(시크릿이 env 로 섞일 수 있다).
        rc, log = self._sh(["kaggle", "datasets", "create", "-p", stage, "-r", "zip"], 900, _UTF8)
        if rc != 0:
            # 이미 있는 데이터셋이면 create 는 실패한다 → 새 버전으로 올린다.
            rc, vlog = self._sh(["kaggle", "datasets", "version", "-p", stage,
                                 "-m", f"freecloud {job.name}", "-r", "zip"], 900, _UTF8)
            log = log + "\n" + vlog
            if rc != 0:
                # ★ 업로드가 실패했는데 커널을 push 하면, 코드 없는 노드에서 job 이
                #   돌거나(입력 미첨부) 다른 provider 로 페일오버해 **조용히 옛 코드로**
                #   학습한다. JOB_CONFIG=ABORT 로 push 前에 끝낸다(원인은 reason/log 에).
                raise _WorkdirError(
                    f"workdir 데이터셋 업로드 실패({ref}) — 커널을 push 하지 않는다. "
                    f"원인: {classify(log).advice}", "JOB_CONFIG", log)
        ready = self._wait_dataset_ready(ref)
        if ready != "ready":
            raise _WorkdirError(
                f"workdir 데이터셋({ref})이 준비되지 않았다 — 커널을 push 하지 않는다. "
                f"상태: {(ready or '(응답 없음)').strip()[:200]}", "JOB_CONFIG", ready)
        return ref, (f"workdir 업로드 완료: {ref} ({len(files)}개, {total/1048576:.1f}MB) "
                     f"→ 노드에서 {REMOTE_WORKDIR}")

    def _wait_dataset_ready(self, ref: str, timeout_s: int = 300) -> str:
        """데이터셋 처리 완료 대기. Kaggle 은 업로드 직후 비동기로 처리한다 —
        ready 전에 커널을 push 하면 입력이 안 붙거나 옛 버전이 붙는다."""
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            _, out = self._sh(["kaggle", "datasets", "status", ref], 60, _UTF8)
            last = out or ""
            low = last.lower()
            if "ready" in low or "complete" in low:
                return "ready"
            if "error" in low:
                return last
            time.sleep(10)
        return last

    def _write_kernel_dir(self, job: Job, dataset_ref: str = "") -> str:
        """work 디렉터리에 kernel.py + kernel-metadata.json 생성 → 경로 반환.

        job.resolved_env(시크릿 HF_TOKEN 등 포함)를 커널 스크립트에 주입한다.
        토큰이 커널 소스에 들어가므로 반드시 is_private=True(아래)로 push한다.

        dataset_ref 가 있으면 dataset_sources 에 붙이고 커널이 그것을 workdir 로
        복사·cd 한다(결함 A).

        ★ 불변: '복사 코드'와 'dataset_sources' 는 **같은 dataset_ref 하나**에서 나온다.
          둘이 어긋나면(=입력 없이 복사) 커널이 FileNotFoundError 로 죽는다 —
          2026-09-23 라이브 회귀가 정확히 그거였다. 분기를 하나로 유지할 것.
        """
        work = tempfile.mkdtemp(prefix="fc-kaggle-")
        title = dataset_ref.split("/")[-1] if dataset_ref else None
        with open(os.path.join(work, "kernel.py"), "w", encoding="utf-8") as f:
            f.write(_build_kernel_script(job.entrypoint, job.resolved_env(), title))
        meta = {
            "id": self.kernel_id(job), "title": self.kernel_slug(job),
            "code_file": "kernel.py", "language": "python", "kernel_type": "script",
            "is_private": True,                 # ★ env에 시크릿 주입되므로 비공개 강제
            "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [dataset_ref] if dataset_ref else [],
            "kernel_sources": [], "competition_sources": [],
        }
        with open(os.path.join(work, "kernel-metadata.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return work

    def _stage_and_write(self, job: Job) -> tuple[str, RunResult | None]:
        """workdir 스테이징 + 커널 디렉터리 작성 → (work, None) 또는 ("", 실패 RunResult).

        kaggle 과 kaggle-ui 가 **같은 경로**를 타도록 여기 한 곳에 모은다
        (설계 §3: kaggle-ui 는 부모 헬퍼 재사용으로 자동 적용).
        """
        try:
            ref, note = self._stage_workdir(job)
        except _WorkdirError as e:
            print(f"[{self.name}] {e}", flush=True)
            return "", RunResult(False, "error", str(e), (e.log or "")[-1200:], e.error_class)
        if note:
            print(f"[{self.name}] {note}", flush=True)
        return self._write_kernel_dir(job, ref or ""), None

    def _push(self, work: str, job: Job) -> tuple[int, str]:
        env = {**_UTF8, **job.resolved_env()}
        return self._sh(["kaggle", "kernels", "push", "-p", work], 300, env)

    # ── B. 산출물 회수 ──────────────────────────────────────────────────────
    def fetch_outputs(self, job: Job, kernel_id: str) -> tuple[str, str]:
        """`kaggle kernels output` 을 ./artifacts/<job>/ 로 받고 (경로, diag) 반환.

        결함 B: 예전 `_read_diag` 는 임시 디렉터리에 받아 diag.txt 만 읽고 **경로를
        버렸다**. 커널 Traceback 을 보려면 사람이 손으로 다시 받아야 했다.
        complete/error 양쪽에서 부르므로 실패해도 예외를 올리지 않는다.
        """
        out = artifacts_dir_for(job)
        os.makedirs(out, exist_ok=True)
        self._sh(["kaggle", "kernels", "output", kernel_id, "-p", out], 180, _UTF8)
        dp = os.path.join(out, "diag.txt")
        diag = ""
        if os.path.exists(dp):
            try:
                diag = open(dp, encoding="utf-8", errors="replace").read()
            except OSError:
                diag = ""
        return out, diag

    @staticmethod
    def missing_artifacts(job: Job, adir: str) -> list[str]:
        """job.artifacts 를 '회수본에 이 파일이 있어야 한다'로 해석 → 빠진 것 목록.

        선언은 원격 경로(`/kaggle/working/out.glb`)인데 회수본은 평평하게 내려오므로
        **파일명 기준**으로 확인한다. 못 찾으면 경고만 한다 — 성공 판정은 success_glob 몫.
        """
        wanted = [a for a in (getattr(job, "artifacts", None) or []) if str(a).strip()]
        if not wanted:
            return []
        have = {os.path.basename(p) for p in _glob.glob(os.path.join(adir, "**", "*"),
                                                        recursive=True)}
        return [a for a in wanted if os.path.basename(str(a).rstrip("/")) not in have]

    @staticmethod
    def success_glob_hits(job: Job, adir: str) -> list[str]:
        """success_glob 평가. cwd 기준으로 먼저 보고, 없으면 회수 디렉터리 기준으로 본다.

        두 기준을 다 보는 이유: 사용자가 `artifacts/**/hello.txt`(cwd 기준 전체 경로)로도,
        `*.txt`(회수본 안쪽)로도 쓴다. 한쪽만 지원하면 조용히 '성공 0건'이 된다.
        """
        pat = (getattr(job, "success_glob", "") or "").strip()
        if not pat:
            return []
        hits = _glob.glob(pat, recursive=True)
        if not hits and not os.path.isabs(pat):
            hits = _glob.glob(os.path.join(adir, pat), recursive=True)
        return sorted(hits)

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
                adir, diag = self.fetch_outputs(job, kid)
                got = self.assigned_gpu(diag)
                want = job.needs.get("gpu")
                if want and got and not self._gpu_matches(want, got):
                    return RunResult(False, "needs_setup",
                                     f"요청={want} 인데 실제={got}. UI에서 가속기 재설정 필요.",
                                     diag[-600:], "GPU_MISMATCH", adir)
                missing = self.missing_artifacts(job, adir)
                if missing:
                    print(f"[{self.name}] 선언된 artifacts 미확인: {', '.join(missing)} "
                          f"(회수본 {adir})", flush=True)
                if job.success_glob and not self.success_glob_hits(job, adir):
                    # 커널이 '완료'로 끝났는데 산출물이 없다 = 코드 문제. 페일오버해도
                    # 다른 provider 에서 똑같이 빈손이다 → ABORT(errors.py NO_ARTIFACT).
                    return RunResult(False, "error",
                                     f"NO_ARTIFACT: 커널은 complete 인데 success_glob"
                                     f"({job.success_glob}) 매치 0 — 산출물이 없다. 회수본={adir}",
                                     diag[-1200:], "NO_ARTIFACT", adir)
                return RunResult(True, "done",
                                 f"Kaggle 완료(GPU={got or '?'}). 산출물 → {adir}",
                                 st[-400:], "", adir)
            if "error" in low:
                adir, diag = self.fetch_outputs(job, kid)
                d = classify(st + "\n" + diag)
                return RunResult(False, "error", d.advice, (diag or st)[-1200:],
                                 d.error_class, adir)
        return RunResult(False, "timeout", f"Kaggle 폴링 {job.max_runtime_s}s 초과.",
                         error_class="TIMEOUT")

    # ── 인터페이스 ───────────────────────────────────────────────────────────
    def probe(self) -> Probe:
        """실제 API 호출 결과까지 본다.

        ⚠️ 과거 버그: rc==127(CLI 부재)만 검사하고 rc!=0(인증 실패)은 그냥 통과시켜
        `fcc auth` 가 "kaggle 인증 OK" 라 답한 뒤 `fcc run` 이 AUTH 로 죽었다.
        probe 는 '붙는다'를 보장해야 의미가 있다 — 거짓 양성은 없느니만 못하다.
        """
        rc, log = self._sh(["kaggle", "kernels", "list", "-m", "--page-size", "1"], 60, _UTF8)
        if rc == 127:
            return Probe(False, "kaggle CLI 미설치(pip install kaggle).")
        if rc == 0:
            return Probe(True, f"kaggle 인증 OK{(' (' + self._user() + ')') if self._user() else ''}.")
        return Probe(False, kaggle_auth_hint(log))

    def run(self, job: Job) -> RunResult:
        if not self._user():
            return RunResult(False, "error", "Kaggle 인증 없음.", error_class="AUTH")
        work, err = self._stage_and_write(job)
        if err is not None:
            return err
        rc, log = self._push(work, job)
        if rc != 0:
            d = classify(log)
            return RunResult(False, "error", d.advice, log[-1200:], d.error_class)
        return self._poll_and_fetch(job, work)



# ── D. OAuth access token (3시간 수명 → refresh_token 으로 자동 갱신) ────────
def _credentials_path() -> str:
    return os.path.expanduser("~/.kaggle/credentials.json")


def _refresh_via_sdk(refresh_token: str) -> tuple[str, str]:
    """kagglesdk 로 access token 재발급 → (access_token, expiration_iso).

    **엔드포인트를 우리가 하드코딩하지 않는다.** SDK 가 아는 것은 SDK 가 하게 두고,
    우리는 그 결과만 받는다(URL 이 바뀌면 SDK 업데이트로 따라간다).
    SDK 의 `KaggleCredentials.refresh_access_token()` 을 쓰지 않는 이유: 그쪽은
    내부 `save()` 를 호출하는데 우리가 만든 객체엔 username/scopes 가 비어 있어
    credentials.json 의 그 필드를 **지워버린다**. 토큰 발급만 빌리고 저장은 우리가 한다.

    테스트는 이 함수 하나만 몽키패치하면 된다(네트워크 0).
    """
    from datetime import datetime, timedelta, timezone
    from kagglesdk.kaggle_client import KaggleClient
    from kagglesdk.kaggle_creds import KaggleCredentials

    with KaggleClient() as client:
        creds = KaggleCredentials(client=client, refresh_token=refresh_token)
        resp = creds.generate_access_token()
    token = str(getattr(resp, "token", "") or "")
    if not token:
        return "", ""
    secs = int(getattr(resp, "expires_in", 0) or 0)
    exp = (datetime.now(timezone.utc) + timedelta(seconds=secs)).isoformat()
    return token, exp


def _expires_within(d: dict, seconds: int = 300) -> bool:
    """만료(또는 seconds 이내 만료)인가. 만료시각이 없거나 못 읽으면 False(옛 동작 유지)."""
    exp = str(d.get("access_token_expiration", ""))
    if not exp:
        return False
    try:
        from datetime import datetime, timedelta, timezone
        t = datetime.fromisoformat(exp.replace("Z", "+00:00"))
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return t <= datetime.now(timezone.utc) + timedelta(seconds=seconds)
    except Exception:
        return False


def _save_credentials(fp: str, d: dict) -> None:
    """credentials.json 원자적 갱신(tmp + replace, mode 0600).

    원자성이 필요한 이유: 이 파일은 kaggle CLI 가 **동시에** 읽는다. 덮어쓰다 죽으면
    반쪽짜리 JSON 이 남아 로그인이 통째로 날아간다.
    """
    os.makedirs(os.path.dirname(fp), exist_ok=True)
    tmp = fp + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    try:
        os.chmod(tmp, 0o600)
    except OSError:
        pass        # Windows 등에서는 무시(원본 CLI 도 같은 처리)
    os.replace(tmp, fp)


def oauth_access_token() -> str:
    """`kaggle auth login` 이 남긴 OAuth access_token. 없거나 갱신 실패면 빈 문자열.

    구조: {refresh_token, access_token, access_token_expiration, username, scopes}

    결함 D(docs/IMPROVEMENTS_2026-09-23.md §0): access token 수명이 **3시간**이라
    예전엔 학습 직전마다 `kaggle auth login --force` 를 사람이 쳐야 했다. 같은 파일에
    refresh_token 이 있고 kagglesdk 에 갱신 경로가 있으니, 만료(또는 5분 이내 만료)면
    여기서 새로 받아 파일을 원자적으로 갱신한다(불변조건 7).
    """
    fp = _credentials_path()
    if not os.path.exists(fp):
        return ""
    try:
        d = json.load(open(fp))
    except Exception:
        return ""

    if not _expires_within(d):
        return str(d.get("access_token", "") or "")

    rt = str(d.get("refresh_token", "") or "")
    if not rt:
        return ""              # 만료 + 갱신 수단 없음 → probe 가 재로그인 안내를 내도록
    try:
        token, exp = _refresh_via_sdk(rt)
    except Exception as e:
        print(f"[kaggle] access token 갱신 실패({type(e).__name__}) — "
              f"`kaggle auth login --force` 로 재로그인하라.", flush=True)
        return ""
    if not token:
        print("[kaggle] access token 갱신 실패(빈 토큰) — "
              "`kaggle auth login --force` 로 재로그인하라.", flush=True)
        return ""

    d["access_token"] = token
    if exp:
        d["access_token_expiration"] = exp
    try:
        _save_credentials(fp, d)
    except OSError as e:
        print(f"[kaggle] credentials.json 저장 실패({e}) — 토큰은 이번 실행에만 쓴다.",
              flush=True)
    print("[kaggle] access token 자동 갱신됨(수명 3시간 → refresh_token 으로 재발급).",
          flush=True)
    return token


def kaggle_auth_hint(log: str) -> str:
    """kaggle CLI 실패 로그 → 사람이 바로 실행할 수 있는 한 줄 안내.

    kaggle CLI 2.2.x 부터 인증 방식이 바뀌었다. 예전 `~/.kaggle/kaggle.json`
    (username+key)만 있으면 되던 것이 이제 OAuth 또는 단일 API 토큰을 요구하고,
    kaggle.json 이 있어도 "Authentication required" 로 거절한다.
    실측(2026-08-21): kaggle.json·access_token 이 둘 다 있는데도 거절 → 재로그인 필요.
    """
    low = (log or "").lower()
    if "authentication required" in low or "401" in low or "unauthorized" in low:
        return ("kaggle 인증 만료/불일치 — `kaggle auth login`(브라우저 OAuth) 으로 재로그인하라. "
                "비대화형이면 kaggle.com/settings/api 에서 새 토큰을 받아 "
                "KAGGLE_API_TOKEN 환경변수 또는 ~/.kaggle/access_token 에 넣는다. "
                "※ CLI 2.2+ 는 예전 kaggle.json(username+key)만으로는 통과하지 않는다.")
    if "403" in low or "forbidden" in low:
        return "kaggle 권한 거부(403) — 계정 상태/전화 인증(Phone verification) 확인 필요."
    if "429" in low or "too many requests" in low:
        return "kaggle 요청 과다(429) — 잠시 후 재시도."
    tail = (log or "").strip().splitlines()
    return "kaggle CLI 실패: " + (tail[-1][:160] if tail else "원인 불명")

# 커널 안에서 실행되는 래퍼: env(시크릿) 주입 + 실제 GPU(nvidia-smi)·예외를 diag.txt에 남긴다.
# str.format을 쓰지 않는다(env dict의 중괄호가 깨짐) — json.dumps + %r 로 안전하게 삽입.
def _build_kernel_script(entrypoint: str, env: dict, workdir_title: str | None = None) -> str:
    """커널 스크립트 생성. workdir_title 이 없으면 **예전과 바이트 동일**(불변조건 3).

    workdir_title 이 있으면 entrypoint 앞에 '입력 데이터셋 → /kaggle/working/workdir 복사
    + cd' 를 끼워넣는다. 복사하는 이유: /kaggle/input 은 **읽기 전용**이라 학습 산출물을
    그 안에 못 쓴다. zip 을 푸는 이유: 하위 디렉터리는 `-r zip` 으로 올라간다(_stage_workdir).
    """
    stage = ""
    if workdir_title:
        stage = (
            "    import shutil, glob, zipfile\n"
            "    _src = '/kaggle/input/%s'\n" % workdir_title +
            "    _dst = %r\n" % REMOTE_WORKDIR +
            "    shutil.copytree(_src, _dst, dirs_exist_ok=True)\n"
            "    for _z in sorted(glob.glob(_dst + '/*.zip')):\n"
            "        zipfile.ZipFile(_z).extractall(_dst)\n"
            "    os.environ['FREECLOUD_WORKDIR'] = _dst\n"
            "    os.chdir(_dst)\n"
        )
    head = (
        "import os, json, subprocess, traceback\n"
        "os.environ.update(json.loads(%r))\n" % json.dumps(env or {}) +
        "open('/kaggle/working/diag.txt','w').write('boot\\n')\n"
        "try:\n"
        "    subprocess.run('nvidia-smi -L >> /kaggle/working/diag.txt 2>&1', shell=True)\n"
    )
    tail = (
        "    subprocess.run(%r, shell=True, check=True)\n" % entrypoint +
        "except Exception:\n"
        "    open('/kaggle/working/diag.txt','a').write(traceback.format_exc())\n"
        "    raise\n"
    )
    return head + stage + tail
