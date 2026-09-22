# fcc 개선 설계 — 2026-09-23 (Laibrio 실학습 연동에서 드러난 결함)

설계=Fable, 구현=Opus, 검증=Fable. 사용자 지시: "fcc 문제점들 수정해서 최신화".

## 0. 실측된 결함 (2026-09-23, HEAD 2017339, Laibrio_AI `run_fcc` 라이브 실행 3회)

| # | 결함 | 실물 |
|---|---|---|
| A | **파일 업로드 없음** | `workdir` 는 `Job.load` 가 abspath 만 하고 **어느 provider 도 읽지 않는다**. Kaggle 에 push 되는 건 생성된 `kernel.py` 한 장(`providers/kaggle.py:_write_kernel_dir`), `dataset_sources=[]` 고정. `examples/job.yaml` 의 entrypoint(`python train_lora.py`)는 그래서 **실행 불가능한 예시**다. Laibrio 는 HF dataset repo 로 우회했다 |
| B | **산출물 회수 없음** | `kaggle kernels output` 은 `_read_diag` 가 임시 디렉터리에 받아 `diag.txt` 만 읽고 경로를 버린다. `job.artifacts` 는 kaggle/kaggle-ui 에서 무시(`warn_unfetched` 조차 호출 안 함). `success_glob` 은 어디서도 안 읽힌다 |
| C | **실패 원인이 안 보인다** | CLI 는 `→ ok=False UNKNOWN 미분류 오류 — …` 한 줄만 찍는다. `RunResult.log_tail`(1,200자) 은 반환 dict 안에만 있고 출력되지 않는다. 실제 원인(커널 Traceback)은 `kaggle kernels output` 을 손으로 받아야 보였다 |
| D | **Kaggle OAuth 갱신 없음** | `oauth_access_token()` 은 `credentials.json` 의 만료시각만 보고 `""` 를 돌려준다. access token 수명이 **3시간**이라 학습 직전마다 `kaggle auth login --force` 가 필요했다. `refresh_token` 이 같은 파일에 있고 `kagglesdk/kaggle_oauth.py` 에 갱신 로직이 있다 |
| E | **entrypoint 코드 버그도 페일오버한다** | 커널이 `TypeError` 로 죽었는데 `UNKNOWN → FAILOVER` 로 분류돼 kaggle-ui 에 30분 쿨다운이 걸리고 다음 라운드를 60초씩 3번 돌았다. 코드 버그는 어느 provider 로 가도 똑같이 죽는다 — 쿼터·쿨다운만 태운다 |
| F | **kaggle-login 이 창을 닫으면 Traceback** | `save_login_state` 폴링 중 사용자가 브라우저를 닫으면 `TargetClosedError` 트레이스백으로 죽는다. 안내 없이 |
| G | **문서 stale** | `docs/AUTH.md` 가 `.freecloud/`(프로젝트 상대) 경로를 말하지만 `paths.py` 는 `~/.freecloud` 다. README 에 토큰 수명·`--force` 안내 없음 |

## 1. 결정

- **A. `workdir` 를 Kaggle private 데이터셋으로 실어 보낸다.** `repo` 가 비고 `workdir` 가 있으면 `kaggle datasets create/version` 로 `{user}/freecloud-{name}-workdir`(isPrivate, license other) 를 만들고 `dataset_sources` 에 붙인다. 생성 커널은 entrypoint 전에 `/kaggle/input/<dataset-title>/` 를 `/kaggle/working/workdir/` 로 복사하고 `cd` 한다(입력은 읽기 전용이라 복사). 제외: `.git/`, `__pycache__/`, `*.pyc`, `.env`, 그리고 `FREECLOUD_WORKDIR_IGNORE`(쉼표 glob) — **`.env` 는 항상 제외**(시크릿 유출 방지). 크기 상한 `workdir_max_mb`(신규 Job 필드 아님 — env `FREECLOUD_WORKDIR_MAX_MB`, 기본 500) 초과 시 JOB_CONFIG 로 즉시 실패. 노드에는 `FREECLOUD_WORKDIR=/kaggle/working/workdir` 를 주입. `examples/job.yaml` 이 그대로 돌게 된다.
- **B. Kaggle 계열은 항상 산출물을 회수한다.** complete/error 모두 `kaggle kernels output` 을 **`./artifacts/<job.name>/`**(cwd 기준, env `FREECLOUD_ARTIFACTS_DIR` 로 변경) 에 받고 `RunResult` 에 `artifacts_dir` 필드를 추가해 CLI 가 경로를 찍는다. `job.artifacts` 는 Kaggle 에서 "이 중 존재 확인할 파일명 목록"으로 해석(없으면 경고). `success_glob` 은 회수 디렉터리에서 평가: complete 인데 glob 매치 0 이면 `ok=False, status="error", error_class="NO_ARTIFACT"`(FAILOVER 아님 → ABORT: 커널이 끝났는데 산출물이 없으면 코드 문제).
- **C. 실패는 로그 꼬리를 찍는다.** `_cmd_run` 이 `attempts[]` 를 순회해 `provider · status · error_class · reason` 과 `log_tail` 마지막 40줄을 들여쓰기해 출력. `fcc run --quiet` 로 끌 수 있다. 신규 `fcc logs <job.yaml|name>`: Kaggle 커널의 `kernels output` + `kernels status` 를 받아 diag.txt·로그 꼬리를 보여준다.
- **D. OAuth refresh.** `oauth_access_token()` 이 만료(또는 5분 이내 만료)이고 `refresh_token` 이 있으면 `kagglesdk` 의 갱신 경로를 호출해 새 access token 을 얻고 `credentials.json` 을 **원자적으로**(tmp+replace, mode 0600) 갱신한다. `kagglesdk` 가 없거나 갱신 실패면 옛 동작(빈 문자열 + 안내). 갱신 성공 시 한 줄 로그. 테스트는 `kagglesdk` 호출을 몽키패치.
- **E. 코드 버그는 ABORT.** `errors.py` 에 `ENTRYPOINT_ERROR` 추가: 로그에 `Traceback (most recent call last)` 가 있고 그 뒤 마지막 예외 줄이 `(TypeError|ValueError|KeyError|AttributeError|AssertionError|IndexError|NameError|ZeroDivisionError|RuntimeError|SystemExit)` 이면 ABORT(쿨다운 없음). 단 **OOM·CUDA·quota 패턴이 먼저 매치되면 그쪽 우선**(순서: 기존 규칙 → ENTRYPOINT_ERROR → UNKNOWN). `UNKNOWN` 은 그대로 FAILOVER 지만 쿨다운을 1800 → **300초**로 줄인다(미분류를 30분 잠그는 건 과하다).
- **F. kaggle-login 견고화.** `TargetClosedError`(및 Playwright `Error` 로 브라우저 종료) 를 잡아 `[FAIL] 브라우저가 로그인 감지 전에 닫혔습니다 — 다시 실행하고 창을 닫지 마세요(자동으로 닫힙니다)` 로 종료(rc 1). 감지 성공 시 기존 `[OK]`.
- **G. 문서.** `docs/AUTH.md` 경로를 `~/.freecloud` 로, README 에 (1) Kaggle access token 3시간·자동 refresh (2) `kaggle auth login --force`(CLI 가 만료 토큰을 '로그인됨'으로 오판) (3) `workdir` 업로드·`artifacts/` 회수·`fcc logs` (4) kaggle-login 은 창을 닫지 말 것. `examples/job.yaml` 주석을 실제 동작에 맞게.

## 2. 불변조건 (pytest 로 고정, `tests/` 에 추가)

1. `Job` 스키마 키는 **늘리지 않는다**(12개 그대로). 새 동작은 기존 필드(`workdir`·`artifacts`·`success_glob`)를 살리는 것으로만.
2. `workdir` 스테이징: `.git`·`__pycache__`·`*.pyc`·`.env` 가 **절대** 포함되지 않는다(테스트가 .env 를 심고 확인). 상한 초과 → `RunResult(False,"error",…,"JOB_CONFIG")`, kaggle CLI 호출 0.
3. 생성 커널 스크립트는 여전히 `compile()` 되고, workdir 가 있을 때만 복사+cd 코드가 들어가며, 없을 때는 **기존과 바이트 동일**(옛 사용자 회귀 0).
4. Kaggle `_poll_and_fetch` 는 complete/error 양쪽에서 산출물 디렉터리를 만들고 `RunResult.artifacts_dir` 를 채운다(`_sh` 몽키패치로 CLI 호출 순서·인자 검증). `success_glob` 매치 0 → NO_ARTIFACT ABORT.
5. `RunResult` 에 `artifacts_dir: str = ""` 추가(기본값 있어 기존 생성자 호환). `orchestrator.run` 반환 dict 의 attempts 에 그대로 실린다.
6. CLI 실패 출력에 `error_class`·`reason`·log_tail 마지막 40줄이 나온다(capsys). `--quiet` 면 안 나온다.
7. `oauth_access_token()`: (a) 유효하면 그대로 (b) 만료+refresh_token → 몽키패치된 갱신 함수 호출 → 새 토큰 반환 + credentials.json 갱신(mode 0600, expiration 갱신) (c) 갱신 실패 → `""` + 기존 안내 (d) kagglesdk 없음 → `""`.
8. `errors.classify`: 커널 Traceback+TypeError 로그 → `ENTRYPOINT_ERROR`/ABORT; 같은 로그에 `CUDA out of memory` 가 있으면 OOM 우선; UNKNOWN cooldown_s == 300. 기존 test_errors 전부 통과.
9. `save_login_state` 가 TargetClosedError 에서 트레이스백 없이 `SystemExit(1)` + 안내 문자열(테스트는 playwright 를 몽키패치한 가짜 컨텍스트로).
10. 기존 테스트 86 노드 전부 통과 + 신규. `pytest tests/ -q` 는 python3(3.11, pytest 9.1 설치됨)로 실행.
11. 라이브 스모크 `examples/smoke_kaggle.yaml` 는 손대지 않는다(P100 경로). `examples/smoke_t4.yaml` 에 `workdir: examples/smoke_workdir`(작은 텍스트 파일 1개) 와 `success_glob: "artifacts/**/hello.txt"` 를 넣어 A·B 를 한 번에 검증할 수 있게 한다 — 이 파일 실행은 검증자(Fable)가 한다.

## 3. 파일

- `freecloud/providers/kaggle.py` — `_stage_workdir(job) -> (ds_slug|None, note)`, `_write_kernel_dir` 에 `dataset_sources` 배선, `_build_kernel_script(entrypoint, env, workdir_title=None)`, `_poll_and_fetch` 회수·success_glob·artifacts_dir, `oauth_access_token` refresh.
- `freecloud/providers/kaggle_ui.py` — 부모 헬퍼 재사용으로 자동 적용되는지 확인(`run()` 이 `_poll_and_fetch` 를 그대로 쓴다). `save_login_state` 예외 처리.
- `freecloud/providers/base.py` — `RunResult.artifacts_dir`.
- `freecloud/errors.py` — ENTRYPOINT_ERROR, UNKNOWN 쿨다운 300.
- `freecloud/cli.py` — 실패 상세 출력, `--quiet`, `logs` 서브커맨드.
- `README.md`, `docs/AUTH.md`, `examples/job.yaml`, `examples/smoke_t4.yaml`, `examples/smoke_workdir/hello.txt`(내용 `hello from workdir`), 그리고 smoke_t4 entrypoint 를 `cat hello.txt && cp hello.txt /kaggle/working/hello.txt && nvidia-smi -L` 로.
- `tests/test_kaggle_workdir.py`, `tests/test_kaggle_artifacts.py`, `tests/test_oauth_refresh.py`, `tests/test_errors.py`(추가 케이스), `tests/test_cli_output.py`, `tests/test_kaggle_login_closed.py`.

## 4. 하지 않는 것

- colab/lightning/modal/saturn 어댑터의 workdir 업로드(각자 다른 전송 방식; 이번엔 Kaggle 만). 단 `warn_unfetched` 는 유지.
- 실제 Kaggle 호출(전부 몽키패치). 라이브 스모크는 검증자가.
- Laibrio_AI 쪽 `run_fcc` 변경(HF 번들 경로는 그대로 유효; workdir 업로드로 갈아타는 건 다음 결정).
