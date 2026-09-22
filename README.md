# FreeCloudComputers (`freecloud`)

무료(비중국) 클라우드 GPU를 **하나의 풀로 묶어**, 한 곳의 할당량이 소진되면 **자동으로 다음
provider에서 이어받아** 작업을 끝내는 오픈소스 오케스트레이터.

> 핵심 원칙 하나가 전부를 좌우한다:
> **컴퓨트 노드는 무상태. 상태(체크포인트)는 외부 공유 스토리지에만 산다.**
> 그래서 Kaggle에서 쿼터가 끝나 죽어도 Lightning/Colab/Modal이 *마지막 체크포인트부터* 이어받는다.

## 왜?

Kaggle 30h/주, Colab 15~30h/주, Lightning 80h/월, Modal $30/월 … 각각은 금방 바닥난다.
하지만 **합치면** 유효 GPU 시간이 크게 늘어난다. `freecloud`는 이걸 무인(unattended)으로 돌린다.

## 설치

```bash
pip install -e .            # CLI 코어
pip install -e '.[hub]'     # + 체크포인트 동기화(HF Hub) — 재개하려면 필요
pip install -e '.[mcp]'     # + Claude/dev-hub 제어용 MCP 래퍼(선택)
```

## 빠른 시작

```bash
freecloud clouds              # 지원/후보 무료 클라우드 카탈로그
freecloud login huggingface   # 체크포인트 저장소 토큰(재개에 필요) — 자세히는 docs/AUTH.md
freecloud login kaggle        # 사이트별 로그인은 처음 한 번만
freecloud auth                # 인증 상태 한눈에(계정/토큰/세션)
freecloud providers           # 각 provider 가용 상태
freecloud run examples/job.yaml
freecloud logs examples/job.yaml   # 커널 상태 + diag/로그 꼬리 + 산출물 회수
freecloud status              # provider별 쿨다운 스냅샷
```

## 로그인/토큰 (처음 한 번만)

로그인은 두 종류뿐 — **토큰 붙여넣기**(huggingface/kaggle/modal/lightning/saturn)와
**브라우저 로그인**(colab/kaggle-ui, 구글 OAuth는 자동화 불가라 headful 1회 → 쿠키 재사용).
등록만 하면 **freecloud가 HF 토큰을 Kaggle 커널 등 원격 노드에 자동 주입**하므로 job.yaml에
토큰을 적지 않아도 체크포인트가 동작합니다. 상세 → **[docs/AUTH.md](docs/AUTH.md)**.

`job.yaml` 한 장으로 정의(→ `examples/job.yaml`):

```yaml
name: demo-lora
entrypoint: "python train_lora.py"
checkpoint_repo: "your-hf-username/demo-lora-ckpt"   # ★ 재개의 핵심
needs: { gpu: T4, min_vram_gb: 16, headless: true }
max_runtime_s: 5400
providers: [lightning, kaggle, colab, modal]
```

## 코드 업로드(`workdir`)와 산출물 회수(`artifacts/`)

`job.yaml` 에 **`workdir` 를 적으면**(그리고 `repo` 가 비어 있으면), **Kaggle 계열은 그
디렉토리를 private 데이터셋으로 올려** 노드에서 `/kaggle/working/workdir` 로 풀고 거기서
`entrypoint` 를 실행합니다(`FREECLOUD_WORKDIR` env 로도 경로를 줍니다). 그래서
`entrypoint: "python train_lora.py"` 같은 평범한 명령이 그대로 돕니다 — 예전에는 생성된
`kernel.py` 한 장만 올라가서 죽었습니다.

> **`workdir` 는 opt-in 입니다.** 키를 적지 않으면(기본값 `""`) 아무것도 올리지 않고
> 커널도 예전과 **바이트 단위로 똑같이** 만들어집니다. 코드는 `entrypoint` 가 알아서
> 확보한다는 뜻입니다(`git clone`, HF Hub 다운로드 등). 예전 기본값은 `"."` 였는데,
> 그러면 "안 적음"과 "cwd 를 올려라"를 구분할 수 없어 workdir 없는 job 의 커널까지
> 붙지도 않은 입력을 복사하려다 `FileNotFoundError` 로 죽었습니다.

- **항상 제외**: `.git/`, `__pycache__/`, `*.pyc`, **`.env`**(시크릿 유출 방지).
- 추가 제외: `FREECLOUD_WORKDIR_IGNORE="*.bin,data,artifacts"`(쉼표 glob).
- 크기 상한: `FREECLOUD_WORKDIR_MAX_MB`(기본 500). 초과하면 **업로드 전에** JOB_CONFIG 로
  즉시 실패합니다(쿼터를 태우지 않음). `0` 이면 업로드 자체를 끕니다.
- ⚠️ `workdir` 는 **실행한 디렉토리 기준**으로 해석됩니다(`Job.load` 가 abspath).
- 업로드(또는 데이터셋 처리)가 실패하면 **커널을 push 하지 않고** JOB_CONFIG 로 중단합니다.
  push 해버리면 코드 없는 노드에서 돌거나, 페일오버해서 조용히 옛 코드로 학습합니다.

실행이 complete 든 error 든 `kaggle kernels output` 을 **`./artifacts/<job.name>/`** 로
받아 둡니다(`FREECLOUD_ARTIFACTS_DIR` 로 루트 변경). 실패해도 커널 Traceback 이 담긴
`diag.txt` 가 손에 남습니다.

- `job.artifacts` — Kaggle 에서는 "회수본에 이 파일이 있어야 한다" 목록으로 읽고, 없으면 경고.
- `job.success_glob` — 회수 디렉토리(그리고 cwd) 기준으로 평가. 커널이 complete 인데
  매치가 0이면 `NO_ARTIFACT` 로 **중단**합니다(페일오버 아님 — 코드가 파일을 안 만든 것이라
  다른 provider 로 가도 똑같이 빈손입니다).

실패하면 CLI 가 provider·status·`error_class`·reason 과 **로그 꼬리 40줄**을 찍습니다
(`fcc run --quiet` 로 끄기). 나중에 다시 보려면 `fcc logs <job.yaml|이름>`.

## 아키텍처

```
job.yaml ─► orchestrator ─► [provider 순회: probe→run→성공?]
               │                    │
               │                    └─ 실패 분류(errors) → RETRY / FAILOVER(쿨다운) / ABORT
               │
               └─ 상태는 provider에 없음 ─► checkpoint(HF Hub) ◄─ 모든 노드가 pull/push
```

| 모듈 | 역할 |
|---|---|
| `providers/base.py` | 어댑터 인터페이스(`probe`/`run`/`fits`). 새 클라우드 = 파일 하나 + registry 한 줄 |
| `providers/{kaggle,colab,lightning,modal}.py` | 4개 구현. Kaggle/Colab은 검증된 페일오버 로직 이식 |
| `providers/kaggle_ui.py` | T4×2 전용 — Playwright로 UI 자동화(opt-in, `kaggle-login` 선행) |
| `providers/saturn.py` | Saturn Cloud(saturn-client recipe) |
| `checkpoint.py` | 재개의 린치핀. HF Hub에 상태 push/pull |
| `auth.py` / `secrets.py` | 통합 로그인/인증 상태 + 시크릿 저장·원격 자동주입 |
| `errors.py` | 중앙 오류 플레이북(단일 SSOT). 오류→행동 매핑 |
| `state.py` | 쿨다운 영속화(죽은 provider 계속 두드리지 않게) |
| `orchestrator.py` | 페일오버 루프 |
| `cli.py` / `mcp_server.py` | CLI 코어 / 선택적 MCP 제어 표면 |

### 재개 계약 (entrypoint가 지켜야 할 것)

원격에서 실행되는 당신의 `entrypoint`는:
1. 시작 시 `Checkpoint.from_env().pull(dir)` — 이전 노드 상태 복원
2. 매 N스텝 `.push(dir, step=step)` — 공유 저장소에 저장

→ `examples/train_lora.py`가 최소 형태. transformers/peft로 바꿔도 구조는 동일.

## 지원 & 후보 클라우드 (2026-07, 비중국)

`freecloud clouds` 출력과 동일. 값은 변동 — 갱신 PR 환영.

| | 클라우드 | 무료 GPU | 한도 | 카드 |
|---|---|---|---|---|
| ✓ | Kaggle | P100 / T4×2 / TPU v5e-8 | ~30h/주 | X |
| ✓ | Colab | T4 16GB | ~15–30h/주 | X |
| ✓ | Lightning AI | L4/T4 | 15크레딧/월 ≈ 80h | X |
| ✓ | Modal | T4/L4/A10G | ~$30/월 | X |
| ✓ | Saturn Cloud | T4 | 반복 무료 | X |
| · | SageMaker Studio Lab | T4 | 4h/세션 · ⚠️ 신규가입 2026-07-30 마감 | X |
| · | Intel Tiber AI Cloud | Gaudi2 / GPU Max | 무료 | X |
| · | HF Spaces ZeroGPU | H200(버스트) | ~5분/일 (데모용) | X |
| · | Paperspace(DO) | M4000 8GB | 6h | X |
| · | Google TRC | TPU 1000+대 | 신청제(오픈소스 조건) | GCP |
| ⚙ | Oracle Always Free | ARM A1 | 영구(컨트롤 플레인) | 필요 |
| ⚙ | GitHub Actions | CPU | 2000분/월(디스패처) | X |

### Kaggle T4×2 — provider `kaggle-ui` (Playwright 자동화)
Kaggle **API/CLI는 단일 GPU(P100)만** 노출한다. **T4×2는 웹 UI 전용**(Settings→Accelerator→
GPU T4 x2→Save & Run All). `freecloud`는 이걸 `kaggle-ui` provider로 자동화한다:

```bash
pip install -e '.[kaggle-ui]' && playwright install chromium
freecloud kaggle-login          # 최초 1회: 브라우저 로그인(2FA 포함) → 세션 저장
# ⚠️ 뜬 브라우저 창을 직접 닫지 말 것 — 로그인이 감지되면 fcc 가 알아서 닫는다.
#    감지 전에 닫으면 세션이 저장되지 않는다([FAIL] 안내 후 rc 1).
# job.yaml 의 providers 에 kaggle-ui 추가 → 무인 T4×2
```

동작: 코드 push → Playwright가 로그인 상태로 에디터를 열어 T4×2 선택 + Save & Run All →
커널 안 `nvidia-smi -L` 로 **실제 T4×2 붙었는지 검증**(불일치면 `needs_setup` 반환).
셀렉터는 Kaggle 에디터 DOM 기준 best-effort — UI 바뀌면 `providers/kaggle_ui.py` 조정.
디버그: `FREECLOUD_KAGGLE_HEADFUL=1` 로 브라우저 띄워 확인.

> UI에서 T4×2가 회색이면(수동 점검): ①전화번호 인증 ②주간 쿼터 소진 ③남의 노트북
> Viewer(→Copy&Edit) ④세션 running 중(→Stop).

## 새 provider 추가

`providers/base.py`의 `Provider`를 상속해 `probe()`/`run()` 구현 → `registry.py`에 한 줄 등록.
`run()`은 **어떤 경로로 끝나도 원격 세션을 stop**(고아 방지)하고, 실패 시 로그를 `RunResult`에 담아
반환한다(예외 던지지 않음). Colab 어댑터의 `finally` 블록이 참고 패턴.

## 상태

v0.1 — CLI 코어 + 6 어댑터 + 체크포인트 + MCP 래퍼 스캐폴드. provider별 라이브 검증은 진행 중.
2026-09-23: Kaggle 계열에 `workdir` 업로드·산출물 회수·실패 로그 출력·OAuth 자동 갱신 추가
(설계 = `docs/IMPROVEMENTS_2026-09-23.md`). workdir 업로드는 아직 Kaggle 계열만 —
colab/lightning/modal/saturn 은 `warn_unfetched` 로 "회수 못 한다"고 말만 한다.

## 라이선스

Apache-2.0

### Kaggle 인증 (CLI 2.2+ 에서 바뀌었다)

`kaggle` CLI 2.2.x 부터 인증 방식이 바뀌어 **예전 `~/.kaggle/kaggle.json`
(username+key) 만으로는 통과하지 않는다.** 파일이 있어도 `Authentication required`
로 거절된다(2026-08-21 실측: `kaggle.json`·`access_token` 이 둘 다 있는 머신에서도 거절).

```bash
kaggle auth login          # 권장 — 브라우저 OAuth, 1회
```

> **알아둘 것**: `kaggle auth login` 은 `~/.kaggle/credentials.json` 을 만들지만
> **CLI 2.2.x 는 이 파일을 `kernels` 계열 명령에서 자동으로 읽지 않는다.**
> 로그인 직후에도 `Authentication required` 가 뜬다(2026-08-21 실측, CLI 2.2.4에서도 동일).
> `freecloud` 는 이 파일의 `access_token` 을 `KAGGLE_API_TOKEN` 으로 자동 주입해 메운다 —
> 사용자가 손으로 export 할 필요 없다.

#### access token 수명은 3시간 — fcc 가 자동으로 갱신한다

`credentials.json` 의 `access_token` 은 **3시간**이면 만료된다. 예전에는 학습을 걸 때마다
사람이 `kaggle auth login --force` 를 쳐야 했다. 이제 fcc 가 만료(또는 5분 이내 만료)를
감지하면 같은 파일의 `refresh_token` 으로 **새 토큰을 받아 파일을 원자적으로 갱신**한다
(`[kaggle] access token 자동 갱신됨` 한 줄이 찍힌다). 엔드포인트는 하드코딩하지 않고
`kagglesdk` 의 갱신 경로를 빌린다.

갱신이 실패하면(=refresh_token 도 만료) 재로그인이 필요하다:

```bash
kaggle auth login --force     # ★ --force 필수
```

> `--force` 없이 `kaggle auth login` 을 치면 CLI 가 **만료된 토큰을 보고도 "이미 로그인됨"
> 이라 답하고 그냥 끝난다.** 그래서 아무리 다시 쳐도 인증이 안 풀리는 것처럼 보인다.

비대화형(CI 등)이면 [kaggle.com/settings/api](https://www.kaggle.com/settings/api) 에서
새 토큰을 발급해 둘 중 하나로 넣는다:

```bash
export KAGGLE_API_TOKEN=xxxxxxxx      # A) 환경변수
echo "xxxxxxxx" > ~/.kaggle/access_token   # B) 파일
```

`fcc auth` 는 이제 **실제 API 호출 결과까지 확인**한다. 예전에는 CLI 존재 여부만 보고
"인증 OK" 라 답한 뒤 `fcc run` 이 AUTH 로 죽는 거짓 양성이 있었다 — 수정됨.
