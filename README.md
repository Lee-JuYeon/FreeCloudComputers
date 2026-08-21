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

v0.1 — CLI 코어 + 4 어댑터 + 체크포인트 + MCP 래퍼 스캐폴드. provider별 라이브 검증은 진행 중.

## 라이선스

Apache-2.0

### Kaggle 인증 (CLI 2.2+ 에서 바뀌었다)

`kaggle` CLI 2.2.x 부터 인증 방식이 바뀌어 **예전 `~/.kaggle/kaggle.json`
(username+key) 만으로는 통과하지 않는다.** 파일이 있어도 `Authentication required`
로 거절된다(2026-08-21 실측: `kaggle.json`·`access_token` 이 둘 다 있는 머신에서도 거절).

```bash
kaggle auth login          # 권장 — 브라우저 OAuth, 1회. 토큰 자동 캐시
```

비대화형(CI 등)이면 [kaggle.com/settings/api](https://www.kaggle.com/settings/api) 에서
새 토큰을 발급해 둘 중 하나로 넣는다:

```bash
export KAGGLE_API_TOKEN=xxxxxxxx      # A) 환경변수
echo "xxxxxxxx" > ~/.kaggle/access_token   # B) 파일
```

`fcc auth` 는 이제 **실제 API 호출 결과까지 확인**한다. 예전에는 CLI 존재 여부만 보고
"인증 OK" 라 답한 뒤 `fcc run` 이 AUTH 로 죽는 거짓 양성이 있었다 — 수정됨.
