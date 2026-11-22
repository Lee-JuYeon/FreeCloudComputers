# 인증 & 토큰 (로그인은 사이트별 딱 한 번)

로그인은 **세 종류**입니다.

## ① 붙여넣기 방식 (토큰 1회 등록 → 무인)
`huggingface`, `lightning`, `saturn`

사이트에서 토큰을 발급받아 한 번 등록하면 끝. 이후 자동. 값은 `~/.freecloud/secrets.json` 에
저장됩니다. 필요한 값이 모두 환경변수에 있고 검증을 통과하면 묻지 않고 그 값을 저장합니다(무인 경로). 일부만 있으면 없는 항목만 묻고, 값이 무효면 새 값을 묻습니다.

```bash
freecloud login huggingface   # write 토큰 붙여넣기 → whoami로 검증 후 저장
freecloud login lightning     # API Key·User ID·Teamspace 3종 → 검증 후 저장
freecloud login saturn        # User Token 저장
```

## ② 공식 CLI 브라우저 로그인 (그 도구의 로그인 명령 1회)
`kaggle`, `modal`

fcc 가 그 도구의 로그인 명령을 대신 돌리고, 끝나면 실제 API 호출로 통과 여부를 확인합니다.
자격증명은 그 도구 자체 저장소(`~/.kaggle/credentials.json`, modal 설정)에 남습니다.

```bash
freecloud login kaggle        # 브라우저 OAuth 1회 → API 호출로 통과 확인(낡은 kaggle.json 은 OAuth 를 가리므로 .bak 으로 치울지 묻는다)
freecloud login modal         # 'modal token new'(브라우저 1회) 실행 → 통과 확인
```

비대화형(CI·파이프)에서는 브라우저를 못 띄우므로 환경변수로 넘깁니다.

```bash
KAGGLE_API_TOKEN=... freecloud login kaggle    # 실제 API 호출로 검증된 뒤에만 secrets.json 에 저장
MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... freecloud login modal   # 확인만 하고 저장하지 않음(modal 이 env 를 직접 읽는다)
```

## ③ 브라우저 쿠키 방식 (구글 로그인 등, headful 1회 → 쿠키 재사용)
`colab`, `kaggle-ui`(T4×2)

구글 OAuth/2FA는 **자동화 불가·금지**(계정 잠금·ToS 위반). 그래서 사람이 최초 1회
브라우저에서 직접 로그인 → 세션 쿠키(`storage_state`)를 저장 → 이후 무인 재사용.

```bash
freecloud login colab         # 브라우저 뜸 → 구글 로그인 → ~/.freecloud/colab_state.json
freecloud login kaggle-ui     # 브라우저 뜸 → Kaggle 로그인 → ~/.freecloud/kaggle_state.json
```

> ⚠️ **브라우저 창을 직접 닫지 마세요.** 로그인이 감지되면 fcc 가 알아서 닫습니다.
> 감지 전에 닫으면 세션이 저장되지 않고 `[FAIL] 브라우저가 로그인 감지 전에 닫혔습니다`
> 로 끝납니다(rc 1) — 다시 실행하면 됩니다.

쿠키는 며칠~몇 주 뒤 만료됩니다. `freecloud auth`가 만료를 감지하면 다시 로그인하세요.

> 🔑 **kaggle-ui 쿠키 수명 = 14일**(2026-10-24 실측). Kaggle 웹 세션 쿠키
> `__Host-KAGGLEID`가 14일이면 만료됩니다. fcc 는 이 쿠키의 만료일을 파일에서 바로 읽어
> (`fcc auth`/`fcc providers` 줄에 `만료 YYYY-MM-DD(N일 남음)` 표시) 만료되었거나 2일 이내면
> **LOGIN_REQUIRED**(ABORT)로 즉시 끝냅니다 — 페일오버·재시도로 쿼터를 태우지 않습니다.
> 조치는 `freecloud kaggle-login` 한 번(약 1분, 브라우저). ⚠️ 과거 버전은 수명 30일인
> `CLIENT-TOKEN`(JWT)만 봐서 16일 된 세션을 "로그인됨"으로 통과시켜 런타임에 404로 죽었습니다.
> kaggle CLI OAuth(`~/.kaggle/credentials.json`)는 이와 **별개**입니다. access token(수명 3시간)은
> 실행 시 `refresh_token` 으로 자동 갱신되지만, `refresh_token` 까지 만료되어 갱신이 실패하면
> `freecloud login kaggle` 로 다시 로그인해야 합니다.

## 상태 확인
```bash
freecloud auth      # 모든 provider 인증 상태 + 저장된 시크릿 이름
freecloud login     # (인자 없음) 같은 상태표 + 미인증인 것만 차례로 "지금 로그인할까요? [y/N]"
```

인자 없는 `freecloud login` 은 미인증 provider 를 하나씩 묻습니다(kaggle-ui·colab 포함).
비대화형(파이프·CI)에서는 표만 찍고 한 번도 묻지 않으며, 미인증이 있으면 종료코드 1 입니다.
CLI·SDK 가 **설치되지 않은** provider 는 `··` 로 따로 표시하고(설치 명령 안내), 묻지도 않고
종료코드에도 세지 않습니다 — 로그인으로 풀 수 있는 문제가 아니기 때문입니다.

---

## Hugging Face 토큰 (가장 중요)

**왜 필요한가?** 작업이 Kaggle에서 죽으면 Colab이 *마지막 체크포인트부터* 이어받아야 하는데,
그 중간 저장창고가 HF Hub입니다. 거기에 저장하려면 **write 토큰**이 필요합니다.

- **발급**: https://huggingface.co/settings/tokens → Type: **Write** (사람이 클릭 — 자동화 불가).
- **등록**: `freecloud login huggingface` 한 번. `whoami()`로 계정/토큰 유효성을 자동 검증.
- **원격 주입(핵심)**: 등록만 하면 **freecloud가 알아서** Kaggle 커널·Colab·Modal·Lightning의
  원격 env로 `HF_TOKEN`을 실어보냅니다. `job.yaml`에 토큰을 적지 않아도 체크포인트가 동작.
  (job에 `checkpoint_repo`가 있으면 자동, 또는 `secrets: [NAME]`으로 추가 지정.)

### provider별 원격 주입 방식
| provider | HF_TOKEN 주입 경로 |
|---|---|
| kaggle | 커널을 **비공개(is_private)**로 push + env를 커널 스크립트에 주입. (수동 대안: Kaggle Secrets) |
| modal | 함수 env로 주입. (수동 대안: `modal secret create`) |
| colab / lightning / saturn | 실행 command 앞에 `export`로 주입 |

## 저장 위치 (`~/.freecloud`)

자격증명·로그인 세션·쿨다운은 **사용자 전역** `~/.freecloud/` 에 삽니다(`FREECLOUD_HOME`
으로 변경). 예전에는 실행 디렉토리 상대(`./.freecloud/`)였는데, 그러면 디렉토리 A 에서
쿼터를 소진하고 B 에서 실행할 때 쿨다운을 못 봅니다 — "Kaggle 쿼터 소진"은 프로젝트가
아니라 **계정**의 속성이니까요. 자세한 이유는 `freecloud/paths.py` 주석.

| 경로 | 내용 |
|---|---|
| `~/.freecloud/secrets.json` | 등록한 토큰 — HF, Lightning 3종(API Key·User ID·Teamspace), Saturn, 비대화형으로 넣은 Kaggle 토큰 |
| `~/.freecloud/kaggle_state.json`, `colab_state.json` | 브라우저 로그인 세션 |
| `~/.freecloud/chrome_profile/` | kaggle-login 이 쓰는 실제 Chrome 프로필 |
| `~/.freecloud/cooldown.json` | provider별 쿨다운 |
| `~/.kaggle/credentials.json` | `freecloud login kaggle` 이 만드는 OAuth 토큰(fcc 가 자동 갱신) |
| `./artifacts/<job.name>/` | 회수한 산출물(`FREECLOUD_ARTIFACTS_DIR` 로 변경) |

## 보안 원칙
- **비밀번호는 저장하지 않음.** 토큰 또는 세션 쿠키만 저장.
- 저장 위치 `~/.freecloud/`(secrets.json, *_state.json)와 `.env`는 **gitignore** — 커밋 안 됨.
- `workdir` 업로드는 **opt-in**(job.yaml 에 키를 적었을 때만)이고, `.env`·`.git`·
  `__pycache__`·`*.pyc` 를 **항상** 제외한다 — 시크릿이 데이터셋으로 새어나가지 않게.
  추가 제외는 `FREECLOUD_WORKDIR_IGNORE`(쉼표 glob).
- env 변수가 있으면 그걸 우선 사용(저장소 파일보다). CI에선 GitHub Secrets 사용.
- Kaggle 커널에 토큰을 심으므로 **반드시 비공개 커널**(코드가 강제). 더 엄격히 하려면 Kaggle Secrets 사용.
- HF 토큰은 가능하면 **해당 체크포인트 repo로 스코프를 좁힌 fine-grained write 토큰** 권장.

## CI(무인)에서
환경변수로 자격증명을 넘길 수 있는 provider만 GitHub Actions에서 완전 무인 동작합니다.
env 가 저장소 파일보다 우선하므로, CI secret 을 아래 이름으로 넣으면 로그인 단계 없이 돕니다.

| provider | 환경변수 |
|---|---|
| huggingface | `HF_TOKEN` |
| kaggle | `KAGGLE_API_TOKEN` |
| lightning | `LIGHTNING_API_KEY`, `LIGHTNING_USER_ID`, `LIGHTNING_TEAMSPACE` |
| modal | `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` — **env 로만** 동작(fcc 는 저장하지 않는다) |
| saturn | `SATURN_TOKEN` |

브라우저 쿠키 방식(colab/kaggle-ui)은
저장된 `storage_state`를 CI secret으로 넣어야 하며 만료 관리가 필요 → 제한적.
`.github/workflows/dispatch.example.yml` 참고.
