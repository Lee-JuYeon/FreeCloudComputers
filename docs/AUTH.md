# 인증 & 토큰 (로그인은 사이트별 딱 한 번)

로그인은 **두 종류뿐**입니다.

## ① 토큰 방식 (붙여넣기 1회 → 무인)
`huggingface`, `kaggle`, `modal`, `lightning`, `saturn`

사이트에서 토큰을 발급받아 한 번 등록하면 끝. 이후 자동.

```bash
freecloud login huggingface   # write 토큰 붙여넣기 → whoami로 검증 후 저장
freecloud login kaggle        # kaggle.json 안내 또는 username/key 입력
freecloud login modal         # 'modal token new'(브라우저 1회) 실행
freecloud login lightning     # API Key 저장 + env 안내
freecloud login saturn        # User Token 저장
```

## ② 브라우저 방식 (구글 로그인 등, headful 1회 → 쿠키 재사용)
`colab`, `kaggle-ui`(T4×2)

구글 OAuth/2FA는 **자동화 불가·금지**(계정 잠금·ToS 위반). 그래서 사람이 최초 1회
브라우저에서 직접 로그인 → 세션 쿠키(`storage_state`)를 저장 → 이후 무인 재사용.

```bash
freecloud login colab         # 브라우저 뜸 → 구글 로그인 → .freecloud/colab_state.json
freecloud login kaggle-ui     # 브라우저 뜸 → Kaggle 로그인 → .freecloud/kaggle_state.json
```

쿠키는 며칠~몇 주 뒤 만료됩니다. `freecloud auth`가 만료를 감지하면 다시 로그인하세요.

## 상태 확인
```bash
freecloud auth      # 모든 provider 인증 상태 + 저장된 시크릿 이름
```

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

## 보안 원칙
- **비밀번호는 저장하지 않음.** 토큰 또는 세션 쿠키만 저장.
- 저장 위치 `.freecloud/`(secrets.json, *_state.json)와 `.env`는 **gitignore** — 커밋 안 됨.
- env 변수가 있으면 그걸 우선 사용(저장소 파일보다). CI에선 GitHub Secrets 사용.
- Kaggle 커널에 토큰을 심으므로 **반드시 비공개 커널**(코드가 강제). 더 엄격히 하려면 Kaggle Secrets 사용.
- HF 토큰은 가능하면 **해당 체크포인트 repo로 스코프를 좁힌 fine-grained write 토큰** 권장.

## CI(무인)에서
토큰 방식 provider만 GitHub Actions에서 완전 무인 동작합니다. 브라우저 방식(colab/kaggle-ui)은
저장된 `storage_state`를 CI secret으로 넣어야 하며 만료 관리가 필요 → 제한적.
`.github/workflows/dispatch.example.yml` 참고.
