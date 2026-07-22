# CI / 디스패처 워크플로

## 활성화됨

`ci.yml` 은 **`.github/workflows/ci.yml` 로 이동해 가동 중**이다 (2026-07-22).
push(main)/PR 마다 pytest 를 Python 3.9 / 3.11 / 3.12 로 돌린다.

> 이전에는 gh 토큰에 `workflow` 스코프가 없어 이 폴더에 묶여 있었다.
> `gh auth refresh -h github.com -s workflow` 로 권한을 추가하고 옮겼다.

## 여기 남은 것

- `dispatch.example.yml` — 무인 디스패처 **예시**. 일부러 활성화하지 않았다.
  `.github/workflows/` 로 옮기면 GitHub 이 실제 워크플로로 인식하는데, 안에
  `schedule: cron "0 */6 * * *"` 가 있어 **시크릿 없이 6시간마다 실행돼 계속 실패**한다.
  쓰려면 먼저 아래 secrets 를 레포 Settings→Secrets 에 등록하고, 그 다음
  `.github/workflows/dispatch.yml` 로 복사할 것:

  ```
  HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY, MODAL_TOKEN_ID, MODAL_TOKEN_SECRET,
  LIGHTNING_USER_ID, LIGHTNING_API_KEY, LIGHTNING_TEAMSPACE, SATURN_TOKEN
  ```

  ⚠️ 토큰 방식 provider(kaggle/modal/lightning/saturn/HF)만 CI 에서 무인 동작한다.
  브라우저 방식(colab/kaggle-ui)은 저장된 storage_state 가 필요해 CI 에선 제한적이다.
