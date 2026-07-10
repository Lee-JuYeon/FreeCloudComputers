# CI / 디스패처 워크플로

GitHub Actions로 쓰려면 이 폴더의 yml을 `.github/workflows/`로 옮기세요.
(자동 푸시가 안 된 이유: 현재 gh 토큰에 `workflow` 스코프가 없음.
 한 번만 `gh auth refresh -s workflow` 후 옮겨 커밋하면 활성화됩니다.)

- `ci.yml` — push/PR마다 pytest (py3.9/3.11/3.12)
- `dispatch.example.yml` — 무인 디스패처 예시(토큰 secrets 등록 필요)
