"""paths — 상태 파일 위치의 SSOT.

freecloud 가 디스크에 남기는 것은 성격이 둘로 갈린다:

  · **사용자 전역** (`~/.freecloud`) — 자격증명(secrets.json), 로그인 세션(kaggle_state.json,
    chrome_profile/), 그리고 **쿨다운(cooldown.json)**.
  · **프로젝트 로컬** (`./.freecloud`) — 실행 로그, 받아온 체크포인트 등 그 작업에 딸린 것.

쿨다운이 전역인 이유가 중요하다. "Kaggle 쿼터 소진"은 *프로젝트*가 아니라 **계정**의 속성이다.
이전처럼 cwd 상대로 두면 디렉토리 A 에서 쿼터를 소진하고 B 로 옮겨 실행할 때 쿨다운을 못 보고
이미 죽은 provider 를 다시 두드린다 — 페일오버 오케스트레이터의 존재 이유가 훼손된다.

과거에는 이 경로 계산식이 secrets/state/kaggle_ui/colab_login 네 곳에 복붙돼 있었고, 그래서
조용히 어긋났다. 여기 한 곳에서만 정한다.

env 오버라이드:
  FREECLOUD_HOME         — 사용자 전역 디렉토리 (테스트 격리·다계정 전환에 사용)
  FREECLOUD_PROJECT_DIR  — 프로젝트 로컬 디렉토리
"""
from __future__ import annotations

import os
import shutil

# 사용자 전역으로 옮겨야 하는 것들(마이그레이션 대상).
_USER_ARTIFACTS = ("secrets.json", "kaggle_state.json", "cooldown.json", "colab_state.json")
_USER_DIRS = ("chrome_profile",)


def user_home() -> str:
    """자격증명·로그인·쿨다운이 사는 곳. 기본 `~/.freecloud`.

    ⚠️ 모듈 임포트 시점에 상수로 굳히지 말 것 — 테스트가 env 로 격리하고, 사용자가 실행 중
    계정을 바꿀 수 있다. 항상 호출해서 쓴다.
    """
    return os.environ.get("FREECLOUD_HOME") or os.path.join(
        os.path.expanduser("~"), ".freecloud")


def project_dir() -> str:
    """이 작업 디렉토리에 딸린 산출물(로그·체크포인트)이 사는 곳. 기본 `./.freecloud`."""
    return os.environ.get("FREECLOUD_PROJECT_DIR") or os.path.join(
        os.getcwd(), ".freecloud")


def user_file(name: str) -> str:
    """사용자 전역 파일 경로(디렉토리는 만들지 않는다 — 쓰는 쪽에서 ensure)."""
    return os.path.join(user_home(), name)


def ensure_user_home() -> str:
    os.makedirs(user_home(), exist_ok=True)
    return user_home()


def migrate_legacy(verbose: bool = True) -> list[str]:
    """구버전(cwd 상대) 상태를 사용자 전역으로 1회 승격.

    복사만 하고 원본은 지우지 않는다 — 남의 자격증명을 말없이 삭제하는 것보다, 남겨두고
    알려주는 편이 안전하다. 이미 전역에 같은 이름이 있으면 건드리지 않는다(덮어쓰기 금지).

    반환: 옮긴 항목 이름 리스트(없으면 빈 리스트).
    """
    legacy = project_dir()
    home = user_home()
    if os.path.abspath(legacy) == os.path.abspath(home) or not os.path.isdir(legacy):
        return []

    moved: list[str] = []
    for name in _USER_ARTIFACTS:
        src, dst = os.path.join(legacy, name), os.path.join(home, name)
        if os.path.isfile(src) and not os.path.exists(dst):
            os.makedirs(home, exist_ok=True)
            shutil.copy2(src, dst)
            moved.append(name)
    for name in _USER_DIRS:
        src, dst = os.path.join(legacy, name), os.path.join(home, name)
        if os.path.isdir(src) and not os.path.exists(dst):
            os.makedirs(home, exist_ok=True)
            shutil.copytree(src, dst)
            moved.append(name + "/")

    if moved and verbose:
        print(f"[freecloud] 기존 상태를 사용자 전역으로 옮겼습니다: {', '.join(moved)}")
        print(f"[freecloud]   {legacy}  ->  {home}")
        print(f"[freecloud] 이제 어느 디렉토리에서 실행해도 같은 로그인·쿨다운을 씁니다.")
        print(f"[freecloud] 원본은 남겨뒀습니다 — 확인 후 지우셔도 됩니다.")
    return moved
