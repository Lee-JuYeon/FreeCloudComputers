"""paths — 사용자 전역/프로젝트 로컬 분리 회귀 테스트.

과거 버그: 경로가 `os.getcwd()/.freecloud` 로 계산되고 그 식이 네 모듈에 복붙돼 있었다.
그래서 (a) 실행 디렉토리를 옮기면 로그인·시크릿을 못 찾고, (b) **쿨다운이 프로젝트마다
갈라져** 이미 쿼터가 소진된 provider 를 다시 두드렸다(페일오버의 존재 이유가 훼손).
"""
import os

import pytest

from freecloud import paths


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "home"
    monkeypatch.setenv("FREECLOUD_HOME", str(h))
    return h


def test_user_home_은_env를_존중한다(home):
    assert paths.user_home() == str(home)


def test_user_home은_cwd가_바뀌어도_고정된다(home, tmp_path, monkeypatch):
    """핵심 회귀: 실행 디렉토리를 옮겨도 같은 자격증명·쿨다운을 봐야 한다."""
    a, b = tmp_path / "projA", tmp_path / "projB"
    a.mkdir(); b.mkdir()

    monkeypatch.chdir(a)
    first = paths.user_home()
    monkeypatch.chdir(b)
    assert paths.user_home() == first


def test_project_dir은_cwd를_따라간다(tmp_path, monkeypatch):
    """반대로 로그·산출물은 그 작업 디렉토리에 남는 게 맞다."""
    monkeypatch.delenv("FREECLOUD_PROJECT_DIR", raising=False)
    a, b = tmp_path / "projA", tmp_path / "projB"
    a.mkdir(); b.mkdir()

    monkeypatch.chdir(a)
    first = paths.project_dir()
    monkeypatch.chdir(b)
    assert paths.project_dir() != first


def test_쿨다운은_디렉토리가_달라도_공유된다(home, tmp_path, monkeypatch):
    """쿼터 소진은 프로젝트가 아니라 계정의 속성이다."""
    from freecloud import state

    a, b = tmp_path / "projA", tmp_path / "projB"
    a.mkdir(); b.mkdir()

    monkeypatch.chdir(a)
    state.set_cooldown("kaggle", 3600, reason="쿼터 소진")

    monkeypatch.chdir(b)
    assert state.cooldown_remaining("kaggle") > 0, "다른 디렉토리에서 쿨다운을 못 봄"


def test_시크릿도_디렉토리를_안_탄다(home, tmp_path, monkeypatch):
    from freecloud import secrets

    a, b = tmp_path / "projA", tmp_path / "projB"
    a.mkdir(); b.mkdir()

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(a)
    secrets.set("HF_TOKEN", "hf_dummy")

    monkeypatch.chdir(b)
    assert secrets.get("HF_TOKEN") == "hf_dummy"


def test_migrate_legacy는_구경로_자격증명을_승격한다(home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    legacy = proj / ".freecloud"
    legacy.mkdir(parents=True)
    (legacy / "secrets.json").write_text('{"HF_TOKEN": "old"}', encoding="utf-8")
    (legacy / "kaggle_state.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(proj)

    moved = paths.migrate_legacy(verbose=False)

    assert set(moved) == {"secrets.json", "kaggle_state.json"}
    assert os.path.isfile(home / "secrets.json")
    # 원본은 지우지 않는다 — 남의 자격증명을 말없이 삭제하지 않는다.
    assert os.path.isfile(legacy / "secrets.json")


def test_migrate_legacy는_기존_전역파일을_덮어쓰지_않는다(home, tmp_path, monkeypatch):
    home.mkdir(parents=True)
    (home / "secrets.json").write_text('{"HF_TOKEN": "current"}', encoding="utf-8")

    proj = tmp_path / "proj"
    legacy = proj / ".freecloud"
    legacy.mkdir(parents=True)
    (legacy / "secrets.json").write_text('{"HF_TOKEN": "stale"}', encoding="utf-8")
    monkeypatch.chdir(proj)

    assert paths.migrate_legacy(verbose=False) == []
    assert "current" in (home / "secrets.json").read_text(encoding="utf-8")


def test_migrate_legacy는_옮길게_없으면_조용하다(home, tmp_path, monkeypatch):
    proj = tmp_path / "proj"
    proj.mkdir()
    monkeypatch.chdir(proj)
    assert paths.migrate_legacy(verbose=False) == []
