"""checkpoint — resume 판정 회귀 테스트.

과거 버그: `pull()` 이 `any(os.scandir(dest))` 로 재개 여부를 정했다. HF 는 저장소 생성 시
`.gitattributes` 를 자동으로 넣고 snapshot_download 는 `.cache/` 를 남기므로, **한 번도
push 한 적 없는 빈 저장소도 resume=True** 로 보고됐다. 학습 스크립트가 이걸로 scratch/resume
을 분기하면 처음 실행인데 재개 경로를 탄다.
"""
from freecloud.checkpoint import Checkpoint, _has_payload, _is_auth_error


def test_빈_저장소는_재개가_아니다(tmp_path):
    """HF 가 자동 생성하는 메타데이터만 있는 상태 = scratch."""
    (tmp_path / ".gitattributes").write_text("*.bin filter=lfs\n", encoding="utf-8")
    (tmp_path / ".cache").mkdir()
    (tmp_path / "README.md").write_text("# repo", encoding="utf-8")

    assert _has_payload(str(tmp_path)) is False


def test_실제_체크포인트가_있으면_재개다(tmp_path):
    (tmp_path / ".gitattributes").write_text("x", encoding="utf-8")
    (tmp_path / "state.txt").write_text("42", encoding="utf-8")

    assert _has_payload(str(tmp_path)) is True


def test_하위디렉토리_체크포인트도_재개로_친다(tmp_path):
    (tmp_path / "ckpt-1000").mkdir()

    assert _has_payload(str(tmp_path)) is True


def test_완전히_빈_디렉토리는_재개가_아니다(tmp_path):
    assert _has_payload(str(tmp_path)) is False


def test_토큰은_저장소로_폴백한다(monkeypatch):
    """env 를 안 쓰고 `fcc login` 으로만 저장한 사용자도 로컬에서 토큰을 찾아야 한다."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    from freecloud import secrets
    monkeypatch.setattr(secrets, "get",
                        lambda n: "hf_from_store" if n == "HF_TOKEN" else None)

    assert Checkpoint("u/r").token == "hf_from_store"


def test_env가_저장소보다_우선한다(monkeypatch):
    """원격 노드는 resolved_env 로 주입받으므로 env 가 이겨야 한다."""
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    assert Checkpoint("u/r").token == "hf_from_env"


def test_401은_인증오류로_분류된다():
    """체크포인트 유실 오해를 막는 안내로 이어지는 분기."""
    assert _is_auth_error(Exception("Client error '401 Unauthorized' for url ..."))
    assert _is_auth_error(Exception("Invalid user token."))
    assert not _is_auth_error(Exception("connection reset by peer"))
