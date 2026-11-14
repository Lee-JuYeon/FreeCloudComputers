import os
import tempfile

import pytest

# 테스트는 실제 홈이 아닌 임시 .freecloud 를 쓰도록 격리(쿨다운/시크릿 파일 오염 방지).
os.environ.setdefault("FREECLOUD_HOME", tempfile.mkdtemp(prefix="fc-test-home-"))
os.environ.setdefault("PYTHONUTF8", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_hf: latest_checkpoint_commit 블록(_block_hf_checkpoint)을 건너뛴다")


@pytest.fixture(autouse=True)
def _block_hf_checkpoint(request, monkeypatch):
    """무진행 워치독·heartbeat 가 실제 HF 를 두드리지 못하게 막는다(네트워크 0).

    기본값 None = '커밋 없음/조회 실패' → 워치독은 '모름(정지 아님)'으로 다루고 heartbeat 는
    'ckpt 조회 실패/없음' 으로 표시한다. 워치독 동작을 검증하는 테스트는 이 함수를 자체
    monkeypatch 로 다시 덮어쓴다(나중 setattr 가 이긴다). 실제 함수 경로를 테스트하는 쪽은
    `@pytest.mark.allow_hf` 로 블록을 건너뛴다(가짜 HfApi 를 직접 심는다)."""
    if request.node.get_closest_marker("allow_hf"):
        return
    from freecloud.providers import kaggle as _K
    monkeypatch.setattr(_K, "latest_checkpoint_commit", lambda repo: None, raising=False)
