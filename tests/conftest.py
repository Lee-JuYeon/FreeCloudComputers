import os
import tempfile

import pytest

# 테스트는 실제 홈이 아닌 임시 .freecloud 를 쓰도록 격리(쿨다운/시크릿 파일 오염 방지).
os.environ.setdefault("FREECLOUD_HOME", tempfile.mkdtemp(prefix="fc-test-home-"))
os.environ.setdefault("PYTHONUTF8", "1")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_hf: latest_checkpoint_commit 블록(_block_hf_checkpoint)을 건너뛴다")
    config.addinivalue_line(
        "markers", "real_stop: stop_running_session 기본 차단을 건너뛴다(가짜 playwright 로 실함수 검증)")
    config.addinivalue_line(
        "markers", "allow_hf_storage: hf_storage._api 기본 차단을 건너뛴다(가짜 HfApi 를 직접 심는다)")


@pytest.fixture(autouse=True)
def _isolate_kaggle_session(request, monkeypatch, tmp_path):
    """kaggle 로그인 세션을 테스트별로 격리하고 실수로 **진짜 브라우저가 뜨는 것**을 막는다.

    · FREECLOUD_KAGGLE_STATE 를 테스트별 tmp 로 지정 — 세션공유 임시홈의 kaggle_state.json 을
      한 테스트가 써두면 뒤 테스트의 진짜 stop_kernel 이 그 쿠키로 Chromium 을 띄우려 했다.
    · kaggle_ui.stop_running_session 을 기본 차단(Playwright 기동 0 보장). 실제 위임을 검증하는
      테스트는 자체 monkeypatch 로 덮어쓴다(나중 setattr 가 이긴다)."""
    monkeypatch.setenv("FREECLOUD_KAGGLE_STATE", str(tmp_path / "kaggle_state.json"))
    if request.node.get_closest_marker("real_stop"):
        return          # 실 stop_running_session 을 가짜 playwright 로 검증하는 테스트
    from freecloud.providers import kaggle_ui as _UI
    monkeypatch.setattr(_UI, "stop_running_session",
                        lambda *a, **k: (False, "테스트: stop_running_session 차단(브라우저 미기동)"),
                        raising=False)


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


@pytest.fixture(autouse=True)
def _block_hf_storage(request, monkeypatch):
    """hf_storage 의 HfApi 생성(_api)을 기본 차단 — 실수로라도 네트워크를 두드리지 못하게.

    storage_report/prune/write_probe/hygiene 등 모든 HF 호출이 이 한 함수를 지난다. 위생 경로를
    검증하는 테스트는 `@pytest.mark.allow_hf_storage` 로 블록을 끄고 가짜 HfApi 를 직접 심는다."""
    if request.node.get_closest_marker("allow_hf_storage"):
        return
    from freecloud import hf_storage as _S

    def _blocked(tok=None):
        raise RuntimeError("테스트: hf_storage._api 차단(네트워크 0)")

    monkeypatch.setattr(_S, "_api", _blocked, raising=False)
