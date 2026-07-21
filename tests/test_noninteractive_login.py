"""비대화형 환경에서 login 이 '멈추지 않고' 실패하는지 회귀 테스트.

과거 버그: `fcc login huggingface` 를 에이전트/파이프에서 돌리면 getpass 가 입력할 수 없는
프롬프트에서 **영원히 멈췄다**(출력조차 없어 진단 불가). 이 CLI 에서 가장 알아채기 어려운
실패 모드였다. 이제는 즉시 SystemExit 로 안내한다.
"""
import pytest

from freecloud import auth


@pytest.fixture
def noninteractive(monkeypatch):
    monkeypatch.setattr(auth, "interactive_available", lambda: False)


@pytest.fixture(autouse=True)
def no_prompt(monkeypatch):
    """테스트가 실수로 진짜 프롬프트에 걸리지 않도록 봉쇄."""
    def boom(*a, **k):
        raise AssertionError("프롬프트가 호출되면 안 된다(멈춤 회귀)")
    monkeypatch.setattr("builtins.input", boom)
    import getpass
    monkeypatch.setattr(getpass, "getpass", boom)


def _clear(monkeypatch, *names):
    for n in names:
        monkeypatch.delenv(n, raising=False)


@pytest.mark.parametrize("fn, envs", [
    (auth.login_huggingface, ("HF_TOKEN", "HUGGINGFACE_TOKEN")),
    (auth.login_lightning, ("LIGHTNING_API_KEY",)),
    (auth.login_saturn, ("SATURN_TOKEN",)),
])
def test_비대화형이면_멈추지_않고_즉시_안내한다(fn, envs, noninteractive, monkeypatch):
    _clear(monkeypatch, *envs)
    with pytest.raises(SystemExit) as e:
        fn()
    assert "터미널" in str(e.value)


def test_env에_토큰이_있으면_프롬프트_없이_진행한다(noninteractive, monkeypatch):
    """무인 경로 — CI 에서 env 로 넘기면 로그인이 성립해야 한다."""
    monkeypatch.setenv("SATURN_TOKEN", "tok_dummy")
    stored = {}
    monkeypatch.setattr(auth.secrets, "set", lambda k, v: stored.__setitem__(k, v))

    auth.login_saturn()  # SystemExit 나면 안 됨

    assert stored["SATURN_TOKEN"] == "tok_dummy"


def test_prompt_secret_자체가_비대화형을_막는다(noninteractive):
    """호출부가 가드를 빠뜨려도 여기서 걸린다."""
    with pytest.raises(SystemExit):
        auth._prompt_secret("아무거나: ")


def test_대화형이면_가드가_통과한다(monkeypatch):
    monkeypatch.setattr(auth, "interactive_available", lambda: True)
    auth._require_interactive("무언가")  # 예외 없어야 함


def test_env토큰이_무효면_프롬프트로_넘어간다(monkeypatch, capsys):
    """회귀: 만료 토큰이 env 에 남아 있으면 새 토큰을 넣을 기회조차 없었다."""
    monkeypatch.setattr(auth, "interactive_available", lambda: True)
    monkeypatch.setenv("HF_TOKEN", "hf_expired")
    monkeypatch.setattr(auth, "verify_hf_token",
                        lambda t: (t == "hf_new", "무효" if t != "hf_new" else "OK"))
    monkeypatch.setattr(auth, "_prompt_secret", lambda p: "hf_new")
    monkeypatch.setattr(auth, "_mirror_hf_cache", lambda t: None)
    stored = {}
    monkeypatch.setattr(auth.secrets, "set", lambda k, v: stored.__setitem__(k, v))

    auth.login_huggingface()

    assert stored["HF_TOKEN"] == "hf_new", "무효 env 토큰에서 멈추면 안 된다"


def test_env가_저장값을_가리면_경고한다(monkeypatch, capsys):
    """secrets.get 은 env 우선 — 낡은 env 가 새 토큰을 조용히 가리는 걸 알려야 한다."""
    monkeypatch.setenv("HF_TOKEN", "hf_stale")
    auth._warn_if_env_shadows("HF_TOKEN", "hf_new")
    out = capsys.readouterr().out
    assert "경고" in out and "HF_TOKEN" in out


def test_env가_같으면_경고하지_않는다(monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_same")
    auth._warn_if_env_shadows("HF_TOKEN", "hf_same")
    assert capsys.readouterr().out == ""
