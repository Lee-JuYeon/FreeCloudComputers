"""fits() GPU 매칭 회귀: T4 job이 P100 provider를 제외하고 T4 provider를 고르는지."""
from freecloud import orchestrator, registry, state
from freecloud.job import Job
from freecloud.providers.base import Provider, Probe, RunResult, _gpu_match


def test_gpu_match_basic():
    assert _gpu_match("T4", "T4")
    assert _gpu_match("T4", "T4x2")            # 접미사
    assert _gpu_match("T4", "T4|L4|A10G")      # 나열
    assert not _gpu_match("T4", "P100")        # P100은 T4 아님
    assert not _gpu_match("T4", "L4|A10G")
    assert _gpu_match("A10G", "T4|L4|A10G")


class Fake(Provider):
    def __init__(self, name, gpu):
        self.name = name
        self.capabilities = {"gpu": gpu, "vram_gb": 16, "headless": True}
        self.calls = 0

    def probe(self):
        return Probe(True, "")

    def run(self, job):
        self.calls += 1
        return RunResult(True, "done", "ok")


def test_t4_job_skips_p100_provider(monkeypatch):
    p100 = Fake("kaggle", "P100")
    t4 = Fake("kaggle-ui", "T4x2")
    mapping = {"kaggle": p100, "kaggle-ui": t4}
    monkeypatch.setattr(registry, "get", lambda n: mapping[n])
    monkeypatch.setattr(registry, "DEFAULT_ORDER", ["kaggle", "kaggle-ui"])
    state.clear_cooldown("kaggle"); state.clear_cooldown("kaggle-ui")
    job = Job(name="t", entrypoint="x", needs={"gpu": "T4"})
    res = orchestrator.run(job, max_rounds=1)
    assert res["ok"] and res["provider"] == "kaggle-ui"   # T4 provider 선택
    assert p100.calls == 0                                 # P100은 UNFIT → 실행 안 함
    assert t4.calls == 1


def test_no_gpu_need_matches_any(monkeypatch):
    p100 = Fake("kaggle", "P100")
    monkeypatch.setattr(registry, "get", lambda n: p100)
    monkeypatch.setattr(registry, "DEFAULT_ORDER", ["kaggle"])
    state.clear_cooldown("kaggle")
    res = orchestrator.run(Job(name="t", entrypoint="x"), max_rounds=1)  # gpu 요구 없음
    assert res["ok"] and p100.calls == 1
