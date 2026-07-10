"""예시 entrypoint — '재개 가능한' 학습 루프의 최소 형태.

핵심 계약(어느 provider에서 죽어도 이어받으려면):
  1) 시작 시 checkpoint.pull() 로 이전 노드 상태 복원(없으면 scratch).
  2) 매 N스텝 checkpoint.push() 로 공유 저장소에 저장.
  3) max_runtime 임박 전에 저장 → 세션이 끊겨도 최신 step 보존.

실제 학습 대신 step 카운터로 메커니즘만 보여준다. transformers/peft로 바꾸면 그대로 재개형.
"""
import json
import os
import time

from freecloud.checkpoint import Checkpoint

CKPT_DIR = "./ckpt"
STATE = os.path.join(CKPT_DIR, "state.json")
TOTAL_STEPS = int(os.environ.get("TOTAL_STEPS", "1000"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "50"))


def load_step() -> int:
    if os.path.exists(STATE):
        return json.load(open(STATE)).get("step", 0)
    return 0


def save_step(step: int) -> None:
    os.makedirs(CKPT_DIR, exist_ok=True)
    json.dump({"step": step}, open(STATE, "w"))


def main() -> None:
    ck = Checkpoint.from_env()            # FREECLOUD_CKPT_REPO 있으면 활성
    if ck:
        ck.pull(CKPT_DIR)                 # [1] 이전 노드 상태 복원
    step = load_step()
    print(f"[train] resume from step={step}/{TOTAL_STEPS}", flush=True)

    while step < TOTAL_STEPS:
        # ...여기서 실제 forward/backward 1스텝... (데모: sleep)
        time.sleep(0.01)
        step += 1
        if step % SAVE_EVERY == 0:
            save_step(step)
            if ck:
                ck.push(CKPT_DIR, step=step)   # [2] 주기적 저장 → 페일오버 대비
            print(f"[train] step {step}", flush=True)

    save_step(step)
    if ck:
        ck.push(CKPT_DIR, step=step)
    print("[train] DONE", flush=True)


if __name__ == "__main__":
    main()
