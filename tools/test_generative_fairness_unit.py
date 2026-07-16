#!/usr/bin/env python3
"""Unit checks for GPU fairness and key-aware micro-batching (no GPU required)."""
from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_fairness_prefers_starved_visual():
    # Six-lane: with no qwen/sarvam↔visual overlap and no qwen↔sarvam overlap,
    # after sarvam finishes, visual is dispatched before qwen (exec order).
    os.environ["AI_ENABLE_QWEN_SARVAM_OVERLAP"] = "false"
    os.environ["AI_ENABLE_VISUAL_QWEN_OVERLAP"] = "false"
    os.environ["AI_ENABLE_VISUAL_SARVAM_OVERLAP"] = "false"
    import runtime.gpu_scheduler as gs

    gs.reset_gpu_scheduler_for_tests()
    sched = gs.GpuScheduler()

    got = []
    ready = threading.Event()

    def gen_then_both():
        with sched.run(model="sarvam", batch_size=1, use_cuda_events=False):
            got.append("sarvam_hold")
            ready.set()
            time.sleep(0.08)

    def waiter(model, label):
        ready.wait()
        time.sleep(0.01)
        with sched.run(model=model, batch_size=1, use_cuda_events=False):
            got.append(label)

    t0 = threading.Thread(target=gen_then_both)
    t1 = threading.Thread(target=waiter, args=("qwen_vl", "generative_next"))
    t2 = threading.Thread(target=waiter, args=("emotion", "visual_next"))
    t0.start()
    time.sleep(0.02)
    t1.start()
    t2.start()
    t0.join()
    t1.join()
    t2.join()
    after = [x for x in got if x.endswith("_next")]
    assert after[0] == "visual_next", after
    print("starved visual preferred ok", got)


async def test_microbatch_affinity():
    from main import MicroBatcher

    def process_fn(payloads):
        langs = {p[1] for p in payloads}
        assert len(langs) == 1, payloads
        return [f"{p[0]}|{p[1]}" for p in payloads]

    batcher = MicroBatcher(
        "sarvam",
        max_batch_size=4,
        max_wait_ms=50,
        process_fn=process_fn,
        key_fn=lambda p: p[1],
    )
    batcher.start()
    try:
        async def submit(text, lang):
            return await batcher.submit((text, lang))

        futs = [
            asyncio.create_task(submit("a", "hi")),
            asyncio.create_task(submit("b", "en")),
            asyncio.create_task(submit("c", "hi")),
            asyncio.create_task(submit("d", "hi")),
        ]
        outs = await asyncio.gather(*futs)
    finally:
        await batcher.stop()

    assert "a|hi" in outs and "c|hi" in outs and "d|hi" in outs
    assert "b|en" in outs
    print("microbatch affinity ok", outs)


def main() -> int:
    test_fairness_prefers_starved_visual()
    asyncio.run(test_microbatch_affinity())
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
