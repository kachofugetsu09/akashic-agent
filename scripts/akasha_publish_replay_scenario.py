"""按批发布的可重建性：崩溃丢掉未发布尾部后按原序重学，必须与逐条发布、完整重建得到同一逻辑状态。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main() -> None:
    """用合成经历跑三条路径并比较含消费进度的逻辑哈希；不访问用户数据，不调用模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source", type=Path, required=True)
    _ = parser.add_argument("--turns", type=int, default=300)
    args = parser.parse_args()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    from plugins.akasha.application.consumer import PUBLISH_EVERY, MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig, Turn, TurnFeedback
    from plugins.akasha.infrastructure.consumption import Applied, Consumption
    from plugins.akasha.infrastructure.persistence import load_consumption, logical_state_sha256

    # 1. 固定经历：跨 Session、长短间隔、remember/forget 反馈；间隔与 dialogue_turn 一样由接纳时间得出。
    rng = np.random.default_rng(20261014)
    prototypes = rng.normal(size=(12, 64)).astype(np.float32)
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns: list[Turn] = []
    entries: list[Applied] = []
    for node in range(args.turns):
        noisy = np.asarray(prototypes[node % 12] + rng.normal(0, .2, 64), dtype=np.float32)
        vector = np.asarray(noisy / np.linalg.norm(noisy), dtype=np.float32)
        gap = float(rng.choice([5.0, 60.0, 3600.0]))
        stamp += timedelta(seconds=gap)
        feedback = TurnFeedback()
        if node > 10 and node % 29 == 0:
            feedback = TurnFeedback(remember_nodes=(node - 5,), remember_boost=2.0)
        elif node > 10 and node % 31 == 0:
            feedback = TurnFeedback(forget_nodes=(node - 7,))
        session = f"scenario:{node % 4}"
        turns.append(Turn(
            node, f"turn:{node}", session, node * 2, f"u:{node}", f"a:{node}",
            stamp.isoformat(), stamp.isoformat(), "user", "assistant", vector, vector.copy(),
            ((f"topic{node % 12}", 2), ("shared", 1)), (("reply", 1),), gap if node else None, feedback,
        ))
        entries.append(Applied(
            learning_binding="scenario-binding", session_id=session, ending=(node * 2 + 1, f"a:{node}"),
            members=((node * 2, f"u:{node}"), (node * 2 + 1, f"a:{node}")), observations=(),
            source_digest=hashlib.sha256(str(node).encode()).hexdigest(),
        ))
    root = Path(tempfile.mkdtemp(prefix="akasha-publish-replay-"))
    config, empty = MemoryConfig(), Consumption(cutover_heads=())

    # 2. 三条路径：逐条发布；按批发布并在未发布尾部崩溃后重启重学；重建式一次发布。
    def run(name: str, *, batch: int, crash_at: int | None = None) -> str:
        path = root / f"{name}.db"
        consumer = MessageConsumer(path, turns=[], state=empty, config=config)
        index = 0
        while index < len(turns):
            assert consumer.apply(turns[index], entries[index], publish=batch == 1)
            index += 1
            if len(consumer.state.applied) % batch == 0:
                consumer.flush()
            if index == crash_at:
                consumer.close()
                published = load_consumption(path)
                assert published is not None
                index = len(published.applied)
                consumer = MessageConsumer(path, turns=turns[:index], state=published, config=config)
                crash_at = None
        consumer.flush()
        consumer.close()
        return logical_state_sha256(path)

    crash_at = 5 * PUBLISH_EVERY + 7
    every = run("every", batch=1)
    batched = run("batched", batch=PUBLISH_EVERY, crash_at=crash_at)
    rebuild = MessageConsumer(root / "rebuild.db", turns=[], state=empty, config=config, deferred_publish=True)
    for turn, entry in zip(turns, entries):
        assert rebuild.apply(turn, entry)
    _ = rebuild.publish_snapshot()
    rebuild.close()
    rebuilt = logical_state_sha256(root / "rebuild.db")

    # 3. 三者含消费进度的逻辑状态必须一致。
    result: dict[str, object] = {
        "turns": args.turns, "publish_every": PUBLISH_EVERY, "crash_at": crash_at,
        "replayed_tail": crash_at % PUBLISH_EVERY, "every_turn": every, "batched_with_crash": batched,
        "rebuild": rebuilt, "equal": every == batched == rebuilt,
    }
    print(json.dumps(result, indent=1))
    if not result["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
