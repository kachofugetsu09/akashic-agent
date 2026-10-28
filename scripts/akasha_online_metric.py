"""Akasha 在线成本：按线上模式（先查询、再学习、每 16 轮发布）学到检查点，测查询、学习与发布的中位数。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import sys
import tempfile
import time
from typing import Any


# 可重放的长期聊天形状：主题原型加噪声的向量、Zipf 词项、短长间隔混合、少量 remember/forget。
def synthetic_turns(count: int, *, dim: int, topics: int = 64, seed: int = 20261028) -> list[Any]:
    import numpy as np
    from plugins.akasha.domain.model import Turn, TurnFeedback

    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(topics, dim)).astype(np.float32)
    vocab = [f"w{index}" for index in range(6000)]
    zipf = 1.0 / np.arange(1, len(vocab) + 1)
    zipf /= zipf.sum()
    topic_words = rng.integers(0, len(vocab), size=(topics, 40))
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns: list[Turn] = []
    topic = 0
    for node in range(count):
        # 1. 会话内多数延续当前主题，偶尔切换，模拟 burst。
        if rng.random() < 0.25:
            topic = int(rng.integers(0, topics))
        gap = float(rng.choice([8.0, 45.0, 600.0, 7200.0, 86400.0], p=[0.45, 0.3, 0.12, 0.08, 0.05]))
        stamp += timedelta(seconds=gap)
        vectors = []
        for _ in range(2):
            vector = prototypes[topic] + rng.normal(0, 0.9, dim).astype(np.float32)
            vectors.append(vector / np.linalg.norm(vector))
        # 2. 词项 = 主题词 + 全局 Zipf 噪声词。
        words: dict[str, int] = {}
        for index in [*rng.choice(topic_words[topic], size=6), *rng.choice(len(vocab), size=10, p=zipf)]:
            words[vocab[int(index)]] = words.get(vocab[int(index)], 0) + 1
        reply: dict[str, int] = {}
        for index in rng.choice(topic_words[topic], size=12):
            reply[vocab[int(index)]] = reply.get(vocab[int(index)], 0) + 1
        feedback = TurnFeedback()
        if node > 20 and node % 97 == 0:
            feedback = TurnFeedback(remember_nodes=(node - 7,), remember_boost=2.0)
        elif node > 20 and node % 131 == 0:
            feedback = TurnFeedback(forget_nodes=(node - 11,))
        turns.append(Turn(
            node, f"turn:{node}", f"s{(node // 40) % 12}", node * 2, f"u{node}", f"a{node}",
            stamp.isoformat(), stamp.isoformat(), f"user {node}", f"assistant {node}",
            None if node % 53 == 7 else vectors[0], vectors[1],
            tuple(sorted(words.items())), tuple(sorted(reply.items())), gap if node else None, feedback,
        ))
    return turns


def _entry(turn: Any) -> Any:
    from plugins.akasha.infrastructure.consumption import Applied

    return Applied(
        learning_binding="metric-binding", session_id=turn.session_key,
        ending=(turn.user_seq + 1, turn.assistant_message_id),
        members=((turn.user_seq, turn.user_message_id), (turn.user_seq + 1, turn.assistant_message_id)),
        observations=(), source_digest=hashlib.sha256(turn.turn_id.encode()).hexdigest(),
    )


def _median(values: list[float]) -> float:
    return round(sorted(values)[len(values) // 2], 2) if values else 0.0


def _hot(profiler: cProfile.Profile, steps: int, top: int) -> list[str]:
    stats: dict[tuple[str, int, str], tuple[int, int, float, float, object]] = pstats.Stats(profiler).stats  # pyright: ignore[reportAttributeAccessIssue]
    rows = sorted(((cum, tot, f"{Path(file).name}:{line}:{name}") for (file, line, name), (_, _, tot, cum, _) in stats.items()),
                  reverse=True)
    return [f"{cum * 1000 / steps:8.2f} {tot * 1000 / steps:8.2f}  {name}" for cum, tot, name in rows[:top]]


# 检查点后的窗口按发布批次交替：计时批不开 profiler，采样批只收热点。
def _window(consumer: Any, turns: list[Any], node: int, size: int, publish_every: int) -> tuple[dict[str, float], cProfile.Profile]:
    query, apply, flush = [], [], []
    profiler = cProfile.Profile()
    for _ in range(size):
        sampled = (node // publish_every) % 2 == 1
        turn = turns[node]
        cue = replace(turn, node_id=consumer.cycle.state_version, turn_id=f"query:{node}", assistant_text="",
                      assistant_dense=None, assistant_terms=(), user_message_id="", assistant_message_id="",
                      committed_at=turn.started_at)
        if sampled:
            profiler.enable()
        t0 = time.perf_counter()
        _ = consumer.cycle.retrieve(cue)
        t1 = time.perf_counter()
        assert consumer.apply(turn, _entry(turn), publish=False)
        t2 = time.perf_counter()
        node += 1
        if len(consumer.state.applied) % publish_every == 0:
            consumer.flush()
            if not sampled:
                flush.append((time.perf_counter() - t2) * 1000)
        if sampled:
            profiler.disable()
        else:
            query.append((t1 - t0) * 1000)
            apply.append((t2 - t1) * 1000)
    learn = _median(apply)
    return {"query_ms": _median(query), "apply_ms": learn, "flush_ms": _median(flush),
            "online_ms_per_turn": round(learn + _median(flush) / publish_every, 2)}, profiler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source", type=Path, required=True)
    _ = parser.add_argument("--checkpoints", default="2000,4000,6000")
    _ = parser.add_argument("--window", type=int, default=64)
    _ = parser.add_argument("--dim", type=int, default=1024)
    _ = parser.add_argument("--top", type=int, default=0, help="每个检查点输出的热点行数")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from plugins.akasha.application.consumer import PUBLISH_EVERY, MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig
    from plugins.akasha.infrastructure.consumption import Consumption

    # 1. 学到每个检查点的墙钟累计就是重建口径；检查点后再测稳态窗口。
    checkpoints = sorted(int(value) for value in args.checkpoints.split(","))
    turns = synthetic_turns(checkpoints[-1] + args.window + 1, dim=args.dim)
    path = Path(tempfile.mkdtemp(prefix="akasha-online-metric-")) / "online.db"
    consumer = MessageConsumer(path, turns=[], state=Consumption(cutover_heads=()), config=MemoryConfig())
    node, learned = 0, 0.0
    for checkpoint in checkpoints:
        started = time.perf_counter()
        while node < checkpoint:
            assert consumer.apply(turns[node], _entry(turns[node]), publish=False)
            node += 1
            if len(consumer.state.applied) % PUBLISH_EVERY == 0:
                consumer.flush()
        learned += time.perf_counter() - started
        row, profiler = _window(consumer, turns, node, args.window, PUBLISH_EVERY)
        node += args.window
        print(json.dumps({"n": checkpoint, "cumulative_learn_s": round(learned, 1), **row}), flush=True)
        if args.top:
            print("\n".join(_hot(profiler, args.window // 2, args.top)), flush=True)
    consumer.close()


if __name__ == "__main__":
    main()
