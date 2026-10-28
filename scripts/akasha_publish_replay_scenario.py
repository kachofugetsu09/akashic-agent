"""在线 ≡ 重建：按批发布、崩溃、重启、发布失败后按原序重学，必须与逐条发布、完整重建得到同一逻辑状态。"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any


# 固定经历：跨 Session、长短间隔、remember/forget 反馈；间隔与 dialogue_turn 一样由接纳时间得出。
def _experience(count: int) -> tuple[list[Any], list[Any]]:
    import numpy as np
    from plugins.akasha.domain.model import Turn, TurnFeedback
    from plugins.akasha.infrastructure.consumption import Applied

    rng = np.random.default_rng(20261014)
    prototypes = rng.normal(size=(12, 64)).astype(np.float32)
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    turns: list[Turn] = []
    entries: list[Applied] = []
    for node in range(count):
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
    return turns, entries


# 线上模式的查询 cue：真实用户输入先查询（只有用户侧），回复结束后同一轮再学习。
def _user_cue(turn: Any, node_id: int) -> Any:
    return replace(
        turn, node_id=node_id, turn_id=f"query:{turn.node_id}", assistant_text="", assistant_dense=None,
        assistant_terms=(), user_message_id="", assistant_message_id="", committed_at=turn.started_at,
    )


class _Interrupts:
    """按批路径的中断计划：崩溃丢掉未发布尾部、干净重启、发布写到一半失败。"""

    def __init__(self, count: int, publish_every: int) -> None:
        self.crashes = {5 * publish_every + 7, (count * 3) // 5 + 3}
        self.restarts = {(count * 2) // 5}
        self.failures = {(count * 4) // 5 // publish_every * publish_every}
        self.fail_next = False
        self.surfaced = 0


def _run(path: Path, turns: list[Any], entries: list[Any], *, batch: int, plan: _Interrupts | None) -> Any:
    from plugins.akasha.application.consumer import MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig
    from plugins.akasha.infrastructure.consumption import Consumption
    from plugins.akasha.infrastructure.persistence import load_consumption

    # 1. 重开只装载磁盘上的完整检查点，再从日志接着学。
    def reopen() -> tuple[Any, int]:
        published = load_consumption(path)
        assert published is not None
        index = len(published.applied)
        return MessageConsumer(path, turns=turns[:index], state=published, config=MemoryConfig()), index

    consumer = MessageConsumer(path, turns=[], state=Consumption(cutover_heads=()), config=MemoryConfig())
    index = 0
    while index < len(turns):
        # 2. 按批路径每轮先查询（只读），再学习。
        if plan is not None and index > 0:
            _ = consumer.cycle.retrieve(_user_cue(turns[index], consumer.cycle.state_version))
        assert consumer.apply(turns[index], entries[index], publish=batch == 1)
        index += 1
        if len(consumer.state.applied) % batch == 0:
            if plan is not None and index in plan.failures:
                # 3. 发布写到一半失败：文件仍是上一检查点，实例停止，重开后重学。
                plan.failures.discard(index)
                plan.fail_next = True
                try:
                    consumer.flush()
                except OSError:
                    plan.surfaced += 1
                    consumer.close()
                    consumer, index = reopen()
                    continue
                raise AssertionError("注入的发布失败没有浮出")
            consumer.flush()
        if plan is not None and index in plan.crashes | plan.restarts:
            # 4. 崩溃不发布尾部；干净重启先发布再关闭。
            if index in plan.restarts:
                consumer.flush()
            plan.crashes.discard(index)
            plan.restarts.discard(index)
            consumer.close()
            consumer, index = reopen()
    consumer.flush()
    return consumer


def main() -> None:
    """用合成经历比较三条路径含消费进度的逻辑哈希，并核对恢复与查询；不访问用户数据，不调用模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--source", type=Path, required=True)
    _ = parser.add_argument("--turns", type=int, default=300)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    import plugins.akasha.infrastructure.persistence as persistence
    from plugins.akasha.application.consumer import PUBLISH_EVERY, MessageConsumer
    from plugins.akasha.domain.model import MemoryConfig
    from plugins.akasha.infrastructure.consumption import Consumption

    # 1. 在发布的最后一步注入一次失败，检验事务回滚后文件仍是上一检查点。
    turns, entries = _experience(args.turns)
    plan = _Interrupts(args.turns, PUBLISH_EVERY)
    original = persistence._write_recalls  # pyright: ignore[reportPrivateUsage]

    def failing(*items: Any, **options: Any) -> None:
        if plan.fail_next:
            plan.fail_next = False
            raise OSError("injected publish failure")
        original(*items, **options)

    persistence._write_recalls = failing  # pyright: ignore[reportPrivateUsage]
    root = Path(tempfile.mkdtemp(prefix="akasha-publish-replay-"))

    # 2. 三条路径：逐条发布；按批发布并带中断；重建式一次发布。
    every = _run(root / "every.db", turns, entries, batch=1, plan=None)
    every.close()
    live = _run(root / "batched.db", turns, entries, batch=PUBLISH_EVERY, plan=plan)
    rebuild = MessageConsumer(root / "rebuild.db", turns=[], state=Consumption(cutover_heads=()),
                              config=MemoryConfig(), deferred_publish=True)
    for turn, entry in zip(turns, entries):
        assert rebuild.apply(turn, entry)
    _ = rebuild.publish_snapshot()
    rebuild.close()

    # 3. 恢复后的查询等于存活态的查询；查询不改变已发布状态。
    probe = _user_cue(turns[-1], live.cycle.state_version)
    live_query = repr(live.cycle.retrieve(probe).completion)
    live.close()
    hashes = {name: persistence.logical_state_sha256(root / f"{name}.db") for name in ("every", "batched", "rebuild")}
    published = persistence.load_consumption(root / "batched.db")
    assert published is not None
    restored = MessageConsumer(root / "batched.db", turns=turns, state=published, config=MemoryConfig())
    restored_query = repr(restored.cycle.retrieve(probe).completion)
    restored.close()
    checks = {
        "logical_equal": len(set(hashes.values())) == 1,
        "restored_query_equal": restored_query == live_query,
        "query_is_read_only": persistence.logical_state_sha256(root / "batched.db") == hashes["batched"],
        "injected_failure_surfaced": plan.surfaced == 1,
        "all_interrupts_ran": not (plan.crashes or plan.restarts or plan.failures),
    }
    print(json.dumps({"turns": args.turns, "publish_every": PUBLISH_EVERY, **hashes, **checks}, indent=1))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
