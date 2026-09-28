"""用固定合成经历验证连续学习、只读检索、重启和发布失败；不访问用户数据。"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time


def main() -> None:
    """在全新输出目录运行实际消费者，并留下可跨版本比较的状态摘要。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=272)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(args.source.resolve()))
    import numpy as np
    from plugins.akasha.application.consumer import MessageConsumer
    from plugins.akasha.application.cycle import MemoryCycle
    import plugins.akasha.application.cycle as cycle_module
    from plugins.akasha.domain.model import ContextState, MemoryConfig, Turn, TurnFeedback
    from plugins.akasha.infrastructure.consumption import Applied, Consumption, Skipped
    import plugins.akasha.infrastructure.persistence as persistence

    rng = np.random.default_rng(20260929)
    config = MemoryConfig()
    turns = []
    entries = []
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # 1. 重复经历、跨 Session、长间隔、remember/forget 与无向量输入共用原算法。
    prototypes = rng.normal(size=(9, 33)).astype(np.float32)
    for node in range(args.turns):
        session = f"scenario:{node % 3}"
        vector = prototypes[node % 9] + rng.normal(0, .07, 33).astype(np.float32)
        vector /= np.linalg.norm(vector)
        gap = 86400.0 if node % 37 == 0 else float(node % 11 + 1)
        stamp += timedelta(seconds=gap)
        user, assistant = f"u:{node}", f"a:{node}"
        feedback = TurnFeedback()
        if node > 9 and node % 19 == 0:
            feedback = TurnFeedback(remember_nodes=(node - 9,), remember_boost=2.0)
        if node > 12 and node % 23 == 0:
            feedback = TurnFeedback(forget_nodes=(node - 12,))
        turns.append(Turn(
            node, f"turn:{node}", session, node * 2, user, assistant,
            stamp.isoformat(), stamp.isoformat(), f"episode {node % 9}", "response",
            None if node > 0 and node % 17 == 0 else vector,
            None if node > 0 and node % 13 == 0 else vector.copy(),
            ((f"topic{node % 9}", 2), ("shared", 1)), (("reply", 1),),
            gap, feedback,
        ))
        entries.append(Applied(
            learning_binding="scenario-binding", session_id=session,
            ending=(node * 2 + 1, assistant), members=((node * 2, user), (node * 2 + 1, assistant)),
            observations=(), source_digest=hashlib.sha256(str(node).encode()).hexdigest(),
        ))
    state = Consumption(cutover_heads=())
    cycle = MemoryCycle(config)
    cycle.context = ContextState((), None, ())
    prefix = args.turns - 16
    for turn, entry in zip(turns[:prefix], entries[:prefix]):
        cycle.commit(turn, None)
        state = state.append(entry)
    path = args.output / "memory.db"

    def export(target: Path, current: MemoryCycle, progress: Consumption) -> str:
        """完整 writer 作为连续发布的独立对照。"""
        assert current.context is not None
        return persistence.write_memory_database(
            target, turns=current.turns, graph=current.graph, events=current.events,
            evidence=current.evidence, captures=[], context=current.context,
            burst_members=current.burst_members, config=config, metadata={},
            recalls=current.recalls, consumption=progress,
        )

    def tables(target: Path) -> dict:
        """比较所有表和 schema，不只比较召回或一个摘要。"""
        with sqlite3.connect(target) as connection:
            schema = connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
            values = {name: sorted(connection.execute(f'SELECT * FROM "{name}"').fetchall(), key=repr)
                      for kind, name, _ in schema if kind == "table"}
            assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        return {"schema": schema, "tables": values}

    export(path, cycle, state)
    consumer = MessageConsumer(path, turns=cycle.turns, state=state, config=config)
    timings = []
    for turn, entry in zip(turns[prefix:], entries[prefix:]):
        start = time.perf_counter()
        assert consumer.apply(turn, entry)
        timings.append(time.perf_counter() - start)
        assert not consumer.apply(turn, entry)
        export(args.output / "full.db", consumer.cycle, consumer.state)
        assert tables(path) == tables(args.output / "full.db")
        restored_turns, restored_state = list(consumer.cycle.turns), consumer.state
        consumer.close()
        consumer = MessageConsumer(path, turns=restored_turns, state=restored_state, config=config)
    # 2. 查询成功和中途失败均不能改变已学习状态；跳过也必须与完整写入等价。
    cue = replace(turns[-1], node_id=args.turns, turn_id="query", feedback=TurnFeedback())
    before = export(args.output / "before-query.db", consumer.cycle, consumer.state)
    completions = [asdict(consumer.cycle.retrieve(replace(cue, user_dense=prototypes[i])).completion)
                   for i in range(9)]
    original_readout = cycle_module.read_pattern_completion

    def fail_readout(**kwargs):
        raise RuntimeError("scenario: interrupted readout")

    cycle_module.read_pattern_completion = fail_readout
    try:
        try:
            consumer.cycle.retrieve(cue)
        except RuntimeError as error:
            assert str(error) == "scenario: interrupted readout"
        else:
            raise AssertionError("readout did not fail")
    finally:
        cycle_module.read_pattern_completion = original_readout
    assert export(args.output / "after-query.db", consumer.cycle, consumer.state) == before
    skipped = consumer.state.mark_skipped(Skipped(session_id="scenario:skip", ending=(1, "skip:1"), reason="missing-embedding"))
    consumer.publish_snapshot_for(skipped)
    consumer.state = skipped
    export(args.output / "full.db", consumer.cycle, consumer.state)
    assert tables(path) == tables(args.output / "full.db")
    final_hash = persistence.logical_state_sha256(path)
    # 3. 模拟发布前和替换后的真实 I/O 失败；重启只接受磁盘上完整的旧或新快照。
    faults = []
    for boundary in ("file-sync", "replace", "directory-sync"):
        original_sync, original_replace = persistence.os.fsync, persistence.os.replace
        count = 0
        disk_before = persistence.sha256_file(path)
        next_state = consumer.state.mark_skipped(Skipped(session_id="fault", ending=(len(faults), f"fault:{boundary}"), reason="scenario"))

        def sync(fd):
            nonlocal count
            count += 1
            if (boundary == "file-sync" and count == 1) or (boundary == "directory-sync" and count == 2):
                raise OSError(f"scenario: {boundary}")
            return original_sync(fd)

        def publish(source, target):
            if boundary == "replace":
                raise OSError("scenario: replace")
            return original_replace(source, target)

        persistence.os.fsync, persistence.os.replace = sync, publish
        try:
            try:
                consumer.publish_snapshot_for(next_state)
            except OSError as error:
                assert str(error) == f"scenario: {boundary}"
            else:
                raise AssertionError("publication did not fail")
        finally:
            persistence.os.fsync, persistence.os.replace = original_sync, original_replace
        actual = persistence.load_consumption(path)
        assert actual == (next_state if boundary == "directory-sync" else consumer.state)
        if boundary != "directory-sync":
            assert persistence.sha256_file(path) == disk_before
        tables(path)
        restored_turns = list(consumer.cycle.turns)
        consumer.close()
        assert actual is not None
        consumer = MessageConsumer(path, turns=restored_turns, state=actual, config=config)
        faults.append({"boundary": boundary, "published_new": boundary == "directory-sync"})
    consumer.close()
    result = {"turns": args.turns, "logical_state_sha256": final_hash,
              "recalls_sha256": hashlib.sha256(json.dumps(completions, sort_keys=True).encode()).hexdigest(),
              "incremental_seconds": timings, "all_tables_equal": True,
              "query_unchanged": True, "faults": faults}
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
