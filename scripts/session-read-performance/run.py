"""Session 读取性能基线与爬坡测量。

在一次性临时目录构造真实 sessions.db，测量 ledger 读取面的端到端耗时、
SQL 语句数与消息解码次数。结果追加到 results.jsonl，供爬坡迭代对账。

口径：
- 数据镜像生产：目标会话按 4 条小输入/输出 + 1 条大 ToolResult（默认 50KB）
  混合，模拟工具密集型长会话；目录另有 --sessions 个普通会话。
- 冷态 = 新建 MessageLog（解码缓存、读连接、共享前缀全部为空），模拟进程
  重启或会话首次被读取；热态 = 同一 log 上追加少量消息后再次读取。
- 每次测量前 gc.collect()；报告 repeats 次的中位数与最小值。
- SQL 语句数通过 sqlite3 连接工厂计数；解码次数包装 MessageLog._decode。
"""

import argparse
import gc
import json
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from plugins.ledger.contract import (
    CallRef,
    ContentPart,
    ContentReferences,
    Input,
    Output,
    ToolResult,
    encode_body,
)
from plugins.ledger.log import MessageLog

QUERIES = [0]
DECODES = [0]

_real_connect = sqlite3.connect


class _CountingConnection(sqlite3.Connection):
    def execute(self, sql, parameters=(), /):
        QUERIES[0] += 1
        return super().execute(sql, parameters)

    def executemany(self, sql, parameters, /):
        QUERIES[0] += 1
        return super().executemany(sql, parameters)


def _counting_connect(*args, **kwargs):
    kwargs.setdefault("factory", _CountingConnection)
    return _real_connect(*args, **kwargs)


sqlite3.connect = _counting_connect

_real_decode = MessageLog._decode


def _counting_decode(self, row):
    DECODES[0] += 1
    return _real_decode(self, row)


MessageLog._decode = _counting_decode


def _lines(size: int) -> str:
    line = "000123 read /src/example/module.py output line with realistic tool text\n"
    repeats = size // len(line) + 1
    return (line * repeats)[:size]


def seed(log: MessageLog, session_id: str, count: int, *, large_bytes: int) -> None:
    """一条真实 append 建会话与模板行，其余用 executemany 批量灌入。"""
    writer = log.writer(
        session_id,
        author="user",
        source="conversation",
        body_types=(Input,),
        content={"text": lambda part: ContentReferences()},
    )
    writer.append(f"{session_id}:0", Input((ContentPart("text", _lines(500)),)))
    db = log._connection
    template = dict(
        db.execute("SELECT * FROM messages WHERE session_key=?", (session_id,)).fetchone()
    )
    columns = tuple(template)
    small_output = encode_body(
        Output((ContentPart("text", _lines(500)),), "complete")
    )
    rows = []
    for index in range(1, count):
        if index % 10 == 9 or index == count - 1:
            body = encode_body(
                ToolResult(
                    CallRef(f"call:{session_id}:{index}", 0),
                    "success",
                    (ContentPart("text", _lines(large_bytes)),),
                )
            )
            author = "assistant"
        elif index % 2:
            body = small_output
            author = "assistant"
        else:
            body = template["body"]
            author = "user"
        row = {
            **template,
            "id": f"{session_id}:{index}",
            "seq": index,
            "author": author,
            "body": body,
        }
        rows.append(tuple(row[column] for column in columns))
    with db:
        db.executemany(
            "INSERT INTO messages ("
            + ",".join(columns)
            + ") VALUES ("
            + ",".join("?" for _ in columns)
            + ")",
            rows,
        )
        db.execute("UPDATE sessions SET next_seq=? WHERE key=?", (count, session_id))


def measure(fn, repeats: int) -> dict:
    times, queries, decodes = [], [], []
    for _ in range(repeats):
        gc.collect()
        QUERIES[0] = DECODES[0] = 0
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
        queries.append(QUERIES[0])
        decodes.append(DECODES[0])
    return {
        "ms_median": round(statistics.median(times), 2),
        "ms_min": round(min(times), 2),
        "queries": int(statistics.median(queries)),
        "decodes": int(statistics.median(decodes)),
    }


def run_cases(path: Path, target: str, count: int, repeats: int) -> dict[str, dict]:
    """每个 case 自己控制 log 生命周期；计时只含被测调用。"""
    cases: dict[str, dict] = {}

    def cold_snapshot():
        log = MessageLog(path)
        try:
            QUERIES[0] = DECODES[0] = 0
            start = time.perf_counter()
            snapshot = log.reader(target).committed_snapshot()
            times.append((time.perf_counter() - start) * 1000)
            queries.append(QUERIES[0])
            decodes.append(DECODES[0])
            assert len(snapshot) == count
        finally:
            log.close()

    times, queries, decodes = [], [], []
    for _ in range(repeats):
        gc.collect()
        cold_snapshot()
    cases["cold_snapshot"] = {
        "ms_median": round(statistics.median(times), 2),
        "ms_min": round(min(times), 2),
        "queries": int(statistics.median(queries)),
        "decodes": int(statistics.median(decodes)),
    }

    def fresh_log_call(call):
        log = MessageLog(path)
        try:
            reader = log.reader(target)
            QUERIES[0] = DECODES[0] = 0
            start = time.perf_counter()
            call(log, reader)
            return (time.perf_counter() - start) * 1000
        finally:
            log.close()

    def timed_fresh(name, call):
        times, queries, decodes = [], [], []
        for _ in range(repeats):
            gc.collect()
            times.append(fresh_log_call(call))
            queries.append(QUERIES[0])
            decodes.append(DECODES[0])
        cases[name] = {
            "ms_median": round(statistics.median(times), 2),
            "ms_min": round(min(times), 2),
            "queries": int(statistics.median(queries)),
            "decodes": int(statistics.median(decodes)),
        }

    timed_fresh(
        "read_tail_50", lambda log, reader: reader.read_tail(limit=50)
    )

    def read_all(log, reader):
        after = -1
        while True:
            page = reader.read_page(after_seq=after, limit=200)
            if page.messages:
                after = page.messages[-1].seq
            if not page.has_more:
                break

    timed_fresh("read_pages_full", read_all)

    timed_fresh("scan_full", lambda log, reader: reader.scan(tuple))

    ref = CallRef(f"call:{target}:{count - 1}", 0)

    def settle_scan(log, reader):
        found = reader.scan(
            lambda rows: any(
                isinstance(message.body, ToolResult) and message.body.call_ref == ref
                for message in rows
            )
        )
        assert found

    timed_fresh("settle_scan", settle_scan)

    def settle_lookup(log, reader):
        assert reader.result_committed(ref)

    probe = MessageLog(path)
    try:
        has_lookup = hasattr(probe.reader(target), "result_committed")
    finally:
        probe.close()
    if has_lookup:
        timed_fresh("settle_lookup", settle_lookup)

    timed_fresh(
        "catalog_sessions_50", lambda log, reader: log.catalog().sessions(limit=50)
    )
    timed_fresh("snapshot_heads", lambda log, reader: log.catalog().snapshot_heads())

    # 热态：共享前缀命中后只补读尾部 4 条新消息（react 稳态每轮路径）。
    log = MessageLog(path)
    try:
        writer = log.writer(
            target,
            author="assistant",
            source="conversation",
            body_types=(Output,),
            content={"text": lambda part: ContentReferences()},
        )
        reader = log.reader(target)
        keep = reader.committed_snapshot()

        def warm_once():
            nonlocal keep
            for index in range(4):
                writer.append(
                    f"{target}:warm:{time.time_ns()}:{index}",
                    Output((ContentPart("text", _lines(500)),), "complete"),
                )
            QUERIES[0] = DECODES[0] = 0
            start = time.perf_counter()
            keep = reader.committed_snapshot()
            return (time.perf_counter() - start) * 1000

        times, queries, decodes = [], [], []
        for _ in range(repeats):
            gc.collect()
            times.append(warm_once())
            queries.append(QUERIES[0])
            decodes.append(DECODES[0])
        cases["warm_snapshot_tail4"] = {
            "ms_median": round(statistics.median(times), 2),
            "ms_min": round(min(times), 2),
            "queries": int(statistics.median(queries)),
            "decodes": int(statistics.median(decodes)),
        }
    finally:
        log.close()
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--messages", type=int, default=20000, help="目标会话消息数")
    parser.add_argument("--sessions", type=int, default=300, help="目录普通会话数")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--large-bytes", type=int, default=50_000)
    parser.add_argument("--label", default=None, help="写入 results.jsonl 的标签")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.jsonl"),
    )
    args = parser.parse_args()

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    label = args.label or sha

    with tempfile.TemporaryDirectory(prefix="akashic-read-perf-") as directory:
        path = Path(directory) / "sessions.db"
        log = MessageLog(path)
        target = "bench:target"
        seed_start = time.perf_counter()
        for index in range(args.sessions):
            seed(log, f"bench:other:{index}", 40, large_bytes=4_000)
        # 目标长会话最后灌入：目录第一页包含它，stats 聚合成本计入测量。
        seed(log, target, args.messages, large_bytes=args.large_bytes)
        log.close()
        seed_ms = round((time.perf_counter() - seed_start) * 1000)
        size_mb = round(path.stat().st_size / 1e6, 1)
        print(
            f"seeded target={args.messages} messages, sessions={args.sessions}, "
            f"db={size_mb}MB in {seed_ms}ms",
            file=sys.stderr,
        )
        cases = run_cases(path, target, args.messages, args.repeats)

    record = {
        "label": label,
        "sha": sha,
        "messages": args.messages,
        "sessions": args.sessions,
        "large_bytes": args.large_bytes,
        "repeats": args.repeats,
        "cases": cases,
    }
    with args.output.open("a") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    width = max(len(name) for name in cases)
    print(f"{'case'.ljust(width)}  ms_median  ms_min  queries  decodes")
    for name, stats in cases.items():
        print(
            f"{name.ljust(width)}  "
            f"{stats['ms_median']:>9}  {stats['ms_min']:>6}  "
            f"{stats['queries']:>7}  {stats['decodes']:>7}"
        )


if __name__ == "__main__":
    main()
