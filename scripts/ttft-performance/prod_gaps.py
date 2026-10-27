"""只读统计生产实例的"发消息→首次模型调用"间隔，按会话规模和日期分组。

用法（在宿主上跑，库以 mode=ro 打开）：
  python3 prod_gaps.py <workspace> [--days 14] [--session <会话 id 后缀>] [--list]
间隔 = 用户 Input 的 ts 到其后第一条 model_calls.started_at（秒级精度）。并发会话的调用可能
落在两者之间，单条可能偏小；按日中位数观察趋势。只读 messages 与 model_calls，不打印正文。
"""

import argparse
import bisect
import sqlite3
import statistics
from collections import defaultdict
from datetime import UTC, datetime


def _at(value: str) -> datetime:
    moment = datetime.fromisoformat(value)
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument("--session", default="")
    parser.add_argument("--list", action="store_true")
    options = parser.parse_args()
    sessions = sqlite3.connect(f"file:{options.workspace}/sessions.db?mode=ro", uri=True)
    models = sqlite3.connect(f"file:{options.workspace}/model-registry.sqlite3?mode=ro", uri=True)
    window = f"-{options.days + 1} days"
    calls = [_at(row[0]) for row in models.execute(
        "SELECT started_at FROM model_calls WHERE started_at > datetime('now', ?) ORDER BY started_at", (window,))]
    sizes = dict(sessions.execute("SELECT session_key, count(*) FROM messages GROUP BY session_key"))
    inputs = sessions.execute(
        "SELECT session_key, seq, ts FROM messages WHERE author='user' AND ts > datetime('now', ?) ORDER BY ts",
        (f"-{options.days} days",)).fetchall()
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for session, seq, ts in inputs:
        if not session.endswith(options.session):
            continue
        sent = _at(ts)
        index = bisect.bisect_left(calls, sent)
        if index == len(calls):
            continue
        gap = (calls[index] - sent).total_seconds()
        size = sizes.get(session, 0)
        bucket = ">=5000" if size >= 5000 else ">=1000" if size >= 1000 else "<1000"
        groups[(sent.date().isoformat(), bucket)].append(gap)
        if options.list:
            print(f"{sent.isoformat()[:19]} {session[-8:]} seq={seq} messages={size} gap_s={gap:.1f}")
    print("date        session_size  n   median_s  max_s")
    for (day, bucket), gaps in sorted(groups.items()):
        print(f"{day}  {bucket:>12}  {len(gaps):<3} {statistics.median(gaps):8.1f}  {max(gaps):6.1f}")


if __name__ == "__main__":
    main()
