"""
扫描 sessions.db 所有 message，用 jieba 切词算 IDF，写入 akasha.db 的 fts_token_idf 表。
"""

from __future__ import annotations

import math
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import jieba

AKASHA_DB = Path.home() / ".akashic" / "workspace" / "memory" / "akasha.db"
SESSIONS_DB = Path.home() / ".akashic" / "workspace" / "sessions.db"


def tokenize(text: str) -> set[str]:
    out: set[str] = set()
    for w in jieba.cut_for_search(text or ""):
        cleaned = "".join(
            c for c in w.strip()
            if c.isalnum() or "一" <= c <= "鿿"
        )
        if len(cleaned) > 1:
            out.add(cleaned.lower())
    return out


def main() -> None:
    if not AKASHA_DB.exists():
        print(f"❌ {AKASHA_DB} 不存在"); sys.exit(1)
    if not SESSIONS_DB.exists():
        print(f"❌ {SESSIONS_DB} 不存在"); sys.exit(1)

    sconn = sqlite3.connect(SESSIONS_DB)
    cur = sconn.cursor()
    print("[scan] messages ...")
    df: dict[str, int] = defaultdict(int)
    n_docs = 0
    for (content,) in cur.execute("SELECT content FROM messages"):
        n_docs += 1
        for tok in tokenize(content):
            df[tok] += 1
        if n_docs % 1000 == 0:
            print(f"  {n_docs} messages, {len(df)} unique tokens")
    sconn.close()
    print(f"  done: {n_docs} messages, {len(df)} unique tokens")

    idf: dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log((n_docs + 1) / (freq + 1)) + 1

    # 写入 akasha.db
    aconn = sqlite3.connect(AKASHA_DB)
    aconn.execute("""
        CREATE TABLE IF NOT EXISTS fts_token_idf (
            token TEXT PRIMARY KEY,
            df INTEGER NOT NULL,
            idf REAL NOT NULL
        )
    """)
    aconn.execute("DELETE FROM fts_token_idf")
    aconn.executemany(
        "INSERT INTO fts_token_idf VALUES (?, ?, ?)",
        [(t, df[t], idf[t]) for t in df],
    )
    aconn.commit()
    print(f"  wrote {len(df)} tokens to akasha.db:fts_token_idf")

    # 分布报告
    sorted_by_idf = sorted(idf.items(), key=lambda x: x[1])
    print(f"\n=== IDF 分布 ===")
    bins = [
        (-1, 1, "极常见"),
        (1, 2, "常见"),
        (2, 3, "中等"),
        (3, 5, "稀有"),
        (5, 99, "极稀有"),
    ]
    for lo, hi, label in bins:
        cnt = sum(1 for _, v in idf.items() if lo <= v < hi)
        print(f"  IDF [{lo:.1f},{hi:.1f}) {label:<8} {cnt} tokens")

    print(f"\n=== 最常见 token (低 IDF，应过滤) ===")
    for tok, v in sorted_by_idf[:20]:
        print(f"  IDF={v:.2f}  df={df[tok]:>5}  {tok}")

    print(f"\n=== 最稀有 token sample (高 IDF，有信息量) ===")
    rare = [t for t in sorted_by_idf if 4.5 < t[1] < 8.5]
    for tok, v in rare[::max(1, len(rare)//20)][:20]:
        print(f"  IDF={v:.2f}  df={df[tok]:>5}  {tok}")

    aconn.close()


if __name__ == "__main__":
    main()
