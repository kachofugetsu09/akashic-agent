#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from proactive_v2.mcp_sources import McpClientPool
from proactive_v2.modules_source import McpGatewaySource


EVENT_ID = "probe_interest_feedback"


def _write_sources(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "proactive_sources.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "server": "feed",
                        "channel": "content",
                        "get_tool": "get_proactive_events",
                        "ack_tool": "acknowledge_events",
                        "enabled": True,
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _seed_feed_db(feed_db: Path) -> None:
    feed_db.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC).isoformat()
    conn = sqlite3.connect(feed_db)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                event_id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                source_name TEXT NOT NULL,
                source_type TEXT NOT NULL,
                title TEXT,
                content TEXT NOT NULL,
                url TEXT,
                author TEXT,
                published_at TEXT,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                emitted_at TEXT,
                content_hash TEXT NOT NULL,
                interest_ok INTEGER,
                interest_scored_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS acked_items (
                event_id TEXT PRIMARY KEY,
                acked_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        conn.execute("DELETE FROM acked_items WHERE event_id = ?", (EVENT_ID,))
        conn.execute("DELETE FROM items WHERE event_id = ?", (EVENT_ID,))
        conn.execute(
            """
            INSERT INTO items (
                event_id, source_id, source_name, source_type, title, content,
                url, author, published_at, first_seen_at, last_seen_at, emitted_at,
                content_hash, interest_ok, interest_scored_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL)
            """,
            (
                EVENT_ID,
                "probe-source",
                "Probe Feed",
                "rss",
                "Probe item",
                "This item exists only for proactive feedback verification.",
                "https://example.com/probe-interest-feedback",
                "probe",
                now,
                now,
                now,
                "probe-hash",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _read_interest(feed_db: Path) -> tuple[int | None, str | None]:
    conn = sqlite3.connect(feed_db)
    try:
        row = conn.execute(
            "SELECT interest_ok, interest_scored_at FROM items WHERE event_id = ?",
            (EVENT_ID,),
        ).fetchone()
        if row is None:
            raise AssertionError("probe item disappeared")
        return row[0], row[1]
    finally:
        conn.close()


async def _run(args: argparse.Namespace) -> None:
    workspace = args.workspace
    feed_root = args.feed_root
    data_dir = args.data_dir
    feed_db = data_dir / "feed_mcp.sqlite3"
    python = feed_root / "mcp" / ".venv" / "bin" / "python"
    if not python.exists():
        raise SystemExit(f"feed MCP venv missing: {python}")
    _write_sources(workspace)
    _seed_feed_db(feed_db)

    pool = McpClientPool(
        workspace,
        extra_server_configs={
            "feed": {
                "command": [
                    "bash",
                    "-lc",
                    f"cd {feed_root} && exec {python} mcp/run_mcp.py",
                ],
                "env": {"AKA_PLUGIN_DATA_DIR": str(data_dir)},
            }
        },
    )
    await pool.connect_all()
    try:
        source = McpGatewaySource(pool, content_limit=5)
        events = await source.feed_fn(limit=5)
        if not any(item.get("event_id") == EVENT_ID for item in events):
            raise AssertionError(f"probe item not fetched: {events}")

        await source.ack_fn(f"feed:{EVENT_ID}", 24, "interesting")
        interest_ok, scored_at = _read_interest(feed_db)
        if interest_ok != 1 or not scored_at:
            raise AssertionError(f"interesting ack not persisted: {(interest_ok, scored_at)}")

        await source.ack_fn(f"feed:{EVENT_ID}", 720, "not_interesting")
        interest_ok, scored_at = _read_interest(feed_db)
        if interest_ok != 0 or not scored_at:
            raise AssertionError(f"not_interesting ack not persisted: {(interest_ok, scored_at)}")

        print(json.dumps({"ok": True, "event_id": EVENT_ID, "interest_ok": interest_ok}, ensure_ascii=False))
    finally:
        await pool.disconnect_all()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, default=Path(os.environ.get("AKASHIC_DEBUG_WORKSPACE", "/sandbox/workspace")))
    parser.add_argument("--feed-root", type=Path, default=Path.home() / ".akashic-plugin" / "cache" / "lab" / "feed" / "0.1.0")
    parser.add_argument("--data-dir", type=Path, default=Path.home() / ".akashic-plugin" / "data" / "feed-lab")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
