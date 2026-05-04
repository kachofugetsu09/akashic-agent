#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import time
import urllib.parse
import urllib.request
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


@dataclass
class ProbePaths:
    repo: Path
    debug_dir: Path
    profile: str

    @property
    def profile_dir(self) -> Path:
        return self.debug_dir / "profiles" / self.profile

    @property
    def config(self) -> Path:
        return self.profile_dir / "config.toml"

    @property
    def workspace(self) -> Path:
        return self.profile_dir / "workspace"

    @property
    def socket(self) -> Path:
        return self.profile_dir / "akashic.sock"

    @property
    def recent_context(self) -> Path:
        return self.workspace / "memory" / "RECENT_CONTEXT.md"

    @property
    def observe_db(self) -> Path:
        return self.workspace / "observe" / "observe.db"

    @property
    def sessions_db(self) -> Path:
        return self.workspace / "sessions.db"

    @property
    def memory_db(self) -> Path:
        return self.workspace / "memory" / "memory2.db"


@dataclass
class Scenario:
    name: str
    turns: list[dict[str, object]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_compose(paths: ProbePaths, args: list[str]) -> None:
    env = {"AKASHIC_DEBUG_PROFILE": paths.profile}
    _ = subprocess.run(
        ["docker", "compose", "-f", str(paths.debug_dir / "docker-compose.yml"), *args],
        cwd=paths.repo,
        env={**dict(os.environ), **env},
        check=True,
    )


def _coerce_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [str(item) for item in items if str(item).strip()]


def _coerce_object_dict(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    items = cast(dict[object, object], value)
    return {str(key): item for key, item in items.items()}


def _coerce_float(value: object, default: float) -> float:
    if not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _legacy_scenario(data: dict[str, object]) -> Scenario:
    phase1 = _coerce_string_list(data.get("phase1"))
    phase2 = _coerce_string_list(data.get("phase2"))
    final_question = str(data.get("final_question") or "").strip()
    turns: list[dict[str, object]] = [
        {"role": "user", "content": text}
        for text in phase1
    ]
    turns.append({"action": "consolidate", "force": False, "archive_all": False})
    turns.extend({"role": "user", "content": text} for text in phase2)
    if final_question:
        turns.append({"role": "user", "content": final_question, "final": True})
    return Scenario(
        name=str(data.get("name") or "legacy"),
        turns=turns,
    )


def _load_scenario(path: Path | None) -> Scenario:
    if path is None:
        raise SystemExit("必须通过 --messages 指定场景 JSON")
    data = cast(object, json.loads(path.read_text(encoding="utf-8")))
    if not isinstance(data, dict):
        raise SystemExit("场景 JSON 顶层必须是 object")
    scenario_data = _coerce_object_dict(cast(object, data))
    if "turns" not in scenario_data:
        return _legacy_scenario(scenario_data)
    turns = scenario_data.get("turns")
    if not isinstance(turns, list) or not turns:
        raise SystemExit("场景 JSON 需要非空 turns 数组")
    turn_items = cast(list[object], turns)
    normalized_turns: list[dict[str, object]] = []
    for index, item in enumerate(turn_items, 1):
        if not isinstance(item, dict):
            raise SystemExit(f"turns[{index}] 必须是 object")
        turn = _coerce_object_dict(cast(object, item))
        if turn.get("action") == "consolidate":
            normalized_turns.append(
                {
                    "action": "consolidate",
                    "force": bool(turn.get("force", False)),
                    "archive_all": bool(turn.get("archive_all", False)),
                    "label": str(turn.get("label") or "manual"),
                }
            )
            continue
        if turn.get("action") == "wait":
            normalized_turns.append(
                {
                    "action": "wait",
                    "seconds": _coerce_float(turn.get("seconds"), 1.0),
                    "label": str(turn.get("label") or "wait"),
                }
            )
            continue
        content = str(turn.get("content") or "").strip()
        if not content:
            raise SystemExit(f"turns[{index}] 缺少 content")
        normalized_turns.append(
            {
                "role": "user",
                "content": content,
                "final": bool(turn.get("final", False)),
                "label": str(turn.get("label") or ""),
            }
        )
    return Scenario(
        name=str(scenario_data.get("name") or path.stem),
        turns=normalized_turns,
    )


def _disable_qq_config(config_path: Path) -> str | None:
    text = config_path.read_text(encoding="utf-8")
    marker = "[channels.qq]\n"
    if marker not in text:
        return None
    head, tail = text.split(marker, 1)
    section, sep, rest = tail.partition("\n[")
    if "enabled = false" in section:
        return None
    if re.search(r"(?m)^enabled\s*=", section):
        section = re.sub(r"(?m)^enabled\s*=.*$", "enabled = false", section, count=1)
    else:
        section = "enabled = false\n" + section
    patched = head + marker + section + (sep + rest if sep else "")
    _ = config_path.write_text(patched, encoding="utf-8")
    return text


async def _send_and_read(
    writer: asyncio.StreamWriter,
    reader: asyncio.StreamReader,
    text: str,
    timeout: int,
) -> str:
    writer.write((json.dumps({"content": text}, ensure_ascii=False) + "\n").encode())
    await writer.drain()
    line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    if not line:
        return "（连接已断开）"
    data = json.loads(line)
    return str(data.get("content") or "")


def _latest_session_key(db_path: Path) -> str:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "select key from sessions order by updated_at desc limit 1"
        ).fetchone()
        return str(row[0]) if row else ""
    finally:
        conn.close()


def _probe_session_key(db_path: Path, previous: str) -> str:
    current = _latest_session_key(db_path)
    if not current:
        raise SystemExit("未找到当前探针 session")
    if previous and current != previous:
        raise SystemExit(f"探针 session 发生变化: {previous} -> {current}")
    return current


def _dashboard_consolidate(
    *,
    dashboard_url: str,
    session_key: str,
    force: bool,
    archive_all: bool,
    timeout: int,
) -> dict[str, Any]:
    quoted = urllib.parse.quote(session_key, safe="")
    req = urllib.request.Request(
        f"{dashboard_url.rstrip('/')}/api/dashboard/sessions/{quoted}/consolidate",
        data=json.dumps({"force": force, "archive_all": archive_all}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _tool_rows(observe_db: Path, session_key: str) -> list[dict[str, Any]]:
    if not observe_db.exists():
        return []
    conn = sqlite3.connect(observe_db)
    try:
        rows = conn.execute(
            """
            select id, user_msg,
                   case
                     when tool_calls is null or tool_calls='' or tool_calls='[]'
                     then 0
                     else json_array_length(tool_calls)
                   end,
                   coalesce(error, '')
            from turns
            where session_key = ?
            order by id
            """,
            (session_key,),
        ).fetchall()
        return [
            {
                "turn": int(row[0]),
                "user": str(row[1] or ""),
                "tool_calls": int(row[2] or 0),
                "error": str(row[3] or ""),
            }
            for row in rows
        ]
    finally:
        conn.close()


def _memory_rows(
    *,
    memory_db: Path,
    observe_db: Path,
    session_key: str,
    started_at: str,
    baseline: dict[str, str],
) -> list[dict[str, str]]:
    if not memory_db.exists():
        return []
    writes = _memory_write_rows(observe_db, session_key, started_at)
    write_rows = _memory_rows_from_writes(memory_db, writes) if writes else []
    updated_rows = _memory_rows_changed_since(memory_db, baseline)
    return _dedupe_memory_rows([*write_rows, *updated_rows])


def _memory_baseline(memory_db: Path) -> dict[str, str]:
    if not memory_db.exists():
        return {}
    conn = sqlite3.connect(memory_db)
    try:
        rows = conn.execute("select id, updated_at from memory_items").fetchall()
        return {str(row[0]): str(row[1] or "") for row in rows}
    except sqlite3.Error:
        return {}
    finally:
        conn.close()


def _memory_rows_changed_since(
    memory_db: Path,
    baseline: dict[str, str],
) -> list[dict[str, str]]:
    conn = sqlite3.connect(memory_db)
    try:
        rows = conn.execute(
            """
            select id, memory_type, summary, source_ref, updated_at
            from memory_items
            order by updated_at
            """
        ).fetchall()
        return [
            {
                "item_id": str(row[0] or ""),
                "memory_type": str(row[1] or ""),
                "summary": str(row[2] or ""),
                "source_ref": str(row[3] or ""),
            }
            for row in rows
            if baseline.get(str(row[0])) != str(row[4] or "")
        ]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _dedupe_memory_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        raw_item_id = row.get("item_id", "")
        item_id = raw_item_id.split(":", 1)[1] if ":" in raw_item_id else raw_item_id
        key = (
            item_id,
            row.get("memory_type", ""),
            row.get("summary", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _memory_write_rows(
    observe_db: Path,
    session_key: str,
    started_at: str,
) -> list[dict[str, str]]:
    if not observe_db.exists():
        return []
    conn = sqlite3.connect(observe_db)
    try:
        rows = conn.execute(
            """
            select action, item_id, memory_type, summary, source_ref
            from memory_writes
            where session_key = ? and ts >= ?
            order by id
            """,
            (session_key, started_at),
        ).fetchall()
        return [
            {
                "action": str(row[0] or ""),
                "item_id": str(row[1] or ""),
                "memory_type": str(row[2] or ""),
                "summary": str(row[3] or ""),
                "source_ref": str(row[4] or ""),
            }
            for row in rows
        ]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _memory_rows_from_writes(
    memory_db: Path,
    writes: list[dict[str, str]],
) -> list[dict[str, str]]:
    item_ids = [
        item_id.split(":", 1)[1]
        for row in writes
        if (item_id := row.get("item_id", "")).startswith(("new:", "reinforced:"))
    ]
    items: dict[str, dict[str, str]] = {}
    if item_ids:
        conn = sqlite3.connect(memory_db)
        try:
            placeholders = ",".join("?" for _ in item_ids)
            rows = conn.execute(
                f"""
                select id, memory_type, summary, source_ref
                from memory_items
                where id in ({placeholders})
                """,
                tuple(item_ids),
            ).fetchall()
            items = {
                str(row[0]): {
                    "memory_type": str(row[1] or ""),
                    "summary": str(row[2] or ""),
                    "source_ref": str(row[3] or ""),
                }
                for row in rows
            }
        finally:
            conn.close()
    result: list[dict[str, str]] = []
    for row in writes:
        raw_item_id = row.get("item_id", "")
        item_id = raw_item_id.split(":", 1)[1] if ":" in raw_item_id else raw_item_id
        item = items.get(item_id, {})
        result.append(
            {
                "action": row.get("action", ""),
                "item_id": raw_item_id,
                "memory_type": item.get("memory_type") or row.get("memory_type", ""),
                "summary": item.get("summary") or row.get("summary", ""),
                "source_ref": item.get("source_ref") or row.get("source_ref", ""),
            }
        )
    return result


def _write_reports(
    *,
    report_md: Path,
    report_json: Path,
    profile: str,
    scenario: Scenario,
    session_key: str,
    records: list[dict[str, str]],
    consolidate_result: dict[str, Any] | None,
    recent_after_consolidate: str,
    recent_final: str,
    tools: list[dict[str, Any]],
    memories: list[dict[str, str]],
) -> None:
    payload: dict[str, object] = {
        "profile": profile,
        "scenario": {
            "name": scenario.name,
        },
        "session_key": session_key,
        "manual_consolidate": consolidate_result,
        "records": records,
        "tool_calls": tools,
        "memory_items": memories,
        "recent_context_after_consolidate": recent_after_consolidate,
        "recent_context_final": recent_final,
    }
    _ = report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        f"# context probe: {profile}",
        "",
        f"- scenario: {scenario.name}",
        f"- session_key: {session_key}",
        f"- manual_consolidate: {json.dumps(consolidate_result, ensure_ascii=False)}",
        "",
        "## 对话记录",
        "",
    ]
    for index, row in enumerate(records, 1):
        lines.extend(
            [
                f"### Turn {index}",
                "",
                "user:",
                "",
                row["user"],
                "",
                "assistant:",
                "",
                row["assistant"],
                "",
            ]
        )
    lines.extend(["## Tool Calls", ""])
    if tools:
        for row in tools:
            lines.append(
                f"- turn {row['turn']}: tools={row['tool_calls']} "
                f"error={row['error']} user={row['user'][:100]}"
            )
    else:
        lines.append("- none")
    lines.extend(["", "## Memory Items", ""])
    if memories:
        for row in memories:
            lines.append(f"- [{row['memory_type']}] {row['summary']}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Recent Context After Manual Consolidate",
            "",
            "```text",
            recent_after_consolidate.rstrip(),
            "```",
            "",
            "## Recent Context Final",
            "",
            "```text",
            recent_final.rstrip(),
            "```",
            "",
        ]
    )
    _ = report_md.write_text("\n".join(lines), encoding="utf-8")


async def _run_probe(args: argparse.Namespace) -> None:
    paths = ProbePaths(
        repo=_repo_root(),
        debug_dir=Path(__file__).resolve().parent,
        profile=args.profile,
    )
    if not paths.config.exists():
        raise SystemExit(f"缺少 profile config: {paths.config}")

    original_config: str | None = None
    proc: subprocess.Popen[bytes] | None = None
    if args.disable_qq:
        original_config = _disable_qq_config(paths.config)
    try:
        if args.reset_workspace:
            _run_compose(
                paths,
                ["run", "--rm", "akashic-debug", "reset-workspace"],
            )
        if args.start_agent:
            proc = subprocess.Popen(
                [
                    "docker",
                    "compose",
                    "-f",
                    str(paths.debug_dir / "docker-compose.yml"),
                    "up",
                    "akashic-debug",
                ],
                cwd=paths.repo,
                env={
                    **dict(os.environ),
                    "AKASHIC_DEBUG_PROFILE": paths.profile,
                },
                stdout=subprocess.DEVNULL if args.quiet_agent else None,
                stderr=subprocess.STDOUT if args.quiet_agent else None,
            )
            deadline = time.time() + args.start_timeout
            while time.time() < deadline and not paths.socket.exists():
                if proc.poll() is not None:
                    raise SystemExit("agent 启动失败，docker compose 已退出")
                await asyncio.sleep(0.5)
            if not paths.socket.exists():
                raise SystemExit(f"等待 socket 超时: {paths.socket}")

        scenario = _load_scenario(args.messages)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        memory_baseline = _memory_baseline(paths.memory_db)
        records: list[dict[str, str]] = []
        consolidate_result: dict[str, Any] | None = None
        recent_after_consolidate = ""
        session_key = ""
        reader, writer = await asyncio.open_unix_connection(str(paths.socket))
        try:
            for index, turn in enumerate(scenario.turns, 1):
                if turn.get("action") == "consolidate":
                    session_key = _probe_session_key(paths.sessions_db, session_key)
                    consolidate_result = _dashboard_consolidate(
                        dashboard_url=args.dashboard_url,
                        session_key=session_key,
                        force=bool(turn.get("force", args.force_consolidate)),
                        archive_all=bool(turn.get("archive_all", args.archive_all)),
                        timeout=args.turn_timeout,
                    )
                    await asyncio.sleep(args.after_consolidate_wait)
                    recent_after_consolidate = (
                        paths.recent_context.read_text(encoding="utf-8")
                        if paths.recent_context.exists()
                        else ""
                    )
                    print(f"turn {index} consolidate ok")
                    continue
                if turn.get("action") == "wait":
                    await asyncio.sleep(_coerce_float(turn.get("seconds"), 1.0))
                    print(f"turn {index} wait ok")
                    continue
                text = str(turn.get("content") or "").strip()
                reply = await _send_and_read(writer, reader, text, args.turn_timeout)
                session_key = _probe_session_key(paths.sessions_db, session_key)
                records.append({"user": text, "assistant": reply})
                print(f"turn {index} ok: {reply[:80]}")
        finally:
            writer.close()
            await writer.wait_closed()

        await asyncio.sleep(args.after_final_wait)
        session_key = _probe_session_key(paths.sessions_db, session_key)
        report_base = args.output or paths.workspace / f"context-probe-{paths.profile}"
        report_md = report_base.with_suffix(".md")
        report_json = report_base.with_suffix(".json")
        _ = report_md.parent.mkdir(parents=True, exist_ok=True)
        recent_final = (
            paths.recent_context.read_text(encoding="utf-8")
            if paths.recent_context.exists()
            else ""
        )
        tools = _tool_rows(paths.observe_db, session_key)
        memories = _memory_rows(
            memory_db=paths.memory_db,
            observe_db=paths.observe_db,
            session_key=session_key,
            started_at=started_at,
            baseline=memory_baseline,
        )
        _write_reports(
            report_md=report_md,
            report_json=report_json,
            profile=paths.profile,
            scenario=scenario,
            session_key=session_key,
            records=records,
            consolidate_result=consolidate_result,
            recent_after_consolidate=recent_after_consolidate,
            recent_final=recent_final,
            tools=tools,
            memories=memories,
        )
        print(f"markdown: {report_md}")
        print(f"json: {report_json}")

        if proc is not None and args.stop_agent:
            _run_compose(paths, ["down"])
            proc = None
    finally:
        if proc is not None and args.stop_agent:
            _run_compose(paths, ["down"])
        if original_config is not None:
            _ = paths.config.write_text(original_config, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 docker/debug 沙盒上下文连续性探针。"
    )
    _ = parser.add_argument("--profile", default="default")
    _ = parser.add_argument("--messages", type=Path, required=True, help="场景 JSON")
    _ = parser.add_argument(
        "--output",
        type=Path,
        help="输出文件前缀，默认写到 profile workspace",
    )
    _ = parser.add_argument("--dashboard-url", default="http://127.0.0.1:2237")
    _ = parser.add_argument("--turn-timeout", type=int, default=240)
    _ = parser.add_argument("--start-timeout", type=int, default=60)
    _ = parser.add_argument("--after-consolidate-wait", type=float, default=3.0)
    _ = parser.add_argument("--after-final-wait", type=float, default=2.0)
    _ = parser.add_argument("--force-consolidate", action="store_true")
    _ = parser.add_argument("--archive-all", action="store_true")
    _ = parser.add_argument("--reset-workspace", action="store_true")
    _ = parser.add_argument("--start-agent", action="store_true")
    _ = parser.add_argument("--stop-agent", action="store_true")
    _ = parser.add_argument("--quiet-agent", action="store_true")
    _ = parser.add_argument(
        "--disable-qq",
        action="store_true",
        help="运行期间临时给 [channels.qq] 加 enabled=false，结束后恢复。",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run_probe(_parse_args()))


if __name__ == "__main__":
    main()
