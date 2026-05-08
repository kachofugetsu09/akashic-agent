#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    def sessions_db(self) -> Path:
        return self.workspace / "sessions.db"

    @property
    def observe_db(self) -> Path:
        return self.workspace / "observe" / "observe.db"

    @property
    def memory_db(self) -> Path:
        return self.workspace / "memory" / "memory2.db"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_compose(paths: ProbePaths, args: list[str]) -> None:
    _ = subprocess.run(
        ["docker", "compose", "-f", str(paths.debug_dir / "docker-compose.yml"), *args],
        cwd=paths.repo,
        env={**dict(os.environ), "AKASHIC_DEBUG_PROFILE": paths.profile},
        check=True,
    )


def _bootstrap_profile(paths: ProbePaths, from_profile: str | None) -> None:
    if paths.config.exists():
        return
    if not from_profile:
        raise SystemExit(f"缺少 profile config: {paths.config}")
    src = paths.debug_dir / "profiles" / from_profile / "config.toml"
    if not src.exists():
        raise SystemExit(f"缺少 bootstrap config: {src}")
    paths.profile_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, paths.config)


def _replace_section_value(
    text: str,
    section_name: str,
    key: str,
    value: str,
) -> str:
    marker = f"[{section_name}]\n"
    if marker not in text:
        return text
    head, tail = text.split(marker, 1)
    section, sep, rest = tail.partition("\n[")
    pattern = rf"(?m)^{re.escape(key)}\s*=.*$"
    replacement = f"{key} = {value}"
    if re.search(pattern, section):
        section = re.sub(pattern, replacement, section, count=1)
    else:
        section = replacement + "\n" + section
    return head + marker + section + (sep + rest if sep else "")


def _isolate_cli_config(config_path: Path) -> str:
    original = config_path.read_text(encoding="utf-8")
    text = original
    text = _replace_section_value(text, "channels.telegram", "token", '""')
    text = _replace_section_value(text, "channels.qq", "bot_uin", '""')
    text = _replace_section_value(text, "channels.qqbot", "app_id", '""')
    text = _replace_section_value(text, "proactive", "enabled", "false")
    config_path.write_text(text, encoding="utf-8")
    return original


def _connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _latest_cli_session_key(db_path: Path) -> str:
    conn = _connect_db(db_path)
    try:
        row = conn.execute(
            """
            select key
            from sessions
            where key like 'cli:%'
            order by updated_at desc
            limit 1
            """
        ).fetchone()
        return str(row["key"]) if row else ""
    finally:
        conn.close()


def _json_loads(value: object) -> Any:
    if not isinstance(value, str) or not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _session_messages(db_path: Path, session_key: str) -> list[dict[str, Any]]:
    conn = _connect_db(db_path)
    try:
        rows = conn.execute(
            """
            select seq, role, content, tool_chain, extra, ts
            from messages
            where session_key = ?
            order by seq
            """,
            (session_key,),
        ).fetchall()
    finally:
        conn.close()
    result: list[dict[str, Any]] = []
    for row in rows:
        extra = _json_loads(row["extra"]) or {}
        result.append(
            {
                "seq": int(row["seq"]),
                "role": str(row["role"]),
                "content": str(row["content"] or ""),
                "tool_chain": _json_loads(row["tool_chain"]),
                "extra": extra,
                "ts": str(row["ts"]),
            }
        )
    return result


def _observe_turns(db_path: Path, session_key: str) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = _connect_db(db_path)
    try:
        rows = conn.execute(
            """
            select id, user_msg, tool_calls, error
            from turns
            where session_key = ?
            order by id
            """,
            (session_key,),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [
        {
            "id": int(row["id"]),
            "user": str(row["user_msg"] or ""),
            "tool_calls": _json_loads(row["tool_calls"]) or [],
            "error": str(row["error"] or ""),
        }
        for row in rows
    ]


def _memory_baseline(memory_db: Path) -> dict[str, str]:
    if not memory_db.exists():
        return {}
    conn = _connect_db(memory_db)
    try:
        rows = conn.execute("select id, updated_at from memory_items").fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()
    return {str(row["id"]): str(row["updated_at"] or "") for row in rows}


def _changed_memory_items(
    memory_db: Path,
    baseline: dict[str, str],
) -> list[dict[str, str]]:
    if not memory_db.exists():
        return []
    conn = _connect_db(memory_db)
    try:
        rows = conn.execute(
            """
            select id, memory_type, summary, source_ref, updated_at
            from memory_items
            order by updated_at
            """
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [
        {
            "id": str(row["id"]),
            "memory_type": str(row["memory_type"] or ""),
            "summary": str(row["summary"] or ""),
            "source_ref": str(row["source_ref"] or ""),
        }
        for row in rows
        if baseline.get(str(row["id"])) != str(row["updated_at"] or "")
    ]


def _memory_writes(
    observe_db: Path,
    session_key: str,
    started_at: str,
) -> list[dict[str, str]]:
    if not observe_db.exists():
        return []
    conn = _connect_db(observe_db)
    try:
        rows = conn.execute(
            """
            select action, item_id, memory_type, summary, source_ref, ts
            from memory_writes
            where session_key = ? and ts >= ?
            order by id
            """,
            (session_key, started_at),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return [
        {
            "action": str(row["action"] or ""),
            "item_id": str(row["item_id"] or ""),
            "memory_type": str(row["memory_type"] or ""),
            "summary": str(row["summary"] or ""),
            "source_ref": str(row["source_ref"] or ""),
            "ts": str(row["ts"] or ""),
        }
        for row in rows
    ]


async def _read_assistant(
    reader: asyncio.StreamReader,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = await asyncio.wait_for(reader.readline(), timeout=deadline - time.monotonic())
        if not line:
            raise RuntimeError("CLI 连接已断开")
        data = json.loads(line)
        if data.get("type") == "assistant":
            return {
                "content": str(data.get("content") or ""),
                "metadata": data.get("metadata") or {},
            }
    raise TimeoutError("等待 assistant 回复超时")


async def _send(
    writer: asyncio.StreamWriter,
    text: str,
) -> None:
    writer.write((json.dumps({"content": text}, ensure_ascii=False) + "\n").encode())
    await writer.drain()


def _extract_marker(text: str) -> str:
    match = re.search(r"AKASHIC_BG_DONE_[0-9a-f]{32}", text)
    return match.group(0) if match else ""


def _has_shell_tool(msg: dict[str, Any]) -> bool:
    extra = msg.get("extra") if isinstance(msg.get("extra"), dict) else {}
    tools_used = extra.get("tools_used") if isinstance(extra, dict) else None
    if isinstance(tools_used, list) and "shell" in tools_used:
        return True
    chain = msg.get("tool_chain")
    if not isinstance(chain, list):
        return False
    for group in chain:
        if not isinstance(group, dict):
            continue
        calls = group.get("calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if isinstance(call, dict) and call.get("name") == "shell":
                return True
    return False


def _build_checks(
    *,
    responses: list[dict[str, Any]],
    messages: list[dict[str, Any]],
    memory_writes: list[dict[str, str]],
    memory_items: list[dict[str, str]],
) -> dict[str, Any]:
    completion_messages = [
        msg
        for msg in messages
        if msg["role"] == "assistant" and "后台命令已" in msg["content"]
    ]
    first_shell_messages = [
        msg
        for msg in messages
        if msg["role"] == "assistant" and _has_shell_tool(msg)
    ]
    marker = ""
    for row in responses:
        marker = _extract_marker(row["content"])
        if marker:
            break
    memory_blob = "\n".join(
        [
            *(row["summary"] + "\n" + row["source_ref"] for row in memory_writes),
            *(row["summary"] + "\n" + row["source_ref"] for row in memory_items),
        ]
    )
    fake_user_rows = [
        msg
        for msg in messages
        if msg["role"] == "user" and "[后台 shell 完成]" in msg["content"]
    ]
    completion_uses_shell = [
        msg
        for msg in completion_messages
        if _has_shell_tool(msg)
    ]
    checks = {
        "first_turn_used_shell": bool(first_shell_messages),
        "has_auto_completion_message": bool(completion_messages),
        "no_fake_user_shell_completion": not fake_user_rows,
        "completion_tools_used_is_empty": not completion_uses_shell,
        "completion_marker_observed": bool(marker),
        "completion_marker_not_in_memory": bool(marker) and marker not in memory_blob,
        "assistant_completion_count": len(completion_messages),
        "exact_marker": marker,
    }
    checks["passed"] = all(
        bool(checks[name])
        for name in (
            "first_turn_used_shell",
            "has_auto_completion_message",
            "no_fake_user_shell_completion",
            "completion_tools_used_is_empty",
            "completion_marker_observed",
            "completion_marker_not_in_memory",
        )
    )
    return checks


def _write_report(
    *,
    report_base: Path,
    payload: dict[str, Any],
) -> None:
    report_base.parent.mkdir(parents=True, exist_ok=True)
    report_json = report_base.with_suffix(".json")
    report_md = report_base.with_suffix(".md")
    report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# shell background probe",
        "",
        "```text",
        "docker/debug profile",
        "  |",
        "  +-- cli user prompt",
        "  |     |",
        "  |     +-- shell auto-promote after 15s",
        "  |",
        "  +-- shell completion inbound",
        "        |",
        "        +-- assistant-only persistence",
        "```",
        "",
        f"- profile: {payload['profile']}",
        f"- session_key: {payload['session_key']}",
        f"- passed: {payload['checks']['passed']}",
        f"- exact_marker: {payload['checks']['exact_marker']}",
        "",
        "## Checks",
        "",
    ]
    for key, value in payload["checks"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Responses", ""])
    for index, row in enumerate(payload["responses"], 1):
        lines.extend([f"### Response {index}", "", row["content"], ""])
    lines.extend(["## Session Messages", ""])
    for row in payload["session_messages"]:
        content = row["content"].replace("\n", "\\n")
        if len(content) > 180:
            content = content[:180] + "..."
        tools_used = row["extra"].get("tools_used") if isinstance(row["extra"], dict) else None
        lines.append(
            f"- seq={row['seq']} role={row['role']} tools_used={tools_used} content={content}"
        )
    lines.extend(["", "## Memory Writes", ""])
    if payload["memory_writes"]:
        for row in payload["memory_writes"]:
            lines.append(f"- {row['action']} [{row['memory_type']}] {row['summary']}")
    else:
        lines.append("- none")
    lines.extend(["", "## Changed Memory Items", ""])
    if payload["changed_memory_items"]:
        for row in payload["changed_memory_items"]:
            lines.append(f"- [{row['memory_type']}] {row['summary']}")
    else:
        lines.append("- none")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"markdown: {report_md}")
    print(f"json: {report_json}")


async def _run_probe(args: argparse.Namespace) -> None:
    paths = ProbePaths(
        repo=_repo_root(),
        debug_dir=Path(__file__).resolve().parent,
        profile=args.profile,
    )
    _bootstrap_profile(paths, args.bootstrap_from)
    original_config = _isolate_cli_config(paths.config) if args.isolate_channels else None
    proc: subprocess.Popen[bytes] | None = None
    try:
        if args.reset_workspace:
            _run_compose(paths, ["run", "--rm", "akashic-debug", "reset-workspace"])
        if args.start_agent:
            paths.socket.unlink(missing_ok=True)
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
                env={**dict(os.environ), "AKASHIC_DEBUG_PROFILE": paths.profile},
                stdout=subprocess.DEVNULL if args.quiet_agent else None,
                stderr=subprocess.STDOUT if args.quiet_agent else None,
            )
            deadline = time.monotonic() + args.start_timeout
            while time.monotonic() < deadline and not paths.socket.exists():
                if proc.poll() is not None:
                    raise SystemExit("agent 启动失败，docker compose 已退出")
                await asyncio.sleep(0.5)
            if not paths.socket.exists():
                raise SystemExit(f"等待 socket 超时: {paths.socket}")

        started_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
        memory_baseline = _memory_baseline(paths.memory_db)
        reader, writer = await asyncio.open_unix_connection(str(paths.socket))
        command = (
            "python -c 'import time, uuid; "
            'print("AKASHIC_BG_START", flush=True); '
            "time.sleep(18); "
            'print("AKASHIC_BG_DONE_"+uuid.uuid4().hex, flush=True)' "'"
        )
        prompt = (
            "请真实调用 shell 工具执行下面这个命令，不要用文字模拟。"
            "不要设置 timeout，不要设置 run_in_background=true，让它按默认 15 秒自动转后台。"
            "拿到 background_task_id 后只告诉我任务已转后台，不要调用 task_output。\n\n"
            f"命令：{command}"
        )
        responses: list[dict[str, Any]] = []
        try:
            await _send(writer, prompt)
            first = await _read_assistant(reader, args.turn_timeout)
            responses.append(first)
            print(f"first response: {first['content'][:120]}")
            deadline = time.monotonic() + args.completion_timeout
            while time.monotonic() < deadline:
                item = await _read_assistant(
                    reader,
                    max(1.0, deadline - time.monotonic()),
                )
                responses.append(item)
                print(f"completion candidate: {item['content'][:120]}")
                if _extract_marker(item["content"]) or "后台命令已" in item["content"]:
                    break
            session_key = _latest_cli_session_key(paths.sessions_db)
            if not session_key:
                raise SystemExit("未找到 CLI session")
        finally:
            writer.close()
            await writer.wait_closed()

        await asyncio.sleep(args.after_completion_wait)
        messages = _session_messages(paths.sessions_db, session_key)
        observe_turns = _observe_turns(paths.observe_db, session_key)
        writes = _memory_writes(paths.observe_db, session_key, started_at)
        changed_items = _changed_memory_items(paths.memory_db, memory_baseline)
        checks = _build_checks(
            responses=responses,
            messages=messages,
            memory_writes=writes,
            memory_items=changed_items,
        )
        payload = {
            "profile": paths.profile,
            "session_key": session_key,
            "prompt": prompt,
            "responses": responses,
            "session_messages": messages,
            "observe_turns": observe_turns,
            "memory_writes": writes,
            "changed_memory_items": changed_items,
            "checks": checks,
        }
        report_base = args.output or paths.workspace / f"shell-background-probe-{paths.profile}"
        _write_report(report_base=report_base, payload=payload)
        if not checks["passed"]:
            raise SystemExit("shell background probe failed")
    finally:
        if proc is not None and args.stop_agent:
            _run_compose(paths, ["down"])
        if original_config is not None:
            paths.config.write_text(original_config, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行后台 shell 完成回灌真实探针。")
    parser.add_argument("--profile", default="shell-bg-probe")
    parser.add_argument("--bootstrap-from", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--turn-timeout", type=float, default=180)
    parser.add_argument("--completion-timeout", type=float, default=90)
    parser.add_argument("--start-timeout", type=float, default=90)
    parser.add_argument("--after-completion-wait", type=float, default=2)
    parser.add_argument("--reset-workspace", action="store_true")
    parser.add_argument("--start-agent", action="store_true")
    parser.add_argument("--stop-agent", action="store_true")
    parser.add_argument("--quiet-agent", action="store_true")
    parser.add_argument("--isolate-channels", action="store_true", default=True)
    parser.add_argument("--no-isolate-channels", dest="isolate_channels", action="store_false")
    return parser.parse_args()


def main() -> None:
    asyncio.run(_run_probe(_parse_args()))


if __name__ == "__main__":
    main()
