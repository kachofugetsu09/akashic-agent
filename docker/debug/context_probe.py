#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.control.client import ControlClient
from agent.plugins.manifest import builtin_plugin_data_dir
from plugins.akasha.config import load_akasha_config
from plugins.akasha.inspector import AkashaInspectorReader


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
    def observe_db(self) -> Path:
        return self.workspace / "observe" / "observe.db"

    @property
    def sessions_db(self) -> Path:
        return self.workspace / "sessions.db"


@dataclass
class Scenario:
    name: str
    turns: list[dict[str, object]]


_FAILED_REPLIES = frozenset(
    {
        "处理消息时出错，请稍后再试。",
        "模型流响应中断，请刷新对话重试。",
    }
)


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


def _ensure_successful_reply(reply: str, turn_index: int) -> None:
    if reply.strip() in _FAILED_REPLIES:
        raise RuntimeError(f"turn {turn_index} 返回运行时失败回复: {reply}")


def _legacy_scenario(data: dict[str, object]) -> Scenario:
    phase1 = _coerce_string_list(data.get("phase1"))
    phase2 = _coerce_string_list(data.get("phase2"))
    final_question = str(data.get("final_question") or "").strip()
    turns: list[dict[str, object]] = [
        {"role": "user", "content": text} for text in phase1
    ]
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
    client: ControlClient,
    thread_id: str,
    text: str,
    timeout: int,
) -> str:
    handle = await client.start_turn(thread_id, text)
    result = await asyncio.wait_for(handle.result(), timeout=timeout)
    return str(result.get("finalResponse") or "")


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


def _akasha_events(workspace: Path, session_key: str) -> list[dict[str, object]]:
    """Read the Akasha events committed for this probe session."""

    # 1. Resolve the same plugin-owned configuration and sidecars as runtime.
    data_root = builtin_plugin_data_dir("akasha", workspace)
    reader = AkashaInspectorReader(
        memory_root=workspace / "memory",
        config=load_akasha_config(data_root / "config.local.toml"),
    )

    # 2. Return the complete bounded probe session, failing on invalid sidecars.
    rows, total = reader.list_turns(session_key=session_key, page_size=50)
    if total > len(rows):
        raise RuntimeError(f"探针 session 的 Akasha 事件超过报告上限: {total}")
    return rows


def _write_reports(
    *,
    report_md: Path,
    report_json: Path,
    profile: str,
    scenario: Scenario,
    session_key: str,
    records: list[dict[str, str]],
    tools: list[dict[str, Any]],
    akasha_events: list[dict[str, object]],
) -> None:
    payload: dict[str, object] = {
        "profile": profile,
        "scenario": {
            "name": scenario.name,
        },
        "session_key": session_key,
        "records": records,
        "tool_calls": tools,
        "akasha_events": akasha_events,
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
    lines.extend(["", "## Akasha Events", ""])
    if akasha_events:
        for row in akasha_events:
            lines.append(
                f"- seq={row['seq']} seeds={row['seed_count']} "
                f"activations={row['activation_count']} query={row['query_text']}"
            )
    else:
        lines.append("- none")
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
        records: list[dict[str, str]] = []
        session_key = ""
        client = await ControlClient.connect(str(paths.socket))
        try:
            thread = await client.start_thread(
                {"probe": "context", "scenario": scenario.name}
            )
            session_key = str(thread["id"])
            for index, turn in enumerate(scenario.turns, 1):
                if turn.get("action") == "wait":
                    await asyncio.sleep(_coerce_float(turn.get("seconds"), 1.0))
                    print(f"turn {index} wait ok")
                    continue
                text = str(turn.get("content") or "").strip()
                reply = await _send_and_read(
                    client, session_key, text, args.turn_timeout
                )
                _ensure_successful_reply(reply, index)
                records.append({"user": text, "assistant": reply})
                print(f"turn {index} ok: {reply[:80]}")
        finally:
            await client.close()

        await asyncio.sleep(args.after_final_wait)
        report_base = args.output or paths.workspace / f"context-probe-{paths.profile}"
        report_md = report_base.with_suffix(".md")
        report_json = report_base.with_suffix(".json")
        _ = report_md.parent.mkdir(parents=True, exist_ok=True)
        tools = _tool_rows(paths.observe_db, session_key)
        akasha_events = _akasha_events(paths.workspace, session_key)
        _write_reports(
            report_md=report_md,
            report_json=report_json,
            profile=paths.profile,
            scenario=scenario,
            session_key=session_key,
            records=records,
            tools=tools,
            akasha_events=akasha_events,
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
    _ = parser.add_argument("--turn-timeout", type=int, default=240)
    _ = parser.add_argument("--start-timeout", type=int, default=60)
    _ = parser.add_argument("--after-final-wait", type=float, default=2.0)
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
