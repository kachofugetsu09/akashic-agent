#!/usr/bin/env python3
"""
shell_background_probe.py

验证新版 shell 后台设计：agent 能自主发现卡死任务并调用 task_stop，
整个生命周期在单次 turn 内完成，不依赖系统自动回调。

被测行为：
  1. agent 对外观无害的命令启动 shell
  2. 命令 15s 后 auto-promote 转后台，agent 拿到 task_id
  3. agent 用 task_output 轮询，发现从无输出（since_last_output_ms=null）
  4. agent 判定卡死，调用 task_stop
  5. agent 给用户一条最终回复，整个 turn 内完成，无孤悬后台进程
"""
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


# ── 路径 ─────────────────────────────────────────────────────────────

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


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# ── Docker / 配置 ─────────────────────────────────────────────────────

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


def _replace_section_value(text: str, section_name: str, key: str, value: str) -> str:
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
    text = _replace_section_value(text, "plugins.qqbot", "app_id", '""')
    text = _replace_section_value(text, "proactive", "enabled", "false")
    config_path.write_text(text, encoding="utf-8")
    return original


# ── DB ────────────────────────────────────────────────────────────────

def _connect_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=10000")
    return conn


def _latest_cli_session_key(db_path: Path) -> str:
    conn = _connect_db(db_path)
    try:
        row = conn.execute(
            "select key from sessions where key like 'cli:%' order by updated_at desc limit 1"
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
            "select seq, role, content, tool_chain, extra, ts from messages "
            "where session_key = ? order by seq",
            (session_key,),
        ).fetchall()
    finally:
        conn.close()
    result: list[dict[str, Any]] = []
    for row in rows:
        result.append({
            "seq": int(row["seq"]),
            "role": str(row["role"]),
            "content": str(row["content"] or ""),
            "tool_chain": _json_loads(row["tool_chain"]),
            "extra": _json_loads(row["extra"]) or {},
            "ts": str(row["ts"]),
        })
    return result


# ── 工具调用提取 ──────────────────────────────────────────────────────

def _flatten_calls(tool_chain: Any) -> list[dict[str, Any]]:
    """从 tool_chain 里按顺序提取所有工具调用。"""
    if not isinstance(tool_chain, list):
        return []
    calls: list[dict[str, Any]] = []
    for group in tool_chain:
        if not isinstance(group, dict):
            continue
        for call in group.get("calls") or []:
            if isinstance(call, dict):
                calls.append(call)
    return calls


def _calls_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for msg in messages:
        if msg["role"] == "assistant":
            calls.extend(_flatten_calls(msg.get("tool_chain")))
    return calls


# ── Checks ────────────────────────────────────────────────────────────

def _build_checks(
    *,
    responses: list[dict[str, Any]],
    messages: list[dict[str, Any]],
) -> dict[str, Any]:
    all_calls = _calls_from_messages(messages)
    call_names = [c.get("name", "") for c in all_calls]

    shell_idx = next((i for i, n in enumerate(call_names) if n == "shell"), -1)
    task_output_idx = next((i for i, n in enumerate(call_names) if n == "task_output"), -1)
    task_stop_idx = next((i for i, n in enumerate(call_names) if n == "task_stop"), -1)

    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    # 旧格式消息："后台命令已..." 由已删除的 process_shell_completion_event 生成
    old_format_messages = [
        m for m in assistant_messages if "后台命令已" in m["content"]
    ]

    # task_output 调用里有没有拿到 status=done（不应该，死循环不会自然结束）
    task_output_done = any(
        _json_loads(c.get("output", "")) is not None
        and isinstance(_json_loads(c.get("output", "")), dict)
        and _json_loads(c.get("output", "")).get("status") == "done"
        for c in all_calls
        if c.get("name") == "task_output"
    )

    checks: dict[str, Any] = {
        # 核心行为验证
        "shell_called": shell_idx >= 0,
        "task_output_called": task_output_idx >= 0,
        "task_stop_called": task_stop_idx >= 0,
        # 顺序：shell → task_output → task_stop
        "correct_call_order": (
            shell_idx >= 0
            and task_output_idx > shell_idx
            and task_stop_idx > task_output_idx
        ),
        # 单 turn 内完成，无孤悬回调产生的额外消息
        "single_assistant_turn": len(assistant_messages) == 1,
        # 旧格式的自动回调消息不再出现
        "no_old_completion_format": len(old_format_messages) == 0,
        # 死循环不应该自然结束，agent 不应该看到 status=done
        "task_not_done_naturally": not task_output_done,
        # 调用序列摘要（供报告查看）
        "call_sequence": call_names,
    }
    checks["passed"] = all(
        bool(checks[k])
        for k in (
            "shell_called",
            "task_output_called",
            "task_stop_called",
            "correct_call_order",
            "single_assistant_turn",
            "no_old_completion_format",
        )
    )
    return checks


# ── CLI 通信 ──────────────────────────────────────────────────────────

async def _read_assistant(reader: asyncio.StreamReader, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        line = await asyncio.wait_for(reader.readline(), timeout=remaining)
        if not line:
            raise RuntimeError("CLI 连接已断开")
        data = json.loads(line)
        if data.get("type") == "assistant":
            return {
                "content": str(data.get("content") or ""),
                "metadata": data.get("metadata") or {},
            }
    raise TimeoutError("等待 assistant 回复超时")


async def _send(writer: asyncio.StreamWriter, text: str) -> None:
    writer.write((json.dumps({"content": text}, ensure_ascii=False) + "\n").encode())
    await writer.drain()


# ── 报告 ──────────────────────────────────────────────────────────────

def _write_report(*, report_base: Path, payload: dict[str, Any]) -> None:
    report_base.parent.mkdir(parents=True, exist_ok=True)
    report_json = report_base.with_suffix(".json")
    report_md = report_base.with_suffix(".md")
    report_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    checks = payload["checks"]
    lines = [
        "# shell background probe",
        "",
        "验证 agent 能自主发现卡死的后台 shell 任务并调用 task_stop，",
        "整个生命周期在单次 turn 内完成，不依赖系统自动回调。",
        "",
        "```",
        "shell(死循环命令)",
        "  └─ auto-promote 15s → task_id",
        "       └─ task_output(block=true) → since_last_output_ms=null",
        "            └─ task_stop → 最终回复（单 turn 完成）",
        "```",
        "",
        f"- profile: {payload['profile']}",
        f"- session_key: {payload['session_key']}",
        f"- passed: **{checks['passed']}**",
        "",
        "## Checks",
        "",
    ]
    for key, value in checks.items():
        if key == "call_sequence":
            lines.append(f"- call_sequence: {' → '.join(value) if value else '(empty)'}")
        else:
            icon = "✓" if value else "✗"
            lines.append(f"- {icon} {key}: {value}")
    lines.extend(["", "## Agent Response", ""])
    for i, r in enumerate(payload["responses"], 1):
        lines.extend([f"### Response {i}", "", r["content"], ""])
    lines.extend(["## Session Messages", ""])
    for row in payload["session_messages"]:
        content = row["content"].replace("\n", "\\n")
        if len(content) > 200:
            content = content[:200] + "..."
        tools = row["extra"].get("tools_used") if isinstance(row["extra"], dict) else None
        lines.append(f"- seq={row['seq']} role={row['role']} tools={tools} | {content}")
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"markdown: {report_md}")
    print(f"json:     {report_json}")


# ── 主流程 ────────────────────────────────────────────────────────────

# 死循环命令：外表像"运行中"，实际永远无输出
# iter(int, 1) 是永远不停的迭代器；time.sleep(1) 让进程安静占着 CPU
_HIDDEN_LOOP_COMMAND = "python3 -c \"import time; [time.sleep(1) for _ in iter(int, 1)]\""

# 发给 agent 的 prompt：不暴露循环意图，agent 只能通过 task_output 观察到什么都没输出
_PROBE_PROMPT = (
    "帮我运行下面这个命令，告诉我它的输出是什么。\n\n"
    f"命令：{_HIDDEN_LOOP_COMMAND}"
)


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
                ["docker", "compose", "-f", str(paths.debug_dir / "docker-compose.yml"),
                 "up", "akashic-debug"],
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

        reader, writer = await asyncio.open_unix_connection(str(paths.socket))
        responses: list[dict[str, Any]] = []
        try:
            print(f"发送 prompt（死循环命令，agent 不知情）：\n  {_HIDDEN_LOOP_COMMAND}\n")
            await _send(writer, _PROBE_PROMPT)
            # agent 需要：15s auto-promote + 至少一次 task_output + task_stop + 回复
            # 给充裕的 turn_timeout（默认 120s）
            response = await _read_assistant(reader, args.turn_timeout)
            responses.append(response)
            print(f"agent 回复：{response['content'][:200]}")
        finally:
            writer.close()
            await writer.wait_closed()

        await asyncio.sleep(2.0)  # 等 DB 写入

        session_key = _latest_cli_session_key(paths.sessions_db)
        if not session_key:
            raise SystemExit("未找到 CLI session")

        messages = _session_messages(paths.sessions_db, session_key)
        checks = _build_checks(responses=responses, messages=messages)

        print("\n── checks ──")
        for k, v in checks.items():
            if k == "call_sequence":
                print(f"  call_sequence: {' → '.join(v) if v else '(empty)'}")
            else:
                icon = "✓" if v else "✗"
                print(f"  {icon} {k}: {v}")
        print(f"\npassed: {checks['passed']}\n")

        payload = {
            "profile": paths.profile,
            "session_key": session_key,
            "prompt": _PROBE_PROMPT,
            "command": _HIDDEN_LOOP_COMMAND,
            "responses": responses,
            "session_messages": messages,
            "checks": checks,
        }
        report_base = args.output or paths.workspace / f"shell-background-probe-{paths.profile}"
        _write_report(report_base=report_base, payload=payload)

        if not checks["passed"]:
            raise SystemExit("shell background probe FAILED")
        print("shell background probe PASSED")

    finally:
        if proc is not None and args.stop_agent:
            _run_compose(paths, ["down"])
        if original_config is not None:
            paths.config.write_text(original_config, encoding="utf-8")


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="测试 agent 能否自主发现卡死后台 shell 任务并调用 task_stop。"
    )
    parser.add_argument("--profile", default="shell-bg-probe")
    parser.add_argument("--bootstrap-from", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--turn-timeout", type=float, default=120,
                        help="等待 agent 完成整个 turn 的超时秒数（含 auto-promote 15s + 轮询 + 回复）")
    parser.add_argument("--start-timeout", type=float, default=90)
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
