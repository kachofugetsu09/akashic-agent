"""Codex app-server adapter. Native sandbox is explicit; no parent tool grants."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast
from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESS_STREAMS
from plugins.jobs.contract import JOB_NOTICES
from plugins.tools.contract import TOOLS
from plugins.subagent.contract import AGENT_RUNNERS, AgentRunner, AgentTask, AgentResult, AgentActivity, AgentWire

api_version = 3
name = "subagent_codex"
version = "1.0.0"
desc = "委托原生 Codex；一次性只读任务，公开过程独立展示"
inject = (AGENT_RUNNERS, PROCESS_STREAMS, TOOLS, JOB_NOTICES)

# MCP servers, hooks, plugins and notify run outside the sandbox. Each job therefore gets
# a throwaway CODEX_HOME that holds only a link to the real auth.json (token refreshes
# write through), and the directory is removed when the wrapper exits.
# The group-wide SIGTERM on close must not kill rm itself, so cleanup ignores it first.
ISOLATED_HOME = r"""set -eu
src="${CODEX_HOME:-$HOME/.codex}"
home=""
trap 'trap "" TERM INT HUP; if [ -n "$home" ]; then rm -rf "$home"; fi' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP
mkdir -p "$HOME/.cache"
home=$(mktemp -d "$HOME/.cache/akashic-codex.XXXXXX")
if [ -f "$src/auth.json" ]; then ln -s "$src/auth.json" "$home/auth.json"; fi
CODEX_HOME="$home" "$@"
"""
LOCKED_CONFIG = ("-c", "mcp_servers={}", "-c", 'web_search="disabled"',
                 "--disable", "apps", "--disable", "plugins", "--disable", "hooks")


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executable: str = "codex"
    model: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=1800, gt=0, le=86400)


def obj(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("invalid Codex object")
    return cast(dict[str, object], value)


def argv(config: Config) -> tuple[str, ...]:
    return ("/bin/sh", "-c", ISOLATED_HOME, "codex-runner", config.executable,
            "app-server", "--listen", "stdio://", *LOCKED_CONFIG)


# JSON-RPC client half: requests wait for their reply, notifications are queued for the turn.
class Session:
    def __init__(self, wire: AgentWire) -> None:
        self.wire = wire
        self.serial = 0
        self.queued: list[dict[str, object]] = []

    async def request(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.serial += 1
        identity = self.serial
        await self.wire.send({"id": identity, "method": method, "params": params})
        while True:
            frame = await self.wire.receive()
            if frame.get("id") == identity and "method" not in frame:
                if "error" in frame:
                    raise RuntimeError("Codex request rejected: " + method)
                return obj(frame.get("result"))
            if "id" in frame and "method" in frame:
                await self.deny(frame)
            else:
                if len(self.queued) >= 1000:
                    raise ValueError("Codex startup event overflow")
                self.queued.append(frame)

    async def next(self) -> dict[str, object]:
        return self.queued.pop(0) if self.queued else await self.wire.receive()

    # Never approve writes, network escalation or interactive questions; unknown requests get an error reply.
    async def deny(self, frame: dict[str, object]) -> None:
        method = frame.get("method")
        if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
            await self.wire.send({"id": frame["id"], "result": {"decision": "decline"}})
        elif method == "item/tool/requestUserInput":
            await self.wire.send({"id": frame["id"], "result": {"answers": {}}})
        else:
            await self.wire.send({"id": frame["id"], "error": {
                "code": -32601, "message": "Unattended request unsupported"}})


# Handshake and thread start; fails closed if any MCP server survived the isolation.
async def open_thread(session: Session, task: AgentTask, config: Config) -> str:
    await session.request("initialize", {"clientInfo": {"name": "akashic", "version": "1.0.0"},
                                         "capabilities": {"experimentalApi": True}})
    await session.wire.send({"method": "initialized", "params": {}})
    params: dict[str, object] = {"cwd": task.cwd, "ephemeral": True,
                                  "approvalPolicy": "never", "sandbox": "read-only"}
    if config.model:
        params["model"] = config.model
    thread = obj((await session.request("thread/start", params)).get("thread"))
    thread_id = thread.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("Codex thread identity missing")
    servers = await session.request("mcpServerStatus/list", {"threadId": thread_id, "detail": "toolsAndAuthOnly"})
    if servers.get("data") != []:
        raise RuntimeError("Codex MCP servers present; refusing to run outside the read-only sandbox")
    return thread_id


async def execute(wire: AgentWire, task: AgentTask, config: Config,
                  emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
    session = Session(wire)
    thread_id = await open_thread(session, task, config)
    turn = obj((await session.request("turn/start", {"threadId": thread_id,
        "input": [{"type": "text", "text": task.prompt}]})).get("turn"))
    turn_id = turn.get("id")
    if not isinstance(turn_id, str) or not turn_id:
        raise ValueError("Codex turn identity missing")
    await emit(AgentActivity("status", "Codex 正在执行（原生只读沙箱）"))
    return await follow_turn(session, thread_id, turn_id, emit)


# Consume this turn's notifications until it completes; other threads and turns are ignored.
async def follow_turn(session: Session, thread_id: str, turn_id: str,
                      emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
    answers: dict[str, str] = {}
    while True:
        frame = await session.next()
        if "id" in frame and "method" in frame:
            await session.deny(frame)
            continue
        params = obj(frame.get("params", {}))
        if params.get("threadId") != thread_id:
            continue
        method = frame.get("method")
        if method == "turn/completed":
            terminal = obj(params.get("turn"))
            if terminal.get("id") != turn_id:
                continue
            text = answers.get("final_answer") or answers.get("partial_answer") or answers.get("unphased", "")
            if terminal.get("status") != "completed" or not text.strip():
                return AgentResult("failed", "Codex 未返回有效完成答案。")
            return AgentResult("completed", text)
        if params.get("turnId") == turn_id and method in {"item/started", "item/completed"}:
            await on_item(obj(params.get("item")), method, answers, emit)


async def on_item(item: dict[str, object], method: object, answers: dict[str, str],
                  emit: Callable[[AgentActivity], Awaitable[None]]) -> None:
    kind = item.get("type")
    if kind == "agentMessage" and method == "item/completed":
        text = item.get("text")
        if not isinstance(text, str):
            raise ValueError("invalid Codex answer")
        phase = item.get("phase")
        if phase in ("final_answer", "partial_answer", None):
            answers["unphased" if phase is None else str(phase)] = text
        else:
            # 1. Commentary and phases added by newer Codex versions are progress, not answers.
            await emit(AgentActivity("message", text))
    elif kind in {"commandExecution", "fileChange", "mcpToolCall", "webSearch"}:
        await emit(AgentActivity("tool", f"{kind} · {'开始' if method == 'item/started' else '结束'}"))


async def apply(ctx: Context) -> None:
    config = Config.model_validate(ctx.config)
    service = ctx.require(AGENT_RUNNERS)

    async def run(task: AgentTask, emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
        async with asyncio.timeout(config.timeout_seconds):
            async with service.connect(ctx, task.id, argv(config), task.cwd, config.env) as wire:
                return await execute(wire, task, config, emit)

    await service.register(ctx, AgentRunner("codex", desc, True, "read-only", run))
