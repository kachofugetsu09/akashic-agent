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


async def execute(wire: AgentWire, task: AgentTask, config: Config,
                  emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
    serial = 0
    queued: list[dict[str, object]] = []

    async def request(method: str, params: dict[str, object]):
        nonlocal serial
        serial += 1
        identity = serial
        await wire.send({"id": identity, "method": method, "params": params})
        while True:
            frame = await wire.receive()
            if frame.get("id") == identity and "method" not in frame:
                if "error" in frame:
                    raise RuntimeError("Codex request rejected: " + method)
                return obj(frame.get("result"))
            if "id" in frame and "method" in frame:
                await deny(frame)
            else:
                if len(queued) >= 1000:
                    raise ValueError("Codex startup event overflow")
                queued.append(frame)

    async def deny(frame: dict[str, object]):
        # Never approve writes, network escalation or interactive questions.
        method = frame.get("method")
        if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
            await wire.send({"id": frame["id"], "result": {"decision": "decline"}})
        elif method == "item/tool/requestUserInput":
            await wire.send({"id": frame["id"], "result": {"answers": {}}})
        else:
            await wire.send({"id": frame["id"], "error": {"code": -32601, "message": "Unattended request unsupported"}})
            raise ValueError("unsupported Codex server request")

    await request("initialize", {"clientInfo": {"name": "akashic", "version": "1.0.0"},
                                 "capabilities": {"experimentalApi": True}})
    await wire.send({"method": "initialized", "params": {}})
    params: dict[str, object] = {"cwd": task.cwd, "ephemeral": True,
                                  "approvalPolicy": "never", "sandbox": "read-only"}
    if config.model:
        params["model"] = config.model
    thread = obj((await request("thread/start", params)).get("thread"))
    thread_id = thread.get("id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("Codex thread identity missing")
    turn = obj((await request("turn/start", {"threadId": thread_id,
        "input": [{"type": "text", "text": task.prompt}]})).get("turn"))
    turn_id = turn.get("id")
    if not isinstance(turn_id, str) or not turn_id:
        raise ValueError("Codex turn identity missing")
    answers: dict[str, str] = {}
    await emit(AgentActivity("status", "Codex 正在执行（原生只读沙箱）"))
    while True:
        frame = queued.pop(0) if queued else await wire.receive()
        if "id" in frame and "method" in frame:
            await deny(frame)
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
        if params.get("turnId") != turn_id:
            continue
        if method in {"item/started", "item/completed"}:
            item = obj(params.get("item"))
            kind = item.get("type")
            if kind == "agentMessage" and method == "item/completed":
                text = item.get("text")
                if not isinstance(text, str):
                    raise ValueError("invalid Codex answer")
                phase = item.get("phase")
                if phase in ("final_answer", "partial_answer", None):
                    answers["unphased" if phase is None else str(phase)] = text
                elif phase == "commentary":
                    await emit(AgentActivity("message", text))
                else:
                    raise ValueError("unknown Codex message phase")
            elif kind in {"commandExecution", "fileChange", "mcpToolCall", "webSearch"}:
                await emit(AgentActivity("tool", f"{kind} · {'开始' if method == 'item/started' else '结束'}"))


async def apply(ctx: Context) -> None:
    config = Config.model_validate(ctx.config)
    service = ctx.require(AGENT_RUNNERS)

    async def run(task: AgentTask, emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
        async with asyncio.timeout(config.timeout_seconds):
            async with service.connect(ctx, task.id, (config.executable, "app-server", "--listen", "stdio://"),
                                       task.cwd, config.env) as wire:
                return await execute(wire, task, config, emit)

    await service.register(ctx, AgentRunner("codex", desc, True, "read-only", run))
