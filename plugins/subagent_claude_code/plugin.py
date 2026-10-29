"""Claude Code stream-json adapter with an explicit read-tool allowlist."""
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
name = "subagent_claude_code"
version = "1.0.0"
desc = "委托 Claude Code；只开放 Read、Glob、Grep，不开放 shell、写入或 MCP 工具"
inject = (AGENT_RUNNERS, PROCESS_STREAMS, TOOLS, JOB_NOTICES)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    executable: str = "claude"
    model: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = Field(default=1800, gt=0, le=86400)


async def execute(wire: AgentWire, task: AgentTask,
                  emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
    await wire.send({"type": "user", "message": {"role": "user", "content": task.prompt}})
    session_id: str | None = None
    while True:
        frame = await wire.receive()
        identity = frame.get("session_id")
        if identity is not None:
            if not isinstance(identity, str) or (session_id is not None and identity != session_id):
                raise ValueError("Claude Code session changed during one-shot task")
            session_id = identity
        kind = frame.get("type")
        if kind == "control_request":
            request_id = frame.get("request_id")
            if not isinstance(request_id, str):
                raise ValueError("invalid Claude Code control request")
            await wire.send({"type": "control_response", "response": {
                "subtype": "error", "request_id": request_id,
                "error": "Interactive requests are not supported by this read-only runner"}})
            raise ValueError("unexpected Claude Code interactive request")
        if kind == "result":
            text = frame.get("result")
            if frame.get("subtype") != "success" or frame.get("is_error") is True or not isinstance(text, str) or not text.strip():
                return AgentResult("failed", "Claude Code 未返回有效完成答案。")
            return AgentResult("completed", text)
        if kind == "system":
            await emit(AgentActivity("status", "Claude Code：" + str(frame.get("subtype", "system"))))
        if kind in {"assistant", "user"}:
            message = frame.get("message")
            if not isinstance(message, dict):
                continue
            blocks = message.get("content")
            if not isinstance(blocks, list):
                continue
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                block = cast(dict[str, object], block)
                if block.get("type") == "text" and kind == "assistant":
                    await emit(AgentActivity("message", str(block.get("text", ""))))
                elif block.get("type") == "tool_use":
                    await emit(AgentActivity("tool", str(block.get("name", "tool")) + " · 开始"))
                elif block.get("type") == "tool_result":
                    await emit(AgentActivity("tool", str(block.get("tool_use_id", "tool")) + " · 结束"))


async def apply(ctx: Context) -> None:
    config = Config.model_validate(ctx.config)
    service = ctx.require(AGENT_RUNNERS)

    async def run(task: AgentTask, emit: Callable[[AgentActivity], Awaitable[None]]) -> AgentResult:
        argv = (config.executable, "--print", "--verbose", "--input-format", "stream-json",
                "--output-format", "stream-json", "--permission-mode", "dontAsk",
                "--tools", "Read,Glob,Grep", "--allowedTools", "Read,Glob,Grep",
                "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
                "--setting-sources", "", "--settings", '{"disableAllHooks":true}',
                "--no-session-persistence")
        if config.model:
            argv += ("--model", config.model)
        async with asyncio.timeout(config.timeout_seconds):
            async with service.connect(ctx, task.id, argv, task.cwd, config.env) as wire:
                return await execute(wire, task, emit)

    await service.register(ctx, AgentRunner("claude_code", desc, True, "read-only", run))
