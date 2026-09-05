from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Mapping

from agent.plugin_composition import (
    MCP_SERVERS,
    RUNTIME_STOPPING,
    TOOL_CATALOG,
    WORKLOADS,
    Context,
    McpServerDefinition,
    PluginToolDefinition,
    Workload,
    WorkloadData,
    WorkloadEnv,
    WorkloadHealth,
    WorkloadLimits,
    WorkloadPort,
)
from agent.tools.base import ToolExecutionContext
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED

from .control import endpoint_name
from .control import request as control_request

api_version = 3
name = "computer"
version = "2.0.0"
desc = "Persistent Linux desktop, browser, and visual control"
author = "Akashic Core"
inject = (WORKLOADS, MCP_SERVERS, TOOL_CATALOG)
skill_roots = ("skills",)
drift_skill_roots = ()
workspace_roots = ()
workspace_files = ()
dashboard_module = "dashboard.py"
web_module = "web_module.js"
web_requires = ("conversation.tools.v1",)
web_provides = ()
web_contract_digests = {
    "conversation.tools.v1": "ed47d69b84e946e27a2e297634e96bcc6afc72a3d3089caac1a14632703efb54",
}

_IMAGE = (
    "ghcr.io/kachofugetsu09/akashic-computer@"
    "sha256:6fd3c605380a3daef5ddebb34f2905ee992d2b4e1490fbfb78dcce9f06a3dadb"
)


async def apply(ctx: Context, config: object) -> None:
    """Register the Computer workload and its narrow MCP adapter."""

    _ = config
    control_name = endpoint_name(ctx.data_root)
    used_turns: set[tuple[str, str]] = set()

    async def run(
        context: ToolExecutionContext, arguments: Mapping[str, object]
    ) -> str:
        """只从 Core 的工具上下文取身份，不允许用户代码指定其他 Session。"""
        if (
            not context.origin_session_key
            or not context.turn_id
            or not context.execution_id
        ):
            raise RuntimeError("Computer requires a live Tool execution context")
        identity = {
            "generation_id": ctx.generation_id,
            "session_id": context.origin_session_key,
            "turn_id": context.turn_id,
            "call_id": context.execution_id,
        }
        used_turns.add((context.origin_session_key, context.turn_id))
        result = await control_request(
            control_name,
            {
                "op": "run",
                "context": identity,
                "code": arguments["code"],
                "timeoutMs": arguments.get("timeout_ms", 60000),
                "reset": arguments.get("reset", False),
            },
        )
        return json.dumps(result, ensure_ascii=False)

    async def end_turn(event) -> None:
        """只收尾实际使用过 Computer 的 Turn，保留标记交付的标签。"""
        key = (event.session_key, event.turn_id)
        if key not in used_turns:
            return
        await control_request(
            control_name,
            {
                "op": "end_turn",
                "context": {
                    "generation_id": ctx.generation_id,
                    "session_id": event.session_key,
                    "turn_id": event.turn_id,
                    "call_id": "end-" + uuid.uuid4().hex,
                },
            },
        )
        used_turns.remove(key)

    await ctx.require(TOOL_CATALOG).register(
        ctx,
        PluginToolDefinition(
            name="computer",
            description=(
                "Run JavaScript against the persistent container browser and Linux desktop. "
                "Use browser.tabs, tab.ax, tab.playwright, tab.dom_cua or sky. "
                "Bindings persist within this Session. Call nodeRepl.write(value) or "
                "nodeRepl.emitImage(bytes) for output. Read the computer skill first."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "maxLength": 131072},
                    "timeout_ms": {"type": "integer", "minimum": 1, "maximum": 110000},
                    "reset": {
                        "type": "boolean",
                        "description": "Reset this Session's JS bindings before running code.",
                    },
                },
                "required": ["code"],
                "additionalProperties": False,
            },
            handler_export="computer",
            risk="external-side-effect",
        ),
        run,
    )
    queue = asyncio.Queue()
    worker = None

    def enqueue_end(event) -> None:
        if (event.session_key, event.turn_id) not in used_turns:
            return
        if worker is not None and worker.done():
            raise RuntimeError("Computer Turn cleanup worker stopped")
        queue.put_nowait((event, ctx.capture_runtime_scope()))

    async def finish_turns() -> None:
        try:
            while True:
                event, scope = await queue.get()
                try:
                    async with scope:
                        await end_turn(event)
                finally:
                    queue.task_done()
        finally:
            while not queue.empty():
                _, scope = queue.get_nowait()
                await scope.close()
                queue.task_done()

    worker = await ctx.spawn(finish_turns(), name="computer-turn-cleanup")
    await ctx.on(AFTER_TURN_COMMITTED, enqueue_end)
    await ctx.on(RUNTIME_STOPPING, lambda _event: queue.join())
    await ctx.require(WORKLOADS).register(
        ctx,
        Workload(
            name="computer",
            image=_IMAGE,
            command=("/opt/computer/start.sh",),
            ports=(
                WorkloadPort("gateway", 8080),
                WorkloadPort("display", 6080),
                WorkloadPort("opencli", 19826, loopback=19825),
            ),
            data=(WorkloadData("state", "/data"),),
            health=WorkloadHealth("gateway", "/health", 90.0),
            limits=WorkloadLimits(0, 0.0, 0),
            user_namespaces=True,
        ),
    )
    await ctx.require(MCP_SERVERS).register(
        ctx,
        McpServerDefinition(
            name="computer",
            command=("mcp_server.py",),
            required_tools=(
                "browser_observe",
                "browser_action",
                "computer_observe",
                "computer_action",
            ),
            candidate_read_only_tools=("browser_observe", "computer_observe"),
            workload_env=(WorkloadEnv("COMPUTER_URL", "computer", "gateway"),),
        ),
    )
