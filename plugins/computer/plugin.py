from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import cast

from agent.plugin_composition import (
    MCP_SERVERS,
    RUNTIME_STARTED,
    ServiceKey,
    WORKLOADS,
    Context,
    McpServerDefinition,
    WorkloadEnv,
    Workload,
    WorkloadData,
    WorkloadHealth,
    WorkloadLimits,
    WorkloadPort,
)
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE
from plugins.tools.api import BoundTool, CallSource, Result
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION, TurnProjection
from session.log import MessageCatalog, OwnerRecord, OwnerStore
from session.message import ContentPart, Input, Message, Output, ToolCall

from .control import endpoint_name, request

COMPUTER_CONTROL = ServiceKey["ComputerControl"]("computer.control.v1")


@dataclass(frozen=True, slots=True)
class ComputerCall:
    """保存给 source driver 的稳定 Turn 身份。"""

    session_id: str
    source: str
    turn_input_id: str


class ComputerControl:
    """拥有 Computer effect，并通过精确 MCP binding 路由驱动。"""

    def __init__(self, ctx: Context):
        self.ctx = ctx

    def _state(self):
        return self.ctx.require(OWNER_STATE).open(self.ctx)

    async def run(self, key: str, control_binding: str, arguments: Mapping[str, object]) -> dict[str, object]:
        """先保存 started，再打开归档 MCP 执行驱动。"""
        identity = _call_identity(arguments)
        owner_key = "computer-use:" + key
        value = {
            "v": 1,
            "phase": "started",
            "control_binding": control_binding,
            "session_id": identity.session_id,
            "source": identity.source,
            "turn_input_id": identity.turn_input_id,
        }
        state = self._state()
        existing = state.read(owner_key)
        if existing is None:
            _ = state.transact(lambda tx: tx.save(owner_key, value, expected_version=None))
        elif existing.value.get("phase") == "ended" and _same_identity(existing.value, value):
            raise RuntimeError("Computer 调用已经结束")
        elif existing.value != value:
            raise ValueError("Computer 调用回执身份不一致")
        async with self.ctx.require(MCP_SERVERS).open(self.ctx, "computer") as server:
            return await request(
                endpoint_name(self.ctx.data_root, server.generation_id),
                {
                    "op": "run",
                    "context": _driver_context(server.generation_id, identity, key),
                    "code": arguments["code"],
                    "timeoutMs": arguments.get("timeout_ms", 60000),
                    "reset": arguments.get("reset", False),
                },
            )

    async def end_turn(self, identity: ComputerCall, call_id: str) -> None:
        """通过同一归档 MCP 关闭已结算 Turn。"""
        async with self.ctx.require(MCP_SERVERS).open(self.ctx, "computer") as server:
            _ = await request(
                endpoint_name(self.ctx.data_root, server.generation_id),
                {
                    "op": "end_turn",
                    "context": _driver_context(server.generation_id, identity, call_id),
                },
            )


def _call_identity(arguments: Mapping[str, object]) -> ComputerCall:
    value = arguments.get("_computer")
    if not isinstance(value, Mapping) or set(value) != {"session_id", "source", "turn_input_id"}:
        raise ValueError("Computer 工具缺少已验证的 Turn 身份")
    if any(not isinstance(value[key], str) or not value[key] for key in value):
        raise ValueError("Computer Turn 身份无效")
    return ComputerCall(value["session_id"], value["source"], value["turn_input_id"])


def _same_identity(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    fields = ("v", "control_binding", "session_id", "source", "turn_input_id")
    return all(left.get(field) == right.get(field) for field in fields)


def _driver_turn_id(identity: ComputerCall) -> str:
    value = json.dumps(
        [identity.session_id, identity.source, identity.turn_input_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "turn:" + hashlib.sha256(value.encode()).hexdigest()


def _driver_context(generation_id: str, identity: ComputerCall, call_id: str) -> dict[str, str]:
    return {
        "generation_id": generation_id,
        "session_id": identity.session_id,
        "source": identity.source,
        "turn_id": _driver_turn_id(identity),
        "turn_input_id": identity.turn_input_id,
        "call_id": call_id,
    }

api_version = 3
name = "computer"
version = "2.0.0"
desc = "Persistent Linux desktop, browser, and visual control"
author = "Akashic Core"
inject = (
    MCP_SERVERS,
    WORKLOADS,
    TOOLS,
    BINDINGS,
    MESSAGE_CATALOG,
    OWNER_STATE,
    TURN_PROJECTION,
)
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
    "sha256:9bd4f6e215b4848e91f0dbfea75a7b227faeba96268c422d62e81a9b64d5ac92"
)


class _ComputerTool:
    """把已验证的 Message Turn 身份交给 Computer effect owner。"""

    def __init__(self, control: ComputerControl, control_binding: str, projection: TurnProjection):
        self._control = control
        self._control_binding = control_binding
        self._projection = projection

    @property
    def idempotent(self) -> bool:
        return False

    async def prepare(
        self, arguments: Mapping[str, object], source: CallSource | None = None
    ) -> Mapping[str, object]:
        if source is None:
            raise ValueError("Computer 工具只能由已提交 Message 调用")
        target = _call_message(source)
        turns = self._projection.project(source.messages, target.source)
        matched = [
            turn for turn in turns if turn.status == "open" and target.message_id in turn.message_ids
        ]
        if len(matched) != 1:
            raise ValueError("Computer ToolCall 不属于唯一 open Turn")
        by_id = {message.message_id: message for message in source.messages}
        inputs = [
            message
            for message in (by_id[item] for item in matched[0].message_ids)
            if isinstance(message.body, Input) and message.seq <= target.seq
        ]
        if not inputs:
            raise ValueError("Computer Turn 缺少前置 Input")
        values = dict(arguments)
        values["_computer"] = {
            "session_id": target.session_id,
            "source": target.source,
            "turn_input_id": inputs[0].message_id,
        }
        return values

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        value = await self._control.run(key, self._control_binding, arguments)
        return Result("success", (ContentPart("text", json.dumps(value, ensure_ascii=False)),))

    async def query(self, key: str) -> Result | None:
        _ = key
        return None


def _call_message(source: CallSource) -> Message:
    """读取 source 前缀末端唯一的 ToolCall。"""
    target = next(
        (message for message in source.messages if message.message_id == source.call_ref.message_id),
        None,
    )
    if target is None or not isinstance(target.body, Output):
        raise ValueError("Computer CallRef 缺少 Output")
    if source.call_ref.part_index >= len(target.body.parts):
        raise ValueError("Computer CallRef 超出 Output")
    if not isinstance(target.body.parts[source.call_ref.part_index], ToolCall):
        raise ValueError("Computer CallRef 不指向 ToolCall")
    if target.seq != source.messages[-1].seq:
        raise ValueError("Computer CallRef 不是 source 前缀末端")
    return target


def _capture(ctx: Context, configuration: Mapping[str, object]) -> Mapping[str, object]:
    """固定 Computer 专属 control binding，随外层 Tool binding 归档。"""
    if configuration:
        raise ValueError("Computer binding 不接受配置")
    identity = ctx.require(BINDINGS).bind(
        COMPUTER_CONTROL,
        {"version": 1},
        contributors=(ctx,),
    )
    return {"control_binding": identity}


@asynccontextmanager
async def _open_target(ctx: Context, state: Mapping[str, object]) -> AsyncIterator[BoundTool]:
    if set(state) != {"control_binding"} or not isinstance(state["control_binding"], str):
        raise ValueError("Computer binding state 无效")
    bindings = ctx.require(BINDINGS)
    async with bindings.open(state["control_binding"], COMPUTER_CONTROL) as (control, _):
        yield _ComputerTool(control, state["control_binding"], ctx.require(TURN_PROJECTION))


async def apply(ctx: Context, config: object) -> None:
    """注册唯一 Computer Tool、专属 control binding 与资源声明。"""
    _ = config
    control = ComputerControl(ctx)
    _ = await ctx.provide(COMPUTER_CONTROL, control)
    await ctx.require(TOOLS).register(
        ctx,
        name="computer",
        description=(
            "Read or operate browser and desktop UI in the persistent container. "
            "Prefer web_search/web_fetch for information available without UI interaction; "
            "use this for interactive or login-dependent pages, visual inspection, or when "
            "search/fetch fails or cannot provide the needed content. "
            "Prefer suitable dedicated skills, connectors, APIs or CLIs when available. "
            "Run JavaScript with the initialized browser and desktop APIs. "
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
        open=lambda state: _open_target(ctx, state),
        capture=lambda options: _capture(ctx, options),
        public=True,
        idempotent=False,
        risk="external-side-effect",
    )
    await ctx.require(MCP_SERVERS).register(
        ctx,
        McpServerDefinition(
            name="computer",
            command=("mcp_server.py",),
            workload_env=(WorkloadEnv("COMPUTER_URL", "computer", "gateway"),),
        ),
    )
    await _register_workload(ctx)
    async def start_follower(_event: object) -> None:
        _ = await ctx.spawn(_start_follower(ctx), name="computer-turn-follower")

    _ = await ctx.on(RUNTIME_STARTED, start_follower)


async def _register_workload(ctx: Context) -> None:
    """注册持久 source-driver workload。"""
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


async def _start_follower(ctx: Context) -> None:
    """扫描 started effects，等待 Message Turn 关闭后重试收尾。"""
    catalog = ctx.require(MESSAGE_CATALOG)
    projection = ctx.require(TURN_PROJECTION)
    state: OwnerStore | None = None

    async def once() -> None:
        nonlocal state
        if state is None:
            state = ctx.require(OWNER_STATE).open(ctx)
        assert state is not None
        groups: dict[tuple[str, str, str, str], list[tuple[str, OwnerRecord]]] = {}
        for key, record in state.list():
            value = record.value
            if not key.startswith("computer-use:") or value.get("phase") != "started":
                continue
            group = (
                cast(str, value.get("control_binding")),
                cast(str, value.get("session_id")),
                cast(str, value.get("source")),
                cast(str, value.get("turn_input_id")),
            )
            groups.setdefault(group, []).append((key, record))
        for records in groups.values():
            # `follow()` wakes without a runtime lease.  Re-open the exact
            # generation for each effect group so the archived binding and
            # MCP endpoint cannot fall back to a different Root.
            async with ctx.runtime_scope():
                await _try_end(ctx, catalog, projection, records)

    async with ctx.runtime_scope():
        await once()
    async for _heads in catalog.follow():
        async with ctx.runtime_scope():
            await once()


async def _try_end(
    ctx: Context,
    catalog: MessageCatalog,
    projection: TurnProjection,
    records: list[tuple[str, OwnerRecord]],
) -> None:
    """只在源 Turn 已闭合后结束同一 Computer group。"""
    key, record = records[0]
    value = record.value
    expected = {"v", "phase", "control_binding", "session_id", "source", "turn_input_id"}
    if (
        set(value) != expected
        or value["v"] != 1
        or value["phase"] != "started"
        or any(
            not isinstance(value[field], str) or not value[field]
            for field in ("control_binding", "session_id", "source", "turn_input_id")
        )
    ):
        raise ValueError("Computer owner record 字段无效")
    reader = catalog.reader(cast(str, value["session_id"]))
    turns = projection.project(reader.snapshot(), cast(str, value["source"]))
    if not any(
        turn.status in {"complete", "quiet", "abandoned"}
        and cast(str, value["turn_input_id"]) in turn.message_ids
        for turn in turns
    ):
        return
    identity = ComputerCall(
        cast(str, value["session_id"]), cast(str, value["source"]), cast(str, value["turn_input_id"])
    )
    group = json.dumps(
        [value["control_binding"], identity.session_id, identity.source, identity.turn_input_id],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    end_id = "end:" + hashlib.sha256(group.encode()).hexdigest()
    binding = cast(str, value["control_binding"])
    try:
        bindings = ctx.require(BINDINGS)
        async with bindings.open(binding, COMPUTER_CONTROL) as (bound, _):
            await bound.end_turn(identity, end_id)
    except Exception as error:
        ctx.report_incident("computer-end-turn", f"{key}: {error}")
        return
    state = ctx.require(OWNER_STATE).open(ctx)
    def commit(tx) -> None:
        for item_key, item_record in records:
            current = tx.read(item_key)
            if (
                current is None
                or current.version != item_record.version
                or current.value != item_record.value
            ):
                continue
            _ = tx.save(
                item_key,
                {**item_record.value, "phase": "ended"},
                expected_version=item_record.version,
            )

    state.transact(commit)
