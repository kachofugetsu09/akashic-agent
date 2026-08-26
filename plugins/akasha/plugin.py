"""Akasha memory kernel mounted through ordinary plugin lifecycle seams."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, cast

from agent.control.context import running_turn_id
from agent.lifecycle.composition import (
    AFTER_REASONING_PREPROCESS_EVENT,
    PROMPT_RENDER_EVENT,
    observe_composition_domain_event,
)
from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx
from agent.plugin_composition import (
    EMBEDDING_MEMORY_PLUGIN,
    INTERACTION_UNDO,
    TEXT_EMBEDDING_SETTINGS,
    CONVERSATION_SEMANTIC_INTEREST,
    RUNTIME_STOPPING,
    TOOL_CATALOG,
    UI_SLOTS,
    Context,
    MobileUiDefinition,
    MobileUiNavigation,
    MobileUiRpcInvalidRequest,
    PluginToolDefinition,
    ConversationSemanticInterest,
    SourceMutationFence,
)
from agent.prompting import PromptSectionRender
from agent.retrieval.events import build_retrieval_completed
from agent.retrieval.protocol import RetrievalRequest
from agent.tools.base import Tool, ToolExecutionContext
from agent.tools.recall_memory import RecallMemoryTool
from core.memory.plugin import ActiveRecallRecord
from core.memory.engine import (
    MemoryQuery,
    MemoryQueryFilters,
    MemoryQueryResult,
    MemoryScope,
    MemoryToolSpec,
)
from core.net.http import SharedHttpResources
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
from bus.events_lifecycle import TurnCommitted

from .config import load_akasha_config
from .engine import AkashaMemoryEngine
from .inspector import AkashaInspectorReader, mobile_summary

api_version = 3
name = "akasha"
version = "3.0.0"
desc = "提供 Akasha 反馈持久化、Inspector 与移动召回视图"
inject = (TOOL_CATALOG, UI_SLOTS, TEXT_EMBEDDING_SETTINGS, INTERACTION_UNDO)
workspace_roots = ("memory",)
workspace_files = ("sessions.db",)
dashboard_module = "dashboard.py"

_MOBILE_RECALL_SCHEMA = "akasha.recall-card.v1"
_MOBILE_RECALL_USER_PREVIEW_CHARS = 100
_MOBILE_RECALL_ASSISTANT_PREVIEW_CHARS = 50


class _MemoryQueryRuntime(Protocol):
    async def query(self, request: MemoryQuery) -> MemoryQueryResult: ...


class _AkashaMobileQuery:
    """Serve bounded mobile projections from one exact Root runtime."""

    def __init__(
        self,
        runtime: AkashaMemoryEngine,
        *,
        memory_root: Path,
        data_root: Path,
    ) -> None:
        self._runtime = runtime
        self._reader = AkashaInspectorReader(
            memory_root=memory_root,
            config=load_akasha_config(data_root / "config.local.toml"),
        )

    def __call__(
        self,
        method: str,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Serve one validated mobile UI query."""

        # 1. Resolve the memory that affected one assistant response.
        if method == "recall.current":
            return self._mobile_recall(
                payload,
                session_id=session_id,
                turn_id=turn_id,
            )

        # 2. Serve the Inspector list and on-demand detail.
        if method == "inspector.recent":
            if payload:
                raise MobileUiRpcInvalidRequest("Akasha inspector.recent 不接受参数")
            items, total = self._inspector().list_turns(page=1, page_size=30)
            return {
                "items": [
                    {
                        **item,
                        "query_preview": _clip(str(item["query_text"]), 180),
                    }
                    for item in items
                ],
                "total": total,
            }
        if method == "inspector.detail":
            if set(payload) != {"query_id"}:
                raise MobileUiRpcInvalidRequest("Akasha inspector.detail 需要 query_id")
            query_id = payload["query_id"]
            if not isinstance(query_id, str) or not query_id.strip():
                raise MobileUiRpcInvalidRequest(
                    "Akasha inspector.detail 的 query_id 必须是非空字符串"
                )
            item = self._inspector().get_turn(query_id.strip())
            if item is None:
                raise MobileUiRpcInvalidRequest("Akasha 检索记录不存在")
            return mobile_summary(item)
        raise MobileUiRpcInvalidRequest(f"Akasha mobile UI 方法无效: {method}")

    def _mobile_recall(
        self,
        payload: dict[str, object],
        *,
        session_id: str | None,
        turn_id: str | None,
    ) -> dict[str, object]:
        """Resolve current or persisted assistant identity to prompt lanes."""

        # 1. Validate request identity before touching active or persisted state.
        if set(payload) - {"message_id"}:
            raise MobileUiRpcInvalidRequest("Akasha recall.current 参数无效")
        if session_id is None:
            raise MobileUiRpcInvalidRequest("Akasha recall.current 需要 session_id")
        message_id = payload.get("message_id")
        if message_id is not None and not isinstance(message_id, str):
            raise MobileUiRpcInvalidRequest(
                "Akasha recall.current 的 message_id 必须是字符串"
            )

        # 2. Synthetic active messages use only the narrow exact-Root port.
        if isinstance(message_id, str) and message_id.startswith("assistant:"):
            if turn_id is None or message_id != f"assistant:{turn_id}":
                return _empty_mobile_recall()
            pending = self._runtime.wait_active_recall(session_id, turn_id)
            if pending is None:
                return _empty_mobile_recall(pending=True)
            return {
                "schema": _MOBILE_RECALL_SCHEMA,
                "query_id": pending.query_id,
                "recall_capture_available": True,
                "left": _mobile_recall_records(pending.dense),
                "right": _mobile_recall_records(pending.completion),
                "tool_left": [],
                "tool_right": [],
            }

        # 3. Persisted messages read only Akasha's deterministic sidecars.
        item = (
            self._inspector().for_assistant_message(session_id, message_id)
            if isinstance(message_id, str)
            else self._inspector().latest_for_session(session_id)
        )
        if item is None:
            return _empty_mobile_recall()
        return {
            "schema": _MOBILE_RECALL_SCHEMA,
            "query_id": item["query_id"],
            "recall_capture_available": item["recall_capture_available"],
            "left": _mobile_recall_lane(cast(list[dict[str, object]], item["left"])),
            "right": _mobile_recall_lane(cast(list[dict[str, object]], item["right"])),
            "tool_left": _mobile_recall_lane(
                cast(list[dict[str, object]], item["tool_left"])
            ),
            "tool_right": _mobile_recall_lane(
                cast(list[dict[str, object]], item["tool_right"])
            ),
        }

    def _inspector(self) -> AkashaInspectorReader:
        """Reuse one sidecar snapshot reader for this exact Root binding."""

        return self._reader


async def apply(ctx: Context, config: object) -> None:
    """Own the Akasha kernel and expose it through ordinary lifecycle effects."""

    # 1. Claim first, so duplicate memory plugins fail before opening any storage.
    _ = config
    _ = await ctx.provide(EMBEDDING_MEMORY_PLUGIN, object())
    runtime, http = _build_runtime(ctx)

    async def bind_undo_fence():
        return ctx.require(INTERACTION_UNDO).bind_source_fence(
            cast(SourceMutationFence, runtime.delete_interaction_source)
        )

    _ = await ctx.effect(bind_undo_fence, label="akasha-interaction-undo")

    async def cleanup_runtime() -> None:
        await _close_owned([http, *runtime.closeables])

    _ = await ctx.effect(lambda: cleanup_runtime, label="akasha-kernel")
    _ = await ctx.provide(
        CONVERSATION_SEMANTIC_INTEREST,
        ConversationSemanticInterest(
            ctx.workspace_file("sessions.db"),
            runtime.embedding_api,
        ),
    )

    # 2. Prompt retrieval and post-commit projection are normal lifecycle listeners.
    queue: asyncio.Queue[TurnCommitted] = asyncio.Queue()

    async def project_commits() -> None:
        while True:
            event = await queue.get()
            try:
                await runtime.project_committed_turn(event)
            finally:
                queue.task_done()

    _ = await ctx.spawn(project_commits(), name="akasha-post-commit")
    _ = await ctx.on(AFTER_TURN_COMMITTED, queue.put_nowait)
    _ = await ctx.on(RUNTIME_STOPPING, lambda _event: queue.join())
    _ = await ctx.on(
        PROMPT_RENDER_EVENT,
        lambda event: _inject_memory(event, runtime),
    )
    _ = await ctx.on(
        AFTER_REASONING_PREPROCESS_EVENT,
        lambda event: _persist_feedback(event, runtime),
    )
    await _register_tools(ctx, runtime)

    # 3. Inspector UI closes over the same Root-owned kernel.
    query = _AkashaMobileQuery(
        runtime,
        memory_root=ctx.workspace_root("memory"),
        data_root=ctx.data_root,
    )
    await ctx.require(UI_SLOTS).register_mobile(
        ctx,
        MobileUiDefinition(
            module="mobile_ui.js",
            stylesheet="mobile_ui.css",
            navigation=MobileUiNavigation(
                label="Akasha Inspector",
                description="查看每轮线索、激活与模式补全",
            ),
            slots=("turn.before_reasoning",),
        ),
        query=query,
    )


def _build_runtime(ctx: Context) -> tuple[AkashaMemoryEngine, SharedHttpResources]:
    """Build one exact-Root Akasha kernel from declared workspace paths."""

    workspace = ctx.workspace_file("sessions.db").parent
    http = SharedHttpResources()
    runtime = AkashaMemoryEngine(
        embedding=ctx.require(TEXT_EMBEDDING_SETTINGS),
        akasha_config=load_akasha_config(ctx.data_root / "config.local.toml"),
        workspace=workspace,
        http_resources=http,
        event_publisher=None,
    )
    return runtime, http


async def _inject_memory(
    event: PromptRenderCtx,
    runtime: _MemoryQueryRuntime,
) -> None:
    """Retrieve and append one ordinary dynamic prompt section."""

    if "memory" in event.disabled_sections:
        return
    request = RetrievalRequest(
        message=event.content,
        session_key=event.session_key,
        channel=event.channel,
        chat_id=event.chat_id,
        history=event.history,
        session_metadata={},
        turn_id=running_turn_id.get(),
        timestamp=event.timestamp,
    )
    result = await runtime.query(
        MemoryQuery(
            text=request.message,
            intent="context",
            scope=MemoryScope(
                session_key=request.session_key,
                channel=request.channel,
                chat_id=request.chat_id,
            ),
            context={"history": request.history, "turn_id": request.turn_id},
            filters=MemoryQueryFilters(),
            timestamp=request.timestamp,
        )
    )
    await observe_composition_domain_event(build_retrieval_completed(request, result))
    block = result.text_block.strip()
    if block:
        event.system_sections_bottom.append(
            PromptSectionRender(
                name="memory",
                content=block,
                is_static=False,
            )
        )


async def _register_tools(ctx: Context, runtime: AkashaMemoryEngine) -> None:
    """Project the kernel's tool profile into the ordinary plugin Tool catalog."""

    profile = runtime.tool_profile()
    specs = tuple(spec for spec in (profile.recall, *profile.tools) if spec is not None)
    tools = ctx.require(TOOL_CATALOG)
    for spec in specs:
        tool = _build_tool(runtime, spec)
        await tools.register(
            ctx,
            PluginToolDefinition(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
                handler_export=f"akasha:{tool.name}",
                risk="read-only" if spec.risk == "read-only" else "read-write",
                always_on=True,
                search_hint=spec.search_hint or None,
            ),
            _tool_handler(tool),
        )


def _build_tool(runtime: AkashaMemoryEngine, spec: MemoryToolSpec) -> Tool:
    cls = spec.tool_class or RecallMemoryTool
    return cast(Tool, cls(runtime, spec))


def _tool_handler(tool: Tool):
    async def handler(
        context: ToolExecutionContext,
        arguments: Mapping[str, object],
    ) -> object:
        _ = context
        return await tool.execute(**dict(arguments))

    return handler


async def _close_owned(closeables: list[object]) -> None:
    for closeable in reversed(closeables):
        closer = getattr(closeable, "aclose", None) or getattr(closeable, "close", None)
        if closer is None:
            continue
        result = closer()
        if inspect.isawaitable(result):
            await result


def _persist_feedback(
    event: AfterReasoningCtx,
    runtime: AkashaMemoryEngine,
) -> None:
    """Move selected-engine feedback into the current pending user row."""

    metadata = runtime.take_turn_user_metadata(running_turn_id.get())
    duplicated = set(event.persist_user_metadata) & set(metadata)
    if duplicated:
        fields = ", ".join(sorted(duplicated))
        raise RuntimeError(f"Akasha user metadata 字段重复: {fields}")
    event.persist_user_metadata.update(metadata)


def _clip(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[:limit] + "..."


def _empty_mobile_recall(*, pending: bool = False) -> dict[str, object]:
    return {
        "schema": _MOBILE_RECALL_SCHEMA,
        "query_id": None,
        "pending": pending,
        "recall_capture_available": False,
        "left": [],
        "right": [],
        "tool_left": [],
        "tool_right": [],
    }


def _mobile_recall_lane(
    value: list[dict[str, object]],
) -> list[dict[str, object]]:
    """把语义层已选出的整条 lane 投影成移动卡片字段。"""

    projected: list[dict[str, object]] = []
    for raw in value:
        item: dict[str, object] = {
            "user_preview": _clip(cast(str, raw["user_text"]), 100),
            "assistant_preview": _clip(cast(str, raw["assistant_preview"]), 50),
            "ts": cast(str, raw["ts"]),
        }
        score = raw.get("score")
        if score is not None:
            item["score"] = score
        projected.append(item)
    return projected


def _mobile_recall_records(
    records: tuple[ActiveRecallRecord, ...],
) -> list[dict[str, object]]:
    """Project frozen runtime records through the bounded card shape."""

    return _mobile_recall_lane(
        [
            {
                "user_text": record.user_text,
                "assistant_preview": record.assistant_preview,
                "ts": record.started_at,
                "score": record.score,
            }
            for record in records
        ]
    )
