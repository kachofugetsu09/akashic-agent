"""Akasha memory kernel mounted through ordinary plugin lifecycle seams."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager
from pathlib import Path
from typing import Protocol, cast

from agent.control.context import running_turn_id
from agent.lifecycle.composition import (
    AFTER_REASONING_PREPROCESS_EVENT,
    PROMPT_RENDER_EVENT,
    observe_composition_domain_event,
)
from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx
from agent.plugin_composition import (
    EMBEDDINGS,
    EMBEDDING_MEMORY_PLUGIN,
    COMMANDS,
    INTERACTION_UNDO,
    CONVERSATION_SEMANTIC_INTEREST,
    RUNTIME_STOPPING,
    RUNTIME_STARTED,
    SNAPSHOT_SEALING,
    TOOL_CATALOG,
    UI_SLOTS,
    Context,
    CommandDefinition,
    CommandInvocation,
    CommandResult,
    DriverUnavailableError,
    Embeddings,
    HealthHandle,
    MobileUiDefinition,
    MobileUiNavigation,
    MobileUiRpcInvalidRequest,
    ModelUnavailableError,
    PluginToolDefinition,
    PluginDiagnosticContext,
    PluginDiagnostics,
    RuntimeScope,
    ConversationSemanticInterest,
    SourceMutationFence,
    ServiceKey,
)
from agent.prompting import PromptSectionRender
from agent.retrieval.events import build_retrieval_completed
from agent.retrieval.protocol import RetrievalRequest
from agent.tools.base import Tool, ToolExecutionContext
from agent.tools.recall_memory import RecallMemoryTool, render_memory_unavailable
from core.memory.plugin import ActiveRecallRecord, ActiveRecallView
from core.memory.engine import (
    MemoryQuery,
    MemoryQueryFilters,
    MemoryQueryResult,
    MemoryScope,
    MemoryToolSpec,
)
from agent.turn_events.after_turn import AFTER_TURN_COMMITTED
from bus.events_lifecycle import TurnCommitted
from session.store import InteractionDeletion
from .config import AkashaConfig, load_akasha_config
from .engine import (
    AkashaMemoryEngine,
    EmbeddingSpaceMismatchError,
    render_feedback_unavailable,
)
from .inspector import AkashaInspectorReader, mobile_summary
from .repair import finish_request, load_request, reindex, save_request

api_version = 3
name = "akasha"
version = "3.0.0"
desc = "提供 Akasha 反馈持久化、Inspector 与移动召回视图"
inject = (COMMANDS, TOOL_CATALOG, UI_SLOTS, EMBEDDINGS, INTERACTION_UNDO)
workspace_roots = ("memory",)
workspace_files = ("sessions.db",)
dashboard_module = "dashboard.py"
web_module = "web_module.js"
web_requires = ("workbench.panels.v2",)
web_provides = ()
web_contract_digests = {
    "workbench.panels.v2": "fb6417c9bf532c1fdb344767d06065d5d3293da85deb64eff1e8088889a33bcb",
}

MEMORY_RECALL = ServiceKey[object]("memory.recall.v1")

_MOBILE_RECALL_SCHEMA = "akasha.recall-card.v1"
_MOBILE_RECALL_USER_PREVIEW_CHARS = 100
_MOBILE_RECALL_ASSISTANT_PREVIEW_CHARS = 50


class _MemoryQueryRuntime(Protocol):
    async def query(self, request: MemoryQuery) -> MemoryQueryResult: ...


class _AkashaToolRuntime(_MemoryQueryRuntime, Protocol):
    def stage_feedback(
        self,
        *,
        turn_id: str,
        action: str,
        message_ids: list[str],
        reason: str,
    ) -> dict[str, object]: ...


class _AkashaRuntimeHandle:
    """等所有 model driver 注册后再构造 kernel。"""

    def __init__(self) -> None:
        self._runtime: AkashaMemoryEngine | None = None
        self._factory: Callable[[], AkashaMemoryEngine] | None = None
        self._embedding_identity: Callable[[], str] | None = None
        self._health: HealthHandle | None = None
        self._unavailable_reason = "Akasha runtime 尚未启动"

    def configure(
        self,
        factory: Callable[[], AkashaMemoryEngine],
        *,
        embedding_identity: Callable[[], str],
    ) -> None:
        if self._factory is not None:
            raise RuntimeError("Akasha runtime 重复配置")
        self._factory = factory
        self._embedding_identity = embedding_identity

    def bind_health(self, health: HealthHandle) -> None:
        self._health = health

    def try_get(self) -> AkashaMemoryEngine | None:
        """Load the kernel or expose one optional, observable unavailable state."""

        try:
            runtime = self.get()
        except (
            DriverUnavailableError,
            EmbeddingSpaceMismatchError,
            ModelUnavailableError,
        ) as error:
            self._unavailable_reason = str(error)
            if self._health is not None:
                self._health.degrade(self._unavailable_reason)
            return None
        self._unavailable_reason = ""
        if self._health is not None and not self._health.healthy:
            self._health.recover()
        return runtime

    def available(self) -> bool:
        return self.try_get() is not None

    @property
    def unavailable_reason(self) -> str:
        return self._unavailable_reason

    def get(self) -> AkashaMemoryEngine:
        if self._runtime is None:
            if self._factory is None:
                raise RuntimeError("Akasha runtime 尚未配置")
            self._runtime = self._factory()
        identity = self._embedding_identity
        if identity is None:
            raise RuntimeError("Akasha embedding identity 尚未配置")
        if identity() != self._runtime.embedding_api.model_id:
            raise EmbeddingSpaceMismatchError(
                "Akasha 默认 embedding 空间已变化，需要重建派生状态"
            )
        return self._runtime

    @property
    def model_id(self) -> str:
        runtime = self.try_get()
        return "" if runtime is None else runtime.embedding_api.model_id

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self.get().embedding_api.embed_batch(texts)

    async def query(self, request: MemoryQuery) -> MemoryQueryResult:
        return await self.get().query(request)

    async def project_committed_turn(self, event: TurnCommitted) -> None:
        runtime = self.try_get()
        if runtime is not None:
            await runtime.project_committed_turn(event)

    async def delete_interaction_source(
        self,
        control_turn_id: str,
        delete: Callable[[], InteractionDeletion | None],
    ) -> InteractionDeletion | None:
        return await self.get().delete_interaction_source(control_turn_id, delete)

    def stage_feedback(
        self,
        *,
        turn_id: str,
        action: str,
        message_ids: list[str],
        reason: str,
    ) -> dict[str, object]:
        return self.get().stage_feedback(
            turn_id=turn_id,
            action=action,
            message_ids=message_ids,
            reason=reason,
        )

    def take_turn_user_metadata(self, turn_id: str) -> dict[str, object]:
        runtime = self.try_get()
        return {} if runtime is None else runtime.take_turn_user_metadata(turn_id)

    def wait_active_recall(
        self,
        session_key: str,
        turn_id: str,
    ) -> ActiveRecallView | None:
        return self.get().wait_active_recall(session_key, turn_id)

    async def close(self) -> None:
        if self._runtime is not None:
            await _close_owned(self._runtime.closeables)

    async def reset(self) -> None:
        """Close one old kernel before an explicit startup repair."""

        runtime = self._runtime
        self._runtime = None
        if runtime is not None:
            await _close_owned(runtime.closeables)

    def degrade(self, reason: str) -> None:
        self._unavailable_reason = reason
        if self._health is not None:
            self._health.degrade(reason)

class _AkashaMobileQuery:
    """Serve bounded mobile projections from one exact Root runtime."""

    def __init__(
        self,
        runtime: _AkashaRuntimeHandle,
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
            return _active_mobile_recall(pending, publishing=False)

        # 3. Persisted messages read only Akasha's deterministic sidecars.
        item = (
            self._inspector().for_assistant_message(session_id, message_id)
            if isinstance(message_id, str)
            else self._inspector().latest_for_session(session_id)
        )
        if item is None:
            if turn_id is None:
                return _empty_mobile_recall()
            pending = self._runtime.wait_active_recall(session_id, turn_id)
            return (
                _empty_mobile_recall(pending=True)
                if pending is None
                else _active_mobile_recall(pending, publishing=True)
            )
        if not cast(bool, item["projection_ready"]):
            if turn_id is None:
                return _empty_mobile_recall(pending=True)
            pending = self._runtime.wait_active_recall(session_id, turn_id)
            return (
                _empty_mobile_recall(pending=True)
                if pending is None
                else _active_mobile_recall(pending, publishing=True)
            )
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
    _ = await ctx.provide(MEMORY_RECALL, object())
    runtime = _AkashaRuntimeHandle()
    runtime.bind_health(await ctx.health("embedding", required=False))
    embeddings = ctx.require(EMBEDDINGS)
    workspace = ctx.workspace_file("sessions.db").parent
    akasha_config = load_akasha_config(ctx.data_root / "config.local.toml")

    async def start_runtime(_event: object) -> None:
        runtime.configure(
            lambda: _build_runtime(
                embeddings=embeddings,
                workspace=workspace,
                akasha_config=akasha_config,
                runtime_scope=ctx.runtime_scope,
            ),
            embedding_identity=lambda: embeddings.describe().identity,
        )
        # 模型未配置、driver 暂离或旧派生索引待重建时，控制面仍可启动。
        _ = runtime.try_get()

    _ = await ctx.on(SNAPSHOT_SEALING, start_runtime)

    async def request_reindex(invocation: CommandInvocation) -> CommandResult:
        if invocation.raw_input.strip() != "confirm":
            return CommandResult(
                kind="error",
                text="此操作会重新生成当前 embedding 空间和 Akasha 派生索引。请使用 /akasha_reindex confirm",
            )
        try:
            request = save_request(ctx.data_root, embeddings.describe())
        except (DriverUnavailableError, ModelUnavailableError, ValueError) as error:
            return CommandResult(kind="error", text=f"无法创建 Akasha reindex 请求：{error}")
        return CommandResult(
            kind="success",
            text=(
                "Akasha reindex 已登记。请重启服务；启动阶段会先备份，再重建 "
                f"{request.embedding_identity}。"
            ),
        )

    await ctx.require(COMMANDS).register(
        ctx,
        CommandDefinition(
            name="akasha_reindex",
            description="显式备份并重建 Akasha embedding 空间",
            handler=request_reindex,
            input_hint="confirm",
        ),
    )

    async def run_requested_reindex() -> None:
        try:
            request = load_request(ctx.data_root)
            if request is None:
                return
            async with ctx.runtime_scope():
                descriptor = embeddings.describe()
            await runtime.reset()
            result = await reindex(
                embeddings=embeddings,
                descriptor=descriptor,
                request=request,
                workspace=workspace,
                data_root=ctx.data_root,
                config=akasha_config,
                runtime_scope=ctx.runtime_scope,
            )
            if runtime.try_get() is None:
                raise RuntimeError(runtime.unavailable_reason)
            finish_request(ctx.data_root)
        except Exception as error:
            reason = f"Akasha reindex 未完成：{error}"
            runtime.degrade(reason)
            ctx.report_incident("akasha.reindex_failed", reason)
            return
        ctx.report_incident(
            "akasha.reindex_completed",
            f"Akasha reindex 完成，embedded={result.embedded_messages}",
        )

    async def start_reindex_worker(_event: object) -> None:
        _ = await ctx.spawn(run_requested_reindex(), name="akasha-reindex")

    _ = await ctx.on(RUNTIME_STARTED, start_reindex_worker)

    async def bind_undo_fence():
        async def delete_source(
            control_turn_id: str,
            delete: Callable[[], object | None],
        ) -> object | None:
            return await runtime.delete_interaction_source(
                control_turn_id,
                cast(Callable[[], InteractionDeletion | None], delete),
            )

        return ctx.require(INTERACTION_UNDO).bind_source_fence(
            cast(SourceMutationFence, delete_source)
        )

    _ = await ctx.effect(bind_undo_fence, label="akasha-interaction-undo")

    async def cleanup_runtime() -> None:
        await runtime.close()

    _ = await ctx.effect(lambda: cleanup_runtime, label="akasha-kernel")
    _ = await ctx.provide(
        CONVERSATION_SEMANTIC_INTEREST,
        ConversationSemanticInterest(
            ctx.workspace_file("sessions.db"),
            runtime,
        ),
    )

    # 2. Prompt retrieval and post-commit projection are normal lifecycle listeners.
    diagnostics = ctx.diagnostics
    queue: asyncio.Queue[
        tuple[TurnCommitted, PluginDiagnosticContext | None, RuntimeScope]
    ] = asyncio.Queue()
    worker_task: asyncio.Task[None] | None = None

    def enqueue_commit(event: TurnCommitted) -> None:
        """Preserve the source listener as the queued projection's parent."""

        if worker_task is not None and worker_task.done():
            raise RuntimeError("Akasha post-commit worker 已停止")
        queue.put_nowait(
            (event, diagnostics.capture(), ctx.capture_runtime_scope())
        )

    async def project_commits() -> None:
        try:
            while True:
                event, parent, scope = await queue.get()
                try:
                    async with scope:
                        with diagnostics.resume(parent):
                            with diagnostics.operation("memory.project_commit"):
                                await runtime.project_committed_turn(event)
                finally:
                    queue.task_done()
        finally:
            while True:
                try:
                    _event, _parent, pending_scope = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await pending_scope.close()
                queue.task_done()

    worker_task = await ctx.spawn(project_commits(), name="akasha-post-commit")
    _ = await ctx.on(AFTER_TURN_COMMITTED, enqueue_commit)
    _ = await ctx.on(RUNTIME_STOPPING, lambda _event: queue.join())
    _ = await ctx.on(
        PROMPT_RENDER_EVENT,
        lambda event: _inject_memory(event, runtime, diagnostics),
    )
    _ = await ctx.on(
        AFTER_REASONING_PREPROCESS_EVENT,
        lambda event: _persist_feedback(
            event,
            runtime,
        ),
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
        available=runtime.available,
    )


def _build_runtime(
    *,
    embeddings: Embeddings,
    workspace: Path,
    akasha_config: AkashaConfig,
    runtime_scope: Callable[[], AbstractAsyncContextManager[None]],
) -> AkashaMemoryEngine:
    """Build one exact-Root Akasha kernel from declared workspace paths."""

    return AkashaMemoryEngine(
        embeddings=embeddings,
        embedding_space=embeddings.describe(),
        runtime_scope=runtime_scope,
        akasha_config=akasha_config,
        workspace=workspace,
        event_publisher=None,
    )


async def _inject_memory(
    event: PromptRenderCtx,
    runtime: _MemoryQueryRuntime,
    diagnostics: PluginDiagnostics,
) -> None:
    """Retrieve and append one ordinary dynamic prompt section."""

    if "memory" in event.disabled_sections:
        return
    if isinstance(runtime, _AkashaRuntimeHandle) and not runtime.available():
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
    with diagnostics.operation("memory.retrieval"):
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
        diagnostics.measure("memory.records", len(result.records))
        for name in (
            "seed_count",
            "dense_count",
            "active_basin_count",
            "completion_count",
            "pushes",
        ):
            value = result.trace.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                diagnostics.measure(f"memory.{name}", value)
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


async def _register_tools(ctx: Context, runtime: _AkashaRuntimeHandle) -> None:
    """Project the kernel's tool profile into the ordinary plugin Tool catalog."""

    profile = AkashaMemoryEngine.tool_profile()
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
            _tool_handler(tool, runtime, recall=spec is profile.recall),
            provided_for=(MEMORY_RECALL if spec is profile.recall else None),
        )


def _build_tool(runtime: _AkashaToolRuntime, spec: MemoryToolSpec) -> Tool:
    cls = spec.tool_class or RecallMemoryTool
    return cast(Tool, cls(runtime, spec))


def _tool_handler(
    tool: Tool,
    runtime: _AkashaRuntimeHandle,
    *,
    recall: bool,
):
    async def handler(
        context: ToolExecutionContext,
        arguments: Mapping[str, object],
    ) -> object:
        _ = context
        if not runtime.available():
            if recall:
                return render_memory_unavailable(runtime.unavailable_reason)
            return render_feedback_unavailable(runtime.unavailable_reason)
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
    runtime: _AkashaRuntimeHandle,
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


def _active_mobile_recall(
    pending: ActiveRecallView,
    *,
    publishing: bool,
) -> dict[str, object]:
    """Project the frozen active lanes through the stable card schema."""

    return {
        "schema": _MOBILE_RECALL_SCHEMA,
        "query_id": pending.query_id,
        "pending": publishing,
        "recall_capture_available": True,
        "left": _mobile_recall_records(pending.dense),
        "right": _mobile_recall_records(pending.completion),
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
