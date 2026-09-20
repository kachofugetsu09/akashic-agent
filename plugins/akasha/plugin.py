"""从消息学习；模型未配置时保持可见的记忆不可用状态。"""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from importlib import import_module
from agent.plugin_composition.ui import UI

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal, Protocol, Self
from functools import partial
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from agent.plugin_composition import EMBEDDINGS, RUNTIME_STARTED, RUNTIME_STOPPING, Context, ServiceKey, UI_SLOTS, MobileUiDefinition, MobileUiNavigation, MobileUiRpcInvalidRequest
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.commands import COMMANDS, CommandDefinition, CommandInvocation, CommandResult
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, OWNER_STATE
from agent.plugin_contracts import Message
from agent.plugin_composition.models import DriverUnavailableError, ModelUnavailableError
from .domain.model import EmbeddingSpaceMismatchError, MemoryRebuildRequiredError
from ._boundaries import CONTENT, TOOLS, TURN_PROJECTION, ContentCapability, ToolCatalog, ToolRef, ToolView

from .application.consumer import MessageConsumer
from .config import AkashaConfig, resolve_memory_path
from .infrastructure.consumption import load_message_nodes
from .inspector import RecallInspector
from .learning import AKASHA_LEARNING, Learning, LearningConfig
from .interest import SEMANTIC_INTEREST, Embed, SemanticInterest
from .recall_tool import RecallArguments, RecallTool, check_recall
from .recalls import Recall, RecallRecords, RecallRecordsRead
from .runtime import MessageMemory, prepare_materials
from .application.snapshot import read_memory
from agent.plugin_composition.models import open_embedding as open_saved_embedding, read_embedding_binding
from .tools import FeedbackArguments, FeedbackTool, check_feedback
from .application.rebuild import manifest_json, rebuild_from_catalog

logger = logging.getLogger(__name__)

# 迁移与插件共同寻址的一次性重放凭据名；两处必须保持一致。
_REPLAY_REQUEST_NAME = ".akasha-replay-request.json"
_REPLAY_STATUS_NAME = ".akasha-replay-status.json"

api_version = 3
name = "akasha"
version = "4.1.0"
desc = "从消息学习并提供普通 Context 材料与记忆工具"
workspace_roots = ("memory",)

MaterialData = Mapping[str, object]


class MaterialRegistry(Protocol):
    async def register(
        self, ctx: Context, *, name: str,
        prepare: Callable[[tuple[Message, ...], str], Awaitable[MaterialData]],
        priority: int = 0, prompt: bool = False, reduce: object | None = None,
    ) -> object: ...


MATERIALS = ServiceKey[MaterialRegistry]("context.materials.v3")
inject = (UI, TURN_PROJECTION, CONTENT, MATERIALS, TOOLS, EMBEDDINGS,
          BINDINGS, MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, OWNER_STATE, UI_SLOTS, COMMANDS)


class Config(BaseModel):
    """同名旧配置由 Manager 一次读取并归档，再转换为现有 Akasha 配置。"""

    model_config = ConfigDict(extra="forbid")
    sources: tuple[str, ...] = Field(
        default=("conversation", "programmatic", "legacy-unattributed"), min_length=1,
    )
    db_path: str = AkashaConfig.db_path
    inject_max_chars: int = AkashaConfig.inject_max_chars
    context_recall_limit: int = AkashaConfig.context_recall_limit
    restart: float = AkashaConfig.restart
    tolerance: float = AkashaConfig.tolerance
    learning_rate: float = AkashaConfig.learning_rate
    activation_power: float = AkashaConfig.activation_power
    recurrent_budget: float = AkashaConfig.recurrent_budget
    reverse_temporal_ratio: float = AkashaConfig.reverse_temporal_ratio
    forgetting_enabled: bool = AkashaConfig.forgetting_enabled

    def settings(self) -> AkashaConfig:
        settings = AkashaConfig(**self.model_dump(exclude={"sources"}))
        settings.validate()
        return settings


class InspectorPage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    page: int = Field(default=1, ge=1, le=1000000)
    page_size: int = Field(default=30, ge=1, le=100)


class RecallBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    embedding_binding: str | None = Field(min_length=1)
    unavailable: str | None = Field(min_length=1)

    @model_validator(mode="after")
    def check(self) -> Self:
        if (self.embedding_binding is None) == (self.unavailable is None):
            raise ValueError("召回 binding 必须包含模型引用或明确不可用原因")
        return self


AKASHA_RECORDS = ServiceKey[Callable[[str], Recall | None]]("akasha.recalls.v1")
AKASHA_RECORDS_VIEW = ServiceKey[Callable[[], RecallRecordsRead]](
    "akasha.recall-records.v1"
)
AKASHA_TOOLS = ServiceKey[ToolView]("akasha.tools.v1")
AKASHA_MEMORY_PATH = ServiceKey[Callable[[], Path]]("akasha.memory-path.v1")


async def apply(ctx: Context) -> None:
    """注册纯学习规则和延迟工具；正式启动事件才取得唯一学习 writer。"""
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        dashboard=lambda: import_module(".dashboard", __package__),
        requires=("workbench.panels.v2",),
        provides=(),
        contract_digests={
            "workbench.panels.v2": "fb6417c9bf532c1fdb344767d06065d5d3293da85deb64eff1e8088889a33bcb",
        },
    )
    config = Config.model_validate(ctx.config)
    catalog: ToolCatalog = ctx.require(TOOLS)
    content: ContentCapability = ctx.require(CONTENT)
    _ = await catalog.declare_group(ctx, description=desc)
    tool_refs: list[ToolRef] = []

    async def request_reindex(invocation: CommandInvocation) -> CommandResult:
        if invocation.raw_input.strip().casefold() != "confirm":
            return CommandResult(
                "error",
                "Akasha 重建会用 canonical 来源整体替换派生学习图；确认后以 /akasha_reindex confirm 执行。",
            )
        return await run_rebuild()

    _ = await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        name="akasha_reindex", description="从 canonical 来源全量重建 Akasha 学习图",
        handler=request_reindex, read_only=False, input_hint="confirm",
    ))
    settings = config.settings()
    memory_path = resolve_memory_path(ctx.workspace_root("memory"), settings.db_path)
    rebuild_backup_root = ctx.data_root / "backups" / "rebuild"
    # 一次性重放凭据必须放在跨 generation 稳定的位置：迁移在 runtime 之前运行，
    # 只能寻址 workspace 内的固定路径，而 ctx.data_root 属于当前 generation。
    rebuild_request_paths = (
        ctx.workspace_root("memory") / _REPLAY_REQUEST_NAME,
        ctx.data_root / _REPLAY_REQUEST_NAME,
    )
    learning = Learning(
        ctx.require(TURN_PROJECTION), owner=ctx.runtime.plugin_id,
        post_commit_effect=content.legacy_post_commit_effect,
    )
    _ = await ctx.provide(AKASHA_LEARNING, learning)
    _ = await content.register(ctx, {
        "name": "akasha",
        "content": {"akasha.feedback": check_feedback, "akasha.recall": check_recall},
    })
    memory: MessageMemory | None = None
    memory_rule: LearningConfig | None = None
    watcher: asyncio.Task[None] | None = None
    running = False
    start_lock = asyncio.Lock()
    health = await ctx.health("embedding", required=False)

    def records() -> RecallRecords:
        return RecallRecords(ctx.require(OWNER_STATE).open(ctx))

    def records_read() -> RecallRecordsRead:
        return RecallRecordsRead(ctx.require(OWNER_STATE).open(ctx))

    # 公开读取函数不暴露 owner transaction；归档 apply 也不会读取正式数据库。
    def read_recall(identity: str) -> Recall | None:
        return records().read(identity)
    _ = await ctx.provide(AKASHA_RECORDS, read_recall)
    _ = await ctx.provide(AKASHA_RECORDS_VIEW, records_read)

    inspector: RecallInspector | None = None

    def get_inspector() -> RecallInspector:
        """返回正式 runtime 绑定的只读查询投影。"""
        if not running or inspector is None:
            raise MobileUiRpcInvalidRequest("Akasha 查询读取尚未启动")
        return inspector

    def query(method: str, payload: dict[str, object], *, session_id: str | None,
              turn_id: str | None) -> dict[str, object]:
        inspector = get_inspector()
        if method == "recall.turn":
            offset = payload.get("offset", 0)
            if (not session_id or set(payload) - {"message_id", "source", "offset"}
                or not {"message_id", "source"} <= set(payload)
                or not isinstance(offset, int) or isinstance(offset, bool) or offset < 0
                or not isinstance(payload["message_id"], str) or not payload["message_id"]
                or not isinstance(payload["source"], str)):
                raise MobileUiRpcInvalidRequest("检索卡片缺少消息或会话")
            return inspector.for_turn(session_id, payload["message_id"], payload["source"],
                                      ctx.require(TURN_PROJECTION), offset=offset)
        if method == "inspector.recent":
            try:
                page = InspectorPage.model_validate(payload)
            except ValidationError as error:
                raise MobileUiRpcInvalidRequest("检索页码或每页数量无效") from error
            return inspector.recent(page=page.page, page_size=page.page_size)
        if method == "inspector.detail":
            if set(payload) != {"query_id"} or not isinstance(payload["query_id"], str):
                raise MobileUiRpcInvalidRequest("请选择一条检索记录")
            detail = inspector.mobile_detail(payload["query_id"])
            if detail is None:
                raise MobileUiRpcInvalidRequest("检索记录不存在，请刷新列表")
            return detail
        raise MobileUiRpcInvalidRequest(f"不支持的 Akasha 查询：{method}")

    _ = await ctx.require(UI_SLOTS).register_mobile(
        ctx, MobileUiDefinition(module="message_ui.js", stylesheet="message_ui.css",
                                slots=("turn.before_reasoning",),
                                navigation=MobileUiNavigation(label="Akasha Inspector",
                                    description="查看实际检索及呈现的原消息")), query=query,
    )

    def select_learning() -> tuple[str, LearningConfig, str]:
        try:
            descriptor = ctx.require(EMBEDDINGS).describe()
            if memory_rule is not None and (descriptor.identity, descriptor.dimensions) != (
                memory_rule.embedding_model, memory_rule.dimension,
            ):
                raise EmbeddingSpaceMismatchError("默认 embedding 空间已变化，需显式重建 Akasha")
        except (ModelUnavailableError, DriverUnavailableError, EmbeddingSpaceMismatchError) as error:
            health.degrade(str(error))
            raise
        except ValueError as error:
            health.degrade(str(error))
            raise EmbeddingSpaceMismatchError(str(error)) from error
        rule = LearningConfig(embedding_model=descriptor.identity, dimension=descriptor.dimensions,
                              sources=config.sources)
        identity = ctx.require(BINDINGS).bind(AKASHA_LEARNING, rule.model_dump())
        return identity, rule, descriptor.model_id

    @asynccontextmanager
    async def open_embedding(model_id: str):
        # 纯学习 binding 会进入另一归档 Root；模型调用回到提供此 opener 的确切 Root。
        async with ctx.runtime_scope():
            async with ctx.require(EMBEDDINGS).bind(model_id=model_id) as model:
                yield model

    def embedder(rule: LearningConfig, model_id: str):
        async def embed(texts: list[str]) -> list[list[float]]:
            async with open_embedding(model_id) as model:
                if model.descriptor.identity != rule.embedding_model:
                    raise ValueError("实际 embedding 模型不匹配固定学习 binding")
                result = await model.embed(texts)
                return [list(vector) for vector in result.vectors]
        return embed

    def select_interest() -> tuple[LearningConfig, Embed]:
        _identity, rule, model_id = select_learning()
        return rule, embedder(rule, model_id)

    _ = await ctx.provide(SEMANTIC_INTEREST, SemanticInterest(
        learning, ctx.require(MESSAGE_CATALOG), ctx.require(MESSAGE_EMBEDDINGS), select_interest,
    ))

    def unavailable() -> MaterialData:
        reminder: Mapping[str, object] = {
            "name": "status", "text": f"## Akasha 状态\n召回不可用：{health.reason}",
            "priority": 300,
        }
        return {
            "reminders": (reminder,),
        }

    async def prepare(snapshot: tuple[Message, ...], source: str) -> MaterialData:
        if running:
            if not await start_if_available():
                return unavailable()
            assert memory is not None
            return await memory.prepare(snapshot, source)
        # 归档和显式程序只查询已发布图的副本，不取得正式学习 writer。
        try:
            identity, rule, model_id = select_learning()
        except (ModelUnavailableError, DriverUnavailableError, EmbeddingSpaceMismatchError):
            return unavailable()
        except MemoryRebuildRequiredError as error:
            health.degrade(str(error))
            return unavailable()
        bindings = ctx.require(BINDINGS)
        query_records = records()
        try:
            async with bindings.open(identity, AKASHA_LEARNING) as (selected, _metadata):
                async with read_memory(
                    memory_path, catalog=ctx.require(MESSAGE_CATALOG),
                    embeddings=ctx.require(MESSAGE_EMBEDDINGS), bindings=bindings,
                    config=settings.memory_config(), embedding_space=(rule.embedding_model, rule.dimension),
                    allow_initial=True,
                ) as (cycle, state):
                    result = await prepare_materials(
                        snapshot, source, cycle=cycle, state=state,
                        catalog=ctx.require(MESSAGE_CATALOG), embeddings=ctx.require(MESSAGE_EMBEDDINGS),
                        bindings=bindings, learning_binding=identity, learning=selected, rule=rule,
                        records=query_records, embed_batch=embedder(rule, model_id),
                        limit=settings.context_recall_limit, max_chars=settings.inject_max_chars,
                    )
        except EmbeddingSpaceMismatchError as error:
            health.degrade(str(error))
            return unavailable()
        health.recover()
        return result

    _ = await ctx.require(MATERIALS).register(ctx, name="akasha", prepare=prepare, priority=400)

    # 1. Feedback 读取已发布目标；归档调用不依赖正式运行事件或内存指针。
    actions: tuple[Literal["remember", "forget"], ...] = ("remember", "forget")
    for action in actions:
        @asynccontextmanager
        async def open_feedback(
            candidates: object, action: Literal["remember", "forget"] = action
        ) -> AsyncGenerator[FeedbackTool]:
            yield FeedbackTool(
                action,
                learning,
                ctx.require(BINDINGS),
                lambda: load_message_nodes(memory_path),
            )

        tool_refs.append(
            await catalog.register(
                ctx,
                name=f"{action}_memory",
                description=(
                    "记住明确确认的内容"
                    if action == "remember"
                    else "遗忘明确撤回的内容"
                ),
                parameters=FeedbackArguments.model_json_schema(),
                open=open_feedback,
                idempotent=True,
            )
        )

    def capture_recall(options: Mapping[str, object]) -> Mapping[str, object]:
        if options:
            raise ValueError("召回工具没有额外 binding 配置")
        try:
            identity = ctx.require(EMBEDDINGS).save_binding(ctx.require(BINDINGS))
        except (ModelUnavailableError, DriverUnavailableError) as error:
            return RecallBinding(embedding_binding=None, unavailable=str(error)).model_dump()
        return RecallBinding(embedding_binding=identity, unavailable=None).model_dump()

    @asynccontextmanager
    async def open_recall(captured: Mapping[str, object]) -> AsyncGenerator[RecallTool]:
        selected = RecallBinding.model_validate(dict(captured))
        bindings = ctx.require(BINDINGS)
        def select() -> tuple[str, str]:
            if selected.embedding_binding is None:
                assert selected.unavailable is not None
                raise ModelUnavailableError(selected.unavailable)
            saved = read_embedding_binding(bindings, selected.embedding_binding)
            rule = LearningConfig(embedding_model=saved.space_identity, dimension=saved.dimensions,
                                  sources=config.sources)
            identity = bindings.bind(AKASHA_LEARNING, rule.model_dump())
            return identity, selected.embedding_binding
        yield RecallTool(
            memory=memory_path, config=settings.memory_config(),
            catalog=ctx.require(MESSAGE_CATALOG), embeddings=ctx.require(MESSAGE_EMBEDDINGS),
            bindings=bindings, select_learning=select, records=records(),
            open_embedding=partial(open_saved_embedding, bindings), max_chars=settings.inject_max_chars,
        )

    tool_refs.append(
        await catalog.register(
            ctx,
            name="recall_memory",
            description="从记忆图召回历史对话，返回原始 Message 引用",
            parameters=RecallArguments.model_json_schema(),
            open=open_recall,
            capture=capture_recall,
            idempotent=True,
            risk="read-only",
        )
    )
    _ = await ctx.provide(AKASHA_TOOLS, catalog.view(*tool_refs))
    # 只读账本按声明的 workspace root 解析学习图；不暴露 writer 或任意路径。
    _ = await ctx.provide(AKASHA_MEMORY_PATH, lambda: memory_path)

    async def close_memory() -> None:
        if memory is not None:
            await memory.close()
    _ = await ctx.effect(lambda: close_memory, label="message-memory")

    async def start_if_available() -> bool:
        """模型设置后在首次实际使用时启用；同一 Root 只取得一个学习 writer。"""
        nonlocal memory, memory_rule
        async with start_lock:
            # 1. 未配置或空间变化只停用记忆；其他数据损坏仍明确失败。
            try:
                identity, rule, model_id = select_learning()
            except (ModelUnavailableError, DriverUnavailableError, EmbeddingSpaceMismatchError):
                return False
            except MemoryRebuildRequiredError as error:
                # 旧消费版本的图只能由显式重建替换，不能假装可用。
                health.degrade(str(error))
                return False
            if memory is not None:
                health.recover()
                return True
            try:
                consumer = await MessageConsumer.load(
                    memory_path, catalog=ctx.require(MESSAGE_CATALOG),
                    embeddings=ctx.require(MESSAGE_EMBEDDINGS), bindings=ctx.require(BINDINGS),
                    config=settings.memory_config(),
                )
            except MemoryRebuildRequiredError as error:
                health.degrade(str(error))
                return False
            runtime_records = records()
            prepared = MessageMemory(
                consumer, catalog=ctx.require(MESSAGE_CATALOG), embeddings=ctx.require(MESSAGE_EMBEDDINGS),
                bindings=ctx.require(BINDINGS), learning_binding=identity, records=runtime_records,
                embed_batch=embedder(rule, model_id), limit=settings.context_recall_limit,
                max_chars=settings.inject_max_chars,
            )
            # 2. 新选择必须与已有图一致；失败先归还 writer，绝不自动重建。
            try:
                await prepared.consume()
            except EmbeddingSpaceMismatchError as error:
                await prepared.close()
                health.degrade(str(error))
                return False
            except MemoryRebuildRequiredError as error:
                await prepared.close()
                health.degrade(str(error))
                return False
            except BaseException:
                await prepared.close()
                raise
            memory, memory_rule = prepared, rule
            health.recover()
            return True

    async def rebuild_now() -> str:
        """全量重放 canonical 来源；失败时已发布学习图保持不变。"""
        nonlocal memory, memory_rule
        # 1. 先确认 embedding 空间可用，避免无谓地停掉在线学习。
        identity, rule, model_id = select_learning()
        async with start_lock:
            # 2. 先归还唯一 writer，再生成候选；同一时刻只有一个学习 writer。
            if memory is not None:
                await memory.close()
                memory, memory_rule = None, None
            report = await rebuild_from_catalog(
                catalog=ctx.require(MESSAGE_CATALOG), embeddings=ctx.require(MESSAGE_EMBEDDINGS),
                bindings=ctx.require(BINDINGS), config=settings.memory_config(),
                learning_binding=identity, embed_batch=embedder(rule, model_id),
                memory_path=memory_path, backup_root=rebuild_backup_root,
            )
        # 3. 用同一启动边界重新装载；装载失败必须让调用者看到。
        if not await start_if_available():
            raise RuntimeError("Akasha 重建后无法重新装载学习图")
        return manifest_json(report)

    async def run_rebuild() -> CommandResult:
        """操作者显式确认后全量重建。"""
        return CommandResult("success", "Akasha 重建完成：" + await rebuild_now())

    def _note_replay(payload: Mapping[str, object]) -> None:
        """把重放状态写进 memory root；这是安装期唯一稳定可读的证据面。"""

        path = rebuild_request_paths[0].with_name(_REPLAY_STATUS_NAME)
        try:
            path.write_text(
                json.dumps({"at": datetime.now(UTC).isoformat(), **payload},
                           ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except OSError:
            return

    async def run_pending_rebuild() -> None:
        """消费迁移登记的一次性重放请求；只有成功后才移除凭据。"""

        pending = tuple(path for path in rebuild_request_paths if path.exists())
        # 重放是安装期动作，必须留下可读证据：容器日志级别可能看不到插件 logger。
        _note_replay({
            "phase": "scan",
            "checked": [str(path) for path in rebuild_request_paths],
            "pending": [str(path) for path in pending],
        })
        if not pending:
            return
        logger.info("Akasha 发现一次性重放请求: %s", [str(path) for path in pending])
        try:
            _note_replay({"phase": "rebuilding", "pending": [str(path) for path in pending]})
            detail = await rebuild_now()
            _note_replay({"phase": "completed", "detail": detail})
        except (ModelUnavailableError, DriverUnavailableError, EmbeddingSpaceMismatchError) as error:
            # 缺模型不阻塞启动：请求保留，下一次真实输入或重启再试。
            _note_replay({"phase": "degraded", "reason": str(error)})
            health.degrade(f"待执行的 Akasha 重放需要可用 embedding 空间: {error}")
            return
        except BaseException as error:
            _note_replay({"phase": "failed", "error": f"{type(error).__name__}: {error}"})
            raise
        for path in pending:
            path.unlink(missing_ok=True)
        health.recover()
        logger.info("Akasha 一次性重放完成: %s", detail)

    # 3. 通知只唤醒消费者；模型后配时下一条输入也会经过同一启动边界。
    async def follow() -> None:
        async for _heads in ctx.require(MESSAGE_CATALOG).follow():
            async with ctx.runtime_scope():
                await run_pending_rebuild()
                if await start_if_available():
                    assert memory is not None
                    _ = await memory.consume()

    async def start(_event: object) -> None:
        nonlocal watcher, running, inspector
        if running:
            raise RuntimeError("Akasha 消息运行重复启动")
        running = True
        async with ctx.runtime_scope():
            # Inspector 只读已保存查询和 Message；它不需要 embedding 或学习 writer。
            runtime_records = records()
            inspector = RecallInspector(read=runtime_records.read, list_records=runtime_records.list,
                                        catalog=ctx.require(MESSAGE_CATALOG))
            await run_pending_rebuild()
            _ = await start_if_available()
        watcher = await ctx.spawn(follow(), name="akasha-messages")

    async def stop(_event: object) -> None:
        nonlocal running
        running = False
        try:
            if watcher is not None:
                _ = watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass
        finally:
            await close_memory()

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
