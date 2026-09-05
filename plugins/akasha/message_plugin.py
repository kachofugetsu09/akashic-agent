"""新链路的 Akasha 组合入口；完整功能验收后统一切换 manifest。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.plugin_composition import EMBEDDINGS, RUNTIME_STARTED, RUNTIME_STOPPING, Context, ServiceKey, UI_SLOTS, MobileUiDefinition, MobileUiNavigation, MobileUiRpcInvalidRequest
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.commands import COMMANDS, CommandDefinition, CommandInvocation, CommandResult
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, OWNER_STATE
from plugins.content.api import ContentSchema
from plugins.content.plugin import CONTENT
from plugins.context.api import Materials
from plugins.context.materials import MATERIALS
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.message import Message

from .application.consumer import MessageConsumer
from .config import AkashaConfig, resolve_memory_path
from .infrastructure.consumption import load_message_nodes
from .inspector import RecallInspector
from .learning import AKASHA_LEARNING, Learning, LearningConfig
from .recall_tool import RecallArguments, RecallTool, check_recall
from .recalls import Recall, RecallRecords
from .runtime import MessageMemory
from .tools import FeedbackArguments, FeedbackTool, check_feedback

api_version = 3
name = "akasha"
version = "4.0.0"
desc = "从消息学习并提供普通 Context 材料与记忆工具"
inject = (TURN_PROJECTION, CONTENT, MATERIALS, TOOLS, EMBEDDINGS,
          BINDINGS, MESSAGE_CATALOG, MESSAGE_EMBEDDINGS, OWNER_STATE, UI_SLOTS, COMMANDS)
workspace_roots = ("memory",)


class Config(BaseModel):
    """同名旧配置由 Manager 一次读取并归档，再转换为现有 Akasha 配置。"""

    model_config = ConfigDict(extra="forbid")
    sources: tuple[str, ...] = Field(default=("conversation",), min_length=1)
    db_path: str = AkashaConfig.db_path
    index_path: str = AkashaConfig.index_path
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


AKASHA_RECORDS = ServiceKey[Callable[[str], Recall | None]]("akasha.recalls.v1")


async def apply(ctx: Context, config: Config) -> None:
    """注册纯学习规则和延迟工具；正式启动事件才取得唯一学习 writer。"""
    async def request_reindex(_invocation: CommandInvocation) -> CommandResult:
        # TODO: 固定旧学习规则与来源的重建合同确认后，再接管旧请求与启动流程。
        return CommandResult("error", "新消息链路尚未接管 Akasha 重建；原学习图与旧重建记录保持不变。")

    _ = await ctx.require(COMMANDS).register(ctx, CommandDefinition(
        name="akasha_reindex", description="查询 Akasha 重建是否可用",
        handler=request_reindex, read_only=True, input_hint="confirm",
    ))
    settings = config.settings()
    memory_path = resolve_memory_path(ctx.workspace_root("memory"), settings.db_path)
    index_path = resolve_memory_path(ctx.workspace_root("memory"), settings.index_path)
    learning = Learning(ctx.require(TURN_PROJECTION), owner=ctx.runtime.plugin_id)
    _ = await ctx.provide(AKASHA_LEARNING, learning)
    _ = await ctx.require(CONTENT).register(ctx, ContentSchema(
        name="akasha", content={"akasha.feedback": check_feedback, "akasha.recall": check_recall},
    ))
    memory: MessageMemory | None = None
    watcher: asyncio.Task[None] | None = None

    def records() -> RecallRecords:
        return RecallRecords(ctx.require(OWNER_STATE).open(ctx))

    # 公开读取函数不暴露 owner transaction；归档 apply 也不会读取正式数据库。
    def read_recall(identity: str) -> Recall | None:
        return records().read(identity)
    _ = await ctx.provide(AKASHA_RECORDS, read_recall)

    inspector: RecallInspector | None = None

    def query(method: str, payload: dict[str, object], *, session_id: str | None,
              turn_id: str | None) -> dict[str, object]:
        if inspector is None:
            raise MobileUiRpcInvalidRequest("Akasha 查询读取尚未启动")
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
                                navigation=MobileUiNavigation(label="Akasha Inspector",
                                    description="查看实际检索及呈现的原消息")), query=query,
    )

    def select_learning() -> tuple[str, LearningConfig, str]:
        descriptor = ctx.require(EMBEDDINGS).describe()
        rule = LearningConfig(embedding_model=descriptor.identity, dimension=descriptor.dimensions,
                              sources=config.sources)
        identity = ctx.require(BINDINGS).bind(AKASHA_LEARNING, rule.model_dump())
        return identity, rule, descriptor.model_id

    def embedder(rule: LearningConfig, model_id: str):
        async def embed(texts: list[str]) -> list[list[float]]:
            async with ctx.require(EMBEDDINGS).bind(model_id=model_id) as model:
                if model.descriptor.identity != rule.embedding_model:
                    raise ValueError("实际 embedding 模型不匹配固定学习 binding")
                result = await model.embed(texts)
                return [list(vector) for vector in result.vectors]
        return embed

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Materials:
        if memory is None:
            raise RuntimeError("Akasha 消息运行尚未启动")
        return await memory.prepare(snapshot, source)

    _ = await ctx.require(MATERIALS).register(ctx, name="akasha", prepare=prepare)

    # 1. Feedback 读取已发布目标；归档调用不依赖正式运行事件或内存指针。
    actions: tuple[Literal["remember", "forget"], ...] = ("remember", "forget")
    for action in actions:
        @asynccontextmanager
        async def open_feedback(candidates: object, action: Literal["remember", "forget"] = action) -> AsyncGenerator[FeedbackTool]:
            yield FeedbackTool(action, learning, ctx.require(BINDINGS),
                               lambda: load_message_nodes(memory_path, index_path))
        _ = await ctx.require(TOOLS).register(
            ctx, name=f"{action}_memory", description="记住明确确认的内容" if action == "remember" else "遗忘明确撤回的内容",
            parameters=FeedbackArguments.model_json_schema(), open=open_feedback,
            idempotent=True, always_on=True,
        )

    @asynccontextmanager
    async def open_recall(candidates: object) -> AsyncGenerator[RecallTool]:
        def select() -> tuple[str, str]:
            identity, _rule, model_id = select_learning()
            return identity, model_id
        yield RecallTool(
            memory=memory_path, legacy_index=index_path, config=settings.memory_config(),
            catalog=ctx.require(MESSAGE_CATALOG), embeddings=ctx.require(MESSAGE_EMBEDDINGS),
            bindings=ctx.require(BINDINGS), select_learning=select, records=records(),
            embedding_api=ctx.require(EMBEDDINGS), max_chars=settings.inject_max_chars,
        )
    _ = await ctx.require(TOOLS).register(
        ctx, name="recall_memory", description="从记忆图召回历史对话，返回原始 Message 引用",
        parameters=RecallArguments.model_json_schema(), open=open_recall, idempotent=True,
        risk="read-only", always_on=True,
    )

    async def close_memory() -> None:
        if memory is not None:
            await memory.close()
    _ = await ctx.effect(lambda: close_memory, label="message-memory")

    # 2. 通知只唤醒消费者；实际消费上界、幂等与进度始终来自耐久日志和学习图。
    async def follow() -> None:
        async for _heads in ctx.require(MESSAGE_CATALOG).follow():
            async with ctx.runtime_scope():
                if memory is None:
                    raise RuntimeError("Akasha 消费任务缺少运行 owner")
                _ = await memory.consume()

    async def start(_event: object) -> None:
        nonlocal memory, watcher, inspector
        if memory is not None:
            raise RuntimeError("Akasha 消息运行重复启动")
        async with ctx.runtime_scope():
            identity, rule, model_id = select_learning()
            consumer = await MessageConsumer.load(
                memory_path, legacy_index=index_path, catalog=ctx.require(MESSAGE_CATALOG),
                embeddings=ctx.require(MESSAGE_EMBEDDINGS), bindings=ctx.require(BINDINGS),
                config=settings.memory_config(),
            )
            runtime_records = records()
            inspector = RecallInspector(read=runtime_records.read, list_records=runtime_records.list,
                                        catalog=ctx.require(MESSAGE_CATALOG))
            memory = MessageMemory(
                consumer, catalog=ctx.require(MESSAGE_CATALOG), embeddings=ctx.require(MESSAGE_EMBEDDINGS),
                bindings=ctx.require(BINDINGS), learning_binding=identity, records=runtime_records,
                embed_batch=embedder(rule, model_id), limit=settings.context_recall_limit,
                max_chars=settings.inject_max_chars,
            )
            _ = await memory.consume()
        watcher = await ctx.spawn(follow(), name="akasha-messages")

    async def stop(_event: object) -> None:
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
