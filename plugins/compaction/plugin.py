"""候选 Message 摘要入口；正式 manifest 在完整迁移验收后切换。"""
from __future__ import annotations

from agent.plugin_composition.models import ModelError
from collections.abc import Callable, Mapping, Sequence
from uuid import uuid4

from core.common.file_io import run_file_io

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
)
from plugins.models.contract import CHAT_MODELS
from agent.plugin_composition.models import (
    ContextLengthError,
    ModelRequest,
    ModelTimeoutError,
    RateLimitError,
    TransportError,
)
from plugins.models.contract import BoundChatModel
from plugins.context.contract import ReductionStatus
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE
from agent.plugin_contracts import Input, Message

from .records import StoredSummary, SummaryLookup, SummaryRecord, SummaryRecords
from .message_summary import SummaryError, closed_groups, source_text, summarize, summary_groups, window_starts
from ._boundaries import (
    COMPACTION_READER, COMPACTION_SUMMARIES, CONTEXT, MATERIALS, TURN_PROJECTION,
    ContextModel, MaterialData, TurnProjection,
)

api_version = 3
name = "compaction"
version = "4.1.0"
desc = "按不可变消息前缀发布摘要，并为已使用的摘要保留原始读取口"
inject = (MATERIALS, CONTEXT, OWNER_STATE, BINDINGS, MESSAGE_CATALOG, CHAT_MODELS, TURN_PROJECTION)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep_recent_tokens: int = Field(default=20_000, gt=0, strict=True)


class _CompactionReader:
    """Markdown 只取得 compaction 的纯读取算法，不取得发布或模型权限。"""

    def __init__(self, *, settled_prefixes: Callable[[tuple[Message, ...]], tuple[int, ...]]):
        self._settled_prefixes = settled_prefixes

    def source_text(self, messages: Sequence[Message]) -> str:
        return source_text(messages)

    def window_starts(
        self, messages: tuple[Message, ...], projection: TurnProjection,
    ) -> tuple[int, ...]:
        return window_starts(messages, projection, settled_prefixes=self._settled_prefixes)

    def summary_groups(
        self, groups: tuple[tuple[Message, ...], ...], snapshot: tuple[Message, ...],
    ) -> tuple[tuple[Message, ...], ...]:
        return summary_groups(groups, snapshot)


async def apply(ctx: Context) -> None:
    """注册只读材料和归档解析；apply 不打开 writer 或调用模型。"""
    config = Config.model_validate(ctx.config)
    context = ctx.require(CONTEXT)

    def records() -> SummaryRecords:
        return SummaryRecords(ctx.require(OWNER_STATE).open(ctx))

    # 状态查询在线程执行；启动时取得窄读取口，不在线程中重新申请 owner 写权限。
    read_record: Callable[[str], StoredSummary | None] | None = None
    read_current: Callable[[str], StoredSummary | None] | None = None

    def read(reference: str) -> StoredSummary | None:
        if read_record is None:
            raise RuntimeError("摘要状态读取口尚未启动")
        return read_record(reference)

    def head(session_id: str) -> StoredSummary | None:
        if read_current is None:
            raise RuntimeError("摘要状态读取口尚未启动")
        return read_current(session_id)

    async def start(_event: object) -> None:
        nonlocal read_current, read_record
        state = records()
        read_record = state.read
        read_current = state.head

    async def stop(_event: object) -> None:
        nonlocal read_current, read_record
        read_record = None
        read_current = None

    _ = await ctx.on(RUNTIME_STARTED, start)
    _ = await ctx.on(RUNTIME_STOPPING, stop)
    lookup = SummaryLookup(read, head)
    _ = await ctx.provide(COMPACTION_SUMMARIES, lookup)
    _ = await ctx.provide(COMPACTION_READER, _CompactionReader(
        settled_prefixes=context.settled_prefixes,
    ))

    def material(record: StoredSummary) -> MaterialData:
        reference = ctx.require(BINDINGS).bind(COMPACTION_SUMMARIES, {
            "record_ref": record.reference, "session_id": record.session_id,
        })
        return {
            "reference": reference,
            "source_message_ids": record.source_message_ids,
            "content": record.content,
        }

    # prepare 的只读缓存：head 行版本不变时复用已核对的完整 lineage；
    # 发布走 records().head/publish 原路径，缓存只服务材料读取。
    head_cache: dict[str, tuple[int, StoredSummary]] = {}

    async def prepare(snapshot: tuple[Message, ...], source: str) -> MaterialData:
        if not snapshot:
            return {}
        session_id = snapshot[0].session_id
        state = records()

        def load() -> StoredSummary | None:
            head = state.read_head(session_id, known=head_cache.get(session_id))
            if head is None:
                head_cache.pop(session_id, None)
                return None
            version, record = head
            head_cache[session_id] = (version, record)
            return record

        # 命中与未命中都要读 head 行版本（SQL），恢复在文件线程完成；
        # 只有纯内存命中才值得省 executor 往返，这里不是（评审 #1142）。
        record = await run_file_io(load)
        if record is None:
            return {}
        return {"summary": material(record)}

    async def compact(snapshot: tuple[Message, ...], materials: MaterialData, request: ModelRequest,
                      model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
                      on_status: ReductionStatus | None = None) -> MaterialData | None:
        """按已结算工具批次选旧前缀，再原子发布一份真实摘要。"""
        # 1. 容量与近期保留均按当前已固定的业务模型判断。
        reminder = context.reminder_content(materials)
        reminder_input_id = (
            next(
                (
                    message.message_id
                    for message in reversed(snapshot)
                    if message.source == source and isinstance(message.body, Input)
                ),
                None,
            )
            if reminder is not None
            else None
        )
        window = projection.context_window
        before = projection.estimate(request)
        if window is None or not snapshot or (not force and before < int(window * 0.74)):
            return None
        state = records()
        parent = await run_file_io(lambda: state.head(snapshot[0].session_id))
        current_summary = materials.get("summary")
        if current_summary is not None and not isinstance(current_summary, Mapping):
            raise TypeError("Context 材料摘要必须是对象")
        if (None if parent is None else material(parent)) != current_summary:
            raise ValueError("本次已取得摘要与当前 Session head 不一致")
        turns = ctx.require(TURN_PROJECTION)
        if parent is None:
            # 摘要模型仍只取一个近期窗口；退出请求的更早资料写入 omitted 分区。
            origin = start = 0
        else:
            covered = context.summary_range(snapshot, parent.source_message_ids)
            origin, start = covered.start, covered.stop
        groups = closed_groups(
            snapshot, turns, settled_prefixes=context.settled_prefixes, after=start,
        )
        selected: tuple[tuple[Message, ...], ...] = ()
        fallback: tuple[tuple[Message, ...], ...] = ()
        retained_tokens = 0
        # 原文保留量来自实际 Model 投影，包含尚未闭合的尾部与当前输入。
        for size in range(len(groups), 0, -1):
            after_seq = groups[size - 1][-1].seq
            tail = projection.render(
                snapshot,
                after_seq=after_seq,
                fresh=True,
                current_reminder=reminder if reminder_input_id is not None else None,
                current_reminder_input_id=reminder_input_id,
            )
            # 先用占位摘要核对完整输入；真实摘要生成后还会再次核对硬容量。
            count = start + sum(len(group) for group in groups[:size])
            planned = {**materials, "summary": {
                "reference": "compaction-plan", "content": "待生成摘要",
                "source_message_ids": tuple(message.message_id for message in snapshot[origin:count]),
            }}
            candidate, error = context.build_attempt(
                snapshot, materials=planned, model=projection, tools=request.tools,
                max_output_tokens=request.max_output_tokens,
                current_reminder_input_id=reminder_input_id,
            )
            if error is not None or projection.estimate(candidate) > int(window * 0.74):
                continue
            if not fallback:
                fallback = groups[:size]
                retained_tokens = projection.estimate(tail)
            if projection.estimate(tail) >= config.keep_recent_tokens:
                selected = groups[:size]
                break
        if not selected:
            selected = fallback
            if selected and on_status is not None:
                await on_status(f"近期原文目标 {config.keep_recent_tokens:,} tokens 无法满足容量；"
                                f"本次保留约 {retained_tokens:,} tokens，当前输入与未结算工具仍保留。", True)
        if not selected:
            raise SummaryError("没有能降低完整请求容量的已结算工具批次切点")
        inputs = summary_groups(selected, snapshot)
        if not inputs:
            raise SummaryError("可选范围没有可用于摘要的资料")
        # 2. 嵌套 execution 复用调用者已经固定的角色，不重读模型配置。
        async with ctx.require(CHAT_MODELS).execution() as execution:
            async def report_fallback(text: str) -> None:
                if on_status is not None:
                    await on_status(text, True)

            text, calls, summarized = await summarize(
                inputs, previous="" if parent is None else parent.content,
                model=model, fallback=execution.chat("default"),
                on_fallback=report_fallback,
            )
        count = start + sum(len(group) for group in selected)
        summary_message_ids = tuple(
            message.message_id for group in summarized for message in group
        )
        summary_id_set = set(summary_message_ids)
        added = snapshot[start:count]
        record = SummaryRecord(
            reference=uuid4().hex, session_id=snapshot[0].session_id,
            generation=1 if parent is None else parent.generation + 1,
            parent=None if parent is None else parent.reference,
            source_message_ids=tuple(message.message_id for message in snapshot[origin:count]),
            summary_message_ids=summary_message_ids,
            omitted_message_ids=tuple(
                message.message_id for message in added
                if message.message_id not in summary_id_set
            ),
            content=text, model_call_ids=calls, trigger="context_overflow" if force else "soft_limit",
            context_window=window, max_output_tokens=request.max_output_tokens,
            keep_recent_tokens=config.keep_recent_tokens, tokens_before=before, tokens_after=0,
        )
        summary = material(record)
        after_materials = dict(materials)
        after_materials["summary"] = summary
        after_request, error = ctx.require(CONTEXT).build_attempt(
            snapshot, materials=after_materials, model=projection,
            tools=request.tools, max_output_tokens=request.max_output_tokens,
            current_reminder_input_id=reminder_input_id,
        )
        if error is not None:
            raise SummaryError(error)
        after = projection.estimate(after_request)
        if after >= before:
            raise SummaryError("摘要没有降低本次完整请求容量")
        if after + request.max_output_tokens > window:
            raise SummaryError(f"摘要后的输入约 {after:,} tokens，加输出预留仍超过硬容量")
        record = record.model_copy(update={"tokens_after": after})
        # 3. binding 可以先固定，但读者只有在摘要事务成功后才取得此引用。
        reader = ctx.require(MESSAGE_CATALOG).reader(record.session_id)
        _ = await state.publish(
            record, reader, parent=parent, summary_range=context.summary_range,
        )
        return summary

    async def reduce(snapshot: tuple[Message, ...], materials: MaterialData, request: ModelRequest,
                     model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
                     on_status: ReductionStatus | None = None) -> MaterialData | None:
        """软水位失败可继续；硬容量或 provider 明确拒绝必须保留失败原因。"""
        window = projection.context_window
        before = projection.estimate(request)
        if window is None or not snapshot or (not force and before < int(window * 0.74)):
            return None
        budget = (f"模型 {model.descriptor.model_id}；窗口 {window:,}；输入估算 {before:,}；"
                  f"输出预留 {request.max_output_tokens:,}；原文目标 {config.keep_recent_tokens:,} tokens")
        if on_status is not None:
            await on_status("正在压缩上下文：" + budget, False)
        notices: list[str] = []
        async def report(text: str, retain: bool) -> None:
            if retain:
                notices.append(text)
            if on_status is not None:
                await on_status(text, retain)

        try:
            return await compact(snapshot, materials, request, model, projection,
                                 source=source, force=force, on_status=report)
        except (RuntimeError, TimeoutError, SummaryError) as error:
            if not (ModelError.matches(error, ContextLengthError, ModelTimeoutError, RateLimitError, TransportError) or isinstance(error, SummaryError)):
                raise
            detail = f"上下文压缩失败：{type(error).__name__}: {error}。{budget}。"
            if force or before + request.max_output_tokens > window:
                raise SummaryError("\n".join((*notices, detail + "本次请求被阻断，未继续业务生成。"))) from error
            if on_status is not None:
                await on_status(detail + "原请求仍满足硬容量，本次继续生成。", True)
            return None

    _ = await ctx.require(MATERIALS).register(ctx, kind="context", name="compaction", prepare=prepare, reduce=reduce, priority=500)
