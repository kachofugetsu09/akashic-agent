"""候选 Message 摘要入口；正式 manifest 在完整迁移验收后切换。"""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, Context
from agent.plugin_composition.models import BoundChatModel, ModelRequest, ModelRole
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, OWNER_STATE
from plugins.context.api import ContextModel, ContextOverflow, Materials, Summary, summary_range
from plugins.context.plugin import CONTEXT
from plugins.context.materials import MATERIALS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.message import Message

from .records import COMPACTION_SUMMARIES, SummaryLookup, SummaryRecord, SummaryRecords
from .message_summary import SummaryError, closed_groups, summarize, summary_groups, window_starts

api_version = 3
name = "compaction"
version = "4.0.0"
desc = "按不可变消息前缀发布摘要，并为已使用的摘要保留原始读取口"
inject = (MATERIALS, CONTEXT, OWNER_STATE, BINDINGS, MESSAGE_CATALOG, CHAT_MODELS, TURN_PROJECTION)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep_recent_tokens: int = Field(default=20_000, gt=0, strict=True)


async def apply(ctx: Context, config: Config) -> None:
    """注册只读材料和归档解析；apply 不打开 writer 或调用模型。"""
    def records() -> SummaryRecords:
        return SummaryRecords(ctx.require(OWNER_STATE).open(ctx))

    def read(reference: str) -> SummaryRecord | None:
        return records().read(reference)

    lookup = SummaryLookup(read)
    _ = await ctx.provide(COMPACTION_SUMMARIES, lookup)

    def material(record: SummaryRecord) -> Summary:
        reference = ctx.require(BINDINGS).bind(COMPACTION_SUMMARIES, {
            "record_ref": record.reference, "session_id": record.session_id,
        })
        return Summary(reference, record.source_message_ids, record.content)

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Materials:
        # TODO(v4-legacy-summary): 旧 ledger/prepare/receipt 缺少新调用及使用出处；
        # 保留旧恢复资料，待明确转换合同后接入，不能补造 Summary head。
        if not snapshot:
            return Materials("")
        record = records().head(snapshot[0].session_id)
        if record is None:
            return Materials("")
        return Materials("", summary=material(record))

    async def reduce(snapshot: tuple[Message, ...], materials: Materials, request: ModelRequest,
                     model: BoundChatModel, projection: ContextModel, *, source: str, force: bool) -> Summary | None:
        """选完整旧前缀、生成摘要，再把不可变记录与 head 一起发布。"""
        # 1. 容量与近期保留均按当前已固定的业务模型判断。
        window = projection.context_window
        before = projection.estimate(request)
        if window is None or not snapshot or (not force and before < int(window * 0.74)):
            return None
        parent = records().head(snapshot[0].session_id)
        if (None if parent is None else material(parent)) != materials.summary:
            raise ValueError("本次已取得摘要与当前 Session head 不一致")
        turns = ctx.require(TURN_PROJECTION)
        if parent is None:
            origin: int | None = None
            # 首次窗口从最近完整单元累加；完整业务输入不得越过软水位或硬边界。
            for index in reversed(window_starts(snapshot, turns)):
                try:
                    candidate = ctx.require(CONTEXT).build(snapshot, materials=materials, model=projection,
                        tools=request.tools, max_output_tokens=request.max_output_tokens,
                        window_start=snapshot[index].message_id)
                except ContextOverflow:
                    break
                if projection.estimate(candidate) > int(window * 0.74):
                    break
                origin = index
            if origin is None:
                raise SummaryError("当前完整工作与固定材料超过首次窗口容量")
            start = origin
        else:
            covered = summary_range(snapshot, parent.source_message_ids)
            origin, start = covered.start, covered.stop
        groups = closed_groups(snapshot, turns, after=start)
        selected: tuple[tuple[Message, ...], ...] = ()
        # 原文保留量来自实际 Model 投影，包含尚未闭合的尾部与当前输入。
        for size in range(len(groups), 0, -1):
            after_seq = groups[size - 1][-1].seq
            tail = projection.render(snapshot, after_seq=after_seq, fresh=True)
            if projection.estimate(tail) >= config.keep_recent_tokens:
                selected = groups[:size]
                break
        if not selected:
            raise SummaryError("近期原文保留量内没有合法摘要切点")
        inputs = summary_groups(selected, snapshot)
        if not inputs:
            raise SummaryError("可选范围没有可用于摘要的资料")
        # 2. 嵌套 execution 复用调用者已经固定的角色，不重读模型配置。
        async with ctx.require(CHAT_MODELS).execution() as execution:
            text, calls = await summarize(inputs, previous="" if parent is None else parent.content,
                                          model=model, fallback=execution.chat(ModelRole.DEFAULT))
        count = start + sum(len(group) for group in selected)
        record = SummaryRecord(
            reference=uuid4().hex, session_id=snapshot[0].session_id,
            generation=1 if parent is None else parent.generation + 1,
            parent=None if parent is None else parent.reference,
            source_message_ids=tuple(message.message_id for message in snapshot[origin:count]),
            content=text, model_call_ids=calls, trigger="context_overflow" if force else "soft_limit",
            context_window=window, max_output_tokens=request.max_output_tokens,
            keep_recent_tokens=config.keep_recent_tokens, tokens_before=before, tokens_after=0,
        )
        summary = material(record)
        try:
            after_request = ctx.require(CONTEXT).build(snapshot, materials=replace(materials, summary=summary),
                model=projection, tools=request.tools, max_output_tokens=request.max_output_tokens)
        except ContextOverflow as overflow:
            after_request = overflow.request
        after = projection.estimate(after_request)
        if after >= before:
            raise SummaryError("摘要没有降低本次完整请求容量")
        if after > int(window * 0.74) or after + request.max_output_tokens > window:
            raise SummaryError("摘要后的完整请求仍超过模型软水位或硬边界")
        record = record.model_copy(update={"tokens_after": after})
        # 3. binding 可以先固定，但读者只有在摘要事务成功后才取得此引用。
        reader = ctx.require(MESSAGE_CATALOG).reader(record.session_id)
        _ = records().publish(record, reader, parent=parent)
        return summary

    _ = await ctx.require(MATERIALS).register(ctx, name="compaction", prepare=prepare, reduce=reduce)
