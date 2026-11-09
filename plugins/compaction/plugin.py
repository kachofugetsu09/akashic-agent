"""候选 Message 摘要入口；正式 manifest 在完整迁移验收后切换。"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import partial
from collections.abc import Awaitable, Callable, Mapping, Sequence
import logging
from typing import Literal, cast
from uuid import uuid4
from weakref import WeakKeyDictionary

from core.common.file_io import run_file_io

from pydantic import BaseModel, ConfigDict, Field, field_validator

from agent.plugin_composition import (
    RUNTIME_STARTED,
    RUNTIME_STOPPING,
    Context,
)
from plugins.models.contract import (
    CHAT_MODELS, MODEL_CATALOG, BoundChatModel, ContentSafetyError, ContextLengthError, ContextModel, ModelChoice, ModelError, ModelExecution,
    ModelRequest, ModelTimeoutError, ModelUnavailableError, ModelUse, content_rejection_reason, is_content_rejection, model_options,
)
from plugins.context.contract import (
    CONTEXT, IDLE_COMPACT_MINUTES, IDLE_COMPACT_TOKENS, MATERIALS_V4, OUTPUT_RESERVE_TOKENS,
    MaterialData, ReductionStatus, SUMMARY_SOFT_LIMIT, idle_gap_anchor, idle_summary_due, output_reserve, summary_due,
)
from plugins.ledger.contract import BINDINGS
from plugins.ledger.contract import MESSAGE_CATALOG, OWNER_STATE
from plugins.ledger.contract import HistoryWindow, Input, Message

from .records import StoredSummary, SummaryLookup, SummaryRecord, SummaryRecords
from .message_summary import (
    BLOCKING_SUMMARY, OPTIONAL_SUMMARY, SUMMARY_DEADLINE_SECONDS, SummaryCall, SummaryCapacityError, SummaryError,
    clip_summary, closed_groups, source_text, summarize, summary_groups, window_starts,
)
from plugins.compaction.contract import COMPACTION_READER, COMPACTION_SUMMARIES
from plugins.turn_projection.contract import TURN_PROJECTION, TurnProjection
from agent.plugin_composition.plugin_config import PLUGIN_CONFIG
from agent.plugin_composition.model import ServiceKey
from plugins.ui.contract import Configuration, ModelSettings, mount_settings

logger = logging.getLogger(__name__)

api_version = 3
name = "compaction"
version = "4.1.0"
desc = "按不可变消息身份发布摘要，并为已使用的摘要保留原始读取口"
inject = (PLUGIN_CONFIG, MATERIALS_V4, CONTEXT, OWNER_STATE, BINDINGS, MESSAGE_CATALOG, CHAT_MODELS, MODEL_CATALOG, TURN_PROJECTION)


SETTINGS = ServiceKey[Configuration]("compaction.settings.v1")
SETTING_FIELDS = (
    "model",
    "keep_recent_tokens",
    "idle_scope",
    "idle_sessions",
    "idle_trigger_minutes",
    "idle_trigger_tokens",
)
# 摘要默认跟随本轮对话的模型（窗口与业务一致）；也可跟随默认或指定一个更便宜的模型。
# 摘要输入按摘要模型自己的窗口裁剪，所以必须是上下文窗口已确认的模型。
SUMMARY_MODEL = ModelUse("compaction", "上下文压缩", follows=("session", "default"), needs=frozenset({"context_window"}))
MAX_SUMMARY_FAILURES = 3
# 空闲压缩挡在回合前且本不必要：期限短，到期软失败。
IDLE_DEADLINE_SECONDS = 120.0
# 按计划节省不到请求估算的这一比例时不值得花钱（新摘要按父摘要原长估算）。
IDLE_MIN_SAVING = 0.2


@dataclass
class _SummaryFailures:
    boundary: tuple[str, str, str, str | None]
    count: int = 0
    detail: str = ""
    recovery: str = ""


# 失败出口跟随原因：摘要模型故障才指向认证和额度，容量问题的文案自带出口。
def _recovery(error: BaseException) -> str:
    if isinstance(error, SummaryCapacityError):
        return ""
    if ModelError.matches(error, ContextLengthError):
        return "摘要请求裁剪后仍被判超窗，请切换窗口更大的摘要模型或开始新对话。"
    return "请检查摘要模型的认证、额度或可用性，或切换摘要模型后重试。"


# 进程内空闲压缩记录：已试过的缺口属于会话，连续失败只对当前业务模型计数。
@dataclass
class _IdleAttempts:
    binding_id: str
    gap: str | None = None
    failures: int = 0


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    keep_recent_tokens: int = Field(default=20_000, gt=0, strict=True)
    idle_scope: Literal["selected", "all", "off"] = Field(default="selected")
    idle_sessions: tuple[str, ...] = Field(default=())
    idle_trigger_minutes: int = Field(default=IDLE_COMPACT_MINUTES, ge=0, strict=True)
    idle_trigger_tokens: int = Field(default=IDLE_COMPACT_TOKENS, gt=0, strict=True)
    model: ModelChoice = ModelChoice(follow="session")

    @field_validator("idle_sessions", mode="before")
    @classmethod
    def _check_idle_sessions(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str) or not isinstance(value, Sequence):
            raise ValueError("绑定对话必须是会话 ID 列表")
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            if not isinstance(item, str):
                raise ValueError("会话 ID 必须是字符串")
            sid = item.strip()
            if sid and sid not in seen:
                seen.add(sid)
                cleaned.append(sid)
        return tuple(cleaned)

    @field_validator("model")
    @classmethod
    def _check_model(cls, value: ModelChoice) -> ModelChoice:
        return SUMMARY_MODEL.check(value)


def _idle_minutes_for(config: Config, snapshot: tuple[Message, ...]) -> int:
    if not snapshot or config.idle_trigger_minutes <= 0 or config.idle_scope == "off":
        return 0
    if config.idle_scope == "all":
        return config.idle_trigger_minutes
    return config.idle_trigger_minutes if snapshot[0].session_id in config.idle_sessions else 0


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


# 摘要模型按插件配置取；回退目标是本轮业务模型与默认模型中与主选不同的那个。
# 1. 配置的模型取不到（含默认模型窗口未确认）时直接用本轮业务模型，并留下提示。
# 2. 主选是业务模型时回退默认模型，否则回退业务模型；两者相同则 summarize 不再回退。
async def summary_models(execution: ModelExecution, business: BoundChatModel, choice: ModelChoice,
                         report: Callable[[str], Awaitable[None]]) -> tuple[BoundChatModel, BoundChatModel]:
    try:
        chosen = await execution.use(SUMMARY_MODEL, choice)
    except (RuntimeError, TimeoutError) as failure:
        if not ModelError.matches(failure, ModelUnavailableError):
            raise
        await report(f"摘要模型不可用：{ModelError.read(failure)}；改用本轮模型 {business.descriptor.model_id}")
        return business, business
    if chosen.descriptor.model_id != business.descriptor.model_id:
        return chosen, business
    try:
        return chosen, await execution.use(SUMMARY_MODEL, ModelChoice(follow="default"))
    except (RuntimeError, TimeoutError) as failure:
        if not ModelError.matches(failure, ModelUnavailableError):
            raise
        return chosen, chosen


async def apply(ctx: Context) -> None:
    """注册只读材料和归档解析；apply 不打开 writer 或调用模型。"""
    config = Config.model_validate(ctx.config)
    context = ctx.require(CONTEXT)
    # 失败预算只属于当前执行 Task；新输入或换模型重新尝试，结束后不保留会话引用。
    failures: WeakKeyDictionary[asyncio.Task[object], _SummaryFailures] = WeakKeyDictionary()

    # 同一执行内的工具步共享预算，避免每一步重复调用已失败的摘要模型。
    def failure_state(snapshot: tuple[Message, ...], source: str, model: BoundChatModel) -> _SummaryFailures:
        task = asyncio.current_task()
        assert task is not None
        latest_input = next((message.message_id for message in reversed(snapshot)
                             if message.source == source and isinstance(message.body, Input)), None)
        boundary = (snapshot[0].session_id, source, model.descriptor.binding_id, latest_input)
        state = failures.get(task)
        if state is None or state.boundary != boundary:
            state = failures[task] = _SummaryFailures(boundary)
        return state

    # 只服务空闲路径，force 与硬水位不读它；有界，满了整体丢弃，代价只是多试一次。
    idle_attempts: dict[str, _IdleAttempts] = {}

    # 换模型清零连续失败；已试过的缺口不随模型改变。binding 含注册表版本，保存模型设置也清零：
    # 用户刚改过密钥或额度配置时应当再试，代价有上限（每次保存至多再付 3 次）。
    def idle_attempts_for(session_id: str, binding_id: str) -> _IdleAttempts:
        attempts = idle_attempts.get(session_id)
        if attempts is None:
            if len(idle_attempts) >= 256:
                idle_attempts.clear()
            attempts = idle_attempts[session_id] = _IdleAttempts(binding_id)
        elif attempts.binding_id != binding_id:
            attempts.binding_id, attempts.failures = binding_id, 0
        return attempts

    # 门控和缩减共用：范围、缺口与阈值都达标，这个缺口没试过，连续失败也未到上限。
    def idle_due(snapshot: tuple[Message, ...], estimated: int, model: BoundChatModel) -> bool:
        if estimated <= config.keep_recent_tokens or not idle_summary_due(
            estimated, snapshot, idle_minutes=_idle_minutes_for(config, snapshot), idle_tokens=config.idle_trigger_tokens,
        ):
            return False
        attempts = idle_attempts_for(snapshot[0].session_id, model.descriptor.binding_id)
        return attempts.failures < MAX_SUMMARY_FAILURES and attempts.gap != idle_gap_anchor(snapshot)

    def records() -> SummaryRecords:
        return SummaryRecords(ctx.require(OWNER_STATE).open(ctx))

    # 状态查询在线程执行；启动时取得窄读取口，不在线程中重新申请 owner 写权限。
    read_state: SummaryRecords | None = None
    read_current: Callable[[str], StoredSummary | None] | None = None

    # 摘要记录不可变：核对过完整父链的记录按引用复用，父链核对遇到已核对祖先即停；有界，满了整体丢弃。
    verified: dict[str, StoredSummary] = {}

    def read(reference: str) -> StoredSummary | None:
        if read_state is None:
            raise RuntimeError("摘要状态读取口尚未启动")
        record = verified.get(reference)
        if record is None:
            record = read_state.read(reference, known=verified.get)
            if record is not None:
                if len(verified) >= 64:
                    verified.clear()
                verified[reference] = record
        return record

    def head(session_id: str) -> StoredSummary | None:
        if read_current is None:
            raise RuntimeError("摘要状态读取口尚未启动")
        return read_current(session_id)

    # head 行版本不变时复用已核对的完整 lineage；发布走 records().head/publish 原路径，缓存只服务读取。
    head_cache: dict[str, tuple[int, StoredSummary]] = {}

    def current_head(state: SummaryRecords, session_id: str) -> StoredSummary | None:
        head = state.read_head(session_id, known=head_cache.get(session_id))
        if head is None:
            _ = head_cache.pop(session_id, None)
            return None
        head_cache[session_id] = head
        return head[1]

    models_status = model_options(ctx, {"model": SUMMARY_MODEL})

    def session_candidates() -> list[dict[str, object]]:
        catalog = ctx.require(MESSAGE_CATALOG)
        state = read_state or records()
        items: list[dict[str, object]] = []
        seen: set[str] = set()
        for entry in catalog.sessions(visibility="listed", limit=200).items:
            seen.add(entry.session_id)
            channel = entry.session_id.split(":", 1)[0] if ":" in entry.session_id else "akashic"
            title = (entry.title or "").strip()
            if not title and entry.first_message is not None and isinstance(entry.first_message.body, Input):
                title = next(
                    (
                        part.value.strip().splitlines()[0].strip()
                        for part in entry.first_message.body.parts
                        if part.kind == "text" and isinstance(part.value, str) and part.value.strip()
                    ),
                    "",
                )
            title = title[:64] if title else entry.session_id
            project = entry.attributes.dimension("project")
            latest = current_head(state, entry.session_id)
            items.append({
                "session_id": entry.session_id,
                "title": title,
                "channel": channel,
                "project": "" if project == "default" else project,
                "message_count": entry.message_count,
                "updated_at": entry.updated_at.isoformat(),
                "compaction_count": 0 if latest is None else latest.generation,
                "last_trigger": None if latest is None else getattr(latest, "trigger", None),
                "last_tokens_before": None if latest is None else getattr(latest, "tokens_before", None),
                "last_tokens_after": None if latest is None else getattr(latest, "tokens_after", None),
            })
        for sid in config.idle_sessions:
            if sid in seen:
                continue
            latest = current_head(state, sid)
            items.append({
                "session_id": sid,
                "title": sid,
                "channel": sid.split(":", 1)[0] if ":" in sid else "akashic",
                "project": "",
                "message_count": 0,
                "updated_at": "",
                "compaction_count": 0 if latest is None else latest.generation,
                "last_trigger": None if latest is None else getattr(latest, "trigger", None),
                "last_tokens_before": None if latest is None else getattr(latest, "tokens_before", None),
                "last_tokens_after": None if latest is None else getattr(latest, "tokens_after", None),
            })
        items.sort(
            key=lambda item: (
                cast(int, item["compaction_count"]) > 0 or cast(int, item["message_count"]) >= 100,
                cast(int, item["message_count"]),
                str(item["updated_at"]),
            ),
            reverse=True,
        )
        return items

    async def status() -> Mapping[str, object]:
        return {
            **(await models_status()),
            "sessions": await run_file_io(session_candidates),
        }

    await mount_settings(ctx, SETTINGS, ModelSettings(ctx, Config, SETTING_FIELDS, status=status))

    async def start(_event: object) -> None:
        nonlocal read_current, read_state
        state = records()
        read_state = state
        read_current = partial(current_head, state)

    async def stop(_event: object) -> None:
        nonlocal read_current, read_state
        read_state = None
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

    async def prepare(snapshot: tuple[Message, ...], source: str) -> MaterialData:
        if not snapshot:
            return {}
        session_id = snapshot[0].session_id
        state = records()
        # 命中与未命中都要读 head 行版本（SQL），恢复在文件线程完成；
        # 只有纯内存命中才值得省 executor 往返，这里不是（评审 #1142）。
        record = await run_file_io(lambda: current_head(state, session_id))
        if record is None:
            return {}
        return {"summary": material(record)}

    # 同一代摘要在同一账本前缀版本下的连续覆盖不变；版本前进（前缀被改写）后重新核对。
    windows: dict[str, HistoryWindow] = {}

    async def history_start(session_id: str) -> HistoryWindow | None:
        state = records()
        catalog = ctx.require(MESSAGE_CATALOG)
        record = await run_file_io(lambda: current_head(state, session_id))
        if record is None:
            return None
        window = windows.get(record.reference)
        if window is None or window.revision is None or window.revision != await run_file_io(catalog.prefix_revision):
            reader = catalog.reader(session_id)
            covered = record.source_message_ids
            window = await run_file_io(lambda: reader.covered_window(covered))
            if len(windows) >= 64:
                windows.clear()
            windows[record.reference] = window
        return window

    async def compact(snapshot: tuple[Message, ...], materials: MaterialData, request: ModelRequest,
                      model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
                      hard_limit: bool, on_status: ReductionStatus | None = None,
                      summary_limit: int | None = None) -> MaterialData | None:
        """按已结算工具批次选未覆盖消息，再原子发布一份真实摘要。"""
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
        if window is None or not snapshot or not summary_due(
            before, window, force=force, snapshot=snapshot,
            idle_minutes=_idle_minutes_for(config, snapshot), idle_tokens=config.idle_trigger_tokens,
        ):
            return None
        soft_due = force or before >= int(window * SUMMARY_SOFT_LIMIT)
        state = records()
        parent = await run_file_io(lambda: state.head(snapshot[0].session_id))
        current_summary = materials.get("summary")
        if current_summary is not None and not isinstance(current_summary, Mapping):
            raise TypeError("Context 材料摘要必须是对象")
        if (None if parent is None else material(parent)) != current_summary:
            raise ValueError("本次已取得摘要与当前 Session head 不一致")
        turns = ctx.require(TURN_PROJECTION)
        parent_ids = frozenset() if parent is None else frozenset(parent.source_message_ids)
        if parent is not None:
            _ = context.summary_indexes(snapshot, parent.source_message_ids)
        # 未覆盖的旧空洞可能刚变为可见；与新尾部一起候选，不按上代末尾跳过它们。
        pending = tuple(message for message in (snapshot if summary_limit is None else snapshot[:summary_limit])
                        if message.message_id not in parent_ids)
        groups = closed_groups(pending, turns, settled_prefixes=context.settled_prefixes)
        if not soft_due and not groups:
            return None
        # 没有新的已结算批次时，0 表示只重写父摘要本身。
        sizes = range(len(groups), 0, -1) if groups else range(0 if parent is not None else -1, -1, -1)
        selected: int | None = None
        fallback: int | None = None
        deepest: int | None = None
        retained_tokens = deepest_tokens = 0
        planned_tokens: dict[int, int] = {}
        # 估算放得下却被 provider 拒绝时估算不可信：按它保留原文会把超窗内容一直留在尾部，只能选最深切点。
        rejected = hard_limit and before + output_reserve(request.max_output_tokens, window) <= window
        # 原文保留量来自实际 Model 投影，包含尚未闭合的尾部与当前输入。
        for size in sizes:
            added = tuple(message for group in groups[:size] for message in group)
            covered = parent_ids | frozenset(message.message_id for message in added)
            tail = projection.render(
                snapshot,
                after_seq=-1, covered_message_ids=covered,
                fresh=True,
                current_reminder=reminder if reminder_input_id is not None else None,
                current_reminder_input_id=reminder_input_id,
            )
            # 先用占位摘要核对完整输入；真实摘要生成后还会再次核对硬容量。
            planned = {**materials, "summary": {
                "reference": "compaction-plan", "content": "待生成摘要",
                "source_message_ids": tuple(message.message_id for message in snapshot if message.message_id in covered),
            }}
            candidate, error = context.build_attempt(
                snapshot, materials=planned, model=projection, tools=request.tools,
                max_output_tokens=request.max_output_tokens,
                current_reminder_input_id=reminder_input_id,
            )
            if error is not None:
                continue
            planned_tokens[size] = projection.estimate(candidate)
            if deepest is None:
                deepest, deepest_tokens = size, projection.estimate(tail)
                # 组边界已结算，工具配对完整；当前 Input 由 keep_input_ids 保留原文。
                if rejected:
                    break
            if planned_tokens[size] > int(window * SUMMARY_SOFT_LIMIT):
                continue
            if fallback is None:
                fallback, retained_tokens = size, projection.estimate(tail)
            if projection.estimate(tail) >= config.keep_recent_tokens:
                selected = size
                break
        if selected is None:
            if not soft_due:
                if fallback is None or fallback == 0:
                    return None
                selected, retained_tokens = fallback, retained_tokens
            else:
                # 没有切点能回到软水位时仍摘要全部已结算内容；是否放得下交给下面的硬容量核对。
                selected, retained_tokens = (fallback, retained_tokens) if fallback is not None else (deepest, deepest_tokens)
                if selected and on_status is not None and not (rejected and retained_tokens >= config.keep_recent_tokens):
                    await on_status(f"上下文空间不足，这一轮只保留了最近约 {retained_tokens:,} tokens 的原文"
                                    f"（目标 {config.keep_recent_tokens:,}），更早的内容改用摘要。"
                                    "当前输入和进行中的工具调用完整保留。", True)
        if selected is None:
            if before + output_reserve(request.max_output_tokens, window) <= window:
                return None
            raise SummaryCapacityError("当前输入与进行中的工具调用本身超出容量，没有可摘要的内容。"
                                       "请放弃本轮后发送较短内容；对话可用 /abandon，程序可用 programmatic/message/abandon")
        if not soft_due:
            # 花钱前核对收益：planned 用占位摘要，新摘要至少按父摘要原长算。
            kept = 0 if parent is None else model.estimate_context_tokens(({"role": "user", "content": parent.content},))
            if before - planned_tokens[selected] - kept < before * IDLE_MIN_SAVING:
                return None
        inputs = summary_groups(groups[:selected], snapshot)
        if selected and not inputs:
            raise SummaryError("可选范围没有可用于摘要的资料")
        # 2. 摘要必须放进剩余空间：模型生成 → 模型缩写 → 按比例截断各节正文，取第一个放得下的。
        reserve = output_reserve(request.max_output_tokens, window)
        room = window - reserve - planned_tokens[selected]
        added = tuple(message for group in groups[:selected] for message in group)
        covered = parent_ids | frozenset(message.message_id for message in added)

        # 校准估算可能被封顶，收益只比未校准估算；放得下仍按校准估算判断。
        raw_before = model.estimate_context_tokens(request.messages, request.tools)

        def measure(text: str, calls: tuple[str, ...], summarized: tuple[tuple[Message, ...], ...],
                    ) -> tuple[SummaryRecord, MaterialData, int, bool]:
            summary_ids = tuple(message.message_id for group in summarized for message in group)
            kept = set(summary_ids)
            omitted = sum(message.message_id not in kept for message in added)
            if omitted:
                text = f"{text}\n\n## 未读取的更早消息：{omitted} 条（摘要请求放不下，这里没有它们的内容）"
            record = SummaryRecord(
                reference=uuid4().hex, session_id=snapshot[0].session_id,
                generation=1 if parent is None else parent.generation + 1,
                parent=None if parent is None else parent.reference,
                source_message_ids=tuple(message.message_id for message in snapshot if message.message_id in covered),
                summary_message_ids=summary_ids,
                omitted_message_ids=tuple(message.message_id for message in added if message.message_id not in kept),
                content=text, model_call_ids=calls,
                trigger="context_overflow" if force else ("soft_limit" if soft_due else "idle"),
                context_window=window, max_output_tokens=request.max_output_tokens,
                keep_recent_tokens=config.keep_recent_tokens, tokens_before=before, tokens_after=0,
            )
            summary = material(record)
            after_request, _ = context.build_attempt(
                snapshot, materials={**materials, "summary": summary}, model=projection,
                tools=request.tools, max_output_tokens=request.max_output_tokens,
                current_reminder_input_id=reminder_input_id,
            )
            smaller = model.estimate_context_tokens(after_request.messages, after_request.tools) < raw_before
            return record, summary, projection.estimate(after_request), smaller

        def fits(after: int, smaller: bool) -> bool:
            return smaller and after + reserve <= window

        async def report_retry(value: Mapping[str, str]) -> None:
            text = value.get("retry_status")
            if text and on_status is not None:
                await on_status(text, False)

        # 阻断回合和空闲路径各有总期限；软水位摘要迟早要做，只受模型自身的生成期限约束。
        seconds = SUMMARY_DEADLINE_SECONDS if hard_limit else None if soft_due else IDLE_DEADLINE_SECONDS
        deadline = None if seconds is None else asyncio.get_running_loop().time() + seconds

        async def generate(groups: tuple[tuple[Message, ...], ...], previous: str, limit: int,
                           call: SummaryCall) -> tuple[str, tuple[str, ...], tuple[tuple[Message, ...], ...]]:
            # 嵌套 execution 复用调用者已经固定的角色，不重读模型配置；
            # 只包住模型调用，摘要 binding 必须回到 compaction 自己的调用方；期限也不覆盖发布事务。
            try:
                async with asyncio.timeout_at(deadline), ctx.require(CHAT_MODELS).execution() as execution:
                    async def report_fallback(text: str) -> None:
                        if on_status is not None:
                            await on_status(text, True)

                    chosen, fallback = await summary_models(execution, model, config.model, report_fallback)
                    return await summarize(groups, previous=previous, limit=limit, model=chosen,
                                           fallback=fallback, on_fallback=report_fallback,
                                           call=call, on_delta=report_retry)
            except TimeoutError as failure:
                if ModelError.matches(failure):
                    raise
                raise ModelTimeoutError("摘要调用超时" if seconds is None else f"摘要超过 {seconds:g} 秒总期限").exception() from failure

        limit = max(1, min(room, OUTPUT_RESERVE_TOKENS))
        # 只有阻断回合的摘要才值得等重试；可选压缩和下面的缩写失败都有退路，不挡在回合前退避。
        text, calls, summarized = await generate(inputs, "" if parent is None else parent.content, limit,
                                                 BLOCKING_SUMMARY if hard_limit else OPTIONAL_SUMMARY)
        record, summary, after, smaller = measure(text, calls, summarized)
        if not fits(after, smaller):
            try:
                shorter, more, _ = await generate((), text, max(1, limit // 2), OPTIONAL_SUMMARY)
            except (RuntimeError, TimeoutError, SummaryError) as failure:
                if not (ModelError.matches(failure) or isinstance(failure, SummaryError)):
                    raise
                logger.warning("摘要缩写失败，改为按比例截断已有摘要: %s", failure)
            else:
                text, calls = shorter, calls + more
                record, summary, after, smaller = measure(text, calls, summarized)
        # 已有真实摘要时本地按比例截断各节正文；最短版本仍放不下才是容量边界。
        for ratio in (0.5, 0.25, 0.0):
            if fits(after, smaller):
                break
            record, summary, after, smaller = measure(clip_summary(text, ratio), calls, summarized)
        if not smaller:
            raise SummaryCapacityError("摘要后请求没有变小：已结算的历史不是主要占用，当前输入或进行中的工具结果本身过大。"
                                       "摘要模型工作正常。请发送较短内容继续，过大的内容结算后会进入摘要；"
                                       "对话可用 /abandon，程序可用 programmatic/message/abandon")
        if after + reserve > window:
            raise SummaryCapacityError(f"摘要缩至最短后输入仍约 {after:,} tokens，当前输入与进行中的工具调用本身超出容量。"
                                       "请放弃本轮后发送较短内容；对话可用 /abandon，程序可用 programmatic/message/abandon")
        record = record.model_copy(update={"tokens_after": after})
        # 3. binding 可以先固定，但读者只有在摘要事务成功后才取得此引用。
        reader = ctx.require(MESSAGE_CATALOG).reader(record.session_id)
        _ = await state.publish(
            record, reader, parent=parent, summary_indexes=context.summary_indexes,
        )
        if record.omitted_message_ids and on_status is not None:
            await on_status(f"摘要请求放不下全部历史：{len(record.omitted_message_ids)} 条更早的消息没有进入摘要，"
                            "模型之后看不到它们。", True)
        return summary

    # 空闲压缩是可选优化：任何异常都转成 notice 后继续回合，只有取消继续向外传。
    async def reduce_idle(snapshot: tuple[Message, ...], materials: MaterialData, request: ModelRequest,
                          model: BoundChatModel, projection: ContextModel, *, source: str,
                          on_status: ReductionStatus | None,
                          summary_limit: int | None) -> MaterialData | None:
        attempts = idle_attempts_for(snapshot[0].session_id, model.descriptor.binding_id)
        attempts.gap = idle_gap_anchor(snapshot)
        try:
            if on_status is not None:
                await on_status(f"正在进行空闲预压缩：模型 {model.descriptor.model_id}；"
                                f"输入估算 {projection.estimate(request):,} tokens", False)
            summary = await compact(snapshot, materials, request, model, projection, source=source, force=False,
                                    hard_limit=False, on_status=on_status, summary_limit=summary_limit)
        except Exception as error:
            attempts.failures += 1
            logger.warning("空闲预压缩失败 session=%s failures=%d: %r", snapshot[0].session_id, attempts.failures, error,
                           exc_info=not isinstance(error, (RuntimeError, TimeoutError, SummaryError)))
            paused = (f"已连续失败 {MAX_SUMMARY_FAILURES} 次，这个对话暂停空闲预压缩；压缩成功、更换模型或保存模型设置后恢复。"
                      if attempts.failures >= MAX_SUMMARY_FAILURES else "")
            if on_status is not None:
                await on_status(f"空闲预压缩未完成：{type(error).__name__}: {error}。这是可选优化，本轮继续生成。" + paused, True)
            return None
        if summary is not None:
            attempts.failures = 0
        return summary

    async def reduce(snapshot: tuple[Message, ...], materials: MaterialData, request: ModelRequest,
                     model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
                     required: bool = True,
                     on_status: ReductionStatus | None = None,
                     summary_limit: int | None = None) -> MaterialData | None:
        """软水位失败可继续；硬容量或 provider 明确拒绝必须保留失败原因。"""
        window = projection.context_window
        before = projection.estimate(request)
        if window is None or not snapshot:
            return None
        # 软水位以下只剩空闲路径：硬水位一定高于软水位，走到这里的请求原样也能发。
        if not summary_due(before, window, force=force):
            if not idle_due(snapshot, before, model):
                return None
            return await reduce_idle(snapshot, materials, request, model, projection, source=source,
                                     on_status=on_status, summary_limit=summary_limit)
        hard_limit = (force and required) or before + output_reserve(request.max_output_tokens, window) > window
        # An optional shrink must not rewrite a request already within the raw-history target.
        if (force or before < int(window * SUMMARY_SOFT_LIMIT)) and not hard_limit and before <= config.keep_recent_tokens:
            return None
        state = failure_state(snapshot, source, model)
        budget = (f"模型 {model.descriptor.model_id}；窗口 {window:,}；输入估算 {before:,}；"
                  f"输出预留 {output_reserve(request.max_output_tokens, window):,}；原文目标 {config.keep_recent_tokens:,} tokens")
        paused = f"本轮上下文压缩连续失败 {MAX_SUMMARY_FAILURES} 次，已暂停自动压缩；新输入或更换模型后重新尝试。"
        if state.count >= MAX_SUMMARY_FAILURES:
            if hard_limit:
                raise SummaryError(state.detail + budget + "。" + paused + "本次请求被阻断，未继续业务生成。" + state.recovery)
            return None
        if on_status is not None:
            await on_status("正在压缩上下文：" + budget, False)
        notices: list[str] = []
        async def report(text: str, retain: bool) -> None:
            if retain:
                notices.append(text)
            if on_status is not None:
                await on_status(text, retain)

        try:
            summary = await compact(snapshot, materials, request, model, projection,
                                    source=source, force=force, hard_limit=hard_limit, on_status=report,
                                    summary_limit=summary_limit)
        except (RuntimeError, TimeoutError, SummaryError) as error:
            if not (ModelError.matches(error) or isinstance(error, SummaryError)):
                raise
            state.count += 1
            state.detail = f"上下文压缩失败：{type(error).__name__}: {error}。"
            state.recovery = _recovery(error)
            detail = state.detail + budget + "。"
            if hard_limit:
                blocked = "\n".join((*notices, detail + "本次请求被阻断，未继续业务生成。"))
                # 摘要模型自身的配置问题（如 404）要留「切换摘要模型」的出口，不引导放弃当前输入。
                unsafe = is_content_rejection(error) and ModelError.matches(error, ContentSafetyError)
                raise SummaryError(content_rejection_reason(blocked) if unsafe else blocked + state.recovery) from error
            if on_status is not None:
                outcome = ("本次压缩没有缩小请求，保留原文继续生成。" if isinstance(error, SummaryCapacityError)
                           else detail + "原请求仍满足硬容量，本次继续生成。")
                await on_status(outcome
                                + (paused if state.count >= MAX_SUMMARY_FAILURES else ""), True)
            return None
        state.count, state.detail = 0, ""
        # 软水位或强制压缩成功同样证明摘要模型可用。
        if summary is not None and snapshot[0].session_id in idle_attempts:
            idle_attempts[snapshot[0].session_id].failures = 0
        return summary

    _ = await ctx.require(MATERIALS_V4).register(ctx, kind="context", name="compaction", prepare=prepare, reduce=reduce,
                                                 history_start=history_start, reduce_due=idle_due, priority=500)
