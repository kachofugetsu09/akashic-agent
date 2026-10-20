from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal, cast
from contextlib import AbstractAsyncContextManager, AbstractContextManager

from core.common.diagnostic_log import log_timing
from core.common.file_io import run_file_io
from functools import partial
from agent.plugin_composition import Context
from plugins.ledger.contract import ArtifactRead
from plugins.ledger.contract import MessageReader, MessageWriters, OwnerState, OwnerTransaction
from plugins.models.contract import ModelRequest
from plugins.models.contract import (
    BoundChatModel,
    ChatModels,
)
from agent.plugin_composition.tasks import Task
from plugins.context.contract import MaterialView, ReductionStatus
from plugins.models.contract import PrepareContent
from plugins.ledger.contract import ContentPart, Control, Input, Message, Output

from plugins.sources.contract import (
    SourceGuard,
)

from plugins.content.contract import Content
from plugins.context.contract import ContextBuilder, MaterialData, SummaryData
from plugins.models.contract import (
    CallReader, ContentRenderer, ContextModel, ModelChecks, ModelContent,
    ModelProjections, ModelSelection, StreamCallback,
)
from plugins.tools.contract import (
    OrderedToolProgram, ToolCatalog, ToolCleanup, ToolPresentation, ToolView,
)
from plugins.turn_projection.contract import TurnProjection


async def run_reply(
    ctx: Context, task: Task, reader: MessageReader, source: str, *,
    models: ChatModels, content: Content, context: ContextBuilder, tools: ToolCatalog,
    cleanup: ToolCleanup,
    check_admission: SourceGuard,
    selection: ModelSelection, tool_program: OrderedToolProgram,
    model_checks: ModelChecks, model_content: ModelContent, model_projection: ModelProjections,
    writers: MessageWriters, owner_state: OwnerState, artifact_reader: ArtifactRead,
    react: Callable[..., Awaitable[Message]],
    materials: AbstractAsyncContextManager[MaterialView],
    turn_projection: TurnProjection,
    render_content: ContentRenderer | None = None,
    prepare_content: PrepareContent | None = None,
    dynamic_content_kinds: frozenset[str] | None = frozenset(),
    read_call: CallReader,
    authorize: Callable[[str, Mapping[str, object]], Awaitable[Mapping[str, object] | str]],
    max_output_tokens: int | None,
    max_steps: int,
    max_parallel_calls: int = 4,
    tool_view: ToolView | None = None,
    tool_names: Sequence[str] | None = None,
    prompt_hints: Sequence[str] = (),
    fixed_bindings: Mapping[str, str] | None = None,
    preview: Callable[[str], AbstractContextManager[StreamCallback]] | None = None,
    reminders: Sequence[Mapping[str, object]] = (),
    terminal_tools: frozenset[str] = frozenset(),
    presentation: ToolPresentation | None = None,
    history: Literal["session", "own"] = "session",
    before_step: Callable[[], Awaitable[None]] | None = None,
) -> Message:
    """普通组合拥有本次程序资源，Source 不必同步签发模型或内容 writer。"""
    mark = partial(log_timing, session_id=reader.session_id, source=source)
    mark("reply.begin")
    # 请求只看可见历史：其他来源未完成的轮次不进入请求；own 时只看本来源，不带会话摘要也不压缩整个会话。
    context = RequestHistory(context, source, own=history == "own")
    if tool_names is not None:
        if tool_view is not None or fixed_bindings is None:
            raise ValueError("旧工具名称只可核对原固定 binding")
        if len(set(tool_names)) != len(tool_names) or set(tool_names) != set(
            fixed_bindings
        ):
            raise ValueError("旧工具名称与原固定 binding 不一致")
    # 1. 内容检查器与模型绑定覆盖整个程序，取消时先排空已开始的工具。
    prompt_hints = tuple(prompt_hints)
    reader = reader.incremental()
    with reader.read_snapshot():
        check_admission()
        source_head = reader.head(source=source)
        through_seq = reader.head()
        saved_selection = reader.metadata()
    def check(transaction: OwnerTransaction | None = None) -> None:
        check_admission(transaction=transaction)

    def read_open(messages: Sequence[Message]) -> tuple[Message, ...]:
        """同一短快照内只取未闭合 Turn 正文，历史分段只保留引用。"""
        turns = turn_projection.project(messages, source, include_closed=False)
        if not turns or turns[-1].status != "open":
            return ()
        # 开放尾段的成员已经在这份固定视图里，直接复用，不再逐 ID 同步点查。
        index = {message.message_id: message for message in messages}
        members: list[Message] = []
        for identity in turns[-1].message_ids:
            message = index.get(identity)
            if message is None:
                raise RuntimeError("同一快照中的 Turn 消息缺失")
            members.append(message)
        return tuple(members)

    # Turn 投影与 react 共用同一份已提交前缀：history_view 持有共享读面直到
    # react 自己的快照接手，历史只解码一次。投影与成员提取是不受调用方约束的
    # 全量扫描，必须留在可排空 worker 内，避免长历史时占住事件循环。
    history_view = await reader.committed_snapshot_async(through_seq=through_seq)
    opened = await run_file_io(lambda: read_open(history_view))
    chosen = selection.read(opened)
    if chosen is None:
        chosen = selection.read_saved(saved_selection or {})
    from_seq = min((message.seq for message in opened), default=source_head + 1)
    keep_input_ids = tuple(item.message_id for item in opened if isinstance(item.body, Input))
    del opened
    mark("reply.selection.end")
    async with (
        cleanup(reader, source, from_seq, task=task, drain=tools.drain_calls),
        content.bind() as view,
        models.execution(model_id=chosen.model_id, reasoning_effort=chosen.reasoning_effort) as execution,
        materials as material_view,
    ):
        model = execution.chat("agent")
        # 推理与正文共用输出额度；在冻结请求前确定预算，Context 同样预留它。
        if max_output_tokens is None:
            max_output_tokens = model.descriptor.capabilities.max_output_tokens or 32768
        mark("tools.menu.begin")
        menu = await tool_program.create_menu(
            reader, source, content=view.checks,
            check_start=check,
            authorize=authorize, view=tool_view, limit=model.max_tool_schemas,
            fixed_bindings=fixed_bindings, presentation=presentation,
            child_permit=task.child_permit if task.has_external_permit else None,
        )
        mark("tools.menu.end", counts={"tools": len(menu.names)})
        if terminal_tools - menu.names:
            raise ValueError("终结工具必须属于本次允许目录")
        output = writers.bind(
            ctx, author="assistant", source=source, body_types=(Output,),
            check_metadata=view.check_metadata,
            content={**view.checks, "model.facts": model_checks.check_facts, "model.tool_rejection": model_checks.check_tool_rejection, "context.summary": context.check_summary, "context.notice": context.check_notice}, check_call=menu.check_call,
        )(reader.session_id)
        task.on_close(output.expire)
        artifacts: Mapping[str, tuple[Mapping[str, Any], ...]] = {}
        def render(part: ContentPart):
            return model_content.render(part, artifacts=artifacts, read_message=reader.get)
        projection = model_projection.create(
            model, source=source, render_content=render if render_content is None else render_content,
            tool_name=menu.name, read_call=read_call, check_summary=context.check_summary, keep_input_ids=keep_input_ids,
            dynamic_content_kinds=dynamic_content_kinds,
            **({"prepare_content": prepare_content, "tool_names": menu.names} if prepare_content is not None else {}),
        )

        # 2. 内容协议提示与解码来自同一 view；Context 仍只接收已取得的材料。
        async def build_materials(messages: tuple[Message, ...]) -> MaterialData:
            nonlocal artifacts
            mark("materials.begin")
            if history == "own":
                messages = own_messages(messages, source)
            result = await material_view.prepare(
                messages, source, caller=ctx, reminders=tuple(reminders),
            )
            if history == "own":
                result = {key: value for key, value in result.items() if key != "summary"}
            if render_content is None:
                summary = cast(SummaryData | None, result.get("summary"))
                start = 0 if summary is None else context.summary_range(messages, cast(tuple[str, ...], summary["source_message_ids"])).stop
                refs = reader.attachments_for(tuple(
                    message.message_id for index, message in enumerate(messages)
                    if index >= start or message.message_id in keep_input_ids
                ))
                if refs:
                    artifacts = await model_content.load_artifacts(
                        artifact_reader, refs,
                        accepts_images="image" in model.descriptor.capabilities.input_modalities,
                        current_artifact_ids=frozenset(
                            ref.artifact_id for ref in reader.attachments_for(keep_input_ids)
                        ),
                    )
            check()
            mark("materials.end")
            return {**result, "system_prompt": "\n\n".join(
                part for part in (cast(str, result["system_prompt"]), *view.prompts, *prompt_hints, menu.system_prompt) if part
            )}

        async def reduce(
            snapshot: tuple[Message, ...], prepared: MaterialData, request: ModelRequest,
            model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
            on_status: ReductionStatus | None = None, summary_limit: int | None = None,
        ) -> SummaryData | None:
            # 摘要按请求视图估算，当前输入等尾部原文保留；其他来源有未完成轮次时只摘要它之前的前缀，
            # 该前缀在真实日志和请求视图里相同，摘要范围两边都连续。
            hidden = hidden_indexes(snapshot, source)
            if hidden:
                first = min(hidden)
                summary_limit = first if summary_limit is None else min(summary_limit, first)
                snapshot = tuple(visible_messages(snapshot, source))
            result = await material_view.reduce(snapshot, prepared, request, model, projection,
                                                source=source, force=force, on_status=on_status,
                                                summary_limit=summary_limit)
            check()
            return result

        try:
            return await react(
                reader, output, model=model, context=context, projection=projection,
                materials=build_materials, content=view, tools=menu,
                max_output_tokens=max_output_tokens, max_steps=max_steps,
                reduce=None if history == "own" else reduce, preview=preview, terminal_tools=terminal_tools,
                max_parallel_calls=max_parallel_calls,
                state=owner_state.open_scoped(ctx, "generation"),
                check_start=check,
                before_step=before_step,
            )
        finally:
            output.expire()


def own_messages(messages: Sequence[Message], source: str) -> tuple[Message, ...]:
    return tuple(message for message in messages if message.source == source)


# 其他来源最后一次结束之后的消息：那是仍在进行的轮次，工具调用可能尚未配对。
def hidden_indexes(messages: Sequence[Message], source: str) -> set[int]:
    finished: set[str] = set()
    hidden: set[int] = set()
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.source == source or message.source in finished:
            continue
        body = message.body
        if isinstance(body, Control) and body.action in {"failure", "abandon"} or isinstance(body, Output) and body.finish != "continue":
            finished.add(message.source)
        else:
            hidden.add(index)
    return hidden


def visible_messages(messages: Sequence[Message], source: str) -> Sequence[Message]:
    hidden = hidden_indexes(messages, source)
    if not hidden:
        return messages
    return tuple(message for index, message in enumerate(messages) if index not in hidden)


class RequestHistory:
    """只改变交给请求组装的历史；读面、提交、竞争检查与恢复仍使用整个会话的真实日志。"""

    def __init__(self, context: ContextBuilder, source: str, *, own: bool):
        self._context = context
        self._source = source
        self._own = own
        self.check_summary = context.check_summary
        self.check_notice = context.check_notice
        self.summary_range = context.summary_range
        self.settled_prefixes = context.settled_prefixes
        self.reminder_content = context.reminder_content

    def _view(self, snapshot: Sequence[Message]) -> Sequence[Message]:
        return own_messages(snapshot, self._source) if self._own else visible_messages(snapshot, self._source)

    def build(self, snapshot: Sequence[Message], **options: Any) -> ModelRequest:
        return self._context.build(self._view(snapshot), **options)

    def build_attempt(self, snapshot: Sequence[Message], **options: Any) -> tuple[ModelRequest, str | None]:
        return self._context.build_attempt(self._view(snapshot), **options)
