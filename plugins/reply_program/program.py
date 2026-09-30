from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, cast
from contextlib import AbstractAsyncContextManager

from core.common.file_io import run_file_io
from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ArtifactRead
from agent.plugin_composition.messages import MessageReader, MessageWriters, OwnerState
from agent.plugin_composition.models import BoundChatModel, ChatModels, ModelRequest
from agent.plugin_composition.tasks import Task
from agent.plugin_contracts.context import MaterialView
from agent.plugin_contracts.models import PrepareContent
from agent.plugin_contracts import ContentPart, Input, Message, Output

from agent.plugin_contracts.sources import SourceCheck

from .inputs import (
    Authorize,
    CallReader,
    Content,
    ContentRenderer,
    ContextBuilder,
    ContextModel,
    Materials,
    ModelChecks,
    ModelContent,
    ModelProjections,
    ModelSelection,
    Preview,
    Reminder,
    Summary,
    ToolCatalog,
    ToolCleanup,
    ToolPresentation,
    ToolProgram,
    ToolView,
    TurnProjection,
)


async def run_reply(
    ctx: Context, task: Task, reader: MessageReader, source: str, *,
    models: ChatModels, content: Content, context: ContextBuilder, tools: ToolCatalog,
    cleanup: ToolCleanup,
    check_source: SourceCheck,
    selection: ModelSelection, tool_program: ToolProgram,
    model_checks: ModelChecks, model_content: ModelContent, model_projection: ModelProjections,
    writers: MessageWriters, owner_state: OwnerState, artifact_reader: ArtifactRead,
    react: Callable[..., Awaitable[Message]],
    materials: AbstractAsyncContextManager[MaterialView],
    turn_projection: TurnProjection,
    render_content: ContentRenderer | None = None,
    prepare_content: PrepareContent | None = None,
    read_call: CallReader,
    authorize: Authorize,
    max_output_tokens: int,
    max_steps: int,
    max_parallel_calls: int = 4,
    tool_view: ToolView | None = None,
    tool_names: Sequence[str] | None = None,
    prompt_hints: Sequence[str] = (),
    fixed_bindings: Mapping[str, str] | None = None,
    preview: Preview | None = None,
    reminders: Sequence[Reminder] = (),
    terminal_tools: frozenset[str] = frozenset(),
    presentation: ToolPresentation | None = None,
) -> Message:
    """普通组合拥有本次程序资源，Source 不必同步签发模型或内容 writer。"""
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
        check_source(task, reader, source, task.boundary_hint)
        source_head = reader.head(source=source)
        through_seq = reader.head()
        saved_selection = reader.metadata()
    def read_open(messages: Iterable[Message]) -> tuple[Message, ...]:
        """同一短快照内只取未闭合 Turn 正文，历史分段只保留引用。"""
        turns = turn_projection.project(messages, source, include_closed=False)
        if not turns or turns[-1].status != "open":
            return ()
        members: list[Message] = []
        for identity in turns[-1].message_ids:
            message = reader.get(identity)
            if message is None:
                raise RuntimeError("同一快照中的 Turn 消息缺失")
            members.append(message)
        return tuple(members)

    opened = await run_file_io(lambda: reader.scan(read_open, through_seq=through_seq, source=source))
    chosen = selection.read(opened)
    if chosen is None:
        chosen = selection.read_saved(saved_selection or {})
    from_seq = min((message.seq for message in opened), default=source_head + 1)
    keep_input_ids = tuple(item.message_id for item in opened if isinstance(item.body, Input))
    del opened
    async with (
        cleanup(reader, source, from_seq, task=task, drain=tools.drain_calls),
        content.bind() as view,
        models.execution(model_id=chosen.model_id, reasoning_effort=chosen.reasoning_effort) as execution,
        materials as material_view,
    ):
        model = execution.chat("agent")
        menu = await tool_program.create_menu(
            reader, source, content=view.checks,
            check_start=lambda transaction: check_source(task, reader, source, source_head, transaction=transaction),
            authorize=authorize, view=tool_view, limit=model.max_tool_schemas,
            fixed_bindings=fixed_bindings, presentation=presentation,
            child_permit=task.child_permit if task.has_external_permit else None,
        )
        if terminal_tools - menu.names:
            raise ValueError("终结工具必须属于本次允许目录")
        output = writers.bind(
            ctx, author="assistant", source=source, body_types=(Output,),
            check_metadata=view.check_metadata,
            content={**view.checks, "model.facts": model_checks.check_facts, "model.tool_rejection": model_checks.check_tool_rejection, "context.summary": context.check_summary}, check_call=menu.check_call,
        )(reader.session_id)
        task.on_close(output.expire)
        artifacts: Mapping[str, tuple[Mapping[str, Any], ...]] = {}
        def render(part: ContentPart):
            return model_content.render(part, artifacts=artifacts, read_message=reader.get)
        projection = model_projection.create(
            model, source=source, render_content=render if render_content is None else render_content,
            tool_name=menu.name, read_call=read_call, check_summary=context.check_summary, keep_input_ids=keep_input_ids,
            **({"prepare_content": prepare_content, "tool_names": menu.names} if prepare_content is not None else {}),
        )

        # 2. 内容协议提示与解码来自同一 view；Context 仍只接收已取得的材料。
        async def build_materials(messages: tuple[Message, ...]) -> Materials:
            nonlocal artifacts
            result = await material_view.prepare(
                messages, source, caller=ctx, reminders=tuple(reminders),
            )
            if render_content is None:
                summary = cast(Summary | None, result.get("summary"))
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
            check_source(task, reader, source, source_head)
            return {**result, "system_prompt": "\n\n".join(
                part for part in (cast(str, result["system_prompt"]), *view.prompts, *prompt_hints, menu.system_prompt) if part
            )}

        async def reduce(
            snapshot: tuple[Message, ...], prepared: Materials, request: ModelRequest,
            model: BoundChatModel, projection: ContextModel, *, source: str, force: bool,
        ) -> Summary | None:
            result = await material_view.reduce(snapshot, prepared, request, model, projection,
                                                source=source, force=force)
            check_source(task, reader, source, source_head)
            return result

        try:
            return await react(
                reader, output, model=model, context=context, projection=projection,
                materials=build_materials, content=view, tools=menu,
                max_output_tokens=max_output_tokens, max_steps=max_steps,
                reduce=reduce, preview=preview, terminal_tools=terminal_tools,
                max_parallel_calls=max_parallel_calls,
                state=owner_state.open_scoped(ctx, "generation"),
                check_start=lambda transaction: check_source(task, reader, source, source_head, transaction=transaction),
            )
        finally:
            output.expire()
