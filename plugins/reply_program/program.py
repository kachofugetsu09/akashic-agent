from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast

from agent.plugin_composition import Context
from agent.plugin_composition.artifacts import ARTIFACT_READ
from agent.plugin_composition.messages import MESSAGE_WRITERS, MessageReader
from agent.plugin_composition.models import BoundChatModel, ChatModels, ModelRequest
from agent.plugin_composition.tasks import Task
from agent.plugin_contracts import ContentPart, Input, Message, Output

from .inputs import (
    Authorize, CallReader, Content, ContentRenderer, ContextBuilder, ContextMaterials,
    ContextModel, Materials, MODEL_CHECKS, MODEL_CONTENT, MODEL_PROJECTION, MODEL_SELECTION,
    SOURCE_CHECK, Preview, Reminder, Summary, ToolCatalog, ToolCleanup, TOOL_PROGRAM, ToolPresentation,
    ToolView, TurnProjection,
)




async def run_reply(
    ctx: Context, task: Task, reader: MessageReader, source: str, *,
    models: ChatModels, content: Content, context: ContextBuilder, tools: ToolCatalog,
    cleanup: ToolCleanup,
    react: Callable[..., Awaitable[Message]],
    materials: ContextMaterials,
    turn_projection: TurnProjection,
    render_content: ContentRenderer | None = None,
    read_call: CallReader,
    authorize: Authorize,
    max_output_tokens: int,
    max_steps: int,
    tool_view: ToolView | None = None,
    tool_names: Sequence[str] | None = None,
    exclude_materials: frozenset[str] = frozenset(),
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
    check_source = ctx.require(SOURCE_CHECK)
    prompt_hints = tuple(prompt_hints)
    reader = reader.incremental()
    source_head = reader.head(source=source)
    snapshot = reader.snapshot()
    turns = turn_projection.project(snapshot, source)
    open_ids: set[str] = set(turns[-1].message_ids) if turns and turns[-1].status == "open" else set()
    chosen = ctx.require(MODEL_SELECTION).read(tuple(message for message in snapshot if message.message_id in open_ids))
    if chosen is None:
        chosen = ctx.require(MODEL_SELECTION).read_saved(reader.metadata() or {})
    from_seq = min((message.seq for message in snapshot if message.message_id in open_ids), default=source_head + 1)
    async with (
        cleanup(ctx, reader, source, from_seq, task=task, drain=tools.drain_calls),
        content.bind() as view,
        models.execution(model_id=chosen.model_id, reasoning_effort=chosen.reasoning_effort) as execution,
        materials.bind(exclude=exclude_materials) as material_view,
    ):
        model = execution.chat("agent")
        writers = ctx.require(MESSAGE_WRITERS)
        keep_input_ids = tuple(
            item.message_id for item in snapshot
            if item.message_id in open_ids and isinstance(item.body, Input)
        )
        menu = ctx.require(TOOL_PROGRAM).create_menu(
            reader, source, content=view.checks,
            check_start=lambda: check_source(task, reader, source, source_head),
            authorize=authorize, view=tool_view, limit=model.max_tool_schemas,
            fixed_bindings=fixed_bindings, presentation=presentation,
            child_permit=task.child_permit if task.has_external_permit else None,
        )
        if terminal_tools - menu.names:
            raise ValueError("终结工具必须属于本次允许目录")
        output = writers.bind(
            ctx, author="assistant", source=source, body_types=(Output,),
            check_metadata=view.check_metadata,
            content={**view.checks, "model.facts": ctx.require(MODEL_CHECKS).check_facts, "model.tool_rejection": ctx.require(MODEL_CHECKS).check_tool_rejection, "context.summary": context.check_summary}, check_call=menu.check_call,
        )(reader.session_id)
        task.on_close(output.expire)
        artifacts: Mapping[str, tuple[Mapping[str, Any], ...]] = {}
        def render(part: ContentPart):
            return ctx.require(MODEL_CONTENT).render(part, artifacts=artifacts, read_message=reader.get)
        projection = ctx.require(MODEL_PROJECTION).create(
            model, source=source, render_content=render if render_content is None else render_content,
            tool_name=menu.name, read_call=read_call, check_summary=context.check_summary, keep_input_ids=keep_input_ids,
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
                    artifacts = await ctx.require(MODEL_CONTENT).load_artifacts(
                        ctx.require(ARTIFACT_READ), refs,
                        accepts_images="image" in model.descriptor.capabilities.input_modalities,
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
            )
        finally:
            output.expire()
