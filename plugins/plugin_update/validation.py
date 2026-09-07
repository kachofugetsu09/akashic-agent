from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import CHAT_MODELS, Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS, Task
from plugins.content.plugin import CONTENT, check_text
from plugins.context.materials import MATERIALS
from plugins.context.plugin import CONTEXT
from plugins.conversation.program import run_reply
from plugins.models.projection import MODEL_CALLS
from plugins.react.plugin import REACT
from plugins.tools.api import Denied
from plugins.tools.menu import check_menu
from plugins.tools.plugin import TOOLS
from plugins.turn_projection.plugin import TURN_PROJECTION
from session.log import SessionAttributes
from session.message import ContentPart, Input, Message, Output

from .tool import InstallInput


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    passed: bool
    reason: str = Field(min_length=1)


class Validation:
    """普通回复程序解释测试要求和结论；Core 不识别这些策略。"""

    def __init__(self, ctx: Context, *, max_steps: int, max_output_tokens: int):
        self._ctx = ctx
        self._max_steps = max_steps
        self._max_output_tokens = max_output_tokens

    async def run(self, identity: str, request: InstallInput) -> Verdict:
        """内部 Session 只写验证消息；实际工具和材料来自该候选 Root。"""
        ctx = self._ctx
        # 1. 输入、菜单与材料选择属于本次隔离程序。
        session_id = "plugin-validation:" + identity
        _ = ctx.require(SESSION_ADMISSION).ensure(ctx, session_id, SessionAttributes("internal", "excluded"))
        reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
        writer = ctx.require(MESSAGE_WRITERS).bind(ctx, author="plugin_update", source="plugin_update",
            body_types=(Input,), content={"text": check_text})(session_id)
        names = (tuple(sorted(cast(str, item["name"]) for item in ctx.require(TOOLS).descriptions()))
                 if request.validation_tools is None else tuple(request.validation_tools))
        check_menu(ctx.require(TOOLS), names)
        _ = writer.append(identity + ":input", Input((ContentPart("text", request.validation_prompt),)))

        async def authorize(binding: str, arguments: Mapping[str, object]) -> Mapping[str, object]:
            tool = cast(Mapping[str, object], ctx.require(BINDINGS).describe(binding, TOOLS)["tool"])
            if tool["name"] not in names:
                raise Denied("当前验证未授予此工具")
            return {"source": "plugin_update", "session_id": session_id}

        async def program(task: Task) -> Message:
            task.on_close(writer.expire)
            return await run_reply(ctx, task, reader, "plugin_update", models=ctx.require(CHAT_MODELS),
                content=ctx.require(CONTENT), context=ctx.require(CONTEXT), tools=ctx.require(TOOLS),
                react=ctx.require(REACT), materials=ctx.require(MATERIALS),
                turn_projection=ctx.require(TURN_PROJECTION), read_call=ctx.require(MODEL_CALLS),
                authorize=authorize, tool_names=names, max_output_tokens=self._max_output_tokens,
                max_steps=self._max_steps, exclude_materials=frozenset(request.excluded_materials),
                prompt_hints=(
                    '你正在验证插件候选。依据实际检查结果作结论。最终只返回 JSON：'
                    '{"passed": true 或 false, "reason": "实际证据和原因"}。',
                ))
        try:
            task = await ctx.require(TASKS).open(ctx).admit(session_id, lambda slot: slot.start(program))
            output = cast(Message, await task.join())
        finally:
            writer.expire()
        # 2. 只解释实际已提交的最终回答；缺失或坏 JSON 不能算验证通过。
        if not isinstance(output.body, Output) or output.body.finish != "complete":
            raise ValueError("验证没有产生完整结论")
        text = "\n".join(cast(str, part.value) for part in output.body.parts
                         if isinstance(part, ContentPart) and part.kind == "text")
        return Verdict.model_validate(json.loads(text))


PLUGIN_VALIDATION = ServiceKey[Validation]("plugin_update.validation")
