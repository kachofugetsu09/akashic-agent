from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping

from typing import cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.bindings import BINDINGS
from agent.plugin_composition.messages import MESSAGE_CATALOG, MESSAGE_WRITERS, SESSION_ADMISSION
from agent.plugin_composition.tasks import TASKS, Task
from .inputs import CONTENT, ALL_TOOLS, TOOLS

from agent.plugin_composition.messages import SessionAttributes
from agent.plugin_contracts import ContentPart, Input, Message, Output

from .tool import InstallInput


REPLY_EXECUTE = ServiceKey[Callable[..., Awaitable[Message]]]("reply.execute.v1")


class Validation:
    """普通回复程序执行调用者的要求；Core 不解释回答内容。"""

    def __init__(self, ctx: Context, *, max_steps: int, max_output_tokens: int):
        self._ctx = ctx
        self._max_steps = max_steps
        self._max_output_tokens = max_output_tokens

    async def run(self, identity: str, request: InstallInput) -> Message:
        """内部 Session 只写验证消息；实际工具和材料来自该候选 Root。"""
        ctx = self._ctx
        # 1. 输入、菜单与材料选择属于本次隔离程序。
        session_id = "plugin-validation:" + identity
        _ = ctx.require(SESSION_ADMISSION).ensure(ctx, session_id, SessionAttributes("internal", "excluded"))
        reader = ctx.require(MESSAGE_CATALOG).reader(session_id)
        writer = ctx.require(MESSAGE_WRITERS).bind(
            ctx,
            author="plugin_update",
            source="plugin_update",
            body_types=(Input,),
            content={"text": ctx.require(CONTENT).check_text},
        )(session_id)
        available = ctx.require(ALL_TOOLS)()
        view = (
            available
            if request.validation_tools is None
            else ctx.require(TOOLS).view(*(available.select(name) for name in request.validation_tools))
        )
        names = frozenset(ref.name for ref in view.refs)
        _ = writer.append(
            identity + ":input",
            Input((ContentPart("text", request.validation_prompt),)),
        )

        async def authorize(binding: str, arguments: Mapping[str, object]) -> Mapping[str, object] | str:
            tool = cast(Mapping[str, object], ctx.require(BINDINGS).describe(binding, TOOLS)["tool"])
            if tool["name"] not in names:
                return '当前验证未授予此工具'
            return {"source": "plugin_update", "session_id": session_id}

        async def program(task: Task) -> Message:
            task.on_close(writer.expire)
            return await ctx.require(REPLY_EXECUTE)(
                             ctx, task, reader, 'plugin_update',
                             authorize=authorize,
                             tool_view=view,
                             max_output_tokens=self._max_output_tokens,
                             max_steps=self._max_steps,
                             exclude_materials=frozenset(request.excluded_materials),
                             prompt_hints=('你正在隔离的 latest 候选中执行普通程序调用。报告实际过程和结果，不需要批准 JSON。发现问题时明确说明，发起升级的 Agent 可以 revert 撤销本次更新。',),
                         )
        try:
            task = await ctx.require(TASKS).open(ctx).admit(session_id, lambda slot: slot.start(program))
            output = cast(Message, await task.join())
        finally:
            writer.expire()
        # 2. 来源只判断普通程序是否正常结束，不把回答文字当作授权协议。
        if not isinstance(output.body, Output) or output.body.finish != "complete":
            raise ValueError("latest 调用没有正常完成")
        return output


PLUGIN_VALIDATION = ServiceKey[Validation]("plugin_update.validation")
