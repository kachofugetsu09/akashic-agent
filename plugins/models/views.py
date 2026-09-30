"""普通内容投影注册表；只固定贡献者和纯函数的生命周期。"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from agent.plugin_composition import Context, Effect
from agent.plugin_composition.model import FiberState
from agent.plugin_contracts import Message
from agent.plugin_contracts.models import (
    CONTENT_VIEWS as CONTENT_VIEWS,
    ContentTransform,
    PrepareContent,
    RenderedContent,
)


class ContentViews:
    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._sources: dict[tuple[str, str], tuple[Context, PrepareContent]] = {}

    async def register(self, ctx: Context, *, name: str, prepare: PrepareContent) -> Effect:
        """贡献者只能注册纯投影；同一内容位置出现两个处理者时明确拒绝。"""
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("内容投影不能跨 Root 注册")
        if not isinstance(name, str) or not name or not callable(prepare):
            raise ValueError("内容投影需要名称和 prepare 函数")
        key = (ctx.runtime.plugin_id, name)

        def setup():
            if key in self._sources:
                raise ValueError(f"内容投影重复: {key}")
            self._sources[key] = (ctx, prepare)
            return lambda: self._sources.pop(key)

        return await ctx.effect(setup, label=f"content-view:{name}")

    def binding_contributors(self) -> tuple[Context, ...]:
        return tuple(dict.fromkeys(ctx for ctx, _ in self._sources.values()))

    @asynccontextmanager
    async def bind(self) -> AsyncIterator[PrepareContent]:
        """一次回复固定实际 ACTIVE 贡献者，排空前不释放其代码作用域。"""
        async with self._ctx.runtime_scope(), AsyncExitStack() as stack:
            sources = tuple(value for _, value in sorted(self._sources.items())
                            if value[0].fiber.state is FiberState.ACTIVE)
            for ctx in dict.fromkeys(ctx for ctx, _ in sources):
                await stack.enter_async_context(ctx.runtime_scope())
            active = True

            def prepare(messages: tuple[Message, ...], source: str, tools: frozenset[str],
                        seen: frozenset[tuple[str, int]]) -> ContentTransform:
                if not active:
                    raise RuntimeError("内容投影视图已关闭")
                transforms = tuple(make(messages, source, tools, seen) for _, make in sources)

                def render(message: Message, index: int) -> RenderedContent | None:
                    if not active:
                        raise RuntimeError("内容投影视图已关闭")
                    result = None
                    for transform in transforms:
                        candidate = transform(message, index)
                        if candidate is not None:
                            if result is not None:
                                raise ValueError(f"内容位置有多个投影 owner: {message.message_id}/{index}")
                            result = candidate
                    return result

                return render

            try:
                yield prepare
            finally:
                active = False
