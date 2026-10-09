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


class BoundViews:
    """一次回复固定的 ACTIVE 内容贡献者及其动态范围声明。

    dynamic_kinds 为 None 表示存在未声明范围的贡献者：下游不能据此
    断定任何消息是静态的（评审 #1110——未知不能合并成已完整声明）。
    """

    def __init__(self, prepare: PrepareContent | None, dynamic_kinds: frozenset[str] | None) -> None:
        self.prepare = prepare
        self.dynamic_kinds = dynamic_kinds


class ContentViews:
    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._sources: dict[tuple[str, str], tuple[Context, PrepareContent, frozenset[str] | None]] = {}

    async def register(self, ctx: Context, *, name: str, prepare: PrepareContent,
                       dynamic_kinds: frozenset[str] | None = None) -> Effect:
        """贡献者只能注册纯投影；同一内容位置出现两个处理者时明确拒绝。

        dynamic_kinds 声明该投影可能返回非 None 的内容 kind；对不含这些
        kind 的消息必须恒返回 None，且同一路径的消息内容不变时结果不变。
        不传入（None）表示未声明：范围未知，下游按全动态处理。
        """
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("内容投影不能跨 Root 注册")
        if not isinstance(name, str) or not name or not callable(prepare):
            raise ValueError("内容投影需要名称和 prepare 函数")
        key = (ctx.runtime.plugin_id, name)

        def setup():
            if key in self._sources:
                raise ValueError(f"内容投影重复: {key}")
            declared = None if dynamic_kinds is None else frozenset(dynamic_kinds)
            self._sources[key] = (ctx, prepare, declared)
            return lambda: self._sources.pop(key)

        return await ctx.effect(setup, label=f"content-view:{name}")

    def binding_contributors(self) -> tuple[Context, ...]:
        return tuple(dict.fromkeys(ctx for ctx, _, _ in self._sources.values()))

    @asynccontextmanager
    async def bind(self) -> AsyncIterator[BoundViews]:
        """一次回复固定实际 ACTIVE 贡献者，排空前不释放其代码作用域。

        没有 ACTIVE 贡献者时 prepare 为 None：调用方走基础渲染路径，
        不为空组合逐消息调用恒空的 transform。
        """
        async with self._ctx.runtime_scope(), AsyncExitStack() as stack:
            sources = tuple(value for _, value in sorted(self._sources.items())
                            if value[0].fiber.state is FiberState.ACTIVE)
            for ctx in dict.fromkeys(ctx for ctx, _, _ in sources):
                await stack.enter_async_context(ctx.runtime_scope())
            active = True

            def prepare(messages: tuple[Message, ...], source: str, tools: frozenset[str],
                        seen: frozenset[tuple[str, int]]) -> ContentTransform:
                if not active:
                    raise RuntimeError("内容投影视图已关闭")
                transforms = tuple(make(messages, source, tools, seen) for _, make, _ in sources)

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
                if not sources:
                    yield BoundViews(None, frozenset())
                    return
                # 任一贡献者未声明范围，组合范围即未知，向下游传 None。
                combined: frozenset[str] | None = (
                    None
                    if any(kinds is None for _, _, kinds in sources)
                    else frozenset().union(*(kinds for _, _, kinds in sources if kinds is not None))
                )
                yield BoundViews(prepare, combined)
            finally:
                active = False
