from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from agent.core.passive_support import build_context_hint_message
from agent.core.types import ContextRequest
from agent.lifecycle.phase import (
    PhaseFrame,
    PhaseModule,
    append_string_exports,
    collect_prefixed_slots,
)
from agent.lifecycle.types import PromptRenderCtx, PromptRenderInput, PromptRenderResult
from agent.prompting import PromptSectionRender
from bus.event_bus import EventBus

if TYPE_CHECKING:
    from agent.context import ContextBuilder


@dataclass
class PromptRenderFrame(PhaseFrame[PromptRenderInput, PromptRenderResult]):
    pass


PromptRenderModules: TypeAlias = list[PhaseModule[PromptRenderFrame]]


_CTX_SLOT = "prompt:ctx"
_RESULT_SLOT = "prompt:result"
_SECTION_TOP_PREFIX = "prompt:section_top:"
_SECTION_BOTTOM_PREFIX = "prompt:section_bottom:"
_EXTRA_HINT_PREFIX = "prompt:extra_hint:"


class _BuildPromptRenderCtxModule:
    produces = (_CTX_SLOT,)

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        input = frame.input
        frame.slots[_CTX_SLOT] = PromptRenderCtx(
            session_key=input.session_key,
            channel=input.channel,
            chat_id=input.chat_id,
            content=input.content,
            media=input.media,
            timestamp=input.timestamp,
            history=input.history,
            skill_names=input.skill_names,
            retrieved_memory_block=input.retrieved_memory_block,
            disabled_sections=set(input.disabled_sections),
            turn_injection_prompt=input.turn_injection_prompt,
            extra_hints=list(input.extra_hints or []),
        )
        return frame


class _EmitPromptRenderCtxModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        ctx = cast(PromptRenderCtx, frame.slots[_CTX_SLOT])
        frame.slots[_CTX_SLOT] = await self._bus.emit(ctx)
        return frame


class _RenderPromptModule:
    requires = (_CTX_SLOT,)
    produces = (_RESULT_SLOT,)

    def __init__(self, context: ContextBuilder) -> None:
        self._context = context

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        ctx = cast(PromptRenderCtx, frame.slots[_CTX_SLOT])
        rendered = self._context.render(
            ContextRequest(
                history=ctx.history,
                current_message=ctx.content,
                media=ctx.media,
                skill_names=ctx.skill_names,
                channel=ctx.channel,
                chat_id=ctx.chat_id,
                message_timestamp=ctx.timestamp,
                retrieved_memory_block=ctx.retrieved_memory_block,
                disabled_sections=ctx.disabled_sections,
                turn_injection_prompt=ctx.turn_injection_prompt,
            ),
            system_sections_top=ctx.system_sections_top,
            system_sections_bottom=ctx.system_sections_bottom,
        )
        messages = list(rendered.messages)
        if ctx.extra_hints:
            messages.append(
                build_context_hint_message(
                    "plugin_hints",
                    "\n".join(ctx.extra_hints),
                )
            )
        frame.slots[_RESULT_SLOT] = PromptRenderResult(messages=messages)
        return frame


class _CollectPromptExportSlotsModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        ctx = cast(PromptRenderCtx, frame.slots[_CTX_SLOT])
        _append_sections(
            ctx.system_sections_top,
            collect_prefixed_slots(frame.slots, _SECTION_TOP_PREFIX),
        )
        _append_sections(
            ctx.system_sections_bottom,
            collect_prefixed_slots(frame.slots, _SECTION_BOTTOM_PREFIX),
        )
        append_string_exports(
            ctx.extra_hints,
            collect_prefixed_slots(frame.slots, _EXTRA_HINT_PREFIX),
        )
        return frame


class _ReturnPromptRenderResultModule:
    requires = (_RESULT_SLOT,)

    async def run(self, frame: PromptRenderFrame) -> PromptRenderFrame:
        frame.output = cast(PromptRenderResult, frame.slots[_RESULT_SLOT])
        return frame


def default_prompt_render_modules(
    bus: EventBus,
    context: ContextBuilder,
    plugin_modules_top: PromptRenderModules | None = None,
    plugin_modules_bottom: PromptRenderModules | None = None,
) -> PromptRenderModules:
    top_modules = plugin_modules_top or []
    bottom_modules = plugin_modules_bottom or []
    return [
        _BuildPromptRenderCtxModule(),
        _EmitPromptRenderCtxModule(bus),
        *top_modules,
        *bottom_modules,
        _CollectPromptExportSlotsModule(),
        _RenderPromptModule(context),
        _ReturnPromptRenderResultModule(),
    ]


def _append_sections(
    target: list[PromptSectionRender],
    exports: dict[str, object],
) -> None:
    for name, value in exports.items():
        if isinstance(value, PromptSectionRender):
            target.append(value)
        elif isinstance(value, str) and value.strip():
            target.append(
                PromptSectionRender(
                    name=name,
                    content=value,
                    is_static=False,
                )
            )
