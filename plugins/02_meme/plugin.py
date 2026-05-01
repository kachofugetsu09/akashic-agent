from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from agent.lifecycle.types import AfterReasoningCtx, PromptRenderCtx
from agent.plugins import Plugin, on_after_reasoning
from agent.prompting import PromptSectionRender
from .runtime import MemeCatalog, MemeDecorator

_CTX_SLOT = "prompt:ctx"
_MEME_RE = re.compile(r"<meme:([a-zA-Z0-9_-]+)>", re.IGNORECASE)


class MemePromptModule:
    requires = (_CTX_SLOT,)
    produces = (_CTX_SLOT,)

    def __init__(self, plugin: "MemePlugin") -> None:
        self._plugin = plugin

    async def run(self, frame: Any) -> Any:
        ctx = frame.slots.get(_CTX_SLOT)
        if not isinstance(ctx, PromptRenderCtx):
            return frame
        block = self._plugin.catalog.build_prompt_block()
        if not block:
            return frame
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="memes",
                content=f"# Memes\n\n{block}",
                is_static=False,
            )
        )
        return frame


class MemePlugin(Plugin):
    name = "meme"
    _catalog: Any = None
    _decorator: Any = None

    async def initialize(self) -> None:
        memes_dir = _workspace(self.context.plugin_dir, self.context.workspace) / "memes"
        self._catalog = MemeCatalog(memes_dir)
        self._decorator = MemeDecorator(self._catalog)

    def prompt_render_modules_bottom(self) -> list[object]:
        return [MemePromptModule(self)]

    @on_after_reasoning()
    async def decorate_meme(self, ctx: AfterReasoningCtx) -> AfterReasoningCtx:
        cleaned, tag = _extract_meme_tag(ctx.reply)
        decorated = self.decorator.decorate(cleaned, meme_tag=tag)
        ctx.reply = decorated.content
        ctx.media.extend(decorated.media)
        ctx.meme_tag = decorated.tag
        return ctx

    @property
    def catalog(self) -> Any:
        if self._catalog is None:
            raise RuntimeError("meme 插件尚未初始化")
        return self._catalog

    @property
    def decorator(self) -> Any:
        if self._decorator is None:
            raise RuntimeError("meme 插件尚未初始化")
        return self._decorator


def _extract_meme_tag(response: str) -> tuple[str, str | None]:
    first = _MEME_RE.search(response)
    cleaned = _MEME_RE.sub("", response).strip()
    if first is None:
        return cleaned, None
    return cleaned, first.group(1).lower()


def _workspace(plugin_dir: Path, configured: Path | None) -> Path:
    if configured is not None:
        return configured
    return cast(Path, plugin_dir.parent.parent)
