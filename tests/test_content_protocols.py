import json
import re
from contextlib import asynccontextmanager
from typing import Any, cast
from pathlib import Path
from tempfile import TemporaryDirectory
import pytest
from agent.plugin_composition import CompositionRoot
from agent.plugin_composition.model import FiberState, PluginRuntime
from plugins.content.plugin import CONTENT, Span, TextProtocol, apply
from session.message import ContentPart

@asynccontextmanager
async def bound_content(definitions):
    """通过公共注册与绑定接口取得真实本地 owner scopes。"""
    root = CompositionRoot("content-test")
    temporary = TemporaryDirectory(prefix="content-protocol-")
    path = Path(temporary.name)

    async def provider(ctx):
        await apply(ctx)

    try:
        await root.mount(provider, name="content-provider")
        for definition in definitions:
            async def consumer(ctx, definition=definition):
                await ctx.require(CONTENT).register(ctx, definition)
            fiber = await root.mount(consumer, name=definition.name, inject=(CONTENT,),
                                     runtime=PluginRuntime(definition.name, "content-test", path, path, path, {}))
            if fiber.state is FiberState.FAILED:
                assert fiber.error is not None
                raise fiber.error
        async with root.context.require(CONTENT).bind() as view:
            yield view
    finally:
        await root.dispose()
        temporary.cleanup()

async def decode_text(text, protocols, references=()):
    async with bound_content(protocols) as view:
        return await view.decode(text, references)

def reference_data(ref: str, resolved_ref=None, retrieval_ref=None):
    return {"ref": ref, "resolved_ref": resolved_ref, "retrieval_ref": retrieval_ref}

def citation_protocol():
    """外部 Citation 形态 fixture：真实引用证据与模型声明分别保存。"""
    pattern = re.compile(r"§cited:(\[[^\]]*\])§|\[§([^\]]+)\]")

    async def decode(source, references):
        matches = list(source.matches(pattern))
        available = {}
        for ref in references:
            if ref.ref in available and available[ref.ref] != ref:
                raise ValueError("conflicting reference proof")
            available[ref.ref] = ref
        spans, citations = [], []
        for match in matches:
            ids = json.loads(match[1]) if match[1] else [match[2]]
            if not isinstance(ids, list) or any(not isinstance(ident, str) or not ident for ident in ids):
                raise ValueError("citation identity")
            citations.extend({
                "ref": ident, "declared": True,
                "retrieval_ref": available[ident].retrieval_ref if ident in available else None,
                "resolved_ref": available[ident].resolved_ref if ident in available else None,
            } for ident in ids)
            spans.append(Span(match.start(), match.end(), ()))
        if not matches:
            citations.extend({
                "ref": ref.ref, "declared": False,
                "retrieval_ref": ref.retrieval_ref, "resolved_ref": ref.resolved_ref,
            } for ref in available.values() if ref.retrieval_ref is not None)
        return spans, {"version": 1, "references": citations} if citations else {}

    return TextProtocol(name="citation", prompt="内部引用协议", decode=decode, content={})

def meme_protocol(picks, image_id=None):
    """图片是普通附件；类别是插件自己的附加信息，不注册 Meme 内容类型。"""
    pattern = re.compile(r"<meme:([^>]+)>")

    async def decode(source, _references):
        matches = list(source.matches(pattern))
        result, metadata = [], {}
        for index, match in enumerate(matches):
            category = match[1].lower()
            image = (image_id or f"image-{len(picks)}") if category == "happy" else None
            parts = ()
            if index == 0:
                picks.append(image)
                metadata = {"category": category}
                if image is not None:
                    parts = (ContentPart("artifact_ref", image),)
            result.append(Span(match.start(), match.end(), parts))
        return result, metadata

    return TextProtocol(name="meme", prompt="可用表情类别：happy", decode=decode, content={})

def visible(parts):
    return "".join(part.value for part in parts if part.kind == "text")

@pytest.mark.asyncio
async def test_meme_and_citation_are_independent_of_registration_order():
    raw = '回答。 §cited:["known","unknown"]§ <meme:HAPPY> <other:literal>'
    references = (reference_data("known", "memory@revision", "retrieval-ticket"),)
    first, metadata = await decode_text(raw, (meme_protocol([]), citation_protocol()), references)
    second, other_metadata = await decode_text(
        raw, (citation_protocol(), meme_protocol([])), references
    )
    assert first == second
    assert metadata == other_metadata
    assert visible(first) == "回答。   <other:literal>"
    citations = cast(dict[str, Any], metadata["citation"])["references"]
    assert citations[0]["declared"] is True
    assert citations[0]["resolved_ref"] == "memory@revision"
    assert citations[1]["resolved_ref"] is None
    assert metadata["meme"] == {"category": "happy"}
    assert [part.value for part in first if part.kind == "artifact_ref"] == [
        "image-0"
    ]
