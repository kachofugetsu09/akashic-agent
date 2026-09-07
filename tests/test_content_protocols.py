from session.message import ContentReferences
import json
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, cast
from pathlib import Path
from tempfile import TemporaryDirectory
from agent.plugin_composition.model import PluginRuntime

import pytest

from agent.plugin_composition import CompositionRoot
from agent.plugins.snapshot import (
    RuntimeSnapshotCompiler,
    RuntimeSnapshotStore,
    bind_runtime_snapshot,
    reset_runtime_snapshot,
)
from plugins.content.plugin import (
    CONTENT,
    ContentSchema,
    Reference,
    Span,
    TextProtocol,
    apply,
)
from session.log import MessageLog
from session.message import ContentPart, Output


@asynccontextmanager
async def bound_content(definitions):
    """通过公共注册与绑定接口取得真实 generation lease。"""
    root = CompositionRoot("content-test")
    temporary = TemporaryDirectory(prefix="content-protocol-")
    path = Path(temporary.name)

    async def provider(ctx):
        await apply(ctx, None)

    store = RuntimeSnapshotStore()
    try:
        await root.mount(provider, name="content-provider")
        for definition in definitions:
            async def consumer(ctx, definition=definition):
                await ctx.require(CONTENT).register(ctx, definition)
            await root.mount(consumer, name=definition.name, inject=(CONTENT,),
                             runtime=PluginRuntime(definition.name, "content-test", path, path, path, {}))
        store.install(RuntimeSnapshotCompiler().compile({}, composition_root=root))
        lease = store.lease()
        token = bind_runtime_snapshot(lease)
        try:
            async with root.context.require(CONTENT).bind() as view:
                yield view
        finally:
            reset_runtime_snapshot(token)
            await lease.release()
    finally:
        await store.close()
        await root.dispose()
        temporary.cleanup()


async def decode_text(text, protocols, references=()):
    async with bound_content(protocols) as view:
        return await view.decode(text, references)


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
    references = (Reference("known", "memory@revision", "retrieval-ticket"),)
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


@pytest.mark.parametrize(
    "raw",
    [
        '`<meme:happy> §cited:["example"]§`',
        '`` ` <meme:happy> §cited:["example"]§ ``',
        '```text\n<meme:happy> §cited:["example"]§\n```',
        '~~~text\r\n<meme:happy> §cited:["example"]§\r\n~~~',
        '    <meme:happy> §cited:["example"]§',
        '> 示例\n<meme:happy> §cited:["example"]§',
    ],
)
@pytest.mark.asyncio
async def test_literal_markers_are_preserved_and_do_not_suppress_retrieval_fallback(
    raw,
):
    picks = []
    parts, metadata = await decode_text(
        raw,
        (meme_protocol(picks), citation_protocol()),
        (Reference("actual", "revision", "ticket"),),
    )
    assert visible(parts) == raw
    assert picks == []
    citations = cast(dict[str, Any], metadata["citation"])["references"]
    assert [(item["ref"], item["declared"]) for item in citations] == [
        ("actual", False)
    ]


@pytest.mark.asyncio
async def test_overlapping_protocols_fail_without_changing_the_original():
    async def first(source, _references):
        return (Span(0, len(source.text), ()),), {}

    async def second(source, _references):
        return (Span(1, len(source.text), ()),), {}

    with pytest.raises(ValueError, match="冲突"):
        await decode_text(
            "immutable",
            (
                TextProtocol(name="one", prompt="", decode=first, content={}),
                TextProtocol(name="two", prompt="", decode=second, content={}),
            ),
        )


@pytest.mark.asyncio
async def test_protocol_cannot_emit_another_owners_content():
    async def decode(_source, _references):
        return (Span(0, 1, (ContentPart("model.facts", {"fake": True}),)),), {}

    with pytest.raises(PermissionError, match="未声明"):
        await decode_text(
            "x", (TextProtocol(name="unrelated", prompt="", decode=decode, content={}),)
        )


@pytest.mark.asyncio
async def test_saved_output_retry_keeps_the_same_image_and_expired_checks_reject_new_writes(
    tmp_path,
):
    picks = []
    log = MessageLog(tmp_path / "messages.db")
    from infra.channels.artifacts import ChannelAttachmentArtifactStore
    from session.artifact_store import ArtifactStore
    from session.artifacts import AttachmentKind
    store = ArtifactStore(tmp_path / "messages.db")
    artifacts = ChannelAttachmentArtifactStore(workspace=tmp_path, metadata_store=store)
    ref = await artifacts.import_bytes(b"\x89PNG\r\n\x1a\nfixture", kind=AttachmentKind.IMAGE,
                                       filename="meme.png", media_type="image/png")
    try:
        async with bound_content((meme_protocol(picks, ref.artifact_id),)) as view:
            writer = log.writer(
                "s",
                author="model",
                source="conversation",
                body_types=(Output,),
                content=view.checks,
                check_metadata=view.check_metadata,
            )
            parts, metadata = await view.decode("回答 <meme:happy>")
            with pytest.raises(PermissionError, match="解码"):
                writer.append("fake", Output(parts, "complete"), metadata={"meme": {"category": "forged"}})
            assert log.catalog().snapshot_heads() == {}
            original = writer.append("reply", Output(parts, "complete"), metadata=metadata)
        assert writer.append("reply", original.body, metadata=original.metadata) == original
        assert picks == [ref.artifact_id]
        with pytest.raises(RuntimeError, match="lease"):
            writer.append("late", original.body, metadata=original.metadata)
    finally:
        store.close()
        log.close()


@pytest.mark.asyncio
async def test_external_identity_registers_via_ordinary_effect_and_real_runtime_lease(tmp_path):
    root = CompositionRoot("content-generation")

    async def provider(ctx):
        await apply(ctx, None)

    async def external(ctx):
        await ctx.require(CONTENT).register(ctx, meme_protocol([]))

    await root.mount(provider, name="independent-content-provider")
    await root.mount(external, name="external-meme", inject=(CONTENT,),
                     runtime=PluginRuntime("external-meme", "content-generation", tmp_path, tmp_path, tmp_path, {}))
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        async with root.context.require(CONTENT).bind() as view:
            assert snapshot.lease_count == 2
            assert view.prompts == ("可用表情类别：happy",)
            parts, metadata = await view.decode("<meme:happy>")
            assert parts == (ContentPart("artifact_ref", "image-0"),)
            assert metadata == {"external-meme": {"category": "happy"}}
        assert snapshot.lease_count == 1
        with pytest.raises(RuntimeError, match="lease"):
            await view.decode("<meme:happy>")
    finally:
        reset_runtime_snapshot(token)
        await lease.release()
        await store.close()
        await root.dispose()


@pytest.mark.asyncio
async def test_plain_schema_needs_no_text_protocol_and_owns_its_kind():
    def check(part):
        if part.value != "saved":
            raise ValueError("invalid structured fact")
        return ContentReferences()

    schema = ContentSchema(name="structured", content={"fact": check})
    async with bound_content((schema,)) as view:
        assert view.prompts == ()
        assert await view.decode("plain") == ((ContentPart("text", "plain"),), {})
        assert view.checks["fact"](ContentPart("fact", "saved")) == ContentReferences()
        with pytest.raises(ValueError, match="invalid structured"):
            view.checks["fact"](ContentPart("fact", "bad"))
    duplicate = ContentSchema(name="other", content={"fact": check})
    with pytest.raises(RuntimeError, match="拓扑未就绪"):
        async with bound_content((schema, duplicate)):
            pytest.fail("duplicate schema registered")


@pytest.mark.asyncio
async def test_citation_fallback_requires_real_retrieval_and_conflicting_proof_fails():
    parts, metadata = await decode_text(
        "answer", (citation_protocol(),), (Reference("direct", "revision"),)
    )
    assert parts == (ContentPart("text", "answer"),)
    assert metadata == {}
    with pytest.raises(ValueError, match="conflicting reference"):
        await decode_text(
            "[§same]",
            (citation_protocol(),),
            (
                Reference("same", "revision-1", "ticket-1"),
                Reference("same", "revision-2", "ticket-2"),
            ),
        )


@pytest.mark.asyncio
async def test_zero_length_span_cannot_inject_visible_text():
    async def decode(source, _references):
        return (
            Span(
                len(source.text), len(source.text), (ContentPart("text", "injected"),)
            ),
        ), {}

    with pytest.raises(ValueError, match="零长度"):
        await decode_text(
            "answer",
            (TextProtocol(name="injector", content={}, prompt="", decode=decode),),
        )


@pytest.mark.asyncio
async def test_escaped_backtick_leaves_a_real_code_delimiter():
    raw = r"\``<meme:happy>`"
    picks = []
    parts, _ = await decode_text(raw, (meme_protocol(picks),))
    assert visible(parts) == raw
    assert picks == []


@pytest.mark.asyncio
async def test_inline_code_cannot_cross_paragraph_boundaries():
    parts, metadata = await decode_text("`<meme:happy>\n\n`", (meme_protocol([]),))
    assert metadata == {"meme": {"category": "happy"}}
    assert visible(parts) == "`\n\n`"
