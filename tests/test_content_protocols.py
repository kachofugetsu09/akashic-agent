from session.message import ContentReferences
import asyncio
import json
import re
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any, cast
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from agent.plugin_composition import CompositionRoot, RUNTIME_STARTED
from agent.plugin_composition.model import FiberState, PluginRuntime
from plugins.content.plugin import (
    CONTENT,
    ContentSchema,
    Span,
    TextProtocol,
    apply,
)
from session.log import MessageLog
from session.message import ContentPart, Output


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
            await root.mount(consumer, name=definition.name, inject=(CONTENT,),
                             runtime=PluginRuntime(definition.name, "content-test", path, path, path, {}))
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
        (reference_data("actual", "revision", "ticket"),),
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
async def test_external_identity_registers_via_ordinary_effect_and_real_local_scope(tmp_path):
    root = CompositionRoot("content-generation")

    async def provider(ctx):
        await apply(ctx)

    async def external(ctx):
        await ctx.require(CONTENT).register(ctx, meme_protocol([]))

    await root.mount(provider, name="independent-content-provider")
    await root.mount(
        external,
        name="external-meme",
        inject=(CONTENT,),
        runtime=PluginRuntime(
            "external-meme", "content-generation", tmp_path, tmp_path, tmp_path, {}
        ),
    )
    try:
        async with root.context.require(CONTENT).bind() as view:
            assert view.prompts == ("可用表情类别：happy",)
            parts, metadata = await view.decode("<meme:happy>")
            assert parts == (ContentPart("artifact_ref", "image-0"),)
            assert metadata == {"external-meme": {"category": "happy"}}
        with pytest.raises(RuntimeError, match="lease"):
            await view.decode("<meme:happy>")
    finally:
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
        "answer", (citation_protocol(),), (reference_data("direct", "revision"),)
    )
    assert parts == (ContentPart("text", "answer"),)
    assert metadata == {}
    with pytest.raises(ValueError, match="conflicting reference"):
        await decode_text(
            "[§same]",
            (citation_protocol(),),
            (
                reference_data("same", "revision-1", "ticket-1"),
                reference_data("same", "revision-2", "ticket-2"),
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


@pytest.mark.asyncio
async def test_dynamic_protocol_freezes_prompt_and_decoder_until_next_bind(tmp_path):
    root = CompositionRoot("dynamic-protocol")
    current = ["first"]
    prepared = []

    def prepare():
        value = current[0]
        prepared.append(value)
        async def decode(source, references):
            return (), {"selected": value}
        return TextProtocol(name="dynamic", prompt=value, decode=decode, content={})

    async def provider(ctx):
        await apply(ctx)

    async def consumer(ctx):
        await ctx.require(CONTENT).register(ctx, prepare(), prepare=prepare)

    try:
        await root.mount(provider, name="content-provider")
        await root.mount(consumer, name="dynamic", inject=(CONTENT,), runtime=PluginRuntime(
            "dynamic", "dynamic-test", tmp_path, tmp_path, tmp_path, {},
        ))
        content = root.context.require(CONTENT)
        async with content.bind() as first:
            assert first.prompts == ("first",)
            current[0] = "second"
            assert first.prompts == ("first",)
            assert cast(Mapping[str, object], (await first.decode("reply"))[1]["dynamic"])["selected"] == "first"
            async with content.bind() as second:
                assert second.prompts == ("second",)
                assert cast(Mapping[str, object], (await second.decode("reply"))[1]["dynamic"])["selected"] == "second"
            assert cast(Mapping[str, object], (await first.decode("reply"))[1]["dynamic"])["selected"] == "first"
        assert prepared == ["first", "first", "second"]
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_content_view_retains_old_contributor_but_new_bind_excludes_unloading(
    tmp_path,
):
    root = CompositionRoot("content-local-drain")
    prepare_calls = []
    cleanup_calls = 0
    unrelated_contexts = []
    unrelated_events = []

    async def provider(ctx):
        await apply(ctx)

    async def unrelated(ctx):
        unrelated_contexts.append(ctx)
        await ctx.on(RUNTIME_STARTED, lambda _event: unrelated_events.append("started"))

    unrelated_fiber = await root.mount(
        unrelated,
        name="unrelated-content",
        runtime=PluginRuntime(
            "unrelated-content", "content-local-drain", tmp_path, tmp_path, tmp_path, {}
        ),
    )
    unrelated_context = unrelated_contexts[0]
    unrelated_events_before = tuple(unrelated_events)

    def prepare():
        prepare_calls.append("prepare")

        async def decode(_source, _references):
            return (), {"selected": "old"}

        return TextProtocol(name="dynamic", prompt="old", decode=decode, content={})

    async def contributor(ctx):
        nonlocal cleanup_calls

        def cleanup():
            nonlocal cleanup_calls
            cleanup_calls += 1

        await ctx.effect(lambda: cleanup, label="content-contributor-resource")
        await ctx.require(CONTENT).register(ctx, prepare(), prepare=prepare)
        await ctx.require(CONTENT).register(
            ctx,
            ContentSchema(name="structured", content={"other": lambda _part: ContentReferences()}),
        )

    await root.mount(provider, name="content-provider")
    contributor_fiber = await root.mount(
        contributor,
        name="contributor",
        inject=(CONTENT,),
        runtime=PluginRuntime(
            "contributor", "content-local-drain", tmp_path, tmp_path, tmp_path, {}
        ),
    )
    content = root.context.require(CONTENT)
    try:
        async with content.bind() as old_view:
            assert old_view.prompts == ("old",)
            assert len(contributor_fiber.context.fiber._fiber._in_flight_calls) == 1
            assert cleanup_calls == 0
            dispose_task = asyncio.create_task(contributor_fiber.dispose())
            drain_started = asyncio.Event()
            asyncio.get_running_loop().call_soon(drain_started.set)
            await drain_started.wait()
            assert contributor_fiber.state is FiberState.UNLOADING

            assert old_view.checks["other"](ContentPart("other", "value")) == ContentReferences()
            assert await old_view.decode("old") == (
                (ContentPart("text", "old"),),
                {"contributor": {"selected": "old"}},
            )
            async with content.bind() as new_view:
                assert new_view.prompts == ()
                assert "other" not in new_view.checks
                assert await new_view.decode("new") == ((ContentPart("text", "new"),), {})
        await dispose_task
        assert contributor_fiber.state is FiberState.DISPOSED
        assert not content._definitions
        assert cleanup_calls == 1
        assert prepare_calls == ["prepare", "prepare"]
        assert unrelated_fiber.state is FiberState.ACTIVE
        assert unrelated_fiber.context is unrelated_context
        assert tuple(unrelated_events) == unrelated_events_before
    finally:
        await root.dispose()


@pytest.mark.asyncio
async def test_content_bind_excludes_loading_contributor_until_started(tmp_path):
    root = CompositionRoot("content-local-loading")
    started = asyncio.Event()
    release = asyncio.Event()
    prepare_calls = []

    async def provider(ctx):
        await apply(ctx)

    def prepare():
        prepare_calls.append("prepare")

        async def decode(_source, _references):
            return (), {}

        return TextProtocol(name="loading", prompt="ready", decode=decode, content={})

    async def contributor(ctx):
        await ctx.require(CONTENT).register(ctx, prepare(), prepare=prepare)

        async def on_started(_event):
            started.set()
            await release.wait()

        await ctx.on(RUNTIME_STARTED, on_started)

    await root.mount(provider, name="content-provider")
    mount_task = asyncio.create_task(
        root.mount(
            contributor,
            name="loading-contributor",
            inject=(CONTENT,),
            runtime=PluginRuntime(
                "loading-contributor", "content-local-loading", tmp_path, tmp_path, tmp_path, {}
            ),
        )
    )
    content = root.context.require(CONTENT)
    try:
        await started.wait()
        assert {
            view.name: view.state for view in root.receipt().fibers
        }["loading-contributor"] is FiberState.LOADING
        async with content.bind() as view:
            assert view.prompts == ()
            assert prepare_calls == ["prepare"]
        release.set()
        await mount_task
        async with content.bind() as view:
            assert view.prompts == ("ready",)
        assert prepare_calls == ["prepare", "prepare"]
    finally:
        release.set()
        await mount_task
        await root.dispose()


@pytest.mark.asyncio
async def test_content_bind_prepare_failure_releases_local_scopes(tmp_path):
    root = CompositionRoot("content-local-failure")
    provider_contexts = []
    contributor_contexts = []
    prepare_calls = 0

    async def provider(ctx):
        provider_contexts.append(ctx)
        await apply(ctx)

    def prepare():
        nonlocal prepare_calls
        prepare_calls += 1
        if prepare_calls > 1:
            raise ValueError("dynamic prepare failed")

        async def decode(_source, _references):
            return (), {}

        return TextProtocol(name="failure", prompt="failure", decode=decode, content={})

    async def contributor(ctx):
        contributor_contexts.append(ctx)
        await ctx.require(CONTENT).register(ctx, prepare(), prepare=prepare)

    await root.mount(provider, name="content-provider")
    await root.mount(
        contributor,
        name="failing-contributor",
        inject=(CONTENT,),
        runtime=PluginRuntime(
            "failing-contributor", "content-local-failure", tmp_path, tmp_path, tmp_path, {}
        ),
    )
    try:
        with pytest.raises(ValueError, match="dynamic prepare failed"):
            async with root.context.require(CONTENT).bind():
                pytest.fail("prepare failure must abort bind")
        assert not provider_contexts[0].fiber._fiber._in_flight_calls
        assert not contributor_contexts[0].fiber._fiber._in_flight_calls
    finally:
        await root.dispose()
