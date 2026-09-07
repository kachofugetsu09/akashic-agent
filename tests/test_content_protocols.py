import json
import re
from contextlib import asynccontextmanager

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

    async def provider(ctx):
        await apply(ctx, None)

    async def consumer(ctx):
        for definition in definitions:
            await ctx.require(CONTENT).register(ctx, definition)

    store = RuntimeSnapshotStore()
    try:
        await root.mount(provider, name="content-provider")
        await root.mount(consumer, name="external-owner", inject=(CONTENT,))
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


async def decode_text(text, protocols, references=()):
    async with bound_content(protocols) as view:
        return await view.decode(text, references)


def citation_protocol():
    """外部 Citation 形态 fixture：真实引用证据与模型声明分别保存。"""
    pattern = re.compile(r"§cited:(\[[^\]]*\])§|\[§([^\]]+)\]")

    def check(part):
        if set(part.value) != {"ref", "declared", "retrieval_ref", "resolved_ref"}:
            raise ValueError("citation schema")
        value = part.value
        if (
            not isinstance(value["ref"], str)
            or not value["ref"]
            or type(value["declared"]) is not bool
        ):
            raise ValueError("citation identity")
        for key in ("retrieval_ref", "resolved_ref"):
            if value[key] is not None and (
                not isinstance(value[key], str) or not value[key]
            ):
                raise ValueError("citation proof")
        if not value["declared"] and value["retrieval_ref"] is None:
            raise ValueError("fallback requires retrieval proof")
        return ()

    async def decode(source, references):
        matches = list(source.matches(pattern))
        available = {}
        for ref in references:
            if ref.ref in available and available[ref.ref] != ref:
                raise ValueError("conflicting reference proof")
            available[ref.ref] = ref
        result = []
        for match in matches:
            ids = json.loads(match[1]) if match[1] else [match[2]]
            parts = tuple(
                ContentPart(
                    "citation",
                    {
                        "ref": ident,
                        "declared": True,
                        "retrieval_ref": (
                            available[ident].retrieval_ref
                            if ident in available
                            else None
                        ),
                        "resolved_ref": (
                            available[ident].resolved_ref
                            if ident in available
                            else None
                        ),
                    },
                )
                for ident in ids
            )
            result.append(Span(match.start(), match.end(), parts))
        if not matches:
            result.append(
                Span(
                    len(source.text),
                    len(source.text),
                    tuple(
                        ContentPart(
                            "citation",
                            {
                                "ref": ref.ref,
                                "declared": False,
                                "retrieval_ref": ref.retrieval_ref,
                                "resolved_ref": ref.resolved_ref,
                            },
                        )
                        for ref in available.values()
                        if ref.retrieval_ref is not None
                    ),
                )
            )
        return result

    return TextProtocol(
        name="citation",
        prompt="内部引用协议",
        decode=decode,
        content={"citation": check},
    )


def meme_protocol(picks):
    """外部 Meme 形态 fixture：只选择第一枚有效标记，消息保存固定选择。"""
    pattern = re.compile(r"<meme:([^>]+)>")

    def check(part):
        if set(part.value) != {"category", "artifact_id"}:
            raise ValueError("meme schema")
        return ()

    async def decode(source, _references):
        matches = list(source.matches(pattern))
        result = []
        for index, match in enumerate(matches):
            category = match[1].lower()
            image = f"image-{len(picks)}" if category == "happy" else None
            if index == 0:
                picks.append(image)
            result.append(
                Span(
                    match.start(),
                    match.end(),
                    (
                        (
                            ContentPart(
                                "meme", {"category": category, "artifact_id": image}
                            ),
                        )
                        if index == 0
                        else ()
                    ),
                )
            )
        return result

    return TextProtocol(
        name="meme",
        prompt="可用表情类别：happy",
        decode=decode,
        content={"meme": check},
    )


def visible(parts):
    return "".join(part.value for part in parts if part.kind == "text")


@pytest.mark.asyncio
async def test_meme_and_citation_are_independent_of_registration_order():
    raw = '回答。 §cited:["known","unknown"]§ <meme:HAPPY> <other:literal>'
    references = (Reference("known", "memory@revision", "retrieval-ticket"),)
    first = await decode_text(raw, (meme_protocol([]), citation_protocol()), references)
    second = await decode_text(
        raw, (citation_protocol(), meme_protocol([])), references
    )
    assert first == second
    assert visible(first) == "回答。   <other:literal>"
    citations = [part.value for part in first if part.kind == "citation"]
    assert citations[0]["declared"] is True
    assert citations[0]["resolved_ref"] == "memory@revision"
    assert citations[1]["resolved_ref"] is None
    assert [part.value["artifact_id"] for part in first if part.kind == "meme"] == [
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
    parts = await decode_text(
        raw,
        (meme_protocol(picks), citation_protocol()),
        (Reference("actual", "revision", "ticket"),),
    )
    assert visible(parts) == raw
    assert picks == []
    citations = [part.value for part in parts if part.kind == "citation"]
    assert [(item["ref"], item["declared"]) for item in citations] == [
        ("actual", False)
    ]


@pytest.mark.asyncio
async def test_overlapping_protocols_fail_without_changing_the_original():
    async def first(source, _references):
        return (Span(0, len(source.text), ()),)

    async def second(source, _references):
        return (Span(1, len(source.text), ()),)

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
        return (Span(0, 1, (ContentPart("model.facts", {"fake": True}),)),)

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
    try:
        async with bound_content((meme_protocol(picks),)) as view:
            writer = log.writer(
                "s",
                author="model",
                source="conversation",
                body_types=(Output,),
                content=view.checks,
            )
            parts = await view.decode("回答 <meme:happy>")
            original = writer.append("reply", Output(parts, "complete"))
        assert writer.append("reply", original.body) == original
        assert picks == ["image-0"]
        with pytest.raises(RuntimeError, match="lease"):
            writer.append("late", original.body)
    finally:
        log.close()


@pytest.mark.asyncio
async def test_external_identity_registers_via_ordinary_effect_and_real_runtime_lease():
    root = CompositionRoot("content-generation")

    async def provider(ctx):
        await apply(ctx, None)

    async def external(ctx):
        await ctx.require(CONTENT).register(ctx, meme_protocol([]))

    await root.mount(provider, name="independent-content-provider")
    await root.mount(external, name="external-meme", inject=(CONTENT,))
    store = RuntimeSnapshotStore()
    snapshot = RuntimeSnapshotCompiler().compile({}, composition_root=root)
    store.install(snapshot)
    lease = store.lease()
    token = bind_runtime_snapshot(lease)
    try:
        async with root.context.require(CONTENT).bind() as view:
            assert snapshot.lease_count == 2
            assert view.prompts == ("可用表情类别：happy",)
            assert any(
                part.kind == "meme" for part in await view.decode("<meme:happy>")
            )
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
        return ()

    schema = ContentSchema(name="structured", content={"fact": check})
    async with bound_content((schema,)) as view:
        assert view.prompts == ()
        assert await view.decode("plain") == (ContentPart("text", "plain"),)
        assert view.checks["fact"](ContentPart("fact", "saved")) == ()
        with pytest.raises(ValueError, match="invalid structured"):
            view.checks["fact"](ContentPart("fact", "bad"))
    duplicate = ContentSchema(name="other", content={"fact": check})
    with pytest.raises(RuntimeError, match="拓扑未就绪"):
        async with bound_content((schema, duplicate)):
            pytest.fail("duplicate schema registered")


@pytest.mark.asyncio
async def test_citation_fallback_requires_real_retrieval_and_conflicting_proof_fails():
    parts = await decode_text(
        "answer", (citation_protocol(),), (Reference("direct", "revision"),)
    )
    assert parts == (ContentPart("text", "answer"),)
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
        )

    with pytest.raises(ValueError, match="零长度"):
        await decode_text(
            "answer",
            (TextProtocol(name="injector", content={}, prompt="", decode=decode),),
        )


@pytest.mark.asyncio
async def test_escaped_backtick_leaves_a_real_code_delimiter():
    raw = r"\``<meme:happy>`"
    picks = []
    assert visible(await decode_text(raw, (meme_protocol(picks),))) == raw
    assert picks == []


@pytest.mark.asyncio
async def test_inline_code_cannot_cross_paragraph_boundaries():
    parts = await decode_text("`<meme:happy>\n\n`", (meme_protocol([]),))
    assert any(part.kind == "meme" for part in parts)
    assert visible(parts) == "`\n\n`"
