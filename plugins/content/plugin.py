from __future__ import annotations

import re
from collections.abc import (
    AsyncGenerator,
    Callable,
    Mapping,
    Sequence,
)
from contextlib import asynccontextmanager
from types import MappingProxyType
from typing import Protocol

from markdown_it import MarkdownIt

from agent.plugin_composition import Context, Effect, ServiceKey
from agent.plugin_composition.bindings import Bindings
from session.message import ContentPart

# 类型属于 Content 的公开 API；不同归档实现共享当前已校验的 binding ABI。
from plugins.content.api import (
    ContentCheck,
    ContentSchema,
    Reference,
    Span,
    TextProtocol,
    TextSource,
)

api_version = 3
name = "content"
version = "1.0.0"
desc = "按固定协议组装内容，各解析器只读取同一份原文"
inject = ()


def check_text(part: ContentPart) -> tuple[str, ...]:
    if not isinstance(part.value, str):
        raise TypeError("text 内容必须是字符串")
    return ()


def _literal_ranges(text: str) -> tuple[tuple[int, int], ...]:
    """保留 Markdown 代码与引用的原始区间，不把示例当成输出协议。"""
    # 1. Block token 的行范围覆盖 fenced/indented code 和 lazy blockquote。
    offsets = [0]
    offsets.extend(match.end() for match in re.finditer(r"\r\n?|\n", text))
    if offsets[-1] < len(text):
        offsets.append(len(text))
    protected: list[tuple[int, int]] = []
    tokens = MarkdownIt("commonmark").parse(text)
    for token in tokens:
        if token.type in {"fence", "code_block", "blockquote_open"}:
            assert token.map is not None
            protected.append((offsets[token.map[0]], offsets[token.map[1]]))

    # 2. Inline code 不跨 Markdown 段；转义只消费一个 backtick。
    for token in tokens:
        if token.type != "inline":
            continue
        assert token.map is not None
        left, right = offsets[token.map[0]], offsets[token.map[1]]
        runs = list(re.finditer(r"`+", text[left:right]))
        index = 0
        while index < len(runs):
            opening = runs[index]
            start = left + opening.start()
            length = len(opening[0])
            escaped = len(text[:start]) - len(text[:start].rstrip("\\"))
            if escaped % 2:
                start += 1
                length -= 1
            if length == 0 or any(a <= start < b for a, b in protected):
                index += 1
                continue
            closing_index = next(
                (
                    other
                    for other in range(index + 1, len(runs))
                    if len(runs[other][0]) == length
                ),
                None,
            )
            if closing_index is None:
                index += 1
                continue
            protected.append((start, left + runs[closing_index].end()))
            index = closing_index + 1
    return tuple(sorted(protected))


async def _decode_text(
    text: str, protocols: Sequence[TextProtocol], references: tuple[Reference, ...] = ()
) -> tuple[ContentPart, ...]:
    """一次组装整份原文，冲突明确失败；不按安装先后串行改写文本。"""
    if not isinstance(text, str):
        raise TypeError("待解码内容必须是字符串")
    references = tuple(references)
    if any(not isinstance(ref, Reference) for ref in references):
        raise TypeError("引用必须来自已校验的 Reference")
    protected = _literal_ranges(text)
    source = TextSource(text, protected)
    spans: list[tuple[str, Span]] = []
    # 1. 每个 decoder 都接收同一个不可变原文和引用集合。
    for protocol in sorted(protocols, key=lambda item: item.name):
        for span in await protocol.decode(source, references):
            if not isinstance(span, Span) or span.end > len(text):
                raise ValueError(f"{protocol.name} 返回越界的内容区间")
            if not source.allows(span.start, span.end):
                continue
            for part in span.parts:
                if part.kind == "text":
                    _ = check_text(part)
                else:
                    try:
                        check = protocol.content[part.kind]
                    except KeyError as exc:
                        raise PermissionError(
                            f"{protocol.name} 未声明内容 {part.kind}"
                        ) from exc
                    _ = check(part)
            spans.append((protocol.name, span))
    # 2. 统一区间排序并检查重叠；没有 priority，也不吞掉未知协议。
    spans.sort(key=lambda item: (item[1].start, item[1].end, item[0]))
    cursor = 0
    parts: list[ContentPart] = []
    for owner, span in spans:
        if span.start < cursor:
            raise ValueError(f"文本协议区间冲突: {owner}")
        if span.start > cursor:
            parts.append(ContentPart("text", text[cursor : span.start]))
        parts.extend(span.parts)
        cursor = span.end
    if cursor < len(text):
        parts.append(ContentPart("text", text[cursor:]))
    return tuple(parts)


class ContentView(Protocol):
    @property
    def prompts(self) -> tuple[str, ...]: ...

    @property
    def checks(self) -> Mapping[str, ContentCheck]: ...

    async def decode(
        self, text: str, references: tuple[Reference, ...] = ()
    ) -> tuple[ContentPart, ...]: ...


class _ContentView:
    """本次请求固定的协议集合；Prompt、解码和提交共用其 generation lease。"""

    def __init__(self, definitions: tuple[ContentSchema, ...]):
        self._definitions = definitions
        self._protocols = tuple(
            item for item in definitions if isinstance(item, TextProtocol)
        )
        self._active = True

    @property
    def prompts(self) -> tuple[str, ...]:
        self._check_active()
        return tuple(protocol.prompt for protocol in self._protocols if protocol.prompt)

    @property
    def checks(self) -> Mapping[str, ContentCheck]:
        self._check_active()
        return MappingProxyType(
            {
                "text": self._live_check(check_text),
                **{
                    kind: self._live_check(check)
                    for definition in self._definitions
                    for kind, check in definition.content.items()
                },
            }
        )

    def _live_check(self, check: ContentCheck) -> ContentCheck:
        def validate(part: ContentPart) -> tuple[str, ...]:
            self._check_active()
            return check(part)

        return validate

    async def decode(
        self, text: str, references: tuple[Reference, ...] = ()
    ) -> tuple[ContentPart, ...]:
        self._check_active()
        parts = await _decode_text(text, self._protocols, references)
        self._check_active()
        return parts

    def _check_active(self) -> None:
        if not self._active:
            raise RuntimeError("内容协议 lease 已释放")

    def close(self) -> None:
        self._active = False


class Content:
    def __init__(self, ctx: Context):
        self._ctx = ctx
        self._definitions: dict[str, tuple[Context, ContentSchema]] = {}

    async def register(self, ctx: Context, definition: ContentSchema) -> Effect:
        """内容与协议共用普通 Effect 注册，schema 只有一个 owner。"""
        if not isinstance(definition, ContentSchema):
            raise TypeError("内容声明必须是 ContentSchema 或 TextProtocol")
        if ctx.root_instance_token is not self._ctx.root_instance_token:
            raise ValueError("内容声明不能跨 composition generation 注册")

        def setup() -> Callable[[], None]:
            if definition.name in self._definitions:
                raise ValueError(f"内容声明重复: {definition.name}")
            kinds = {
                kind for _, item in self._definitions.values() for kind in item.content
            }
            if kinds.intersection(definition.content):
                raise ValueError("内容 schema 必须有唯一 owner")
            self._definitions[definition.name] = (ctx, definition)

            def cleanup() -> None:
                del self._definitions[definition.name]

            return cleanup

        return await ctx.effect(setup, label=f"content:{definition.name}")

    def describe(self) -> Mapping[str, object]:
        return {
            name: {
                "kinds": tuple(sorted(definition.content)),
                "prompt": (
                    definition.prompt if isinstance(definition, TextProtocol) else None
                ),
            }
            for name, (_, definition) in sorted(self._definitions.items())
        }

    def save_binding(self, bindings: Bindings) -> str:
        """固定实际注册者及协议选择，恢复不能从当前安装补全遗漏的 decoder。"""
        return bindings.bind(
            CONTENT,
            self.describe(),
            contributors=tuple(ctx for ctx, _ in self._definitions.values()),
        )

    @asynccontextmanager
    async def bind(self) -> AsyncGenerator[ContentView]:
        """调用方把 bind 保持到 append 完成，未提交结果不跨 lease 恢复。"""
        async with self._ctx.runtime_scope():
            view = _ContentView(
                tuple(self._definitions[key][1] for key in sorted(self._definitions))
            )
            try:
                yield view
            finally:
                view.close()


CONTENT = ServiceKey[Content]("content.v1")


@asynccontextmanager
async def open_content(
    bindings: Bindings, binding_id: str
) -> AsyncGenerator[ContentView]:
    """打开已固定的内容协议；动态配置改变时明确拒绝不同的解析选择。"""
    async with bindings.open(binding_id, CONTENT) as (content, metadata):
        if content.describe() != metadata:
            raise ValueError("归档内容协议与固定描述不一致")
        async with content.bind() as view:
            yield view


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.provide(CONTENT, Content(ctx))
