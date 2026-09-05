from __future__ import annotations

import re
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast

from agent.turn_effects import PostCommitEffect, post_commit_effect
from session.artifacts import check_artifact_id
from session.message import ContentPart, ContentReferences, Control, Message


def _history_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("历史 extra 包含重复 JSON 字段")
        result[key] = value
    return result


def check_turn_input(part: ContentPart) -> ContentReferences:
    """迁入 Input 指向原执行 item；metadata 是经迁移校验的旧领域事实。"""
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("history.turn_input 必须是对象")
    value = cast(Mapping[str, object], value)
    if set(value) != {"record_id", "item_id", "metadata"}:
        raise ValueError("history.turn_input schema 无效")
    if any(not isinstance(value[key], str) or not value[key] for key in ("record_id", "item_id")):
        raise ValueError("history.turn_input 缺少旧 record/item 身份")
    if not isinstance(value["metadata"], Mapping):
        raise ValueError("history.turn_input metadata 必须是对象")
    metadata = cast(Mapping[str, object], value["metadata"])
    if "skip_post_memory" in metadata:
        raise ValueError("旧执行排除尚未完成 effects 迁移")
    _ = post_commit_effect(metadata)
    return ContentReferences()


def legacy_post_commit_effect(message: Message) -> PostCommitEffect | None:
    """从迁移保留的原文读取旧投影资格，不猜来源或恢复已退役的标记。"""
    if isinstance(message.body, Control):
        return None
    imported = [part for part in message.body.parts
                if isinstance(part, ContentPart) and part.kind == "history.turn_input"]
    parts = [part for part in message.body.parts
             if isinstance(part, ContentPart) and part.kind == "history.provenance"]
    if imported:
        if len(imported) != 1 or parts:
            raise ValueError("历史 Input 只能有一个来源证明")
        _ = check_turn_input(imported[0])
        value = cast(Mapping[str, object], imported[0].value)
        return post_commit_effect(cast(Mapping[str, object], value["metadata"]))
    if not parts:
        return None
    if len(parts) != 1:
        raise ValueError("历史消息必须只有一个 provenance 对象")
    raw_value = parts[0].value
    if not isinstance(raw_value, Mapping):
        raise ValueError("历史 provenance 必须是对象")
    value = cast(Mapping[str, object], raw_value)
    # 1. 迁移内容是持久读取边界；先验证已知格式与保全的原始字节。
    if (set(value) != {"schema", "role", "content_was_null", "extra", "extra_sha256"}
            or value["schema"] != "sessions.messages.v0"
            or value["role"] not in ("user", "assistant")
            or type(value["content_was_null"]) is not bool):
        raise ValueError("历史 provenance 格式未知")
    raw, digest = value["extra"], value["extra_sha256"]
    if raw is None:
        if digest is not None:
            raise ValueError("历史空 extra 的 digest 不一致")
        return PostCommitEffect.ALLOW
    if not isinstance(raw, str) or hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
        raise ValueError("历史 extra 原文与 digest 不一致")
    extra: object = json.loads(raw, object_pairs_hook=_history_object)
    if not isinstance(extra, dict):
        raise ValueError("历史 extra 必须是 JSON 对象")
    extra = cast(dict[str, object], extra)
    if "skip_post_memory" in extra:
        raise ValueError("历史记忆排除尚未完成 effects 迁移")
    # 2. 只解释迁移已确认的 effects 原语；新来源的权限由其自身合同拥有。
    return post_commit_effect(extra)


@dataclass(frozen=True, slots=True)
class Reference:
    """调用方实际取得的引用证据；模型声明不能自己产生这些权限。"""

    ref: str
    resolved_ref: str | None = None
    retrieval_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ref, str) or not self.ref:
            raise ValueError("引用身份不能为空")
        if any(
            value is not None and (not isinstance(value, str) or not value)
            for value in (self.resolved_ref, self.retrieval_ref)
        ):
            raise ValueError("引用的解析目标与查询凭据必须是非空字符串或 None")


@dataclass(frozen=True, slots=True)
class Span:
    """替换原文的一个精确区间；零长度区间用于附加引用等非文本事实。"""

    start: int
    end: int
    parts: tuple[ContentPart, ...]

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end < self.start
        ):
            raise ValueError("内容区间必须是有序的非负整数")
        parts = tuple(self.parts)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("文本协议只能产生内容块，不能提出工具调用")
        if self.start == self.end and any(part.kind == "text" for part in parts):
            raise ValueError("零长度区间只能附加非文本事实")
        object.__setattr__(self, "parts", parts)


ContentCheck = Callable[[ContentPart], ContentReferences]


@dataclass(frozen=True, slots=True)
class TextSource:
    """原文与其字面区间，供协议同时判定显式声明和召回兜底。"""

    text: str
    literals: tuple[tuple[int, int], ...]

    def allows(self, start: int, end: int) -> bool:
        return not any(left < end and start < right for left, right in self.literals)

    def matches(self, pattern: re.Pattern[str]) -> Iterator[re.Match[str]]:
        return (
            match
            for match in pattern.finditer(self.text)
            if self.allows(match.start(), match.end())
        )


TextDecoder = Callable[[TextSource, tuple[Reference, ...]], Awaitable[Sequence[Span]]]


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentSchema:
    """内容 owner 的纯 schema 声明，结构化生产者不需要文本协议。"""

    name: str
    content: Mapping[str, ContentCheck]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("内容声明名不能为空")
        if any(
            not isinstance(key, str) or not key or not callable(check)
            for key, check in self.content.items()
        ):
            raise TypeError("内容声明必须映射 kind 到纯 schema 校验函数")
        if "text" in self.content or "tool_call" in self.content:
            raise ValueError("内容声明不能重定义基础文本或工具调用")
        object.__setattr__(self, "content", MappingProxyType(dict(self.content)))


@dataclass(frozen=True, slots=True, kw_only=True)
class TextProtocol(ContentSchema):
    """文本协议一起提供提示、解析与产生的内容 schema。"""

    prompt: str
    decode: TextDecoder

    def __post_init__(self) -> None:
        ContentSchema.__post_init__(self)
        if not isinstance(self.prompt, str) or not callable(self.decode):
            raise TypeError("文本协议必须提供提示文本和 decoder")


def check_artifact(part: ContentPart) -> ContentReferences:
    """正文只存 ID；完整元数据由 Artifact owner 保存和读取。"""
    return ContentReferences(artifact_ids=(check_artifact_id(part.value),))
