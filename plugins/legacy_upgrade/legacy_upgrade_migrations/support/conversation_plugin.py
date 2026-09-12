"""Frozen channel provenance check used while importing historical turns."""

from collections.abc import Mapping
from typing import cast

from agent.plugin_contracts import ContentPart, ContentReferences


def check_origin(part: ContentPart) -> ContentReferences:
    """Validate the persisted channel identity without consulting a live channel plugin."""
    raw_value = part.value
    if not isinstance(raw_value, Mapping):
        raise ValueError("channel.origin 必须是对象")
    value = cast(Mapping[str, object], raw_value)
    if set(value) != {"channel", "chat_id", "sender"} or any(
        not isinstance(item, str) or not item for item in value.values()
    ):
        raise ValueError("channel.origin 身份无效")
    return ContentReferences()
