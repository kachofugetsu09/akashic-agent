"""Frozen base content checks used while importing historical turns."""

from agent.plugin_contracts import ContentPart, ContentReferences


def check_text(part: ContentPart) -> ContentReferences:
    """Validate the immutable text part without consulting the live content plugin."""
    if not isinstance(part.value, str):
        raise TypeError("text 内容必须是字符串")
    return ContentReferences()
