from collections.abc import Mapping, Sequence
from typing import cast

from agent.plugin_composition.models import ChatModelSelection
from session.message import ContentPart, ContentReferences, Input, Message


def check_selection(part: ContentPart) -> ContentReferences:
    """模型偏好是用户选择事实，实际可用性由 Model owner 在调用前验证。"""
    raw_value = part.value
    if not isinstance(raw_value, Mapping):
        raise ValueError("model.selection 必须是对象")
    value = cast(Mapping[str, object], raw_value)
    if set(value) != {"model_id", "reasoning_effort"} or any(
        item is not None and (not isinstance(item, str) or not item)
        for item in value.values()
    ):
        raise ValueError("model.selection 字段无效")
    return ContentReferences()


def selection(messages: Sequence[Message]) -> ChatModelSelection | None:
    """读取给定输入范围的最后一次显式选择；None 表示没有选择事实。"""
    for message in reversed(messages):
        if isinstance(message.body, Input):
            for part in reversed(message.body.parts):
                if part.kind == "model.selection":
                    value = cast(Mapping[str, str | None], part.value)
                    return ChatModelSelection(value["model_id"], value["reasoning_effort"])
    return None
