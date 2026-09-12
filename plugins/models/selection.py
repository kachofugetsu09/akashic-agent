from collections.abc import Mapping, MutableMapping, Sequence
from typing import cast

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.models import ChatModelSelection
from agent.plugin_contracts import ContentPart, ContentReferences, Input, Message


SESSION_MODEL_SELECTION_KEY = "model_selection"
LEGACY_MODEL_OVERRIDE_KEY = "model_runtime_override"


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


def read_saved(metadata: Mapping[str, object]) -> ChatModelSelection:
    """读取持久选择，并在显式选择前兼容旧的 runtime override。"""
    raw = metadata.get(SESSION_MODEL_SELECTION_KEY)
    if raw is not None:
        if not isinstance(raw, Mapping):
            raise ValueError("session model_selection 必须是对象")
        value = cast(Mapping[str, object], raw)
        if value.get("schema_version") != 1:
            raise ValueError("session model_selection schema_version 无效")
        model_ref = value.get("model_ref", "")
        effort = value.get("reasoning_effort", "")
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise ValueError("session model_selection.model_ref 必须是非空字符串")
        if not isinstance(effort, str):
            raise ValueError("session model_selection.reasoning_effort 必须是字符串")
        return ChatModelSelection(model_ref.strip(), effort.strip() or None)

    legacy = metadata.get(LEGACY_MODEL_OVERRIDE_KEY)
    if legacy is None:
        return ChatModelSelection()
    if not isinstance(legacy, str) or not legacy.strip():
        raise ValueError("session model_runtime_override 必须是非空字符串")
    return ChatModelSelection(legacy.strip(), None)


def write_saved(
    metadata: MutableMapping[str, object], selection: ChatModelSelection,
) -> None:
    """写入显式选择，或在清除时只删除选择字段。"""
    _ = metadata.pop(LEGACY_MODEL_OVERRIDE_KEY, None)
    if not selection.model_id:
        if selection.reasoning_effort:
            raise ValueError("默认模型不能单独覆盖推理强度")
        _ = metadata.pop(SESSION_MODEL_SELECTION_KEY, None)
        return
    metadata[SESSION_MODEL_SELECTION_KEY] = {
        "schema_version": 1,
        "model_ref": selection.model_id,
        "reasoning_effort": selection.reasoning_effort or "",
    }


class SelectionOwner:
    check = staticmethod(check_selection)
    read = staticmethod(selection)
    read_saved = staticmethod(read_saved)
    write_saved = staticmethod(write_saved)


MODEL_SELECTION = ServiceKey[SelectionOwner]("models.selection.v1")
