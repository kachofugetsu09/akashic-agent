from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping


SESSION_MODEL_SELECTION_KEY = "model_selection"
LEGACY_MODEL_OVERRIDE_KEY = "model_runtime_override"


@dataclass(frozen=True)
class SessionModelSelection:
    """Describe the model choice persisted for one conversation."""

    model_ref: str = ""
    reasoning_effort: str = ""


def read_session_model_selection(
    metadata: MutableMapping[str, object],
) -> SessionModelSelection:
    """Read the structured selection while accepting the legacy string field."""

    raw = metadata.get(SESSION_MODEL_SELECTION_KEY)
    if raw is not None:
        if not isinstance(raw, dict):
            raise ValueError("session model_selection 必须是对象")
        if raw.get("schema_version") != 1:
            raise ValueError("session model_selection schema_version 无效")
        model_ref = raw.get("model_ref", "")
        effort = raw.get("reasoning_effort", "")
        if not isinstance(model_ref, str) or not model_ref.strip():
            raise ValueError("session model_selection.model_ref 必须是非空字符串")
        if not isinstance(effort, str):
            raise ValueError("session model_selection.reasoning_effort 必须是字符串")
        return SessionModelSelection(model_ref.strip(), effort.strip())

    legacy = metadata.get(LEGACY_MODEL_OVERRIDE_KEY)
    if legacy is None:
        return SessionModelSelection()
    if not isinstance(legacy, str) or not legacy.strip():
        raise ValueError("session model_runtime_override 必须是非空字符串")
    return SessionModelSelection(legacy.strip(), "")


def write_session_model_selection(
    metadata: MutableMapping[str, object],
    selection: SessionModelSelection,
) -> None:
    """Persist one explicit selection, or follow the global default when empty."""

    metadata.pop(LEGACY_MODEL_OVERRIDE_KEY, None)
    if not selection.model_ref:
        if selection.reasoning_effort:
            raise ValueError("默认模型不能单独覆盖推理强度")
        metadata.pop(SESSION_MODEL_SELECTION_KEY, None)
        return
    metadata[SESSION_MODEL_SELECTION_KEY] = {
        "schema_version": 1,
        "model_ref": selection.model_ref,
        "reasoning_effort": selection.reasoning_effort,
    }


__all__ = [
    "SessionModelSelection",
    "read_session_model_selection",
    "write_session_model_selection",
]
