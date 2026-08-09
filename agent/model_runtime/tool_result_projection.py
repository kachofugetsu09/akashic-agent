from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


TOOL_RESULT_ARCHIVE_CHAR_THRESHOLD = 8192
TOOL_RESULT_READ_DEFAULT_CHARS = 4000
TOOL_RESULT_READ_MAX_CHARS = 6000


@dataclass(frozen=True, slots=True)
class ToolResultPromptProjection:
    """Return one provider-only message projection and its savings."""

    messages: tuple[dict[str, Any], ...]
    masked_result_count: int
    masked_chars: int


def tool_result_placeholder(artifact_id: str) -> str:
    """Render the stable minimal placeholder exposed to the model."""

    if not artifact_id:
        raise ValueError("tool result artifact id 不能为空")
    return json.dumps(
        {"tool_result_ref": artifact_id},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def project_tool_results(
    messages: Sequence[Mapping[str, Any]],
    *,
    artifact_ids_by_call: Mapping[str, str],
    masked_call_ids: set[str],
) -> ToolResultPromptProjection:
    """Replace selected archived tool outputs in a provider-only copy."""

    projected: list[dict[str, Any]] = []
    masked_count = 0
    masked_chars = 0
    for message in messages:
        item = dict(message)
        call_id = item.get("tool_call_id")
        content = item.get("content")
        if (
            item.get("role") == "tool"
            and isinstance(call_id, str)
            and call_id in masked_call_ids
            and isinstance(content, str)
        ):
            artifact_id = artifact_ids_by_call.get(call_id)
            if artifact_id is not None:
                placeholder = tool_result_placeholder(artifact_id)
                if content != placeholder:
                    item["content"] = placeholder
                    masked_count += 1
                    masked_chars += max(0, len(content) - len(placeholder))
        projected.append(item)
    return ToolResultPromptProjection(
        messages=tuple(projected),
        masked_result_count=masked_count,
        masked_chars=masked_chars,
    )
