from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
)
from agent.plugin_contracts import (
    ContentPart,
    Control,
    Message,
    ToolCall,
    freeze_json,
    json_value,
)
from agent.plugin_contracts.models import MODEL_CONTENT as MODEL_CONTENT


def describe_artifacts(
    refs: Sequence[AttachmentRef],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """Render attachment identities as text without reading or encoding their files."""
    content: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for ref in refs:
        label: dict[str, object] = {"artifact": asdict(ref)}
        if ref.kind is AttachmentKind.IMAGE:
            label["image_status"] = "图片占位符；未提供图片内容，不能据此判断画面"
        content[ref.artifact_id] = ({"type": "text", "text": json.dumps(label, ensure_ascii=False)},)
    return cast(Mapping[str, tuple[Mapping[str, Any], ...]], freeze_json(content))


def render_content(
    part: ContentPart,
    *,
    artifacts: Mapping[str, tuple[Mapping[str, Any], ...]],
    read_message: Callable[[str], Message | None] | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """基础正文与附件按协议投影；其余已声明内容作为带 kind 的低信任数据。"""
    if part.kind == "text":
        return ({"type": "text", "text": part.value},)
    if part.kind == "artifact_ref":
        return artifacts[cast(str, part.value)]
    if part.kind in {"model.selection", "tool.selection", "context.summary", "history.record", "history.turn_input"}:
        return ()
    if part.kind == "reply_ref":
        if read_message is None:
            raise RuntimeError("回复引用投影需要当前 Session 的消息读取口")
        target = read_message(cast(str, part.value))
        text = None if target is None or isinstance(target.body, Control) else "\n".join(
            cast(str, item.value) for item in target.body.parts
            if not isinstance(item, ToolCall) and item.kind == "text"
        )
        return ({"type": "text", "text": json.dumps(
            {"reply_to": part.value, "quoted_text": text, "available": target is not None},
            ensure_ascii=False,
        )},)
    if part.kind == "model.facts":
        raise ValueError("model.facts 必须由 Model replay owner 单独处理")
    return (
        {
            "type": "text",
            "text": json.dumps(
                {"kind": part.kind, "value": json_value(part.value)},
                ensure_ascii=False,
            ),
        },
    )


class ContentOwner:
    """Project message content and attachment labels without file access."""

    describe_artifacts = staticmethod(describe_artifacts)
    render = staticmethod(render_content)
