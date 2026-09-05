from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
import json
from typing import Any, cast

from agent.media import (
    MAX_IMAGE_DATA_URI_TOTAL_BYTES,
    MAX_IMAGE_FILE_BYTES,
    encode_image_bytes,
    validate_image_attachment_budget,
)
from agent.plugin_composition.channels import (
    AttachmentKind,
    AttachmentRef,
    ChannelAttachmentReadPort,
)
from session.message import ContentPart, Control, Message, ToolCall, freeze_json
from session.message_codec import json_value


async def load_artifacts(
    reader: ChannelAttachmentReadPort,
    refs: Sequence[AttachmentRef],
    *,
    accepts_images: bool,
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """先取得只读附件材料，后续投影只读该值；图片变化不会重写原 artifact。"""
    images = [ref for ref in refs if ref.kind is AttachmentKind.IMAGE]
    if accepts_images:
        validate_image_attachment_budget([ref.size_bytes for ref in images])
    content: dict[str, tuple[Mapping[str, Any], ...]] = {}
    encoded_bytes = 0
    for ref in refs:
        # 1. 文件和不支持视觉的模型保留精确附件身份及明确的能力说明。
        label: dict[str, object] = {"artifact": asdict(ref)}
        blocks: list[Mapping[str, Any]] = []
        if ref.kind is AttachmentKind.IMAGE and not accepts_images:
            label["image_status"] = "当前模型不接收图片；原附件仍可按 artifact_id 读取"
        blocks.append({"type": "text", "text": json.dumps(label, ensure_ascii=False)})
        if ref.kind is AttachmentKind.IMAGE and accepts_images:
            # 2. Artifact owner 校验 hash；Model 只对获得的 bytes 做请求图转换。
            lease = await reader.acquire(ref)
            try:
                raw = await lease.read_bytes(max_bytes=MAX_IMAGE_FILE_BYTES)
            finally:
                await lease.aclose()
            uri = await asyncio.to_thread(encode_image_bytes, raw)
            encoded_bytes += len(uri)
            if encoded_bytes > MAX_IMAGE_DATA_URI_TOTAL_BYTES:
                raise ValueError("模型请求图片合计超过编码字节上限")
            blocks.append({"type": "image_url", "image_url": {"url": uri}})
        content[ref.artifact_id] = tuple(blocks)
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
