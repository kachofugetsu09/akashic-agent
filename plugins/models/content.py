from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

from agent.media import (
    MAX_IMAGE_DATA_URI_TOTAL_BYTES,
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGE_TOTAL_BYTES,
    encode_image_bytes,
)
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
from agent.plugin_composition.artifacts import ArtifactRead
from agent.plugin_contracts.models import MODEL_CONTENT as MODEL_CONTENT


async def load_artifacts(
    reader: ArtifactRead,
    refs: Sequence[AttachmentRef],
    *,
    accepts_images: bool,
    current_artifact_ids: frozenset[str],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """有界构造本次图片视图；当前 Input 优先，其余按最近引用优先。"""
    # 1. 一个 artifact 的所有出现共用投影，资源预算按实际出现次数计费。
    counts = Counter(ref.artifact_id for ref in refs)
    recent = {ref.artifact_id: ref for ref in reversed(refs)}
    ordered = sorted(recent.values(), key=lambda ref: ref.artifact_id not in current_artifact_ids)
    content: dict[str, tuple[Mapping[str, Any], ...]] = {}
    raw_bytes = encoded_bytes = 0
    exhausted = False
    for ref in ordered:
        label: dict[str, object] = {"artifact": asdict(ref)}
        uri = None
        if ref.kind is AttachmentKind.IMAGE:
            if not accepts_images:
                label["image_status"] = "当前模型不接收图片；未提供图片内容"
            else:
                current = ref.artifact_id in current_artifact_ids
                raw_cost = ref.size_bytes * counts[ref.artifact_id]
                fits = (not exhausted and ref.size_bytes <= MAX_IMAGE_FILE_BYTES
                        and raw_bytes + raw_cost <= MAX_IMAGE_TOTAL_BYTES)
                if fits:
                    # 2. 字节完整性由 Artifact owner 核验，关闭租约后在 worker 转图。
                    lease = await reader.acquire(ref)
                    try:
                        raw = await lease.read_bytes(max_bytes=MAX_IMAGE_FILE_BYTES)
                    finally:
                        await lease.aclose()
                    candidate = await asyncio.to_thread(encode_image_bytes, raw)
                    del raw
                    encoded_cost = len(candidate) * counts[ref.artifact_id]
                    fits = encoded_bytes + encoded_cost <= MAX_IMAGE_DATA_URI_TOTAL_BYTES
                    if fits:
                        uri = candidate
                        raw_bytes += raw_cost
                        encoded_bytes += encoded_cost
                    del candidate
                if not fits:
                    if current:
                        raise ValueError("当前输入图片超过临时图片视图资源预算，请减少或压缩图片")
                    # 3. 旧图停止加载，明确说明未提供像素；原消息与原图保持不变。
                    exhausted = True
                    label["image_status"] = "本次未提供图片内容；临时图片视图资源预算已用完"
        blocks: list[Mapping[str, Any]] = [
            {"type": "text", "text": json.dumps(label, ensure_ascii=False)},
        ]
        if uri is not None:
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
    if part.kind in {"model.selection", "tool.selection", "context.summary", "context.notice", "history.record", "history.turn_input"}:
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
    """模型正文与附件解释使用原函数，权限由传入的只读端口限定。"""

    load_artifacts = staticmethod(load_artifacts)
    render = staticmethod(render_content)
