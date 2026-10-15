from __future__ import annotations

import asyncio
import codecs
import json
import posixpath
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any, cast

from plugins.models.contract import (
    MAX_IMAGE_DATA_URI_TOTAL_BYTES,
    MAX_IMAGE_FILE_BYTES,
    MAX_IMAGE_TOTAL_BYTES,
    encode_image_bytes,
)
from plugins.channels.contract import (
    AttachmentKind,
    AttachmentRef,
)
from plugins.ledger.contract import (
    ContentPart,
    Control,
    Message,
    ToolCall,
    json_value,
)
from core.common.frozen_json import freeze_json
from plugins.ledger.contract import ArtifactRead
from plugins.models.contract import MODEL_CONTENT as MODEL_CONTENT


# 文本类附件直接把内容交给模型：模型没有按 artifact 读文件的工具，只给元数据等于没传。
# 历史里的附件每轮都会重新投影，预算按 token 成本保守取（48KB 约 1.2 万 token）。
MAX_TEXT_FILE_BYTES = 48 * 1024
MAX_TEXT_TOTAL_BYTES = 160 * 1024
_TEXT_MEDIA_TYPES = frozenset({
    "application/json", "application/ld+json", "application/xml", "application/yaml", "application/x-yaml",
    "application/toml", "application/javascript", "application/x-javascript", "application/typescript",
    "application/x-sh", "application/x-python", "application/sql", "application/x-ndjson",
})
# mimetypes 猜不出或猜错（.ts 会被当成 video/mp2t）的常见源码与配置扩展名。
_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv", ".tsv", ".json", ".jsonl", ".ndjson", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".env", ".xml", ".html", ".css", ".scss", ".py", ".pyi", ".js", ".mjs",
    ".cjs", ".jsx", ".ts", ".tsx", ".mts", ".vue", ".svelte", ".go", ".rs", ".java", ".kt", ".swift", ".c",
    ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".lua", ".sh", ".bash", ".zsh", ".fish", ".sql",
    ".graphql", ".proto", ".diff", ".patch", ".gitignore", ".dockerfile", ".tex", ".r", ".dart", ".zig",
})


def _text_like(ref: AttachmentRef) -> bool:
    media = (ref.media_type or "").split(";", 1)[0].strip().lower()
    if media.startswith("text/") or media in _TEXT_MEDIA_TYPES or media.endswith(("+json", "+xml")):
        return True
    name = (ref.filename or "").lower()
    extension = posixpath.splitext(name)[1] or (name if name.startswith(".") else "")
    return extension in _TEXT_EXTENSIONS or name in {"dockerfile", "makefile", "readme", "license"}


# 读前缀并按 UTF-8 解码；带 NUL 或含非法字节的视为二进制，不交给模型。
# 截断处可能落在合法多字节字符中间：增量解码器只把这种未写完的尾部留在缓冲里丢掉，非法字节照常报错。
def _decode_text(raw: bytes, *, truncated: bool) -> str | None:
    if b"\x00" in raw:
        return None
    try:
        return codecs.getincrementaldecoder("utf-8")().decode(raw, final=not truncated)
    except UnicodeDecodeError:
        return None


# 有预算时读入文本附件内容，写回附件标签里的状态说明；返回实际计入预算的字节数与内容块。
# 同一附件在历史里出现几次，正文就投影几次：单份额度按出现次数分摊，返回的成本已乘上次数。
async def _text_view(reader: ArtifactRead, ref: AttachmentRef, label: dict[str, object], *,
                     remaining: int, occurrences: int) -> tuple[int, Mapping[str, Any] | None]:
    # 1. 预算用完时只说明未提供，原附件不变
    if remaining // occurrences <= 0:
        label["text_status"] = "本次未提供文本内容；文本附件预算已用完"
        return 0, None
    if ref.size_bytes == 0:
        label["text_status"] = "空文件"
        return 0, None
    limit = min(MAX_TEXT_FILE_BYTES, remaining // occurrences)
    truncated = ref.size_bytes > limit
    # 2. 只读前缀：超长文件给开头部分并写明
    lease = await reader.acquire(ref)
    try:
        raw = await lease.read_chunk(offset=0, max_bytes=min(limit, ref.size_bytes))
    finally:
        await lease.aclose()
    text = _decode_text(raw, truncated=truncated)
    if text is None:
        label["text_status"] = "不是 UTF-8 文本；未提供内容"
        return 0, None
    if truncated:
        label["text_status"] = f"内容过长，只提供前 {len(raw) // 1024}KB（共 {ref.size_bytes // 1024}KB）"
    # 3. 内容放在独立文本块里，与附件标签分开，避免 JSON 转义把代码弄得难读
    return len(raw) * occurrences, {"type": "text", "text": f"<attachment_text artifact_id=\"{ref.artifact_id}\">\n{text}\n</attachment_text>"}


async def load_artifacts(
    reader: ArtifactRead,
    refs: Sequence[AttachmentRef],
    *,
    accepts_images: bool,
    current_artifact_ids: frozenset[str],
) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    """有界构造本次图片与文本附件视图；当前 Input 优先，其余按最近引用优先。"""
    # 1. 一个 artifact 的所有出现共用投影，资源预算按实际出现次数计费。
    counts = Counter(ref.artifact_id for ref in refs)
    recent = {ref.artifact_id: ref for ref in reversed(refs)}
    ordered = sorted(recent.values(), key=lambda ref: ref.artifact_id not in current_artifact_ids)
    content: dict[str, tuple[Mapping[str, Any], ...]] = {}
    raw_bytes = encoded_bytes = text_bytes = 0
    exhausted = False
    for ref in ordered:
        label: dict[str, object] = {"artifact": asdict(ref)}
        uri = None
        text_block: Mapping[str, Any] | None = None
        if ref.kind is AttachmentKind.FILE and _text_like(ref):
            cost, text_block = await _text_view(reader, ref, label, remaining=MAX_TEXT_TOTAL_BYTES - text_bytes,
                                                occurrences=counts[ref.artifact_id])
            text_bytes += cost
        elif ref.kind is AttachmentKind.IMAGE:
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
        if text_block is not None:
            blocks.append(text_block)
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
