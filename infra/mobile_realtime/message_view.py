from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import cast

MAX_MESSAGE_PAYLOAD_BYTES = 240 * 1024


def message_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def message_chunks(payload: dict[str, object]) -> Iterator[dict[str, object]]:
    """按帧预算分组完整消息；超大消息只换成整条 JSON 的下载引用。"""
    # 1. 引用摘要与 HTTP 下载共用同一 JSON 编码。
    items: list[dict[str, object]] = []
    after_seq = cast(int, payload["after_seq"])
    for row in cast(list[dict[str, object]], payload["items"]):
        content = message_json(row)
        if len(content) > 64 * 1024:
            reference: dict[str, object] = {"id": row["id"], "session_id": row["session_id"], "seq": row["seq"],
                   "message_ref": {"version": 2, "encoding": "utf-8", "media_type": "application/json",
                                   "byte_length": len(content), "sha256": hashlib.sha256(content).hexdigest()}}
            row = reference
        candidate: dict[str, object] = {**payload, "items": [*items, row], "after_seq": after_seq,
                     "next_after_seq": row["seq"], "has_more": row["seq"] != payload["through_seq"]}
        if len(message_json(candidate)) > MAX_MESSAGE_PAYLOAD_BYTES:
            if not items:
                raise ValueError("单条消息引用超过同步帧上限")
            # 2. 当前组交付后继续剩余项，固定 through_seq 不变。
            yield {**payload, "items": items, "after_seq": after_seq,
                   "next_after_seq": items[-1]["seq"], "has_more": True}
            after_seq = cast(int, items[-1]["seq"])
            items = []
        items.append(row)
    result: dict[str, object] = {**payload, "items": items, "after_seq": after_seq,
              "next_after_seq": items[-1]["seq"] if items else after_seq}
    if len(message_json(result)) > MAX_MESSAGE_PAYLOAD_BYTES:
        raise ValueError("单条消息引用超过同步帧上限")
    yield result


def bounded_reply_status(payload: dict[str, object]) -> dict[str, object]:
    """仅缩短临时草稿并显式标记；活动身份和已提交 Message 不受影响。"""
    if len(message_json(payload)) <= MAX_MESSAGE_PAYLOAD_BYTES:
        return payload
    # 1. 复制展示值；不改 ReplyRead 所拥有的完整草稿。
    items: list[dict[str, object]] = [{**item, "preview": dict(cast(dict[str, object], item["preview"])) if item["preview"] is not None else None}
             for item in cast(list[dict[str, object]], payload["items"])]
    result: dict[str, object] = {**payload, "items": items}
    previews = [cast(dict[str, object], item["preview"]) for item in items if item["preview"] is not None]
    limit = max((len(cast(str, preview[key])) for preview in previews for key in ("text", "thinking")), default=0)
    # 2. 按 Unicode 字符收缩，直到完整 JSON 能放入一帧。
    while len(message_json(result)) > MAX_MESSAGE_PAYLOAD_BYTES:
        if limit == 0:
            raise ValueError("回复活动身份超过同步帧上限")
        limit //= 2
        for preview in previews:
            for key in ("text", "thinking"):
                value = cast(str, preview[key])
                if len(value) > limit:
                    preview[key] = value[:limit]
                    preview["truncated"] = True
    return result
