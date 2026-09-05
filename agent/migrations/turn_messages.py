"""旧执行事实只增归档；仅凭独立入站证据恢复暂停的用户输入。"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from agent.migrations.session_db_backup import backup_sqlite_database
from agent.turn_effects import post_commit_effect
from plugins.content.api import check_artifact, check_turn_input, legacy_post_commit_effect
from plugins.content.plugin import check_text
from plugins.conversation.plugin import check_origin
from session.log import MessageLog, OwnerTransaction, _sql  # pyright: ignore[reportPrivateUsage]
from session.message import ContentPart, ContentReferences, Control, Input, Message, Output
from session.message_codec import json_value

_OWNER = "migration:turn-messages-v1"
_TURNS = """CREATE TABLE turns (
    id TEXT PRIMARY KEY, session_key TEXT NOT NULL, status TEXT NOT NULL,
    input_json TEXT NOT NULL, items_json TEXT NOT NULL, usage_json TEXT,
    error_json TEXT, final_response TEXT, created_at TEXT NOT NULL,
    started_at TEXT, completed_at TEXT
)"""
_HANDOFFS = """CREATE TABLE inbound_handoffs (
    handoff_id TEXT PRIMARY KEY, dedupe_key TEXT UNIQUE, channel TEXT NOT NULL,
    sender TEXT NOT NULL, chat_id TEXT NOT NULL, session_key TEXT NOT NULL,
    content TEXT NOT NULL, timestamp TEXT NOT NULL, media_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
)"""


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False, default=lambda blob: {"sqlite_blob": blob.hex()})
    return hashlib.sha256(payload.encode()).hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"旧执行 JSON 字段重复: {key}")
        result[key] = value
    return result


def _load(raw: str) -> Any:
    return json.loads(raw, object_pairs_hook=_pairs)


def _rows(connection: sqlite3.Connection, name: str) -> list[dict[str, Any]]:
    return [dict(row) for row in connection.execute(f'SELECT * FROM "{name}" ORDER BY rowid')]


def _check(connection: sqlite3.Connection) -> None:
    if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
        raise RuntimeError("旧执行迁移发现 SQLite 完整性错误")
    if list(connection.execute("PRAGMA foreign_key_check")):
        raise RuntimeError("旧执行迁移发现外键错误")


def _schema(connection: sqlite3.Connection, table: str, expected: str) -> bool:
    row = connection.execute("SELECT sql FROM sqlite_master WHERE name=?", (table,)).fetchone()
    if row is None:
        return False
    if _sql(row[0]) != _sql(expected):
        raise RuntimeError(f"旧执行迁移遇到未知 {table} schema lineage")
    return True


def check_record(part: ContentPart) -> ContentReferences:
    """历史归档 writer 只接受完整旧行和对应摘要，不接受原生工具调用。"""
    raw = json_value(part.value)
    if not isinstance(raw, dict):
        raise ValueError("history.record 必须是对象")
    value = cast(dict[str, Any], raw)
    if set(value) != {"schema", "row", "sha256"} or value["schema"] != "sessions.turns.v0":
        raise ValueError("history.record schema 无效")
    row = value["row"]
    if set(row) != {"id", "session_key", "status", "input_json", "items_json", "usage_json",
                    "error_json", "final_response", "created_at", "started_at", "completed_at"}:
        raise ValueError("history.record 未保存完整旧行")
    if any(item is not None and not isinstance(item, str) for item in row.values()):
        raise ValueError("history.record 旧标量类型无效")
    if _digest(row) != value["sha256"]:
        raise ValueError("history.record digest 不匹配")
    return ContentReferences()


def _messages(log: MessageLog) -> dict[str, tuple[Message, dict[str, Any] | None]]:
    """解读已迁入 Message 的旧精确引用，原始 extra digest 必须仍然成立。"""
    result: dict[str, tuple[Message, dict[str, Any] | None]] = {}
    for session_id in log.catalog().snapshot_heads():
        for message in log.reader(session_id).snapshot():
            extra = None
            if not isinstance(message.body, Control):
                parts = [p for p in message.body.parts if isinstance(p, ContentPart)
                         and p.kind == "history.provenance"]
                if parts:
                    if len(parts) != 1:
                        raise ValueError("旧消息有重复 provenance")
                    value = cast(dict[str, Any], json_value(parts[0].value))
                    raw = value["extra"]
                    if raw is not None:
                        if hashlib.sha256(raw.encode()).hexdigest() != value["extra_sha256"]:
                            raise ValueError("旧消息 extra digest 损坏")
                        extra = _load(raw)
            result[message.message_id] = message, extra
    return result


def _plan(rows: list[dict[str, Any]], handoffs: list[dict[str, Any]],
          messages: dict[str, tuple[Message, dict[str, Any] | None]], log: MessageLog) -> list[dict[str, Any]]:
    """执行链只证明顺序；每条新 Input 还必须有独立持久入站身份。"""
    by_id: dict[str, tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = {}
    latest: dict[str, str] = {}
    for row in rows:
        if any(not isinstance(row[key], str) or not row[key] for key in ("id", "session_key", "created_at")):
            raise ValueError("旧执行缺少稳定身份或接纳时间")
        raw_data, raw_items = _load(row["input_json"]), _load(row["items_json"])
        if not isinstance(raw_data, dict) or not isinstance(raw_items, list):
            raise ValueError(f"旧执行 input/items 格式损坏: {row['id']}")
        data = cast(dict[str, Any], raw_data)
        if set(data) != {"input", "metadata"} or not isinstance(data["input"], str) or not isinstance(data["metadata"], dict):
            raise ValueError(f"旧执行 input 格式损坏: {row['id']}")
        items = cast(list[dict[str, Any]], raw_items)
        if any(not isinstance(item, dict) or set(item) != {"id", "type", "data"}
                or not isinstance(item["id"], str) or not item["id"]
                or item["type"] not in {"userMessage", "assistantMessage", "reasoning", "toolCall", "error"}
                or not isinstance(item["data"], dict) for item in items):
            raise ValueError(f"旧执行 items 格式损坏: {row['id']}")
        if row["status"] not in {"queued", "in_progress", "completed", "interrupted", "failed", "cancelled"}:
            raise ValueError(f"旧执行 status 未知: {row['id']}")
        by_id[row["id"]] = row, data["metadata"], items
        latest[row["session_key"]] = row["id"]
    result: list[dict[str, Any]] = []
    for session_id, tail_id in latest.items():
        tail, metadata, _ = by_id[tail_id]
        if tail["status"] == "completed":
            continue
        interaction = metadata.get("interactionId", tail_id)
        if not isinstance(interaction, str) or not interaction:
            raise ValueError(f"旧 interaction 身份无效: {tail_id}")
        chain: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        seen: set[str] = set()
        current = tail_id
        while current is not None:
            if current in seen or current not in by_id:
                raise ValueError(f"旧执行链缺失或有环: {tail_id}/{current}")
            seen.add(current)
            row, meta, items = by_id[current]
            if row["session_key"] != session_id or meta.get("interactionId", current) != interaction:
                raise ValueError(f"旧执行链跨 Session 或 interaction: {tail_id}/{current}")
            chain.insert(0, (row, meta, items))
            current = meta.get("continuedFromTurnId")
        # 已落最终 Message 是提交事实；failed 状态不覆盖它，也不凭 final_response 文本猜测。
        final = [m for m, extra in messages.values() if m.session_id == session_id
                 and isinstance(m.body, Output) and m.body.finish != "continue"
                 and extra is not None and extra.get("control_turn_id") == interaction]
        if len(final) > 1:
            raise ValueError(f"旧执行有多个最终 Message 引用: {tail_id}")
        if final:
            continue
        tools = [(row["id"], item) for row, _, items in chain for item in items if item["type"] == "toolCall"]
        if tools:
            # TODO: 旧 ToolCall status/resultPreview 不是领域 receipt；确认真实效果后才能续接。
            identities = [(record, item["id"], item["data"].get("status")) for record, item in tools]
            raise RuntimeError(f"旧可续接执行缺少工具领域 terminal receipt，停止迁移: {identities}")
        inputs = [(row, meta, item) for row, meta, items in chain for item in items if item["type"] == "userMessage"]
        reason = None
        if (not inputs or any(type(item["data"].get("ordinal")) is not int for _, _, item in inputs)
                or [item["data"].get("ordinal") for _, _, item in inputs] != list(range(len(inputs)))):
            reason = "missing_continuous_input_ordinals"
        mapped: list[dict[str, Any]] = []
        used_messages: set[str] = set()
        for row, meta, item in inputs if reason is None else ():
            data = item["data"]
            if not isinstance(data.get("content"), str) or not isinstance(data.get("metadata"), dict):
                raise ValueError(f"旧用户输入格式损坏: {row['id']}/{item['id']}")
            inbound = data["metadata"]
            attachment_ids = inbound.get("attachment_ids", [])
            if not isinstance(attachment_ids, list) or any(not isinstance(ref, str) or not ref for ref in cast(list[object], attachment_ids)):
                raise ValueError(f"旧用户输入 attachment_ids 无效: {item['id']}")
            client = inbound.get("client_message_id")
            # 每项独立匹配真实 handoff；Control metadata 中同名 channel 字段不构成接纳证据。
            evidence = [h for h in handoffs if h["session_key"] == session_id and client
                        and _load(h["metadata_json"]).get("client_message_id") == client]
            if len(evidence) > 1:
                raise ValueError(f"旧用户输入有多个入站身份: {item['id']}")
            if not evidence:
                reason = "missing_independent_channel_receipt"
                break
            evidence = evidence[0]
            accepted_media = _load(evidence["media_json"])
            media = data.get("media", [])
            for values in (accepted_media, media):
                if not isinstance(values, list) or any(not isinstance(value, str) for value in cast(list[object], values)):
                    raise ValueError(f"旧用户输入 media 必须是字符串数组: {item['id']}")
            if accepted_media != media:
                raise ValueError(f"旧用户输入 media 与入站 receipt 冲突: {item['id']}")
            if _load(evidence["metadata_json"]) != inbound:
                raise ValueError(f"旧用户输入 metadata 与入站 receipt 冲突: {item['id']}")
            if evidence["content"] != data["content"] or (
                meta.get("channel") != evidence["channel"] or meta.get("chatId") != evidence["chat_id"]
                or meta.get("sender") != evidence["sender"]
            ):
                raise ValueError(f"旧用户输入与入站 receipt 冲突: {item['id']}")
            candidates = {m.message_id for m, extra in messages.values()
                          if m.session_id == session_id and isinstance(m.body, Input) and extra is not None
                          and extra.get("control_turn_id") == interaction
                          and extra.get("turn_input_ordinal") == data["ordinal"]
                          and extra.get("client_message_id") == client}
            for key in ("message_id", "persisted_user_message_id", "sessionMessageId"):
                if key in inbound:
                    ref = inbound[key]
                    if ref not in messages or messages[ref][0].session_id != session_id or not isinstance(messages[ref][0].body, Input):
                        raise ValueError(f"旧用户输入引用无效: {item['id']}/{ref}")
                    candidates.add(ref)
            if len(candidates) > 1:
                raise ValueError(f"旧用户输入精确引用不一致: {item['id']}")
            if candidates:
                identity = next(iter(candidates))
                if identity in used_messages:
                    raise ValueError(f"旧输入重复映射同一 Message: {identity}")
                used_messages.add(identity)
                message, extra = messages[identity]
                assert isinstance(message.body, Input)
                text = [part.value for part in message.body.parts if part.kind == "text"]
                # 旧持久 owner 允许经入站 metadata 固定的 display_content 替换展示正文。
                display = inbound.get("display_content")
                expected_text = display if isinstance(display, str) else data["content"]
                actual_refs = [ref.artifact_id for ref in log.reader(session_id).attachments(identity)]
                if text != [expected_text] or actual_refs != attachment_ids:
                    raise ValueError(f"旧输入与已落 Message 的正文或附件冲突: {identity}")
                for key, expected in (("client_message_id", client), ("turn_input_ordinal", data["ordinal"]),
                                      ("control_turn_id", interaction)):
                    if extra is not None and key in extra and extra[key] != expected:
                        raise ValueError(f"旧输入引用的 Message 身份冲突: {identity}/{key}")
                if legacy_post_commit_effect(message) != post_commit_effect(inbound):
                    raise ValueError(f"旧输入与已落 Message 的沉淀资格冲突: {identity}")
            if not candidates and data.get("media") and not inbound.get("attachment_ids"):
                reason = "unmapped_legacy_media"
                break
            mapped.append({"record_id": row["id"], "item": item, "handoff_id": evidence["handoff_id"],
                           "message_id": next(iter(candidates), None),
                           "origin": {key: evidence[key] for key in ("channel", "chat_id", "sender")}})
        result.append({"tail_id": tail_id, "session_id": session_id, "reason": reason,
                       "inputs": mapped if reason is None else []})
    return result


def migrate_turn_messages(workspace: Path) -> dict[str, Any] | None:
    """备份、核验、同库追加与复核；旧 turns 永不 UPDATE、DELETE 或 DROP。"""
    path = workspace / "sessions.db"
    if not path.exists():
        return
    # 1. 构造 MessageLog 之前备份；未知结构与坏 JSON 不产生任何权威写入。
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        _check(connection)
        if not _schema(connection, "turns", _TURNS):
            return
        rows = sorted(_rows(connection, "turns"), key=lambda row: (row["created_at"], row["id"]))
        if not rows:
            return
        handoffs = _rows(connection, "inbound_handoffs") if _schema(connection, "inbound_handoffs", _HANDOFFS) else []
        original = _digest(rows)
    backup = workspace / "backups/turn-messages-v1" / uuid4().hex
    _ = backup_sqlite_database(path, backup, migration="20260906_01_turn_messages")
    with closing(MessageLog(path)) as log:
        state = log.owner(_OWNER)
        def commit(tx: OwnerTransaction) -> dict[str, Any]:
            connection = log._connection  # pyright: ignore[reportPrivateUsage]
            current = sorted(_rows(connection, "turns"), key=lambda row: (row["created_at"], row["id"]))
            if _digest(current) != original:
                raise RuntimeError("旧执行在迁移前发生变化")
            receipt = tx.read("manifest")
            if receipt is not None:
                value = cast(dict[str, Any], json_value(receipt.value))
                if receipt.version != 0 or value["turns_sha256"] != original:
                    raise RuntimeError("旧执行迁移 receipt 与源记录不一致")
                for identity, digest in value["messages"]:
                    row = connection.execute("SELECT * FROM messages WHERE id=?", (identity,)).fetchone()
                    if row is None or _digest(dict(row)) != digest:
                        raise RuntimeError(f"旧执行迁移后的 Message 缺失或改变: {identity}")
                _check(connection)
                return value
            sessions = {row[0] for row in connection.execute("SELECT key FROM sessions")}
            if any(row["session_key"] not in sessions for row in rows):
                raise RuntimeError("旧执行引用缺少原 Session，拒绝创建替代会话")
            protected = {name: _digest(_rows(connection, name)) for name in
                         ("turns", "attachments", "message_attachments", "message_embeddings")}
            before_messages = _rows(connection, "messages")
            before_links = _rows(connection, "message_attachments")
            current_handoffs = _rows(connection, "inbound_handoffs") if handoffs else []
            if _digest(current_handoffs) != _digest(handoffs):
                raise RuntimeError("入站 handoff 在迁移前发生变化")
            plans = _plan(rows, handoffs, _messages(log), log)
            # 2. 每条旧行完整归档；quiet/history 没有执行者或学习资格。
            for row in rows:
                writer = log.writer(row["session_key"], author=_OWNER, source="history", body_types=(Output,),
                                    content={"history.record": check_record})
                part = ContentPart("history.record", {"schema": "sessions.turns.v0", "row": row, "sha256": _digest(row)})
                _ = tx.append(writer, "history:turn:" + row["id"], Output((part,), "quiet"))
            mapping: list[dict[str, str]] = []
            pauses: list[str] = []
            for plan in plans:
                if plan["reason"] is not None:
                    continue
                reader = log.reader(plan["session_id"])
                input_seqs: list[int] = []
                for entry in plan["inputs"]:
                    item = entry["item"]
                    identity = entry["message_id"]
                    if identity is None:
                        identity = "history:turn-input:" + entry["record_id"] + ":" + item["id"]
                        data = item["data"]
                        parts = (ContentPart("text", data["content"]), ContentPart("channel.origin", entry["origin"]),
                                 ContentPart("history.turn_input", {"record_id": entry["record_id"], "item_id": item["id"],
                                                                   "metadata": data["metadata"]}),
                                 *(ContentPart("artifact_ref", ref) for ref in data["metadata"].get("attachment_ids", [])))
                        writer = log.writer(reader.session_id, author=_OWNER, source="conversation", body_types=(Input,),
                                            content={"text": check_text, "channel.origin": check_origin,
                                                     "history.turn_input": check_turn_input, "artifact_ref": check_artifact})
                        _ = tx.append(writer, identity, Input(parts))
                    accepted = reader.get(identity)
                    if accepted is None:
                        raise RuntimeError("旧输入映射没有对应的持久 Message")
                    input_seqs.append(accepted.seq)
                    mapping.append({"record_id": entry["record_id"], "item_id": item["id"],
                                    "message_id": identity, "handoff_id": entry["handoff_id"]})
                # 原消息保持 legacy source；迁移 pause 可覆盖已映射的全局旧 seq，不重写 Message。
                cutoff = max(*input_seqs, reader.head(source="conversation"))
                controls = log.writer(reader.session_id, author=_OWNER, source="conversation", body_types=(Control,), content={})
                pause = tx.append(controls, "history:turn-pause:" + plan["tail_id"], Control("pause", cutoff),
                                  expected_source_head=reader.head(source="conversation"))
                pauses.append(pause.message_id)
            # 3. 旧正文、原始 JSON、向量与附件不变；提交 receipt 让 yoyo 落账前崩溃可重入。
            for row in before_messages:
                if dict(connection.execute("SELECT * FROM messages WHERE id=?", (row["id"],)).fetchone()) != row:
                    raise RuntimeError("旧执行迁移改写了已有 Message")
            for name, digest in protected.items():
                if name != "message_attachments" and _digest(_rows(connection, name)) != digest:
                    raise RuntimeError(f"旧执行迁移改写受保护表: {name}")
            after_links = _rows(connection, "message_attachments")
            if any(row not in after_links for row in before_links):
                raise RuntimeError("旧执行迁移改变了原 Message 附件关系")
            _check(connection)
            result: dict[str, Any] = {"turns_sha256": original, "backup": str(backup), "input_mapping": mapping,
                      "pauses": pauses, "unmapped": [{"tail_id": p["tail_id"], "reason": p["reason"]}
                                                    for p in plans if p["reason"] is not None],
                      "messages": [[row["id"], _digest(row)] for row in _rows(connection, "messages")]}
            _ = tx.save("manifest", result, expected_version=None)
            return result
        return state.transact(commit)
