from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Sequence, cast
from uuid import uuid4

from agent.control.errors import TurnNotFoundError, TurnStateTransitionError
from agent.control.models import (
    TurnError,
    TurnItem,
    TurnRecord,
    TurnStatus,
    TurnUsage,
    parse_rfc3339,
)

logger = logging.getLogger(__name__)

_SOURCE_PLAN_DIGEST_RE = re.compile(r"[0-9a-f]{64}")
_ATTACHMENT_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_ATTACHMENT_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_ATTACHMENT_MEDIA_TYPE_RE = re.compile(
    r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+"
)


def validate_message_delivery_id(value: object) -> str:
    """Validate the shared delivery identity stored in Session message extras."""

    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("delivery_id 必须是 1..128 字符串")
    return value


class InteractionDeleteRequiredError(ValueError):
    """要求调用方按完整 interaction 执行原子撤销。"""

    def __init__(self, message_id: str, control_turn_id: str) -> None:
        super().__init__(
            f"message {message_id} 属于 interaction {control_turn_id}，必须整组撤销"
        )
        self.message_id = message_id
        self.control_turn_id = control_turn_id


class SessionAdmissionConflictError(ValueError):
    """拒绝在 session 仍被 active turn 持有时执行破坏性删除。"""

    def __init__(self, session_key: str, *, audit_id: str | None = None) -> None:
        super().__init__(f"session 正在处理消息，暂时不能删除: {session_key}")
        self.session_key = session_key
        self.audit_id = audit_id


@dataclass(frozen=True)
class ChannelIdentityWriteReceipt:
    """记录一次可按提交版本精确撤销的 Channel identity 写入。"""

    channel: str
    identity: str
    chat_id: str
    session_key: str
    committed_created_at: str
    committed_updated_at: str
    committed_last_consolidated: int
    committed_metadata: str
    previous_identity_chat_id: str | None
    previous_identity_updated_at: str | None
    previous_session: tuple[str, str, int, str | None] | None
    previous_lineage_counts: tuple[int, ...]


class SessionCompactionPrepareConflictError(ValueError):
    """拒绝在 source plan 被 durable compaction prepare 锁定时修改消息。"""

    def __init__(
        self,
        session_key: str,
        source_ref: str,
        *,
        audit_id: str | None = None,
    ) -> None:
        super().__init__(
            "session 存在 pending compaction prepare，暂时不能修改消息: "
            f"{session_key}:{source_ref}"
        )
        self.session_key = session_key
        self.source_ref = source_ref
        self.audit_id = audit_id


@dataclass(frozen=True)
class InteractionDeletion:
    """记录一次完整 interaction 撤销及其游标变化。"""

    control_turn_id: str
    session_key: str
    message_ids: tuple[str, ...]
    first_user_message_id: str
    old_last_consolidated: int
    new_last_consolidated: int
    backup_path: str
    audit_id: str | None = None


@dataclass(frozen=True)
class SessionDeleteAudit:
    """记录一次 session 删除命令及其可恢复证据。"""

    audit_id: str
    targets: tuple[str, ...]
    message_ids: tuple[str, ...]
    compactions: tuple[dict[str, Any], ...]
    action_source: str
    cascade: bool
    backup_path: str | None
    started_at: str
    completed_at: str
    result: str
    deleted_count: int
    error: str | None = None


@dataclass(frozen=True)
class SourceMutationAudit:
    """记录一次 canonical source 消息编辑或物理删除。"""

    audit_id: str
    operation: str
    session_key: str
    message_ids: tuple[str, ...]
    action_source: str
    backup_path: str | None
    completed_at: str


@dataclass(frozen=True)
class AttachmentArtifactRecord:
    """描述一个已发布且不可变的附件 artifact。"""

    artifact_id: str
    storage_key: str
    kind: str
    filename: str | None
    media_type: str | None
    size_bytes: int
    sha256: str
    state: str
    created_at: str


@dataclass(frozen=True)
class AttachmentImportRecord:
    """记录一次 attachment 文件发布与 metadata commit 的恢复状态。"""

    artifact_id: str
    storage_key: str
    expected_size_bytes: int
    expected_sha256: str
    phase: str
    created_at: str
    updated_at: str
    error: str | None


@dataclass(frozen=True)
class AttachmentIntegrityReport:
    """汇总 SessionDB attachment authority 的只读完整性证据。"""

    artifact_count: int
    binding_count: int
    bound_message_count: int
    incomplete_import_ids: tuple[str, ...]


@dataclass(frozen=True)
class CompactionPrepare:
    """Durable fence protecting one receipt-before-ledger crash window."""

    session_key: str
    session_created_at: str
    generation: int
    parent_generation: int
    source_ref: str
    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    retained_tail: tuple[dict[str, Any], ...]
    prepared_at: str


@dataclass(frozen=True)
class SessionCompaction:
    """A persisted, immutable model-context checkpoint for one session."""

    session_key: str
    generation: int
    parent_generation: int
    created_at: str
    trigger: str
    summary_format_version: int
    summary: str
    source_ref: str
    source_plan_digest: str
    source_from_seq: int
    consolidated_through_seq: int
    source_message_ids: tuple[str, ...]
    retained_tail: tuple[dict[str, Any], ...]
    model_runtime_id: str
    model: str
    context_window: int
    threshold_tokens: int
    hard_input_tokens: int
    keep_recent_tokens: int
    tokens_before: int
    tokens_after: int
    summary_usage: dict[str, Any]
    invalidated_at: str | None = None
    invalidated_reason: str | None = None

    @property
    def active(self) -> bool:
        return self.invalidated_at is None


@dataclass(frozen=True)
class CompactionHead:
    """Store-owned cursor and monotonic next generation for one session."""

    session_key: str
    parent_generation: int
    next_generation: int


_FTS_CAPABILITY_ERROR_MARKERS = (
    "no such module: fts5",
    "no such tokenizer: trigram",
    "unknown tokenizer: trigram",
)
_MESSAGE_COLUMN_FIELDS = frozenset(
    {"id", "session_key", "seq", "role", "content", "timestamp", "tool_chain"}
)
_RETIRED_ASSISTANT_EXTRA_FIELDS = frozenset({"react_compaction"})
_TURN_TRANSITIONS = {
    TurnStatus.QUEUED: frozenset({TurnStatus.IN_PROGRESS, TurnStatus.CANCELLED}),
    TurnStatus.IN_PROGRESS: frozenset(
        {
            TurnStatus.COMPLETED,
            TurnStatus.INTERRUPTED,
            TurnStatus.FAILED,
            TurnStatus.CANCELLED,
        }
    ),
}


def _resolve_path_text(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.startswith(("http://", "https://")):
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except OSError:
        return ""


def _decode_session_metadata(
    raw: str | bytes | bytearray | None,
    session_key: str,
) -> dict[str, Any]:
    """解析并校验 sessions.metadata 的 JSON object 契约。"""

    metadata = _decode_json_payload(
        raw,
        fallback="{}",
        field="session metadata",
        identifier=session_key,
    )
    if not isinstance(metadata, dict):
        raise ValueError(f"session metadata 必须是 JSON object: {session_key}")
    return cast(dict[str, Any], metadata)


def _decode_message_extra(
    raw: str | bytes | bytearray | None,
    message_id: str,
) -> dict[str, Any]:
    """解析并校验一条消息的 extra JSON object。"""

    extra = _decode_json_payload(
        raw,
        fallback="{}",
        field="message extra",
        identifier=message_id,
    )
    if not isinstance(extra, dict):
        raise ValueError(f"message extra 必须是 JSON object: {message_id}")
    extra_dict = cast(dict[str, Any], extra)
    reserved_fields = _MESSAGE_COLUMN_FIELDS.intersection(extra_dict)
    if reserved_fields:
        fields = ", ".join(sorted(reserved_fields))
        raise ValueError(f"message extra 不得覆盖消息列字段 ({fields}): {message_id}")
    media: object = extra_dict.get("media")
    if "media" in extra_dict and (
        not isinstance(media, list)
        or not all(isinstance(item, str) for item in cast(list[object], media))
    ):
        raise ValueError(f"message media 必须是字符串数组: {message_id}")
    _message_attachment_ids(extra_dict, message_id)
    source_refs: object = extra_dict.get("source_refs")
    if "source_refs" in extra_dict and (
        not isinstance(source_refs, list)
        or not all(isinstance(item, dict) for item in cast(list[object], source_refs))
    ):
        raise ValueError(f"message source_refs 必须是对象数组: {message_id}")
    proactive = extra_dict.get("proactive")
    if "proactive" in extra_dict and not isinstance(proactive, bool):
        raise ValueError(f"message proactive 必须是布尔值: {message_id}")
    if "delivery_id" in extra_dict:
        try:
            _ = validate_message_delivery_id(extra_dict["delivery_id"])
        except ValueError as error:
            raise ValueError(
                f"message delivery_id 必须是 1..128 字符串: {message_id}"
            ) from error
    for field in ("state_summary_tag", "reasoning_content"):
        value = extra_dict.get(field)
        if field in extra_dict and not isinstance(value, str):
            raise ValueError(f"message {field} 必须是字符串: {message_id}")
    if "model_state" in extra_dict:
        _validate_model_state(extra_dict["model_state"], message_id)
    return extra_dict


def _validate_new_message_extra(
    role: object,
    extra: object,
    message_id: str,
) -> None:
    """Reject retired assistant metadata on newly persisted messages."""

    if not isinstance(extra, dict):
        return
    _message_attachment_ids(extra, message_id)
    if role != "assistant":
        return
    retired = _RETIRED_ASSISTANT_EXTRA_FIELDS.intersection(extra)
    if retired:
        fields = ", ".join(sorted(retired))
        raise ValueError(f"assistant extra 字段已退役: {fields}: {message_id}")


def _message_attachment_ids(
    extra: Mapping[str, object],
    message_id: str,
) -> tuple[str, ...]:
    """校验并返回消息声明的有序 artifact identity。"""

    raw_ids = extra.get("attachment_ids")
    if raw_ids is None:
        return ()
    if not isinstance(raw_ids, list):
        raise ValueError(f"message attachment_ids 必须是字符串数组: {message_id}")
    artifact_ids = tuple(raw_ids)
    if any(
        not isinstance(artifact_id, str)
        or _ATTACHMENT_ID_RE.fullmatch(artifact_id) is None
        for artifact_id in artifact_ids
    ):
        raise ValueError(
            f"message attachment_ids 必须是 1..256 字符安全 identity: {message_id}"
        )
    normalized = cast(tuple[str, ...], artifact_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"message attachment_ids 不得重复: {message_id}")
    return normalized


def _message_attachment_binding_rows(
    *,
    message_id: str,
    role: object,
    extra: Mapping[str, object],
) -> list[tuple[str, int, str, str]]:
    """构造一条消息的有序 attachment binding rows。"""

    artifact_ids = _message_attachment_ids(extra, message_id)
    if not artifact_ids:
        return []
    if role == "user":
        direction = "inbound"
    elif role == "assistant":
        direction = "outbound"
    else:
        raise ValueError(f"只有 user/assistant message 可以绑定 attachment: {message_id}")
    return [
        (message_id, ordinal, artifact_id, direction)
        for ordinal, artifact_id in enumerate(artifact_ids)
    ]


def _attachment_storage_key(artifact_id: str, storage_key: str) -> PurePosixPath:
    """校验 Core artifact 的唯一 workspace-relative storage identity。"""

    storage_path = PurePosixPath(storage_key)
    if storage_path.parts != (
        "uploads",
        "artifacts",
        f"{artifact_id}.bin",
    ):
        raise ValueError(
            "attachment storage_key 必须是 uploads/artifacts/<artifact_id>.bin"
        )
    return storage_path


def _message_control_turn_id(
    raw: str | bytes | bytearray | None,
    message_id: str,
) -> str | None:
    extra = _decode_message_extra(raw, message_id)
    value = extra.get("control_turn_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"message control_turn_id 必须是非空字符串: {message_id}")
    return value


def _validate_deletable_interaction(
    control_turn_id: str,
    rows: list[sqlite3.Row],
) -> str:
    """校验显式 transcript 完整性并返回首条 user message ID。"""

    # 1. 每一行都必须属于目标 interaction，且只能有一个 terminal assistant。
    decoded = [
        (row, _decode_message_extra(row["extra"], str(row["id"]))) for row in rows
    ]
    if any(extra.get("control_turn_id") != control_turn_id for _, extra in decoded):
        raise ValueError(f"interaction transcript 身份不一致: {control_turn_id}")
    users = [(row, extra) for row, extra in decoded if row["role"] == "user"]
    terminals = [
        (row, extra)
        for row, extra in decoded
        if row["role"] == "assistant" and extra.get("turn_terminal") is True
    ]
    if not users or len(terminals) != 1 or len(decoded) != len(users) + 1:
        raise ValueError(f"interaction transcript 结构无效: {control_turn_id}")

    # 2. ordinal、input count 和 assistant 顺序必须保持完整提交合同。
    raw_ordinals = [extra.get("turn_input_ordinal") for _, extra in users]
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in raw_ordinals
    ):
        raise ValueError(f"interaction input ordinal 不连续: {control_turn_id}")
    ordinals = cast(list[int], raw_ordinals)
    if sorted(ordinals) != list(range(len(users))):
        raise ValueError(f"interaction input ordinal 不连续: {control_turn_id}")
    assistant, assistant_extra = terminals[0]
    if assistant_extra.get("turn_input_count") != len(users):
        raise ValueError(f"interaction input count 不匹配: {control_turn_id}")
    if any(int(row["seq"]) >= int(assistant["seq"]) for row, _ in users):
        raise ValueError(f"interaction assistant 顺序无效: {control_turn_id}")
    first_user = next(row for row, extra in users if extra["turn_input_ordinal"] == 0)
    return str(first_user["id"])


def _latest_completed_interaction_id(rows: list[sqlite3.Row]) -> str | None:
    """返回最后一个结构完整的 terminal interaction identity。"""

    for row in reversed(rows):
        if row["role"] != "assistant":
            continue
        message_id = str(row["id"])
        extra = _decode_message_extra(row["extra"], message_id)
        if extra.get("turn_terminal") is not True:
            continue
        control_turn_id = _message_control_turn_id(row["extra"], message_id)
        if control_turn_id is None:
            continue
        interaction_rows = [
            candidate
            for candidate in rows
            if _message_control_turn_id(
                candidate["extra"],
                str(candidate["id"]),
            )
            == control_turn_id
        ]
        _ = _validate_deletable_interaction(control_turn_id, interaction_rows)
        return control_turn_id
    return None


def _validate_model_state(value: object, message_id: str) -> None:
    """在数据库边界校验 Responses continuation state。"""
    if not isinstance(value, dict):
        raise ValueError(f"message model_state 必须是 JSON object: {message_id}")
    state = cast(dict[str, object], value)
    if state.get("schema_version") != 1:
        raise ValueError(f"message model_state schema_version 无效: {message_id}")
    for field in ("runtime_id", "transport", "model"):
        if not isinstance(state.get(field), str) or not state[field]:
            raise ValueError(
                f"message model_state.{field} 必须是非空字符串: {message_id}"
            )
    items = state.get("items")
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"message model_state.items 必须是对象数组: {message_id}")
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 2 * 1024 * 1024:
        raise ValueError(f"message model_state 超过 2 MiB: {message_id}")


def _decode_json_payload(
    raw: str | bytes | bytearray | None,
    *,
    fallback: str,
    field: str,
    identifier: str,
) -> object:
    """在 SQLite 反序列化边界统一转换 JSON 损坏错误。"""

    try:
        return json.loads(fallback if raw is None else raw)
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
        raise ValueError(f"{field} JSON 损坏: {identifier}") from exc


def _decode_turn_input(raw: object, turn_id: str) -> tuple[str, dict[str, Any]]:
    """解析并校验 turn 输入及其 metadata。"""
    payload = _decode_json_payload(
        cast(str | bytes | bytearray | None, raw),
        fallback="{}",
        field="turn input",
        identifier=turn_id,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"turn input 必须是 JSON object: {turn_id}")
    data = cast(dict[str, object], payload)
    input_text = data.get("input")
    metadata = data.get("metadata")
    if not isinstance(input_text, str):
        raise ValueError(f"turn input.input 必须是字符串: {turn_id}")
    if not isinstance(metadata, dict):
        raise ValueError(f"turn input.metadata 必须是 JSON object: {turn_id}")
    return input_text, cast(dict[str, Any], metadata)


def _decode_turn_items(raw: object, turn_id: str) -> list[TurnItem]:
    """解析并校验 turn item 数组。"""
    payload = _decode_json_payload(
        cast(str | bytes | bytearray | None, raw),
        fallback="[]",
        field="turn items",
        identifier=turn_id,
    )
    if not isinstance(payload, list):
        raise ValueError(f"turn items 必须是 JSON array: {turn_id}")
    return [TurnItem.from_dict(item) for item in cast(list[object], payload)]


def _decode_turn_usage(raw: object, turn_id: str) -> TurnUsage | None:
    if raw is None:
        return None
    payload = _decode_json_payload(
        cast(str | bytes | bytearray, raw),
        fallback="null",
        field="turn usage",
        identifier=turn_id,
    )
    return TurnUsage.from_dict(payload)


def _decode_turn_error(raw: object, turn_id: str) -> TurnError | None:
    if raw is None:
        return None
    payload = _decode_json_payload(
        cast(str | bytes | bytearray, raw),
        fallback="null",
        field="turn error",
        identifier=turn_id,
    )
    return TurnError.from_dict(payload)


def _decode_required_turn_time(raw: object, field_name: str, turn_id: str) -> datetime:
    value = parse_rfc3339(raw, field_name)
    if value is None:
        raise ValueError(f"turn {field_name} 不能为空: {turn_id}")
    return value


def _validate_source_plan_digest(value: object) -> str:
    """Validate the canonical SHA-256 source-plan identity at the write boundary."""

    if not isinstance(value, str) or _SOURCE_PLAN_DIGEST_RE.fullmatch(value) is None:
        raise ValueError("compaction source_plan_digest 必须是 64 位小写 SHA-256")
    return value


def _validate_source_mutation_digest(value: object) -> str:
    """Validate the authorized source-mutation snapshot identity."""

    if not isinstance(value, str) or _SOURCE_PLAN_DIGEST_RE.fullmatch(value) is None:
        raise ValueError("compaction source_mutation_digest 必须是 64 位小写 SHA-256")
    return value


def _required_source_plan_digest(value: object, *, identifier: str) -> str:
    """Decode one persisted source-plan digest without normalizing corrupted state."""

    try:
        return _validate_source_plan_digest(value)
    except ValueError as exc:
        raise ValueError(f"compaction source_plan_digest 无效: {identifier}") from exc


def _decode_message_tool_chain(
    raw: str | bytes | bytearray,
    message_id: str,
) -> list[dict[str, object]]:
    """解析并校验消息工具链的容器结构。"""

    tool_chain = _decode_json_payload(
        raw,
        fallback="[]",
        field="message tool_chain",
        identifier=message_id,
    )
    if not isinstance(tool_chain, list):
        raise ValueError(f"message tool_chain 必须是 JSON array: {message_id}")

    # 1. 校验每个工具轮次和调用容器。
    raw_groups = cast(list[object], tool_chain)
    for group_index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, dict):
            raise ValueError(
                f"message tool_chain[{group_index}] 必须是 JSON object: {message_id}"
            )
        group = cast(dict[str, object], raw_group)
        raw_calls = group.get("calls")
        if not isinstance(raw_calls, list):
            raise ValueError(
                f"message tool_chain[{group_index}].calls 必须是 JSON array: {message_id}"
            )
        # 2. 校验调用字段；稀疏工具链仍可用于消息查询展示。
        calls = cast(list[object], raw_calls)
        for call_index, raw_call in enumerate(calls):
            if not isinstance(raw_call, dict):
                raise ValueError(
                    "message tool_chain[{}].calls[{}] 必须是 JSON object: {}".format(
                        group_index, call_index, message_id
                    )
                )
            call = cast(dict[str, object], raw_call)
            for field in ("call_id", "name"):
                if field in call and not isinstance(call[field], str):
                    raise ValueError(
                        "message tool_chain[{}].calls[{}].{} 必须是字符串: {}".format(
                            group_index, call_index, field, message_id
                        )
                    )
            arguments = call.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                raise ValueError(
                    "message tool_chain[{}].calls[{}].arguments 必须是 JSON object: {}".format(
                        group_index, call_index, message_id
                    )
                )
            result = call.get("result")
            if result is not None and not isinstance(result, str):
                raise ValueError(
                    "message tool_chain[{}].calls[{}].result 必须是字符串: {}".format(
                        group_index, call_index, message_id
                    )
                )

        # 3. 可选的展示字段若存在，也必须保持字符串契约。
        for field in ("text", "reasoning_content"):
            value = group.get(field)
            if value is not None and not isinstance(value, str):
                raise ValueError(
                    f"message tool_chain[{group_index}].{field} 必须是字符串: {message_id}"
                )
        if "model_state" in group:
            _validate_model_state(group["model_state"], message_id)
    return cast(list[dict[str, object]], tool_chain)


class SessionStore:
    """SQLite-backed store for session metadata and messages."""

    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._closed = False
        self._has_fts = False
        self._conn.execute("PRAGMA foreign_keys = ON")
        foreign_keys = self._conn.execute("PRAGMA foreign_keys").fetchone()
        if foreign_keys is None or int(foreign_keys[0]) != 1:
            raise RuntimeError("SessionStore 无法启用 SQLite foreign key enforcement")
        self._init_schema()

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except sqlite3.Error as cleanup_error:
                logger.warning(
                    "SessionStore 析构关闭失败 db=%s err=%s",
                    self.db_path,
                    cleanup_error,
                )

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    key               TEXT PRIMARY KEY,
                    created_at        TEXT NOT NULL,
                    updated_at        TEXT NOT NULL,
                    last_consolidated INTEGER NOT NULL DEFAULT 0,
                    metadata          TEXT
                )
                """)
            self._ensure_session_columns()
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_admissions (
                    admission_id TEXT PRIMARY KEY,
                    session_key  TEXT NOT NULL,
                    created_at   TEXT NOT NULL
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS inbound_handoffs (
                    handoff_id  TEXT PRIMARY KEY,
                    dedupe_key  TEXT UNIQUE,
                    channel     TEXT NOT NULL,
                    sender      TEXT NOT NULL,
                    chat_id     TEXT NOT NULL,
                    session_key TEXT NOT NULL,
                    content     TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    media_json  TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_inbound_handoffs_session
                ON inbound_handoffs(session_key, created_at)
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_admissions_session
                ON session_admissions(session_key)
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_identities (
                    channel    TEXT NOT NULL,
                    identity   TEXT NOT NULL,
                    chat_id    TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(channel, identity)
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS channel_identity_migrations (
                    channel     TEXT PRIMARY KEY,
                    migrated_at TEXT NOT NULL
                )
                """)
            # SessionStore owns this fresh runtime audit schema; it is not a
            # user-data migration and is created before any management command.
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_delete_audits (
                    audit_id      TEXT PRIMARY KEY,
                    targets_json  TEXT NOT NULL,
                    message_ids_json TEXT NOT NULL,
                    compactions_json TEXT NOT NULL,
                    action_source TEXT NOT NULL,
                    cascade       INTEGER NOT NULL CHECK (cascade IN (0, 1)),
                    backup_path   TEXT,
                    started_at    TEXT NOT NULL,
                    completed_at  TEXT NOT NULL,
                    result        TEXT NOT NULL,
                    deleted_count INTEGER NOT NULL,
                    error         TEXT
                )
                """)
            audit_columns = {
                str(row["name"])
                for row in self._conn.execute(
                    "PRAGMA table_info(session_delete_audits)"
                ).fetchall()
            }
            if "message_ids_json" not in audit_columns:
                self._conn.execute(
                    "ALTER TABLE session_delete_audits ADD COLUMN "
                    "message_ids_json TEXT NOT NULL DEFAULT '[]'"
                )
            if "compactions_json" not in audit_columns:
                self._conn.execute(
                    "ALTER TABLE session_delete_audits ADD COLUMN "
                    "compactions_json TEXT NOT NULL DEFAULT '[]'"
                )
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_delete_audits_time
                ON session_delete_audits(completed_at, audit_id)
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_source_mutation_audits (
                    audit_id      TEXT PRIMARY KEY,
                    operation     TEXT NOT NULL,
                    session_key   TEXT NOT NULL,
                    message_ids_json TEXT NOT NULL,
                    action_source TEXT NOT NULL,
                    backup_path   TEXT,
                    completed_at  TEXT NOT NULL
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source_mutation_audits_lookup
                ON session_source_mutation_audits(session_key, completed_at, audit_id)
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_compaction_prepares (
                    session_key TEXT NOT NULL,
                    session_created_at TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    parent_generation INTEGER NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_from_seq INTEGER NOT NULL,
                    consolidated_through_seq INTEGER NOT NULL,
                    source_message_ids_json TEXT NOT NULL,
                    retained_tail_json TEXT NOT NULL,
                    prepared_at TEXT NOT NULL,
                    PRIMARY KEY (session_key, generation),
                    UNIQUE (session_key, source_ref)
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_compaction_prepares_ref
                ON session_compaction_prepares(session_key, source_ref)
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id          TEXT PRIMARY KEY,
                    session_key TEXT NOT NULL,
                    seq         INTEGER NOT NULL,
                    role        TEXT NOT NULL,
                    content     TEXT,
                    tool_chain  TEXT,
                    extra       TEXT,
                    ts          TEXT NOT NULL,
                    UNIQUE (session_key, seq)
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS attachments (
                    artifact_id TEXT PRIMARY KEY,
                    storage_key TEXT NOT NULL UNIQUE,
                    kind TEXT NOT NULL CHECK (kind IN ('image', 'file')),
                    filename TEXT,
                    media_type TEXT,
                    size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
                    sha256 TEXT NOT NULL CHECK (
                        length(sha256) = 64
                        AND sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                    state TEXT NOT NULL CHECK (state = 'ready'),
                    created_at TEXT NOT NULL
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS attachment_imports (
                    artifact_id TEXT PRIMARY KEY,
                    storage_key TEXT NOT NULL UNIQUE,
                    expected_size_bytes INTEGER NOT NULL
                        CHECK (expected_size_bytes >= 0),
                    expected_sha256 TEXT NOT NULL CHECK (
                        length(expected_sha256) = 64
                        AND expected_sha256 NOT GLOB '*[^0-9a-f]*'
                    ),
                    phase TEXT NOT NULL CHECK (
                        phase IN (
                            'prepared', 'file_published', 'artifact_committed'
                        )
                    ),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT
                )
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS message_attachments (
                    message_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
                    artifact_id TEXT NOT NULL,
                    direction TEXT NOT NULL
                        CHECK (direction IN ('inbound', 'outbound')),
                    PRIMARY KEY (message_id, ordinal),
                    FOREIGN KEY (message_id)
                        REFERENCES messages(id) ON DELETE CASCADE,
                    FOREIGN KEY (artifact_id)
                        REFERENCES attachments(artifact_id)
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_message_attachments_artifact
                ON message_attachments(artifact_id, message_id, ordinal)
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS session_compactions (
                    session_key TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    parent_generation INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    summary_format_version INTEGER NOT NULL,
                    summary TEXT NOT NULL,
                    source_ref TEXT NOT NULL,
                    source_plan_digest TEXT NOT NULL
                        CHECK (
                            length(source_plan_digest) = 64
                            AND source_plan_digest NOT GLOB '*[^0-9a-f]*'
                        ),
                    source_from_seq INTEGER NOT NULL,
                    consolidated_through_seq INTEGER NOT NULL,
                    source_message_ids_json TEXT NOT NULL,
                    retained_tail_json TEXT NOT NULL,
                    model_runtime_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    context_window INTEGER NOT NULL,
                    threshold_tokens INTEGER NOT NULL,
                    hard_input_tokens INTEGER NOT NULL,
                    keep_recent_tokens INTEGER NOT NULL,
                    tokens_before INTEGER NOT NULL,
                    tokens_after INTEGER NOT NULL,
                    summary_usage_json TEXT NOT NULL,
                    invalidated_at TEXT,
                    invalidated_reason TEXT,
                    PRIMARY KEY (session_key, generation),
                    UNIQUE (session_key, source_ref)
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_compactions_active
                ON session_compactions(session_key, invalidated_at, generation)
                """)
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id             TEXT PRIMARY KEY,
                    session_key    TEXT NOT NULL,
                    status         TEXT NOT NULL,
                    input_json     TEXT NOT NULL,
                    items_json     TEXT NOT NULL,
                    usage_json     TEXT,
                    error_json     TEXT,
                    final_response TEXT,
                    created_at     TEXT NOT NULL,
                    started_at     TEXT,
                    completed_at   TEXT
                )
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session_created
                ON turns(session_key, created_at, id)
                """)
            self._conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_status
                ON turns(status)
                """)
            self._ensure_next_seq_values()
            self._ensure_fts()
            self._conn.commit()

    def reserve_inbound_handoff(
        self,
        *,
        handoff_id: str,
        dedupe_key: str | None,
        channel: str,
        sender: str,
        chat_id: str,
        session_key: str,
        content: str,
        timestamp: str,
        media_json: str,
        metadata_json: str,
        created_at: str,
    ) -> tuple[str, bool]:
        """Durably reserve one inbound message before exposing it to MessageBus."""

        fields = (
            handoff_id,
            dedupe_key,
            channel,
            sender,
            chat_id,
            session_key,
            content,
            timestamp,
            media_json,
            metadata_json,
            created_at,
        )
        if not all(
            isinstance(value, str) and value for value in fields if value is not None
        ):
            raise ValueError("inbound handoff fields must be non-empty strings")
        identity = {
            "dedupe_key": dedupe_key,
            "channel": channel,
            "sender": sender,
            "chat_id": chat_id,
            "session_key": session_key,
            "content": content,
            "timestamp": timestamp,
            "media_json": media_json,
            "metadata_json": metadata_json,
        }
        stable_identity = {
            key: value for key, value in identity.items() if key != "timestamp"
        }

        def validate_existing(row: sqlite3.Row, *, include_timestamp: bool) -> None:
            expected_identity = identity if include_timestamp else stable_identity
            for column, expected in expected_identity.items():
                if row[column] != expected:
                    raise RuntimeError(
                        "inbound handoff identity conflict: "
                        f"handoff_id={handoff_id} field={column}"
                    )

        with self._lock:
            existing_by_id = self._conn.execute(
                "SELECT * FROM inbound_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if dedupe_key is not None:
                existing_by_dedupe = self._conn.execute(
                    "SELECT * FROM inbound_handoffs WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
            else:
                existing_by_dedupe = None
            if (
                existing_by_id is not None
                and existing_by_dedupe is not None
                and existing_by_id["handoff_id"] != existing_by_dedupe["handoff_id"]
            ):
                raise RuntimeError(
                    "inbound handoff identity conflict: handoff_id and dedupe_key differ"
                )
            existing = existing_by_id or existing_by_dedupe
            if existing is not None:
                validate_existing(
                    existing, include_timestamp=existing_by_id is not None
                )
                return str(existing["handoff_id"]), False
            cursor = self._conn.execute(
                """
                INSERT INTO inbound_handoffs(
                    handoff_id, dedupe_key, channel, sender, chat_id,
                    session_key, content, timestamp, media_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                fields,
            )
            row = self._conn.execute(
                "SELECT * FROM inbound_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            ).fetchone()
            if row is None and dedupe_key is not None:
                row = self._conn.execute(
                    "SELECT * FROM inbound_handoffs WHERE dedupe_key = ?",
                    (dedupe_key,),
                ).fetchone()
            if row is None:
                self._conn.rollback()
                raise RuntimeError(f"inbound handoff disappeared: {handoff_id}")
            try:
                validate_existing(
                    row,
                    include_timestamp=row["handoff_id"] == handoff_id,
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return str(row["handoff_id"]), cursor.rowcount == 1

    def list_inbound_handoffs(
        self,
        *,
        limit: int | None = None,
    ) -> list[dict[str, str | None]]:
        """按 durable 到达顺序读取有限页的 pending inbound handoff。"""

        if limit is not None and (
            not isinstance(limit, int) or isinstance(limit, bool) or limit < 1
        ):
            raise ValueError("inbound handoff limit 必须是正整数")
        limit_sql = "" if limit is None else " LIMIT ?"

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT handoff_id, dedupe_key, channel, sender, chat_id,
                       session_key, content, timestamp, media_json,
                       metadata_json, created_at
                FROM inbound_handoffs
                ORDER BY created_at ASC, handoff_id ASC
                """ + limit_sql,
                () if limit is None else (limit,),
            ).fetchall()
        return [{key: cast(str | None, row[key]) for key in row.keys()} for row in rows]

    def has_inbound_handoff(
        self,
        *,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        """Check whether a client message still owns an uncompleted handoff."""

        dedupe_key = f"{session_key}:{client_message_id}"
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM inbound_handoffs WHERE dedupe_key = ?",
                (dedupe_key,),
            ).fetchone()
        return row is not None

    def complete_inbound_handoff(self, handoff_id: str) -> None:
        """Release a handoff only after its worker has finished processing."""

        if not isinstance(handoff_id, str) or not handoff_id:
            raise ValueError("handoff_id must be a non-empty string")
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM inbound_handoffs WHERE handoff_id = ?",
                (handoff_id,),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise RuntimeError(f"inbound handoff not found: {handoff_id}")
            self._conn.commit()

    def _ensure_session_columns(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(sessions)").fetchall()
        existing = {str(row["name"]) for row in rows}
        if "last_user_at" not in existing:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN last_user_at TEXT")
        if "last_proactive_at" not in existing:
            self._conn.execute("ALTER TABLE sessions ADD COLUMN last_proactive_at TEXT")
        if "next_seq" not in existing:
            self._conn.execute(
                "ALTER TABLE sessions ADD COLUMN next_seq INTEGER NOT NULL DEFAULT 0"
            )

    def _ensure_next_seq_values(self) -> None:
        rows = self._conn.execute("SELECT key, next_seq FROM sessions").fetchall()
        for row in rows:
            session_key = str(row["key"])
            current = int(row["next_seq"] or 0)
            seq_row = self._conn.execute(
                "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            required = int((seq_row["next_seq"] if seq_row else 0) or 0)
            if current < required:
                self._conn.execute(
                    "UPDATE sessions SET next_seq = ? WHERE key = ?",
                    (required, session_key),
                )

    def _ensure_fts(self) -> None:
        """确保全文索引可用，并仅在创建或修复时重建已有消息。"""

        needs_rebuild = False
        try:
            # 1. 发现旧索引或缺失索引时，准备一次性重建。
            existing = self._conn.execute(
                "SELECT name, sql FROM sqlite_master "
                "WHERE type='table' AND name='messages_fts'"
            ).fetchone()
            needs_rebuild = existing is None
            if existing:
                table_sql = "".join(str(existing["sql"] or "").split()).lower()
                is_trigram = "tokenize='trigram'" in table_sql
                if not is_trigram:
                    self._conn.execute("DROP TABLE IF EXISTS messages_fts")
                    for trig in ("messages_ai", "messages_ad", "messages_au"):
                        self._conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
                    needs_rebuild = True

                trigger_rows = self._conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'trigger' AND name IN (?, ?, ?)",
                    ("messages_ai", "messages_ad", "messages_au"),
                ).fetchall()
                needs_rebuild = needs_rebuild or len(trigger_rows) < 3

            # 2. 确保索引和消息写入触发器存在。
            self._conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    content,
                    content='messages',
                    content_rowid='rowid',
                    tokenize='trigram'
                )
                """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                END
                """)
            self._conn.execute("""
                CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                    INSERT INTO messages_fts(messages_fts, rowid, content)
                    VALUES('delete', old.rowid, old.content);
                    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
                END
                """)
            # 3. 正常重启依赖触发器增量维护，避免重复扫描整张 messages 表。
            if needs_rebuild:
                self._conn.execute(
                    "INSERT INTO messages_fts(messages_fts) VALUES('rebuild')"
                )
            self._conn.commit()
            self._has_fts = True
        except sqlite3.OperationalError as exc:
            if not any(
                marker in str(exc).lower() for marker in _FTS_CAPABILITY_ERROR_MARKERS
            ):
                raise
            logger.warning(
                "SQLite FTS5/trigram 不可用，已禁用 session 全文检索: %s", exc
            )
            self._has_fts = False

    def _has_message_embeddings_locked(self) -> bool:
        row = self._conn.execute("""
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'message_embeddings'
            """).fetchone()
        return row is not None

    def _delete_message_embeddings_locked(self, message_ids: list[str]) -> None:
        if not message_ids or not self._has_message_embeddings_locked():
            return
        placeholders = ",".join("?" for _ in message_ids)
        self._conn.execute(
            f"DELETE FROM message_embeddings WHERE message_id IN ({placeholders})",
            tuple(message_ids),
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()

    def session_exists(self, key: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM sessions WHERE key = ?", (key,)
            ).fetchone()
        return row is not None

    def upsert_session(
        self,
        key: str,
        *,
        created_at: str,
        updated_at: str,
        metadata: dict[str, Any],
    ) -> None:
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                VALUES (?, ?, ?, 0, ?)
                ON CONFLICT(key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    metadata = excluded.metadata
                """,
                (key, created_at, updated_at, payload),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_compaction_prepare(row: sqlite3.Row) -> CompactionPrepare:
        """Decode one durable compaction prepare fence at the SQLite boundary."""

        # 1. Validate scalar identity before any normalization or JSON decoding.
        scalar_strings = {
            "session_key": row["session_key"],
            "session_created_at": row["session_created_at"],
            "source_ref": row["source_ref"],
            "prepared_at": row["prepared_at"],
        }
        if any(not isinstance(value, str) for value in scalar_strings.values()):
            raise ValueError("compaction prepare identity 字段类型无效")
        scalar_ints = {
            "generation": row["generation"],
            "parent_generation": row["parent_generation"],
            "source_from_seq": row["source_from_seq"],
            "consolidated_through_seq": row["consolidated_through_seq"],
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in scalar_ints.values()
        ):
            raise ValueError("compaction prepare numeric identity 字段无效")

        # 2. JSON columns must remain text/blob; NULL must not enter fallback decoding.
        raw_source_ids = row["source_message_ids_json"]
        raw_retained_tail = row["retained_tail_json"]
        json_types = (str, bytes, bytearray)
        if not isinstance(raw_source_ids, json_types):
            raise ValueError("compaction prepare source_message_ids_json 类型无效")
        if not isinstance(raw_retained_tail, json_types):
            raise ValueError("compaction prepare retained_tail_json 类型无效")

        # 3. Decode JSON payloads at the SQLite trust boundary.
        identifier = f"{scalar_strings['session_key']}:{scalar_ints['generation']}"
        source_ids = _decode_json_payload(
            raw_source_ids,
            fallback="[]",
            field="compaction prepare source_message_ids",
            identifier=identifier,
        )
        retained_tail = _decode_json_payload(
            raw_retained_tail,
            fallback="[]",
            field="compaction prepare retained_tail",
            identifier=identifier,
        )
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) and item for item in source_ids
        ):
            raise ValueError("compaction prepare source_message_ids 无效")
        if not isinstance(retained_tail, list) or not all(
            isinstance(item, dict) for item in retained_tail
        ):
            raise ValueError("compaction prepare retained_tail 无效")

        # 4. Reuse the write-boundary contract for source ids/tail and validate timestamp.
        SessionStore._validate_prepare_payload(
            session_key=scalar_strings["session_key"],
            session_created_at=scalar_strings["session_created_at"],
            generation=scalar_ints["generation"],
            parent_generation=scalar_ints["parent_generation"],
            source_ref=scalar_strings["source_ref"],
            source_from_seq=scalar_ints["source_from_seq"],
            consolidated_through_seq=scalar_ints["consolidated_through_seq"],
            source_message_ids=tuple(cast(str, item) for item in source_ids),
            retained_tail=tuple(cast(dict[str, Any], item) for item in retained_tail),
        )
        try:
            prepared_at = datetime.fromisoformat(scalar_strings["prepared_at"])
        except ValueError as exc:
            raise ValueError("compaction prepare prepared_at 无效") from exc
        if prepared_at.tzinfo is None:
            raise ValueError("compaction prepare prepared_at 必须包含时区")
        return CompactionPrepare(
            session_key=scalar_strings["session_key"],
            session_created_at=scalar_strings["session_created_at"],
            generation=scalar_ints["generation"],
            parent_generation=scalar_ints["parent_generation"],
            source_ref=scalar_strings["source_ref"],
            source_from_seq=scalar_ints["source_from_seq"],
            consolidated_through_seq=scalar_ints["consolidated_through_seq"],
            source_message_ids=tuple(cast(str, item) for item in source_ids),
            retained_tail=tuple(cast(dict[str, Any], item) for item in retained_tail),
            prepared_at=scalar_strings["prepared_at"],
        )

    @staticmethod
    def _same_compaction_prepare_identity(
        left: CompactionPrepare,
        right: CompactionPrepare,
    ) -> bool:
        """Compare replayable prepare identity while allowing a new attempt timestamp."""

        return (
            left.session_key == right.session_key
            and left.session_created_at == right.session_created_at
            and left.generation == right.generation
            and left.parent_generation == right.parent_generation
            and left.source_ref == right.source_ref
            and left.source_from_seq == right.source_from_seq
            and left.consolidated_through_seq == right.consolidated_through_seq
            and left.source_message_ids == right.source_message_ids
            and left.retained_tail == right.retained_tail
        )

    @staticmethod
    def _validate_prepare_payload(
        *,
        session_key: str,
        session_created_at: str,
        generation: int,
        parent_generation: int,
        source_ref: str,
        source_from_seq: int,
        consolidated_through_seq: int,
        source_message_ids: list[str] | tuple[str, ...],
        retained_tail: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> None:
        """Validate the durable prepare identity before acquiring the write lock."""

        if (
            not session_key.strip()
            or not session_created_at.strip()
            or not source_ref.strip()
        ):
            raise ValueError("compaction prepare identity 不能为空")
        if (
            not isinstance(generation, int)
            or isinstance(generation, bool)
            or generation < 1
        ):
            raise ValueError("compaction prepare generation 无效")
        if (
            not isinstance(parent_generation, int)
            or isinstance(parent_generation, bool)
            or parent_generation < 0
        ):
            raise ValueError("compaction prepare parent_generation 无效")
        if (
            not isinstance(source_from_seq, int)
            or isinstance(source_from_seq, bool)
            or not isinstance(consolidated_through_seq, int)
            or isinstance(consolidated_through_seq, bool)
            or source_from_seq < 0
            or consolidated_through_seq < source_from_seq
        ):
            raise ValueError("compaction prepare source seq 边界无效")
        if not source_message_ids or any(
            not isinstance(item, str) or not item.strip() for item in source_message_ids
        ):
            raise ValueError("compaction prepare source_message_ids 无效")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or not isinstance(item.get("seq"), int)
            or isinstance(item.get("seq"), bool)
            or not isinstance(item.get("message"), dict)
            or not isinstance(item.get("unit_ref"), str)
            or not item["unit_ref"]
            for item in retained_tail
        ):
            raise ValueError("compaction prepare retained_tail 无效")

    def _validate_compaction_prepare_locked(
        self,
        prepared: CompactionPrepare,
        *,
        source_mutation_digest: str | None = None,
    ) -> CompactionPrepare | None:
        """Validate a pending fence against the current session head and source rows."""

        # 1. incarnation、cursor 和 generation 必须仍然指向同一个 session head。
        session_row = self._conn.execute(
            "SELECT created_at, last_consolidated FROM sessions WHERE key = ?",
            (prepared.session_key,),
        ).fetchone()
        if session_row is None:
            raise KeyError(f"session 不存在: {prepared.session_key}")
        if str(session_row["created_at"]) != prepared.session_created_at:
            raise ValueError("compaction prepare session incarnation 冲突")
        current_cursor = int(session_row["last_consolidated"] or 0)
        if prepared.parent_generation != current_cursor:
            raise ValueError("compaction prepare parent_generation 与 cursor 冲突")
        max_row = self._conn.execute(
            "SELECT COALESCE(MAX(generation), 0) AS generation "
            "FROM session_compactions WHERE session_key = ?",
            (prepared.session_key,),
        ).fetchone()
        max_generation = int(max_row["generation"] if max_row else 0)
        if prepared.generation != max_generation + 1:
            raise ValueError("compaction prepare generation 不匹配 ledger head")

        # 2. source snapshot 必须仍由同一批 canonical SessionDB rows 支撑。
        if source_mutation_digest is not None:
            actual_digest = self._source_mutation_digest_locked(
                prepared.session_key,
                self._compaction_source_ids(
                    prepared.source_message_ids,
                    prepared.retained_tail,
                ),
            )
            if actual_digest != source_mutation_digest:
                raise RuntimeError("compaction source snapshot 在 prepare 前发生变化")

        # 3. source plan 必须仍由 canonical SessionDB rows 完整支撑。
        self._validate_compaction_provenance_locked(
            prepared.session_key,
            source_message_ids=prepared.source_message_ids,
            retained_tail=prepared.retained_tail,
            source_from_seq=prepared.source_from_seq,
            consolidated_through_seq=prepared.consolidated_through_seq,
        )

        # 4. 重试只允许复用完全相同的 prepare identity。
        existing = self._conn.execute(
            "SELECT * FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepared.session_key, prepared.generation),
        ).fetchone()
        if existing is None:
            return None
        existing_value = self._row_to_compaction_prepare(existing)
        if not self._same_compaction_prepare_identity(existing_value, prepared):
            raise ValueError("compaction prepare identity 冲突")
        return existing_value

    def _insert_compaction_prepare_locked(self, prepared: CompactionPrepare) -> None:
        """Insert a validated pending fence while the SQLite write transaction is held."""

        self._conn.execute(
            "INSERT INTO session_compaction_prepares("
            "session_key, session_created_at, generation, parent_generation, "
            "source_ref, source_from_seq, consolidated_through_seq, "
            "source_message_ids_json, retained_tail_json, prepared_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                prepared.session_key,
                prepared.session_created_at,
                prepared.generation,
                prepared.parent_generation,
                prepared.source_ref,
                prepared.source_from_seq,
                prepared.consolidated_through_seq,
                json.dumps(list(prepared.source_message_ids), ensure_ascii=False),
                json.dumps(list(prepared.retained_tail), ensure_ascii=False),
                prepared.prepared_at,
            ),
        )

    def prepare_compaction(
        self,
        *,
        session_key: str,
        session_created_at: str,
        generation: int,
        parent_generation: int,
        source_ref: str,
        source_from_seq: int,
        consolidated_through_seq: int,
        source_message_ids: list[str] | tuple[str, ...],
        retained_tail: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        source_mutation_digest: str | None = None,
    ) -> CompactionPrepare:
        """Durably fence canonical sources before a cross-file receipt write."""

        # 1. 校验外部 prepare identity，再复制可变 JSON 输入。
        self._validate_prepare_payload(
            session_key=session_key,
            session_created_at=session_created_at,
            generation=generation,
            parent_generation=parent_generation,
            source_ref=source_ref,
            source_from_seq=source_from_seq,
            consolidated_through_seq=consolidated_through_seq,
            source_message_ids=source_message_ids,
            retained_tail=retained_tail,
        )
        if source_mutation_digest is not None:
            source_mutation_digest = _validate_source_mutation_digest(
                source_mutation_digest
            )
        prepared = CompactionPrepare(
            session_key=session_key.strip(),
            session_created_at=session_created_at.strip(),
            generation=generation,
            parent_generation=parent_generation,
            source_ref=source_ref.strip(),
            source_from_seq=source_from_seq,
            consolidated_through_seq=consolidated_through_seq,
            source_message_ids=tuple(source_message_ids),
            retained_tail=tuple(dict(item) for item in retained_tail),
            prepared_at=datetime.now().astimezone().isoformat(),
        )
        # 2. 以 immutable value 进入 SQLite 事务，避免调用方后续修改 source plan。
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._validate_compaction_prepare_locked(
                    prepared,
                    source_mutation_digest=source_mutation_digest,
                )
                if existing is not None:
                    self._conn.rollback()
                    return existing
                # 3. 只有通过同一事务校验的 prepare 才能进入 durable fence。
                self._insert_compaction_prepare_locked(prepared)
                self._conn.commit()
                return prepared
            except BaseException:
                self._conn.rollback()
                raise

    def get_compaction_prepare(
        self,
        session_key: str,
        *,
        source_ref: str | None = None,
        generation: int | None = None,
    ) -> CompactionPrepare | None:
        """Read one pending prepare fence by its incarnation-scoped identity."""

        if (source_ref is None) == (generation is None):
            raise ValueError("compaction prepare 必须按 source_ref 或 generation 查询")
        with self._lock:
            if source_ref is not None:
                row = self._conn.execute(
                    "SELECT * FROM session_compaction_prepares "
                    "WHERE session_key = ? AND source_ref = ?",
                    (session_key, source_ref),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM session_compaction_prepares "
                    "WHERE session_key = ? AND generation = ?",
                    (session_key, generation),
                ).fetchone()
        return self._row_to_compaction_prepare(row) if row is not None else None

    def _clear_orphan_compaction_prepare(self, prepare: CompactionPrepare) -> bool:
        """Release one pre-effect orphan fence after proving its identity."""

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # 1. Re-read the full row so a stale recovery handle cannot clear a new fence.
                row = self._conn.execute(
                    "SELECT * FROM session_compaction_prepares "
                    "WHERE session_key = ? AND generation = ?",
                    (prepare.session_key, prepare.generation),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                if self._row_to_compaction_prepare(row) != prepare:
                    raise ValueError("compaction prepare identity 冲突")
                # 2. Only the runtime's no-receipt pre-effect path may remove this row.
                self._conn.execute(
                    "DELETE FROM session_compaction_prepares "
                    "WHERE session_key = ? AND generation = ?",
                    (prepare.session_key, prepare.generation),
                )
                self._conn.commit()
                return True
            except BaseException:
                self._conn.rollback()
                raise

    def _assert_compaction_prepare_locked(
        self,
        prepare: CompactionPrepare,
    ) -> None:
        """Prove the pending fence still owns the checkpoint write set."""

        row = self._conn.execute(
            "SELECT * FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        ).fetchone()
        if row is None:
            raise RuntimeError("compaction prepare 在 checkpoint 提交前丢失")
        existing = self._row_to_compaction_prepare(row)
        if not self._same_compaction_prepare_identity(existing, prepare):
            raise ValueError("compaction prepare identity 冲突")

    def _delete_compaction_prepare_locked(self, prepare: CompactionPrepare) -> None:
        """Delete the verified fence inside the checkpoint transaction."""

        self._assert_compaction_prepare_locked(prepare)
        self._conn.execute(
            "DELETE FROM session_compaction_prepares "
            "WHERE session_key = ? AND generation = ?",
            (prepare.session_key, prepare.generation),
        )

    def _require_no_pending_compaction_prepare_locked(
        self,
        session_keys: list[str],
    ) -> None:
        """Reject destructive mutation while any targeted session has a pending fence."""

        if not session_keys:
            return
        placeholders = ",".join("?" for _ in session_keys)
        row = self._conn.execute(
            "SELECT session_key, source_ref FROM session_compaction_prepares "
            f"WHERE session_key IN ({placeholders}) LIMIT 1",
            tuple(session_keys),
        ).fetchone()
        if row is not None:
            raise SessionCompactionPrepareConflictError(
                str(row["session_key"]),
                str(row["source_ref"]),
            )

    @staticmethod
    def _row_to_compaction(row: sqlite3.Row) -> SessionCompaction:
        """Decode one ledger row at the SQLite boundary."""

        source_ids = _decode_json_payload(
            row["source_message_ids_json"],
            fallback="[]",
            field="compaction source_message_ids",
            identifier=f"{row['session_key']}:{row['generation']}",
        )
        retained_tail = _decode_json_payload(
            row["retained_tail_json"],
            fallback="[]",
            field="compaction retained_tail",
            identifier=f"{row['session_key']}:{row['generation']}",
        )
        usage = _decode_json_payload(
            row["summary_usage_json"],
            fallback="{}",
            field="compaction summary_usage",
            identifier=f"{row['session_key']}:{row['generation']}",
        )
        if not isinstance(source_ids, list) or not all(
            isinstance(item, str) and item for item in source_ids
        ):
            raise ValueError(
                "compaction source_message_ids 必须是非空字符串数组: "
                f"{row['session_key']}:{row['generation']}"
            )
        if not isinstance(retained_tail, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and bool(item["id"])
            and isinstance(item.get("seq"), int)
            and isinstance(item.get("message"), dict)
            and isinstance(item.get("unit_ref"), str)
            and bool(item["unit_ref"])
            for item in retained_tail
        ):
            raise ValueError(
                "compaction retained_tail 必须是带 id/seq/unit_ref/message 的对象数组: "
                f"{row['session_key']}:{row['generation']}"
            )
        if not isinstance(usage, dict):
            raise ValueError(
                "compaction summary_usage 必须是 JSON object: "
                f"{row['session_key']}:{row['generation']}"
            )
        return SessionCompaction(
            session_key=str(row["session_key"]),
            generation=int(row["generation"]),
            parent_generation=int(row["parent_generation"]),
            created_at=str(row["created_at"]),
            trigger=str(row["trigger"]),
            summary_format_version=int(row["summary_format_version"]),
            summary=str(row["summary"]),
            source_ref=str(row["source_ref"]),
            source_plan_digest=_required_source_plan_digest(
                row["source_plan_digest"],
                identifier=f"{row['session_key']}:{row['generation']}",
            ),
            source_from_seq=int(row["source_from_seq"]),
            consolidated_through_seq=int(row["consolidated_through_seq"]),
            source_message_ids=tuple(str(item) for item in source_ids),
            retained_tail=tuple(cast(dict[str, Any], item) for item in retained_tail),
            model_runtime_id=str(row["model_runtime_id"]),
            model=str(row["model"]),
            context_window=int(row["context_window"]),
            threshold_tokens=int(row["threshold_tokens"]),
            hard_input_tokens=int(row["hard_input_tokens"]),
            keep_recent_tokens=int(row["keep_recent_tokens"]),
            tokens_before=int(row["tokens_before"]),
            tokens_after=int(row["tokens_after"]),
            summary_usage=cast(dict[str, Any], usage),
            invalidated_at=(
                str(row["invalidated_at"])
                if row["invalidated_at"] is not None
                else None
            ),
            invalidated_reason=(
                str(row["invalidated_reason"])
                if row["invalidated_reason"] is not None
                else None
            ),
        )

    def persist_compaction(
        self,
        *,
        session_key: str,
        trigger: str,
        summary: str,
        source_ref: str,
        source_plan_digest: str,
        source_from_seq: int,
        consolidated_through_seq: int,
        source_message_ids: list[str] | tuple[str, ...],
        retained_tail: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        model_runtime_id: str,
        model: str,
        context_window: int,
        threshold_tokens: int,
        hard_input_tokens: int,
        keep_recent_tokens: int,
        tokens_before: int,
        tokens_after: int,
        summary_usage: dict[str, Any],
        parent_generation: int | None = None,
        generation: int | None = None,
        summary_format_version: int = 1,
        prepare: CompactionPrepare | None = None,
        source_mutation_digest: str | None = None,
    ) -> SessionCompaction:
        """Insert one immutable checkpoint and advance the session cursor atomically."""

        # 1. Validate the boundary payload before acquiring the write transaction.
        if not session_key.strip() or not source_ref.strip() or not summary.strip():
            raise ValueError("compaction session_key/source_ref/summary 不能为空")
        source_plan_digest = _validate_source_plan_digest(source_plan_digest)
        if source_mutation_digest is not None:
            source_mutation_digest = _validate_source_mutation_digest(
                source_mutation_digest
            )
        if not source_message_ids or any(
            not isinstance(item, str) or not item.strip() for item in source_message_ids
        ):
            raise ValueError("compaction source_message_ids 必须是非空字符串数组")
        if (
            not isinstance(source_from_seq, int)
            or isinstance(source_from_seq, bool)
            or not isinstance(consolidated_through_seq, int)
            or isinstance(consolidated_through_seq, bool)
            or source_from_seq < 0
            or consolidated_through_seq < source_from_seq
        ):
            raise ValueError("compaction source seq 边界无效")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or not item["id"]
            or not isinstance(item.get("seq"), int)
            or not isinstance(item.get("message"), dict)
            or not isinstance(item.get("unit_ref"), str)
            or not item["unit_ref"]
            for item in retained_tail
        ):
            raise ValueError(
                "compaction retained_tail 必须是带 id/seq/unit_ref/message 的对象数组"
            )
        if not isinstance(summary_usage, dict):
            raise ValueError("compaction summary_usage 必须是 JSON object")
        encoded_ids = json.dumps(
            list(source_message_ids), ensure_ascii=False, separators=(",", ":")
        )
        encoded_tail = json.dumps(
            list(retained_tail), ensure_ascii=False, separators=(",", ":")
        )
        encoded_usage = json.dumps(
            summary_usage, ensure_ascii=False, separators=(",", ":")
        )
        if prepare is not None:
            if (
                prepare.session_key != session_key
                or prepare.source_ref != source_ref
                or prepare.source_from_seq != source_from_seq
                or prepare.consolidated_through_seq != consolidated_through_seq
                or prepare.source_message_ids != tuple(source_message_ids)
                or prepare.retained_tail != tuple(retained_tail)
                or (generation is not None and prepare.generation != generation)
                or (
                    parent_generation is not None
                    and prepare.parent_generation != parent_generation
                )
            ):
                raise ValueError("compaction checkpoint 与 prepare identity 冲突")
        now = datetime.now().astimezone().isoformat()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                session_row = self._conn.execute(
                    "SELECT created_at, last_consolidated FROM sessions WHERE key = ?",
                    (session_key,),
                ).fetchone()
                if session_row is None:
                    raise KeyError(f"session 不存在: {session_key}")
                if prepare is not None:
                    if str(session_row["created_at"]) != prepare.session_created_at:
                        raise ValueError(
                            "compaction checkpoint session incarnation 冲突"
                        )
                    self._assert_compaction_prepare_locked(prepare)
                else:
                    self._require_no_pending_compaction_prepare_locked([session_key])
                current_cursor = int(session_row["last_consolidated"] or 0)
                if source_mutation_digest is not None:
                    actual_digest = self._source_mutation_digest_locked(
                        session_key,
                        self._compaction_source_ids(source_message_ids, retained_tail),
                    )
                    if actual_digest != source_mutation_digest:
                        raise RuntimeError(
                            "compaction source snapshot 在 persist 前发生变化"
                        )
                self._validate_compaction_provenance_locked(
                    session_key,
                    source_message_ids=source_message_ids,
                    retained_tail=retained_tail,
                    source_from_seq=source_from_seq,
                    consolidated_through_seq=consolidated_through_seq,
                )
                existing = self._conn.execute(
                    "SELECT * FROM session_compactions "
                    "WHERE session_key = ? AND source_ref = ?",
                    (session_key, source_ref),
                ).fetchone()
                if existing is not None:
                    existing_value = self._row_to_compaction(existing)
                    expected_parent = (
                        existing_value.parent_generation
                        if parent_generation is None
                        else int(parent_generation)
                    )
                    if (
                        (
                            generation is not None
                            and int(generation) != existing_value.generation
                        )
                        or existing_value.trigger != str(trigger)
                        or existing_value.parent_generation != expected_parent
                        or existing_value.summary_format_version
                        != int(summary_format_version)
                        or existing_value.summary != summary
                        or existing_value.source_plan_digest != source_plan_digest
                        or existing_value.source_from_seq != int(source_from_seq)
                        or existing_value.consolidated_through_seq
                        != int(consolidated_through_seq)
                        or existing_value.source_message_ids
                        != tuple(source_message_ids)
                        or existing_value.retained_tail != tuple(retained_tail)
                        or existing_value.model_runtime_id != model_runtime_id
                        or existing_value.model != model
                        or existing_value.context_window != int(context_window)
                        or existing_value.threshold_tokens != int(threshold_tokens)
                        or existing_value.hard_input_tokens != int(hard_input_tokens)
                        or existing_value.keep_recent_tokens != int(keep_recent_tokens)
                        or existing_value.tokens_before != int(tokens_before)
                        or existing_value.tokens_after != int(tokens_after)
                        or existing_value.summary_usage != summary_usage
                    ):
                        raise ValueError(
                            f"compaction source_ref 内容冲突: {session_key}:{source_ref}"
                        )
                    if prepare is not None:
                        self._delete_compaction_prepare_locked(prepare)
                        self._conn.commit()
                    else:
                        self._conn.rollback()
                    return existing_value
                max_row = self._conn.execute(
                    "SELECT COALESCE(MAX(generation), 0) AS generation "
                    "FROM session_compactions WHERE session_key = ?",
                    (session_key,),
                ).fetchone()
                max_generation = int(max_row["generation"] if max_row else 0)
                if generation is None:
                    generation = max_generation + 1
                if int(generation) <= max_generation:
                    raise ValueError(
                        f"compaction generation 必须单调递增: {session_key}:{generation}"
                    )
                if parent_generation is None:
                    parent_generation = current_cursor
                if int(parent_generation) != current_cursor:
                    raise ValueError(
                        "compaction parent_generation 与当前 cursor 不一致: "
                        f"session={session_key} cursor={current_cursor} parent={parent_generation}"
                    )
                self._conn.execute(
                    """
                    INSERT INTO session_compactions (
                        session_key, generation, parent_generation, created_at,
                        trigger, summary_format_version, summary, source_ref,
                        source_plan_digest,
                        source_from_seq, consolidated_through_seq,
                        source_message_ids_json, retained_tail_json,
                        model_runtime_id, model, context_window, threshold_tokens,
                        hard_input_tokens, keep_recent_tokens, tokens_before,
                        tokens_after, summary_usage_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_key,
                        int(generation),
                        int(parent_generation),
                        now,
                        str(trigger),
                        int(summary_format_version),
                        summary.strip(),
                        source_ref.strip(),
                        source_plan_digest,
                        int(source_from_seq),
                        int(consolidated_through_seq),
                        encoded_ids,
                        encoded_tail,
                        model_runtime_id,
                        model,
                        int(context_window),
                        int(threshold_tokens),
                        int(hard_input_tokens),
                        int(keep_recent_tokens),
                        int(tokens_before),
                        int(tokens_after),
                        encoded_usage,
                    ),
                )
                self._conn.execute(
                    "UPDATE sessions SET last_consolidated = ?, updated_at = ? WHERE key = ?",
                    (int(generation), now, session_key),
                )
                if prepare is not None:
                    self._delete_compaction_prepare_locked(prepare)
                row = self._conn.execute(
                    "SELECT * FROM session_compactions WHERE session_key = ? AND generation = ?",
                    (session_key, int(generation)),
                ).fetchone()
                if row is None:
                    raise RuntimeError("compaction checkpoint insert 后无法读取")
                self._conn.commit()
                return self._row_to_compaction(row)
            except BaseException:
                self._conn.rollback()
                raise

    def _validate_compaction_provenance_locked(
        self,
        session_key: str,
        *,
        source_message_ids: list[str] | tuple[str, ...],
        retained_tail: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        source_from_seq: int,
        consolidated_through_seq: int,
    ) -> None:
        """Validate source ids/seqs while the compaction transaction owns the Store lock."""

        ids = {str(item) for item in source_message_ids}
        ids.update(str(item["id"]) for item in retained_tail)
        if not ids:
            raise ValueError("compaction provenance 不能为空")
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT id, seq FROM messages WHERE session_key = ? AND id IN ({placeholders})",
            (session_key, *sorted(ids)),
        ).fetchall()
        by_id = {str(row["id"]): int(row["seq"]) for row in rows}
        missing = sorted(ids - by_id.keys())
        if missing:
            raise ValueError(
                "compaction provenance 引用不存在的 canonical message: "
                + ",".join(missing)
            )
        for item in retained_tail:
            message_id = str(item["id"])
            if by_id[message_id] != int(item["seq"]):
                raise ValueError(
                    "compaction retained_tail seq 与 canonical message 不一致: "
                    f"{message_id}:{item['seq']}!={by_id[message_id]}"
                )
        source_seqs = [by_id[str(message_id)] for message_id in source_message_ids]
        if min(source_seqs) != int(source_from_seq) or max(source_seqs) != int(
            consolidated_through_seq
        ):
            raise ValueError(
                "compaction source seq 边界与 canonical message 不一致: "
                f"{source_from_seq}-{consolidated_through_seq}!="
                f"{min(source_seqs)}-{max(source_seqs)}"
            )

    def validate_compaction_provenance(
        self,
        session_key: str,
        *,
        source_message_ids: list[str] | tuple[str, ...],
        retained_tail: list[dict[str, Any]] | tuple[dict[str, Any], ...],
        source_from_seq: int,
        consolidated_through_seq: int,
    ) -> None:
        """Fail early on missing provenance; persist_compaction repeats it atomically."""

        with self._lock:
            self._validate_compaction_provenance_locked(
                session_key,
                source_message_ids=source_message_ids,
                retained_tail=retained_tail,
                source_from_seq=source_from_seq,
                consolidated_through_seq=consolidated_through_seq,
            )

    def get_compaction(
        self, session_key: str, generation: int
    ) -> SessionCompaction | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_compactions WHERE session_key = ? AND generation = ?",
                (session_key, int(generation)),
            ).fetchone()
        return self._row_to_compaction(row) if row is not None else None

    def get_active_compaction(self, session_key: str) -> SessionCompaction | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT last_consolidated FROM sessions WHERE key = ?",
                (session_key,),
            ).fetchone()
            if cursor is None or int(cursor["last_consolidated"] or 0) == 0:
                return None
            row = self._conn.execute(
                "SELECT * FROM session_compactions WHERE session_key = ? AND generation = ?",
                (session_key, int(cursor["last_consolidated"])),
            ).fetchone()
        if row is None:
            raise ValueError(
                "session last_consolidated 未指向 compaction generation: "
                f"{session_key}:{cursor['last_consolidated']}"
            )
        value = self._row_to_compaction(row)
        if not value.active:
            raise ValueError(
                "session last_consolidated 指向已失效 compaction generation: "
                f"{session_key}:{value.generation}"
            )
        return value

    def get_compaction_head(self, session_key: str) -> CompactionHead:
        """Read the current cursor and never-reused next generation atomically."""

        with self._lock:
            session_row = self._conn.execute(
                "SELECT last_consolidated FROM sessions WHERE key = ?",
                (session_key,),
            ).fetchone()
            if session_row is None:
                raise KeyError(f"session 不存在: {session_key}")
            cursor = int(session_row["last_consolidated"] or 0)
            max_row = self._conn.execute(
                "SELECT COALESCE(MAX(generation), 0) AS generation "
                "FROM session_compactions WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            max_generation = int(max_row["generation"] if max_row else 0)
            if cursor < 0 or cursor > max_generation:
                raise ValueError(
                    "session compaction cursor 超出 ledger head: "
                    f"{session_key}:{cursor}>{max_generation}"
                )
            if cursor:
                row = self._conn.execute(
                    "SELECT invalidated_at FROM session_compactions "
                    "WHERE session_key = ? AND generation = ?",
                    (session_key, cursor),
                ).fetchone()
                if row is None or row["invalidated_at"] is not None:
                    raise ValueError(
                        "session compaction cursor 未指向有效 generation: "
                        f"{session_key}:{cursor}"
                    )
        return CompactionHead(
            session_key=session_key,
            parent_generation=cursor,
            next_generation=max_generation + 1,
        )

    def list_compactions(
        self, session_key: str, *, include_invalidated: bool = True
    ) -> list[SessionCompaction]:
        where = "" if include_invalidated else " AND invalidated_at IS NULL"
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_compactions WHERE session_key = ?"
                + where
                + " ORDER BY generation",
                (session_key,),
            ).fetchall()
        return [self._row_to_compaction(row) for row in rows]

    def list_compaction_briefs(
        self,
        session_keys: Sequence[str],
    ) -> dict[str, dict[str, Any] | None]:
        """Return the active compaction brief for each requested session key."""

        keys = [str(key).strip() for key in session_keys if str(key).strip()]
        if not keys:
            return {}
        placeholders = ",".join("?" for _ in keys)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT
                    s.key AS session_key,
                    c.generation,
                    c.trigger,
                    c.tokens_before,
                    c.tokens_after,
                    c.summary,
                    c.model,
                    c.created_at
                FROM sessions s
                LEFT JOIN session_compactions c
                  ON c.session_key = s.key
                 AND c.generation = s.last_consolidated
                 AND c.invalidated_at IS NULL
                WHERE s.key IN ({placeholders})
                """,
                tuple(keys),
            ).fetchall()
        briefs: dict[str, dict[str, Any] | None] = {}
        for row in rows:
            key = str(row["session_key"])
            if row["generation"] is None:
                briefs[key] = None
                continue
            briefs[key] = {
                "generation": int(row["generation"]),
                "trigger": row["trigger"],
                "tokens_before": int(row["tokens_before"] or 0),
                "tokens_after": int(row["tokens_after"] or 0),
                "summary_preview": str(row["summary"] or ""),
                "model": row["model"],
                "created_at": row["created_at"],
            }
        return briefs

    def _invalidate_compactions_for_messages_locked(
        self,
        session_key: str,
        message_ids: set[str],
        *,
        reason: str,
    ) -> tuple[int, int]:
        """Invalidate a checkpoint and descendants whose provenance was deleted."""

        if not message_ids:
            return 0, 0
        rows = self._conn.execute(
            "SELECT * FROM session_compactions WHERE session_key = ? ORDER BY generation",
            (session_key,),
        ).fetchall()
        first_hit: int | None = None
        for row in rows:
            checkpoint = self._row_to_compaction(row)
            if not checkpoint.active:
                continue
            retained_ids = {
                str(item.get("id"))
                for item in checkpoint.retained_tail
                if item.get("id")
            }
            if message_ids.intersection(
                checkpoint.source_message_ids
            ) or message_ids.intersection(retained_ids):
                first_hit = checkpoint.generation
                break
        if first_hit is None:
            return 0, 0
        invalidated_at = datetime.now().astimezone().isoformat()
        self._conn.execute(
            """
            UPDATE session_compactions
            SET invalidated_at = ?, invalidated_reason = ?
            WHERE session_key = ? AND generation >= ? AND invalidated_at IS NULL
            """,
            (invalidated_at, reason, session_key, first_hit),
        )
        previous = self._conn.execute(
            """
            SELECT MAX(generation) AS generation
            FROM session_compactions
            WHERE session_key = ? AND generation < ? AND invalidated_at IS NULL
            """,
            (session_key, first_hit),
        ).fetchone()
        new_cursor = int(previous["generation"] or 0) if previous else 0
        self._conn.execute(
            "UPDATE sessions SET last_consolidated = ?, updated_at = ? WHERE key = ?",
            (new_cursor, invalidated_at, session_key),
        )
        return first_hit, new_cursor

    def get_session_meta(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT key, created_at, updated_at, last_consolidated, metadata, last_user_at, last_proactive_at FROM sessions WHERE key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "key": row["key"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "last_consolidated": int(row["last_consolidated"] or 0),
            "metadata": _decode_session_metadata(row["metadata"], str(row["key"])),
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
        }

    def create_turn(self, record: TurnRecord) -> TurnRecord:
        """持久化一个 queued turn 并返回数据库中的正式记录。"""
        if record.status is not TurnStatus.QUEUED:
            raise TurnStateTransitionError("turn 创建时必须处于 queued 状态")
        if record.started_at is not None or record.completed_at is not None:
            raise TurnStateTransitionError(
                "queued turn 不得包含 started_at/completed_at"
            )
        if (
            record.usage is not None
            or record.error is not None
            or record.final_response is not None
        ):
            raise TurnStateTransitionError(
                "queued turn 不得包含 usage/error/final_response"
            )

        # 1. 在写入前完成所有 JSON 编码，序列化失败时数据库保持不变。
        input_json = json.dumps(
            {"input": record.input, "metadata": record.metadata},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        items_json = json.dumps(
            [item.to_dict() for item in record.items],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        usage_json = (
            json.dumps(
                record.usage.to_dict(), ensure_ascii=False, separators=(",", ":")
            )
            if record.usage is not None
            else None
        )
        error_json = (
            json.dumps(
                record.error.to_dict(), ensure_ascii=False, separators=(",", ":")
            )
            if record.error is not None
            else None
        )

        # 2. 单条 INSERT 建立不可变 turn identity。
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO turns (
                    id, session_key, status, input_json, items_json,
                    usage_json, error_json, final_response,
                    created_at, started_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.thread_id,
                    record.status.value,
                    input_json,
                    items_json,
                    usage_json,
                    error_json,
                    record.final_response,
                    record.created_at.isoformat(),
                    None,
                    None,
                ),
            )
            self._conn.commit()
        stored = self.read_turn(record.id)
        if stored is None:
            raise RuntimeError(f"turn 创建后无法读取: {record.id}")
        return stored

    def read_turn(self, turn_id: str) -> TurnRecord | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response,
                       created_at, started_at, completed_at
                FROM turns
                WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
        return self._row_to_turn(row) if row is not None else None

    def append_active_turn_item(
        self,
        turn_id: str,
        *,
        thread_id: str,
        item: TurnItem,
    ) -> TurnRecord:
        """原子追加 active turn item，并拒绝终态或 thread 漂移。"""

        # 1. 在同一事务读取并校验当前 active turn。
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response,
                       created_at, started_at, completed_at
                FROM turns
                WHERE id = ?
                """,
                (turn_id,),
            ).fetchone()
            if row is None or str(row["session_key"]) != thread_id:
                raise TurnNotFoundError(f"turn 不属于 thread: {thread_id}/{turn_id}")
            status = TurnStatus(str(row["status"]))
            if status not in {TurnStatus.QUEUED, TurnStatus.IN_PROGRESS}:
                raise TurnStateTransitionError(
                    f"terminal turn 不得追加 item: {turn_id}/{status.value}"
                )
            items = _decode_turn_items(row["items_json"], turn_id)
            if any(existing.id == item.id for existing in items):
                raise ValueError(f"turn item id 重复: {turn_id}/{item.id}")

            # 2. status CAS 保证 append 不会跨过并发 terminal transition。
            items.append(item)
            payload = json.dumps(
                [entry.to_dict() for entry in items],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cursor = self._conn.execute(
                "UPDATE turns SET items_json = ? WHERE id = ? AND status = ?",
                (payload, turn_id, status.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise TurnStateTransitionError(
                    f"turn item append CAS 失败: {turn_id}/{status.value}"
                )
            self._conn.commit()

        stored = self.read_turn(turn_id)
        if stored is None:
            raise RuntimeError(f"turn item 追加后无法读取: {turn_id}")
        return stored

    def replace_active_turn_item(
        self,
        turn_id: str,
        *,
        thread_id: str,
        item: TurnItem,
    ) -> TurnRecord:
        """原子替换 active turn 中同 identity item 的最新 checkpoint。"""

        # 1. 在同一事务定位 active turn 和既有 item identity。
        with self._lock:
            row = self._conn.execute(
                "SELECT session_key, status, items_json FROM turns WHERE id = ?",
                (turn_id,),
            ).fetchone()
            if row is None or str(row["session_key"]) != thread_id:
                raise TurnNotFoundError(f"turn 不属于 thread: {thread_id}/{turn_id}")
            status = TurnStatus(str(row["status"]))
            if status not in {TurnStatus.QUEUED, TurnStatus.IN_PROGRESS}:
                raise TurnStateTransitionError(
                    f"terminal turn 不得更新 item: {turn_id}/{status.value}"
                )
            items = _decode_turn_items(row["items_json"], turn_id)
            matches = [
                index for index, existing in enumerate(items) if existing.id == item.id
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"turn item identity 无法唯一解析: {turn_id}/{item.id}"
                )

            # 2. status CAS 保证 started/completed 更新不跨过终态。
            items[matches[0]] = item
            payload = json.dumps(
                [entry.to_dict() for entry in items],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            cursor = self._conn.execute(
                "UPDATE turns SET items_json = ? WHERE id = ? AND status = ?",
                (payload, turn_id, status.value),
            )
            if cursor.rowcount != 1:
                self._conn.rollback()
                raise TurnStateTransitionError(
                    f"turn item update CAS 失败: {turn_id}/{status.value}"
                )
            self._conn.commit()

        stored = self.read_turn(turn_id)
        if stored is None:
            raise RuntimeError(f"turn item 更新后无法读取: {turn_id}")
        return stored

    def transition_turn(
        self,
        turn_id: str,
        *,
        expected_status: TurnStatus,
        status: TurnStatus,
        thread_id: str | None = None,
        items: list[TurnItem] | None = None,
        usage: TurnUsage | None = None,
        error: TurnError | None = None,
        final_response: str | None = None,
        now: datetime | None = None,
    ) -> TurnRecord:
        """用单条 CAS 更新 turn 状态，状态漂移时明确失败。"""
        expected_status = TurnStatus(expected_status)
        status = TurnStatus(status)
        allowed = _TURN_TRANSITIONS.get(expected_status, frozenset())
        if status not in allowed:
            raise TurnStateTransitionError(
                f"非法 turn 状态转换: {expected_status.value} -> {status.value}"
            )
        if status is TurnStatus.FAILED and error is None:
            raise TurnStateTransitionError("failed turn 必须包含 error")
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("turn transition 时间必须包含时区")
        timestamp = timestamp.astimezone(UTC)

        # 1. 只更新本次调用明确拥有的终态字段。
        set_parts = ["status = ?"]
        params: list[object] = [status.value]
        if status is TurnStatus.IN_PROGRESS:
            set_parts.append("started_at = ?")
            params.append(timestamp.isoformat())
        if status.is_terminal:
            set_parts.append("completed_at = ?")
            params.append(timestamp.isoformat())
        if items is not None:
            set_parts.append("items_json = ?")
            params.append(
                json.dumps(
                    [item.to_dict() for item in items],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        if usage is not None:
            set_parts.append("usage_json = ?")
            params.append(
                json.dumps(usage.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
        if error is not None:
            set_parts.append("error_json = ?")
            params.append(
                json.dumps(error.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
        if final_response is not None:
            set_parts.append("final_response = ?")
            params.append(final_response)

        # 2. status 和可选 thread identity 共同构成 compare-and-set 条件。
        where_parts = ["id = ?", "status = ?"]
        params.extend([turn_id, expected_status.value])
        if thread_id is not None:
            where_parts.append("session_key = ?")
            params.append(thread_id)
        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE turns SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}",
                tuple(params),
            )
            if cursor.rowcount != 1:
                current = self._conn.execute(
                    "SELECT session_key, status FROM turns WHERE id = ?", (turn_id,)
                ).fetchone()
                self._conn.rollback()
                if current is None:
                    raise TurnNotFoundError(f"turn 不存在: {turn_id}")
                if thread_id is not None and str(current["session_key"]) != thread_id:
                    raise TurnNotFoundError(
                        f"turn 不属于 thread: {thread_id}/{turn_id}"
                    )
                raise TurnStateTransitionError(
                    f"turn CAS 失败，期望 {expected_status.value}，实际 {current['status']}: {turn_id}"
                )
            self._conn.commit()

        # 3. 返回同一连接提交后可重读的正式记录。
        stored = self.read_turn(turn_id)
        if stored is None:
            raise RuntimeError(f"turn 转换后无法读取: {turn_id}")
        return stored

    def list_turns(
        self,
        thread_id: str,
        *,
        limit: int = 100,
        before: tuple[str, str] | None = None,
    ) -> list[TurnRecord]:
        """按创建时间倒序读取一个 thread 的稳定 turn 页面。"""
        if limit <= 0 or limit > 200:
            raise ValueError("turn list limit 必须在 1..200")
        where = "session_key = ?"
        params: list[object] = [thread_id]
        if before is not None:
            where += " AND (created_at < ? OR (created_at = ? AND id < ?))"
            params.extend([before[0], before[0], before[1]])
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response,
                       created_at, started_at, completed_at
                FROM turns
                WHERE {where}
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        return [self._row_to_turn(row) for row in rows]

    def recover_in_progress_turns(
        self,
        *,
        now: datetime | None = None,
    ) -> list[TurnRecord]:
        """把上一 runtime 遗留的 queued/in_progress turn 原子收敛为终态。"""
        timestamp = now or datetime.now(UTC)
        if timestamp.tzinfo is None:
            raise ValueError("turn recovery 时间必须包含时区")
        timestamp = timestamp.astimezone(UTC)

        # 1. 在一个事务内闭合遗留 item，并用 status CAS 提交终态。
        recovered_ids: list[str] = []
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, status, items_json FROM turns
                WHERE status IN (?, ?)
                ORDER BY created_at, id
                """,
                (TurnStatus.QUEUED.value, TurnStatus.IN_PROGRESS.value),
            ).fetchall()
            for row in rows:
                turn_id = str(row["id"])
                previous_status = TurnStatus(str(row["status"]))
                items = _decode_turn_items(row["items_json"], turn_id)
                closed_items = [
                    (
                        TurnItem(
                            item.kind,
                            item.id,
                            {**item.data, "status": TurnStatus.INTERRUPTED.value},
                        )
                        if item.data.get("status") == TurnStatus.IN_PROGRESS.value
                        else item
                    )
                    for item in items
                ]
                # 2. queued 从未开始执行，收敛为 cancelled；执行中收敛为 interrupted。
                target_status = (
                    TurnStatus.CANCELLED
                    if previous_status is TurnStatus.QUEUED
                    else TurnStatus.INTERRUPTED
                )
                cursor = self._conn.execute(
                    """
                    UPDATE turns
                    SET status = ?, items_json = ?, completed_at = ?
                    WHERE id = ? AND status = ?
                    """,
                    (
                        target_status.value,
                        json.dumps(
                            [item.to_dict() for item in closed_items],
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                        timestamp.isoformat(),
                        turn_id,
                        previous_status.value,
                    ),
                )
                if cursor.rowcount != 1:
                    self._conn.rollback()
                    raise TurnStateTransitionError(
                        f"turn recovery CAS 失败: {turn_id}/{previous_status.value}"
                    )
                recovered_ids.append(turn_id)
            self._conn.commit()

        # 3. 从提交后的权威行恢复严格领域对象。
        recovered = [self.read_turn(turn_id) for turn_id in recovered_ids]
        if any(record is None for record in recovered):
            raise RuntimeError("turn recovery 提交后无法重读")
        return cast(list[TurnRecord], recovered)

    def delete_thread_turns(self, thread_id: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM turns WHERE session_key = ?", (thread_id,)
            )
            self._conn.commit()
        return int(cursor.rowcount or 0)

    def _row_to_turn(self, row: sqlite3.Row) -> TurnRecord:
        """在 SQLite 边界把 turn 行恢复成严格领域对象。"""
        turn_id = str(row["id"])
        input_text, metadata = _decode_turn_input(row["input_json"], turn_id)
        return TurnRecord(
            id=turn_id,
            thread_id=str(row["session_key"]),
            status=TurnStatus(str(row["status"])),
            input=input_text,
            metadata=metadata,
            items=_decode_turn_items(row["items_json"], turn_id),
            usage=_decode_turn_usage(row["usage_json"], turn_id),
            error=_decode_turn_error(row["error_json"], turn_id),
            final_response=cast(str | None, row["final_response"]),
            created_at=_decode_required_turn_time(
                row["created_at"], "created_at", turn_id
            ),
            started_at=parse_rfc3339(row["started_at"], "started_at"),
            completed_at=parse_rfc3339(row["completed_at"], "completed_at"),
        )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT key, created_at, updated_at, last_user_at, last_proactive_at
                FROM sessions
                ORDER BY updated_at DESC
                """).fetchall()
        return [
            {
                "key": str(row["key"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
            }
            for row in rows
        ]

    def list_sessions_for_dashboard(
        self,
        *,
        q: str = "",
        channel: str = "",
        updated_from: str = "",
        updated_to: str = "",
        has_proactive: bool | None = None,
        page: int = 1,
        page_size: int = 50,
        sort_by: str = "updated_at",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))
        offset = (safe_page - 1) * safe_page_size
        safe_sort_by = (
            sort_by
            if sort_by
            in {
                "updated_at",
                "created_at",
                "last_user_at",
                "last_proactive_at",
            }
            else "updated_at"
        )
        safe_sort_order = "ASC" if str(sort_order).lower() == "asc" else "DESC"

        params: list[Any] = []
        where_parts: list[str] = []
        query = (q or "").strip()
        if query:
            where_parts.append("(s.key LIKE ? OR COALESCE(s.metadata, '') LIKE ?)")
            like = f"%{query}%"
            params.extend([like, like])
        if channel:
            where_parts.append("s.key LIKE ?")
            params.append(f"{channel}:%")
        if updated_from:
            where_parts.append("s.updated_at >= ?")
            params.append(updated_from)
        if updated_to:
            where_parts.append("s.updated_at <= ?")
            params.append(updated_to)
        if has_proactive is True:
            where_parts.append("s.last_proactive_at IS NOT NULL")
        if has_proactive is False:
            where_parts.append("s.last_proactive_at IS NULL")

        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""
        count_sql = f"""
            SELECT COUNT(1) AS c
            FROM sessions s
            {where_sql}
        """
        data_sql = f"""
            SELECT
                s.key,
                s.created_at,
                s.updated_at,
                s.last_consolidated,
                s.metadata,
                s.last_user_at,
                s.last_proactive_at,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.session_key = s.key
                      AND m.role = 'user'
                      AND TRIM(COALESCE(m.content, '')) != ''
                    ORDER BY m.seq ASC
                    LIMIT 1
                ) AS first_message_content,
                (
                    SELECT COUNT(1)
                    FROM messages m
                    WHERE m.session_key = s.key
                ) AS message_count
            FROM sessions s
            {where_sql}
            ORDER BY s.{safe_sort_by} {safe_sort_order}, s.key ASC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            count_row = self._conn.execute(count_sql, tuple(params)).fetchone()
            rows = self._conn.execute(
                data_sql,
                tuple([*params, safe_page_size, offset]),
            ).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [
            {
                "key": str(row["key"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_consolidated": int(row["last_consolidated"] or 0),
                "metadata": _decode_session_metadata(row["metadata"], str(row["key"])),
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
                "first_message_content": row["first_message_content"],
                "message_count": int(row["message_count"] or 0),
            }
            for row in rows
        ], total

    def create_session(
        self,
        *,
        key: str,
        metadata: dict[str, Any] | None = None,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> dict[str, Any]:
        now = datetime.now().astimezone().isoformat()
        payload = json.dumps(metadata or {}, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    key,
                    created_at,
                    updated_at,
                    last_consolidated,
                    metadata,
                    last_user_at,
                    last_proactive_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    now,
                    now,
                    0,
                    payload,
                    last_user_at,
                    last_proactive_at,
                ),
            )
            self._conn.commit()
        meta = self.get_session_meta(key)
        if meta is None:
            raise ValueError(f"session 创建失败: {key}")
        return meta

    def update_session(
        self,
        key: str,
        *,
        metadata: dict[str, Any] | None = None,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> dict[str, Any] | None:
        set_parts = ["updated_at = ?"]
        params: list[Any] = [datetime.now().astimezone().isoformat()]
        if metadata is not None:
            set_parts.append("metadata = ?")
            params.append(json.dumps(metadata, ensure_ascii=False))
        if last_user_at is not None:
            set_parts.append("last_user_at = ?")
            params.append(last_user_at)
        if last_proactive_at is not None:
            set_parts.append("last_proactive_at = ?")
            params.append(last_proactive_at)
        params.append(key)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE sessions SET {', '.join(set_parts)} WHERE key = ?",
                tuple(params),
            )
            self._conn.commit()
        if cur.rowcount <= 0:
            return None
        return self.get_session_meta(key)

    @staticmethod
    def _normalize_delete_targets(keys: list[str]) -> tuple[str, ...]:
        targets: list[str] = []
        for raw_key in keys:
            key = str(raw_key).strip()
            if key and key not in targets:
                targets.append(key)
        return tuple(targets)

    @staticmethod
    def _normalize_action_source(action_source: str) -> str:
        if not isinstance(action_source, str) or not action_source.strip():
            raise ValueError("action_source 必须是非空字符串")
        return action_source.strip()

    @staticmethod
    def _delete_audit_value(
        *,
        audit_id: str,
        targets: tuple[str, ...],
        message_ids: tuple[str, ...],
        compactions: tuple[dict[str, Any], ...],
        action_source: str,
        cascade: bool,
        backup_path: Path | None,
        started_at: str,
        result: str,
        deleted_count: int,
        error: str | None = None,
    ) -> SessionDeleteAudit:
        return SessionDeleteAudit(
            audit_id=audit_id,
            targets=targets,
            message_ids=message_ids,
            compactions=compactions,
            action_source=action_source,
            cascade=cascade,
            backup_path=str(backup_path) if backup_path is not None else None,
            started_at=started_at,
            completed_at=datetime.now().astimezone().isoformat(),
            result=result,
            deleted_count=deleted_count,
            error=error,
        )

    def _insert_delete_audit_locked(self, audit: SessionDeleteAudit) -> None:
        self._conn.execute(
            """
            INSERT INTO session_delete_audits(
                audit_id, targets_json, message_ids_json, compactions_json,
                action_source, cascade, backup_path, started_at, completed_at,
                result, deleted_count, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit.audit_id,
                json.dumps(list(audit.targets), ensure_ascii=False),
                json.dumps(list(audit.message_ids), ensure_ascii=False),
                json.dumps(list(audit.compactions), ensure_ascii=False),
                audit.action_source,
                int(audit.cascade),
                audit.backup_path,
                audit.started_at,
                audit.completed_at,
                audit.result,
                audit.deleted_count,
                audit.error,
            ),
        )

    def _persist_delete_audit(self, audit: SessionDeleteAudit) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._insert_delete_audit_locked(audit)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

    def _row_to_delete_audit(self, row: sqlite3.Row) -> SessionDeleteAudit:
        targets = _decode_json_payload(
            row["targets_json"],
            fallback="[]",
            field="session delete audit targets",
            identifier=str(row["audit_id"]),
        )
        if not isinstance(targets, list) or not all(
            isinstance(target, str) and target for target in targets
        ):
            raise ValueError(f"session delete audit targets 无效: {row['audit_id']}")
        message_ids = _decode_json_payload(
            row["message_ids_json"],
            fallback="[]",
            field="session delete audit message ids",
            identifier=str(row["audit_id"]),
        )
        if not isinstance(message_ids, list) or not all(
            isinstance(message_id, str) and message_id for message_id in message_ids
        ):
            raise ValueError(
                f"session delete audit message ids 无效: {row['audit_id']}"
            )
        compactions = _decode_json_payload(
            row["compactions_json"],
            fallback="[]",
            field="session delete audit compactions",
            identifier=str(row["audit_id"]),
        )
        if not isinstance(compactions, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("session_key"), str)
            and isinstance(item.get("generation"), int)
            and isinstance(item.get("parent_generation"), int)
            and isinstance(item.get("source_ref"), str)
            and isinstance(item.get("source_message_ids"), list)
            for item in compactions
        ):
            raise ValueError(
                f"session delete audit compactions 无效: {row['audit_id']}"
            )
        return SessionDeleteAudit(
            audit_id=str(row["audit_id"]),
            targets=tuple(cast(str, target) for target in targets),
            message_ids=tuple(cast(str, message_id) for message_id in message_ids),
            compactions=tuple(cast(dict[str, Any], item) for item in compactions),
            action_source=str(row["action_source"]),
            cascade=bool(int(row["cascade"])),
            backup_path=(
                str(row["backup_path"]) if row["backup_path"] is not None else None
            ),
            started_at=str(row["started_at"]),
            completed_at=str(row["completed_at"]),
            result=str(row["result"]),
            deleted_count=int(row["deleted_count"]),
            error=str(row["error"]) if row["error"] is not None else None,
        )

    def get_session_delete_audit(self, audit_id: str) -> SessionDeleteAudit | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_delete_audits WHERE audit_id = ?",
                (audit_id,),
            ).fetchone()
        return None if row is None else self._row_to_delete_audit(row)

    def list_session_delete_audits(
        self, *, limit: int = 100
    ) -> list[SessionDeleteAudit]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("session delete audit limit 必须是正整数")
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM session_delete_audits
                ORDER BY completed_at DESC, audit_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._row_to_delete_audit(row) for row in rows]

    @staticmethod
    def _normalize_source_ids(
        source_ids: list[str] | tuple[str, ...],
    ) -> tuple[str, ...]:
        if isinstance(source_ids, (str, bytes)):
            raise ValueError("source_ids 必须是字符串数组")
        normalized = tuple(dict.fromkeys(str(item).strip() for item in source_ids))
        if not normalized or any(not item for item in normalized):
            raise ValueError("source_ids 必须包含非空字符串")
        return normalized

    @staticmethod
    def _compaction_source_ids(
        source_message_ids: list[str] | tuple[str, ...],
        retained_tail: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    ) -> tuple[str, ...]:
        """Return the ordered, de-duplicated source rows covered by a checkpoint."""

        return SessionStore._normalize_source_ids(
            [
                *source_message_ids,
                *(str(item["id"]) for item in retained_tail),
            ]
        )

    def _source_mutation_digest_locked(
        self,
        session_key: str,
        source_ids: tuple[str, ...],
    ) -> str:
        """Hash raw canonical message rows while the Store lock is held."""

        placeholders = ",".join("?" for _ in source_ids)
        rows = self._conn.execute(
            "SELECT id, session_key, seq, role, content, tool_chain, extra, ts "
            f"FROM messages WHERE id IN ({placeholders})",
            source_ids,
        ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        missing = [message_id for message_id in source_ids if message_id not in by_id]
        if missing:
            raise ValueError(
                "compaction source snapshot 缺少 canonical message: "
                + ",".join(missing)
            )
        payload: list[dict[str, Any]] = []
        for message_id in source_ids:
            row = by_id[message_id]
            if str(row["session_key"]) != session_key:
                raise ValueError(
                    "compaction source snapshot 跨 session: "
                    f"{message_id}:{row['session_key']}!={session_key}"
                )
            payload.append(
                {
                    "id": str(row["id"]),
                    "session_key": str(row["session_key"]),
                    "seq": int(row["seq"]),
                    "role": str(row["role"]),
                    "content": row["content"],
                    "tool_chain": row["tool_chain"],
                    "extra": row["extra"],
                    "ts": str(row["ts"]),
                }
            )
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def source_mutation_digest(
        self,
        session_key: str,
        source_ids: list[str] | tuple[str, ...],
    ) -> str:
        """Hash the current raw canonical rows covered by a compaction checkpoint."""

        if not isinstance(session_key, str) or not session_key.strip():
            raise ValueError("session_key 必须是非空字符串")
        normalized = self._normalize_source_ids(source_ids)
        with self._lock:
            return self._source_mutation_digest_locked(session_key.strip(), normalized)

    def _record_source_mutation_locked(
        self,
        *,
        operation: str,
        session_key: str,
        message_ids: tuple[str, ...],
        action_source: str,
        backup_path: Path | None,
    ) -> SourceMutationAudit:
        audit = SourceMutationAudit(
            audit_id=uuid4().hex,
            operation=operation,
            session_key=session_key,
            message_ids=message_ids,
            action_source=action_source,
            backup_path=str(backup_path) if backup_path is not None else None,
            completed_at=datetime.now().astimezone().isoformat(),
        )
        self._conn.execute(
            """
            INSERT INTO session_source_mutation_audits(
                audit_id, operation, session_key, message_ids_json, action_source,
                backup_path, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                audit.audit_id,
                audit.operation,
                audit.session_key,
                json.dumps(list(audit.message_ids), ensure_ascii=False),
                audit.action_source,
                audit.backup_path,
                audit.completed_at,
            ),
        )
        return audit

    def _row_to_source_mutation_audit(
        self,
        row: sqlite3.Row,
    ) -> SourceMutationAudit:
        message_ids = _decode_json_payload(
            row["message_ids_json"],
            fallback="[]",
            field="source mutation audit message ids",
            identifier=str(row["audit_id"]),
        )
        if not isinstance(message_ids, list) or not all(
            isinstance(item, str) and item for item in message_ids
        ):
            raise ValueError(
                f"source mutation audit message ids 无效: {row['audit_id']}"
            )
        return SourceMutationAudit(
            audit_id=str(row["audit_id"]),
            operation=str(row["operation"]),
            session_key=str(row["session_key"]),
            message_ids=tuple(cast(str, item) for item in message_ids),
            action_source=str(row["action_source"]),
            backup_path=(
                str(row["backup_path"]) if row["backup_path"] is not None else None
            ),
            completed_at=str(row["completed_at"]),
        )

    def find_authorized_source_mutations(
        self,
        *,
        session_key: str,
        source_ids: list[str] | tuple[str, ...],
        prepared_at: str,
    ) -> list[SourceMutationAudit]:
        """Return committed mutation audits intersecting the prepared source plan."""

        if not isinstance(session_key, str) or not session_key.strip():
            raise ValueError("session_key 必须是非空字符串")
        if not isinstance(prepared_at, str) or not prepared_at.strip():
            raise ValueError("prepared_at 必须是非空字符串")
        requested = set(self._normalize_source_ids(source_ids))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT *
                FROM session_source_mutation_audits
                WHERE session_key = ?
                  AND completed_at >= ?
                ORDER BY completed_at, audit_id
                """,
                (session_key.strip(), prepared_at.strip()),
            ).fetchall()
        audits = [self._row_to_source_mutation_audit(row) for row in rows]
        return [audit for audit in audits if requested.intersection(audit.message_ids)]

    def _snapshot_delete_lineage_locked(
        self,
        targets: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[dict[str, Any], ...]]:
        """Capture canonical message IDs and compaction lineage before deletion."""

        placeholders = ",".join("?" for _ in targets)
        message_rows = self._conn.execute(
            f"""
            SELECT id
            FROM messages
            WHERE session_key IN ({placeholders})
            ORDER BY session_key, seq
            """,
            targets,
        ).fetchall()
        compaction_rows = self._conn.execute(
            f"""
            SELECT *
            FROM session_compactions
            WHERE session_key IN ({placeholders})
            ORDER BY session_key, generation
            """,
            targets,
        ).fetchall()
        compactions: list[dict[str, Any]] = []
        for row in compaction_rows:
            checkpoint = self._row_to_compaction(row)
            compactions.append(
                {
                    "session_key": checkpoint.session_key,
                    "generation": checkpoint.generation,
                    "parent_generation": checkpoint.parent_generation,
                    "source_ref": checkpoint.source_ref,
                    "source_message_ids": list(checkpoint.source_message_ids),
                }
            )
        return (
            tuple(str(row["id"]) for row in message_rows),
            tuple(compactions),
        )

    def _delete_sessions_with_audit(
        self,
        targets: tuple[str, ...],
        *,
        cascade: bool,
        action_source: str,
    ) -> SessionDeleteAudit:
        action_source = self._normalize_action_source(action_source)
        audit_id = uuid4().hex
        started_at = datetime.now().astimezone().isoformat()
        backup_path: Path | None = None
        message_ids: tuple[str, ...] = ()
        compactions: tuple[dict[str, Any], ...] = ()
        try:
            with self._lock:
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    if not targets:
                        audit = self._delete_audit_value(
                            audit_id=audit_id,
                            targets=targets,
                            message_ids=message_ids,
                            compactions=compactions,
                            action_source=action_source,
                            cascade=cascade,
                            backup_path=None,
                            started_at=started_at,
                            result="not_found",
                            deleted_count=0,
                        )
                        self._insert_delete_audit_locked(audit)
                        self._conn.commit()
                        return audit

                    # 1. 读取待删除 lineage，并在同一写事务中确认 admission。
                    placeholders = ",".join("?" for _ in targets)
                    existing_rows = self._conn.execute(
                        f"SELECT key FROM sessions WHERE key IN ({placeholders})",
                        targets,
                    ).fetchall()
                    if not existing_rows:
                        audit = self._delete_audit_value(
                            audit_id=audit_id,
                            targets=targets,
                            message_ids=message_ids,
                            compactions=compactions,
                            action_source=action_source,
                            cascade=cascade,
                            backup_path=None,
                            started_at=started_at,
                            result="not_found",
                            deleted_count=0,
                        )
                        self._insert_delete_audit_locked(audit)
                        self._conn.commit()
                        return audit

                    message_ids, compactions = self._snapshot_delete_lineage_locked(
                        targets
                    )
                    self._require_sessions_not_admitted_locked(list(targets))
                    self._require_no_pending_compaction_prepare_locked(list(targets))

                    if not cascade:
                        row = self._conn.execute(
                            f"""
                            SELECT
                                (SELECT COUNT(1) FROM messages
                                 WHERE session_key IN ({placeholders})) +
                                (SELECT COUNT(1) FROM turns
                                 WHERE session_key IN ({placeholders})) +
                                (SELECT COUNT(1) FROM session_compactions
                                 WHERE session_key IN ({placeholders})) +
                                (SELECT COUNT(1) FROM session_compaction_prepares
                                 WHERE session_key IN ({placeholders})) AS c
                            """,
                            tuple([*targets, *targets, *targets, *targets]),
                        ).fetchone()
                        count = int((row["c"] if row else 0) or 0)
                        if count > 0:
                            raise ValueError(
                                "选中的 session 仍有 messages、turns、compactions 或 pending prepare，"
                                "需使用 cascade 删除"
                            )

                    # 2. 在任何物理减少前创建并校验不可覆盖的完整快照。
                    backup_path = self._backup_before_session_delete_locked()
                    if cascade and self._has_message_embeddings_locked():
                        self._conn.execute(
                            f"""
                            DELETE FROM message_embeddings
                            WHERE message_id IN (
                                SELECT id FROM messages
                                WHERE session_key IN ({placeholders})
                            )
                            """,
                            targets,
                        )
                    if cascade:
                        self._delete_message_attachment_bindings_locked(message_ids)
                        self._conn.execute(
                            f"DELETE FROM messages WHERE session_key IN ({placeholders})",
                            targets,
                        )
                        self._conn.execute(
                            f"DELETE FROM turns WHERE session_key IN ({placeholders})",
                            targets,
                        )
                        self._conn.execute(
                            f"DELETE FROM session_compactions WHERE session_key IN ({placeholders})",
                            targets,
                        )
                        self._conn.execute(
                            f"DELETE FROM session_compaction_prepares "
                            f"WHERE session_key IN ({placeholders})",
                            targets,
                        )
                    self._conn.execute(
                        f"DELETE FROM channel_identities "
                        f"WHERE (channel || ':' || chat_id) IN ({placeholders})",
                        targets,
                    )
                    cur = self._conn.execute(
                        f"DELETE FROM sessions WHERE key IN ({placeholders})",
                        targets,
                    )
                    audit = self._delete_audit_value(
                        audit_id=audit_id,
                        targets=targets,
                        message_ids=message_ids,
                        compactions=compactions,
                        action_source=action_source,
                        cascade=cascade,
                        backup_path=backup_path,
                        started_at=started_at,
                        result="committed",
                        deleted_count=int(cur.rowcount or 0),
                    )
                    self._insert_delete_audit_locked(audit)
                    self._conn.commit()
                    return audit
                except BaseException:
                    self._conn.rollback()
                    raise
        except BaseException as exc:
            failed = self._delete_audit_value(
                audit_id=audit_id,
                targets=targets,
                message_ids=message_ids,
                compactions=compactions,
                action_source=action_source,
                cascade=cascade,
                backup_path=backup_path,
                started_at=started_at,
                result="rejected" if isinstance(exc, ValueError) else "failed",
                deleted_count=0,
                error=str(exc),
            )
            self._persist_delete_audit(failed)
            setattr(exc, "audit_id", audit_id)
            if isinstance(
                exc,
                (SessionAdmissionConflictError, SessionCompactionPrepareConflictError),
            ):
                exc.audit_id = audit_id
            raise

    def delete_session_with_audit(
        self,
        key: str,
        *,
        cascade: bool = False,
        action_source: str = "session.store.delete_session",
    ) -> SessionDeleteAudit:
        return self._delete_sessions_with_audit(
            self._normalize_delete_targets([key]),
            cascade=cascade,
            action_source=action_source,
        )

    def delete_session(
        self,
        key: str,
        *,
        cascade: bool = False,
        action_source: str = "session.store.delete_session",
    ) -> bool:
        return (
            self.delete_session_with_audit(
                key,
                cascade=cascade,
                action_source=action_source,
            ).result
            == "committed"
        )

    def delete_sessions_batch_with_audit(
        self,
        keys: list[str],
        *,
        cascade: bool = False,
        action_source: str = "session.store.delete_sessions_batch",
    ) -> SessionDeleteAudit:
        return self._delete_sessions_with_audit(
            self._normalize_delete_targets(keys),
            cascade=cascade,
            action_source=action_source,
        )

    def delete_sessions_batch(
        self,
        keys: list[str],
        *,
        cascade: bool = False,
        action_source: str = "session.store.delete_sessions_batch",
    ) -> int:
        return self.delete_sessions_batch_with_audit(
            keys,
            cascade=cascade,
            action_source=action_source,
        ).deleted_count

    def acquire_session_admission(self, key: str, admission_id: str) -> bool:
        """仅在会话仍存在时创建处理租约，并阻止并发删除。"""

        # 1. 用写事务串行化“存在校验”和租约创建
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT 1 FROM sessions WHERE key = ?",
                    (key,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False

                # 2. 租约落库后，其他连接的删除操作才能继续竞争写锁
                self._conn.execute(
                    """
                    INSERT INTO session_admissions(admission_id, session_key, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (admission_id, key, datetime.now(UTC).isoformat()),
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return True

    def release_session_admission(self, admission_id: str) -> None:
        """释放已完成入站消息持有的会话处理租约。"""

        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM session_admissions WHERE admission_id = ?",
                (admission_id,),
            )
            self._conn.commit()
        if cur.rowcount != 1:
            raise RuntimeError(f"session admission 不存在: {admission_id}")

    def clear_session_admissions(self) -> None:
        """在唯一 runtime 启动时清理上次异常退出遗留的处理租约。"""

        with self._lock:
            self._conn.execute("DELETE FROM session_admissions")
            self._conn.commit()

    def _require_sessions_not_admitted_locked(self, keys: list[str]) -> None:
        placeholders = ",".join("?" for _ in keys)
        row = self._conn.execute(
            f"""
            SELECT session_key
            FROM session_admissions
            WHERE session_key IN ({placeholders})
            LIMIT 1
            """,
            tuple(keys),
        ).fetchone()
        if row is not None:
            raise SessionAdmissionConflictError(str(row["session_key"]))

    def update_presence(
        self,
        key: str,
        *,
        last_user_at: str | None = None,
        last_proactive_at: str | None = None,
    ) -> None:
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (
                    key,
                    created_at,
                    updated_at,
                    last_consolidated,
                    metadata,
                    last_user_at,
                    last_proactive_at
                )
                VALUES (?, ?, ?, 0, '{}', ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    last_user_at = COALESCE(excluded.last_user_at, sessions.last_user_at),
                    last_proactive_at = COALESCE(excluded.last_proactive_at, sessions.last_proactive_at)
                """,
                (key, now, now, last_user_at, last_proactive_at),
            )
            self._conn.commit()

    def get_presence(self, key: str) -> dict[str, str | None] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT last_user_at, last_proactive_at
                FROM sessions
                WHERE key = ?
                """,
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {
            "last_user_at": row["last_user_at"],
            "last_proactive_at": row["last_proactive_at"],
        }

    def list_presence(self) -> dict[str, dict[str, str | None]]:
        with self._lock:
            rows = self._conn.execute("""
                SELECT key, last_user_at, last_proactive_at
                FROM sessions
                WHERE last_user_at IS NOT NULL OR last_proactive_at IS NOT NULL
                """).fetchall()
        return {
            str(row["key"]): {
                "last_user_at": row["last_user_at"],
                "last_proactive_at": row["last_proactive_at"],
            }
            for row in rows
        }

    def most_recent_user_at(self) -> str | None:
        with self._lock:
            row = self._conn.execute("""
                SELECT MAX(last_user_at) AS last_user_at
                FROM sessions
                WHERE last_user_at IS NOT NULL
                """).fetchone()
        if row is None:
            return None
        return row["last_user_at"]

    def get_channel_metadata(self, channel: str) -> list[dict[str, Any]]:
        like_key = f"{channel}:%"
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT key, metadata, updated_at
                FROM sessions
                WHERE key LIKE ?
                ORDER BY updated_at ASC, key ASC
                """,
                (like_key,),
            ).fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"])
            chat_id = key.split(":", 1)[-1] if ":" in key else key
            results.append(
                {
                    "key": key,
                    "chat_id": chat_id,
                    "metadata": _decode_session_metadata(row["metadata"], key),
                    "updated_at": str(row["updated_at"]),
                }
            )
        return results

    def get_channel_identities(self, channel: str) -> dict[str, str]:
        """Return the unique durable identity owner for one channel."""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT identity, chat_id
                FROM channel_identities
                WHERE channel = ?
                ORDER BY identity
                """,
                (channel,),
            ).fetchall()
        return {str(row["identity"]): str(row["chat_id"]) for row in rows}

    def channel_identity_migration_completed(self, channel: str) -> bool:
        """Return whether legacy metadata has already lost routing authority."""

        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM channel_identity_migrations WHERE channel = ?",
                (channel,),
            ).fetchone()
        return row is not None

    def seed_channel_identities(
        self,
        channel: str,
        mapping: Mapping[str, tuple[str, str]],
    ) -> None:
        """Copy legacy metadata once without deleting any historical field."""

        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                migrated = self._conn.execute(
                    "SELECT 1 FROM channel_identity_migrations WHERE channel = ?",
                    (channel,),
                ).fetchone()
                if migrated is not None:
                    return
                self._conn.executemany(
                    """
                    INSERT INTO channel_identities(channel, identity, chat_id, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        (channel, identity, chat_id, updated_at)
                        for identity, (chat_id, updated_at) in mapping.items()
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO channel_identity_migrations(channel, migrated_at)
                    VALUES (?, ?)
                    """,
                    (channel, datetime.now(UTC).isoformat()),
                )

    def persist_channel_identity(
        self,
        *,
        channel: str,
        identity: str,
        chat_id: str,
        session_key: str,
        created_at: str,
        updated_at: str,
        metadata: Mapping[str, Any],
    ) -> ChannelIdentityWriteReceipt:
        """Atomically update target Session metadata and the unique identity row."""

        metadata_payload = json.dumps(dict(metadata), ensure_ascii=False)
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                previous = self._conn.execute(
                    """
                    SELECT chat_id, updated_at FROM channel_identities
                    WHERE channel = ? AND identity = ?
                    """,
                    (channel, identity),
                ).fetchone()
                previous_session = self._conn.execute(
                    """
                    SELECT created_at, updated_at, last_consolidated, metadata
                    FROM sessions
                    WHERE key = ?
                    """,
                    (session_key,),
                ).fetchone()
                previous_counts = self._session_lineage_counts_locked(session_key)
                occupied_versions = {
                    str(row["updated_at"])
                    for row in (previous, previous_session)
                    if row is not None
                }
                committed_updated_at = updated_at
                while committed_updated_at in occupied_versions:
                    committed_time = datetime.fromisoformat(committed_updated_at)
                    if committed_time.tzinfo is None:
                        committed_time = committed_time.replace(tzinfo=UTC)
                    committed_updated_at = (
                        committed_time.astimezone(UTC) + timedelta(microseconds=1)
                    ).isoformat()
                committed_created_at = (
                    created_at
                    if previous_session is None
                    else str(previous_session["created_at"])
                )
                committed_last_consolidated = (
                    0
                    if previous_session is None
                    else int(previous_session["last_consolidated"])
                )
                self._conn.execute(
                    """
                    INSERT INTO sessions (
                        key, created_at, updated_at, last_consolidated, metadata
                    ) VALUES (?, ?, ?, 0, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        metadata = excluded.metadata
                    """,
                    (
                        session_key,
                        created_at,
                        committed_updated_at,
                        metadata_payload,
                    ),
                )
                self._conn.execute(
                    """
                    INSERT INTO channel_identities(channel, identity, chat_id, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(channel, identity) DO UPDATE SET
                        chat_id = excluded.chat_id,
                        updated_at = excluded.updated_at
                    """,
                    (channel, identity, chat_id, committed_updated_at),
                )
                self._conn.execute(
                    """
                    INSERT INTO channel_identity_migrations(channel, migrated_at)
                    VALUES (?, ?)
                    ON CONFLICT(channel) DO NOTHING
                    """,
                    (channel, committed_updated_at),
                )
        return ChannelIdentityWriteReceipt(
            channel=channel,
            identity=identity,
            chat_id=chat_id,
            session_key=session_key,
            committed_created_at=committed_created_at,
            committed_updated_at=committed_updated_at,
            committed_last_consolidated=committed_last_consolidated,
            committed_metadata=metadata_payload,
            previous_identity_chat_id=(
                None if previous is None else str(previous["chat_id"])
            ),
            previous_identity_updated_at=(
                None if previous is None else str(previous["updated_at"])
            ),
            previous_session=(
                None
                if previous_session is None
                else (
                    str(previous_session["created_at"]),
                    str(previous_session["updated_at"]),
                    int(previous_session["last_consolidated"]),
                    (
                        None
                        if previous_session["metadata"] is None
                        else str(previous_session["metadata"])
                    ),
                )
            ),
            previous_lineage_counts=previous_counts,
        )

    def rollback_channel_identity(
        self,
        receipt: ChannelIdentityWriteReceipt,
    ) -> bool:
        """撤销仍由同一提交版本拥有的 identity/session 写入。"""

        if not isinstance(receipt, ChannelIdentityWriteReceipt):
            raise TypeError("channel identity rollback receipt 类型无效")
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                current_identity = self._conn.execute(
                    """
                    SELECT chat_id, updated_at FROM channel_identities
                    WHERE channel = ? AND identity = ?
                    """,
                    (receipt.channel, receipt.identity),
                ).fetchone()
                current_session = self._conn.execute(
                    """
                    SELECT created_at, updated_at, last_consolidated, metadata
                    FROM sessions WHERE key = ?
                    """,
                    (receipt.session_key,),
                ).fetchone()
                if (
                    current_identity is None
                    or current_session is None
                    or str(current_identity["chat_id"]) != receipt.chat_id
                    or str(current_identity["updated_at"])
                    != receipt.committed_updated_at
                    or str(current_session["created_at"])
                    != receipt.committed_created_at
                    or str(current_session["updated_at"])
                    != receipt.committed_updated_at
                    or int(current_session["last_consolidated"])
                    != receipt.committed_last_consolidated
                    or str(current_session["metadata"] or "")
                    != receipt.committed_metadata
                    or self._session_lineage_counts_locked(receipt.session_key)
                    != receipt.previous_lineage_counts
                ):
                    return False

                # 1. 恢复本 attempt 覆盖前的唯一 identity owner。
                if receipt.previous_identity_chat_id is None:
                    self._conn.execute(
                        """
                        DELETE FROM channel_identities
                        WHERE channel = ? AND identity = ?
                          AND chat_id = ? AND updated_at = ?
                        """,
                        (
                            receipt.channel,
                            receipt.identity,
                            receipt.chat_id,
                            receipt.committed_updated_at,
                        ),
                    )
                else:
                    self._conn.execute(
                        """
                        UPDATE channel_identities
                        SET chat_id = ?, updated_at = ?
                        WHERE channel = ? AND identity = ?
                          AND chat_id = ? AND updated_at = ?
                        """,
                        (
                            receipt.previous_identity_chat_id,
                            receipt.previous_identity_updated_at,
                            receipt.channel,
                            receipt.identity,
                            receipt.chat_id,
                            receipt.committed_updated_at,
                        ),
                    )

                # 2. 恢复既有 Session，或删除仅由本 attempt 创建的空壳。
                if receipt.previous_session is None:
                    self._conn.execute(
                        """
                        DELETE FROM sessions
                        WHERE key = ? AND updated_at = ? AND metadata = ?
                        """,
                        (
                            receipt.session_key,
                            receipt.committed_updated_at,
                            receipt.committed_metadata,
                        ),
                    )
                else:
                    (
                        previous_created_at,
                        previous_updated_at,
                        previous_last_consolidated,
                        previous_metadata,
                    ) = receipt.previous_session
                    self._conn.execute(
                        """
                        UPDATE sessions
                        SET created_at = ?, updated_at = ?,
                            last_consolidated = ?, metadata = ?
                        WHERE key = ? AND updated_at = ? AND metadata = ?
                        """,
                        (
                            previous_created_at,
                            previous_updated_at,
                            previous_last_consolidated,
                            previous_metadata,
                            receipt.session_key,
                            receipt.committed_updated_at,
                            receipt.committed_metadata,
                        ),
                    )
        return True

    def _session_lineage_counts_locked(
        self,
        session_key: str,
    ) -> tuple[int, ...]:
        """读取 rollback fence 需要的 Session 子状态计数。"""

        tables = (
            "messages",
            "turns",
            "session_compactions",
            "session_compaction_prepares",
            "session_admissions",
            "inbound_handoffs",
            "session_source_mutation_audits",
        )
        return tuple(
            int(
                self._conn.execute(
                    f"SELECT COUNT(1) FROM {table} WHERE session_key = ?",
                    (session_key,),
                ).fetchone()[0]
            )
            for table in tables
        )

    def count_messages(self, session_key: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(1) AS c FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
        return int((row["c"] if row else 0) or 0)

    def begin_attachment_import(
        self,
        *,
        artifact_id: str,
        storage_key: str,
        expected_size_bytes: int,
        expected_sha256: str,
        created_at: str,
    ) -> AttachmentImportRecord:
        """在写文件前持久化或复用完全相同的 attachment import intent。"""

        if _ATTACHMENT_ID_RE.fullmatch(artifact_id) is None:
            raise ValueError("attachment artifact_id 必须是 1..256 字符安全 identity")
        _attachment_storage_key(artifact_id, storage_key)
        if (
            not isinstance(expected_size_bytes, int)
            or isinstance(expected_size_bytes, bool)
            or expected_size_bytes < 0
        ):
            raise ValueError("attachment expected_size_bytes 必须是非负整数")
        if _ATTACHMENT_SHA256_RE.fullmatch(expected_sha256) is None:
            raise ValueError("attachment expected_sha256 必须是 64 位小写十六进制")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("attachment import created_at 不得为空")
        with self._lock:
            with self._conn:
                existing = self._conn.execute(
                    """
                    SELECT artifact_id, storage_key, expected_size_bytes,
                           expected_sha256, phase, created_at, updated_at, error
                    FROM attachment_imports
                    WHERE artifact_id = ?
                    """,
                    (artifact_id,),
                ).fetchone()
                if existing is not None:
                    if (
                        str(existing["storage_key"]) != storage_key
                        or int(existing["expected_size_bytes"])
                        != expected_size_bytes
                        or str(existing["expected_sha256"]) != expected_sha256
                    ):
                        raise RuntimeError(
                            f"attachment import identity 已漂移: {artifact_id}"
                        )
                    return AttachmentImportRecord(
                        artifact_id=str(existing["artifact_id"]),
                        storage_key=str(existing["storage_key"]),
                        expected_size_bytes=int(existing["expected_size_bytes"]),
                        expected_sha256=str(existing["expected_sha256"]),
                        phase=str(existing["phase"]),
                        created_at=str(existing["created_at"]),
                        updated_at=str(existing["updated_at"]),
                        error=(
                            None
                            if existing["error"] is None
                            else str(existing["error"])
                        ),
                    )
                self._conn.execute(
                    """
                    INSERT INTO attachment_imports (
                        artifact_id, storage_key, expected_size_bytes,
                        expected_sha256, phase, created_at, updated_at, error
                    ) VALUES (?, ?, ?, ?, 'prepared', ?, ?, NULL)
                    """,
                    (
                        artifact_id,
                        storage_key,
                        expected_size_bytes,
                        expected_sha256,
                        created_at,
                        created_at,
                    ),
                )
        record = self.attachment_import(artifact_id)
        if record is None:
            raise RuntimeError(f"attachment import 未返回记录: {artifact_id}")
        return record

    def mark_attachment_import_file_published(
        self,
        artifact_id: str,
        *,
        updated_at: str,
    ) -> None:
        """在目录 fsync 后持久化 file_published 恢复边界。"""

        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("attachment import updated_at 不得为空")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE attachment_imports
                    SET phase = 'file_published', updated_at = ?, error = NULL
                    WHERE artifact_id = ? AND phase = 'prepared'
                    """,
                    (updated_at, artifact_id),
                )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"attachment import 不在 prepared: {artifact_id}"
            )

    def record_attachment_import_error(
        self,
        artifact_id: str,
        *,
        error: str,
        updated_at: str,
    ) -> None:
        """记录非终态 import 错误，不删除或伪终结已有 bytes。"""

        if not isinstance(error, str) or not error:
            raise ValueError("attachment import error 不得为空")
        if not isinstance(updated_at, str) or not updated_at:
            raise ValueError("attachment import updated_at 不得为空")
        with self._lock:
            with self._conn:
                cur = self._conn.execute(
                    """
                    UPDATE attachment_imports
                    SET updated_at = ?, error = ?
                    WHERE artifact_id = ? AND phase != 'artifact_committed'
                    """,
                    (updated_at, error, artifact_id),
                )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"attachment import 不存在或已经 committed: {artifact_id}"
            )

    def attachment_import(self, artifact_id: str) -> AttachmentImportRecord | None:
        """读取一个 attachment import 的 durable 恢复状态。"""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT artifact_id, storage_key, expected_size_bytes,
                       expected_sha256, phase, created_at, updated_at, error
                FROM attachment_imports
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return AttachmentImportRecord(
            artifact_id=str(row["artifact_id"]),
            storage_key=str(row["storage_key"]),
            expected_size_bytes=int(row["expected_size_bytes"]),
            expected_sha256=str(row["expected_sha256"]),
            phase=str(row["phase"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error=None if row["error"] is None else str(row["error"]),
        )

    def incomplete_attachment_imports(self) -> tuple[AttachmentImportRecord, ...]:
        """列出所有不能自动删除或自动重绑的未完成 import。"""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT artifact_id
                FROM attachment_imports
                WHERE phase != 'artifact_committed'
                ORDER BY created_at, artifact_id
                """
            ).fetchall()
        records = tuple(
            self.attachment_import(str(row["artifact_id"])) for row in rows
        )
        return tuple(record for record in records if record is not None)

    def validate_attachment_metadata_integrity(self) -> AttachmentIntegrityReport:
        """验证 FK、message projection、binding 顺序与 import terminal state。"""

        with self._lock:
            # 1. SQLite FK 与 ready artifact metadata 必须自洽。
            foreign_key_errors = self._conn.execute(
                "PRAGMA foreign_key_check"
            ).fetchall()
            if foreign_key_errors:
                raise ValueError(
                    f"attachment foreign key check 失败: {len(foreign_key_errors)}"
                )
            artifacts = self._conn.execute(
                """
                SELECT artifact_id, storage_key, size_bytes, sha256, state
                FROM attachments
                ORDER BY artifact_id
                """
            ).fetchall()
            for artifact in artifacts:
                if (
                    str(artifact["state"]) != "ready"
                ):
                    raise ValueError(
                        "attachment metadata 非法: " + str(artifact["artifact_id"])
                    )
                _attachment_storage_key(
                    str(artifact["artifact_id"]),
                    str(artifact["storage_key"]),
                )
            import_rows = self._conn.execute(
                """
                SELECT artifact_id, storage_key, expected_size_bytes,
                       expected_sha256, phase
                FROM attachment_imports
                ORDER BY artifact_id
                """
            ).fetchall()
            imports = {str(row["artifact_id"]): row for row in import_rows}
            artifact_ids = {str(row["artifact_id"]) for row in artifacts}
            for artifact in artifacts:
                artifact_id = str(artifact["artifact_id"])
                intent = imports.get(artifact_id)
                if (
                    intent is None
                    or str(intent["phase"]) != "artifact_committed"
                    or str(intent["storage_key"]) != str(artifact["storage_key"])
                    or int(intent["expected_size_bytes"])
                    != int(artifact["size_bytes"])
                    or str(intent["expected_sha256"]) != str(artifact["sha256"])
                ):
                    raise ValueError(
                        f"attachment committed intent 已漂移: {artifact_id}"
                    )
            terminal_without_artifact = sorted(
                artifact_id
                for artifact_id, intent in imports.items()
                if str(intent["phase"]) == "artifact_committed"
                and artifact_id not in artifact_ids
            )
            if terminal_without_artifact:
                raise ValueError(
                    "attachment committed intent 缺少 artifact: "
                    + ", ".join(terminal_without_artifact)
                )

            # 2. durable binding 是 owner；message extra 只能是完全一致的投影。
            messages = self._conn.execute(
                """
                SELECT id, role, extra
                FROM messages
                ORDER BY session_key, seq
                """
            ).fetchall()
            binding_rows = self._conn.execute(
                """
                SELECT message_id, ordinal, artifact_id, direction
                FROM message_attachments
                ORDER BY message_id, ordinal
                """
            ).fetchall()
            bindings: dict[str, list[sqlite3.Row]] = {}
            for binding in binding_rows:
                bindings.setdefault(str(binding["message_id"]), []).append(binding)
            bound_message_count = 0
            for message in messages:
                message_id = str(message["id"])
                projection = _message_attachment_ids(
                    _decode_message_extra(message["extra"], message_id),
                    message_id,
                )
                durable = tuple(
                    str(binding["artifact_id"])
                    for binding in bindings.get(message_id, [])
                )
                if projection != durable:
                    raise ValueError(
                        f"message attachment projection 已漂移: {message_id}"
                    )
                if durable:
                    bound_message_count += 1
                    if message["role"] not in {"user", "assistant"}:
                        raise ValueError(
                            f"带 attachment 的 message role 无效: {message_id}"
                        )
                    expected_direction = (
                        "inbound" if message["role"] == "user" else "outbound"
                    )
                    if any(
                        str(binding["direction"]) != expected_direction
                        for binding in bindings[message_id]
                    ):
                        raise ValueError(
                            f"message attachment direction 已漂移: {message_id}"
                        )
            incomplete_rows = self._conn.execute(
                """
                SELECT artifact_id
                FROM attachment_imports
                WHERE phase != 'artifact_committed'
                ORDER BY created_at, artifact_id
                """
            ).fetchall()
        return AttachmentIntegrityReport(
            artifact_count=len(artifacts),
            binding_count=len(binding_rows),
            bound_message_count=bound_message_count,
            incomplete_import_ids=tuple(
                str(row["artifact_id"]) for row in incomplete_rows
            ),
        )

    def register_ready_attachment(
        self,
        *,
        artifact_id: str,
        storage_key: str,
        kind: str,
        filename: str | None,
        media_type: str | None,
        size_bytes: int,
        sha256: str,
        created_at: str,
    ) -> AttachmentArtifactRecord:
        """登记一个已原子发布且不可变的附件 artifact。"""

        # 1. SessionDB 只接受 Core artifact namespace 的稳定元数据。
        if _ATTACHMENT_ID_RE.fullmatch(artifact_id) is None:
            raise ValueError("attachment artifact_id 必须是 1..256 字符安全 identity")
        _attachment_storage_key(artifact_id, storage_key)
        if kind not in {"image", "file"}:
            raise ValueError("attachment kind 必须是 image 或 file")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise ValueError("attachment size_bytes 必须是非负整数")
        if _ATTACHMENT_SHA256_RE.fullmatch(sha256) is None:
            raise ValueError("attachment sha256 必须是 64 位小写十六进制")
        if filename is not None and (
            not isinstance(filename, str)
            or not filename
            or filename != filename.strip()
            or len(filename) > 255
            or "/" in filename
            or "\\" in filename
            or "\x00" in filename
            or any(ord(char) < 32 or ord(char) == 127 for char in filename)
        ):
            raise ValueError("attachment filename 必须是 1..255 字符的纯文件名或 None")
        if media_type is not None and (
            not isinstance(media_type, str)
            or len(media_type) > 255
            or _ATTACHMENT_MEDIA_TYPE_RE.fullmatch(media_type) is None
        ):
            raise ValueError("attachment media_type 必须是合法 MIME type 或 None")
        if not isinstance(created_at, str) or not created_at:
            raise ValueError("attachment created_at 不得为空")

        # 2. INSERT-only publication 禁止复用 identity 或 storage path。
        with self._lock:
            with self._conn:
                intent = self._conn.execute(
                    """
                    SELECT storage_key, expected_size_bytes, expected_sha256, phase
                    FROM attachment_imports
                    WHERE artifact_id = ?
                    """,
                    (artifact_id,),
                ).fetchone()
                if intent is None:
                    raise RuntimeError(
                        f"attachment 缺少 durable import intent: {artifact_id}"
                    )
                if str(intent["phase"]) != "file_published":
                    raise RuntimeError(
                        f"attachment file 尚未发布: {artifact_id}:{intent['phase']}"
                    )
                if (
                    str(intent["storage_key"]) != storage_key
                    or int(intent["expected_size_bytes"]) != size_bytes
                    or str(intent["expected_sha256"]) != sha256
                ):
                    raise RuntimeError(
                        f"attachment publication 与 durable intent 不一致: {artifact_id}"
                    )
                self._conn.execute(
                    """
                    INSERT INTO attachments (
                        artifact_id, storage_key, kind, filename, media_type,
                        size_bytes, sha256, state, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', ?)
                    """,
                    (
                        artifact_id,
                        storage_key,
                        kind,
                        filename,
                        media_type,
                        size_bytes,
                        sha256,
                        created_at,
                    ),
                )
                self._conn.execute(
                    """
                    UPDATE attachment_imports
                    SET phase = 'artifact_committed', updated_at = ?, error = NULL
                    WHERE artifact_id = ? AND phase = 'file_published'
                    """,
                    (created_at, artifact_id),
                )
        record = self.get_attachment(artifact_id)
        if record is None:
            raise RuntimeError(f"attachment publication 未返回记录: {artifact_id}")
        return record

    def get_attachment(self, artifact_id: str) -> AttachmentArtifactRecord | None:
        """读取一个 artifact 的权威 metadata。"""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT artifact_id, storage_key, kind, filename, media_type,
                       size_bytes, sha256, state, created_at
                FROM attachments
                WHERE artifact_id = ?
                """,
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        return AttachmentArtifactRecord(
            artifact_id=str(row["artifact_id"]),
            storage_key=str(row["storage_key"]),
            kind=str(row["kind"]),
            filename=None if row["filename"] is None else str(row["filename"]),
            media_type=None if row["media_type"] is None else str(row["media_type"]),
            size_bytes=int(row["size_bytes"]),
            sha256=str(row["sha256"]),
            state=str(row["state"]),
            created_at=str(row["created_at"]),
        )

    def list_attachments(self) -> tuple[AttachmentArtifactRecord, ...]:
        """按 identity 列出 ready artifact metadata，不暴露任意 SQL。"""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT artifact_id, storage_key, kind, filename, media_type,
                       size_bytes, sha256, state, created_at
                FROM attachments
                ORDER BY artifact_id
                """
            ).fetchall()
        return tuple(
            AttachmentArtifactRecord(
                artifact_id=str(row["artifact_id"]),
                storage_key=str(row["storage_key"]),
                kind=str(row["kind"]),
                filename=(
                    None if row["filename"] is None else str(row["filename"])
                ),
                media_type=(
                    None if row["media_type"] is None else str(row["media_type"])
                ),
                size_bytes=int(row["size_bytes"]),
                sha256=str(row["sha256"]),
                state=str(row["state"]),
                created_at=str(row["created_at"]),
            )
            for row in rows
        )

    def message_attachment_ids(self, message_id: str) -> tuple[str, ...]:
        """按消息内顺序读取 durable attachment bindings。"""

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT artifact_id
                FROM message_attachments
                WHERE message_id = ?
                ORDER BY ordinal
                """,
                (message_id,),
            ).fetchall()
        return tuple(str(row["artifact_id"]) for row in rows)

    def _require_ready_attachments_locked(
        self,
        artifact_ids: set[str],
    ) -> None:
        """在当前消息事务内确认全部 artifact 已发布。"""

        if not artifact_ids:
            return
        placeholders = ",".join("?" for _ in artifact_ids)
        rows = self._conn.execute(
            f"""
            SELECT artifact_id
            FROM attachments
            WHERE artifact_id IN ({placeholders}) AND state = 'ready'
            """,
            tuple(sorted(artifact_ids)),
        ).fetchall()
        ready = {str(row["artifact_id"]) for row in rows}
        missing = sorted(artifact_ids - ready)
        if missing:
            raise ValueError("message 引用了未发布的 attachment: " + ", ".join(missing))

    def _delete_message_attachment_bindings_locked(
        self,
        message_ids: Sequence[str],
    ) -> None:
        """只删除消息绑定，永不删除 artifact metadata 或 bytes。"""

        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        self._conn.execute(
            f"DELETE FROM message_attachments WHERE message_id IN ({placeholders})",
            tuple(message_ids),
        )

    def _next_seq_locked(self, session_key: str) -> int:
        meta = self._conn.execute(
            "SELECT next_seq FROM sessions WHERE key = ?",
            (session_key,),
        ).fetchone()
        row = self._conn.execute(
            "SELECT COALESCE(MAX(seq) + 1, 0) AS next_seq FROM messages WHERE session_key = ?",
            (session_key,),
        ).fetchone()
        from_messages = int((row["next_seq"] if row else 0) or 0)
        if meta is None:
            return from_messages
        return max(int(meta["next_seq"] or 0), from_messages)

    def next_seq(self, session_key: str) -> int:
        with self._lock:
            return self._next_seq_locked(session_key)

    def insert_message(
        self,
        session_key: str,
        *,
        role: str,
        content: str,
        ts: str,
        seq: int,
        tool_chain: Any | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        message_id = f"{session_key}:{seq}"
        normalized_extra = extra or {}
        _validate_new_message_extra(role, normalized_extra, message_id)
        binding_rows = _message_attachment_binding_rows(
            message_id=message_id,
            role=role,
            extra=normalized_extra,
        )
        tool_chain_payload = (
            json.dumps(tool_chain, ensure_ascii=False)
            if tool_chain is not None
            else None
        )
        extra_payload = json.dumps(normalized_extra, ensure_ascii=False)
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._require_ready_attachments_locked(
                    {row[2] for row in binding_rows}
                )
                self._conn.execute(
                    """
                    INSERT INTO messages (
                        id, session_key, seq, role, content, tool_chain, extra, ts
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        session_key,
                        seq,
                        role,
                        content,
                        tool_chain_payload,
                        extra_payload,
                        ts,
                    ),
                )
                if binding_rows:
                    self._conn.executemany(
                        """
                        INSERT INTO message_attachments (
                            message_id, ordinal, artifact_id, direction
                        ) VALUES (?, ?, ?, ?)
                        """,
                        binding_rows,
                    )
                self._conn.execute(
                    """
                    UPDATE sessions
                    SET next_seq = CASE WHEN next_seq < ? THEN ? ELSE next_seq END
                    WHERE key = ?
                    """,
                    (int(seq) + 1, int(seq) + 1, session_key),
                )
        row = {
            "id": message_id,
            "session_key": session_key,
            "seq": seq,
            "role": role,
            "content": content,
            "timestamp": ts,
        }
        if tool_chain is not None:
            row["tool_chain"] = tool_chain
        if normalized_extra:
            row.update(normalized_extra)
        return row

    def _prepare_message_batch(
        self,
        key: str,
        *,
        start_seq: int,
        messages: list[dict[str, Any]],
    ) -> tuple[
        list[tuple[Any, ...]],
        list[dict[str, Any]],
        list[tuple[str, int, str, str]],
    ]:
        """序列化待写消息并构造提交成功后的内存行。"""

        insert_rows: list[tuple[Any, ...]] = []
        result_rows: list[dict[str, Any]] = []
        binding_rows: list[tuple[str, int, str, str]] = []

        # 1. 先完成序列化，失败时不改变数据库和内存消息。
        for offset, message in enumerate(messages):
            seq = start_seq + offset
            message_id = f"{key}:{seq}"
            tool_chain = message.get("tool_chain")
            extra = message["extra"]
            _validate_new_message_extra(message.get("role"), extra, message_id)
            binding_rows.extend(
                _message_attachment_binding_rows(
                    message_id=message_id,
                    role=message.get("role"),
                    extra=extra,
                )
            )
            tool_chain_payload = (
                json.dumps(tool_chain, ensure_ascii=False)
                if tool_chain is not None
                else None
            )
            insert_rows.append(
                (
                    message_id,
                    key,
                    seq,
                    str(message["role"]),
                    str(message["content"]),
                    tool_chain_payload,
                    json.dumps(extra, ensure_ascii=False),
                    str(message["timestamp"]),
                )
            )
            row = {
                "id": message_id,
                "session_key": key,
                "seq": seq,
                "role": str(message["role"]),
                "content": str(message["content"]),
                "timestamp": str(message["timestamp"]),
            }
            if tool_chain is not None:
                row["tool_chain"] = tool_chain
            row.update(extra)
            result_rows.append(row)
        return insert_rows, result_rows, binding_rows

    def persist_session(
        self,
        key: str,
        *,
        created_at: str,
        updated_at: str,
        metadata: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """原子更新 session 元数据并追加新消息。"""

        metadata_payload = json.dumps(metadata, ensure_ascii=False)
        result_rows: list[dict[str, Any]] = []

        # 1. 元数据和消息必须同成同败，避免磁盘留下半个 turn。
        with self._lock:
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    """
                    INSERT INTO sessions (key, created_at, updated_at, last_consolidated, metadata)
                    VALUES (?, ?, ?, 0, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        updated_at = excluded.updated_at,
                        metadata = excluded.metadata
                    """,
                    (key, created_at, updated_at, metadata_payload),
                )
                if messages:
                    start_seq = self._next_seq_locked(key)
                    insert_rows, result_rows, binding_rows = self._prepare_message_batch(
                        key,
                        start_seq=start_seq,
                        messages=messages,
                    )
                    self._require_ready_attachments_locked(
                        {row[2] for row in binding_rows}
                    )
                    self._conn.executemany(
                        """
                        INSERT INTO messages (id, session_key, seq, role, content, tool_chain, extra, ts)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                    if binding_rows:
                        self._conn.executemany(
                            """
                            INSERT INTO message_attachments (
                                message_id, ordinal, artifact_id, direction
                            ) VALUES (?, ?, ?, ?)
                            """,
                            binding_rows,
                        )
                    next_seq = start_seq + len(insert_rows)
                    self._conn.execute(
                        """
                        UPDATE sessions
                        SET next_seq = CASE WHEN next_seq < ? THEN ? ELSE next_seq END
                        WHERE key = ?
                        """,
                        (next_seq, next_seq, key),
                    )
        return result_rows

    def fetch_session_messages(self, session_key: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ?
                ORDER BY seq ASC
                """,
                (session_key,),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def mobile_history_snapshot(self, session_key: str) -> tuple[int, int]:
        """返回当前会话历史的消息数与不可变 seq 高水位。"""

        with self._lock:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS total, COALESCE(MAX(seq), -1) AS max_seq
                FROM messages
                WHERE session_key = ?
                """,
                (session_key,),
            ).fetchone()
        if row is None:
            raise RuntimeError("mobile history snapshot 查询未返回结果")
        return int(row["total"]), int(row["max_seq"])

    def mobile_history_count_through(self, session_key: str, through_seq: int) -> int:
        """统计固定 seq 高水位内仍存在的历史消息。"""

        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS total FROM messages WHERE session_key = ? AND seq <= ?",
                (session_key, through_seq),
            ).fetchone()
        if row is None:
            raise RuntimeError("mobile history count 查询未返回结果")
        return int(row["total"])

    def list_mobile_history_page(
        self,
        *,
        session_key: str,
        after_seq: int,
        through_seq: int,
        page_size: int,
    ) -> list[dict[str, Any]]:
        """在固定 seq 高水位内按游标读取一页历史。"""

        safe_page_size = max(1, min(int(page_size), 200))
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ? AND seq > ? AND seq <= ?
                ORDER BY seq ASC, id ASC
                LIMIT ?
                """,
                (session_key, after_seq, through_seq, safe_page_size),
            ).fetchall()
        return [self._row_to_message(row) for row in rows]

    def list_messages_for_dashboard(
        self,
        *,
        session_key: str | None = None,
        q: str = "",
        role: str = "",
        page: int = 1,
        page_size: int = 25,
        sort_by: str = "ts",
        sort_order: str = "desc",
    ) -> tuple[list[dict[str, Any]], int]:
        safe_page = max(1, int(page))
        safe_page_size = max(1, min(int(page_size), 200))
        offset = (safe_page - 1) * safe_page_size
        safe_sort = "ASC" if str(sort_order).lower() == "asc" else "DESC"
        safe_sort_by = (
            sort_by if sort_by in {"ts", "seq", "role", "session_key"} else "ts"
        )

        params: list[Any] = []
        where_parts: list[str] = []
        if session_key:
            where_parts.append("session_key = ?")
            params.append(session_key)
        term = (q or "").strip()
        if term:
            where_parts.append("content LIKE ?")
            params.append(f"%{term}%")
        if role:
            where_parts.append("role = ?")
            params.append(role)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        count_sql = f"SELECT COUNT(1) AS c FROM messages {where_sql}"
        data_sql = f"""
            SELECT id, session_key, seq, role, content, tool_chain, extra, ts
            FROM messages
            {where_sql}
            ORDER BY {safe_sort_by} {safe_sort}, seq {safe_sort}, id ASC
            LIMIT ? OFFSET ?
        """
        with self._lock:
            count_row = self._conn.execute(count_sql, tuple(params)).fetchone()
            rows = self._conn.execute(
                data_sql,
                tuple([*params, safe_page_size, offset]),
            ).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [self._row_to_message(row) for row in rows], total

    def list_chat_history_page(
        self,
        *,
        session_key: str,
        before_seq: int | None,
        page_size: int,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        """读取会话尾页或指定 seq 之前的一页消息。"""

        # 1. 用不可复用的 seq 游标固定本页边界
        params: list[Any] = [session_key]
        before_sql = ""
        if before_seq is not None:
            before_sql = "AND seq < ?"
            params.append(before_seq)

        # 2. 倒序多取一条判断前页，再恢复为展示顺序
        with self._lock:
            count_row = self._conn.execute(
                "SELECT COUNT(1) AS c FROM messages WHERE session_key = ?",
                (session_key,),
            ).fetchone()
            rows = self._conn.execute(
                f"""
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ? {before_sql}
                ORDER BY seq DESC, id DESC
                LIMIT ?
                """,
                tuple([*params, page_size + 1]),
            ).fetchall()
        has_more = len(rows) > page_size
        visible_rows = list(reversed(rows[:page_size]))
        total = int((count_row["c"] if count_row else 0) or 0)
        return [self._row_to_message(row) for row in visible_rows], total, has_more

    def media_path_exists(self, path: str | Path) -> bool:
        target = _resolve_path_text(path)
        if not target:
            return False
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, extra FROM messages WHERE extra LIKE ?",
                ('%"media"%',),
            ).fetchall()
        for row in rows:
            message_id = str(row["id"])
            extra = _decode_message_extra(row["extra"], message_id)
            if "media" not in extra:
                continue
            media = extra["media"]
            if not isinstance(media, list) or not all(
                isinstance(item, str) for item in media
            ):
                raise ValueError(f"message media 必须是字符串数组: {message_id}")
            for item in cast(list[str], media):
                if _resolve_path_text(item) == target:
                    return True
        return False

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE id = ?
                """,
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_message(row)

    def get_message_by_client_id(
        self,
        session_key: str,
        client_message_id: str,
    ) -> dict[str, Any] | None:
        """按会话和移动端消息 ID 解析唯一的 canonical 消息。"""

        # 1. client_message_id 只存在于受校验的 extra JSON 中
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ?
                  AND json_extract(extra, '$.client_message_id') = ?
                ORDER BY seq ASC
                LIMIT 2
                """,
                (session_key, client_message_id),
            ).fetchall()

        # 2. 重复标识违反移动消息幂等契约，不能静默选中其中一条
        if len(rows) > 1:
            raise RuntimeError(
                f"同一会话存在重复 client_message_id: {session_key} {client_message_id}"
            )
        return None if not rows else self._row_to_message(rows[0])

    def has_turn_user_input_by_client_id(
        self,
        session_key: str,
        client_message_id: str,
    ) -> bool:
        """判断移动入站是否已经进入任一 durable execution attempt。"""

        # 1. turns.items_json 是中断前用户输入的权威落点；只匹配 user item。
        with self._lock:
            row = self._conn.execute(
                """
                SELECT 1
                FROM turns AS turn_record, json_each(turn_record.items_json) AS item
                WHERE turn_record.session_key = ?
                  AND json_extract(item.value, '$.type') = 'userMessage'
                  AND json_extract(
                        item.value,
                        '$.data.metadata.client_message_id'
                      ) = ?
                LIMIT 1
                """,
                (session_key, client_message_id),
            ).fetchone()
        return row is not None

    def find_turn_by_client_message_id(
        self,
        session_key: str,
        client_message_id: str,
    ) -> TurnRecord | None:
        """按 turns.items_json 的 userMessage client_message_id 返回唯一 turn。

        阶段1：0 条匹配返回 None，调用方按未建立 turn 正常准入；
        阶段2：唯一匹配返回该权威 TurnRecord；
        阶段3：同一会话同 client_message_id 命中多条 turn 违反幂等契约，
        fail-loud，绝不静默选中其中一条。
        """

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, status, input_json, items_json,
                       usage_json, error_json, final_response,
                       created_at, started_at, completed_at
                FROM turns AS turn_record
                WHERE turn_record.session_key = ?
                  AND EXISTS (
                      SELECT 1 FROM json_each(turn_record.items_json) AS item
                      WHERE json_extract(item.value, '$.type') = 'userMessage'
                        AND json_extract(
                              item.value,
                              '$.data.metadata.client_message_id'
                            ) = ?
                  )
                ORDER BY turn_record.created_at DESC, turn_record.id DESC
                LIMIT 2
                """,
                (session_key, client_message_id),
            ).fetchall()
        if len(rows) > 1:
            raise RuntimeError(
                f"同一会话存在重复 client_message_id turn: {session_key} {client_message_id}"
            )
        return None if not rows else self._row_to_turn(rows[0])

    def get_message_by_delivery_id(
        self,
        session_key: str,
        delivery_id: str,
    ) -> dict[str, Any] | None:
        """按会话和主动投递 ID 解析唯一的 canonical 消息。"""

        # 1. 只允许主动 assistant 消息拥有可引用的投递身份
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, session_key, seq, role, content, tool_chain, extra, ts
                FROM messages
                WHERE session_key = ?
                  AND role = 'assistant'
                  AND json_extract(extra, '$.proactive') = 1
                  AND json_extract(extra, '$.delivery_id') = ?
                ORDER BY seq ASC
                LIMIT 2
                """,
                (session_key, delivery_id),
            ).fetchall()

        # 2. 重复身份违反一次主动投递只产生一条消息的契约
        if len(rows) > 1:
            raise RuntimeError(
                f"同一会话存在重复 delivery_id: {session_key} {delivery_id}"
            )
        return None if not rows else self._row_to_message(rows[0])

    def update_message(
        self,
        message_id: str,
        *,
        role: str | None = None,
        content: str | None = None,
        tool_chain: Any | None = None,
        extra: dict[str, Any] | None = None,
        ts: str | None = None,
        action_source: str = "session.store.message_edit",
    ) -> dict[str, Any] | None:
        set_parts: list[str] = []
        params: list[Any] = []
        if role is not None:
            set_parts.append("role = ?")
            params.append(role)
        if content is not None:
            set_parts.append("content = ?")
            params.append(content)
        if tool_chain is not None:
            set_parts.append("tool_chain = ?")
            params.append(json.dumps(tool_chain, ensure_ascii=False))
        if extra is not None:
            set_parts.append("extra = ?")
            params.append(json.dumps(extra, ensure_ascii=False))
        if ts is not None:
            set_parts.append("ts = ?")
            params.append(ts)
        if not set_parts:
            return self.get_message(message_id)
        normalized_source = self._normalize_action_source(action_source)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT session_key, role, content FROM messages WHERE id = ?",
                    (message_id,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                session_key = str(row["session_key"])
                self._require_no_pending_compaction_prepare_locked([session_key])
                self._require_sessions_not_admitted_locked([session_key])
                if extra is not None:
                    _validate_new_message_extra(
                        role if role is not None else row["role"],
                        extra,
                        message_id,
                    )
                    attachment_rows = self._conn.execute(
                        """
                        SELECT artifact_id
                        FROM message_attachments
                        WHERE message_id = ?
                        ORDER BY ordinal
                        """,
                        (message_id,),
                    ).fetchall()
                    durable_ids = tuple(
                        str(attachment_row["artifact_id"])
                        for attachment_row in attachment_rows
                    )
                    if _message_attachment_ids(extra, message_id) != durable_ids:
                        raise ValueError(
                            "message attachment binding 不允许由 message_edit 改写: "
                            f"{message_id}"
                        )
                if role is not None:
                    attachment_count = self._conn.execute(
                        """
                        SELECT COUNT(1) AS c
                        FROM message_attachments
                        WHERE message_id = ?
                        """,
                        (message_id,),
                    ).fetchone()
                    if int(attachment_count["c"] if attachment_count else 0) > 0:
                        if role not in {"user", "assistant"}:
                            raise ValueError(
                                "带 attachment 的 message role 只能是 user/assistant: "
                                f"{message_id}"
                            )
                        self._conn.execute(
                            """
                            UPDATE message_attachments
                            SET direction = ?
                            WHERE message_id = ?
                            """,
                            ("inbound" if role == "user" else "outbound", message_id),
                        )
                params.append(message_id)
                cur = self._conn.execute(
                    f"UPDATE messages SET {', '.join(set_parts)} WHERE id = ?",
                    tuple(params),
                )
                if content is not None and content != str(row["content"] or ""):
                    self._delete_message_embeddings_locked([message_id])
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE key = ?",
                    (datetime.now().astimezone().isoformat(), session_key),
                )
                self._record_source_mutation_locked(
                    operation="message_edit",
                    session_key=session_key,
                    message_ids=(message_id,),
                    action_source=normalized_source,
                    backup_path=None,
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        if cur.rowcount <= 0:
            return None
        return self.get_message(message_id)

    def delete_message(
        self,
        message_id: str,
        *,
        action_source: str = "session.store.message_delete",
    ) -> bool:
        normalized_source = self._normalize_action_source(action_source)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT session_key, extra FROM messages WHERE id = ?",
                    (message_id,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return False
                session_key = str(row["session_key"])
                self._require_no_pending_compaction_prepare_locked([session_key])
                control_turn_id = _message_control_turn_id(row["extra"], message_id)
                if control_turn_id is not None:
                    raise InteractionDeleteRequiredError(message_id, control_turn_id)
                self._require_sessions_not_admitted_locked([session_key])
                backup_path = self._backup_before_delete_locked("message-deletions")
                self._delete_message_attachment_bindings_locked([message_id])
                cur = self._conn.execute(
                    "DELETE FROM messages WHERE id = ?",
                    (message_id,),
                )
                self._delete_message_embeddings_locked([message_id])
                self._conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE key = ?",
                    (datetime.now().astimezone().isoformat(), session_key),
                )
                self._record_source_mutation_locked(
                    operation="message_delete",
                    session_key=session_key,
                    message_ids=(message_id,),
                    action_source=normalized_source,
                    backup_path=backup_path,
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return cur.rowcount > 0

    def delete_messages_batch(
        self,
        ids: list[str],
        *,
        action_source: str = "session.store.message_batch_delete",
    ) -> int:
        clean_ids = [
            str(message_id).strip() for message_id in ids if str(message_id).strip()
        ]
        if not clean_ids:
            return 0
        normalized_source = self._normalize_action_source(action_source)
        placeholders = ",".join("?" for _ in clean_ids)
        now = datetime.now().astimezone().isoformat()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                explicit_rows = self._conn.execute(
                    f"""
                    SELECT id, session_key, extra
                    FROM messages
                    WHERE id IN ({placeholders})
                    ORDER BY session_key, seq
                    """,
                    tuple(clean_ids),
                ).fetchall()
                grouped: dict[str, list[str]] = {}
                for row in explicit_rows:
                    grouped.setdefault(str(row["session_key"]), []).append(
                        str(row["id"])
                    )
                if not grouped:
                    self._conn.rollback()
                    return 0
                self._require_no_pending_compaction_prepare_locked(list(grouped))
                for row in explicit_rows:
                    message_id = str(row["id"])
                    control_turn_id = _message_control_turn_id(row["extra"], message_id)
                    if control_turn_id is not None:
                        raise InteractionDeleteRequiredError(
                            message_id,
                            control_turn_id,
                        )
                self._require_sessions_not_admitted_locked(list(grouped))
                backup_path = self._backup_before_delete_locked("message-deletions")
                self._delete_message_attachment_bindings_locked(clean_ids)
                cur = self._conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    tuple(clean_ids),
                )
                self._delete_message_embeddings_locked(clean_ids)
                for session_key in grouped:
                    self._conn.execute(
                        "UPDATE sessions SET updated_at = ? WHERE key = ?",
                        (now, session_key),
                    )
                for session_key in grouped:
                    self._record_source_mutation_locked(
                        operation="message_batch_delete",
                        session_key=session_key,
                        message_ids=tuple(grouped[session_key]),
                        action_source=normalized_source,
                        backup_path=backup_path,
                    )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
        return int(cur.rowcount or 0)

    def delete_interaction(
        self,
        control_turn_id: str,
        *,
        action_source: str = "session.store.interaction_delete",
        expected_latest_session_key: str | None = None,
    ) -> InteractionDeletion | None:
        """校验并原子撤销一个显式 interaction 及其派生 embedding。"""

        if not isinstance(control_turn_id, str) or not control_turn_id.strip():
            raise ValueError("control_turn_id 必须是非空字符串")
        normalized_turn_id = control_turn_id.strip()
        normalized_source = self._normalize_action_source(action_source)
        normalized_session_key = (
            None
            if expected_latest_session_key is None
            else expected_latest_session_key.strip()
        )
        if expected_latest_session_key is not None and not normalized_session_key:
            raise ValueError("expected_latest_session_key 必须是非空字符串")
        now = datetime.now().astimezone().isoformat()

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # 1. 解析完整 transcript，并拒绝跨 session 或非连续结构。
                rows = self._conn.execute(
                    """
                    SELECT id, session_key, seq, role, extra
                    FROM messages
                    WHERE json_extract(COALESCE(extra, '{}'), '$.control_turn_id') = ?
                    ORDER BY session_key, seq
                    """,
                    (normalized_turn_id,),
                ).fetchall()
                if not rows:
                    self._conn.rollback()
                    return None
                session_keys = {str(row["session_key"]) for row in rows}
                if len(session_keys) != 1:
                    raise ValueError(
                        f"interaction 跨越多个 session: {normalized_turn_id}"
                    )
                session_key = next(iter(session_keys))
                session_rows = self._conn.execute(
                    """
                    SELECT id, seq, role, extra
                    FROM messages
                    WHERE session_key = ?
                    ORDER BY seq
                    """,
                    (session_key,),
                ).fetchall()
                if normalized_session_key is not None:
                    if session_key != normalized_session_key:
                        raise RuntimeError(
                            "latest interaction session 已变化: "
                            f"{normalized_session_key}!={session_key}"
                        )
                    latest_turn_id = _latest_completed_interaction_id(session_rows)
                    if latest_turn_id != normalized_turn_id:
                        raise RuntimeError(
                            "latest interaction 已变化: "
                            f"{normalized_turn_id}!={latest_turn_id or 'none'}"
                        )
                positions = [
                    index
                    for index, row in enumerate(session_rows)
                    if _message_control_turn_id(row["extra"], str(row["id"]))
                    == normalized_turn_id
                ]
                if positions != list(range(positions[0], positions[-1] + 1)):
                    raise ValueError(
                        f"interaction transcript 不连续: {normalized_turn_id}"
                    )
                turn_rows = [session_rows[index] for index in positions]
                first_user_message_id = _validate_deletable_interaction(
                    normalized_turn_id,
                    turn_rows,
                )

                # 2. Cursor 只由 ledger provenance 驱动；迁移负责清理 legacy 值。
                meta = self._conn.execute(
                    "SELECT last_consolidated FROM sessions WHERE key = ?",
                    (session_key,),
                ).fetchone()
                if meta is None:
                    raise ValueError(f"interaction session 不存在: {session_key}")
                old_cursor = int(meta["last_consolidated"])
                new_cursor = old_cursor

                # 2. active admission 与删除共用同一写事务，避免当前 turn 在删除后提交。
                self._require_no_pending_compaction_prepare_locked([session_key])
                self._require_sessions_not_admitted_locked([session_key])
                message_ids = tuple(str(row["id"]) for row in turn_rows)
                backup_path = self._backup_before_interaction_delete_locked()

                # 3. 正文、逐消息 embedding 与游标在同一事务中提交。
                placeholders = ",".join("?" for _ in message_ids)
                self._delete_message_attachment_bindings_locked(message_ids)
                self._conn.execute(
                    f"DELETE FROM messages WHERE id IN ({placeholders})",
                    message_ids,
                )
                self._delete_message_embeddings_locked(list(message_ids))
                _first_invalidated, ledger_cursor = (
                    self._invalidate_compactions_for_messages_locked(
                        session_key,
                        set(message_ids),
                        reason=f"interaction_deleted:{normalized_turn_id}",
                    )
                )
                if _first_invalidated:
                    new_cursor = ledger_cursor
                self._conn.execute(
                    """
                    UPDATE sessions
                    SET last_consolidated = ?, updated_at = ?
                    WHERE key = ?
                    """,
                    (new_cursor, now, session_key),
                )
                audit = self._record_source_mutation_locked(
                    operation="interaction_delete",
                    session_key=session_key,
                    message_ids=message_ids,
                    action_source=normalized_source,
                    backup_path=backup_path,
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise

        deletion = InteractionDeletion(
            control_turn_id=normalized_turn_id,
            session_key=session_key,
            message_ids=message_ids,
            first_user_message_id=first_user_message_id,
            old_last_consolidated=old_cursor,
            new_last_consolidated=new_cursor,
            backup_path=str(backup_path),
            audit_id=audit.audit_id,
        )
        logger.info(
            "interaction deleted control_turn_id=%s session_key=%s messages=%d "
            "last_consolidated=%d->%d backup_path=%s",
            deletion.control_turn_id,
            deletion.session_key,
            len(deletion.message_ids),
            deletion.old_last_consolidated,
            deletion.new_last_consolidated,
            deletion.backup_path,
        )
        return deletion

    def latest_completed_interaction_id(self, session_key: str) -> str | None:
        """读取一个既有 Session 最后的 completed interaction identity。"""

        if not isinstance(session_key, str) or not session_key.strip():
            raise ValueError("session_key 必须是非空字符串")
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, seq, role, extra
                FROM messages
                WHERE session_key = ?
                ORDER BY seq
                """,
                (session_key.strip(),),
            ).fetchall()
        return _latest_completed_interaction_id(rows)

    def _backup_before_interaction_delete_locked(self) -> Path:
        """创建 interaction 删除前的完整 SessionDB SQLite 快照。"""

        return self._backup_before_delete_locked("interaction-deletions")

    def _backup_before_session_delete_locked(self) -> Path:
        """创建 session 删除前的完整 SessionDB SQLite 快照。"""

        return self._backup_before_delete_locked("session-deletions")

    def _backup_before_delete_locked(self, category: str) -> Path:
        """在事务尚未写入时创建并验证不可覆盖的 SQLite 快照。"""

        backup_root = Path(self.db_path).parent / "backups" / category
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_id = uuid4().hex
        backup_path = backup_root / f"sessions-{backup_id}.db"
        candidate_path = backup_root / f".sessions-{backup_id}.db.tmp"
        candidate_path.touch(mode=0o600, exist_ok=False)
        try:
            # 当前连接已进入 BEGIN IMMEDIATE；同一连接作为备份源会让 SQLite
            # backup API 永久等待，因此在事务尚未写入时通过独立连接读取快照。
            source = sqlite3.connect(self.db_path)
            try:
                target = sqlite3.connect(candidate_path)
                try:
                    source.backup(target)
                    rows = target.execute("PRAGMA integrity_check").fetchall()
                    if rows != [("ok",)]:
                        raise RuntimeError(
                            f"{category} 删除备份 integrity_check 失败: {rows[:3]}"
                        )
                finally:
                    target.close()
            finally:
                source.close()
            _ = candidate_path.replace(backup_path)
        except (OSError, RuntimeError, sqlite3.Error):
            candidate_path.unlink(missing_ok=True)
            raise
        return backup_path

    def fetch_by_ids_with_context(
        self, ids: list[str], context: int
    ) -> list[dict[str, Any]]:
        """Fetch messages by ID, expanding each hit by ±context rows in its session.

        Returns messages ordered by (session_key, seq).
        Each dict includes ``in_source_ref: bool`` to distinguish hits from context.
        """
        if not ids:
            return []
        if context == 0:
            result = self.fetch_by_ids(ids)
            for m in result:
                m["in_source_ref"] = True
            return result

        id_set = set(ids)
        session_seqs: dict[str, set[int]] = {}
        for msg_id in ids:
            parts = msg_id.rsplit(":", 1)
            if len(parts) != 2:
                continue
            sk, seq_str = parts
            try:
                seq = int(seq_str)
            except ValueError:
                continue
            if sk not in session_seqs:
                session_seqs[sk] = set()
            session_seqs[sk].add(seq)

        if not session_seqs:
            return []

        results: list[dict[str, Any]] = []
        with self._lock:
            for sk, seqs in session_seqs.items():
                expanded: set[int] = set()
                for seq in seqs:
                    for s in range(max(0, seq - context), seq + context + 1):
                        expanded.add(s)
                placeholders = ",".join("?" * len(expanded))
                rows = self._conn.execute(
                    f"SELECT id, session_key, seq, role, content, tool_chain, extra, ts "
                    f"FROM messages WHERE session_key = ? AND seq IN ({placeholders}) ORDER BY seq",
                    [sk, *expanded],
                ).fetchall()
                for row in rows:
                    msg = self._row_to_message(row)
                    msg["in_source_ref"] = msg["id"] in id_set
                    results.append(msg)
        return results

    def fetch_by_ids(self, ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        order_expr = " ".join(f"WHEN ? THEN {i}" for i in range(len(ids)))
        sql = (
            "SELECT id, session_key, seq, role, content, tool_chain, extra, ts FROM messages "
            f"WHERE id IN ({placeholders}) ORDER BY CASE id {order_expr} END"
        )
        with self._lock:
            rows = self._conn.execute(sql, tuple(ids + ids)).fetchall()
        return [self._row_to_message(row) for row in rows]

    def search_messages(
        self,
        query: str,
        *,
        session_key: str | None = None,
        role: str | None = None,
        limit: int = 10,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        limit = max(1, min(int(limit), 100))
        offset = max(0, int(offset))
        params: list[Any] = []
        where_parts: list[str] = []
        if session_key:
            where_parts.append("m.session_key = ?")
            params.append(session_key)
        if role:
            where_parts.append("m.role = ?")
            params.append(role)
        where_sql = f"WHERE {' AND '.join(where_parts)}" if where_parts else ""

        # Split into individual terms for both FTS and LIKE paths.
        terms = [t for t in query.split() if t]
        if not terms:
            terms = [query]

        term_conditions_or = " OR ".join("m.content LIKE ?" for _ in terms)
        score_expr = " + ".join(
            f"(CASE WHEN m.content LIKE ? THEN 1 ELSE 0 END)" for _ in terms
        )
        if self._has_fts:
            # 长词走 FTS，短词继续走 LIKE，再把两路结果合并去重。
            fts_terms = [t for t in terms if len(t) >= 3]
            if fts_terms:
                fts_query = " OR ".join(fts_terms)
                connector = "AND" if where_sql else "WHERE"
                count_params = [fts_query] + params[:]
                count_sql = (
                    "SELECT COUNT(1) AS c "
                    "FROM messages m "
                    "LEFT JOIN ("
                    "    SELECT rowid FROM messages_fts WHERE messages_fts MATCH ?"
                    ") fts ON m.rowid = fts.rowid "
                    f"{where_sql} {connector} (fts.rowid IS NOT NULL OR ({term_conditions_or})) "
                )
                count_params.extend(f"%{t}%" for t in terms)
                fts_params: list[Any] = []
                fts_sql = (
                    "SELECT m.id, m.session_key, m.seq, m.role, m.content, m.tool_chain, m.extra, m.ts, "
                    f"({score_expr}) AS match_score, "
                    "fts.rank_score AS rank_score "
                    "FROM messages m "
                    "LEFT JOIN ("
                    "    SELECT rowid, bm25(messages_fts) AS rank_score "
                    "    FROM messages_fts WHERE messages_fts MATCH ?"
                    ") fts ON m.rowid = fts.rowid "
                    f"{where_sql} {connector} (fts.rowid IS NOT NULL OR ({term_conditions_or})) "
                    "ORDER BY match_score DESC, "
                    "CASE WHEN rank_score IS NULL THEN 1 ELSE 0 END ASC, "
                    "rank_score ASC, m.seq DESC LIMIT ? OFFSET ?"
                )
                fts_params.extend(f"%{t}%" for t in terms)
                fts_params.append(fts_query)
                fts_params.extend(params[:])
                fts_params.extend(f"%{t}%" for t in terms)
                fts_params.extend([limit, offset])
                try:
                    with self._lock:
                        count_row = self._conn.execute(
                            count_sql, tuple(count_params)
                        ).fetchone()
                        rows = self._conn.execute(fts_sql, tuple(fts_params)).fetchall()
                    total = int((count_row["c"] if count_row else 0) or 0)
                    return [self._row_to_message(row) for row in rows], total
                except sqlite3.OperationalError:
                    pass

        # LIKE fallback: OR across all terms so any hit surfaces; rank by match count descending.
        like_params = params[:]
        count_params = params[:]
        connector = "AND" if where_sql else "WHERE"
        count_sql = f"SELECT COUNT(1) AS c FROM messages m {where_sql} {connector} ({term_conditions_or}) "
        count_params.extend(f"%{t}%" for t in terms)
        like_sql = (
            f"SELECT m.id, m.session_key, m.seq, m.role, m.content, m.tool_chain, m.extra, m.ts, "
            f"({score_expr}) AS match_score "
            f"FROM messages m {where_sql} {connector} ({term_conditions_or}) "
            f"ORDER BY match_score DESC, m.seq DESC LIMIT ? OFFSET ?"
        )
        # score_expr binds: one %t% per term; term_conditions_or binds: one %t% per term
        like_params.extend(f"%{t}%" for t in terms)  # for score_expr
        like_params.extend(f"%{t}%" for t in terms)  # for WHERE OR
        like_params.extend([limit, offset])
        with self._lock:
            count_row = self._conn.execute(count_sql, tuple(count_params)).fetchone()
            rows = self._conn.execute(like_sql, tuple(like_params)).fetchall()
        total = int((count_row["c"] if count_row else 0) or 0)
        return [self._row_to_message(row) for row in rows], total

    def _row_to_message(self, row: sqlite3.Row) -> dict[str, Any]:
        """校验并反序列化一行消息，拒绝损坏的 JSON 载荷。"""

        message_id = str(row["id"])
        role = row["role"]
        if role not in {"user", "assistant"}:
            raise ValueError(f"message role 无效: {message_id}")
        content = row["content"]
        if not isinstance(content, str):
            raise ValueError(f"message content 必须是字符串: {message_id}")
        timestamp = row["ts"]
        if not isinstance(timestamp, str):
            raise ValueError(f"message timestamp 必须是字符串: {message_id}")
        message: dict[str, Any] = {
            "id": message_id,
            "session_key": row["session_key"],
            "seq": int(row["seq"]),
            "role": role,
            "content": content,
            "timestamp": timestamp,
        }
        raw_tool_chain = row["tool_chain"]
        if raw_tool_chain is not None:
            tool_chain = _decode_message_tool_chain(raw_tool_chain, message_id)
            if tool_chain:
                message["tool_chain"] = tool_chain
        extra_dict = _decode_message_extra(row["extra"], message_id)
        if extra_dict:
            message.update(extra_dict)
        return message
