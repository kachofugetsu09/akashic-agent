"""把已提交的旧摘要接入普通 Summary owner；旧账本与原消息始终保留。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from pathlib import Path
from typing import Any, TypedDict, cast
from uuid import uuid4

from agent.migrations.session_db_backup import (
    backup_sqlite_database,
    validate_table_schema,
)
from plugins.compaction.records import ImportedSummaryRecord, SummaryRecords
from agent.plugin_contracts.context import summary_range
from session.log import MessageLog, OwnerTransaction
from session.message_codec import json_value

_OWNER = "plugin:compaction"
_RECEIPT = "migration:legacy-summary-v1"


class _Schema(TypedDict):
    columns: tuple[tuple[str, str, int, int], ...]
    named_indexes: dict[str, tuple[tuple[str, ...], int]]
    auto_indexes: tuple[tuple[str, tuple[str, ...]], ...]
    sql_fragments: tuple[str, ...]


_FINAL_LEDGER_SCHEMA: _Schema = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("created_at", "TEXT", 1, 0),
        ("trigger", "TEXT", 1, 0),
        ("summary_format_version", "INTEGER", 1, 0),
        ("summary", "TEXT", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_plan_digest", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("model_runtime_id", "TEXT", 1, 0),
        ("model", "TEXT", 1, 0),
        ("context_window", "INTEGER", 1, 0),
        ("threshold_tokens", "INTEGER", 1, 0),
        ("hard_input_tokens", "INTEGER", 1, 0),
        ("keep_recent_tokens", "INTEGER", 1, 0),
        ("tokens_before", "INTEGER", 1, 0),
        ("tokens_after", "INTEGER", 1, 0),
        ("summary_usage_json", "TEXT", 1, 0),
        ("invalidated_at", "TEXT", 0, 0),
        ("invalidated_reason", "TEXT", 0, 0),
    ),
    "named_indexes": {
        "idx_session_compactions_active": (
            ("session_key", "invalidated_at", "generation"),
            0,
        ),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (
        "CHECK (length(source_plan_digest) = 64 AND "
        "source_plan_digest NOT GLOB '*[^0-9a-f]*')",
    ),
}
_PREPARE_SCHEMA: _Schema = {
    "columns": (
        ("session_key", "TEXT", 1, 1),
        ("session_created_at", "TEXT", 1, 0),
        ("generation", "INTEGER", 1, 2),
        ("parent_generation", "INTEGER", 1, 0),
        ("source_ref", "TEXT", 1, 0),
        ("source_from_seq", "INTEGER", 1, 0),
        ("consolidated_through_seq", "INTEGER", 1, 0),
        ("source_message_ids_json", "TEXT", 1, 0),
        ("retained_tail_json", "TEXT", 1, 0),
        ("prepared_at", "TEXT", 1, 0),
    ),
    "named_indexes": {
        "idx_session_compaction_prepares_ref": (("session_key", "source_ref"), 0),
    },
    "auto_indexes": (
        ("pk", ("session_key", "generation")),
        ("u", ("session_key", "source_ref")),
    ),
    "sql_fragments": (),
}


def _digest(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _json(raw: str) -> Any:
    """旧 JSON 在迁移边界拒绝重复字段与非标准数值。"""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        if len(dict(items)) != len(items):
            raise ValueError("旧摘要 JSON 字段重复")
        return dict(items)

    def invalid(value: str) -> None:
        raise ValueError("旧摘要 JSON 数值无效: " + value)

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _source(
    connection: sqlite3.Connection,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """核对唯一旧 schema 与已提交游标；空 prepare 不冒充已完成摘要。"""
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
    if "session_compactions" not in tables:
        return [], {}
    for table, schema in (
        ("session_compactions", _FINAL_LEDGER_SCHEMA),
        ("session_compaction_prepares", _PREPARE_SCHEMA),
    ):
        validate_table_schema(connection, table=table, **schema)
    if connection.execute(
        "SELECT count(*) FROM session_compaction_prepares"
    ).fetchone()[0]:
        raise ValueError("旧摘要仍有未完成 prepare；不能把未提交内容发布成摘要")
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM session_compactions ORDER BY session_key,generation"
        )
    ]
    if not rows:
        return [], {}
    cursors = dict(connection.execute("SELECT key,last_consolidated FROM sessions"))
    for value in cursors.values():
        if type(value) is not int or value < 0:
            raise ValueError("旧摘要游标必须是非负整数")
    return rows, cursors


def _plan(
    log: MessageLog, rows: list[dict[str, Any]], cursors: dict[str, int]
) -> list[ImportedSummaryRecord]:
    """只转换当前游标的有效祖先链，逐代固定连续的累计来源。"""
    indexed = {(row["session_key"], row["generation"]): row for row in rows}
    if any(row["session_key"] not in cursors for row in rows):
        raise ValueError("旧摘要缺少原 Session")
    records: list[ImportedSummaryRecord] = []
    for session_id, cursor in sorted(cursors.items()):
        if cursor == 0:
            continue
        chain: list[dict[str, Any]] = []
        current = cursor
        while current:
            row = indexed.get((session_id, current))
            if row is None or row["invalidated_at"] is not None:
                raise ValueError("旧摘要游标或父链指向缺失、失效的 generation")
            parent = row["parent_generation"]
            if type(parent) is not int or not 0 <= parent < current:
                raise ValueError("旧摘要父链 generation 必须严格递减")
            chain.append(row)
            current = parent
        snapshot = log.reader(session_id).snapshot()
        cumulative: tuple[str, ...] = ()
        parent_ref: str | None = None
        for row in reversed(chain):
            if row["summary_format_version"] != 1:
                raise ValueError("旧摘要格式版本未知")
            ids: list[Any] = _json(row["source_message_ids_json"])
            if (
                not isinstance(ids, list)
                or not ids
                or any(not isinstance(i, str) or not i for i in ids)
            ):
                raise ValueError("旧摘要来源必须是非空 Message ID 数组")
            for name in ("retained_tail_json", "summary_usage_json"):
                _ = _json(row[name])
            cumulative += tuple(ids)
            covered = summary_range(snapshot, cumulative)
            own = snapshot[covered.stop - len(ids) : covered.stop]
            if (
                own[0].seq != row["source_from_seq"]
                or own[-1].seq != row["consolidated_through_seq"]
            ):
                raise ValueError("旧摘要来源 Message 与原序号边界不一致")
            reference = "legacy-compaction:" + _digest(
                [session_id, row["generation"], row["source_ref"]]
            )
            record = ImportedSummaryRecord.model_validate_json(
                json.dumps(
                    {
                        "version": 0,
                        "reference": reference,
                        "session_id": session_id,
                        "generation": row["generation"],
                        "parent": parent_ref,
                        "source_message_ids": cumulative,
                        "content": row["summary"],
                        "legacy": {
                            "schema": "sessions.session_compactions.v1",
                            "row": row,
                            "sha256": _digest(row),
                        },
                    },
                    ensure_ascii=False,
                )
            )
            records.append(record)
            parent_ref = reference
    return records


def migrate_legacy_summaries(workspace: Path) -> dict[str, Any] | None:
    """备份后原子追加 imported 摘要与 head；重试只核对原记录，不倒退后续 head。"""
    path = workspace / "sessions.db"
    if not path.exists():
        return None
    # 1. 旧事实必须在任何写入前可完整读取，备份不修改原库。
    with closing(sqlite3.connect(path)) as connection:
        connection.row_factory = sqlite3.Row
        rows, cursors = _source(connection)
        if not rows:
            return None
    backup = workspace / "backups/legacy-summaries-v1" / uuid4().hex
    _ = backup_sqlite_database(path, backup, migration="20260908_01_legacy_summaries")
    original = _digest([rows, cursors])
    with closing(MessageLog(path)) as log:
        state = log.owner(_OWNER)
        records = SummaryRecords(state)

        def commit(tx: OwnerTransaction) -> dict[str, Any]:
            connection = log._connection  # pyright: ignore[reportPrivateUsage]
            current_rows, current_cursors = _source(connection)
            if _digest([current_rows, current_cursors]) != original:
                raise ValueError("旧摘要来源在迁移前发生变化")
            receipt = tx.read(_RECEIPT)
            if receipt is not None:
                saved = cast(dict[str, Any], json_value(receipt.value))
                if receipt.version != 0 or saved["source_sha256"] != original:
                    raise ValueError("旧摘要迁移回执与原事实不一致")
                for reference, digest in saved["records"]:
                    record = records.read(reference)
                    if (
                        record is None
                        or _digest(record.model_dump(mode="json")) != digest
                    ):
                        raise ValueError("已导入的旧摘要缺失或改变")
                return saved
            planned = _plan(log, rows, cursors)
            # 2. 任何现有 head 都必须由其 owner 解释，迁移不得覆盖。
            for session_id in {record.session_id for record in planned}:
                if tx.read("head:" + session_id) is not None:
                    raise ValueError("旧摘要转换与已有 Summary head 冲突")
            heads: dict[str, str] = {}
            evidence: list[list[str]] = []
            for record in planned:
                value = cast(Mapping[str, object], record.model_dump(mode="json"))
                _ = tx.save("summary:" + record.reference, value, expected_version=None)
                heads[record.session_id] = record.reference
                evidence.append([record.reference, _digest(value)])
            for session_id, reference in heads.items():
                _ = tx.save(
                    "head:" + session_id,
                    {"reference": reference},
                    expected_version=None,
                )
            result: dict[str, Any] = {
                "source_sha256": original,
                "records": evidence,
                "heads": heads,
                "backup": str(backup),
            }
            _ = tx.save(_RECEIPT, result, expected_version=None)
            # 3. 一个提交点发布全部摘要，消息、旧行、游标和其他 owner 不写入。
            if [row[0] for row in connection.execute("PRAGMA integrity_check")] != [
                "ok"
            ]:
                raise ValueError("旧摘要迁移 SQLite 完整性检查失败")
            if list(connection.execute("PRAGMA foreign_key_check")):
                raise ValueError("旧摘要迁移外键检查失败")
            return result

        return state.transact(commit)
