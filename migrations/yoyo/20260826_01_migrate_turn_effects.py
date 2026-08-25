import json
import sqlite3
from uuid import uuid4

from yoyo import step

from agent.migrations.context import current_migration_context
from agent.migrations.session_db_backup import backup_sqlite_database

__depends__ = {"20260825_01_migrate_proactive_delivery_target"}
__transactional__ = False

_MIGRATION_NAME = "migrate-turn-effects"
_LEGACY_KEY = "skip_post_memory"
_EFFECTS_KEY = "effects"
_POST_COMMIT_KEY = "post_commit"
_SUPPRESS = "suppress"
_ALLOW = "allow"
_RETIRED_DEFAULT_RECEIPTS = "interaction_memory_reconciliations"


class _Rewrite:
    """Describe one JSON column replacement owned by the migration."""

    def __init__(self, *, table: str, key: str, column: str, payload: str) -> None:
        self.table = table
        self.key = key
        self.column = column
        self.payload = payload


def _decode_object(raw: object, *, field: str) -> dict[str, object]:
    """Decode one persisted JSON object at the SessionDB trust boundary."""

    if raw is None or raw == "":
        return {}
    if not isinstance(raw, str):
        raise ValueError(f"{field} 必须是 JSON 文本")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{field} 必须是 JSON object")
    return payload


def _legacy_value(metadata: dict[str, object], *, field: str) -> bool | None:
    """Read only the historical boolean contract from wide metadata."""

    if _LEGACY_KEY not in metadata:
        return None
    value = metadata[_LEGACY_KEY]
    if not isinstance(value, bool):
        if field.startswith("sessions.metadata"):
            return None
        raise ValueError(f"{field}.{_LEGACY_KEY} 必须是 boolean")
    return value


def _set_suppress(metadata: dict[str, object], *, field: str) -> bool:
    """Add the generic effect while rejecting a contradictory declaration."""

    raw_effects = metadata.get(_EFFECTS_KEY)
    if raw_effects is None:
        effects: dict[str, object] = {}
    elif isinstance(raw_effects, dict):
        effects = dict(raw_effects)
    else:
        raise ValueError(f"{field}.{_EFFECTS_KEY} 必须是 object")
    current = effects.get(_POST_COMMIT_KEY)
    if current not in (None, _SUPPRESS):
        if current == _ALLOW:
            raise RuntimeError(f"{field} 同时声明 legacy suppress 与 post_commit allow")
        raise ValueError(f"{field}.{_EFFECTS_KEY}.{_POST_COMMIT_KEY} 值无效")
    if current == _SUPPRESS:
        return False
    effects[_POST_COMMIT_KEY] = _SUPPRESS
    metadata[_EFFECTS_KEY] = effects
    return True


def _render_metadata(
    raw: object,
    *,
    field: str,
    force_suppress: bool,
) -> str | None:
    """Remove one legacy flag and optionally project its suppression effect."""

    metadata = _decode_object(raw, field=field)
    legacy = _legacy_value(metadata, field=field)
    if legacy is None and not force_suppress:
        return None
    changed = False
    if legacy is not None:
        del metadata[_LEGACY_KEY]
        changed = True
    if force_suppress or legacy is True:
        changed = _set_suppress(metadata, field=field) or changed
    if not changed:
        return None
    return json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
        is not None
    )


def _has_columns(
    connection: sqlite3.Connection,
    table: str,
    required: set[str],
) -> bool:
    """Return whether one optional historical table has the canonical columns."""

    if not _table_exists(connection, table):
        return False
    columns = {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}
    return required.issubset(columns)


def _plan_rewrites(connection: sqlite3.Connection) -> list[_Rewrite]:
    """Build the complete deterministic SessionDB rewrite before any mutation."""

    if not _has_columns(
        connection,
        "sessions",
        {"key", "metadata"},
    ) or not _has_columns(
        connection,
        "messages",
        {"id", "session_key", "seq", "extra"},
    ):
        return []

    # 1. Resolve historical session-wide exclusions and retire boolean markers.
    excluded_sessions: set[str] = set()
    rewrites: list[_Rewrite] = []
    session_rows = connection.execute(
        "SELECT key, metadata FROM sessions ORDER BY key"
    ).fetchall()
    for key, raw_metadata in session_rows:
        session_key = str(key)
        metadata = _decode_object(
            raw_metadata,
            field=f"sessions.metadata[{session_key}]",
        )
        legacy = _legacy_value(
            metadata,
            field=f"sessions.metadata[{session_key}]",
        )
        if session_key.startswith("scheduler:") or legacy is True:
            excluded_sessions.add(session_key)
        if legacy is not None:
            del metadata[_LEGACY_KEY]
            rewrites.append(
                _Rewrite(
                    table="sessions",
                    key=session_key,
                    column="metadata",
                    payload=json.dumps(
                        metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )

    # 2. Project every historical message onto the generic Turn effect.
    message_rows = connection.execute(
        "SELECT id, session_key, extra FROM messages ORDER BY session_key, seq, id"
    ).fetchall()
    for message_id, session_key, raw_extra in message_rows:
        rendered = _render_metadata(
            raw_extra,
            field=f"messages.extra[{message_id}]",
            force_suppress=str(session_key) in excluded_sessions,
        )
        if rendered is not None:
            rewrites.append(
                _Rewrite(
                    table="messages",
                    key=str(message_id),
                    column="extra",
                    payload=rendered,
                )
            )

    # 3. Keep durable programmatic Turn replay on the same canonical effect.
    if _has_columns(
        connection,
        "turns",
        {"id", "session_key", "input_json"},
    ):
        turn_rows = connection.execute(
            "SELECT id, session_key, input_json FROM turns ORDER BY id"
        ).fetchall()
        for turn_id, session_key, raw_input in turn_rows:
            turn_input = _decode_object(
                raw_input,
                field=f"turns.input_json[{turn_id}]",
            )
            raw_metadata = turn_input.get("metadata")
            if not isinstance(raw_metadata, dict):
                raise ValueError(f"turns.input_json[{turn_id}].metadata 必须是 object")
            rendered_metadata = _render_metadata(
                json.dumps(raw_metadata, ensure_ascii=False),
                field=f"turns.input_json[{turn_id}].metadata",
                force_suppress=str(session_key) in excluded_sessions,
            )
            if rendered_metadata is None:
                continue
            turn_input["metadata"] = json.loads(rendered_metadata)
            rewrites.append(
                _Rewrite(
                    table="turns",
                    key=str(turn_id),
                    column="input_json",
                    payload=json.dumps(
                        turn_input,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )
            )
    return rewrites


def _apply_rewrites(
    connection: sqlite3.Connection,
    rewrites: list[_Rewrite],
) -> None:
    """Apply a prevalidated plan and retire the Default-only receipt table."""

    for rewrite in rewrites:
        identity = "key" if rewrite.table == "sessions" else "id"
        cursor = connection.execute(
            f"UPDATE {rewrite.table} SET {rewrite.column} = ? WHERE {identity} = ?",
            (rewrite.payload, rewrite.key),
        )
        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Turn effect migration target changed: {rewrite.table}:{rewrite.key}"
            )

    if _plan_rewrites(connection):
        raise RuntimeError("Turn effect migration left canonical rewrites pending")
    connection.execute(f"DROP TABLE IF EXISTS {_RETIRED_DEFAULT_RECEIPTS}")
    if _table_exists(connection, _RETIRED_DEFAULT_RECEIPTS):
        raise RuntimeError("Turn effect migration left Default Memory receipts behind")


def migrate_turn_effects(_connection: object) -> None:
    """Replace historical memory exclusions with the generic Turn effect."""

    _ = _connection
    current = current_migration_context()
    sessions_db = current.workspace / "sessions.db"
    if not sessions_db.exists():
        return

    # 1. Preflight every target before allocating a recovery artifact.
    preflight = sqlite3.connect(sessions_db)
    try:
        rewrites = _plan_rewrites(preflight)
        retire_receipts = _table_exists(preflight, _RETIRED_DEFAULT_RECEIPTS)
    finally:
        preflight.close()
    if not rewrites and not retire_receipts:
        return

    # 2. Persist an online, integrity-checked copy before the only data write.
    backup_sqlite_database(
        sessions_db,
        current.workspace / "backups" / _MIGRATION_NAME / uuid4().hex,
        migration=_MIGRATION_NAME,
    )

    # 3. Recompute under the write lock and commit the full rewrite atomically.
    connection = sqlite3.connect(sessions_db)
    try:
        connection.execute("BEGIN IMMEDIATE")
        try:
            locked_rewrites = _plan_rewrites(connection)
            _apply_rewrites(connection, locked_rewrites)
            if connection.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                raise RuntimeError("Turn effect migration integrity_check 失败")
        except BaseException:
            connection.rollback()
            raise
        connection.commit()
    finally:
        connection.close()


steps = [step(migrate_turn_effects)]
