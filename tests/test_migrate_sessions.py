from pathlib import Path

from scripts.migrate_sessions_to_sqlite import migrate
from session.store import SessionStore


def _write_jsonl(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_migrate_basic_and_keep_original(tmp_path: Path):
    jsonl = tmp_path / "telegram_7674283004.jsonl"
    _write_jsonl(
        jsonl,
        [
            '{"_type":"metadata","key":"telegram:7674283004","created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-02T00:00:00+00:00","last_consolidated":2,"metadata":{}}',
            '{"role":"user","content":"hello","timestamp":"2026-01-01T10:00:00+00:00"}',
            '{"role":"assistant","content":"hi","timestamp":"2026-01-01T10:00:01+00:00"}',
        ],
    )

    db_path = tmp_path / "sessions.db"
    migrate(sessions_dir=tmp_path, db_path=db_path, workers=2)

    store = SessionStore(db_path)
    msgs = store.fetch_session_messages("telegram:7674283004")
    assert len(msgs) == 2
    assert msgs[0]["id"] == "telegram:7674283004:0"
    assert msgs[1]["id"] == "telegram:7674283004:1"
    assert jsonl.exists()
    assert (tmp_path / "telegram_7674283004.jsonl.migrated").exists()


def test_migrate_skips_eval_and_idempotent(tmp_path: Path):
    _write_jsonl(
        tmp_path / "eval_abc123.jsonl",
        [
            '{"_type":"metadata","key":"eval:abc123","created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","last_consolidated":0,"metadata":{}}',
            '{"role":"user","content":"eval msg"}',
        ],
    )
    _write_jsonl(
        tmp_path / "tg_1.jsonl",
        [
            '{"_type":"metadata","key":"tg:1","created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","last_consolidated":0,"metadata":{}}',
            '{"role":"user","content":"only once"}',
        ],
    )

    db_path = tmp_path / "sessions.db"
    migrate(sessions_dir=tmp_path, db_path=db_path)
    migrate(sessions_dir=tmp_path, db_path=db_path)

    store = SessionStore(db_path)
    assert store.fetch_session_messages("eval:abc123") == []
    assert len(store.fetch_session_messages("tg:1")) == 1


def test_migrate_multiple_sessions_concurrently(tmp_path: Path):
    _write_jsonl(
        tmp_path / "telegram_1.jsonl",
        [
            '{"_type":"metadata","key":"telegram:1","created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","last_consolidated":0,"metadata":{}}',
            '{"role":"user","content":"A1"}',
        ],
    )
    _write_jsonl(
        tmp_path / "telegram_2.jsonl",
        [
            '{"_type":"metadata","key":"telegram:2","created_at":"2026-01-01T00:00:00+00:00","updated_at":"2026-01-01T00:00:00+00:00","last_consolidated":0,"metadata":{}}',
            '{"role":"user","content":"B1"}',
        ],
    )

    db_path = tmp_path / "sessions.db"
    results = migrate(sessions_dir=tmp_path, db_path=db_path, workers=2)

    migrated_keys = {r.session_key for r in results if r.status == "migrated"}
    assert migrated_keys == {"telegram:1", "telegram:2"}
    store = SessionStore(db_path)
    assert len(store.fetch_session_messages("telegram:1")) == 1
    assert len(store.fetch_session_messages("telegram:2")) == 1
