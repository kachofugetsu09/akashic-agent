"""旧有效摘要恢复后可供 Context 使用，原消息与失效历史不变。"""

from contextlib import closing
import json
import sqlite3

import pytest

from agent.migrations.legacy_summaries import migrate_legacy_summaries
from agent.migrations.session_attributes import migrate as migrate_attributes
from plugins.compaction.records import SummaryLookup, SummaryRecords
from plugins.content.plugin import check_text
from session.log import MessageLog, OwnerTransaction
from session.message import ContentPart, Input
from session.store import SessionStore


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    old = tmp_path / "legacy.db"
    with closing(SessionStore(old)):
        pass
    path = root / "sessions.db"
    with (
        closing(sqlite3.connect(old)) as source,
        closing(sqlite3.connect(path)) as target,
    ):
        for table in ("sessions", "session_compactions", "session_compaction_prepares"):
            target.execute(
                source.execute(
                    "SELECT sql FROM sqlite_master WHERE name=?", (table,)
                ).fetchone()[0]
            )
        for (sql,) in source.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL AND tbl_name IN ('session_compactions','session_compaction_prepares')"
        ):
            target.execute(sql)
        target.commit()
    with closing(MessageLog(path)):
        pass
    migrate_attributes(path, root / "backups/test-attributes")
    with closing(MessageLog(path)) as log:
        writer = log.writer(
            "s",
            author="user",
            source="conversation",
            body_types=(Input,),
            content={"text": check_text},
        )
        for number in range(4):
            writer.append(
                "u" + str(number), Input((ContentPart("text", "原文" + str(number)),))
            )
    return root


def summary(root, generation, parent, ids, *, invalidated=None):
    with closing(sqlite3.connect(root / "sessions.db")) as db:
        seqs = [
            db.execute("SELECT seq FROM messages WHERE id=?", (i,)).fetchone()[0]
            for i in ids
        ]
        row = {
            "session_key": "s",
            "generation": generation,
            "parent_generation": parent,
            "created_at": "2026-09-01T00:00:00+00:00",
            "trigger": "soft_limit",
            "summary_format_version": 1,
            "summary": "原摘要" + str(generation),
            "source_ref": "source:" + str(generation),
            "source_plan_digest": "a" * 64,
            "source_from_seq": seqs[0],
            "consolidated_through_seq": seqs[-1],
            "source_message_ids_json": json.dumps(ids),
            "retained_tail_json": "[]",
            "model_runtime_id": "old-runtime",
            "model": "old-model",
            "context_window": 100000,
            "threshold_tokens": 74000,
            "hard_input_tokens": 90000,
            "keep_recent_tokens": 20000,
            "tokens_before": 75000,
            "tokens_after": 30000,
            "summary_usage_json": '{ "input_tokens": 1 }',
            "invalidated_at": invalidated,
            "invalidated_reason": None,
        }
        db.execute(
            "INSERT INTO session_compactions ("
            + ",".join(row)
            + ") VALUES ("
            + ",".join("?" for _ in row)
            + ")",
            tuple(row.values()),
        )
        db.execute(
            "UPDATE sessions SET last_consolidated=? WHERE key='s'", (generation,)
        )
        db.commit()
    return row


def dump(root):
    with closing(sqlite3.connect(root / "sessions.db")) as db:
        return "\n".join(db.iterdump())


def test_active_ancestor_chain_survives_restart_and_reentry_without_fake_calls(
    workspace,
):
    stale = summary(workspace, 1, 0, ["u0"], invalidated="2026-09-02")
    first = summary(workspace, 3, 0, ["u0", "u1"])
    last = summary(workspace, 4, 3, ["u2"])
    with closing(MessageLog(workspace / "sessions.db")) as log:
        original = log.reader("s").snapshot()
    result = migrate_legacy_summaries(workspace)
    assert len(result["records"]) == 2
    after = dump(workspace)
    assert migrate_legacy_summaries(workspace) == result
    assert dump(workspace) == after
    with closing(MessageLog(workspace / "sessions.db")) as log:
        records = SummaryRecords(log.owner("plugin:compaction"))
        head = records.head("s")
        assert head.content == last["summary"] and head.generation == 4
        assert head.source_message_ids == ("u0", "u1", "u2")
        assert records.read(head.parent).generation == 3
        assert head.legacy.row.model_dump() == last
        assert "model_call_ids" not in head.model_dump()
        assert "max_output_tokens" not in head.model_dump()
        assert (
            SummaryLookup(records.read, records.head).resolve(
                {"record_ref": head.reference, "session_id": "s"}, session_id="s"
            )
            == head
        )
        assert log.reader("s").snapshot() == original
    with closing(sqlite3.connect(workspace / "sessions.db")) as db:
        db.row_factory = sqlite3.Row
        assert [
            dict(row)
            for row in db.execute(
                "SELECT * FROM session_compactions ORDER BY generation"
            )
        ] == [stale, first, last]


@pytest.mark.parametrize(
    "change", ["missing", "order", "bad_json", "schema", "prepare", "head", "parent"]
)
def test_invalid_source_or_conflicting_head_has_no_partial_publication(
    workspace, change
):
    summary(workspace, 1, 0, ["u0", "u1"])
    with closing(sqlite3.connect(workspace / "sessions.db")) as db:
        if change == "missing":
            db.execute(
                'UPDATE session_compactions SET source_message_ids_json=\'["u0","missing"]\''
            )
        elif change == "order":
            db.execute(
                'UPDATE session_compactions SET source_message_ids_json=\'["u1","u0"]\''
            )
        elif change == "bad_json":
            db.execute("UPDATE session_compactions SET summary_usage_json='{'")
        elif change == "schema":
            db.execute("ALTER TABLE session_compactions ADD COLUMN unknown TEXT")
        elif change == "prepare":
            db.execute(
                "INSERT INTO session_compaction_prepares VALUES ('s','stamp',2,1,'pending',2,2,'[\"u2\"]','[]','stamp')"
            )
        elif change == "head":
            db.execute(
                "INSERT INTO owner_records VALUES ('plugin:compaction','head:s',0,'{\"reference\":\"existing\"}')"
            )
        else:
            db.execute("UPDATE session_compactions SET parent_generation=1")
        db.commit()
    before = dump(workspace)
    with pytest.raises((ValueError, RuntimeError)):
        migrate_legacy_summaries(workspace)
    assert dump(workspace) == before


def test_write_failure_rolls_back_every_import_then_retry_succeeds(
    workspace, monkeypatch
):
    summary(workspace, 1, 0, ["u0"])
    before = dump(workspace)
    original = OwnerTransaction.save

    def fail(self, key, value, *, expected_version):
        if key == "head:s":
            raise OSError("injected commit failure")
        return original(self, key, value, expected_version=expected_version)

    with monkeypatch.context() as patch:
        patch.setattr(OwnerTransaction, "save", fail)
        with pytest.raises(OSError, match="injected"):
            migrate_legacy_summaries(workspace)
    assert dump(workspace) == before
    assert len(migrate_legacy_summaries(workspace)["records"]) == 1


def test_empty_legacy_ledger_does_not_create_a_summary(workspace):
    before = dump(workspace)
    assert migrate_legacy_summaries(workspace) is None
    assert dump(workspace) == before
