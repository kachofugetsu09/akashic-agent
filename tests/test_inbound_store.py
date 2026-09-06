"""输入交接 owner 重开真实既有 schema，未知结构保持原样并报错。"""

import sqlite3
from contextlib import closing

import pytest

from session.admissions import SessionAdmissions
from session.inbound_store import InboundHandoffStore
from session.log import MessageLog, SessionAttributes


@pytest.mark.parametrize("owner,table,index", [
    (InboundHandoffStore, "inbound_handoffs", "idx_inbound_handoffs_session"),
    (SessionAdmissions, "session_admissions", "idx_session_admissions_session"),
])
@pytest.mark.parametrize("damage", ["missing_index", "extra_column"])
def test_inbound_owners_reject_unknown_schema_without_repair(tmp_path, owner, table, index, damage):
    path = tmp_path / "sessions.db"
    owner(path).close()
    with closing(sqlite3.connect(path)) as connection:
        with connection:
            if damage == "missing_index":
                connection.execute(f"DROP INDEX {index}")
            else:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN unknown TEXT")
        before = connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall()
        with pytest.raises(RuntimeError, match="schema"):
            owner(path)
        assert connection.execute("SELECT type,name,sql FROM sqlite_master ORDER BY name").fetchall() == before


def test_opening_admission_owner_does_not_clear_live_or_abandoned_leases(tmp_path):
    path = tmp_path / "sessions.db"
    log = MessageLog(path)
    log.ensure_session("channel:one", SessionAttributes())
    log.close()
    admissions = SessionAdmissions(path)
    admission_id = admissions.acquire("channel:one")
    admissions.close()
    reopened = SessionAdmissions(path)
    try:
        reopened.release_admission(admission_id)
        with pytest.raises(RuntimeError, match="不存在"):
            reopened.release_admission(admission_id)
        with pytest.raises(KeyError, match="不存在"):
            reopened.acquire("channel:missing")
    finally:
        reopened.close()
