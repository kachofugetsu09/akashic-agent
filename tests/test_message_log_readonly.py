"""关闭后的原消息仍由 MessageLog owner 解码；查询不创建或改写库。"""
import sqlite3
from contextlib import closing

import pytest

from session.log import MessageLog, read_persisted_messages
from session.message import ContentPart, Input, Output
from tests.test_message_log import text_schema


def test_read_closed_messages_preserves_original_rows_without_initialization(tmp_path, monkeypatch):
    path = tmp_path / "evidence #问号?.db"
    with closing(MessageLog(path)) as log:
        inputs = log.writer("s", author="user", source="program", body_types=(Input,),
                            content={"text": text_schema})
        inputs.append("input", Input((ContentPart("text", "原始问题"),)))
        outputs = log.writer("s", author="assistant", source="program", body_types=(Output,),
                             content={"text": text_schema})
        outputs.append("answer", Output((ContentPart("text", "original answer"),), "complete"))
        other = log.writer("other", author="user", source="program", body_types=(Input,), content={})
        other.append("other-input", Input(()))
        expected = log.reader("s").snapshot()
    before = path.read_bytes()
    files = set(tmp_path.iterdir())

    def reject_initialization(self, path):
        pytest.fail("只读查询不得初始化 MessageLog")

    monkeypatch.setattr(MessageLog, "__init__", reject_initialization)
    assert read_persisted_messages(path, "s") == expected
    assert read_persisted_messages(path, "absent") == ()
    assert path.read_bytes() == before
    assert set(tmp_path.iterdir()) == files


@pytest.mark.parametrize("contents", [None, b"", b"not a SQLite database"])
def test_missing_or_invalid_database_fails_without_creating_or_repairing_it(tmp_path, contents):
    path = tmp_path / "sessions.db"
    if contents is not None:
        path.write_bytes(contents)
    with pytest.raises(sqlite3.DatabaseError):
        read_persisted_messages(path, "s")
    if contents is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == contents


@pytest.mark.parametrize("field", ["body", "metadata"])
def test_corrupt_message_uses_original_decoder_and_preserves_evidence(tmp_path, field):
    path = tmp_path / "sessions.db"
    with closing(MessageLog(path)) as log:
        inputs = log.writer("s", author="user", source="program", body_types=(Input,), content={})
        inputs.append("input", Input(()))
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(f"UPDATE messages SET {field} = ? WHERE id = ?", ("not JSON", "input"))
    before = path.read_bytes()
    with pytest.raises(ValueError):
        read_persisted_messages(path, "s")
    assert path.read_bytes() == before
