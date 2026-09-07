import sqlite3

import pytest

from session.log import MessageConflict, MessageLog
from session.message import Input, Output, ToolCall


@pytest.fixture
def log(tmp_path):
    log = MessageLog(tmp_path / "sessions.db")
    try:
        yield log
    finally:
        log.close()


def test_owner_cursor_and_message_commit_or_rollback_together(log):
    owner = log.owner("consumer")
    writer = log.writer(
        "s", author="service", source="s", body_types=(Input,), content={}
    )

    def fail(tx):
        tx.append(writer, "input", Input(()))
        tx.save("cursor", {"message_id": "input"}, expected_version=None)
        raise ValueError("failed batch")

    with pytest.raises(ValueError):
        owner.transact(fail)
    assert owner.read("cursor") is None
    assert log.reader("s").read() == ()

    def commit(tx):
        receipt = tx.append(writer, "input", Input(()))
        tx.save("cursor", {"message_id": receipt.message_id}, expected_version=None)
        return receipt

    receipt = owner.transact(commit)
    assert owner.read("cursor").value == {"message_id": receipt.message_id}
    assert log.reader("s").read() == (receipt,)
    assert log.owner("unrelated").list() == ()


def test_caught_partial_insert_failure_still_rolls_back(log):
    writer = log.writer(
        "s",
        author="model",
        source="s",
        body_types=(Output,),
        content={},
        check_call=lambda _: None,
    )

    def operation(tx):
        try:
            tx.append(
                writer, "bad", Output((ToolCall("missing-binding", {}),), "continue")
            )
        except sqlite3.IntegrityError:
            pass

    with pytest.raises(RuntimeError, match="必须回滚"):
        log.owner("tool").transact(operation)
    assert log.reader("s").read() == ()


def test_owner_cas_and_expired_transaction_reject_changes(log):
    owner = log.owner("owner")
    saved = owner.transact(
        lambda tx: tx.save("key", {"nested": [1]}, expected_version=None)
    )
    with pytest.raises(TypeError):
        saved.value["nested"][0] = 2
    with pytest.raises(MessageConflict):
        owner.transact(lambda tx: tx.save("key", {"nested": []}, expected_version=None))
    assert owner.read("key") == saved
    transaction = owner.transact(lambda tx: tx)
    with pytest.raises(RuntimeError, match="已结束"):
        transaction.save("key", {}, expected_version=saved.version)


def test_async_callback_cannot_hold_transaction_across_io(log):
    called = []

    async def operation(tx):
        called.append(True)
        tx.save("key", {}, expected_version=None)

    with pytest.raises(TypeError, match="同步"):
        log.owner("owner").transact(operation)
    assert called == []
    assert log.owner("owner").read("key") is None
