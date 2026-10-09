"""真实消息与 owner 提交在 SIGKILL 后完整回读；只写临时数据库。"""
import multiprocessing
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory

from session.log import MessageLog
from session.message import ContentPart, ContentReferences, Input


def records(path: Path) -> tuple[list[tuple], list[tuple]]:
    """读取完整消息和 owner 行，保留原身份、正文、版本与顺序。"""
    with sqlite3.connect(path) as connection:
        return (
            connection.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall(),
            connection.execute("SELECT * FROM owner_records ORDER BY owner,key").fetchall(),
        )


def writer(path: Path, pipe) -> None:
    """在真实 owner 事务中共同提交消息与 intent，保持连接打开等待强杀。"""
    log = MessageLog(path)
    append = log.writer(
        "session", author="user", source="source", body_types=(Input,),
        content={"text": lambda _: ContentReferences()},
    )
    owner = log.owner("wal-recovery")
    try:
        # 1. 每次事务同时发布实际消息和相应 owner 状态。
        for index in range(30):
            def commit(transaction):
                transaction.append(
                    append, f"message-{index}",
                    Input((ContentPart("text", f"retained message {index}"),)),
                )
                transaction.save(f"call-{index}", {"phase": "started"}, expected_version=None)

            owner.transact(commit)
        mode = log._connection.execute("PRAGMA synchronous").fetchone()[0]
        pipe.send((mode, records(path)))
        pipe.recv()
    finally:
        log.close()


def check(path: Path) -> None:
    """提交回执返回后强杀写进程，再从实际数据库重开核对全部记录。"""
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=writer, args=(path, child))
    process.start()
    child.close()
    try:
        # 2. 只在子进程确认提交后强杀，不用时间延迟猜测提交点。
        if not parent.poll(30):
            raise TimeoutError(f"消息写进程未交付提交回执，exitcode={process.exitcode}")
        mode, before = parent.recv()
        assert mode == 1 and len(before[0]) == len(before[1]) == 30
        process.kill()
        process.join(10)
        assert not process.is_alive()
        assert records(path) == before
        # 3. 重开生产 MessageLog 后，恢复读取不得减少或改写旧记录。
        reopened = MessageLog(path)
        try:
            assert len(reopened.reader("session").snapshot()) == 30
            assert len(reopened.owner("wal-recovery").list()) == 30
            assert records(path) == before
        finally:
            reopened.close()
        print("PASS: NORMAL, 30 atomic messages + owner intents, SIGKILL recovery, exact rows retained")
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-message-wal-") as directory:
        check(Path(directory) / "sessions.db")
