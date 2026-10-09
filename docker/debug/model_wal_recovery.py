"""真实模型账本跨越 checkpoint 目标、固定长读快照，再强停进程核对恢复。"""

from __future__ import annotations

import multiprocessing
import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from agent.plugin_composition.models import (
    BoundModelDescriptor, CapabilitySources, LLMResponse, ModelCapabilities, ModelRequest,
)
from plugins.models.store import ModelsStore


def records(path: Path) -> list[tuple]:
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as connection:
        return connection.execute("SELECT * FROM model_calls ORDER BY id").fetchall()


def writer(path: Path, pipe) -> None:
    """只使用账本真实准入和结算入口；本地受控响应不发起模型请求。"""
    store = ModelsStore(path, path.parent / "backups")
    store.initialize()
    descriptor = BoundModelDescriptor(
        "binding", "snapshot", 0, "model", "connection", "driver", "1",
        "scenario", "scenario-model", "agent", None,
        ModelCapabilities(), CapabilitySources(), "digest",
    )

    def complete(index: int) -> None:
        call = store.start_call(descriptor, ModelRequest(messages=[]))
        store.finish_call(
            call, usage=None, failure=None,
            response=LLMResponse(f"response {index}: " + "retained text " * 1600),
        )

    try:
        # 1. 父进程在首个已提交响应处固定只读快照。
        complete(0)
        with store._connect() as connection:
            pages = connection.execute("PRAGMA wal_autocheckpoint").fetchone()[0]
            page_size = connection.execute("PRAGMA page_size").fetchone()[0]
            synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        pipe.send((pages, page_size, synchronous))
        assert pipe.recv() == "pinned"
        for index in range(1, 81):
            complete(index)
        pending = store.start_call(descriptor, ModelRequest(messages=[]))
        pipe.send((pending, records(path)))
        # 2. 长读释放后继续正常提交，让 SQLite 自己 checkpoint 和复用。
        assert pipe.recv() == "released"
        for index in range(81, 101):
            complete(index)
        pipe.send(records(path))
        pipe.recv()
    finally:
        store.close()


def receive(pipe, process):
    if not pipe.poll(30):
        raise TimeoutError(f"模型账本进程未交付收据，exitcode={process.exitcode}")
    return pipe.recv()


def check(root: Path) -> None:
    """长读不阻塞提交，SIGKILL 后完整账目、响应和未结算 intent 保持不变。"""
    path = root / "models.sqlite3"
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=writer, args=(path, child))
    process.start()
    child.close()
    try:
        pages, page_size, synchronous = receive(parent, process)
        assert synchronous == 1, "场景必须使用 NORMAL 进程崩溃保证"
        reader = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        try:
            reader.execute("BEGIN")
            first = reader.execute("SELECT * FROM model_calls ORDER BY id").fetchall()
            assert len(first) == 1
            parent.send("pinned")
            pending, before_release = receive(parent, process)
            assert reader.execute("SELECT * FROM model_calls ORDER BY id").fetchall() == first
            assert len(before_release) == 82 and records(path) == before_release
            assert Path(str(path) + "-wal").stat().st_size > pages * page_size
        finally:
            reader.close()
        parent.send("released")
        before_crash = receive(parent, process)
        assert len(before_crash) == 102
        # 3. 不运行 close/checkpoint 清理；由下一进程从现存 WAL 恢复。
        process.kill()
        process.join(10)
        assert not process.is_alive()
        reopened = ModelsStore(path, root / "backups")
        try:
            reopened.initialize()
            reopened.integrity_check()
            assert records(path) == before_crash
            assert reopened.read_call(pending)["state"] == "started"
        finally:
            reopened.close()
        print(f"PASS: {pages}-page passive target, pinned reader, 101 responses + pending intent, SIGKILL recovery")
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()


if __name__ == "__main__":
    with TemporaryDirectory(prefix="akashic-model-wal-") as directory:
        check(Path(directory))
