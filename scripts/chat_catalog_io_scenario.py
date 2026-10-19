"""Check chat catalog HTTP reads under controlled SQLite reads in an installed runtime."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import closing, contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
# AKASHIC_PERF_ROOT 指向旧提交 checkout 时，用旧代码跑 --baseline，证明屏障能抓住旧行为。
CODE_ROOT = Path(os.environ.get("AKASHIC_PERF_ROOT", ROOT)).resolve()
sys.path.insert(0, str(CODE_ROOT))


class ReadBarrier:
    """Pause actual SQLite statements; queries still read their real saved rows."""

    def __init__(self, *, cancel: bool = False):
        self.loop = asyncio.get_running_loop()
        self.cancel = cancel
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self.reads = []

    @contextmanager
    def install(self):
        """Attach a trace only to ledger read-only connections, then restore it."""
        original = sqlite3.connect

        def connect(database, *args, **kwargs):
            connection = original(database, *args, **kwargs)
            if "sessions.db?mode=ro" in str(database):
                used = False

                def trace(statement):
                    nonlocal used
                    if used or not statement.lstrip().upper().startswith("SELECT"):
                        return
                    used = True
                    heartbeat = threading.Event()
                    started = time.perf_counter()
                    observation = {"on_loop": threading.get_ident() == self.loop_thread}
                    self.reads.append(observation)

                    def tick():
                        observation["loop_delay_ms"] = (time.perf_counter() - started) * 1000
                        heartbeat.set()
                        self.entered.set()

                    self.loop.call_soon_threadsafe(tick)
                    # The bound ends a blocked baseline; success requires the loop receipt.
                    observation["heartbeat_before_read"] = heartbeat.wait(1)
                    if self.cancel:
                        observation["released"] = self.release.wait(5)

                connection.set_trace_callback(trace)
            return connection

        self.loop_thread = threading.get_ident()
        sqlite3.connect = connect
        try:
            yield
        finally:
            self.release.set()
            sqlite3.connect = original


def saved_rows(workspace: Path) -> dict[str, list]:
    """Keep complete session and message rows; chat catalog reads never rewrite them."""
    with closing(sqlite3.connect(workspace / "sessions.db")) as db:
        return {table: db.execute(f"SELECT * FROM {table} ORDER BY 1").fetchall()
                for table in ("sessions", "messages")}


def seed_catalog(workspace: Path) -> None:
    """在真实 runtime 启动前写入确定性目录夹具，并固定全部时间戳。"""
    from plugins.ledger.contract import ContentPart, ContentReferences, Input, Output
    from plugins.ledger.log import MessageLog, SessionAttributes

    log = MessageLog(workspace / "sessions.db")
    try:
        content = {"text": lambda _: ContentReferences()}
        sessions = ("akashic:io-a", "akashic:io-b", "akashic:io-c")
        for index, key in enumerate(sessions):
            log.ensure_session(key, SessionAttributes())
            writer = log.writer(key, author="user", source="conversation",
                                body_types=(Input, Output), content=content)
            writer.append(f"{key}-in-1", Input((ContentPart("text", f"问题 {index}"),)))
            writer.append(f"{key}-out-1", Output((ContentPart("text", f"回答 {index}-1"),), "complete"))
            if index < 2:
                writer.append(f"{key}-out-2", Output((ContentPart("text", f"回答 {index}-2"),), "complete"))
    finally:
        log.close()
    # 固定 created_at/updated_at/ts，让两次运行（新旧代码）的响应体逐字节一致。
    fixed_day = "2026-10-20T00:00:00+00:00"
    with closing(sqlite3.connect(workspace / "sessions.db")) as db, db:
        for index, key in enumerate(sessions):
            db.execute("UPDATE sessions SET created_at=?, updated_at=? WHERE key=?",
                       (fixed_day, f"2026-10-20T0{3 - index}:00:00+00:00", key))
            for seq in range(3):
                db.execute("UPDATE messages SET ts=? WHERE session_key=? AND seq=?",
                           (f"2026-10-20T0{3 - index}:00:0{seq + 1}+00:00", key, seq))


def reset_read_pool(log) -> None:
    """关闭空闲只读连接，让每个测量窗口都在新连接上装自己的 trace。"""
    with log._read_admission:  # pyright: ignore[reportPrivateUsage]
        idle, log._idle_reads = log._idle_reads, []  # pyright: ignore[reportPrivateUsage]
    for read in idle:
        read.connection.close()


async def run(distribution: Path, directory: Path, baseline: bool) -> dict:
    """Install plugins, seed a deterministic catalog, and query through the real chat HTTP API."""
    import httpx
    from agent.config import Config
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from plugins.ledger.contract import MESSAGE_CATALOG
    from scripts.install_plugin_distribution import ensure_bundle

    # 1. Build fresh runtime state and a deterministic saved catalog.
    workspace, home, config = directory / "workspace", directory / "home", directory / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION=str(distribution),
                      AKASHIC_PLUGIN_BUNDLE="base", AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local",
                      PYTHONPATH=os.pathsep.join((str(CODE_ROOT), str(CODE_ROOT / "sdk/python/src"))))
    os.environ.pop("AKASHIC_WEB_PORT", None)
    config.write_text("[runtime]\n")
    init_workspace(config_path=config, workspace=workspace)
    ensure_bundle(distribution, distribution / "bundles/base.toml", workspace=workspace,
                  plugins_home=home, config_path=config, receipt_path=workspace / "runtime/distribution-install.json")
    seed_catalog(workspace)
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    try:
        await app.start()
        catalog = app.core.plugin_manager.live_root.context.require(MESSAGE_CATALOG)
        log = catalog._log  # pyright: ignore[reportPrivateUsage] 场景只排空空闲读连接，不读取事实。
        socket = workspace / "runtime" / "web-chat.sock"
        for _ in range(200):
            if socket.exists():
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("chat API listener 未就绪")

        transport = httpx.AsyncHTTPTransport(uds=str(socket))
        async with httpx.AsyncClient(transport=transport, base_url="http://chat", timeout=30,
                                     trust_env=False) as client:
            async def expect(method: str, url: str, *, params: dict | None = None) -> bytes:
                response = await client.request(method, url, params=params)
                assert response.status_code == 200, (url, response.status_code, response.text)
                return response.content

            # Warm the exact endpoints once so the measurement windows observe steady reads.
            await expect("GET", "/api/chat/sessions")
            await expect("GET", "/api/chat/sessions/akashic:io-a/messages")
            before = saved_rows(workspace)

            # 2. Every catalog page and message tail read must leave the loop available.
            probes = (
                ("GET", "/api/chat/sessions", None),
                ("GET", "/api/chat/sessions", {"page_size": 2}),
                ("GET", "/api/chat/sessions/akashic:io-a/messages", None),
                ("GET", "/api/chat/sessions/akashic:io-a/messages", {"page_size": 2}),
            )
            observations: dict[str, list] = {}
            bodies: dict[str, str] = {}
            for method, url, params in probes:
                reset_read_pool(log)
                barrier = ReadBarrier()
                with barrier.install():
                    body = await expect(method, url, params=params)
                assert barrier.reads, f"{url} 没有捕获到真实读取"
                if baseline:
                    assert any(read["on_loop"] and not read["heartbeat_before_read"] for read in barrier.reads)
                else:
                    assert all(not read["on_loop"] and read["heartbeat_before_read"] for read in barrier.reads)
                key = url + ("?" + json.dumps(params, sort_keys=True) if params else "")
                observations[key] = barrier.reads
                bodies[key] = hashlib.sha256(body).hexdigest()

            missing = await client.get("/api/chat/sessions/akashic:none/messages")
            assert missing.status_code == 404, "未知会话仍返回 404"
            assert saved_rows(workspace) == before

            # 3. Caller cancellation may return, but the storage read must drain first.
            cancel_drained = False
            if not baseline:
                reset_read_pool(log)
                barrier = ReadBarrier(cancel=True)
                with barrier.install():
                    reading = asyncio.create_task(catalog.sessions_async(
                        prefix="akashic:", visibility="listed", after=None, limit=50))
                    try:
                        await asyncio.wait_for(barrier.entered.wait(), 3)
                        reading.cancel()
                        tick = asyncio.Event()
                        asyncio.get_running_loop().call_soon(tick.set)
                        await tick.wait()
                        # run_file_io 排空：实际读取结束前，被取消的协程不结束、连接不归还。
                        assert not reading.done(), "取消没有等待实际读取排空"
                        assert not any(read.get("released") for read in barrier.reads)
                        barrier.release.set()
                        try:
                            await asyncio.wait_for(reading, 3)
                        except asyncio.CancelledError:
                            pass
                        else:
                            raise AssertionError("读取未经取消语义结束")
                        assert all(read.get("released") for read in barrier.reads)
                        cancel_drained = True
                    finally:
                        barrier.release.set()
                        await asyncio.gather(reading, return_exceptions=True)
                assert saved_rows(workspace) == before
        return {"baseline": baseline, "reads": observations, "response_sha256": bodies,
                "saved_rows_unchanged": True, "cancel_drained": cancel_drained,
                "missing_session_rejected": True}
    finally:
        await app.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distribution", type=Path)
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()
    directory = Path(tempfile.mkdtemp(prefix="akashic-chat-catalog-io-"))
    result = asyncio.run(run(args.distribution.resolve(), directory, args.baseline))
    print(json.dumps({**result, "evidence": str(directory)}))
