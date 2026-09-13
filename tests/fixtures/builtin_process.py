"""在独立进程启动真实内置 App；stdout 只额外发布测试连接地址。"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import signal
import sys

from agent.config_models import Config
from bootstrap.app import AppRuntime
from bootstrap.runtime_readiness import RuntimeReadiness


async def producers(app: AppRuntime) -> None:
    """stdin 只模拟插件生产者提交；判断、发送和结算全部由实际内置服务完成。"""
    from agent.plugins.snapshot import lease_runtime_snapshot
    from plugins.drift.plugin import DRIFT_PROPOSALS
    from plugins.eventmail.plugin import EVENTMAIL_ALERT_SOURCE, EVENTMAIL_CONTENT_SOURCE

    reader = asyncio.StreamReader()
    transport, _ = await asyncio.get_running_loop().connect_read_pipe(
        lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
    try:
        while line := await reader.readline():
            command = json.loads(line)
            assert app.core is not None and app.core.plugin_manager is not None
            async with lease_runtime_snapshot(app.core.plugin_manager.snapshot_store) as snapshot:
                ctx = snapshot.composition_root.context
                now = datetime.now(timezone.utc)
                if command["type"] == "drift":
                    result = ctx.require(DRIFT_PROPOSALS).propose("fixture-duty", "1", {"summary": "真实职责"}, now)
                elif command["type"] == "alert":
                    result = ctx.require(EVENTMAIL_ALERT_SOURCE).bind("fixture").report(
                        event_id="alarm", payload={"summary": "真实告警"}, observed_at=now)
                elif command["type"] == "content":
                    result = ctx.require(EVENTMAIL_CONTENT_SOURCE).bind("fixture").submit("batch", [
                        {"item_id": "article", "revision": "1", "not_before": now,
                         "payload": {"title": "真实文章", "url": "https://example.com/original",
                                     "published_at": now.isoformat(), "preprocess_score": 0.9}}])
                else:
                    raise ValueError("未知 fixture producer")
            print(json.dumps({"fixture_receipt": result}, default=str), flush=True)
    finally:
        transport.close()


def install_crash(root: Path, phase: str) -> None:
    """只在测试子进程的真实事务边界杀进程；正式生产代码不含测试分支。"""
    from session.log import OwnerStore, OwnerTransaction
    from session.message import ToolResult

    marker = root / "crash-reached.json"
    if marker.exists():
        return
    if phase not in {"before_tool_result", "after_tool_result"}:
        raise ValueError("未知 fixture crash phase")
    append, transact = OwnerTransaction.append, OwnerStore.transact
    pending = False

    def crash() -> None:
        marker.write_text(json.dumps({"phase": phase, "pid": os.getpid()}))
        os.kill(os.getpid(), signal.SIGKILL)

    def record(self, writer, message_id, body, **kwargs):
        nonlocal pending
        if isinstance(body, ToolResult):
            if phase == "before_tool_result":
                crash()
            pending = True
        return append(self, writer, message_id, body, **kwargs)

    def commit(self, callback):
        result = transact(self, callback)
        if pending:
            crash()
        return result

    OwnerTransaction.append = record
    OwnerStore.transact = commit


async def serve(root: Path, endpoint: str) -> None:
    """使用正式启动和关闭路径，只在首次启动配置 loopback 模型。"""
    # 1. 所有运行数据和插件安装目录都在调用方的临时目录内。
    workspace = root / "workspace"
    settings_path = root / "fixture-config.json"
    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
    if not (root / "config.toml").is_file():
        raise RuntimeError("fixture parent 必须先完成正式安装与配置")

    if settings.get("crash_phase"):
        install_crash(root, settings["crash_phase"])

    ready = asyncio.Event()
    stop = asyncio.Event()

    class Readiness(RuntimeReadiness):
        def mark_ready(self) -> None:
            super().mark_ready()
            ready.set()

    app = AppRuntime(Config(), workspace, readiness=Readiness(workspace, f"fixture-{os.getpid()}"))
    # stdout is the fixture's one-line connection protocol.  The dashboard
    # server is real, but its access log must stay on the inherited stderr
    # side so a request cannot corrupt that protocol.
    if app.dashboard_server is not None:
        app.dashboard_server.config.access_log = False
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    loop.add_signal_handler(signal.SIGINT, stop.set)
    running = asyncio.create_task(app.run())
    waiting = asyncio.create_task(ready.wait())
    stopped = asyncio.create_task(stop.wait())
    producer = None
    try:
        # 2. readiness 必须来自真实 App，提前退出直接传播实际失败。
        done, _ = await asyncio.wait((running, waiting), timeout=30, return_when=asyncio.FIRST_COMPLETED)
        if running in done:
            await running
            raise RuntimeError("App 在 ready 前退出")
        if not ready.is_set():
            raise TimeoutError("App 未发布 readiness")
        assert app.core is not None and app.app_server is not None
        # The dashboard is created inside App.run, so the construction-time
        # flag above cannot affect its already configured Uvicorn logger.
        # Disable the shared access logger before fixture-side HTTP setup;
        # stdout remains reserved for the JSON connection line.
        logging.getLogger("uvicorn.access").disabled = True
        chat_socket = workspace / "runtime" / "chat.sock"
        print(json.dumps({"fixture_ready": True, "pid": os.getpid(),
                          "endpoint": str(app.app_server.endpoint),
                          "dashboard": app.dashboard_server.config.uds,
                          "chat": str(chat_socket)}), flush=True)
        producer = asyncio.create_task(producers(app))
        done, _ = await asyncio.wait((running, stopped, producer), return_when=asyncio.FIRST_COMPLETED)
        if producer in done:
            await producer
        if running in done:
            await running
    finally:
        # 3. 正常结束通过 App.run 的 finally 清理；强杀由父测试显式执行。
        waiting.cancel()
        stopped.cancel()
        running.cancel()
        if producer is not None:
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        await asyncio.gather(waiting, stopped, return_exceptions=True)
        try:
            await running
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    asyncio.run(serve(Path(sys.argv[1]), sys.argv[2]))
