"""临时 UDS、真实进程和磁盘故障实验；只写 TemporaryDirectory。"""
from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
import sys
import tempfile
import threading
from unittest.mock import patch

import grpc
import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.host_bridge import client as bridge_client, filesystem, monitor
from agent.host_bridge import host_bridge_pb2_grpc as rpc
from agent.host_bridge.client import HostBridgeRpcError, HostBridgeShellProcessManager
from agent.host_bridge.server import HostBridgeService
from bootstrap.app import _run_primary_tasks
from bootstrap.dashboard_api import create_dashboard_app
from bootstrap.web_shell import create_web_shell_app
from bootstrap.web_runtime import dashboard_socket_path
from core.common import file_io

COMMIT = "a" * 40
DIGEST = "b" * 64
TOKEN = "local-experiment"


class FaultService(HostBridgeService):
    """仅在真实 handler 边界注入传输错误，业务仍由生产代码执行。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probe_error = None
        self.heartbeat_error = None
        self.drop_exec_reply = False
        self.calls: dict[str, int] = {}
        self.changed = asyncio.Condition()

    async def record(self, method):
        async with self.changed:
            self.calls[method] = self.calls.get(method, 0) + 1
            self.changed.notify_all()

    async def until(self, method, count):
        async with asyncio.timeout(3), self.changed:
            await self.changed.wait_for(lambda: self.calls.get(method, 0) >= count)

    async def Probe(self, request, context):
        reply = await super().Probe(request, context)
        await self.record("probe")
        if self.probe_error is not None:
            await context.abort(self.probe_error, "实验：探测故障")
        return reply

    async def Heartbeat(self, request, context):
        await self.record("heartbeat")
        if self.heartbeat_error is not None:
            await context.abort(self.heartbeat_error, "实验：心跳故障")
        return await super().Heartbeat(request, context)

    async def OpenManager(self, request, context):
        await self.record("open")
        return await super().OpenManager(request, context)

    async def Exec(self, request, context):
        await self.record("exec")
        reply = await super().Exec(request, context)
        if self.drop_exec_reply:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "实验：命令执行后丢失响应")
        return reply

    async def FileTool(self, request, context):
        try:
            return await super().FileTool(request, context)
        finally:
            await self.record("file_finished")


async def expect_rpc(code, awaitable):
    """明确核对状态码，不把任何异常都算作预期失败。"""
    try:
        await awaitable
    except HostBridgeRpcError as exc:
        assert exc.code is code, str(exc)
        return exc
    raise AssertionError(f"预期 {code.name}")


async def until(predicate):
    """外部服务和线程的完成信号有期限；不靠固定等待猜测完成。"""
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.005)


async def run() -> None:
    """每个断言观察真实边界结果，最后排空线程、RPC 和子进程。"""
    results = []
    with tempfile.TemporaryDirectory(prefix="bridge-reliability-") as temporary:
        root = Path(temporary)
        socket = root / "bridge.sock"
        service = FaultService(TOKEN, 0.1, root / "artifacts", release_commit=COMMIT,
                               toolchain_digest=DIGEST, runtime_checkout=ROOT,
                               bridge_python=Path(sys.executable))
        server = grpc.aio.server()
        rpc.add_HostBridgeServicer_to_server(service, server)
        assert server.add_insecure_port(f"unix:{socket}")
        await server.start()
        clients = []
        tasks = []
        dashboard = None

        def client(boot="experiment", token=TOKEN):
            value = HostBridgeShellProcessManager(socket, boot, token, COMMIT, DIGEST)
            clients.append(value)
            return value

        async def command(owner, text):
            return await owner.exec_command(command=text, argv=["/bin/sh", "-c", text],
                cwd=root, env={}, tty=False, yield_time_ms=1000, max_output_tokens=1000,
                hard_timeout_s=20, owner_session_key="experiment")

        try:
            monitor._MONITOR_INTERVAL_S = 0.01
            bridge_client._HEARTBEAT_INTERVAL_S = 0.01
            control = client()
            await control.claim_boot()
            status = monitor.HostBridgeStatus(state="checking")
            service.probe_error = grpc.StatusCode.DEADLINE_EXCEEDED
            monitoring = asyncio.create_task(monitor._monitor(socket, "experiment", TOKEN, COMMIT, DIGEST, status=status))
            sibling = asyncio.create_task(asyncio.Event().wait())
            primary = asyncio.create_task(_run_primary_tasks([monitoring, sibling]))
            tasks.append(primary)
            await service.until("probe", 2)
            assert not primary.done() and not sibling.done()
            assert status.state == "degraded" and status.code == "DEADLINE_EXCEEDED"
            assert not service._managers, "健康探测不能创建 execution manager"
            workspace = root / "dashboard"
            app = create_dashboard_app(workspace, host_bridge_status=status.snapshot)
            dashboard_socket = dashboard_socket_path(workspace)
            dashboard_socket.parent.mkdir(parents=True, exist_ok=True)
            dashboard = uvicorn.Server(uvicorn.Config(app, uds=str(dashboard_socket), log_level="error"))
            dashboard_task = asyncio.create_task(dashboard.serve())
            tasks.append(dashboard_task)
            await until(lambda: dashboard.started)
            shell = create_web_shell_app(root / "config.json", workspace)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=shell), base_url="http://local",
                                         headers={"sec-fetch-site": "same-origin"}) as web:
                response = await web.get("/api/runtime/host-bridge")
                assert response.json()["state"] == "degraded"
                service.probe_error = None
                await service.until("probe", 4)
                assert (await web.get("/api/runtime/host-bridge")).json()["state"] == "healthy"
            results.append("probe_deadline_core_survives_and_public_shell_http_recovers")

            # 真正断开 UDS transport，然后重建监听；保留 service 的 boot/lease owner。
            await server.stop(0)
            await expect_rpc(grpc.StatusCode.UNAVAILABLE, control.probe())
            await until(lambda: status.state == "degraded")
            assert not primary.done() and not sibling.done()
            server = grpc.aio.server()
            rpc.add_HostBridgeServicer_to_server(service, server)
            assert server.add_insecure_port(f"unix:{socket}")
            await server.start()
            await until(lambda: status.state == "healthy")
            results.append("real_transport_disconnect_and_reconnect")

            # 并发首次调用只登记一次；短暂心跳故障保持同一个 execution owner。
            worker = client()
            before = service.calls.get("open", 0)
            await asyncio.gather(*(worker.active_execution_ids() for _ in range(8)))
            assert service.calls["open"] == before + 1
            identity = (worker._boot_id, worker._manager_id)
            original = service._managers[identity]
            service.heartbeat_error = grpc.StatusCode.UNAVAILABLE
            count = service.calls.get("heartbeat", 0)
            await service.until("heartbeat", count + 2)
            service.heartbeat_error = None
            await service.until("heartbeat", count + 4)
            assert worker._lease_error is None and service._managers[identity] is original
            result = await command(worker, "printf recovered")
            assert result.exit_code == 0
            results.append("one_acquire_and_heartbeat_recovery")

            # 超过 lease 的失联必须终结旧 manager，不能重建一个空执行表。
            await worker._stop_heartbeat()
            original.last_seen = 0
            reaper = asyncio.create_task(service.reap_expired())
            tasks.append(reaper)
            async with asyncio.timeout(2):
                while identity in service._managers:
                    await asyncio.sleep(0.01)
            reaper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reaper
            opens = service.calls["open"]
            await expect_rpc(grpc.StatusCode.NOT_FOUND, worker.active_execution_ids())
            await expect_rpc(grpc.StatusCode.NOT_FOUND, worker.shutdown())
            assert service.calls["open"] == opens and identity not in service._managers
            results.append("expired_manager_rejected_without_reopen_or_false_cleanup")

            # 业务执行后响应丢失：文件只能增加一次，客户端不得重发。
            writer = client()
            service.drop_exec_reply = True
            error = await expect_rpc(grpc.StatusCode.UNAVAILABLE,
                                    command(writer, "printf x >> once.txt"))
            assert "不得自动重发" in str(error)
            service.drop_exec_reply = False
            assert (root / "once.txt").read_text() == "x"
            assert service.calls["exec"] == 2
            results.append("lost_exec_reply_no_replay")

            # 单次命令参数/内部失败不能把仍存在的 manager 宣判为丢失。
            await expect_rpc(grpc.StatusCode.INTERNAL, writer.exec_command(
                command="true", argv=["/bin/sh", "-c", "true"], cwd=root / "missing",
                env={}, tty=False, yield_time_ms=1000, max_output_tokens=100,
                hard_timeout_s=20, owner_session_key="experiment"))
            assert (await command(writer, "printf still-alive")).exit_code == 0
            results.append("business_error_does_not_poison_manager")

            # 四种文件操作走真实磁盘；慢写入期间 Probe 与同文件串行性都保留。
            io = client()
            entered = asyncio.Event()
            release = threading.Event()
            loop = asyncio.get_running_loop()
            real_write = filesystem.atomic_write_text
            def slow_write(path, content, **kwargs):
                if content == "first":
                    loop.call_soon_threadsafe(entered.set)
                    if not release.wait(3):
                        raise TimeoutError("实验未释放慢写")
                return real_write(path, content, **kwargs)
            with patch.object(filesystem, "atomic_write_text", slow_write):
                writing = asyncio.create_task(io.execute_file_tool("write_file", allowed_dir=root,
                    arguments={"path": "slow.txt", "content": "first"}))
                tasks.append(writing)
                try:
                    await asyncio.wait_for(entered.wait(), 2)
                    await asyncio.wait_for(control.probe(), 0.3)
                    writing.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await writing
                    lease = service._managers[(io._boot_id, io._manager_id)]
                    assert lease.active_operations == 1 and not lease.operations_drained.is_set()
                    second = asyncio.create_task(io.execute_file_tool("write_file", allowed_dir=root,
                        arguments={"path": "slow.txt", "content": "second"}))
                    tasks.append(second)
                    await until(lambda: any(state.users == 2 for state in filesystem._FILE_MUTATION_LOCKS.values()))
                    shutdown = asyncio.create_task(io.shutdown())
                    tasks.append(shutdown)
                    await until(lambda: lease.reaping)
                    assert not lease.operations_drained.is_set() and not shutdown.done()
                    release.set()
                    await second
                    assert not (await shutdown).failures
                    assert (root / "slow.txt").read_text() == "second"
                finally:
                    release.set()
            io = client()
            edited = await io.execute_file_tool("edit_file", allowed_dir=root,
                arguments={"path": "slow.txt", "old_text": "second", "new_text": "third"})
            assert isinstance(edited, str)
            read = await io.execute_file_tool("read_file", allowed_dir=root, arguments={"path": "slow.txt"})
            listing = await io.execute_file_tool("list_dir", allowed_dir=root, arguments={"path": "."})
            assert "third" in str(read) and "slow.txt" in str(listing)
            assert not filesystem._FILE_MUTATION_LOCKS and not file_io._FILE_IO_SLOTS
            results.append("slow_disk_probe_cancel_drain_and_four_file_operations")

            # 认证错误不能被恢复策略吞掉；旧 boot 的所有执行入口被 fencing。
            await expect_rpc(grpc.StatusCode.PERMISSION_DENIED, client(token="wrong").probe())
            await control.close_transport()
            primary.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await primary
            next_boot = client("next-boot")
            await next_boot.claim_boot()
            await expect_rpc(grpc.StatusCode.PERMISSION_DENIED, io.probe())
            await expect_rpc(grpc.StatusCode.PERMISSION_DENIED, io.active_execution_ids())
            assert not service._managers
            results.append("auth_and_boot_fencing_stay_fatal")
            print(json.dumps({"result": "passed", "experiments": results}, ensure_ascii=False))
        finally:
            if dashboard is not None:
                dashboard.should_exit = True
                await asyncio.wait_for(dashboard_task, 3)
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            for value in clients:
                await value.close_transport()
            await service.shutdown()
            await server.stop(0)


if __name__ == "__main__":
    asyncio.run(run())
