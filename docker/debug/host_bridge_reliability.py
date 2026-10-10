"""临时 UDS、真实进程和磁盘故障实验；只写 TemporaryDirectory。"""
from __future__ import annotations

import asyncio
import contextlib
from io import StringIO
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import threading
import struct
from unittest.mock import patch

import grpc
import httpx
import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent.host_bridge import client as bridge_client, filesystem
from plugins.host_execution import monitor
from agent.host_bridge import transport
from agent.host_bridge import host_bridge_pb2 as pb
from agent.host_bridge.factory import HostBridgeRpcError
from agent.host_bridge.client import HostBridgeShellProcessManager
from agent.host_bridge.server import HostBridgeService
from bootstrap.app import _run_primary_tasks
from fastapi import FastAPI
from bootstrap.web_shell import create_web_shell_app
from bootstrap.web_runtime import dashboard_socket_path
from core.common import file_io
from core.common.diagnostic_log import AkashicJsonFormatter, diagnostic_context

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
        self.hold_open_reply = False
        self.open_reply_ready = asyncio.Event()
        self.open_reply_cancelled = asyncio.Event()

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
        reply = await super().OpenManager(request, context)
        if self.hold_open_reply:
            self.open_reply_ready.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.open_reply_cancelled.set()
        return reply

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
        server = transport.Server(service)
        await server.start(socket)
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

            # 超时或取消后立即关闭，不能靠下一次发送带走取消帧的空尾段。
            for outcome in ("deadline", "cancel"):
                closing = client()
                service.hold_open_reply = True
                service.open_reply_ready.clear()
                service.open_reply_cancelled.clear()
                before = service.calls.get("open", 0)
                call = asyncio.create_task(closing._channel.call(
                    "OpenManager", pb.ContextRequest(context=closing._request_context()),
                    timeout=0.2 if outcome == "deadline" else None,
                ))
                tasks.append(call)
                await asyncio.wait_for(service.open_reply_ready.wait(), 2)
                if outcome == "cancel":
                    call.cancel()
                    try:
                        await call
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError("取消必须结束本次 RPC 等待")
                else:
                    try:
                        await call
                    except transport.RpcError as error:
                        assert error.code is grpc.StatusCode.DEADLINE_EXCEEDED
                    else:
                        raise AssertionError("延迟响应必须超过 deadline")
                await asyncio.wait_for(service.open_reply_cancelled.wait(), 2)
                await asyncio.wait_for(closing.close_transport(), 2)
                identity = (closing._boot_id, closing._manager_id)
                assert identity in service._managers, "关闭连接不能删除已登记的 manager"
                assert service.calls["open"] == before + 1, "超时或取消不得重放请求"
                service.hold_open_reply = False
                await control._channel.call(
                    "ShutdownManager", pb.ContextRequest(context=closing._request_context()),
                )
                results.append(f"{outcome}_then_immediate_close_without_replay")

            # 空 manager 的清理响应没有 Protobuf 载荷；服务端也必须能排空关闭。
            empty = client()
            assert not await empty.active_execution_ids()
            assert not (await asyncio.wait_for(empty.shutdown(), 2)).failures
            await asyncio.wait_for(server.stop(), 2)
            server = transport.Server(service)
            await server.start(socket)
            results.append("empty_reply_then_client_and_server_close")

            status = monitor.HostBridgeStatus(state="checking")
            service.probe_error = grpc.StatusCode.DEADLINE_EXCEEDED
            monitoring = asyncio.create_task(monitor._monitor(client(), status=status))
            sibling = asyncio.create_task(asyncio.Event().wait())
            primary = asyncio.create_task(_run_primary_tasks([monitoring, sibling]))
            tasks.append(primary)
            await service.until("probe", 2)
            assert not primary.done() and not sibling.done()
            assert status.state == "degraded" and status.code == "DEADLINE_EXCEEDED"
            assert not service._managers, "健康探测不能创建 execution manager"
            workspace = root / "dashboard"
            app = FastAPI()
            @app.get("/api/runtime/host-bridge")
            async def read_status():
                return status.snapshot()
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
            await server.stop()
            await expect_rpc(grpc.StatusCode.UNAVAILABLE, control.probe())
            await until(lambda: status.state == "degraded")
            assert not primary.done() and not sibling.done()
            server = transport.Server(service)
            await server.start(socket)
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

            # 同一连接的长命令不阻塞后续请求；取消只结束等待，execution 仍可清理。
            concurrent = client()
            pending = asyncio.create_task(command(concurrent, "printf ready > parallel-ready; sleep 20"))
            await until(lambda: (root / "parallel-ready").exists())
            async with asyncio.timeout(1):
                short = await command(concurrent, "printf short")
                assert short.output == b"short"
                await asyncio.gather(*(concurrent.probe() for _ in range(160)))
            pending.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pending
            assert await concurrent.active_execution_ids()
            assert not (await concurrent.terminate_owner("experiment")).failures
            assert not await concurrent.active_execution_ids()
            results.append("multiplexed_calls_and_cancellation_keep_execution_owner")

            # 真实 deadline 后，命令副作用仍只发生一次，并能用原 manager 查找与清理。
            request = pb.ExecRequest(context=concurrent._request_context(),
                command="printf x >> deadline-once; sleep 20",
                argv=["/bin/sh", "-c", "printf x >> deadline-once; sleep 20"], cwd=str(root),
                tty=False, yield_time_ms=10000, max_output_tokens=1000,
                hard_timeout_s=30, owner_session_key="deadline")
            try:
                await concurrent._channel.call("Exec", request, timeout=0.5)
            except transport.RpcError as exc:
                assert exc.code is grpc.StatusCode.DEADLINE_EXCEEDED
            else:
                raise AssertionError("真实长命令必须超过 deadline")
            assert (root / "deadline-once").read_text() == "x"
            assert await concurrent.active_execution_ids()
            assert not (await concurrent.terminate_owner("deadline")).failures
            results.append("deadline_keeps_execution_and_does_not_replay")

            lost = asyncio.create_task(command(concurrent, "printf x >> disconnected-once; sleep 20"))
            await until(lambda: (root / "disconnected-once").exists())
            await server.stop()
            await expect_rpc(grpc.StatusCode.UNAVAILABLE, lost)
            server = transport.Server(service)
            await server.start(socket)
            assert (root / "disconnected-once").read_text() == "x"
            assert await concurrent.active_execution_ids()
            assert not (await concurrent.terminate_owner("experiment")).failures
            results.append("disconnect_during_exec_keeps_handle_without_replay")

            checker = bridge_client.HostBridgeRequirementsChecker(socket, "experiment", TOKEN, COMMIT, DIGEST)
            available = await asyncio.to_thread(checker.check_requirements, ["sh"], ["PATH"])
            assert available.available_bins == ("sh",) and available.available_env == ("PATH",)
            results.append("synchronous_requirements_use_same_wire_contract")

            # 畸形 Protobuf 不能执行命令，也不能把下一条请求的响应错配。
            reader, stream = await asyncio.open_unix_connection(socket)
            def packet(kind, code, call_id, payload=b""):
                return struct.pack("!IBBQ", len(payload), kind, code, call_id) + payload
            methods = {method.name: index + 1 for index, method in enumerate(
                pb.DESCRIPTOR.services_by_name["HostBridge"].methods)}
            before = service.calls["exec"]
            stream.write(packet(0, 0, 0, f"Bearer {TOKEN}".encode()) + packet(1, methods["Exec"], 1, b"\xff"))
            kind, code, call_id, payload = await transport._read(reader)
            assert (kind, code, call_id) == (2, 3, 1) and payload
            assert service.calls["exec"] == before
            stream.write(packet(1, methods["Probe"], 2, pb.ContextRequest(context=concurrent._request_context()).SerializeToString()))
            kind, code, call_id, payload = await transport._read(reader)
            assert (kind, code, call_id) == (2, 0, 2)
            assert pb.IdentityReply.FromString(payload).release_commit == COMMIT
            stream.close()
            await stream.wait_closed()
            # 超限长度在分配或等待消息体之前拒绝。
            reader, stream = await asyncio.open_unix_connection(socket)
            stream.write(struct.pack("!IBBQ", 16 * 1024 * 1024 + 1, 0, 0, 0))
            async with asyncio.timeout(1):
                assert await reader.read() == b""
            stream.close()
            await stream.wait_closed()
            results.append("invalid_protobuf_and_oversized_frame_never_execute")

            # 真实成功/失败 RPC 的延后日志仍携带调用上下文与原始异常。
            captured = StringIO()
            handler = logging.StreamHandler(captured)
            handler.setFormatter(AkashicJsonFormatter(
                ("levelname", "name", "message", "process"),
                rename_fields={"levelname": "level", "name": "logger", "process": "pid", "exc_info": "exception"},
            ))
            rpc_logger = logging.getLogger("agent.host_bridge.server")
            rpc_logger.addHandler(handler)
            try:
                with diagnostic_context(session="diagnostic-session", turn="diagnostic-turn"):
                    assert (await command(concurrent, "printf diagnostic")).output == b"diagnostic"
                    await expect_rpc(grpc.StatusCode.INTERNAL, concurrent.exec_command(
                        command="true", argv=["/bin/sh", "-c", "true"], cwd=root / "missing",
                        env={}, tty=False, yield_time_ms=1000, max_output_tokens=100,
                        hard_timeout_s=20, owner_session_key="diagnostic"))
                await asyncio.sleep(0)  # 排空已入队的回调，不等待墙钟时间。
                rows = [json.loads(line) for line in captured.getvalue().splitlines()]
                scoped = [row for row in rows if row.get("session") == "diagnostic-session"]
                assert all(row["turn"] == "diagnostic-turn" and row["request_id"] for row in scoped)
                assert {row["event"] for row in scoped} >= {"host_bridge.rpc_started", "host_bridge.rpc_completed"}
                # 既有错误边界退出 session/turn scope 后，用 request_id 关联原调用。
                request_ids = {row["request_id"] for row in scoped}
                failed = [row for row in rows if row.get("event") == "host_bridge.rpc_failed" and row.get("request_id") in request_ids]
                assert failed and "FileNotFoundError" in json.dumps(failed)
            finally:
                rpc_logger.removeHandler(handler)
                handler.close()
            results.append("deferred_diagnostics_keep_context_and_exception")

            # 真正启动宿主进程：展示字段保留空值，Core 的身份和 PATH 不越界。
            env_reader = client()
            text = 'printf "%s|%s|%s|%s" "$AKASHIC_CALL_CONTEXT" "${GH_PAGER-unset}" "$HOME" "${HB_UNUSED-unset}"'
            result = await env_reader.exec_command(
                command=text, argv=["/bin/sh", "-c", text], cwd=root,
                env={"AKASHIC_CALL_CONTEXT": "context-marker", "GH_PAGER": "",
                     "HOME": "/untrusted-core-home", "PATH": "/untrusted-core-bin",
                     "HB_UNUSED": "x" * 65536},
                tty=False, yield_time_ms=1000, max_output_tokens=1000,
                hard_timeout_s=20, owner_session_key="environment-check",
            )
            expected = f"context-marker||{os.environ.get('HOME', '')}|{os.environ.get('HB_UNUSED', 'unset')}"
            assert result.exit_code == 0 and result.output.decode() == expected
            assert not (await env_reader.shutdown()).failures
            results.append("execution_environment_keeps_host_identity_and_empty_values")

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
            await server.stop()


if __name__ == "__main__":
    asyncio.run(run())
