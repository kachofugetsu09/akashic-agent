"""私有 UDS 上的 Protobuf 请求与响应；连接不拥有执行或重试。"""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass, field
import logging
import socket
import struct
from pathlib import Path
from typing import Any

import grpc
from google.protobuf.message import DecodeError, Message
from google.protobuf.message_factory import GetMessageClass

from agent.host_bridge import host_bridge_pb2 as pb

_HEADER = struct.Struct("!IBBQ")
_HELLO, _REQUEST, _REPLY, _CANCEL = range(4)
_MAX_BYTES = 16 * 1024 * 1024
_MAX_CALLS = 128
_BUSINESS_METHODS = {"Exec", "WriteStdin", "FileTool"}
_METHODS = tuple(pb.DESCRIPTOR.services_by_name["HostBridge"].methods)
_BY_NAME = {method.name: (index + 1, method) for index, method in enumerate(_METHODS)}
_CODES = {code.value[0]: code for code in grpc.StatusCode}
logger = logging.getLogger(__name__)


class RpcError(RuntimeError):
    def __init__(self, code: grpc.StatusCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _frame(kind: int, code: int, call_id: int, payload: bytes = b"") -> tuple[bytes, bytes]:
    if len(payload) > _MAX_BYTES:
        raise RpcError(grpc.StatusCode.RESOURCE_EXHAUSTED, "Host Bridge 消息超过 16 MiB")
    return _HEADER.pack(len(payload), kind, code, call_id), payload


def _header(data: bytes) -> tuple[int, int, int, int]:
    size, kind, code, call_id = _HEADER.unpack(data)
    if size > _MAX_BYTES or kind not in {_HELLO, _REQUEST, _REPLY, _CANCEL}:
        raise RpcError(grpc.StatusCode.INVALID_ARGUMENT, "Host Bridge 帧无效")
    return size, kind, code, call_id


async def _read(reader: asyncio.StreamReader) -> tuple[int, int, int, bytes]:
    size, kind, code, call_id = _header(await reader.readexactly(_HEADER.size))
    return kind, code, call_id, await reader.readexactly(size)


@dataclass(frozen=True)
class RpcContext:
    """认证仍由 service 在每次请求入口检查，连接只保存原始凭据。"""
    authorization: str

    async def abort(self, code: grpc.StatusCode, detail: str) -> None:
        raise RpcError(code, detail)


@dataclass
class _Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    pending: dict[int, asyncio.Future[tuple[int, bytes]]] = field(default_factory=dict)
    task: asyncio.Task[None] | None = None


def _cancel(connection: _Connection | None, call_id: int) -> None:
    if connection is not None and call_id and not connection.writer.is_closing():
        try:
            connection.writer.writelines(_frame(_CANCEL, 0, call_id))
        except OSError:
            connection.writer.close()


class Channel:
    """复用一个连接；丢失连接只影响在途请求，绝不重发操作。"""
    def __init__(self, path: Path, token: str) -> None:
        self.path = path
        self.token = token
        self._connection: _Connection | None = None
        self._lock = asyncio.Lock()
        self._slots = asyncio.Semaphore(_MAX_CALLS)
        self._next_id = 0
        self._closed = False

    async def _connect(self) -> _Connection:
        """为后续新调用建立连接，旧连接的等待不会迁移或重放。"""
        async with self._lock:
            if self._closed:
                raise RpcError(grpc.StatusCode.UNAVAILABLE, "Host Bridge channel 已关闭")
            if self._connection is None:
                hello = _frame(_HELLO, 0, 0, f"Bearer {self.token}".encode())
                reader, writer = await asyncio.open_unix_connection(self.path, limit=1024 * 1024)
                try:
                    writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
                    writer.writelines(hello)
                except OSError:
                    writer.close()
                    raise
                connection = _Connection(reader, writer)
                self._connection = connection
                connection.task = asyncio.create_task(self._receive(connection))
                return connection
            return self._connection

    async def _receive(self, connection: _Connection) -> None:
        """按调用编号分发响应；连接关闭会明确失败所有尚未完成的调用。"""
        error = RpcError(grpc.StatusCode.UNAVAILABLE, "Host Bridge 连接已断开，操作状态可能未知")
        try:
            while True:
                kind, code, call_id, payload = await _read(connection.reader)
                if kind != _REPLY or code not in _CODES or not 0 < call_id <= self._next_id:
                    raise RpcError(grpc.StatusCode.INTERNAL, "Host Bridge 响应帧无效")
                future = connection.pending.get(call_id)
                # 调用取消后，服务端仍可能已经发送终态；响应不能交给后续调用。
                if future is not None and not future.done():
                    future.set_result((code, payload))
        except (OSError, asyncio.IncompleteReadError):
            pass
        except RpcError as exc:
            error = exc
        finally:
            if self._connection is connection:
                self._connection = None
            for future in connection.pending.values():
                if not future.done():
                    future.set_exception(error)
            connection.writer.close()

    async def call(self, method: str, request: Message, *, timeout: float | None = None) -> Any:
        """发送一次请求，独立等待响应；取消和 deadline 只停止这次远程等待。"""
        index, descriptor = _BY_NAME[method]
        connection = None
        call_id = 0
        try:
            # 业务背压不能挡住心跳、探测和停止，否则慢命令会饿死自己的 lease。
            slots = self._slots if method in _BUSINESS_METHODS else nullcontext()
            async with asyncio.timeout(timeout), slots:
                connection = await self._connect()
                self._next_id += 1
                call_id = self._next_id
                future = asyncio.get_running_loop().create_future()
                payload = _frame(_REQUEST, index, call_id, request.SerializeToString())
                connection.pending[call_id] = future
                connection.writer.writelines(payload)
                await connection.writer.drain()
                code, response = await future
                if code:
                    raise RpcError(_CODES[code], response.decode("utf-8"))
                return GetMessageClass(descriptor.output_type).FromString(response)
        except TimeoutError as exc:
            _cancel(connection, call_id)
            raise RpcError(grpc.StatusCode.DEADLINE_EXCEEDED, "Host Bridge 请求超时，操作状态可能未知") from exc
        except asyncio.CancelledError:
            _cancel(connection, call_id)
            raise
        except OSError as exc:
            raise RpcError(grpc.StatusCode.UNAVAILABLE, str(exc)) from exc
        finally:
            if connection is not None:
                connection.pending.pop(call_id, None)

    async def close(self) -> None:
        """关闭 channel 持有的连接，并等待读任务交还所有在途等待。"""
        async with self._lock:
            self._closed = True
            connection = self._connection
            if connection is not None:
                connection.writer.close()
        if connection is not None:
            if connection.task is not None:
                await connection.task
            await connection.writer.wait_closed()


def call_sync(path: Path, token: str, method: str, request: Message, timeout: float) -> Any:
    """同步能力探测复用同一帧合同；此短连接由调用方作用域关闭。"""
    def read_exact(stream: socket.socket, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            chunk = stream.recv(size - len(data))
            if not chunk:
                raise RpcError(grpc.StatusCode.UNAVAILABLE, "Host Bridge 连接已断开")
            data.extend(chunk)
        return bytes(data)

    index, descriptor = _BY_NAME[method]
    try:
        with socket.socket(socket.AF_UNIX) as stream:
            stream.settimeout(timeout)
            stream.connect(str(path))
            stream.sendall(b"".join((*_frame(_HELLO, 0, 0, f"Bearer {token}".encode()),
                                    *_frame(_REQUEST, index, 1, request.SerializeToString()))))
            size, kind, code, call_id = _header(read_exact(stream, _HEADER.size))
            if kind != _REPLY or call_id != 1 or code not in _CODES:
                raise RpcError(grpc.StatusCode.INTERNAL, "Host Bridge 响应帧无效")
            payload = read_exact(stream, size)
            if code:
                raise RpcError(_CODES[code], payload.decode("utf-8"))
            return GetMessageClass(descriptor.output_type).FromString(payload)
    except TimeoutError as exc:
        raise RpcError(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc)) from exc
    except OSError as exc:
        raise RpcError(grpc.StatusCode.UNAVAILABLE, str(exc)) from exc


class Server:
    """复用现有 service；并发、背压和断线取消仅管理 RPC，不接管 execution。"""
    def __init__(self, service: Any) -> None:
        self.service = service
        self._server: asyncio.Server | None = None
        self._connections: dict[asyncio.Task[None], asyncio.StreamWriter] = {}
        self._closing = False

    async def start(self, path: Path) -> None:
        self._server = await asyncio.start_unix_server(self._accept, path, limit=1024 * 1024)

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._closing:
            writer.close()
            return
        try:
            writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024 * 1024)
        except OSError:
            writer.close()
            raise
        task = asyncio.create_task(self._serve(reader, writer))
        self._connections[task] = writer
        task.add_done_callback(self._finished)

    def _finished(self, task: asyncio.Task[None]) -> None:
        # 尚未开始的任务也可能被取消，socket 不能只依赖协程 finally 来关闭。
        self._connections.pop(task).close()
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Host Bridge 连接处理失败", exc_info=error)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """每个连接独立跟踪在途 RPC，取消仍能越过其他正在等待的请求。"""
        active: dict[int, asyncio.Task[None]] = {}
        try:
            kind, code, call_id, payload = await _read(reader)
            if (kind, code, call_id) != (_HELLO, 0, 0):
                raise RpcError(grpc.StatusCode.INVALID_ARGUMENT, "Host Bridge 缺少连接凭据")
            context = RpcContext(payload.decode("utf-8"))
            while True:
                kind, code, call_id, payload = await _read(reader)
                if kind == _CANCEL and code == 0 and not payload:
                    if task := active.get(call_id):
                        task.cancel()
                    continue
                if kind != _REQUEST or not 1 <= code <= len(_METHODS) or not call_id or call_id in active:
                    raise RpcError(grpc.StatusCode.INVALID_ARGUMENT, "Host Bridge 请求帧无效")
                if len(active) >= _MAX_CALLS and _METHODS[code - 1].name in _BUSINESS_METHODS:
                    writer.writelines(_frame(_REPLY, grpc.StatusCode.RESOURCE_EXHAUSTED.value[0], call_id, b"too many active calls"))
                    await writer.drain()
                    continue
                task = asyncio.create_task(self._dispatch(code, call_id, payload, context, writer))
                active[call_id] = task
                task.add_done_callback(lambda done, key=call_id: active.pop(key))
        except (OSError, asyncio.IncompleteReadError):
            pass
        except (RpcError, UnicodeDecodeError) as exc:
            # 畸形连接不能继续解析；所有在途调用看到明确断线而非伪造的成功。
            logger.warning("Host Bridge 拒绝无效连接: %s", type(exc).__name__)
            writer.close()
        finally:
            tasks = tuple(active.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    async def _dispatch(self, code: int, call_id: int, payload: bytes, context: RpcContext, writer: asyncio.StreamWriter) -> None:
        """字段校验、身份和领域错误继续由原 RPC handler 负责。"""
        descriptor = _METHODS[code - 1]
        try:
            try:
                request = GetMessageClass(descriptor.input_type).FromString(payload)
            except DecodeError as exc:
                raise RpcError(grpc.StatusCode.INVALID_ARGUMENT, "Host Bridge Protobuf 消息无效") from exc
            response = await getattr(self.service, descriptor.name)(request, context)
            frame = _frame(_REPLY, 0, call_id, response.SerializeToString())
        except RpcError as exc:
            frame = _frame(_REPLY, exc.code.value[0], call_id, exc.detail.encode("utf-8"))
        except Exception as exc:
            logger.exception("Host Bridge 响应编码失败: %s", descriptor.name)
            frame = _frame(_REPLY, grpc.StatusCode.INTERNAL.value[0], call_id, str(exc).encode("utf-8"))
        try:
            writer.writelines(frame)
            await writer.drain()
        except OSError:
            # 连接读循环负责关闭和取消其余等待；不重发已执行操作。
            writer.close()

    async def stop(self) -> None:
        """先停止接纳，再关闭 socket 并排空 RPC，包含尚未开始的读任务。"""
        self._closing = True
        if self._server is not None:
            self._server.close()
        tasks = tuple(self._connections)
        for writer in self._connections.values():
            writer.close()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._server is not None:
            await self._server.wait_closed()
