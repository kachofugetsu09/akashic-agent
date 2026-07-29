"""McpClient: 管理单个 MCP server 的 stdio 子进程连接和 JSON-RPC 通信。"""

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from utils.process_group import (
    OwnedProcessGroup,
    owned_process_env,
    process_group_spawn_kwargs,
)

logger = logging.getLogger(__name__)

_RECV_TIMEOUT = 30.0
_CONNECT_TIMEOUT = 8.0
_DISCONNECT_TIMEOUT = 5.0
_STREAM_LIMIT = 4 * 1024 * 1024  # 4 MB，防止大响应触发 StreamReader 行限
_MCP_PROTOCOL_VERSION = "2025-11-25"
_SUPPORTED_PROTOCOL_VERSIONS = frozenset(
    {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"}
)
_STRUCTURED_CONTENT_PROTOCOL_VERSIONS = frozenset(
    {"2025-06-18", "2025-11-25"}
)
_MCP_CONTENT_BLOCK_TYPES = frozenset(
    {"text", "image", "resource"}
)


@dataclass
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


class McpToolExecutionError(RuntimeError):
    """MCP 远端工具已执行但返回失败结果。"""


def _infer_cwd(command: list[str]) -> str | None:
    """从 command 中找第一个绝对路径文件，返回其父目录作为 cwd。"""
    for arg in command:
        p = Path(arg)
        if p.is_absolute() and p.is_file():
            return str(p.parent)
    return None


async def _wait_for_leader_exit(process: asyncio.subprocess.Process) -> int | None:
    """等待 leader returncode，不被仍持有 stdio pipe 的孙进程阻塞。"""
    wait_task = asyncio.create_task(process.wait())
    try:
        while process.returncode is None and not wait_task.done():
            await asyncio.sleep(0.05)
        if wait_task.done():
            _ = await wait_task
        return process.returncode
    finally:
        if not wait_task.done():
            _ = wait_task.cancel()
            _ = await asyncio.gather(wait_task, return_exceptions=True)


class McpClient:
    """启动并管理一个 stdio MCP server 子进程，处理 JSON-RPC 通信。"""

    def __init__(
        self,
        name: str,
        command: list[str],
        env: dict[str, str] | None = None,
        cwd: str | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.env = env or {}
        # cwd 未指定时从 command 中推断，避免子进程继承 agent 工作目录
        self.cwd = cwd or _infer_cwd(command)
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 1
        self._call_lock = asyncio.Lock()
        self._tool_infos: list[McpToolInfo] = []
        self._recent_stdout: deque[str] = deque(maxlen=8)
        self._recent_stderr: deque[str] = deque(maxlen=8)
        self._stderr_task: asyncio.Task[None] | None = None
        self._process_group: OwnedProcessGroup | None = None
        self._process_watch_task: asyncio.Task[None] | None = None
        self._disconnect_task: asyncio.Task[None] | None = None
        self._disconnecting = False
        self._protocol_version: str | None = None

    @property
    def tool_infos(self) -> list[McpToolInfo]:
        return self._tool_infos

    @property
    def connected(self) -> bool:
        return self._process is not None and self._process.returncode is None

    async def connect(self) -> list[McpToolInfo]:
        try:
            return await asyncio.wait_for(
                self._connect_impl(),
                timeout=_CONNECT_TIMEOUT,
            )
        except BaseException:
            await self.disconnect()
            raise

    async def _connect_impl(self) -> list[McpToolInfo]:
        """启动子进程，完成握手，获取工具列表。"""
        proc_env = owned_process_env(self.env)
        logger.debug("[mcp] 启动 %r: %s  cwd=%s", self.name, self.command, self.cwd)
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            cwd=self.cwd,
            limit=_STREAM_LIMIT,
            **process_group_spawn_kwargs(),
        )
        self._process_group = OwnedProcessGroup.from_process(self._process)
        self._stderr_task = asyncio.create_task(
            self._drain_stderr(),
            name=f"mcp_stderr:{self.name}",
        )
        if self._process_group.group_id is not None:
            self._process_watch_task = asyncio.create_task(
                self._watch_process_exit(self._process, self._process_group),
                name=f"mcp_process:{self.name}",
            )

        # initialize 握手
        init_id = self._new_id()
        await self._send(
            {
                "jsonrpc": "2.0",
                "id": init_id,
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "clientInfo": {"name": "akashic-agent", "version": "1.0"},
                },
            }
        )
        init_response = await self._recv(expected_id=init_id, stage="initialize")
        init_result = self._response_result(init_response, "initialize")
        protocol_version = init_result.get("protocolVersion")
        if protocol_version not in _SUPPORTED_PROTOCOL_VERSIONS:
            raise RuntimeError(
                f"MCP server {self.name!r} 返回了不支持的协议版本："
                f"{protocol_version!r}"
            )
        self._protocol_version = cast(str, protocol_version)

        # initialized 通知（无 id，不等响应）
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 获取工具列表
        list_id = self._new_id()
        await self._send(
            {"jsonrpc": "2.0", "id": list_id, "method": "tools/list", "params": {}}
        )
        response = await self._recv(expected_id=list_id, stage="tools/list")
        result = self._response_result(response, "tools/list")
        raw_tools: object = result.get("tools", [])
        if not isinstance(raw_tools, list):
            raise RuntimeError(f"MCP server {self.name!r} tools/list.tools 不是 list")
        tool_infos: list[McpToolInfo] = []
        for raw_tool in cast(list[object], raw_tools):
            if not isinstance(raw_tool, dict):
                raise RuntimeError(f"MCP server {self.name!r} 返回了无效 tool")
            tool = cast(dict[str, Any], raw_tool)
            name = tool.get("name")
            schema = tool.get("inputSchema")
            description = tool.get("description", "")
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(schema, dict)
            ):
                raise RuntimeError(
                    f"MCP server {self.name!r} 返回了无效 tool（需要 name 和 inputSchema）"
                )
            schema = cast(dict[str, Any], schema)
            if schema.get("type") != "object":
                raise RuntimeError(
                    f"MCP server {self.name!r} 返回了无效 tool（需要 object inputSchema）"
                )
            if not isinstance(description, str):
                raise RuntimeError(
                    f"MCP server {self.name!r} 返回了无效 tool.description"
                )
            tool_infos.append(
                McpToolInfo(
                    name=name,
                    description=description,
                    input_schema=cast(dict[str, Any], schema),
                )
            )
        self._tool_infos = tool_infos
        logger.debug(
            "[mcp] %r 已连接，工具：%s", self.name, [t.name for t in self._tool_infos]
        )
        return self._tool_infos

    async def call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        timeout: float | None = None,
    ) -> str:
        """调用远端工具，返回结果字符串。"""
        async with self._call_lock:
            call_id = self._new_id()
            await self._send(
                {
                    "jsonrpc": "2.0",
                    "id": call_id,
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }
            )
            resp = await self._recv(
                expected_id=call_id,
                stage=f"tools/call:{tool_name}",
                timeout=timeout,
            )

        if "error" in resp:
            err: object = resp["error"]
            if not isinstance(err, dict):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"error（类型={type(err).__name__}，值={err!r}）"
                )
            error_object = cast(dict[str, object], err)
            detail = error_object.get("message", error_object)
            raise McpToolExecutionError(
                f"MCP server {self.name!r} tools/call:{tool_name} JSON-RPC error: "
                f"{detail}"
            )

        result = self._response_result(resp, f"tools/call:{tool_name}")
        raw_content = result.get("content")
        if not isinstance(raw_content, list):
            raise RuntimeError(
                f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                f"content（类型={type(raw_content).__name__}，值={raw_content!r}）"
            )

        rendered: list[str] = []
        for index, raw_block in enumerate(cast(list[object], raw_content)):
            if not isinstance(raw_block, dict):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"content[{index}]（类型={type(raw_block).__name__}，"
                    f"值={raw_block!r}）"
                )
            block = cast(dict[str, object], raw_block)
            rendered.append(
                self._render_content_block(
                    block,
                    tool_name=tool_name,
                    index=index,
                )
            )

        if "structuredContent" in result:
            if self._protocol_version not in _STRUCTURED_CONTENT_PROTOCOL_VERSIONS:
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"structuredContent（协议 {self._protocol_version or '未协商'} "
                    "不支持）"
                )
            structured_content = result["structuredContent"]
            if not isinstance(structured_content, dict):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    "structuredContent（需要 object）"
                )

        is_error = result.get("isError", False)
        if not isinstance(is_error, bool):
            raise RuntimeError(
                f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                f"isError（类型={type(is_error).__name__}，值={is_error!r}）"
            )
        output = "\n".join(rendered)
        if is_error:
            raise McpToolExecutionError(
                f"MCP server {self.name!r} tools/call:{tool_name} 执行失败: "
                f"{output or '服务端未返回错误内容'}"
            )
        return output

    def _render_content_block(
        self,
        block: dict[str, object],
        *,
        tool_name: str,
        index: int,
    ) -> str:
        """校验 MCP 内容块并转换为工具文本。"""
        block_type = block.get("type")
        if not isinstance(block_type, str) or block_type not in _MCP_CONTENT_BLOCK_TYPES:
            raise RuntimeError(
                f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                f"content[{index}].type（值={block_type!r}）"
            )

        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"content[{index}].text（类型={type(text).__name__}，"
                    f"值={text!r}）"
                )
        elif block_type == "image":
            for field in ("data", "mimeType"):
                if not isinstance(block.get(field), str):
                    raise RuntimeError(
                        f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                        f"content[{index}].{field}"
                    )
        else:
            resource = block.get("resource")
            if not isinstance(resource, dict):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"content[{index}].resource"
                )
            resource_value = cast(dict[str, object], resource)
            if not isinstance(resource_value.get("uri"), str):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"content[{index}].resource.uri"
                )
            mime_type = resource_value.get("mimeType")
            if mime_type is not None and not isinstance(mime_type, str):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"content[{index}].resource.mimeType"
                )
            text = resource_value.get("text")
            blob = resource_value.get("blob")
            if (isinstance(text, str)) == (isinstance(blob, str)):
                raise RuntimeError(
                    f"MCP server {self.name!r} tools/call:{tool_name} 返回了无效 "
                    f"content[{index}].resource（需要 text 或 blob）"
                )

        return cast(str, block["text"]) if block_type == "text" else json.dumps(
            block,
            ensure_ascii=False,
            sort_keys=True,
        )

    async def disconnect(self) -> None:
        """抗取消地终止 MCP 进程组，并只在回收成功后释放 ownership。"""
        task = self._disconnect_task
        if task is None:
            if self._process is None:
                return
            task = asyncio.create_task(
                self._disconnect_impl(),
                name=f"mcp_disconnect:{self.name}",
            )
            self._disconnect_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            _ = await task
            raise
        finally:
            if task.done() and self._disconnect_task is task:
                self._disconnect_task = None

    async def _disconnect_impl(self) -> None:
        """关闭 stdio 后清理稳定 PGID，并聚合通信与回收错误。"""
        process = self._process
        if process is None:
            return
        process_group = self._process_group or OwnedProcessGroup.from_process(process)
        self._process_group = process_group
        self._disconnecting = True
        errors: list[BaseException] = []
        hard_kill = False
        cleanup_succeeded = False

        # 1. 先关闭 stdin，给规范 MCP server 一个自然退出机会
        try:
            if process.stdin is not None:
                process.stdin.close()
        except BaseException as error:
            errors.append(error)
            hard_kill = True
        if not hard_kill and os.name != "nt":
            try:
                _ = await asyncio.wait_for(
                    _wait_for_leader_exit(process),
                    timeout=_DISCONNECT_TIMEOUT,
                )
            except TimeoutError:
                pass
            except BaseException as error:
                errors.append(error)
                hard_kill = True

        # 2. leader 即使已经退出，仍按保存的 PGID 回收 wrapper 后代
        try:
            if hard_kill:
                await process_group.kill(timeout_s=_DISCONNECT_TIMEOUT)
            else:
                await process_group.terminate(timeout_s=_DISCONNECT_TIMEOUT)
        except BaseException as error:
            errors.append(error)
        else:
            cleanup_succeeded = True

        # 3. 只有进程组回收成功，才释放 process 与诊断 watcher ownership
        if cleanup_succeeded:
            self._process = None
            self._process_group = None
            self._protocol_version = None
            process_watch_task = self._process_watch_task
            self._process_watch_task = None
            if process_watch_task is not None:
                _ = await asyncio.gather(process_watch_task, return_exceptions=True)
            stderr_task = self._stderr_task
            self._stderr_task = None
            if stderr_task is not None:
                if not stderr_task.done():
                    _ = stderr_task.cancel()
                _ = await asyncio.gather(stderr_task, return_exceptions=True)
        self._disconnecting = False

        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(f"MCP server {self.name!r} 清理失败", errors)

    async def _watch_process_exit(
        self,
        process: asyncio.subprocess.Process,
        process_group: OwnedProcessGroup,
    ) -> None:
        """leader 意外退出后立即清理仍存活的 wrapper 后代。"""
        while process.returncode is None:
            await asyncio.sleep(0.05)
        exit_code = process.returncode
        if self._process is not process or self._disconnecting:
            return
        logger.error(
            "[mcp] %r 进程意外退出: exit=%s，开始清理进程组 %s",
            self.name,
            exit_code,
            process_group.group_id,
        )
        try:
            await process_group.terminate(timeout_s=_DISCONNECT_TIMEOUT)
        except Exception:
            logger.exception(
                "[mcp] %r 意外退出后的进程组清理失败",
                self.name,
            )

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._process and self._process.stdin
        serialized = json.dumps(payload, ensure_ascii=False)
        logger.debug(
            "[mcp:%s] -> %s",
            self.name,
            serialized[:400],
        )
        self._process.stdin.write((serialized + "\n").encode())
        await self._process.stdin.drain()

    async def _recv(
        self,
        expected_id: int | None = None,
        stage: str = "recv",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        assert self._process and self._process.stdout
        recv_timeout = _RECV_TIMEOUT if timeout is None else timeout
        while True:
            try:
                line = await asyncio.wait_for(
                    self._process.stdout.readline(), timeout=recv_timeout
                )
            except asyncio.TimeoutError as e:
                raise TimeoutError(
                    self._build_timeout_message(stage, expected_id, recv_timeout)
                ) from e
            if not line:
                raise ConnectionError(f"MCP server {self.name!r} 意外关闭了 stdout")
            text = line.decode().strip()
            if not text:
                continue
            self._recent_stdout.append(text[:500])
            try:
                raw_message: object = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("[mcp:%s] 非 JSON 输出: %s", self.name, text[:200])
                continue
            if not isinstance(raw_message, dict):
                raise RuntimeError(f"MCP server {self.name!r} 返回了非对象响应")
            msg = cast(dict[str, Any], raw_message)
            # 跳过通知（有 method 但无 id）
            if "method" in msg and "id" not in msg:
                logger.debug("[mcp:%s] <- notification: %s", self.name, text[:400])
                continue
            if expected_id is not None and msg.get("id") != expected_id:
                logger.debug(
                    "[mcp:%s] <- skip id=%r expect=%r: %s",
                    self.name,
                    msg.get("id"),
                    expected_id,
                    text[:400],
                )
                continue
            logger.debug("[mcp:%s] <- %s", self.name, text[:400])
            return msg

    def _response_result(
        self,
        response: dict[str, Any],
        stage: str,
    ) -> dict[str, Any]:
        if "error" in response:
            raise RuntimeError(
                f"MCP server {self.name!r} {stage} 失败: {response['error']}"
            )
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError(
                f"MCP server {self.name!r} {stage} 返回了无效 result"
            )
        return cast(dict[str, Any], result)

    async def _drain_stderr(self) -> None:
        """后台读取 stderr，防止缓冲区阻塞。"""
        assert self._process and self._process.stderr
        while True:
            line = await self._process.stderr.readline()
            if not line:
                break
            text = line.decode().rstrip()
            self._recent_stderr.append(text[:500])
            logger.debug("[mcp:%s] stderr: %s", self.name, text)

    def _build_timeout_message(
        self,
        stage: str,
        expected_id: int | None,
        timeout: float,
    ) -> str:
        details = [
            f"MCP server {self.name!r} 在阶段 {stage!r} 等待响应超时（{timeout:.0f}s）",
        ]
        if expected_id is not None:
            details.append(f"expected_id={expected_id}")
        if self.command:
            details.append(f"command={self.command!r}")
        if self.cwd:
            details.append(f"cwd={self.cwd}")
        if self._recent_stdout:
            details.append("recent_stdout=" + " | ".join(self._recent_stdout))
        if self._recent_stderr:
            details.append("recent_stderr=" + " | ".join(self._recent_stderr))
        return "; ".join(details)
