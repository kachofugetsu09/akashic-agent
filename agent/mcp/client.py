"""McpClient: 管理单个 MCP server 的 stdio 子进程连接和 JSON-RPC 通信。"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_RECV_TIMEOUT = 30.0


@dataclass
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


def _infer_cwd(command: list[str]) -> str | None:
    """从 command 中找第一个绝对路径文件，返回其父目录作为 cwd。"""
    for arg in command:
        p = Path(arg)
        if p.is_absolute() and p.is_file():
            return str(p.parent)
    return None


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
        self._tool_infos: list[McpToolInfo] = []

    @property
    def tool_infos(self) -> list[McpToolInfo]:
        return self._tool_infos

    async def connect(self) -> list[McpToolInfo]:
        """启动子进程，完成握手，获取工具列表。"""
        proc_env = {**os.environ, **self.env}
        logger.info("[mcp] 启动 %r: %s  cwd=%s", self.name, self.command, self.cwd)
        self._process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
            cwd=self.cwd,
        )
        asyncio.create_task(self._drain_stderr())

        # initialize 握手
        init_id = self._new_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "akasic-agent", "version": "1.0"},
            },
        })
        await self._recv(expected_id=init_id)

        # initialized 通知（无 id，不等响应）
        await self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

        # 获取工具列表
        list_id = self._new_id()
        await self._send({"jsonrpc": "2.0", "id": list_id, "method": "tools/list", "params": {}})
        resp = await self._recv(expected_id=list_id)

        raw_tools = resp.get("result", {}).get("tools", [])
        self._tool_infos = [
            McpToolInfo(
                name=t["name"],
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {"type": "object", "properties": {}}),
            )
            for t in raw_tools
        ]
        logger.info("[mcp] %r 已连接，工具：%s", self.name, [t.name for t in self._tool_infos])
        return self._tool_infos

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """调用远端工具，返回结果字符串。"""
        call_id = self._new_id()
        await self._send({
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        resp = await self._recv(expected_id=call_id)

        if "error" in resp:
            err = resp["error"]
            return f"MCP error ({self.name}/{tool_name}): {err.get('message', err)}"

        content = resp.get("result", {}).get("content", [])
        if isinstance(content, list):
            return "\n".join(
                block.get("text", str(block)) if isinstance(block, dict) else str(block)
                for block in content
            )
        return str(resp.get("result", ""))

    async def disconnect(self) -> None:
        """终止子进程。"""
        if self._process is None:
            return
        try:
            self._process.terminate()
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except Exception as e:
            logger.warning("[mcp] 断开 %r 时出错: %s", self.name, e)
        finally:
            self._process = None

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    async def _send(self, payload: dict[str, Any]) -> None:
        assert self._process and self._process.stdin
        self._process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
        await self._process.stdin.drain()

    async def _recv(self, expected_id: int | None = None) -> dict[str, Any]:
        assert self._process and self._process.stdout
        while True:
            line = await asyncio.wait_for(
                self._process.stdout.readline(), timeout=_RECV_TIMEOUT
            )
            if not line:
                raise ConnectionError(f"MCP server {self.name!r} 意外关闭了 stdout")
            text = line.decode().strip()
            if not text:
                continue
            try:
                msg = json.loads(text)
            except json.JSONDecodeError:
                logger.debug("[mcp:%s] 非 JSON 输出: %s", self.name, text[:200])
                continue
            # 跳过通知（有 method 但无 id）
            if "method" in msg and "id" not in msg:
                continue
            if expected_id is not None and msg.get("id") != expected_id:
                continue
            return msg

    async def _drain_stderr(self) -> None:
        """后台读取 stderr，防止缓冲区阻塞。"""
        assert self._process and self._process.stderr
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                logger.debug("[mcp:%s] stderr: %s", self.name, line.decode().rstrip())
        except Exception:
            pass
