from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.mcp.client import McpClient, _infer_cwd
from agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _read_xls,
    _read_xlsx,
    _resolve_path,
)
from bus.events import OutboundMessage
from bus.queue import MessageBus
from channels.ipc_server import IPCServerChannel


class _Pipe:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _Proc:
    def __init__(self, stdout_lines: list[bytes], stderr_lines: list[bytes] | None = None) -> None:
        self.stdin = _Pipe()
        self.stdout = _Pipe(stdout_lines)
        self.stderr = _Pipe(stderr_lines)
        self.terminated = False

    def terminate(self) -> None:
        self.terminated = True

    async def wait(self) -> None:
        return None


@pytest.mark.asyncio
async def test_filesystem_tools_cover_core_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    text_file = base / "a.txt"
    text_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    assert _resolve_path("a.txt", base) == text_file.resolve()
    with pytest.raises(PermissionError):
        _resolve_path("../x", base)

    monkeypatch.setitem(
        __import__("sys").modules,
        "openpyxl",
        SimpleNamespace(
            load_workbook=lambda *args, **kwargs: type(
                "_WB",
                (),
                {
                    "sheetnames": ["S1"],
                    "__getitem__": lambda self, name: SimpleNamespace(
                        iter_rows=lambda values_only=True: [(1, 2), (None, None)]
                    ),
                },
            )()
        ),
    )
    monkeypatch.setitem(
        __import__("sys").modules,
        "xlrd",
        SimpleNamespace(
            open_workbook=lambda *args, **kwargs: SimpleNamespace(
                sheets=lambda: [SimpleNamespace(name="S1", nrows=1, ncols=2, cell_value=lambda r, c: f"{r}-{c}")]
            )
        ),
    )
    assert "Sheet" in _read_xlsx(base / "a.xlsx")
    assert "0-0" in _read_xls(base / "a.xls")

    reader = ReadFileTool(base)
    content = await reader.execute("a.txt", offset=1, limit=1)
    assert "line2" in content
    assert "第 2" in content
    assert "不存在" in await reader.execute("missing.txt")
    assert "不是文件" in await reader.execute(".")

    sop = MagicMock()
    sop.is_sop_file.return_value = True
    sop.reindex = AsyncMock(return_value="ok")
    writer = WriteFileTool(base, sop)
    result = await writer.execute("b.txt", "hello")
    assert "已写入" in result

    editor = EditFileTool(base, sop)
    assert "未找到 old_text" in await editor.execute("b.txt", "x", "y")
    assert "已成功编辑" in await editor.execute("b.txt", "hello", "world")
    dup = base / "dup.txt"
    dup.write_text("x\nx\n", encoding="utf-8")
    assert "出现了 2 次" in await editor.execute("dup.txt", "x", "y")

    lister = ListDirTool(base)
    assert "📄 a.txt" in await lister.execute(".")
    empty = base / "empty"
    empty.mkdir()
    assert "为空" in await lister.execute("empty")
    assert "不是目录" in await lister.execute("a.txt")


@pytest.mark.asyncio
async def test_ipc_server_channel_covers_connection_command_and_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bus = MessageBus()
    loop = SimpleNamespace()
    channel = IPCServerChannel(bus, str(tmp_path / "agent.sock"), loop)

    server = SimpleNamespace(close=MagicMock(), wait_closed=AsyncMock())
    monkeypatch.setattr("channels.ipc_server.asyncio.start_unix_server", AsyncMock(return_value=server))
    chmod = MagicMock()
    monkeypatch.setattr("channels.ipc_server.os.chmod", chmod)
    await channel.start()
    chmod.assert_called_once()
    await channel.stop()
    server.close.assert_called_once()

    reader = SimpleNamespace(
        readline=AsyncMock(
            side_effect=[
                b'{"content":"hello"}\n',
                b'{"type":"command","command":"noop"}\n',
                b'{"type":"command","command":"unknown"}\n',
                b'not json\n',
                b"",
            ]
        )
    )
    writes: list[bytes] = []
    writer = SimpleNamespace(
        get_extra_info=lambda name: "peer",
        write=lambda data: writes.append(data),
        drain=AsyncMock(),
        close=MagicMock(),
        is_closing=lambda: False,
    )
    await channel._handle_connection(reader, writer)
    inbound = await bus.consume_inbound()
    assert inbound.content == "hello"
    assert any("command_result" in payload.decode() for payload in writes)

    assert any("未知命令" in payload.decode() for payload in writes)

    msg = OutboundMessage(channel="cli", chat_id="missing", content="hi")
    await channel._on_response(msg)
    chat_id = next(iter(channel._writers.keys()), None)
    if chat_id:
        await channel._on_response(OutboundMessage(channel="cli", chat_id=chat_id, content="hi"))


@pytest.mark.asyncio
async def test_mcp_client_and_loop_factory_cover_core_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    script = tmp_path / "server.py"
    script.write_text("print(1)", encoding="utf-8")
    assert _infer_cwd(["python", str(script)]) == str(tmp_path)
    assert _infer_cwd(["python", "srv.py"]) is None

    proc = _Proc(
        [
            b'{"jsonrpc":"2.0","id":1,"result":{}}\n',
            b'{"jsonrpc":"2.0","method":"note"}\n',
            b'{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"tool1","description":"desc","inputSchema":{"type":"object"}}]}}\n',
            b'not json\n',
            b'{"jsonrpc":"2.0","id":3,"result":{"content":[{"text":"ok"}]}}\n',
        ],
        [b"warn\n", b""],
    )
    monkeypatch.setattr("agent.mcp.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    client = McpClient("docs", ["python", str(script)], env={"X": "1"})
    infos = await client.connect()
    assert infos[0].name == "tool1"
    assert proc.stdin.writes
    assert await client.call("tool1", {"q": "x"}) == "ok"
    await client.disconnect()
    assert proc.terminated is True

    proc = _Proc([b""])
    monkeypatch.setattr("agent.mcp.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    client = McpClient("docs", ["python", str(script)])
    client._process = proc
    with pytest.raises(ConnectionError):
        await client._recv(expected_id=1)
