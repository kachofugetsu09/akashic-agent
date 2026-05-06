from __future__ import annotations

import asyncio
import base64
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from typing import cast

import pytest
import agent.mcp.client as mcp_client_module

from agent.mcp.client import McpClient, _infer_cwd
from agent.tool_runtime import append_tool_result
from agent.tools.base import ToolResult
from agent.tools.filesystem import (
    EditFileTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
    _IMAGE_TARGET_B64_LEN,
    _READ_MAX_BYTES,
    _READ_MAX_LINES,
    _FILE_MUTATION_LOCKS,
    _resolve_path,
    _run_with_file_mutation_lock,
)
from agent.tools.vision import _encode_image_data_uri
from bus.events import OutboundMessage
from bus.queue import MessageBus
from infra.channels.ipc_server import IPCServerChannel


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


def _as_text(value: str | ToolResult) -> str:
    if isinstance(value, ToolResult):
        return value.text
    return value


@pytest.mark.asyncio
async def test_filesystem_tools_cover_core_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    base = tmp_path / "base"
    base.mkdir()
    text_file = base / "a.txt"
    text_file.write_text("line1\nline2\nline3\n", encoding="utf-8")

    assert _resolve_path("a.txt", base) == text_file.resolve()
    with pytest.raises(PermissionError):
        _resolve_path("../x", base)

    reader = ReadFileTool(base)
    content = await reader.execute("a.txt", offset=1, limit=1)
    assert "line2" in _as_text(content)
    assert "第 2" in _as_text(content)
    assert "不存在" in _as_text(await reader.execute("missing.txt"))
    assert "不是文件" in _as_text(await reader.execute("."))

    image = base / "a.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    image_result = await reader.execute("a.png")
    assert isinstance(image_result, ToolResult)
    assert "已读取图片文件" in image_result.text
    assert image_result.content_blocks[0]["type"] == "image_url"
    assert image_result.content_blocks[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    weird_image = base / "image.bin"
    weird_image.write_bytes(b"\x89PNG\r\n\x1a\nrest")
    weird_image_result = await reader.execute("image.bin")
    assert isinstance(weird_image_result, ToolResult)
    assert weird_image_result.content_blocks[0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )

    fake_image = base / "fake.png"
    fake_image.write_text("secret text", encoding="utf-8")
    fake_image_result = await reader.execute("fake.png")
    assert isinstance(fake_image_result, str)
    assert "secret text" in fake_image_result

    svg = base / "icon.svg"
    svg.write_text("<svg><rect width='10' height='10'/></svg>\n", encoding="utf-8")
    svg_result = await reader.execute("icon.svg")
    assert isinstance(svg_result, str)
    assert "<svg>" in svg_result

    from PIL import Image

    big = base / "big.png"
    noisy = Image.effect_noise((4000, 3000), 100).convert("RGB")
    noisy.save(big, format="PNG")
    big_result = await reader.execute("big.png")
    assert isinstance(big_result, ToolResult)
    assert "已自动压缩" in big_result.text
    big_url = big_result.content_blocks[0]["image_url"]["url"]
    assert big_url.startswith("data:image/jpeg;base64,")
    assert len(big_url.split(",", 1)[1]) <= _IMAGE_TARGET_B64_LEN

    # 验证行号前缀格式（改动九）
    full_content = await reader.execute("a.txt")
    full_content = _as_text(full_content)
    assert "     1\u2192line1" in full_content, "read_file 应输出 '     1→line1' 格式的行号前缀"
    assert "     2\u2192line2" in full_content
    assert "     3\u2192line3" in full_content

    # 验证字节截断后提示语包含 limit 分页引导
    from agent.tools import filesystem as _fs_mod
    orig_max_bytes = _fs_mod._READ_MAX_BYTES
    _fs_mod._READ_MAX_BYTES = 25  # 强制触发普通字节截断，但不触发首行超长分支
    truncated = await reader.execute("a.txt")
    _fs_mod._READ_MAX_BYTES = orig_max_bytes
    truncated = _as_text(truncated)
    assert "limit=N" in truncated, "截断提示应引导用户用 limit=N 分页，而非 offset 续读"
    assert "字节数超限" in truncated
    assert "本次返回" in truncated
    assert "字节" in truncated
    assert "offset=0 limit=100" in truncated

    orig_max_lines = _fs_mod._READ_MAX_LINES
    _fs_mod._READ_MAX_LINES = 2
    truncated_lines = await reader.execute("a.txt")
    _fs_mod._READ_MAX_LINES = orig_max_lines
    truncated_lines = _as_text(truncated_lines)
    assert "行数超限" in truncated_lines
    assert "本次返回" in truncated_lines

    long_line = base / "long_line.txt"
    long_line.write_text("x" * (_READ_MAX_BYTES + 1), encoding="utf-8")
    long_line_result = await reader.execute("long_line.txt")
    long_line_result = _as_text(long_line_result)
    assert "首行超过 10KB" in long_line_result

    boundary = base / "boundary.txt"
    boundary.write_text("x" * (_READ_MAX_BYTES - 1), encoding="utf-8")
    boundary_result = await reader.execute("boundary.txt")
    boundary_result = _as_text(boundary_result)
    assert "首行超过 10KB" not in boundary_result
    assert "字节数超限" in boundary_result

    bad_utf8 = base / "bad.txt"
    bad_utf8.write_bytes(b"ok\xffoops\n")
    bad_utf8_result = await reader.execute("bad.txt")
    bad_utf8_result = _as_text(bad_utf8_result)
    assert "替代字符" in bad_utf8_result
    assert "oops" in bad_utf8_result

    binary = base / "data.dat"
    binary.write_bytes(b"\x00\x01\x02\x03hello")
    binary_result = await reader.execute("data.dat")
    binary_result = _as_text(binary_result)
    assert "二进制文件" in binary_result
    assert "xxd" in binary_result

    text_no_read_bytes = base / "stream.txt"
    text_no_read_bytes.write_text("alpha\nbeta\n", encoding="utf-8")
    orig_read_bytes = Path.read_bytes

    def _guard_read_bytes(self: Path):
        if self == text_no_read_bytes:
            raise AssertionError("text path should stream via open(), not Path.read_bytes()")
        return orig_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", _guard_read_bytes)
    streamed = await reader.execute("stream.txt")
    assert "alpha" in _as_text(streamed)
    monkeypatch.setattr(Path, "read_bytes", orig_read_bytes)

    writer = WriteFileTool(base)
    result = await writer.execute("b.txt", "hello")
    assert "已写入" in result

    editor = EditFileTool(base)
    assert "未找到 old_text" in await editor.execute("b.txt", "x", "y")
    result = await editor.execute("b.txt", "hello", "world")
    assert "已成功编辑" in result
    assert "替换 1 处" in result, "edit_file 应在结果中报告替换数量"
    assert "```diff" in result
    assert "--- b.txt (before)" in result
    assert "+++ b.txt (after)" in result
    assert "-hello" in result
    assert "+world" in result
    assert text_file.read_text(encoding="utf-8") == "line1\nline2\nline3\n"

    dup = base / "dup.txt"
    dup.write_text("x\nx\n", encoding="utf-8")
    assert "出现了 2 次" in await editor.execute("dup.txt", "x", "y")

    # 验证 replace_all=True（改动十）
    dup.write_text("x\nx\n", encoding="utf-8")
    result_all = await editor.execute("dup.txt", "x", "z", replace_all=True)
    assert "替换 2 处" in result_all, "replace_all=true 应替换所有匹配并报告数量"
    assert dup.read_text(encoding="utf-8") == "z\nz\n"

    crlf = base / "crlf.txt"
    crlf.write_bytes(b"hello\r\nworld\r\n")
    result_crlf = await editor.execute("crlf.txt", "hello\nworld\n", "hi\nworld\n")
    assert "已成功编辑" in result_crlf
    assert "-hello" in result_crlf
    assert "+hi" in result_crlf
    assert crlf.read_bytes() == b"hi\r\nworld\r\n"

    bom = base / "bom.txt"
    bom.write_bytes("\ufeffhello\r\n".encode("utf-8"))
    result_bom = await editor.execute("bom.txt", "hello\n", "world\n")
    assert "已成功编辑" in result_bom
    assert bom.read_bytes() == "\ufeffworld\r\n".encode("utf-8")

    mixed = base / "mixed.txt"
    mixed.write_bytes(b"left\r\nright\nleft\nright\n")
    result_mixed = await editor.execute("mixed.txt", "left\nright\n", "x\ny\n")
    assert "已成功编辑" in result_mixed
    assert "替换 1 处" in result_mixed
    assert mixed.read_bytes() == b"left\r\nright\nx\ny\n"

    lister = ListDirTool(base)
    assert "📄 a.txt" in await lister.execute(".")
    empty = base / "empty"
    empty.mkdir()
    assert "为空" in await lister.execute("empty")
    assert "不是目录" in await lister.execute("a.txt")


def test_vision_rejects_extension_only_image(tmp_path: Path):
    fake_image = tmp_path / "secret.png"
    fake_image.write_text("secret text", encoding="utf-8")

    with pytest.raises(ValueError, match="不支持的图片格式"):
        _encode_image_data_uri(fake_image)


def test_vision_rejects_forged_magic_bytes_image(tmp_path: Path):
    fake_image = tmp_path / "secret.png"
    fake_image.write_bytes(b"\x89PNG\r\n\x1a\nsecret text")

    with pytest.raises(ValueError, match="图片文件无法解码"):
        _encode_image_data_uri(fake_image)


def test_vision_reencodes_image_before_sending(tmp_path: Path):
    from PIL import Image

    image = tmp_path / "with_tail.png"
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image)
    image.write_bytes(image.read_bytes() + b"secret text")

    data_uri = _encode_image_data_uri(image)
    payload = data_uri.split(",", 1)[1]

    assert b"secret text" not in base64.b64decode(payload)


def test_vision_rejects_image_when_compression_still_exceeds_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    from PIL import Image
    from agent.tools import vision

    image = tmp_path / "large.png"
    Image.new("RGB", (32, 32), (255, 0, 0)).save(image)
    monkeypatch.setattr(vision, "_VL_MAX_DATA_URI_BYTES", 10)

    with pytest.raises(ValueError, match="压缩后仍然过大"):
        _encode_image_data_uri(image)


def test_append_tool_result_supports_multimodal_blocks() -> None:
    messages: list[dict] = []
    append_tool_result(
        messages,
        tool_call_id="call_1",
        tool_name="read_file",
        content=ToolResult(
            text="[已读取图片文件 a.png，图片内容已提供给多模态模型]",
            content_blocks=[
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
        ),
    )
    assert messages[0]["role"] == "tool"
    assert messages[0]["content"].startswith("[已读取图片文件")
    assert messages[1]["role"] == "user"
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][1]["type"] == "image_url"


@pytest.mark.asyncio
async def test_file_mutation_lock_serializes_same_file_and_allows_different_files(
    tmp_path: Path,
):
    _FILE_MUTATION_LOCKS.clear()
    shared = tmp_path / "shared.txt"
    other = tmp_path / "other.txt"
    order: list[str] = []

    async def _job(name: str, path: Path, delay: float) -> None:
        async def _run() -> None:
            order.append(f"{name}:start")
            await asyncio.sleep(delay)
            order.append(f"{name}:end")

        await _run_with_file_mutation_lock(path, _run)

    shared_a = asyncio.create_task(_job("shared_a", shared, 0.05))
    shared_b = asyncio.create_task(_job("shared_b", shared, 0.0))
    other_task = asyncio.create_task(_job("other", other, 0.0))
    await asyncio.gather(shared_a, shared_b, other_task)

    assert order.index("shared_a:end") < order.index("shared_b:start")
    assert order.index("other:start") < order.index("shared_a:end")
    assert not _FILE_MUTATION_LOCKS


@pytest.mark.asyncio
async def test_ipc_server_channel_covers_connection_command_and_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    bus = MessageBus()
    loop = SimpleNamespace()
    channel = IPCServerChannel(bus, str(tmp_path / "agent.sock"), None)

    server = SimpleNamespace(close=MagicMock(), wait_closed=AsyncMock())
    chmod = MagicMock()
    monkeypatch.setattr("infra.channels.ipc_server.os.chmod", chmod)
    if sys.platform == "win32":
        monkeypatch.setattr(
            "infra.channels.ipc_server.asyncio.start_server",
            AsyncMock(return_value=server),
        )
    else:
        monkeypatch.setattr(
            "infra.channels.ipc_server.asyncio.start_unix_server",
            AsyncMock(return_value=server),
        )
    await channel.start()
    if sys.platform == "win32":
        chmod.assert_not_called()
    else:
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
        wait_closed=AsyncMock(),
        is_closing=lambda: False,
    )
    await channel._handle_connection(
        cast(asyncio.StreamReader, reader), cast(asyncio.StreamWriter, writer)
    )
    inbound = await bus.consume_inbound()
    assert inbound.content == "hello"
    assert any("command_result" in payload.decode() for payload in writes)
    assert any('"ok": false' in payload.decode() for payload in writes)
    assert any("unknown command" in payload.decode() for payload in writes)

    msg = OutboundMessage(channel="cli", chat_id="missing", content="hi")
    await channel._on_response(msg)
    chat_id = next(iter(channel._writers.keys()), None)
    if chat_id:
        await channel._on_response(OutboundMessage(channel="cli", chat_id=chat_id, content="hi"))


@pytest.mark.asyncio
async def test_ipc_server_uses_tcp_for_explicit_host_port_on_all_platforms(
    monkeypatch: pytest.MonkeyPatch,
):
    bus = MessageBus()
    channel = IPCServerChannel(bus, "127.0.0.1:8765", None)

    server = SimpleNamespace(close=MagicMock(), wait_closed=AsyncMock())
    start_server = AsyncMock(return_value=server)
    start_unix_server = AsyncMock(side_effect=AssertionError("explicit TCP endpoint should not use unix sockets"))
    chmod = MagicMock()

    monkeypatch.setattr("infra.channels.ipc_server.asyncio.start_server", start_server)
    if hasattr(asyncio, "start_unix_server"):
        monkeypatch.setattr("infra.channels.ipc_server.asyncio.start_unix_server", start_unix_server)
    monkeypatch.setattr("infra.channels.ipc_server.os.chmod", chmod)

    await channel.start()
    start_server.assert_awaited_once()
    start_unix_server.assert_not_called()
    chmod.assert_not_called()
    await channel.stop()
    server.close.assert_called_once()


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


@pytest.mark.asyncio
async def test_mcp_recv_timeout_includes_stage_and_recent_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    script = tmp_path / "server.py"
    script.write_text("print(1)", encoding="utf-8")
    proc = _Proc([])
    client = McpClient("docs", ["python", str(script)])
    client._process = proc
    client._recent_stdout.append('{"jsonrpc":"2.0","method":"note"}')
    client._recent_stderr.append("GitHub MCP Server running on stdio")

    async def raise_timeout(awaitable, *args, **kwargs):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(mcp_client_module.asyncio, "wait_for", raise_timeout)
    with pytest.raises(TimeoutError) as exc:
        await client._recv(expected_id=1, stage="initialize", timeout=12.0)
    text = str(exc.value)
    assert "initialize" in text
    assert "12s" in text
    assert "expected_id=1" in text
    assert "recent_stderr=GitHub MCP Server running on stdio" in text
