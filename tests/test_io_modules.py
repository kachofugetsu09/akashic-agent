from __future__ import annotations

import asyncio
import base64
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from typing import cast

import pytest
import agent.mcp.client as mcp_client_module
import agent.tools.filesystem as filesystem_module

from agent.mcp.client import McpClient, McpToolExecutionError, _infer_cwd
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


class _Pipe:
    def __init__(self, lines: list[bytes] | None = None) -> None:
        self._lines = list(lines or [])
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

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
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

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
    b_file = base / "b.txt"
    b_file.chmod(0o751)
    result = await writer.execute("b.txt", "\ufeffhello\r\n")
    assert "已写入" in result
    assert b_file.read_bytes() == "\ufeffhello\r\n".encode("utf-8")
    assert stat.S_IMODE(b_file.stat().st_mode) == 0o751

    editor = EditFileTool(base)
    assert "未找到 old_text" in await editor.execute("b.txt", "x", "y")
    assert "不是文件" in await editor.execute(".", "x", "y")
    result = await editor.execute("b.txt", "hello", "world")
    assert "已成功编辑" in result
    assert "替换 1 处" in result, "edit_file 应在结果中报告替换数量"
    assert "```diff" in result
    assert "--- b.txt (before)" in result
    assert "+++ b.txt (after)" in result
    assert "-hello" in result
    assert "+world" in result
    assert b_file.read_bytes() == "\ufeffworld\r\n".encode("utf-8")
    assert stat.S_IMODE(b_file.stat().st_mode) == 0o751
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
async def test_file_mutation_lock_releases_after_failure_and_cancellation(
    tmp_path: Path,
):
    _FILE_MUTATION_LOCKS.clear()
    path = tmp_path / "shared.txt"

    async def fail() -> None:
        raise AssertionError("callback failed")

    with pytest.raises(AssertionError, match="callback failed"):
        await _run_with_file_mutation_lock(path, fail)
    assert not _FILE_MUTATION_LOCKS

    entered = asyncio.Event()

    async def wait_forever() -> None:
        entered.set()
        await asyncio.Event().wait()

    task = asyncio.create_task(_run_with_file_mutation_lock(path, wait_forever))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not _FILE_MUTATION_LOCKS


@pytest.mark.asyncio
async def test_file_mutation_lock_keeps_waiter_key_until_waiter_acquires(
    tmp_path: Path,
):
    _FILE_MUTATION_LOCKS.clear()
    path = tmp_path / "shared.txt"
    first_started = asyncio.Event()
    first_release = asyncio.Event()
    second_started = asyncio.Event()
    second_release = asyncio.Event()
    third_started = asyncio.Event()

    async def first() -> None:
        first_started.set()
        await first_release.wait()

    async def second() -> None:
        second_started.set()
        await second_release.wait()

    async def third() -> None:
        third_started.set()

    first_task = asyncio.create_task(_run_with_file_mutation_lock(path, first))
    await first_started.wait()
    second_task = asyncio.create_task(_run_with_file_mutation_lock(path, second))
    await asyncio.sleep(0)
    first_release.set()
    third_task: asyncio.Task[None] | None = None
    try:
        await first_task
        await second_started.wait()
        third_task = asyncio.create_task(_run_with_file_mutation_lock(path, third))
        await asyncio.sleep(0)
        assert not third_started.is_set()
    finally:
        second_release.set()
        await second_task
        if third_task is not None:
            await third_task
    assert not _FILE_MUTATION_LOCKS


@pytest.mark.asyncio
async def test_filesystem_tools_propagate_internal_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    base = tmp_path / "base"
    base.mkdir()
    file_path = base / "file.txt"
    file_path.write_text("content\n", encoding="utf-8")

    def fail_scan(*args: object) -> None:
        raise AssertionError("scan programming error")

    monkeypatch.setattr(filesystem_module, "_scan_text_file", fail_scan)
    with pytest.raises(AssertionError, match="scan programming error"):
        await ReadFileTool(base).execute("file.txt")

    def fail_atomic(*args: object, **kwargs: object) -> None:
        raise AssertionError("write programming error")

    monkeypatch.setattr(filesystem_module, "atomic_write_text", fail_atomic)
    with pytest.raises(AssertionError, match="write programming error"):
        await WriteFileTool(base).execute("new.txt", "content")
    with pytest.raises(AssertionError, match="write programming error"):
        await EditFileTool(base).execute("file.txt", "content", "updated")

    def fail_iterdir(*args: object, **kwargs: object) -> None:
        raise AssertionError("list programming error")

    monkeypatch.setattr(Path, "iterdir", fail_iterdir)
    with pytest.raises(AssertionError, match="list programming error"):
        await ListDirTool(base).execute(".")


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
            b'{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25"}}\n',
            b'{"jsonrpc":"2.0","method":"note"}\n',
            b'{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"tool1","description":"desc","inputSchema":{"type":"object"}}]}}\n',
            b'not json\n',
            b'{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text","text":"ok"}]}}\n',
        ],
        [b"warn\n", b""],
    )
    monkeypatch.setattr("agent.mcp.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    client = McpClient("docs", ["python", str(script)], env={"X": "1"})
    infos = await client.connect()
    assert infos[0].name == "tool1"
    initialize = json.loads(proc.stdin.writes[0])
    assert initialize["params"]["protocolVersion"] == "2025-11-25"
    assert await client.call("tool1", {"q": "x"}) == "ok"
    await client.disconnect()
    assert proc.stdin.closed is True
    assert proc.terminated is False

    proc = _Proc([b""])
    monkeypatch.setattr("agent.mcp.client.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
    client = McpClient("docs", ["python", str(script)])
    client._process = proc
    with pytest.raises(ConnectionError):
        await client._recv(expected_id=1)


@pytest.mark.asyncio
async def test_mcp_send_serializes_once_before_logging_and_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TrackingPipe(_Pipe):
        def __init__(self) -> None:
            super().__init__()
            self.events: list[tuple[str, bytes | None]] = []

        def write(self, data: bytes) -> None:
            self.events.append(("write", data))
            super().write(data)

        async def drain(self) -> None:
            self.events.append(("drain", None))
            await super().drain()

    proc = _Proc([])
    proc.stdin = TrackingPipe()
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc
    payload = {
        "jsonrpc": "2.0",
        "method": "测试",
        "params": {"text": "🙂中文" * 200},
    }
    original_dumps = mcp_client_module.json.dumps
    dump_calls: list[dict[str, object]] = []

    def counting_dumps(value, *args, **kwargs):
        if value is payload:
            dump_calls.append(kwargs)
        return original_dumps(value, *args, **kwargs)

    debug = MagicMock()
    monkeypatch.setattr(mcp_client_module.json, "dumps", counting_dumps)
    monkeypatch.setattr(mcp_client_module.logger, "debug", debug)

    await client._send(payload)

    serialized = original_dumps(payload, ensure_ascii=False)
    assert dump_calls == [{"ensure_ascii": False}]
    assert proc.stdin.writes == [(serialized + "\n").encode()]
    assert proc.stdin.events == [("write", proc.stdin.writes[0]), ("drain", None)]
    debug.assert_called_once_with("[mcp:%s] -> %s", "docs", serialized[:400])

    unsupported = _Proc([])
    unsupported_client = McpClient("docs", ["python", "server.py"])
    unsupported_client._process = unsupported
    debug.reset_mock()
    with pytest.raises(TypeError):
        await unsupported_client._send({"unsupported": object()})
    assert unsupported.stdin.writes == []
    debug.assert_not_called()


@pytest.mark.asyncio
async def test_mcp_client_rejects_unsupported_protocol_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proc = _Proc(
        [b'{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2099-01-01"}}\n']
    )
    monkeypatch.setattr(
        "agent.mcp.client.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    client = McpClient("future", ["python", "server.py"])

    with pytest.raises(RuntimeError, match="不支持的协议版本"):
        await client.connect()

    assert client.connected is False
    assert client._protocol_version is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "invalid_field"),
    [
        ({"name": "tool"}, "inputSchema"),
        ({"name": "tool", "inputSchema": {"type": "string"}}, "inputSchema"),
        (
            {"name": "tool", "inputSchema": {"type": "object"}, "description": 1},
            "description",
        ),
    ],
)
async def test_mcp_client_rejects_invalid_tool_schema(
    monkeypatch: pytest.MonkeyPatch,
    tool: dict[str, object],
    invalid_field: str,
) -> None:
    proc = _Proc(
        [
            b'{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25"}}\n',
            (
                '{"jsonrpc":"2.0","id":2,"result":{"tools":['
                + json.dumps(tool)
                + "]}}\n"
            ).encode(),
        ]
    )
    monkeypatch.setattr(
        "agent.mcp.client.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    client = McpClient("broken", ["python", "server.py"])

    with pytest.raises(RuntimeError, match=invalid_field):
        await client.connect()


@pytest.mark.asyncio
async def test_mcp_client_disconnect_escalates_after_graceful_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    proc = _Proc([])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc
    wait_count = 0

    async def wait_for(awaitable, *args, **kwargs):
        nonlocal wait_count
        wait_count += 1
        if wait_count == 1:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(mcp_client_module.asyncio, "wait_for", wait_for)

    await client.disconnect()

    assert proc.stdin.closed is True
    assert proc.terminated is True
    assert proc.killed is False


@pytest.mark.asyncio
async def test_mcp_client_disconnect_kills_after_terminate_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    proc = _Proc([])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc
    wait_count = 0

    async def wait_for(awaitable, *args, **kwargs):
        nonlocal wait_count
        wait_count += 1
        if wait_count <= 2:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(mcp_client_module.asyncio, "wait_for", wait_for)

    await client.disconnect()

    assert proc.stdin.closed is True
    assert proc.terminated is True
    assert proc.killed is True


@pytest.mark.asyncio
async def test_mcp_client_disconnect_reports_cleanup_error() -> None:
    proc = _Proc([])
    proc.stdin.close = MagicMock(side_effect=OSError("stdin close failed"))
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc

    with pytest.raises(OSError, match="stdin close failed"):
        await client.disconnect()

    assert proc.killed is True
    assert client._process is None


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["initialize", "tools/list"])
async def test_mcp_client_rejects_json_rpc_error_and_closes(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    responses = (
        [b'{"jsonrpc":"2.0","id":1,"error":{"code":-1,"message":"bad init"}}\n']
        if stage == "initialize"
        else [
            b'{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25"}}\n',
            b'{"jsonrpc":"2.0","id":2,"error":{"code":-1,"message":"bad list"}}\n',
        ]
    )
    proc = _Proc(responses)
    monkeypatch.setattr(
        "agent.mcp.client.asyncio.create_subprocess_exec",
        AsyncMock(return_value=proc),
    )
    client = McpClient("broken", ["python", "server.py"])

    with pytest.raises(RuntimeError, match=stage):
        await client.connect()

    assert client._process is None
    assert client._stderr_task is None
    assert proc.stdin.closed is True


@pytest.mark.asyncio
async def test_mcp_client_serializes_calls_on_same_server():
    class ConcurrentReadPipe(_Pipe):
        def __init__(self) -> None:
            super().__init__(
                [
                    b'{"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"a"}]}}\n',
                    b'{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"b"}]}}\n',
                ]
            )
            self.reading = False

        async def readline(self) -> bytes:
            if self.reading:
                raise RuntimeError("concurrent stdout read")
            self.reading = True
            try:
                await asyncio.sleep(0)
                return await super().readline()
            finally:
                self.reading = False

    proc = _Proc([])
    proc.stdout = ConcurrentReadPipe()
    client = McpClient("fitbit", ["python", "server.py"])
    client._process = proc

    results = await asyncio.gather(
        client.call("get_proactive_events", {}),
        client.call("get_sleep_context", {}),
    )

    assert results == ["a", "b"]


@pytest.mark.asyncio
async def test_mcp_call_raises_for_json_rpc_error_object() -> None:
    error = {"code": -1, "message": "bad call"}
    response = json.dumps({"jsonrpc": "2.0", "id": 1, "error": error}).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc

    with pytest.raises(McpToolExecutionError) as exc_info:
        await client.call("search", {})

    message = str(exc_info.value)
    assert "JSON-RPC error" in message
    assert "docs" in message
    assert "tools/call:search" in message
    assert "bad call" in message


@pytest.mark.asyncio
@pytest.mark.parametrize("error", ["server unavailable", ["invalid", "error"]])
async def test_mcp_call_rejects_non_object_error(error: object) -> None:
    import json

    response = json.dumps({"jsonrpc": "2.0", "id": 1, "error": error}).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc

    with pytest.raises(RuntimeError) as exc_info:
        await client.call("search", {})

    message = str(exc_info.value)
    assert "docs" in message
    assert "tools/call:search" in message
    assert type(error).__name__ in message
    assert repr(error) in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "invalid_path"),
    [
        ("plain text", "result"),
        ({}, "content"),
        ({"content": "plain text"}, "content"),
        ({"content": ["plain text"]}, "content[0]"),
        ({"content": [{}]}, "content[0].type"),
        ({"content": [{"type": "unknown"}]}, "content[0].type"),
        ({"content": [{"type": "audio"}]}, "content[0].type"),
        ({"content": [{"type": "resource_link"}]}, "content[0].type"),
        ({"content": [{"type": "text"}]}, "content[0].text"),
        ({"content": [{"type": "image", "mimeType": "image/png"}]}, "content[0].data"),
        ({"content": [{"type": "image", "data": "AAAA"}]}, "content[0].mimeType"),
        ({"content": [{"type": "resource"}]}, "content[0].resource"),
        (
            {"content": [{"type": "resource", "resource": {"text": "body"}}]},
            "content[0].resource.uri",
        ),
        (
            {"content": [{"type": "resource", "resource": {"uri": "r"}}]},
            "content[0].resource（需要 text 或 blob）",
        ),
        ({"content": [{"type": "text", "text": 123}]}, "content[0].text"),
        ({"content": [], "structuredContent": {}}, "structuredContent"),
        ({"content": [], "isError": "true"}, "isError"),
    ],
)
async def test_mcp_call_rejects_invalid_result_structure(
    result: object,
    invalid_path: str,
) -> None:
    response = json.dumps({"jsonrpc": "2.0", "id": 1, "result": result}).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc

    with pytest.raises(RuntimeError) as exc_info:
        await client.call("search", {})

    message = str(exc_info.value)
    assert "docs" in message
    assert "tools/call:search" in message
    assert invalid_path in message


@pytest.mark.asyncio
async def test_mcp_call_renders_valid_content_blocks() -> None:
    response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {"type": "text", "text": "ok"},
                    {"type": "image", "data": "AAAA", "mimeType": "image/png"},
                    {
                        "type": "resource",
                        "resource": {
                            "uri": "resource://report",
                            "mimeType": "text/plain",
                            "text": "report",
                        },
                    },
                ],
                "isError": False,
            },
        }
    ).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc

    result = await client.call("search", {})

    assert result.startswith("ok\n")
    assert '"type": "image"' in result
    assert '"uri": "resource://report"' in result


@pytest.mark.asyncio
async def test_mcp_call_accepts_structured_content_for_negotiated_protocol() -> None:
    response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "ok"}],
                "structuredContent": {"result": "ok"},
                "isError": False,
            },
        }
    ).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc
    client._protocol_version = "2025-11-25"

    assert await client.call("search", {}) == "ok"


@pytest.mark.asyncio
async def test_mcp_call_rejects_non_object_structured_content() -> None:
    response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [],
                "structuredContent": [],
            },
        }
    ).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc
    client._protocol_version = "2025-11-25"

    with pytest.raises(RuntimeError, match="structuredContent（需要 object）"):
        await client.call("search", {})


@pytest.mark.asyncio
async def test_mcp_call_raises_for_remote_tool_error() -> None:
    response = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "服务端失败：限流"}],
                "isError": True,
            },
        },
        ensure_ascii=False,
    ).encode()
    proc = _Proc([response + b"\n"])
    client = McpClient("docs", ["python", "server.py"])
    client._process = proc

    with pytest.raises(McpToolExecutionError) as exc_info:
        await client.call("search", {})

    message = str(exc_info.value)
    assert "docs" in message
    assert "tools/call:search" in message
    assert "服务端失败：限流" in message


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
