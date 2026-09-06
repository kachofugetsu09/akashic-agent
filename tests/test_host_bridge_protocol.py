from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
import sys
from typing import Any, AsyncIterator

import grpc
import pytest

from agent.host_bridge import host_bridge_pb2 as pb
from agent.host_bridge import host_bridge_pb2_grpc as rpc
from agent.host_bridge.protocol import (
    CHANNEL_OPTIONS,
    decode_execution,
    encode_file_result,
)
from agent.host_bridge.server import HostBridgeService
from agent.tools.base import ToolResult
from agent.tools.unified_exec import ShellProcessManager

META = (("authorization", "Bearer test-token"),)


def request_context() -> pb.RequestContext:
    return pb.RequestContext(
        boot_id="test-boot",
        manager_id="test-manager",
        request_id="test-request",
        expected_release_commit="a" * 40,
        expected_toolchain_digest="b" * 64,
    )


def exec_request(**changes: Any) -> pb.ExecRequest:
    values: dict[str, Any] = dict(
        context=request_context(),
        command="test",
        argv=[sys.executable, "-c", "pass"],
        tty=False,
        yield_time_ms=1000,
        max_output_tokens=1000,
        hard_timeout_s=30,
        owner_session_key="test-session",
    )
    values.update(changes)
    return pb.ExecRequest(**values)


@asynccontextmanager
async def running_service(tmp_path: Path) -> AsyncIterator[rpc.HostBridgeStub]:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "main.py").write_text("# test fixture\n")
    service = HostBridgeService(
        "test-token",
        4,
        tmp_path / "artifacts",
        release_commit="a" * 40,
        toolchain_digest="b" * 64,
        runtime_checkout=checkout,
        bridge_python=Path(sys.executable),
    )
    server = grpc.aio.server(options=CHANNEL_OPTIONS)
    rpc.add_HostBridgeServicer_to_server(service, server)
    socket = tmp_path / "bridge.sock"
    assert server.add_insecure_port(f"unix:{socket}") == 1
    await server.start()
    try:
        async with grpc.aio.insecure_channel(
            f"unix:{socket}", options=CHANNEL_OPTIONS
        ) as channel:
            stub = rpc.HostBridgeStub(channel)
            await stub.ClaimBoot(
                pb.ContextRequest(context=request_context()), metadata=META
            )
            yield stub
    finally:
        await server.stop(0)
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [(), (("authorization", "wrong"),), META + META])
async def test_authentication_metadata_rejects_missing_wrong_or_duplicate_token(
    tmp_path: Path, metadata: tuple
) -> None:
    async with running_service(tmp_path) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.Inspect(
                pb.ContextRequest(context=request_context()), metadata=metadata
            )
        assert error.value.code() == grpc.StatusCode.PERMISSION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing",
    ["context", "tty", "yield_time_ms", "max_output_tokens", "hard_timeout_s"],
)
async def test_missing_exec_fields_cannot_spawn(tmp_path: Path, missing: str) -> None:
    async with running_service(tmp_path) as stub:
        request = exec_request()
        request.ClearField(missing)
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.Exec(request, metadata=META)
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        active = await stub.ActiveExecutions(
            pb.ContextRequest(context=request_context()), metadata=META
        )
        assert not active.execution_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "changes", [{"max_output_tokens": -1}, {"hard_timeout_s": 0}, {"argv": []}]
)
async def test_invalid_exec_fields_cannot_spawn(tmp_path: Path, changes: dict) -> None:
    async with running_service(tmp_path) as stub:
        with pytest.raises(grpc.aio.AioRpcError) as error:
            await stub.Exec(exec_request(**changes), metadata=META)
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        active = await stub.ActiveExecutions(
            pb.ContextRequest(context=request_context()), metadata=META
        )
        assert not active.execution_ids


@pytest.mark.asyncio
async def test_wire_preserves_empty_binary_output_and_exit_status(
    tmp_path: Path,
) -> None:
    async with running_service(tmp_path) as stub:
        empty = decode_execution(
            await stub.Exec(
                exec_request(max_output_tokens=0, yield_time_ms=0), metadata=META
            )
        )
        assert (
            empty.output == b"" and empty.exit_code == 0 and empty.execution_id is None
        )
        assert empty.output_path is None
        binary = decode_execution(
            await stub.Exec(
                exec_request(
                    argv=[
                        sys.executable,
                        "-c",
                        "import sys; sys.stdout.buffer.write(bytes(range(256))); sys.exit(7)",
                    ]
                ),
                metadata=META,
            )
        )
        assert binary.output == bytes(range(256)) and binary.exit_code == 7
        killed = decode_execution(
            await stub.Exec(
                exec_request(
                    argv=[
                        sys.executable,
                        "-c",
                        "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
                    ]
                ),
                metadata=META,
            )
        )
        assert killed.exit_code == -15


@pytest.mark.asyncio
async def test_cancelled_rpc_keeps_one_execution_for_stdin_and_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    admitted = asyncio.Event()
    cancelled = asyncio.Event()
    original = ShellProcessManager._collect_until_deadline
    calls = 0

    async def collect(manager, execution, deadline):
        nonlocal calls
        calls += 1
        if calls == 1:
            admitted.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return await original(manager, execution, deadline)

    monkeypatch.setattr(ShellProcessManager, "_collect_until_deadline", collect)
    async with running_service(tmp_path) as stub:
        call = stub.Exec(
            exec_request(
                argv=[
                    "/bin/bash",
                    "-c",
                    "read line; printf 'REPLY:%s\\n' \"$line\"; read done",
                ],
                tty=True,
            ),
            metadata=META,
        )
        await asyncio.wait_for(admitted.wait(), 5)
        assert call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        await asyncio.wait_for(cancelled.wait(), 5)
        active = await stub.ActiveExecutions(
            pb.ContextRequest(context=request_context()), metadata=META
        )
        assert len(active.execution_ids) == 1
        execution_id = active.execution_ids[0]
        reply = await stub.WriteStdin(
            pb.WriteStdinRequest(
                context=request_context(),
                execution_id=execution_id,
                chars="hello\n",
                yield_time_ms=250,
                max_output_tokens=1000,
                owner_session_key="test-session",
            ),
            metadata=META,
        )
        result = decode_execution(reply)
        assert result.execution_id == execution_id and b"REPLY:hello" in result.output
        stopped = await stub.Stop(
            pb.StopRequest(
                context=request_context(),
                execution_id=execution_id,
                owner_session_key="test-session",
            ),
            metadata=META,
        )
        assert stopped.stopped
        active = await stub.ActiveExecutions(
            pb.ContextRequest(context=request_context()), metadata=META
        )
        assert not active.execution_ids
        assert calls == 2


@pytest.mark.asyncio
async def test_file_wire_preserves_empty_values_defaults_and_business_errors(
    tmp_path: Path,
) -> None:
    async with running_service(tmp_path) as stub:
        path = str(tmp_path / "file.txt")
        common: dict[str, Any] = dict(
            context=request_context(), allowed_dir=str(tmp_path)
        )
        written = await stub.FileTool(
            pb.FileRequest(**common, write=pb.WriteFile(path=path, content="")),
            metadata=META,
        )
        assert written.WhichOneof("result") == "text"
        assert Path(path).read_bytes() == b""
        await stub.FileTool(
            pb.FileRequest(**common, write=pb.WriteFile(path=path, content="a\na\n")),
            metadata=META,
        )
        for replace_all in (None, False):
            edit = pb.EditFile(path=path, old_text="a", new_text="b")
            if replace_all is not None:
                edit.replace_all = replace_all
            await stub.FileTool(pb.FileRequest(**common, edit=edit), metadata=META)
            assert Path(path).read_text() == "a\na\n"
        await stub.FileTool(
            pb.FileRequest(
                **common,
                edit=pb.EditFile(
                    path=path, old_text="a", new_text="", replace_all=True
                ),
            ),
            metadata=META,
        )
        assert Path(path).read_text() == "\n\n"
        for limit in (None, 1):
            read = await stub.FileTool(
                pb.FileRequest(
                    **common, read=pb.ReadFile(path=path, offset=0, limit=limit)
                ),
                metadata=META,
            )
            assert read.WhichOneof("result") == "text"
        missing = await stub.FileTool(
            pb.FileRequest(**common, read=pb.ReadFile(path=path + ".missing")),
            metadata=META,
        )
        assert "不存在" in missing.text
        for request in (
            pb.FileRequest(**common),
            pb.FileRequest(**common, write=pb.WriteFile(path=path)),
            pb.FileRequest(**common, read=pb.ReadFile(path=path, limit=0)),
        ):
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await stub.FileTool(request, metadata=META)
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT


def test_malformed_execution_reply_is_not_a_success_default() -> None:
    for reply in (
        pb.ExecutionReply(),
        pb.ExecutionReply(
            output=b"",
            wall_time_ms=0,
            original_token_count=0,
            output_omitted_bytes=0,
            finish_reason="natural",
        ),
        pb.ExecutionReply(
            output=b"",
            wall_time_ms=0,
            original_token_count=0,
            output_omitted_bytes=0,
            execution_id=0,
            finish_reason="natural",
        ),
    ):
        with pytest.raises(ValueError):
            decode_execution(reply)


def test_unexpected_file_content_is_not_silently_dropped() -> None:
    for result in (
        ToolResult(content_blocks=[{"type": "text", "text": "unexpected"}]),
        ToolResult(content_blocks=[]),
        ToolResult(
            content_blocks=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image", "detail": "high"},
                }
            ]
        ),
    ):
        with pytest.raises(RuntimeError):
            encode_file_result(result)


@pytest.mark.asyncio
async def test_v1_endpoint_is_rejected_instead_of_falling_back(tmp_path: Path) -> None:
    async with running_service(tmp_path):
        async with grpc.aio.insecure_channel(
            f"unix:{tmp_path / 'bridge.sock'}", options=CHANNEL_OPTIONS
        ) as channel:
            old_call = channel.unary_unary("/akashic.host.v1.HostBridge/Inspect")
            with pytest.raises(grpc.aio.AioRpcError) as error:
                await old_call(b"", metadata=META)
            assert error.value.code() == grpc.StatusCode.UNIMPLEMENTED


@pytest.mark.asyncio
async def test_cancelled_stdin_rpc_does_not_replay_written_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = asyncio.Event()
    cancelled = asyncio.Event()
    original = ShellProcessManager._collect_until_deadline
    calls = 0

    async def collect(manager, execution, deadline):
        nonlocal calls
        calls += 1
        if calls == 1:
            # write_stdin 只在完成输入写入之后进入 collector。
            written.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
        return await original(manager, execution, deadline)

    async with running_service(tmp_path) as stub:
        started = await stub.Exec(
            exec_request(
                argv=[
                    "/bin/bash",
                    "-c",
                    'while read line; do [ "$line" = stop ] && break; printf \'SEEN:%s\\n\' "$line"; done',
                ],
                tty=True,
                yield_time_ms=250,
            ),
            metadata=META,
        )
        execution_id = decode_execution(started).execution_id
        assert execution_id is not None
        monkeypatch.setattr(ShellProcessManager, "_collect_until_deadline", collect)
        common: dict[str, Any] = dict(
            context=request_context(),
            execution_id=execution_id,
            yield_time_ms=250,
            max_output_tokens=1000,
            owner_session_key="test-session",
        )
        call = stub.WriteStdin(
            pb.WriteStdinRequest(**common, chars="once\n"), metadata=META
        )
        await asyncio.wait_for(written.wait(), 5)
        assert call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        await asyncio.wait_for(cancelled.wait(), 5)
        completed = decode_execution(
            await stub.WriteStdin(
                pb.WriteStdinRequest(**common, chars="stop\n"), metadata=META
            )
        )
        assert completed.exit_code == 0
        assert completed.output.count(b"SEEN:once") == 1
        assert calls == 2
        active = await stub.ActiveExecutions(
            pb.ContextRequest(context=request_context()), metadata=META
        )
        assert not active.execution_ids
