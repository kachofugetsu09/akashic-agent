"""Gateway 命令通过已发布端点使用 SDK；不启动 Core 或读取业务配置。"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import sys
from typing import cast
from uuid import uuid4

from akashic_sdk import AsyncAkashic, RemoteError
from agent.plugin_composition import load_endpoint_plan
from core.common.file_io import run_file_io
from .socket import is_tcp_endpoint
from .token import read_secret


def find_endpoint(workspace: Path) -> str:
    """只连接唯一已就绪的 JSON-RPC 端点；缺席或冲突明确失败。"""
    endpoints = tuple(endpoint for endpoint in load_endpoint_plan(workspace / "runtime/endpoints.json")
                      if endpoint.protocol in {"jsonrpc+unix", "jsonrpc+tcp"})
    if len(endpoints) != 1:
        raise ConnectionError(f"Gateway 需要唯一已发布端点，当前有 {len(endpoints)} 个")
    return endpoints[0].address


def read_token(workspace: Path, endpoint: str) -> str | None:
    """Unix 不用 token；TCP 边界只允许 loopback 并只读原 secret。"""
    return read_secret(workspace / ".app-server-token") if is_tcp_endpoint(endpoint) else None


# 只有终态 Output 或 Control 可能结束原 Input；其他追加不必回查结果。
def _may_end_input(event: dict[str, object]) -> bool:
    items = event.get("items")
    if not isinstance(items, list):
        return True
    for row in cast(list[object], items):
        body = cast(dict[str, object], row).get("body") if isinstance(row, dict) else None
        if not isinstance(body, dict):
            return True
        kind = cast(dict[str, object], body).get("kind")
        if kind == "control" or (kind == "output" and cast(dict[str, object], body).get("finish") != "continue"):
            return True
    return False


async def _wait_exec_result(client: AsyncAkashic, session_id: str, input_id: str,
                            *, json_events: bool) -> dict[str, object]:
    """从当前结果的 seq 继续跟随；订阅建立期间的新消息仍能补读。"""
    query: dict[str, object] = {"session_id": session_id, "input_id": input_id}
    result = cast(dict[str, object], await client.request("programmatic/message/result", query))
    if result["status"] != "open":
        return result
    async with await client.session_follow(session_id, after_seq=cast(int, result["through_seq"])) as feed:
        async for event in feed.events():
            if json_events:
                print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)
            if event["type"] == "messages.appended" and _may_end_input(event):
                result = cast(dict[str, object], await client.request("programmatic/message/result", query))
                if result["status"] != "open":
                    return result
    raise ConnectionError("消息订阅已关闭；使用原 Session 和 Input 身份恢复查询")


async def _exec_until_stop(client: AsyncAkashic, session_id: str, input_id: str,
                           *, json_events: bool) -> tuple[dict[str, object], bool]:
    """显式 SIGINT 提交 pause；普通连接关闭只停止本地读取。"""
    interrupt = asyncio.Event()
    loop = asyncio.get_running_loop()
    previous = signal.getsignal(signal.SIGINT)
    native_handler = False
    try:
        loop.add_signal_handler(signal.SIGINT, interrupt.set)
        native_handler = True
    except NotImplementedError:
        def on_sigint(_signal: int, _frame: object) -> None:
            _ = loop.call_soon_threadsafe(interrupt.set)
        _ = signal.signal(signal.SIGINT, on_sigint)
    result_task = asyncio.create_task(_wait_exec_result(client, session_id, input_id,
                                                       json_events=json_events), name="exec-result")
    interrupt_task = asyncio.create_task(interrupt.wait(), name="exec-sigint")
    stopped = False
    try:
        done, _ = await asyncio.wait((result_task, interrupt_task), return_when=asyncio.FIRST_COMPLETED)
        if interrupt_task in done and not result_task.done():
            stopped = True
            _ = await client.request("programmatic/message/pause", {
                "session_id": session_id, "message_id": uuid4().hex,
            })
        return await result_task, stopped
    finally:
        _ = result_task.cancel()
        _ = interrupt_task.cancel()
        _ = await asyncio.gather(result_task, interrupt_task, return_exceptions=True)
        if native_handler:
            _ = loop.remove_signal_handler(signal.SIGINT)
        _ = signal.signal(signal.SIGINT, previous)


async def run_exec(args: list[str], workspace: Path) -> int:
    """通过普通程序来源提交 Message，按原 Input 的持久结果退出。"""
    # 1. 每个可重试写入都有调用方身份；CLI 不接受来源或学习属性覆盖。
    parser = argparse.ArgumentParser(prog="exec")
    _ = parser.add_argument("prompt", nargs="?")
    _ = parser.add_argument("--new", action="store_true")
    _ = parser.add_argument("--session")
    _ = parser.add_argument("--message-id")
    _ = parser.add_argument("--resume")
    _ = parser.add_argument("--persist-memory", action="store_true")
    _ = parser.add_argument("--detach", action="store_true")
    output = parser.add_mutually_exclusive_group()
    _ = output.add_argument("--json", action="store_true")
    _ = output.add_argument("--final-only", action="store_true")
    _ = parser.add_argument("--endpoint")
    options = parser.parse_args(args[1:])
    if not options.new and options.session is None:
        raise ValueError("exec 需要 --new 或 --session ID")
    if options.persist_memory and not options.new:
        raise ValueError("--persist-memory 只能在 --new 准入时选择")
    if options.detach and options.final_only:
        raise ValueError("--detach 不能与 --final-only 一起使用")
    if options.resume is not None:
        if options.new or options.prompt is not None:
            raise ValueError("--resume 只引用原 Session 的 Input，不接收新 prompt")
    elif options.prompt is None:
        raise ValueError("exec 缺少 prompt；使用 - 从 stdin 读取")
    session_id = options.session or "programmatic:" + uuid4().hex
    message_id = options.message_id or uuid4().hex
    input_id = options.resume or message_id
    endpoint = options.endpoint
    if endpoint is None:
        endpoint = await run_file_io(lambda: find_endpoint(workspace))
    token = await run_file_io(lambda: read_token(workspace, endpoint))
    identity: dict[str, object] = {"session_id": session_id, "message_id": message_id, "input_id": input_id}
    print(json.dumps({"type": "message.submitting", **identity}, ensure_ascii=False),
          file=sys.stdout if options.json else sys.stderr, flush=True)

    # 2. 先固定 Session 属性，再提交输入；ACK 不等默认回复。
    async with await AsyncAkashic.connect(endpoint, workspace_token=token) as client:
        if options.new:
            _ = await client.request("programmatic/session/admit", {
                "session_id": session_id, "persist_memory": options.persist_memory,
            })
        if options.resume is not None:
            receipt = await client.request("programmatic/message/resume", identity)
        else:
            prompt = await run_file_io(sys.stdin.read) if options.prompt == "-" else options.prompt
            receipt = await client.request("programmatic/message/send", {
                "session_id": session_id, "message_id": message_id, "text": prompt,
            })
        if options.json:
            print(json.dumps({"type": "message.accepted", "receipt": receipt}, ensure_ascii=False), flush=True)
        if options.detach:
            return 0

        # 3. 完成、暂停、失败都来自日志；读取关闭不会伪造成功。
        result, stopped = await _exec_until_stop(client, session_id, input_id, json_events=options.json)
        if options.json:
            print(json.dumps({"type": "message.result", **result}, ensure_ascii=False), flush=True)
        elif result["status"] in {"complete", "quiet"}:
            ending = cast(int, result["ending_seq"])
            page = await client.message_read(session_id, after_seq=ending - 1, through_seq=ending, limit=1)
            row = page["items"][0]
            if row["id"] != result["ending_message_id"]:
                raise RuntimeError("程序结果引用与读取的 Message 不一致")
            print("\n".join(part["value"] for part in row["body"]["parts"] if part["kind"] == "text"))
        else:
            print(json.dumps(result, ensure_ascii=False), file=sys.stderr)
        if stopped or result["status"] == "pause":
            return 130
        return 0 if result["status"] in {"complete", "quiet"} else 1


async def request(workspace: Path, method: str, params: dict[str, object]) -> dict[str, object]:
    """发起一次管理操作；响应失败不提交本地替代选择。"""
    endpoint = await run_file_io(lambda: find_endpoint(workspace))
    token = await run_file_io(lambda: read_token(workspace, endpoint))
    async with await AsyncAkashic.connect(endpoint, workspace_token=token) as client:
        result = await client.request(method, params)
    if not isinstance(result, dict):
        raise RuntimeError(f"{method} 响应无效")
    return cast(dict[str, object], result)


async def exec_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    """提交程序输入；控制错误和连接失败返回命令失败。"""
    try:
        return await run_exec(["exec", *arguments], workspace)
    except (ValueError, ConnectionError, OSError, RemoteError) as error:
        print(str(error), file=sys.stderr)
        return 2


async def management(arguments: tuple[str, ...], *, workspace: Path, config_path: Path,
                     command: str) -> int:
    """命令只验证参数；安装、查询和卸载由运行实例提交。"""
    parser = argparse.ArgumentParser(prog=command)
    if command == "plugin-install":
        parser.add_argument("--source", required=True)
        parser.add_argument("--marketplace", default="local")
        parser.add_argument("--ref", default="")
        parser.add_argument("--sparse", default="")
        parser.add_argument("--update-id", default=uuid4().hex)
    else:
        parser.add_argument("identity", nargs="?" if command == "plugin-status" else None)
        parser.add_argument("--json", action="store_true")
    options = parser.parse_args(arguments)
    if command == "plugin-install":
        method = "plugin/install"
        params = {"source": options.source, "marketplace": options.marketplace, "ref": options.ref,
                  "sparse": [item.strip() for item in options.sparse.split(",") if item.strip()],
                  "update_id": options.update_id}
    elif command == "plugin-status":
        method = "plugin/status" if options.identity is None else "plugin/update"
        params = {} if options.identity is None else {"update_id": options.identity}
    else:
        method, params = "plugin/uninstall", {"plugin_id": options.identity}
    try:
        result = await request(workspace, method, params)
    except (ConnectionError, OSError) as error:
        print(str(error), file=sys.stderr)
        print("Gateway 无可连接端点。用 plugin-doctor 查看实际 ID，运行 "
              "plugin-enable gateway@<marketplace> 后重启实例。", file=sys.stderr)
        return 1
    except (ValueError, RuntimeError, RemoteError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


async def install_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    return await management(arguments, workspace=workspace, config_path=config_path, command="plugin-install")


async def status_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    return await management(arguments, workspace=workspace, config_path=config_path, command="plugin-status")


async def uninstall_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    return await management(arguments, workspace=workspace, config_path=config_path, command="plugin-uninstall")


async def app_server_main(arguments: tuple[str, ...], *, workspace: Path, config_path: Path) -> int:
    """标准协议输出与宿主日志分流，正式宿主仍拥有启动、锁和停止。"""
    parser = argparse.ArgumentParser(prog="app-server")
    parser.add_argument("--stdio", action="store_true", required=True)
    parser.parse_args(arguments)
    core_root = Path(os.environ["AKASHIC_CORE_ROOT"])
    output_fd = os.dup(sys.stdout.fileno())
    os.set_inheritable(output_fd, True)
    os.environ["AKASHIC_GATEWAY_STDIO"] = str(output_fd)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    os.execv(sys.executable, [sys.executable, str(core_root / "main.py"), "gateway",
        "--workspace", str(workspace), "--config", str(config_path)])
