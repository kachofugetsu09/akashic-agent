"""在插件与独立 MCP 进程间传递调用、取消和收尾协议。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


def endpoint_name(data_root: Path, generation_id: str) -> str:
    """按数据目录和实际 MCP generation 路由控制 socket。"""
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("Computer generation id 不能为空")
    return (
        "akashic-computer-"
        + hashlib.sha256(
            (str(data_root.resolve()) + "\0" + generation_id).encode()
        ).hexdigest()[:32]
    )


async def request(name: str, payload: Mapping[str, object]) -> dict[str, object]:
    """取消时等待驱动释放回执，再向 Core 传播 CancelledError。"""
    reader, writer = await asyncio.open_unix_connection(
        "\0" + name, limit=8 * 1024 * 1024
    )
    try:
        writer.write(json.dumps(payload).encode() + b"\n")
        await writer.drain()
        try:
            async with asyncio.timeout(330 if payload.get("op") == "end_turn" else 170):
                raw = await reader.readline()
        except asyncio.CancelledError:
            writer.write(b'{"cancel":true}\n')
            await writer.drain()
            cleanup = asyncio.create_task(reader.readline())
            try:
                async with asyncio.timeout(45):
                    raw = await asyncio.shield(cleanup)
            finally:
                cleanup.cancel()
                await asyncio.gather(cleanup, return_exceptions=True)
            receipt = json.loads(raw)
            if receipt.get("error") or receipt.get("released") is not True:
                raise RuntimeError(
                    f"Computer cancellation release was not confirmed: {receipt}"
                )
            raise
        if not raw:
            raise RuntimeError("Computer control connection closed before settlement")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError("Computer control returned an invalid response")
        if "error" in value:
            raise RuntimeError(str(value["error"]))
        return value
    finally:
        writer.close()
        await writer.wait_closed()
