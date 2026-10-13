from __future__ import annotations

import asyncio
import logging
import os
import re

from agent.plugin_composition import Context
from plugins.host_execution.contract import PROCESSES
from plugins.models.contract import MODEL_DRIVERS
from plugins.ui.contract import UI

from .driver import definition

api_version = 3
name = "codex"
version = "1.0.0"
desc = "ChatGPT login, Codex catalog, and Responses transport"
author = "Akashic Core"
inject = (MODEL_DRIVERS, PROCESSES)
workspace_roots = ()
workspace_files = ()


async def _register_ui(ctx: Context) -> None:
    await ctx.require(UI).register(
        ctx, web="web_module.js",
        requires=("models.connection-types.v1",),
        provides=(),
        contract_digests={
            "models.connection-types.v1": "8c304d85090a65a4a66cd777a5c2a88e6908de147964dc91e7f84336979ea561",
        },
    )


async def apply(ctx: Context) -> None:
    """注册模型驱动；配置界面只影响可选子分支。"""
    # 1. 每代首次真实使用时读取，避免候选校验执行宿主命令；并发请求共用同一版本。
    client_version: str | None = None
    lock = asyncio.Lock()

    async def read_version() -> str:
        """同一代驱动共享宿主版本，失败后允许重新读取。"""
        nonlocal client_version
        async with ctx.runtime_scope(), lock:
            if client_version is None:
                client_version = await _load_version(ctx)
                logging.getLogger(__name__).info("Codex 使用宿主 CLI 版本 %s", client_version)
            return client_version

    # 2. 目录与 Responses 从同一读取入口取得版本，不保存到连接配置。
    _ = await ctx.require(MODEL_DRIVERS).register(ctx, definition(read_version))
    _ = await ctx.inject((UI,), _register_ui, name="ui")


async def _load_version(ctx: Context) -> str:
    """经宿主进程 owner 读取 CLI 版本；失败或取消后回收本次执行。"""
    processes = ctx.require(PROCESSES)
    # 1. 直接执行固定 argv；Bridge 保留宿主 PATH，本地模式使用当前进程 PATH。
    try:
        result = await processes.exec_command(
            ctx, "codex-version", command="codex --version", argv=["codex", "--version"],
            cwd=None, env={"PATH": os.environ.get("PATH", os.defpath)}, tty=False,
            yield_time_ms=10_000, max_output_tokens=256, hard_timeout_s=5,
        )
    finally:
        report = await processes.terminate_owner(ctx, "codex-version")
        if report.failures:
            raise RuntimeError(f"Codex 版本查询进程未完成清理: {report.failures}")
    # 2. 只接纳完整版本输出；缺少 CLI 或异常输出不降级为写死的版本。
    if result.exit_code != 0 or result.output_omitted_bytes:
        raise RuntimeError(
            f"读取宿主 Codex CLI 版本失败（exit={result.exit_code}, {result.finish_reason}）；"
            "请确认 codex --version 可在宿主执行环境运行"
        )
    match = re.fullmatch(
        rb"codex-cli ([0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?)",
        result.output.strip(),
    )
    if match is None:
        raise RuntimeError("宿主 codex --version 返回了无效版本输出")
    return match[1].decode("ascii")
