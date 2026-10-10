from __future__ import annotations

import platform
from collections.abc import Mapping
from datetime import timedelta
from typing import cast

from agent.plugin_composition import Context
from agent.plugin_contracts import Input, Message, json_value
from agent.plugin_contracts.context import (
    MATERIALS_V4 as MATERIALS,
)

from plugins.runtime_inspection.contract import DOCUMENTS, Document

from .persona import read_veda_file, initialize_veda_if_missing
from .text import build_behavior_rules, build_identity, build_telegram_rendering_prompt

api_version = 3
name = "prompt"
version = "1.0.0"
desc = "每次请求读取人格与行为规则，附带已接纳输入的时间和渠道事实"
workspace_files = ("memory/VEDA.md",)


inject = (MATERIALS,)


async def apply(ctx: Context) -> None:
    """只贡献已获授的 Prompt 和只读环境材料，不取得任何消息 writer。"""
    initialize_veda_if_missing(ctx.runtime.workspace)
    await ctx.inject((DOCUMENTS,), publish_documents, name="documents")

    # 路径与固定段落在本 generation 内解析一次；人格文本按文件签名缓存，热更新随 apply 重建。
    veda_path = ctx.workspace_file("memory/VEDA.md")
    fixed = (build_identity(workspace=ctx.runtime.workspace), build_behavior_rules())
    cached: list[tuple[tuple[int, int, int] | None, str]] = []

    def persona() -> str:
        try:
            stat = veda_path.stat()
            signature: tuple[int, int, int] | None = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        except FileNotFoundError:
            signature = None
        if cached and cached[0][0] == signature and signature is not None:
            return cached[0][1]
        text = "\n\n".join((read_veda_file(veda_path), *fixed))
        cached[:] = [(signature, text)]
        return text

    async def prepare(snapshot: tuple[Message, ...], source: str) -> Mapping[str, object]:
        # 1. 文件是人格唯一真源；签名变化时重读，已返回字符串在本次请求中保持不变。
        prompt = persona()
        values: dict[str, object] = {"architecture": platform.machine()}
        latest = next((item for item in reversed(snapshot)
                       if item.source == source and isinstance(item.body, Input)), None)
        if latest is not None:
            # 2. 用持久接纳时间解释相对日期；不改原文，也不猜外部发送时间或设备。
            ts = latest.recorded_at.astimezone()
            values.update({
                "input_id": latest.message_id, "request_time": ts.isoformat(),
                "time_basis": "输入接纳时间，不代表渠道发送时间",
                "today": ts.date().isoformat(), "yesterday": (ts - timedelta(days=1)).date().isoformat(),
                "tomorrow": (ts + timedelta(days=1)).date().isoformat(),
                "weekday": ts.strftime("%A"),
            })
            assert isinstance(latest.body, Input)
            origin = next((part for part in latest.body.parts if part.kind == "channel.origin"), None)
            if origin is not None:
                values["channel_origin"] = json_value(origin.value)
                channel = cast(Mapping[str, str], origin.value)["channel"]
                if channel == "telegram" or channel.startswith("telegram_"):
                    prompt += build_telegram_rendering_prompt()
        text = "## 当前环境\n" + "\n".join(f"- {key}: {value}" for key, value in values.items())
        environment: Mapping[str, object] = {
            "name": "environment", "text": text, "priority": 100,
        }
        return {
            "system_prompt": prompt,
            "reminders": (environment,),
        }

    _ = await ctx.require(MATERIALS).register(ctx, kind="context", name="default_prompt", prepare=prepare, prompt=True, priority=100)


async def publish_documents(ctx: Context) -> None:
    """只发布本插件拥有的人格文件；检查能力缺席不影响 Prompt。"""
    def read(limit: int) -> bytes:
        with ctx.workspace_file("memory/VEDA.md").open("rb") as file:
            return file.read(limit)

    await ctx.require(DOCUMENTS).register(ctx, Document(
        "veda", "VEDA 人格", "memory/VEDA.md", "identity", "Agent 的人格真源。", read, 300,
    ))
