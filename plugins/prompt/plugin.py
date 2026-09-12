from __future__ import annotations

import platform
from collections.abc import Awaitable, Callable, Mapping
from datetime import timedelta
from typing import Protocol, cast

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_contracts import Input, Message
from agent.plugin_contracts import json_value

from .persona import read_veda_file
from .text import build_behavior_rules, build_identity, build_telegram_rendering_prompt

api_version = 3
name = "prompt"
version = "1.0.0"
desc = "每次请求读取人格与行为规则，附带已接纳输入的时间和渠道事实"
workspace_files = ("memory/VEDA.md",)


class MaterialRegistry(Protocol):
    async def register(
        self, ctx: Context, *, name: str,
        prepare: Callable[[tuple[Message, ...], str], Awaitable[Mapping[str, object]]],
        priority: int = 0, prompt: bool = False, reduce: object | None = None,
    ) -> object: ...


MATERIALS = ServiceKey[MaterialRegistry]("context.materials.v3")
inject = (MATERIALS,)


async def apply(ctx: Context, config: object) -> None:
    """只贡献已获授的 Prompt 和只读环境材料，不取得任何消息 writer。"""
    async def prepare(snapshot: tuple[Message, ...], source: str) -> Mapping[str, object]:
        # 1. 文件是人格唯一真源；已返回字符串在本次请求中保持不变。
        prompt = "\n\n".join((
            read_veda_file(ctx.workspace_file("memory/VEDA.md")),
            build_identity(workspace=ctx.runtime.workspace), build_behavior_rules(),
        ))
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

    _ = await ctx.require(MATERIALS).register(ctx, name="default_prompt", prepare=prepare, prompt=True, priority=100)
