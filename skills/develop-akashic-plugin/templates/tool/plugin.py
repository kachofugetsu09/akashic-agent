"""最小工具插件：注册一个模型可调用的工具。复制到工作区后改 name、工具名、参数与 invoke。"""

from __future__ import annotations

import json
import random
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart
from plugins.tools.contract import TOOLS, CallSource

api_version = 3
name = "dice_tool"
version = "0.1.0"
desc = "掷骰子"
inject = (TOOLS,)


class RollInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sides: int = Field(default=6, ge=2, le=1000, description="骰子面数")


@dataclass(frozen=True, slots=True)
class Result:
    outcome: Literal["success", "denied", "error", "interrupted"]
    parts: tuple[ContentPart, ...]


class RollTool:
    idempotent = False

    # prepare 只校验并规范参数；返回值就是 invoke 收到的 arguments。
    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None) -> Mapping[str, object]:
        return RollInput.model_validate(dict(arguments)).model_dump()

    async def invoke(self, key: str, arguments: Mapping[str, object]) -> Result:
        sides = int(str(arguments["sides"]))
        value = random.randint(1, sides)
        return Result("success", (ContentPart("text", json.dumps({"sides": sides, "value": value})),))

    # 无法确定原调用是否生效时返回 None；不要伪造结果。
    async def query(self, key: str) -> Result | None:
        return None


async def apply(ctx: Context) -> None:
    catalog = ctx.require(TOOLS)
    # always_on=False 时模型需先 load_tools(插件 ID) 再调用；常用工具可设为 True。
    _ = await catalog.declare_group(ctx, always_on=False, description=desc)

    @asynccontextmanager
    async def open_tool(_state: Mapping[str, object]) -> AsyncIterator[RollTool]:
        yield RollTool()

    _ = await catalog.register(
        ctx, name="roll_dice", description="掷一次骰子并返回点数",
        parameters=RollInput.model_json_schema(), open=open_tool,
    )
