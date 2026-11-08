from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from typing import cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from agent.plugin_composition import Context
from plugins.ledger.contract import ContentPart, json_value
from plugins.tools.contract import TOOLS, CallSource, Result
from .contract import USER_INFORMATION, InformationProvider

api_version = 3
name = "userinformation"
version = "1.0.0"
desc = "统一用户信息入口：发现来源及查询 schema，读取睡眠、游戏、位置等证据；不自动注入状态"
inject = (TOOLS,)


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    provider: str = Field(min_length=1)
    query: str = Field(min_length=1)
    arguments: dict[str, object] = Field(default_factory=dict)


class Empty(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Providers:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.entries: dict[str, tuple[Context, InformationProvider]] = {}

    async def register(self, ctx: Context, provider: InformationProvider):
        if ctx.root_instance_token is not self.ctx.root_instance_token or ctx.require(USER_INFORMATION) is not self:
            raise ValueError("Information provider must belong to this composition")
        identity = ctx.runtime.plugin_id
        def setup():
            if identity in self.entries:
                raise ValueError(f"Duplicate information provider: {identity}")
            self.entries[identity] = (ctx, provider)
            def close():
                del self.entries[identity]
            return close
        return await ctx.effect(setup, label=f"userinformation:{identity}")

    async def call(self, identity: str, query: Query | None = None):
        entry = self.entries.get(identity)
        if entry is None:
            return {"status": "unavailable", "provider": identity, "error": "Provider not registered"}
        owner, provider = entry
        try:
            async with asyncio.timeout(20), owner.runtime_scope():
                value = await provider.describe() if query is None else await provider.query(query.query, query.arguments)
                encoded = json.dumps(json_value(value), ensure_ascii=False, allow_nan=False)
                if len(encoded.encode()) > 131072:
                    raise ValueError("Provider result exceeds 128 KiB; narrow or paginate the query")
                return {"provider": identity, "result": json.loads(encoded)}
        except (TimeoutError, OSError) as error:
            return {"provider": identity, "status": "unavailable", "error": str(error) or type(error).__name__}


class InformationTool:
    idempotent = False
    def __init__(self, providers: Providers, listing: bool):
        self.providers, self.listing = providers, listing

    async def prepare(self, arguments: Mapping[str, object], source: CallSource | None = None):
        try:
            return (Empty if self.listing else Query).model_validate(json_value(arguments)).model_dump()
        except ValidationError as error:
            return str(error)

    async def invoke(self, key: str, arguments: Mapping[str, object]):
        now = datetime.now(UTC).isoformat()
        if self.listing:
            items = await asyncio.gather(*(self.providers.call(i) for i in sorted(self.providers.entries)))
            value = {"queried_at": now, "providers": items}
        else:
            query = Query.model_validate(json_value(arguments))
            value = {"queried_at": now, **await self.providers.call(query.provider, query)}
        failed = "error" in value or (isinstance(value.get("result"), dict) and "error" in value["result"])
        return Result("error" if failed else "success", (ContentPart("text", json.dumps(value, ensure_ascii=False)),))

    async def query(self, key: str):
        return None


async def apply(ctx: Context) -> None:
    providers = Providers(ctx)
    await ctx.provide(USER_INFORMATION, providers)
    catalog = ctx.require(TOOLS)
    await catalog.declare_group(ctx, description=desc)
    for tool_name, listing, description in (
        ("user_information_list", True, "发现当前用户信息来源、可用查询及参数 schema；不读取完整隐私数据"),
        ("user_information_query", False, "按 list 返回的 provider、query 和参数查询用户信息。结果是带时间的观测，不保证实时；缺失不能解释为清醒或离线"),
    ):
        @asynccontextmanager
        async def open_tool(state: Mapping[str, object], listing: bool = listing):
            yield InformationTool(providers, listing)
        await catalog.register(ctx, name=tool_name, description=description,
            parameters=(Empty if listing else Query).model_json_schema(), open=open_tool, parallel=True)
