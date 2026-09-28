"""插件自有配置服务的窄 HTTP 合同，不声明业务字段。"""
from __future__ import annotations

from typing import Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from agent.plugin_composition import DashboardContext, ServiceKey


class Configuration(Protocol):
    async def read(self) -> dict[str, object]: ...
    async def save(self, request_id: str, expected_input: str, values: dict[str, object]) -> dict[str, object]: ...
    def receipt(self, request_id: str) -> dict[str, object]: ...


class ConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    expected_input: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: dict[str, object]


def register_routes(app: FastAPI, context: DashboardContext, key: ServiceKey[Configuration], prefix: str) -> None:
    """HTTP 边界只校验请求信封；插件解释字段、处理业务错误。"""
    @app.get(prefix)
    async def read():
        return await context.require(key).read()

    @app.post(prefix)
    async def save(request: ConfigRequest):
        try:
            return await context.require(key).save(request.request_id, request.expected_input, request.values)
        except ValidationError as error:
            raise HTTPException(422, error.errors(include_input=False, include_url=False, include_context=False)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get(prefix + "/receipts/{request_id}")
    async def receipt(request_id: str):
        try:
            return context.require(key).receipt(request_id)
        except KeyError as error:
            raise HTTPException(404, "配置回执不存在") from error
