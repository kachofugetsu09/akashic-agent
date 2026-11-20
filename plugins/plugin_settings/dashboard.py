from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contract import SETTINGS_FORMS, Authorize, SettingsEntry

inject = (SETTINGS_FORMS,)
PREFIX = "/api/dashboard/plugin_settings"


class ConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    expected_input: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: dict[str, object]


class FinishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    callback: str = Field(min_length=1, max_length=8192)


# 表单目录与每张表单的读、存、回执；HTTP 边界只校验信封，字段由贡献插件的 Configuration 解释。
def register(app, context):
    def entry(key: str) -> SettingsEntry:
        forms = context.require(SETTINGS_FORMS)
        try:
            return forms.entry(key)
        except KeyError as error:
            raise HTTPException(404, "这个插件没有声明设置表单") from error

    @app.get(PREFIX + "/forms")
    async def forms():
        return {"forms": [item.body() for item in context.require(SETTINGS_FORMS).entries()]}

    @app.get(PREFIX + "/{key}/config")
    async def read(key: str):
        return await entry(key).settings.read()

    @app.post(PREFIX + "/{key}/config")
    async def save(key: str, request: ConfigRequest):
        try:
            return await entry(key).settings.save(request.request_id, request.expected_input, request.values)
        except ValidationError as error:
            raise HTTPException(422, error.errors(include_input=False, include_url=False, include_context=False)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    def authorization(key: str) -> tuple[SettingsEntry, Authorize]:
        item = entry(key)
        if item.authorize is None:
            raise HTTPException(404, "这个插件没有声明授权")
        return item, item.authorize

    # 授权在贡献插件的运行作用域里执行；ValueError 是用户能改正的错误（回调网址不对、授权已过期）。
    @app.post(PREFIX + "/{key}/authorize/start")
    async def authorize_start(key: str):
        item, authorize = authorization(key)
        try:
            async with item.context.runtime_scope():
                return {"url": await authorize.start()}
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post(PREFIX + "/{key}/authorize/finish")
    async def authorize_finish(key: str, request: FinishRequest):
        item, authorize = authorization(key)
        try:
            async with item.context.runtime_scope():
                await authorize.finish(request.callback.strip())
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return {"ok": True}

    @app.get(PREFIX + "/{key}/config/receipts/{request_id}")
    async def receipt(key: str, request_id: str):
        try:
            return entry(key).settings.receipt(request_id)
        except KeyError as error:
            raise HTTPException(404, "配置回执不存在") from error
