from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .contract import SETTINGS_FORMS, SettingsEntry

inject = (SETTINGS_FORMS,)
PREFIX = "/api/dashboard/plugin_settings"


class ConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: str = Field(min_length=1, max_length=128)
    expected_input: str = Field(pattern=r"^[0-9a-f]{64}$")
    values: dict[str, object]


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

    @app.get(PREFIX + "/{key}/config/receipts/{request_id}")
    async def receipt(key: str, request_id: str):
        try:
            return entry(key).settings.receipt(request_id)
        except KeyError as error:
            raise HTTPException(404, "配置回执不存在") from error
