"""声明式设置表单的 Self-CLI operation：与 Dashboard 路由走同一张表单的 Configuration 与 Authorize。"""
from __future__ import annotations

import json
from typing import cast

from pydantic import ValidationError

from agent.plugin_composition import Context
from plugins.operations.contract import (
    OPERATIONS_V1,
    OperationHandler,
    OperationRequest,
    OperationResult,
    OperationSpec,
    OptionSpec,
)

from .contract import AuthorizeUndeclared, SettingsEntry, SettingsForms

_KEY = OptionSpec("key", "表单键，即插件 ID（settings form list 的 id）", required=True)
_REQUEST_ID = OptionSpec("request_id", "本次保存的幂等请求 ID；重放同一 ID 返回同一回执", required=True)

def _entry(forms: SettingsForms, req: OperationRequest) -> SettingsEntry | OperationResult:
    key = str(req.args["key"])
    try:
        return forms.entry(key)
    except KeyError:
        return OperationResult(False, f"{key} 的设置暂不可用，可能正在应用新配置，请稍后重试", error_code="unavailable")


# 保存：保留 request_id + expected_input 的乐观并发与回执语义；校验错误不回显输入（secret 字段）。
def _save(forms: SettingsForms) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        entry = _entry(forms, req)
        if isinstance(entry, OperationResult):
            return entry
        values = req.args["values"]
        if not isinstance(values, dict):
            return OperationResult(False, "--values 须为 JSON 对象", error_code="invalid_arguments")
        fields = cast(dict[str, object], values)
        try:
            data = await entry.settings.save(str(req.args["request_id"]), str(req.args["expected_input"]), fields)
        except ValidationError as error:
            detail = error.errors(include_input=False, include_url=False, include_context=False)
            return OperationResult(False, json.dumps(detail, ensure_ascii=False), error_code="invalid")
        return OperationResult(True, f"{entry.key} 设置已提交", data=data)
    return handle


def _authorize_start(forms: SettingsForms) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        entry = _entry(forms, req)
        if isinstance(entry, OperationResult):
            return entry
        try:
            url = await entry.authorize_start()
        except AuthorizeUndeclared as error:
            return OperationResult(False, str(error), error_code="not_found")
        return OperationResult(True, f"请把授权链接交给用户打开并同意，再把跳转后的完整网址交回: {url}", data={"url": url})
    return handle


def _authorize_finish(forms: SettingsForms) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        entry = _entry(forms, req)
        if isinstance(entry, OperationResult):
            return entry
        try:
            await entry.authorize_finish(str(req.args["callback"]))
        except AuthorizeUndeclared as error:
            return OperationResult(False, str(error), error_code="not_found")
        return OperationResult(True, f"{entry.key} 授权完成", data={"ok": True})
    return handle


def _read(forms: SettingsForms, receipt: bool) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        entry = _entry(forms, req)
        if isinstance(entry, OperationResult):
            return entry
        if not receipt:
            return OperationResult(True, f"{entry.key} 当前设置已读取", data=await entry.settings.read())
        try:
            data = entry.settings.receipt(str(req.args["request_id"]))
        except KeyError:
            return OperationResult(False, "配置回执不存在", error_code="not_found")
        return OperationResult(True, f"{entry.key} 配置回执已读取", data=data)
    return handle


async def register_ops(ctx: Context, forms: SettingsForms) -> None:
    async def list_forms(_req: OperationRequest) -> OperationResult:
        items = [item.body() for item in forms.entries()]
        return OperationResult(True, f"共 {len(items)} 张设置表单", data={"forms": items})

    specs = (
        OperationSpec("plugin_settings.forms", ("settings", "form", "list"), "列出各插件声明的设置表单与字段", list_forms,
                      mirrors=("http:plugin_settings:GET /api/dashboard/plugin_settings/forms",)),
        OperationSpec("plugin_settings.read", ("settings", "show"), "读取一张表单的当前设置、默认值与 expected_input", _read(forms, False), (_KEY,),
                      mirrors=("http:plugin_settings:GET /api/dashboard/plugin_settings/{key}/config",)),
        OperationSpec("plugin_settings.save", ("settings", "save"), "保存一张表单的设置（secret 字段只写入，不回显）", _save(forms), (
            _KEY, _REQUEST_ID,
            OptionSpec("expected_input", "settings show 返回的 expected_input（64 位十六进制）", required=True),
            OptionSpec("values", "字段值 JSON 对象；空值回到默认", "json", required=True),
        ), mirrors=("http:plugin_settings:POST /api/dashboard/plugin_settings/{key}/config",)),
        OperationSpec("plugin_settings.receipt", ("settings", "receipt"), "读取一次设置保存的回执", _read(forms, True),
                      (_KEY, OptionSpec("request_id", "保存时的 request_id", required=True)),
                      mirrors=("http:plugin_settings:GET /api/dashboard/plugin_settings/{key}/config/receipts/{request_id}",)),
        OperationSpec("plugin_settings.authorize_start", ("settings", "authorize", "start"), "开始第三方授权，返回给用户打开的授权链接",
                      _authorize_start(forms), (_KEY,), mirrors=("http:plugin_settings:POST /api/dashboard/plugin_settings/{key}/authorize/start",)),
        OperationSpec("plugin_settings.authorize_finish", ("settings", "authorize", "finish"), "用用户交回的回调网址完成授权",
                      _authorize_finish(forms), (_KEY, OptionSpec("callback", "用户同意后浏览器跳转到的完整网址", required=True)),
                      mirrors=("http:plugin_settings:POST /api/dashboard/plugin_settings/{key}/authorize/finish",)),
    )
    operations = ctx.require(OPERATIONS_V1)
    for spec in specs:
        _ = await operations.register(ctx, spec)
