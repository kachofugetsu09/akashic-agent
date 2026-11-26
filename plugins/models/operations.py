"""模型设置与用量的 Self-CLI operation：与 Dashboard HTTP、Gateway 共用 model_settings_http 的同一组领域入口。"""
from __future__ import annotations

import json
import types
from typing import Literal, Union, get_args, get_origin

from fastapi import HTTPException
from pydantic import BaseModel

from agent.plugin_composition import Context
from plugins.models.contract import CacheUsageRead
from plugins.operations.contract import (
    OPERATIONS_V1,
    OperationHandler,
    OperationRequest,
    OperationResult,
    OperationSpec,
    OptionSpec,
    QueryHandler,
    QueryMirror,
    register_query_mirrors,
)

from .model_settings_http import COMMAND_PAYLOADS, QUERY_PAYLOADS, ModelControl, run_operation, usage_overview

_FIELD_HINTS = {
    "expected_revision": "当前模型目录修订号（model catalog show 的 revision）",
    "connection_id": "连接 ID",
    "model_id": "模型 ID",
    "credential": "凭据 JSON，如 {\"api_key\": \"...\"}；只写入，不回显",
    "driver_config": "驱动配置 JSON",
    "role": "JSON：\"default\" 设全局默认模型，null 设默认向量模型",
}


# 载荷字段 → CLI 参数：整数/布尔/字符串原样，Literal 给出可选值，可为 null 的必填字段与结构化字段走 JSON。
def payload_options(payload: type[BaseModel]) -> tuple[OptionSpec, ...]:
    options: list[OptionSpec] = []
    for name, field in payload.model_fields.items():
        if name == "type":
            continue
        args = get_args(field.annotation)
        nullable = get_origin(field.annotation) in (Union, types.UnionType) and type(None) in args
        inner = next((arg for arg in args if arg is not type(None)), field.annotation) if nullable else field.annotation
        choices = tuple(str(v) for v in get_args(inner)) if get_origin(inner) is Literal else ()
        value_type: Literal["string", "bool", "int", "json"] = (
            "json" if nullable and field.is_required() else
            "bool" if inner is bool else "int" if inner is int else
            "string" if inner is str or choices else "json"
        )
        options.append(OptionSpec(
            name, _FIELD_HINTS.get(name, name), value_type,
            required=field.is_required(), choices=choices if value_type == "string" else (),
        ))
    return tuple(options)


def _failure(error: HTTPException) -> OperationResult:
    detail = error.detail if isinstance(error.detail, str) else json.dumps(error.detail, ensure_ascii=False)
    return OperationResult(False, detail, error_code=f"http_{error.status_code}")


# 同一 run_operation 入口；summary 只报修订号与状态，不回显载荷（凭据、API Key）。
def _handler(control: ModelControl, name: str) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        try:
            data = await run_operation(control, name, req.args)
        except HTTPException as error:
            return _failure(error)
        facts = [f"{key}={data[key]}" for key in ("revision", "status") if data.get(key) is not None]
        return OperationResult(True, f"{req.operation_id} 完成" + (f"：{' '.join(facts)}" if facts else ""), data=data)
    return handle


def _usage_handler(usage: CacheUsageRead) -> OperationHandler:
    async def handle(req: OperationRequest) -> OperationResult:
        try:
            data = await usage_overview(usage, str(req.args["since"]))
        except RuntimeError as error:
            return OperationResult(False, str(error), error_code="unavailable")
        return OperationResult(True, "用量概览已读取", data=data)
    return handle


# 设置·模型页的读写入口：每个 HTTP 路由与 /command 子动作各一个 operation。
async def register_settings_ops(ctx: Context, control: ModelControl, usage: CacheUsageRead) -> None:
    def run(name: str) -> OperationHandler:
        return _handler(control, name)

    def opts(name: str) -> tuple[OptionSpec, ...]:
        return payload_options(COMMAND_PAYLOADS[name] if name in COMMAND_PAYLOADS else QUERY_PAYLOADS[name][0])

    specs = (
        OperationSpec("models.usage_overview", ("model", "usage", "overview"), "读取设置·用量的概览（since 为带时区的 ISO 时刻）", _usage_handler(usage),
                      (OptionSpec("since", "范围起点，带时区的 ISO 时刻", required=True),), mirrors=("http:models:GET /api/dashboard/models/usage",)),
        OperationSpec("models.call_show", ("model", "call", "show"), "读取一次模型调用的统计", run("call_stats"),
                      opts("call_stats"), mirrors=("http:models:GET /calls/{call_id}", "gateway:models/call_stats")),
        OperationSpec("models.catalog", ("model", "catalog", "show"), "读取模型目录：连接、模型、默认绑定与修订号", run("catalog"),
                      mirrors=("http:models:GET /catalog", "gateway:models/catalog")),
        OperationSpec("models.quotas", ("model", "quota", "list"), "读取各连接的额度", run("quotas"),
                      opts("quotas"), mirrors=("http:models:GET /quotas", "gateway:models/quotas")),
        OperationSpec("models.discover", ("model", "connection", "discover"), "用未保存的连接参数发现可用模型", run("discover"),
                      opts("discover"), mirrors=("http:models:POST /discover", "gateway:models/discover")),
        OperationSpec("models.discover_saved", ("model", "connection", "discover-saved"), "用已保存连接发现可用模型", run("discover_saved"),
                      opts("discover_saved"), mirrors=("http:models:POST /discover_saved", "gateway:models/discover_saved")),
        OperationSpec("models.probe_embedding", ("model", "embedding", "probe"), "试算向量模型维度与归一化", run("probe_embedding"),
                      opts("probe_embedding"), mirrors=("http:models:POST /probe_embedding", "gateway:models/probe_embedding")),
        OperationSpec("models.connection_add", ("model", "connection", "add"), "新增连接", run("add_connection"), opts("add_connection"),
                      mirrors=("http:models:POST /command#add_connection",)),
        OperationSpec("models.connection_create_with_model", ("model", "connection", "create-with-model"), "新增连接并加入一个模型", run("create_connection_with_model"), opts("create_connection_with_model"),
                      mirrors=("http:models:POST /command#create_connection_with_model",)),
        OperationSpec("models.connection_update", ("model", "connection", "update"), "修改连接", run("update_connection"), opts("update_connection"),
                      mirrors=("http:models:POST /command#update_connection",)),
        OperationSpec("models.connection_disable", ("model", "connection", "disable"), "停用连接", run("disable_connection"), opts("disable_connection"),
                      mirrors=("http:models:POST /command#disable_connection",)),
        OperationSpec("models.connection_enable", ("model", "connection", "enable"), "启用连接", run("enable_connection"), opts("enable_connection"),
                      mirrors=("http:models:POST /command#enable_connection",)),
        OperationSpec("models.connection_catalog_mode", ("model", "connection", "catalog-mode"), "设置连接列出全部或已挑选模型", run("set_catalog_mode"), opts("set_catalog_mode"),
                      mirrors=("http:models:POST /command#set_catalog_mode",)),
        OperationSpec("models.connection_sync", ("model", "connection", "sync"), "按服务端同步连接的模型", run("sync_models"), opts("sync_models"),
                      mirrors=("http:models:POST /command#sync_models",)),
        OperationSpec("models.connection_auth_start", ("model", "connection", "auth-start"), "开始连接登录，返回授权挑战", run("start_auth"), opts("start_auth"),
                      mirrors=("http:models:POST /command#start_auth",)),
        OperationSpec("models.connection_auth_finish", ("model", "connection", "auth-finish"), "完成连接登录", run("finish_auth"), opts("finish_auth"),
                      mirrors=("http:models:POST /command#finish_auth",)),
        OperationSpec("models.connection_auth_cancel", ("model", "connection", "auth-cancel"), "取消连接登录", run("cancel_auth"), opts("cancel_auth"),
                      mirrors=("http:models:POST /command#cancel_auth",)),
        OperationSpec("models.model_add", ("model", "add"), "加入模型", run("add_model"), opts("add_model"),
                      mirrors=("http:models:POST /command#add_model",)),
        OperationSpec("models.model_update", ("model", "update"), "修改模型的上下文、输出上限与图片输入", run("update_model"), opts("update_model"),
                      mirrors=("http:models:POST /command#update_model",)),
        OperationSpec("models.model_remove", ("model", "remove"), "移除模型", run("remove_model"), opts("remove_model"),
                      mirrors=("http:models:POST /command#remove_model",)),
        OperationSpec("models.model_verify", ("model", "verify"), "重新验证模型可用", run("verify_model"), opts("verify_model"),
                      mirrors=("http:models:POST /command#verify_model",)),
        OperationSpec("models.default_set", ("model", "default", "set"), "设置全局默认模型（role=\"default\"）或默认向量模型（role=null）", run("set_default"), opts("set_default"),
                      mirrors=("http:models:POST /command#set_default",)),
    )
    operations = ctx.require(OPERATIONS_V1)
    for item in specs:
        _ = await operations.register(ctx, item)


# 聊天界面的插件查询原样发布：参数名即界面载荷键。
CHAT_MIRRORS = (
    QueryMirror("usage.message", ("model", "usage", "message"), "读取一条回答的输出 Token", (OptionSpec("message_id", "消息 ID", required=True),)),
    QueryMirror("usage.insights", ("model", "usage", "insights"), "读取用量洞察与会话排行", (
        OptionSpec("since", "范围起点，带时区的 ISO 时刻"), OptionSpec("limit", "会话行数", "int"),
        OptionSpec("order", "排序", choices=("input", "context", "recent")))),
    QueryMirror("usage.spread", ("model", "usage", "spread"), "读取构成概览", (OptionSpec("since", "范围起点，带时区的 ISO 时刻"),)),
    QueryMirror("usage.status", ("model", "usage", "status"), "读取当前会话的用量短值"),
    QueryMirror("usage.session", ("model", "usage", "session"), "读取一个会话的用量（默认当前会话）", (OptionSpec("session_id", "会话 ID"),)),
    QueryMirror("usage.composition", ("model", "usage", "composition"), "读取一次调用的上下文构成", (
        OptionSpec("call_id", "调用 ID", required=True), OptionSpec("session_id", "会话 ID（默认当前会话）"))),
    QueryMirror("chat.models", ("model", "chat", "list"), "列出可选的聊天模型与当前会话的选择", (OptionSpec("catalog", "已有目录指纹；一致时省掉目录"),)),
    QueryMirror("call.latest", ("model", "call", "latest"), "读取当前会话最近一次模型调用"),
    QueryMirror("call.stats", ("model", "call", "stats"), "读取一次模型调用的统计（聊天界面口径）", (OptionSpec("call_id", "调用 ID", required=True),)),
)


async def register_chat_ops(ctx: Context, query: QueryHandler) -> None:
    await register_query_mirrors(ctx, ctx.require(OPERATIONS_V1), query=query, mirrors=CHAT_MIRRORS)


__all__ = ["register_chat_ops", "register_settings_ops", "payload_options"]
