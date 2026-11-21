"""运行时能力投影：把工具、命令、界面、设置、RPC 入口汇总成确定性的只读 JSON，供迁移前后对比。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import astuple
from typing import cast

from agent.plugin_composition import Context
from agent.plugin_composition.runtime_catalog import RUNTIME_CATALOG
from plugins.commands.contract import COMMANDS
from plugins.gateway.contract import NAMES
from plugins.plugin_settings.contract import SETTINGS_FORMS
from plugins.tools.contract import ALL_TOOLS
from plugins.ui.contract import UI, UI_SLOTS, WEB_UI

JsonObject = dict[str, object]

_RPC_PREFIX = "gateway.rpc:"
# Gateway 在服务构造时固定登记、不经服务键发布的控制方法。
_FIXED_RPC = ("runtime/cancel-stop", "runtime/prepare-stop")


# 冻结的 JSON 值（Mapping、tuple）转成普通 JSON 值，键排序后哈希才稳定。
def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        items = cast(Mapping[object, object], value)
        return {str(key): _plain(items[key]) for key in sorted(items, key=str)}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in cast(Iterable[object], value)]
    return value


# 规范化 JSON 的 sha256；不依赖字典顺序与空白。
def stable_hash(value: object) -> str:
    text = json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 模型可见工具：名字、owner、参数 schema 与描述各自的哈希。
def _tools(ctx: Context) -> list[JsonObject]:
    rows: list[JsonObject] = []
    with ctx.borrow(ALL_TOOLS) as view:
        refs = () if view is None else view().refs
        for ref in refs:
            description = ref.description
            rows.append({
                "name": ref.name,
                "owner": description["owner"],
                "schema_sha256": stable_hash(description["parameters"]),
                "description_sha256": stable_hash(description["description"]),
                "idempotent": description["idempotent"],
            })
    return sorted(rows, key=lambda row: str(row["name"]))


# 斜杠命令：名字、别名、提示与 owner。
def _commands(ctx: Context) -> list[JsonObject]:
    with ctx.borrow(COMMANDS) as commands:
        descriptors = () if commands is None else commands.freeze().descriptors
    return sorted(({
        "name": item.name, "owner": item.owner, "aliases": sorted(item.aliases),
        "input_hint": item.input_hint, "description": item.description,
    } for item in descriptors), key=lambda row: str(row["name"]))


# 聊天界面插槽：每个插件声明的插槽与模块哈希。
def _slots(ctx: Context) -> list[JsonObject]:
    with ctx.borrow(UI_SLOTS) as slots:
        bindings = () if slots is None else slots.bindings()
    return sorted(({
        "owner": item.descriptor.owner, "slots": sorted(item.descriptor.slots),
        "module_sha256": item.descriptor.module_sha256,
        "stylesheet_sha256": item.descriptor.stylesheet_sha256,
    } for item in bindings), key=lambda row: str(row["owner"]))


# Dashboard 绑定：每个插件挂的 HTTP/WebSocket 路由，不含框架自带的文档路由。
def _dashboards(ctx: Context) -> list[JsonObject]:
    with ctx.borrow(UI) as ui:
        bindings = () if ui is None else ui.bindings()
    rows: list[JsonObject] = []
    for item in bindings:
        routes = {f"{','.join(sorted(getattr(route, 'methods', None) or ('WS',)))} {route.path}"
                  for route in item.routes if str(route.path).startswith("/api/")}
        rows.append({"owner": item.plugin_id, "has_web": item.has_web,
                     "navigation_paths": sorted(item.navigation_paths), "routes": sorted(routes)})
    return sorted(rows, key=lambda row: str(row["owner"]))


# Web 模块：取已缓存的 bootstrap，只留插件、哈希与它要求或提供的前端合同。
async def _web_modules(ctx: Context) -> list[JsonObject]:
    with ctx.borrow(WEB_UI) as web:
        if web is None:
            return []
        payload = cast(JsonObject, json.loads(await web.bootstrap()))
    modules = cast(list[JsonObject], payload["modules"])
    return sorted(({
        "owner": item["pluginId"], "module_sha256": item["moduleSha256"],
        "stylesheet_sha256": item["stylesheetSha256"],
        "requires": sorted(cast(list[str], item["requires"])),
        "provides": sorted(cast(list[str], item["provides"])),
        "contract_digests": item["contractDigests"],
    } for item in modules), key=lambda row: str(row["owner"]))


# 声明式设置表单：id、owner、标题与字段。
def _settings(ctx: Context) -> list[JsonObject]:
    with ctx.borrow(SETTINGS_FORMS) as forms:
        entries = () if forms is None else forms.entries()
    rows = [entry.body() for entry in entries]
    return sorted(rows, key=lambda row: str(row["id"]))


# RPC 方法名：Gateway 固定方法加各插件以服务键发布的方法。
def _rpc_methods(ctx: Context) -> list[JsonObject]:
    services = cast(list[JsonObject], ctx.require(RUNTIME_CATALOG)(ctx)["services"])
    rows: list[JsonObject] = [{"name": name, "owner": "gateway"}
                              for name in (*astuple(NAMES), *_FIXED_RPC)]
    rows += [{"name": str(item["key"]).removeprefix(_RPC_PREFIX), "owner": item["owner"]}
             for item in services if str(item["key"]).startswith(_RPC_PREFIX)]
    return sorted(rows, key=lambda row: str(row["name"]))


# 汇总各 owner 的当前登记；缺席的可选服务给空列表。
async def read_capabilities(ctx: Context) -> JsonObject:
    return {
        "tools": _tools(ctx),
        "commands": _commands(ctx),
        "ui_slots": _slots(ctx),
        "web_modules": await _web_modules(ctx),
        "dashboards": _dashboards(ctx),
        "settings_forms": _settings(ctx),
        "rpc_methods": _rpc_methods(ctx),
    }
