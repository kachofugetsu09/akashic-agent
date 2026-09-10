"""模型 facts 的公开结构合同。

`model.facts` 的校验与展示投影是只读纯函数：只解释内容块的值，不读写会话、
不调用模型、不执行工具。Core 的通道视图与多个业务插件都需要它，因此由合同层
拥有，实现留在 `plugins/models/projection.py`（再导出）。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from agent.plugin_contracts.message import ContentPart, ContentReferences


def check_facts(part: ContentPart) -> ContentReferences:
    """验证存储边界的 replay 数据；它不能包含可执行消息或角色声明。"""
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("model.facts 必须是对象")
    value = cast(Mapping[str, object], value)
    old_fields = {"call_record_id", "tool_ids", "thinking", "continuation"}
    new_fields = old_fields | {"wire_tool_calls", "reminder"}
    if set(value) not in (old_fields, new_fields):
        raise ValueError("model.facts 字段无效")
    if not isinstance(value["call_record_id"], str) or not value["call_record_id"]:
        raise ValueError("model.facts 缺少调用记录")
    ids = value["tool_ids"]
    if not isinstance(ids, Mapping):
        raise ValueError("模型工具 ID 必须按 Output 位置记录")
    ids = cast(Mapping[str, object], ids)
    for index, identity in ids.items():
        if (
            not index.isdecimal()
            or str(int(index)) != index
            or not isinstance(identity, str)
            or not identity
        ):
            raise ValueError("模型工具 ID 或位置无效")
    if len(set(ids.values())) != len(ids):
        raise ValueError("同一响应的模型工具 ID 不能重复")
    if "wire_tool_calls" in value:
        wire = value["wire_tool_calls"]
        if not isinstance(wire, Mapping) or set(wire) - set(ids):
            raise ValueError("wire 工具调用必须对应实际 ToolCall")
        for index, raw in cast(Mapping[str, object], wire).items():
            if not isinstance(raw, Mapping):
                raise ValueError("wire 工具调用必须是对象")
            call = cast(Mapping[str, object], raw)
            if (
                set(call) != {"name", "arguments"}
                or not isinstance(call["name"], str)
                or not call["name"]
                or not isinstance(call["arguments"], Mapping)
            ):
                raise ValueError("wire 工具调用字段无效")
        if value["reminder"] is not None and not isinstance(value["reminder"], str):
            raise ValueError("模型请求 reminder 必须是文本或 None")
    if value["thinking"] is not None and not isinstance(value["thinking"], str):
        raise ValueError("模型思考必须是文本或 None")
    continuation = value["continuation"]
    if continuation is not None:
        if not isinstance(continuation, Mapping):
            raise ValueError("模型 continuation 必须是对象")
        continuation = cast(Mapping[str, object], continuation)
        if (
            set(continuation) != {"binding_id", "payload"}
            or not isinstance(continuation["binding_id"], str)
            or not continuation["binding_id"]
            or not isinstance(continuation["payload"], Mapping)
        ):
            raise ValueError("模型 continuation 无效")
    return ContentReferences()


def display_facts(part: ContentPart) -> dict[str, object]:
    """页面只取得调用记录与思考文本，不能取得 provider continuation。"""
    _ = check_facts(part)
    value = cast(Mapping[str, object], part.value)
    return {"call_record_id": value["call_record_id"], "thinking": value["thinking"]}
