from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.models import (
    BoundChatModel,
    LLMResponse,
    ModelContinuation,
    ModelRequest,
)
from agent.plugin_contracts import (
    ContentReferences,
    CallRef,
    ContentPart,
    Control,
    Input,
    Message,
    Output,
    ToolCall,
    ToolResult,
)
from session.message_codec import json_value
from plugins.context.api import check_summary
from .store import ModelCallReader

ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]
CallReader = Callable[[str], Mapping[str, Any]]
MODEL_CALLS = ServiceKey[CallReader]("models.calls.v1")
MODEL_CALL_HISTORY = ServiceKey[Callable[[str, int], tuple[Mapping[str, Any], ...]]](
    "models.call-history.v1"
)


def response_facts(
    response: LLMResponse,
    call_indices: Sequence[int],
    *,
    reminder: str | None = None,
    wire_tool_calls: Mapping[str, Mapping[str, object]] = {},
) -> ContentPart:
    """只保存调用账指针与协议重放所需事实，计费数据仍由 Model store 拥有。"""
    if response.call_record_id is None:
        raise ValueError("模型响应尚未结算调用记录")
    indices = tuple(call_indices)
    if len(indices) != len(response.tool_calls) or len(set(indices)) != len(indices):
        raise ValueError("模型工具调用与 Output 位置不匹配")
    if any(type(index) is not int or index < 0 for index in indices):
        raise ValueError("模型工具调用位置必须是非负整数")
    continuation = response.continuation
    return ContentPart(
        "model.facts",
        {
            "call_record_id": response.call_record_id,
            "tool_ids": {
                str(index): call.id for index, call in zip(indices, response.tool_calls)
            },
            "wire_tool_calls": wire_tool_calls,
            "reminder": reminder,
            "thinking": response.thinking,
            "continuation": (
                None
                if continuation is None
                else {
                    "binding_id": continuation.binding_id,
                    "payload": continuation.payload,
                }
            ),
        },
    )


def check_tool_rejection(part: ContentPart) -> ContentReferences:
    """模型协议拒绝只保存原始请求与错误，不引用 binding 或工具效果。"""
    value = part.value
    if (
        not isinstance(value, Mapping)
        or set(value) != {"name", "arguments", "error"}
        or not isinstance(value["name"], str) or not value["name"]
        or not isinstance(value["arguments"], Mapping)
        or not isinstance(value["error"], str) or not value["error"]
    ):
        raise ValueError("模型工具协议拒绝字段无效")
    return ContentReferences()


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


class MessageProjection:
    """Model 的只读历史投影；不读写会话、不执行工具，也不调用模型。"""

    def __init__(
        self,
        model: BoundChatModel,
        *,
        source: str,
        render_content: ContentRenderer,
        tool_name: Callable[[str], str],
        read_call: CallReader,
        keep_input_ids: tuple[str, ...] = (),
    ):
        self._model = model
        self._source = source
        self._render_content = render_content
        self._tool_name = tool_name
        self._read_call = read_call
        self._keep_input_ids = keep_input_ids
        self._last_rows: tuple[Mapping[str, Any], ...] = ()

    @property
    def context_window(self) -> int | None:
        return self._model.descriptor.capabilities.context_window

    @property
    def max_tool_schemas(self) -> int | None:
        return self._model.max_tool_schemas

    def estimate(self, request: ModelRequest) -> int:
        return self._model.estimate_context_tokens(request.messages, request.tools)

    def facts(
        self,
        response: LLMResponse,
        call_indices: Sequence[int],
        *,
        reminder: str | None = None,
        actual_calls: Sequence[ToolCall | ContentPart] | None = None,
    ) -> ContentPart:
        """只为当前模型已成功结算的响应生成可持久 replay 内容。"""
        if actual_calls is not None and len(actual_calls) != len(response.tool_calls):
            raise ValueError("模型 wire 调用与实际 ToolCall 数量不匹配")
        wire: dict[str, Mapping[str, object]] = {}
        for index, original, actual in zip(
            call_indices,
            response.tool_calls,
            () if actual_calls is None else actual_calls,
        ):
            if isinstance(actual, ContentPart):
                _ = check_tool_rejection(actual)
                continue
            actual_name = self._tool_name(actual.binding_id)
            if (
                original.name != actual_name
                or json_value(original.arguments) != json_value(actual.arguments)
            ):
                wire[str(index)] = {
                    "name": original.name,
                    "arguments": original.arguments,
                }
        facts = response_facts(
            response,
            call_indices,
            reminder=reminder,
            wire_tool_calls=wire,
        )
        assert response.call_record_id is not None
        receipt = self._read_call(response.call_record_id)
        if receipt["state"] != "success" or (
            receipt["binding"]["binding_id"] != self._model.descriptor.binding_id
        ):
            raise ValueError("模型响应不属于当前已结算调用")
        return facts

    def render(self, messages: tuple[Message, ...], *, after_seq: int,
               summary_reference: str | None = None, fresh: bool = False) -> ModelRequest:
        """按日志重建协议；交错输入保留，工具观察只在请求视图中与调用成组。"""
        # 当前工作输入由 Turn owner 选定；摘要只替换历史，不吞掉本次要求。
        keep = set(self._keep_input_ids)
        if len(keep) != len(self._keep_input_ids) or keep != {
            message.message_id for message in messages
            if message.message_id in keep and isinstance(message.body, Input)
            and message.source == self._source
        }:
            raise ValueError("保留输入必须是当前来源的真实 Input，且不能重复")
        # 放弃只撤销未结束前缀的执行协议；可读正文仍属于聊天历史。
        pending: dict[str, list[Message]] = {}
        abandoned: set[str] = set()
        abandoned_calls: set[CallRef] = set()
        for message in messages:
            body = message.body
            if isinstance(body, Output):
                if body.finish == "continue":
                    pending.setdefault(message.source, []).append(message)
                else:
                    pending[message.source] = []
            elif isinstance(body, Control) and body.action == "abandon":
                outputs = pending.get(message.source, [])
                for output in outputs:
                    if output.seq <= body.through_seq:
                        abandoned.add(output.message_id)
                        assert isinstance(output.body, Output)
                        abandoned_calls.update(
                            CallRef(output.message_id, index)
                            for index, part in enumerate(output.body.parts)
                            if isinstance(part, ToolCall)
                        )
                pending[message.source] = [output for output in outputs if output.seq > body.through_seq]
        # 1. 读取完整前缀的 replay facts，摘要不能删除 provider 仍需要的状态。
        facts: dict[str, Mapping[str, Any]] = {}
        continuation: ModelContinuation | None = None
        continuation_summary: str | None = None
        continuation_seq = -1
        results: dict[CallRef, Message] = {}
        recorded_facts: dict[str, Mapping[str, Any]] = {}
        for message in messages:
            if not isinstance(message.body, Output) or message.message_id in abandoned:
                continue
            recorded = [part for part in message.body.parts
                        if isinstance(part, ContentPart) and part.kind == "model.facts"]
            if len(recorded) > 1:
                raise ValueError("同一 Output 出现多个 model.facts")
            if recorded:
                _ = check_facts(recorded[0])
                recorded_facts[message.message_id] = cast(Mapping[str, Any], recorded[0].value)
        # 调用账 owner 批量读取窄字段；插件自带 reader 保持原调用合同。
        read_call = self._read_call
        if isinstance(read_call, ModelCallReader):
            receipts = read_call.replay(tuple(value["call_record_id"] for value in recorded_facts.values()))
            read_call = receipts.__getitem__
        for message in messages:
            body = message.body
            if isinstance(body, Control) and body.action == "abandon" and message.source == self._source:
                continuation = None
            if isinstance(body, ToolResult):
                if body.call_ref in abandoned_calls:
                    continue
                if body.call_ref in results:
                    raise ValueError("同一工具调用出现多个结果")
                results[body.call_ref] = message
            if not isinstance(body, Output) or message.message_id in abandoned:
                continue
            value = recorded_facts.get(message.message_id)
            if value is None:
                continue
            receipt = read_call(value["call_record_id"])
            if receipt["state"] != "success":
                raise ValueError("已提交模型事实必须引用成功结算的真实调用")
            indices = {
                str(index)
                for index, part in enumerate(body.parts)
                if isinstance(part, ToolCall) or (isinstance(part, ContentPart) and part.kind == "model.tool_rejection")
            }
            if set(value["tool_ids"]) != indices:
                raise ValueError("模型工具 ID 不匹配实际 Output 调用位置")
            state = value["continuation"]
            message_continuation = (
                None
                if state is None
                else ModelContinuation(state["binding_id"], state["payload"])
            )
            if (
                message_continuation is not None
                and message_continuation.binding_id != receipt["binding"]["binding_id"]
            ):
                raise ValueError("continuation 不属于记录中的模型")
            if message.source == self._source:
                continuation = message_continuation
                continuation_seq = message.seq
                summaries = [
                    part for part in body.parts
                    if isinstance(part, ContentPart) and part.kind == "context.summary"
                ]
                if len(summaries) > 1:
                    raise ValueError("同一模型 Output 只能使用一份摘要")
                continuation_summary = (
                    check_summary(summaries[0]).binding_ids[0] if summaries else None
                )
            facts[message.message_id] = value
        # 摘要明确开启新请求；原 opaque 保存在日志，只续接同一摘要后的响应。
        if fresh or summary_reference is not None and (
            continuation_summary != summary_reference or continuation_seq <= after_seq
        ):
            continuation = None
        if continuation is not None:
            if continuation.binding_id != self._model.descriptor.binding_id:
                raise ValueError("当前模型不能接续另一 binding 的 opaque 状态")
            if after_seq >= 0 and summary_reference is None:
                raise ValueError(
                    "当前投影不能证明摘要与 opaque continuation 可共同重放"
                )

        # 2. 只在请求中调整 call/result 邻接顺序，不产生新消息或伪造观察。
        rows: list[Mapping[str, Any]] = []
        used_results: set[str] = set()
        for message in messages:
            if message.seq <= after_seq and message.message_id not in keep:
                continue
            body = message.body
            if isinstance(body, (Control, ToolResult)):
                continue
            blocks: list[Mapping[str, Any]] = []
            calls: list[dict[str, Any]] = []
            observations: list[Mapping[str, Any]] = []
            model_facts = facts.get(message.message_id)
            if model_facts is not None and model_facts.get("reminder") is not None:
                rows.append({"role": "user", "content": model_facts["reminder"]})
            for index, part in enumerate(body.parts):
                if isinstance(part, ContentPart):
                    if part.kind == "model.tool_rejection":
                        _ = check_tool_rejection(part)
                        if message.message_id in abandoned:
                            continue
                        if model_facts is None:
                            raise ValueError("模型协议拒绝缺少 model.facts")
                        identity = model_facts["tool_ids"][str(index)]
                        rejected = cast(Mapping[str, Any], part.value)
                        calls.append({
                            "id": identity, "type": "function",
                            "function": {"name": rejected["name"], "arguments": json.dumps(
                                json_value(rejected["arguments"]), ensure_ascii=False, separators=(",", ":"),
                            )},
                        })
                        observations.append({"role": "tool", "tool_call_id": identity, "content": [
                            {"type": "text", "text": "调用未执行：" + rejected["error"]},
                        ]})
                    elif part.kind != "model.facts":
                        blocks.extend(self._render_content(part))
                    continue
                if message.message_id in abandoned:
                    continue
                ref = CallRef(message.message_id, index)
                identity = (
                    model_facts["tool_ids"][str(index)]
                    if model_facts is not None
                    else "call_"
                    + hashlib.sha256(
                        json.dumps([ref.message_id, ref.part_index]).encode()
                    ).hexdigest()[:32]
                )
                wire = None if model_facts is None else model_facts.get("wire_tool_calls")
                raw_call = None if wire is None else wire.get(str(index))
                calls.append({
                    "id": identity,
                    "type": "function",
                    "function": {
                        "name": (
                            self._tool_name(part.binding_id)
                            if raw_call is None
                            else raw_call["name"]
                        ),
                        "arguments": json.dumps(
                            json_value(
                                part.arguments
                                if raw_call is None
                                else raw_call["arguments"]
                            ),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                })
                observation = results.get(ref)
                if observation is None:
                    raise ValueError("模型请求包含未结算的工具调用")
                if observation.seq <= message.seq:
                    raise ValueError("工具结果不能早于调用")
                result = cast(ToolResult, observation.body)
                result_blocks: list[Mapping[str, Any]] = []
                if result.outcome != "success":
                    status = f"工具状态: {result.outcome}"
                    if result.outcome in {"error", "interrupted"}:
                        status += "。原调用可能已经产生效果；先检查当前状态，再决定下一步，不要直接重复执行原操作。"
                    result_blocks.append({"type": "text", "text": status})
                for item in result.parts:
                    result_blocks.extend(self._render_content(item))
                observations.append(
                    {"role": "tool", "tool_call_id": identity, "content": result_blocks}
                )
                used_results.add(observation.message_id)
            if blocks or calls:
                row: dict[str, Any] = {
                    "role": "user" if isinstance(body, Input) else "assistant",
                    "content": blocks,
                }
                if calls:
                    row["tool_calls"] = calls
                if model_facts is not None and model_facts["thinking"] is not None:
                    row["reasoning_content"] = model_facts["thinking"]
                rows.append(row)
                rows.extend(observations)
        if any(
            message.seq > after_seq and message.message_id not in used_results
            for message in results.values()
        ):
            raise ValueError("工具结果缺少本次视图中的真实调用")
        # 本轮仍重读账本和渲染动态内容；值未变的行复用已冻结表示。
        prior = self._last_rows
        rows = [prior[index] if index < len(prior) and _same_json(row, prior[index]) else row
                for index, row in enumerate(rows)]
        request = ModelRequest(messages=rows, continuation=continuation)
        self._last_rows = tuple(request.messages)
        return request


def _same_json(value: Any, saved: Any) -> bool:
    """与已冻结 JSON 比较；数组忽略容器形式，标量保留准确类型。"""
    if value is saved:
        return True
    if isinstance(saved, Mapping):
        if not isinstance(value, Mapping):
            return False
        current = cast(Mapping[object, Any], value)
        previous = cast(Mapping[str, Any], saved)
        return (len(current) == len(previous)
                and all(isinstance(key, str) and key in previous and _same_json(item, previous[key])
                        for key, item in current.items()))
    if isinstance(saved, tuple):
        if not isinstance(value, (list, tuple)):
            return False
        items = cast(Sequence[Any], value)
        old_items = cast(tuple[Any, ...], saved)
        return len(items) == len(old_items) and all(
            _same_json(item, old) for item, old in zip(items, old_items))
    return type(value) is type(saved) and value == saved
