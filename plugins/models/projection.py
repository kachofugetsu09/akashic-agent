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
from session.message import (
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

ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]
CallReader = Callable[[str], Mapping[str, Any]]
MODEL_CALLS = ServiceKey[CallReader]("models.calls.v1")


def response_facts(response: LLMResponse, call_indices: Sequence[int]) -> ContentPart:
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


def check_facts(part: ContentPart) -> ContentReferences:
    """验证存储边界的 replay 数据；它不能包含可执行消息或角色声明。"""
    value = part.value
    if not isinstance(value, Mapping):
        raise ValueError("model.facts 必须是对象")
    value = cast(Mapping[str, object], value)
    if set(value) != {"call_record_id", "tool_ids", "thinking", "continuation"}:
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

    @property
    def context_window(self) -> int | None:
        return self._model.descriptor.capabilities.context_window

    @property
    def max_tool_schemas(self) -> int | None:
        return self._model.max_tool_schemas

    def estimate(self, request: ModelRequest) -> int:
        return self._model.estimate_context_tokens(request.messages, request.tools)

    def facts(self, response: LLMResponse, call_indices: Sequence[int]) -> ContentPart:
        """只为当前模型已成功结算的响应生成可持久 replay 内容。"""
        facts = response_facts(response, call_indices)
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
            recorded = [
                part
                for part in body.parts
                if isinstance(part, ContentPart) and part.kind == "model.facts"
            ]
            if len(recorded) > 1:
                raise ValueError("同一 Output 出现多个 model.facts")
            if not recorded:
                continue
            _ = check_facts(recorded[0])
            value = cast(Mapping[str, Any], recorded[0].value)
            receipt = self._read_call(value["call_record_id"])
            if receipt["state"] != "success":
                raise ValueError("已提交模型事实必须引用成功结算的真实调用")
            indices = {
                str(index)
                for index, part in enumerate(body.parts)
                if isinstance(part, ToolCall)
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
            for index, part in enumerate(body.parts):
                if isinstance(part, ContentPart):
                    if part.kind != "model.facts":
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
                calls.append(
                    {
                        "id": identity,
                        "type": "function",
                        "function": {
                            "name": self._tool_name(part.binding_id),
                            "arguments": json.dumps(
                                json_value(part.arguments),
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                    }
                )
                observation = results.get(ref)
                if observation is None:
                    raise ValueError("模型请求包含未结算的工具调用")
                if observation.seq <= message.seq:
                    raise ValueError("工具结果不能早于调用")
                result = cast(ToolResult, observation.body)
                result_blocks: list[Mapping[str, Any]] = []
                if result.outcome != "success":
                    result_blocks.append(
                        {"type": "text", "text": f"工具状态: {result.outcome}"}
                    )
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
        return ModelRequest(messages=rows, continuation=continuation)
