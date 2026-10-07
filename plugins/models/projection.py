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
    read_content_refs,
)
from agent.plugin_contracts import (
    CallRef,
    ContentPart,
    ContentReferences,
    Control,
    Input,
    Message,
    Output,
    ToolCall,
    ToolResult,
    json_value,
)
from agent.plugin_contracts.models import (
    PrepareContent,
    RenderedContent,
    MODEL_CALLS as MODEL_CALLS,
    MODEL_CHECKS as MODEL_MESSAGE_CHECKS,  # noqa: F401 - 显式再导出给本插件消费者。
    MODEL_PROJECTION as MODEL_PROJECTION,
)

from .store import ModelCallReader

ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]
CallReader = Callable[[str], Mapping[str, Any]]
ContentCheck = Callable[[ContentPart], ContentReferences]
DisplayRenderer = Callable[[ContentPart], Mapping[str, object]]
MODEL_CALL_HISTORY = ServiceKey[Callable[[str, int], tuple[Mapping[str, Any], ...]]](
    "models.call-history.v1"
)
MODEL_DISPLAY = ServiceKey[DisplayRenderer]("message.display:model.facts")


def response_facts(
    response: LLMResponse,
    call_indices: Sequence[int],
    *,
    reminder: str | None = None,
    reminder_input_id: str | None = None,
    wire_tool_calls: Mapping[str, Mapping[str, object]] = {},
    content_refs: tuple[tuple[str, int], ...] = (),
    content_transformed: bool = False,
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
    if reminder_input_id is not None and reminder is None:
        raise ValueError("reminder Input 只能标记实际保存的 reminder")
    facts: dict[str, object] = {
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
    }
    if content_refs:
        facts["content_refs"] = content_refs
    if content_transformed:
        facts["content_transformed"] = True
    if reminder_input_id is not None:
        if not reminder_input_id:
            raise ValueError("reminder Input 身份不能为空")
        assert reminder is not None
        facts["reminder_input_id"] = reminder_input_id
        facts["reminder_sha256"] = hashlib.sha256(reminder.encode("utf-8")).hexdigest()
    return ContentPart("model.facts", facts)


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
    reminder_identity_fields = {"reminder_input_id", "reminder_sha256"}
    if set(value) - {"content_refs", "content_transformed"} not in (old_fields, new_fields, new_fields | reminder_identity_fields):
        raise ValueError("model.facts 字段无效")
    if not isinstance(value["call_record_id"], str) or not value["call_record_id"]:
        raise ValueError("model.facts 缺少调用记录")
    if "content_refs" in value:
        _ = read_content_refs(value["content_refs"])
    if "content_transformed" in value and type(value["content_transformed"]) is not bool:
        raise ValueError("内容投影标记必须是 bool")
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
    if "reminder_input_id" in value:
        reminder = value["reminder"]
        input_id = value["reminder_input_id"]
        digest = value["reminder_sha256"]
        if (
            not isinstance(reminder, str)
            or not isinstance(input_id, str)
            or not input_id
            or not isinstance(digest, str)
            or len(digest) != 64
            or hashlib.sha256(reminder.encode("utf-8")).hexdigest() != digest
        ):
            raise ValueError("模型请求 reminder 身份无效")
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
        check_summary: ContentCheck,
        keep_input_ids: tuple[str, ...] = (),
        prepare_content: PrepareContent | None = None,
        tool_names: frozenset[str] = frozenset(),
    ):
        self._model = model
        self._source = source
        self._render_content = render_content
        self._tool_name = tool_name
        self._read_call = read_call
        self._check_summary = check_summary
        self._keep_input_ids = keep_input_ids
        self._prepare_content = prepare_content
        self._tool_names = tool_names
        self._last_rows: tuple[Mapping[str, Any], ...] = ()
        self._facts: dict[str, tuple[Message, Mapping[str, Any] | None]] = {}
        self._arguments: dict[int, tuple[Mapping[str, Any], str]] = {}
        self._segments: dict[str, _Segment] = {}

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
        reminder_input_id: str | None = None,
        actual_calls: Sequence[ToolCall | ContentPart] | None = None,
        content_refs: tuple[tuple[str, int], ...] = (),
        content_transformed: bool = False,
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
            reminder_input_id=reminder_input_id,
            wire_tool_calls=wire,
            content_refs=content_refs,
            content_transformed=content_transformed,
        )
        assert response.call_record_id is not None
        receipt = self._read_call(response.call_record_id)
        if receipt["state"] != "success" or (
            receipt["binding"]["binding_id"] != self._model.descriptor.binding_id
        ):
            raise ValueError("模型响应不属于当前已结算调用")
        return facts

    def render(
        self,
        messages: tuple[Message, ...],
        *,
        after_seq: int,
        summary_reference: str | None = None,
        fresh: bool = False,
        current_reminder: str | None = None,
        current_reminder_input_id: str | None = None,
        current_context: str | None = None,
    ) -> ModelRequest:
        """按日志重建协议；交错输入保留，工具观察只在请求视图中与调用成组。"""
        # 当前工作输入由 Turn owner 选定；摘要只替换历史，不吞掉本次要求。
        keep = set(self._keep_input_ids)
        if len(keep) != len(self._keep_input_ids) or keep != {
            message.message_id for message in messages
            if message.message_id in keep and isinstance(message.body, Input)
            and message.source == self._source
        }:
            raise ValueError("保留输入必须是当前来源的真实 Input，且不能重复")
        if (current_reminder is None) != (current_reminder_input_id is None):
            raise ValueError("当前 reminder 与 Input 身份必须同时提供")
        latest_input = next(
            (message.message_id for message in reversed(messages)
             if message.source == self._source and isinstance(message.body, Input)),
            None,
        )
        current_reminder_identity: tuple[str, str] | None = None
        if current_reminder_input_id is not None:
            if current_reminder_input_id != latest_input:
                raise ValueError("当前 reminder 必须属于本来源最新 Input")
            assert current_reminder is not None
            current_reminder_identity = (
                current_reminder_input_id,
                hashlib.sha256(current_reminder.encode("utf-8")).hexdigest(),
            )
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
        continuation_transformed = False
        results: dict[CallRef, Message] = {}
        # 只复用同一不可变 Message 的静态检查；调用账和内容贡献者仍每轮读取。
        current_facts: dict[str, tuple[Message, Mapping[str, Any] | None]] = {}
        recorded_facts: dict[str, Mapping[str, Any]] = {}
        response_metadata: dict[str, Mapping[str, Any]] = {}
        for message in messages:
            if not isinstance(message.body, Output):
                continue
            previous = self._facts.get(message.message_id)
            if previous is not None and previous[0] is message:
                value = previous[1]
            else:
                recorded = [part for part in message.body.parts
                            if isinstance(part, ContentPart) and part.kind == "model.facts"]
                if len(recorded) > 1:
                    raise ValueError("同一 Output 出现多个 model.facts")
                value = None
                if recorded:
                    _ = check_facts(recorded[0])
                    value = cast(Mapping[str, Any], recorded[0].value)
            current_facts[message.message_id] = (message, value)
            if value is not None:
                recorded_facts[message.message_id] = value
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
                # 已放弃调用的结算回执仍要配对渲染；call/result 邻接关系不能断。
                if body.call_ref in results:
                    raise ValueError("同一工具调用出现多个结果")
                results[body.call_ref] = message
            if not isinstance(body, Output):
                continue
            value = recorded_facts.get(message.message_id)
            if value is None:
                continue
            receipt = read_call(value["call_record_id"])
            if receipt["state"] != "success":
                raise ValueError("已提交模型事实必须引用成功结算的真实调用")
            if receipt["binding"]["binding_id"] == self._model.descriptor.binding_id:
                metadata = receipt.get("provider_metadata")
                if metadata is not None and not value.get("content_transformed", False):
                    response_metadata[message.message_id] = metadata
            previous = self._facts.get(message.message_id)
            if previous is None or previous[0] is not message:
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
            if message.source == self._source and message.message_id not in abandoned:
                continuation = message_continuation
                continuation_transformed = value.get("content_transformed", False)
                continuation_seq = message.seq
                summaries = [
                    part for part in body.parts
                    if isinstance(part, ContentPart) and part.kind == "context.summary"
                ]
                if len(summaries) > 1:
                    raise ValueError("同一模型 Output 只能使用一份摘要")
                continuation_summary = (
                    self._check_summary(summaries[0]).binding_ids[0] if summaries else None
                )
            facts[message.message_id] = value
        # 摘要明确开启新请求；原 opaque 保存在日志，只续接同一摘要后的响应。
        if fresh or continuation_transformed or summary_reference is not None and (
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

        # 2. 只投影实际进入请求的块；首次完整展示证据跟随请求，而非 render 调用。
        seen = {
            tuple(ref) for message in messages if message.source == self._source
            for ref in recorded_facts.get(message.message_id, {}).get("content_refs", ())
        }
        content_refs: list[tuple[str, int]] = []
        changed_content = False
        transform = (None if self._prepare_content is None else
                     self._prepare_content(messages, self._source, self._tool_names, frozenset(seen)))

        def render(message: Message, index: int,
                   out_refs: list[tuple[str, int]]) -> tuple[Mapping[str, Any], ...]:
            nonlocal changed_content
            assert not isinstance(message.body, Control)
            part = message.body.parts[index]
            assert isinstance(part, ContentPart)
            rendered = None if transform is None else transform(message, index)
            if rendered is not None:
                changed_content = True
            if rendered is None:
                blocks = tuple(self._render_content(part))
                rendered = RenderedContent(blocks, complete=(
                    part.kind == "text" and blocks == ({"type": "text", "text": part.value},)
                ))
            ref = (message.message_id, index)
            if rendered.blocks and rendered.complete and ref not in seen:
                out_refs.append(ref)
                seen.add(ref)
            return rendered.blocks

        current_arguments: dict[int, tuple[Mapping[str, Any], str]] = {}

        def encode_arguments(arguments: Mapping[str, Any]) -> str:
            """参数来自深冻结的消息，只保留当前窗口实际使用的编码。"""
            previous = self._arguments.get(id(arguments))
            encoded = (previous[1] if previous is not None and previous[0] is arguments else
                       json.dumps(json_value(arguments), ensure_ascii=False, separators=(",", ":")))
            current_arguments[id(arguments)] = (arguments, encoded)
            return encoded

        # 3. 只在请求中调整 call/result 邻接顺序，不产生新消息或伪造观察。
        rows: list[Mapping[str, Any]] = []
        used_results: set[str] = set()
        replayed_reminders: set[tuple[str, str]] = set()
        context_added = False
        # 静态内容下按消息缓存渲染分段：不可变前缀既不重建也不深比较。
        # 动态视图、artifact/reply 引用、当前输入相关的消息仍每轮重建。
        static_content = transform is None
        cached_segments = self._segments if static_content else {}
        new_segments: dict[str, _Segment] = {}
        for message in messages:
            if message.seq <= after_seq and message.message_id not in keep:
                continue
            body = message.body
            if isinstance(body, (Control, ToolResult)):
                continue
            model_facts = facts.get(message.message_id)
            reminder_identity = (
                (
                    (cast(str, model_facts["reminder_input_id"]),
                     cast(str, model_facts["reminder_sha256"]))
                    if "reminder_input_id" in model_facts
                    else None
                )
                if model_facts is not None and model_facts.get("reminder") is not None
                else None
            )
            current_touched = (
                message.message_id == latest_input
                or (
                    reminder_identity is not None
                    and current_reminder_identity is not None
                    and reminder_identity == current_reminder_identity
                    and current_context is not None
                )
            )
            call_refs = tuple(
                CallRef(message.message_id, index)
                for index, part in enumerate(body.parts)
                if isinstance(part, ToolCall)
            )
            call_deps = tuple(
                (ref, id(results.get(ref)), ref in abandoned_calls)
                for ref in call_refs
            )
            entry = None if current_touched else cached_segments.get(message.message_id)
            if (
                entry is not None
                and entry.message is message
                and entry.facts is model_facts
                and entry.has_metadata == (message.message_id in response_metadata)
                and entry.call_deps == call_deps
            ):
                rows.extend(entry.rows)
                for ref in entry.new_refs:
                    if ref not in seen:
                        content_refs.append(ref)
                        seen.add(ref)
                used_results.update(entry.used)
                replayed_reminders.update(entry.reminders)
                new_segments[message.message_id] = entry
                continue
            msg_rows: list[Mapping[str, Any]] = []
            msg_refs: list[tuple[str, int]] = []
            msg_used: list[str] = []
            msg_reminders: list[tuple[str, str]] = []
            blocks: list[Mapping[str, Any]] = []
            calls: list[dict[str, Any]] = []
            observations: list[Mapping[str, Any]] = []
            if reminder_identity is not None or (
                model_facts is not None and model_facts.get("reminder") is not None
            ):
                if (
                    reminder_identity is None
                    or reminder_identity not in replayed_reminders
                ):
                    msg_rows.append({"role": "user", "content": model_facts["reminder"]})
                    if reminder_identity is not None:
                        replayed_reminders.add(reminder_identity)
                        msg_reminders.append(reminder_identity)
                    if (current_reminder_identity is not None
                            and reminder_identity == current_reminder_identity and current_context is not None):
                        msg_rows.append({"role": "user", "content": current_context})
                        context_added = True
            for index, part in enumerate(body.parts):
                if isinstance(part, ContentPart):
                    if part.kind == "model.tool_rejection":
                        _ = check_tool_rejection(part)
                        if model_facts is None:
                            raise ValueError("模型协议拒绝缺少 model.facts")
                        identity = model_facts["tool_ids"][str(index)]
                        rejected = cast(Mapping[str, Any], part.value)
                        calls.append({
                            "id": identity, "type": "function",
                            "function": {"name": rejected["name"], "arguments": encode_arguments(rejected["arguments"])},
                        })
                        observations.append({"role": "tool", "tool_call_id": identity, "content": [
                            {"type": "text", "text": "调用未执行：" + rejected["error"]},
                        ]})
                    elif part.kind != "model.facts":
                        blocks.extend(render(message, index, msg_refs))
                    continue
                ref = CallRef(message.message_id, index)
                if ref in abandoned_calls:
                    # 放弃前缀的调用不进入 wire 协议；已有耐久回执时如实保留
                    # 真实状态与内容，只有无回执时才说明效果未知。
                    observation = results.get(ref)
                    if observation is not None:
                        used_results.add(observation.message_id)
                        msg_used.append(observation.message_id)
                        settled = cast(ToolResult, observation.body)
                        if settled.outcome == "denied":
                            blocks.append({"type": "text", "text": (
                                "一次工具调用随来源前缀放弃，结算为 denied："
                                "工具未启动，未产生外部效果。"
                            )})
                        elif settled.outcome == "success":
                            blocks.append({"type": "text", "text": (
                                "一次工具调用在放弃前已完成，真实结果如下；"
                                "外部效果已经发生。"
                            )})
                            for item_index, _item in enumerate(settled.parts):
                                blocks.extend(render(observation, item_index, msg_refs))
                        else:
                            blocks.append({"type": "text", "text": (
                                f"一次工具调用随来源前缀放弃，结算为 {settled.outcome}："
                                "外部效果可能已经发生，不能据此重跑。"
                            )})
                            for item_index, _item in enumerate(settled.parts):
                                blocks.extend(render(observation, item_index, msg_refs))
                    else:
                        blocks.append({"type": "text", "text": (
                            "一次工具调用随来源前缀放弃而中断；外部效果未结算，状态未知，"
                            "不能据此重跑。"
                        )})
                    continue
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
                        "arguments": encode_arguments(
                            part.arguments if raw_call is None else raw_call["arguments"],
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
                for item_index, _item in enumerate(result.parts):
                    result_blocks.extend(render(observation, item_index, msg_refs))
                observations.append(
                    {"role": "tool", "tool_call_id": identity, "content": result_blocks}
                )
                used_results.add(observation.message_id)
                msg_used.append(observation.message_id)
            if blocks or calls:
                row: dict[str, Any] = {
                    "role": "user" if isinstance(body, Input) else "assistant",
                    "content": blocks,
                }
                if calls:
                    row["tool_calls"] = calls
                if (message.message_id in response_metadata and model_facts is not None
                        and len(calls) == len(model_facts["tool_ids"])):
                    row["provider_metadata"] = response_metadata[message.message_id]
                if model_facts is not None and model_facts["thinking"] is not None:
                    row["reasoning_content"] = model_facts["thinking"]
                msg_rows.append(row)
                msg_rows.extend(observations)
            if (current_reminder is None and current_context is not None
                    and message.message_id == latest_input):
                msg_rows.append({"role": "user", "content": current_context})
                context_added = True
            rows.extend(msg_rows)
            content_refs.extend(msg_refs)
            if (
                static_content
                and not current_touched
                and _static_parts(message)
                and all(
                    _static_parts(observation)
                    for ref in call_refs
                    if (observation := results.get(ref)) is not None
                )
            ):
                new_segments[message.message_id] = _Segment(
                    message=message,
                    facts=model_facts,
                    has_metadata=message.message_id in response_metadata,
                    call_deps=call_deps,
                    rows=tuple(msg_rows),
                    new_refs=tuple(msg_refs),
                    used=tuple(msg_used),
                    reminders=tuple(msg_reminders),
                )
        self._segments = new_segments
        # 首次使用和变化后的材料追加；成功 Output 的既有事实固定后续回放位置。
        if current_reminder is not None and current_reminder_identity not in replayed_reminders:
            rows.append({"role": "user", "content": current_reminder})
        if current_context is not None and not context_added:
            rows.append({"role": "user", "content": current_context})
        if any(
            message.seq > after_seq
            and message.message_id not in used_results
            and cast(ToolResult, message.body).call_ref not in abandoned_calls
            for message in results.values()
        ):
            raise ValueError("工具结果缺少本次视图中的真实调用")
        # 本轮仍重读账本和渲染动态内容；值未变的行复用已冻结表示。
        prior = self._last_rows
        rows = [prior[index] if index < len(prior) and _same_json(row, prior[index]) else row
                for index, row in enumerate(rows)]
        # 内容视图变化后从完整投影开始，不混用仍保留旧正文的 opaque 会话。
        request = ModelRequest(messages=rows, continuation=None if changed_content else continuation,
                               content_refs=tuple(content_refs), content_transformed=changed_content)
        self._last_rows = tuple(request.messages)
        self._facts = current_facts
        self._arguments = current_arguments
        return request


class _Segment:
    """一条消息在静态内容下渲染出的 wire 行分段；deps 未命中即整体重建。"""

    __slots__ = (
        "message", "facts", "has_metadata", "call_deps",
        "rows", "new_refs", "used", "reminders",
    )

    def __init__(
        self,
        *,
        message: Message,
        facts: Mapping[str, Any] | None,
        has_metadata: bool,
        call_deps: tuple[tuple[CallRef, int, bool], ...],
        rows: tuple[Mapping[str, Any], ...],
        new_refs: tuple[tuple[str, int], ...],
        used: tuple[str, ...],
        reminders: tuple[tuple[str, str], ...],
    ) -> None:
        self.message = message
        self.facts = facts
        self.has_metadata = has_metadata
        self.call_deps = call_deps
        self.rows = rows
        self.new_refs = new_refs
        self.used = used
        self.reminders = reminders


def _static_parts(message: Message) -> bool:
    """artifact/reply 引用按渲染时的外部状态解析，不参与分段缓存。"""
    return all(
        not isinstance(part, ContentPart) or part.kind not in ("artifact_ref", "reply_ref")
        for part in message.body.parts
    )


def _same_json(value: Any, saved: Any) -> bool:
    """与已冻结 JSON 比较；数组忽略容器形式，标量保留准确类型。"""
    if value is saved:
        return True
    # 冻结行只含 JSON 值；常见标量不需要执行容器的 ABC 查询。
    saved_type = type(saved)
    if saved_type in (str, int, float, bool, type(None)):
        return type(value) is saved_type and value == saved
    if isinstance(saved, dict):
        if not isinstance(value, dict) and not isinstance(value, Mapping):
            return False
        if len(value) != len(saved):
            return False
        for key, item in value.items():
            if not isinstance(key, str) or key not in saved or not _same_json(item, saved[key]):
                return False
        return True
    if isinstance(saved, tuple):
        if not isinstance(value, (list, tuple)) or len(value) != len(saved):
            return False
        for item, old in zip(value, saved):
            if not _same_json(item, old):
                return False
        return True
    return type(value) is saved_type and value == saved



class ProjectionOwner:
    """直接签发模型 owner 的投影，不建立第二份请求状态。"""

    create = staticmethod(MessageProjection)


class MessageChecksOwner:
    check_facts = staticmethod(check_facts)
    check_tool_rejection = staticmethod(check_tool_rejection)
