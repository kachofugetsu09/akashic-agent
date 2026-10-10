from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import Any, cast

from agent.plugin_composition import ServiceKey
from agent.plugin_composition.messages import MessageSnapshot
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
from plugins.models.contract import (
    ContentTransform,
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
        dynamic_content_kinds: frozenset[str] | None = frozenset(),
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
        self._dynamic_content_kinds = dynamic_content_kinds
        self._last_estimate: tuple[ModelRequest, int] | None = None
        self._facts: dict[str, tuple[Message, Mapping[str, Any] | None]] = {}
        self._view: _RenderView | None = None
        self._fold: _Fold | None = None

    @property
    def context_window(self) -> int | None:
        return self._model.descriptor.capabilities.context_window

    @property
    def max_tool_schemas(self) -> int | None:
        return self._model.max_tool_schemas

    def estimate(self, request: ModelRequest) -> int:
        # 估算是不可变请求的纯函数；构建链会对同一请求对象重复估算。
        saved = self._last_estimate
        if saved is not None and saved[0] is request:
            return saved[1]
        value = self._model.estimate_context_tokens(request.messages, request.tools)
        self._last_estimate = (request, value)
        return value

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
        messages: Sequence[Message],
        *,
        after_seq: int,
        summary_reference: str | None = None,
        fresh: bool = False,
        current_reminder: str | None = None,
        current_reminder_input_id: str | None = None,
        current_context: str | None = None,
    ) -> ModelRequest:
        """按日志重建协议；交错输入保留，工具观察只在请求视图中与调用成组。"""
        keep = set(self._keep_input_ids)
        if (current_reminder is None) != (current_reminder_input_id is None):
            raise ValueError("当前 reminder 与 Input 身份必须同时提供")
        # 编排折叠：不可变前缀的预扫描（abandon、facts、调用账与 continuation）
        # 只在追加尾部增量 apply；前缀身份变化或 abandon 进入尾部时整折叠重建，
        # 重建路径与原逐轮全量扫描逐行等价。参数（after_seq/keep/reminder/summary）
        # 不进折叠——它们只作用于折叠输出之上的每轮后处理。
        inherited = self._facts
        fold = self._fold
        if fold is not None and not _fold_compatible(fold, messages):
            fold = None
        if fold is None:
            fold = _Fold()
            self._fold_rebuild(fold, messages, inherited)
        else:
            try:
                for message in messages[fold.count:]:
                    self._fold_apply(fold, message, inherited)
            except _FoldStale:
                fold = _Fold()
                self._fold_rebuild(fold, messages, inherited)
            else:
                fold.prefix = messages
        self._fold = fold
        # 当前工作输入由 Turn owner 选定；摘要只替换历史，不吞掉本次要求。
        if len(keep) != len(self._keep_input_ids) or not keep <= fold.inputs:
            raise ValueError("保留输入必须是当前来源的真实 Input，且不能重复")
        latest_input = fold.latest_input
        current_reminder_identity: tuple[str, str] | None = None
        if current_reminder_input_id is not None:
            if current_reminder_input_id != latest_input:
                raise ValueError("当前 reminder 必须属于本来源最新 Input")
            assert current_reminder is not None
            current_reminder_identity = (
                current_reminder_input_id,
                hashlib.sha256(current_reminder.encode("utf-8")).hexdigest(),
            )
        continuation = fold.continuation
        continuation_summary = fold.cont_summary
        continuation_seq = fold.cont_seq
        continuation_transformed = fold.cont_transformed
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

        # 2. 渲染读面只重编译变化；失败后丢弃未完成的派生状态，下次仍如实校验。
        try:
            rows, content_refs, changed_content = self._render_rows(
                messages, fold, after_seq=after_seq,
                current_reminder=current_reminder,
                current_reminder_identity=current_reminder_identity,
                current_context=current_context,
            )
            request = ModelRequest(
                messages=rows, continuation=None if changed_content else continuation,
                content_refs=content_refs, content_transformed=changed_content,
            )
        except BaseException:
            self._view = None
            raise
        self._facts = fold.facts
        return request

    def _render_rows(
        self, messages: Sequence[Message], fold: _Fold, *, after_seq: int,
        current_reminder: str | None,
        current_reminder_identity: tuple[str, str] | None,
        current_context: str | None,
    ) -> tuple[list[Mapping[str, Any]], tuple[tuple[str, int], ...], bool]:
        """维护当前窗口的有序分段；日志变化编译正文，实时材料只在装配时插入。"""
        # 1. 窗口或日志前缀改变时重建同一读面，旧工具结果也重新检查可见调用。
        view = self._view
        if view is None or view.fold is not fold or view.after_seq != after_seq:
            view = _RenderView(fold, after_seq)
            self._view = view
        tail = messages[view.count:]
        transform = (None if self._prepare_content is None else self._prepare_content(
            tuple(messages), self._source, self._tool_names, frozenset(fold.content_refs),
        ))
        cache_ok = transform is None or bool(self._dynamic_content_kinds)
        dirty = set(view.dynamic) if cache_ok else set(view.units)
        for message in tail:
            if isinstance(message.body, ToolResult):
                caller = message.body.call_ref.message_id
                if caller in view.units:
                    dirty.add(caller)
        # 动态 renderer 仍按消息顺序调用；旧单元先于新增尾部，不能按 set 顺序执行。
        changed = [view.units[key].message
                   for key in sorted(dirty, key=lambda key: view.units[key].index)]
        changed.extend(message for message in tail
                       if isinstance(message.body, (Input, Output))
                       and (message.seq > after_seq or message.message_id in self._keep_input_ids))

        # 2. 冷构建和追加共用编译操作；一份有序单元表拥有冻结行与首个提醒载体。
        for message in changed:
            previous = view.units.get(message.message_id)
            if previous is None:
                index = len(view.units)
                facts = fold.recorded.get(message.message_id)
                reminder = None if facts is None else facts.get("reminder")
                if reminder is not None and facts is not None and "reminder_input_id" in facts:
                    identity = (cast(str, facts["reminder_input_id"]), cast(str, facts["reminder_sha256"]))
                    if identity in view.reminders:
                        reminder = None
                    else:
                        view.reminders[identity] = message.message_id
            else:
                index, reminder = previous.index, previous.reminder
            unit = self._compile_unit(message, fold, transform, position=index,
                                      reminder=reminder, cache_ok=cache_ok, previous=previous)
            view.units[message.message_id] = unit
            if unit.dynamic:
                view.dynamic.add(message.message_id)
            else:
                view.dynamic.discard(message.message_id)

        # 3. 不变窗口中只检查新结果；窗口变化从 count=0 重建，不能漏掉旧结果变孤儿。
        for message in tail:
            if not isinstance(message.body, ToolResult) or message.seq <= after_seq:
                continue
            ref = message.body.call_ref
            caller = view.units.get(ref.message_id)
            if (ref not in fold.abandoned_calls
                    and (caller is None or message.message_id not in caller.used)):
                raise ValueError("工具结果缺少本次视图中的真实调用")

        # 4. 拼接已编译行与本轮首显证据；完整展示候选与持久 seen 分属不同事实。
        groups: list[tuple[Mapping[str, Any], ...]] = []
        refs: list[tuple[str, int]] = []
        seen = set(fold.content_refs)
        changed_content = False
        for unit in view.units.values():
            groups.append(unit.rows)
            changed_content |= unit.transformed
            for ref in unit.refs:
                if ref not in seen:
                    refs.append(ref)
                    seen.add(ref)
        carrier = (view.reminders.get(current_reminder_identity)
                   if current_reminder_identity is not None else fold.latest_input)
        context_unit = None if carrier is None else view.units.get(carrier)
        if current_context is not None and context_unit is not None:
            position = 1 if current_reminder_identity is not None else len(context_unit.rows)
            unit_rows = context_unit.rows
            groups[context_unit.index] = (*unit_rows[:position],
                {"role": "user", "content": current_context}, *unit_rows[position:])
        rows = list(chain.from_iterable(groups))
        if current_reminder is not None and current_reminder_identity not in view.reminders:
            rows.append({"role": "user", "content": current_reminder})
        if current_context is not None and context_unit is None:
            rows.append({"role": "user", "content": current_context})
        view.count = len(messages)
        return rows, tuple(refs), changed_content

    def _compile_unit(
        self, message: Message, fold: _Fold, transform: ContentTransform | None, *,
        position: int, reminder: str | None, cache_ok: bool, previous: _RenderedMessage | None,
    ) -> _RenderedMessage:
        """把一条可见消息及其工具观察编译成冻结行；不保存实时上下文。"""
        # 1. 转换器每轮由完整历史准备；首显候选在装配时与持久 seen 求差。
        changed_content = False

        def render(message: Message, index: int,
                   out_refs: list[tuple[str, int]]) -> tuple[Mapping[str, Any], ...]:
            nonlocal changed_content
            assert not isinstance(message.body, Control)
            part = message.body.parts[index]
            assert isinstance(part, ContentPart)
            rendered = None if transform is None else transform(message, index)
            if rendered is not None:
                changed_content = True
            else:
                blocks = tuple(self._render_content(part))
                rendered = RenderedContent(blocks, complete=(
                    part.kind == "text" and blocks == ({"type": "text", "text": part.value},)
                ))
            if rendered.blocks and rendered.complete:
                out_refs.append((message.message_id, index))
            return rendered.blocks

        def encode_arguments(arguments: Mapping[str, Any]) -> str:
            return json.dumps(json_value(arguments), ensure_ascii=False, separators=(",", ":"))

        # 2. 现有协议配对与错误保持在唯一编译路径中。
        body = cast(Input | Output, message.body)
        model_facts = fold.recorded.get(message.message_id)
        results = fold.results
        abandoned_calls = fold.abandoned_calls
        response_metadata = fold.response_metadata
        msg_rows: list[Mapping[str, Any]] = []
        msg_refs: list[tuple[str, int]] = []
        msg_used: list[str] = []
        blocks: list[Mapping[str, Any]] = []
        calls: list[dict[str, Any]] = []
        observations: list[Mapping[str, Any]] = []
        if reminder is not None:
            msg_rows.append({"role": "user", "content": reminder})
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

        # 3. 行在此冻结；动态 kind、外部引用与实际转换命中均保留逐轮重算。
        dynamic_kinds = self._dynamic_content_kinds
        dynamic = (not cache_ok or changed_content or not _static_parts(message, dynamic_kinds)
                   or any(not _static_parts(observation, dynamic_kinds)
                          for part_index, part in enumerate(body.parts)
                          if isinstance(part, ToolCall)
                          and (observation := results.get(CallRef(message.message_id, part_index))) is not None))
        # 动态内容值未变时保留其行身份，避免下游重新编码整段；只比较本次重编译的单元。
        if previous is not None:
            prior = previous.rows
            msg_rows = [prior[i] if i < len(prior) and _same_json(row, prior[i]) else row
                        for i, row in enumerate(msg_rows)]
        return _RenderedMessage(
            message, position, reminder, tuple(ModelRequest(messages=msg_rows).messages),
            tuple(msg_refs), tuple(msg_used), changed_content, dynamic,
        )

    def _fold_rebuild(
        self,
        fold: _Fold,
        messages: Sequence[Message],
        inherited: Mapping[str, tuple[Message, Mapping[str, Any] | None]],
    ) -> None:
        """预扫描的全量形式；与原逐轮三趟扫描逐行等价。"""
        # 放弃只撤销未结束前缀的执行协议；可读正文仍属于聊天历史。
        pending: dict[str, list[Message]] = {}
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
                        fold.abandoned.add(output.message_id)
                        assert isinstance(output.body, Output)
                        fold.abandoned_calls.update(
                            CallRef(output.message_id, index)
                            for index, part in enumerate(output.body.parts)
                            if isinstance(part, ToolCall)
                        )
                pending[message.source] = [output for output in outputs if output.seq > body.through_seq]
        for message in messages:
            self._fold_meta(fold, message)
            self._fold_facts(fold, message, inherited)
        # 调用账 owner 批量读取窄字段；插件自带 reader 保持原调用合同。
        read_call = self._read_call
        if isinstance(read_call, ModelCallReader):
            fold.receipts.update(read_call.replay(
                tuple(value["call_record_id"] for value in fold.recorded.values())
            ))
        for message in messages:
            self._fold_main(fold, message, inherited)
        fold.count = len(messages)
        fold.prefix = messages

    def _fold_apply(
        self,
        fold: _Fold,
        message: Message,
        inherited: Mapping[str, tuple[Message, Mapping[str, Any] | None]],
    ) -> None:
        """把一条追加消息折进编排状态；abandon 影响已折前缀，整体失效重建。"""
        if isinstance(message.body, Control) and message.body.action == "abandon":
            raise _FoldStale
        self._fold_meta(fold, message)
        self._fold_facts(fold, message, inherited)
        self._fold_main(fold, message, inherited)
        fold.count += 1

    def _fold_meta(self, fold: _Fold, message: Message) -> None:
        if isinstance(message.body, Input) and message.source == self._source:
            fold.inputs.add(message.message_id)
            fold.latest_input = message.message_id

    def _fold_facts(
        self,
        fold: _Fold,
        message: Message,
        inherited: Mapping[str, tuple[Message, Mapping[str, Any] | None]],
    ) -> None:
        """只复用同一不可变 Message 的静态检查；内容贡献者随 facts 一次取齐。"""
        if not isinstance(message.body, Output):
            return
        previous = inherited.get(message.message_id)
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
        fold.facts[message.message_id] = (message, value)
        if value is not None:
            fold.recorded[message.message_id] = value
            if message.source == self._source:
                for ref in value.get("content_refs", ()):
                    fold.content_refs.add(tuple(ref))

    def _fold_receipt(self, fold: _Fold, call_id: str) -> Mapping[str, Any]:
        """成功结算的回执不可变，每条记录在折叠生命周期内只读一次。"""
        receipt = fold.receipts.get(call_id)
        if receipt is None:
            read_call = self._read_call
            if isinstance(read_call, ModelCallReader):
                receipt = read_call.replay((call_id,))[call_id]
            else:
                receipt = read_call(call_id)
            fold.receipts[call_id] = receipt
        return receipt

    def _fold_main(
        self,
        fold: _Fold,
        message: Message,
        inherited: Mapping[str, tuple[Message, Mapping[str, Any] | None]],
    ) -> None:
        body = message.body
        if isinstance(body, Control) and body.action == "abandon" and message.source == self._source:
            fold.continuation = None
        if isinstance(body, ToolResult):
            # 已放弃调用的结算回执仍要配对渲染；call/result 邻接关系不能断。
            if body.call_ref in fold.results:
                raise ValueError("同一工具调用出现多个结果")
            fold.results[body.call_ref] = message
        if not isinstance(body, Output):
            return
        value = fold.recorded.get(message.message_id)
        if value is None:
            return
        receipt = self._fold_receipt(fold, value["call_record_id"])
        if receipt["state"] != "success":
            raise ValueError("已提交模型事实必须引用成功结算的真实调用")
        if receipt["binding"]["binding_id"] == self._model.descriptor.binding_id:
            metadata = receipt.get("provider_metadata")
            if metadata is not None and not value.get("content_transformed", False):
                fold.response_metadata[message.message_id] = metadata
        previous = inherited.get(message.message_id)
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
        if message.source == self._source and message.message_id not in fold.abandoned:
            fold.continuation = message_continuation
            fold.cont_transformed = value.get("content_transformed", False)
            fold.cont_seq = message.seq
            summaries = [
                part for part in body.parts
                if isinstance(part, ContentPart) and part.kind == "context.summary"
            ]
            if len(summaries) > 1:
                raise ValueError("同一模型 Output 只能使用一份摘要")
            fold.cont_summary = (
                self._check_summary(summaries[0]).binding_ids[0] if summaries else None
            )


_EXTERNAL_PART_KINDS = frozenset({"artifact_ref", "reply_ref"})


class _FoldStale(Exception):
    """追加尾部出现 abandon：编排折叠失效，回退全量重建。"""


class _Fold:
    """render 编排层的不可变前缀折叠；字段即原三趟预扫描的终态。"""

    __slots__ = (
        "count", "prefix", "inputs", "latest_input",
        "abandoned", "abandoned_calls", "facts", "recorded", "results",
        "receipts", "response_metadata", "content_refs",
        "continuation", "cont_summary", "cont_seq", "cont_transformed",
    )

    def __init__(self) -> None:
        self.count = 0
        self.prefix: Sequence[Message] = ()
        self.inputs: set[str] = set()
        self.latest_input: str | None = None
        self.abandoned: set[str] = set()
        self.abandoned_calls: set[CallRef] = set()
        self.facts: dict[str, tuple[Message, Mapping[str, Any] | None]] = {}
        self.recorded: dict[str, Mapping[str, Any]] = {}
        self.results: dict[CallRef, Message] = {}
        self.receipts: dict[str, Mapping[str, Any]] = {}
        self.response_metadata: dict[str, Mapping[str, Any]] = {}
        self.content_refs: set[tuple[str, int]] = set()
        self.continuation: ModelContinuation | None = None
        self.cont_summary: str | None = None
        self.cont_seq = -1
        self.cont_transformed = False


def _fold_compatible(fold: _Fold, messages: Sequence[Message]) -> bool:
    """存储读面证明前缀；普通序列仍逐条核对，不能只比较首尾身份。"""
    prefix = fold.prefix
    if type(messages) is MessageSnapshot and type(prefix) is MessageSnapshot:
        return messages.extends(prefix)
    if fold.count > len(messages):
        return False
    for index in range(fold.count):
        if messages[index] is not prefix[index]:
            return False
    return True


@dataclass(frozen=True, slots=True)
class _RenderedMessage:
    """一个可见消息的编译结果；index 是当前窗口内的稳定装配位置。"""

    message: Message
    index: int
    reminder: str | None
    rows: tuple[Mapping[str, Any], ...]
    refs: tuple[tuple[str, int], ...]
    used: tuple[str, ...]
    transformed: bool
    dynamic: bool


class _RenderView:
    """当前窗口的有序编译结果；只随追加推进，日志或窗口变化时重建。"""

    def __init__(self, fold: _Fold, after_seq: int):
        self.fold = fold
        self.after_seq = after_seq
        self.count = 0
        self.units: dict[str, _RenderedMessage] = {}
        self.reminders: dict[tuple[str, str], str] = {}
        self.dynamic: set[str] = set()


def _static_parts(message: Message, dynamic_kinds: frozenset[str] | None = None) -> bool:
    """artifact/reply 引用按渲染时的外部状态解析，动态声明 kind 由视图逐轮展开；
    两类都不参与分段缓存。None 表示范围未知，调用方在 cache_ok 处已拦截。"""
    excluded = (dynamic_kinds or frozenset()) | _EXTERNAL_PART_KINDS
    return all(
        not isinstance(part, ContentPart) or part.kind not in excluded
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
