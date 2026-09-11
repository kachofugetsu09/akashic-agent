"""模型调用与投影的公开结构合同。

`models.calls.v1` / `models.call-history.v1` 是模型调用记录的公开名字，插件之间
共享调用读取与投影渲染的词汇。值词汇、纯函数与 Protocol 由合同层拥有；具体实现
（`MessageProjection`、`ModelCallReader`）留在 `plugins/models/`。

`MessageProjection` 的完整实现依赖 models 插件的存储（`ModelCallReader` 的
isinstance 判断），因此本层只提供 `MessageProjectionPort` 供消费者注解；需要
构造投影的调用点应经 ServiceKey 取得，属后续批次。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.message import ContentPart, ContentReferences, Message

if TYPE_CHECKING:
    from agent.plugin_composition.models import LLMResponse, ModelRequest

ContentRenderer = Callable[[ContentPart], Sequence[Mapping[str, Any]]]
CallReader = Callable[[str], Mapping[str, Any]]

MODEL_CALLS = ServiceKey[CallReader]("models.calls.v1")
MODEL_CALL_HISTORY = ServiceKey[Callable[[str, int], tuple[Mapping[str, Any], ...]]](
    "models.call-history.v1"
)


@runtime_checkable
class MessageProjectionPort(Protocol):
    """只读历史投影；不读写会话、不执行工具、不调用模型。"""

    @property
    def context_window(self) -> int | None:
        """该模型的上下文窗口；无限制时为 None。"""
        ...

    def render(
        self,
        messages: tuple[Message, ...],
        *,
        after_seq: int,
        summary_reference: str | None = None,
        fresh: bool = False,
    ) -> ModelRequest:
        """把完整事实投影为一次模型请求；after_seq 是摘要覆盖末尾，-1 表示无覆盖。"""
        ...


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

