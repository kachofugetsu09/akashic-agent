from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.models import ModelRequest
from session.message import (
    ContentPart,
    Message,
    Output,
    ToolCall,
    ToolResult,
)
from session.message_codec import json_value

api_version = 3
name = "context"
version = "1.0.0"
desc = "用固定日志和已取得材料组成请求，不执行检索或摘要"
inject = ()


@dataclass(frozen=True, slots=True)
class Summary:
    """摘要 owner 已持久发布的内容与精确覆盖范围。"""

    reference: str
    source_message_ids: tuple[str, ...]
    content: str

    def __post_init__(self) -> None:
        ids = tuple(self.source_message_ids)
        if not self.reference or not isinstance(self.reference, str):
            raise ValueError("摘要必须有持久来源引用")
        if not ids or any(not isinstance(item, str) or not item for item in ids):
            raise ValueError("摘要必须声明覆盖的消息")
        if len(set(ids)) != len(ids):
            raise ValueError("摘要消息引用不能重复")
        if not isinstance(self.content, str) or not self.content:
            raise ValueError("摘要正文不能为空")
        object.__setattr__(self, "source_message_ids", ids)


@dataclass(frozen=True, slots=True)
class Materials:
    """权限已由组合确定的 Prompt，以及保持低信任的检索材料。"""

    system_prompt: str
    context: tuple[ContentPart, ...] = ()
    summary: Summary | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str):
            raise TypeError("system Prompt 必须是字符串")
        parts = tuple(self.context)
        if any(not isinstance(part, ContentPart) for part in parts):
            raise TypeError("检索材料必须是已校验的内容块")
        if self.summary is not None and not isinstance(self.summary, Summary):
            raise TypeError("摘要必须来自已发布的 Summary")
        object.__setattr__(self, "context", parts)


class ContextModel(Protocol):
    """Model 的只读请求投影；这里没有 complete 或工具执行权。"""

    @property
    def context_window(self) -> int | None: ...

    @property
    def max_tool_schemas(self) -> int | None: ...

    def render(self, messages: tuple[Message, ...], *, after_seq: int) -> ModelRequest:
        """接收完整事实；after_seq 是摘要覆盖末尾，-1 表示没有覆盖。

        Model owner 按自身协议保留覆盖区间内仍必需的 opaque replay，
        无法满足时明确拒绝；不能把 after_seq 当成丢弃这些事实的授权。
        """
        ...

    def estimate(self, request: ModelRequest) -> int: ...


class ContextOverflow(ValueError):
    def __init__(self, estimated_tokens: int, output_tokens: int, capacity: int):
        self.estimated_tokens = estimated_tokens
        self.output_tokens = output_tokens
        self.capacity = capacity
        super().__init__(
            f"请求需要约 {estimated_tokens}+{output_tokens} tokens，容量 {capacity}"
        )


def _summary_cutoff(snapshot: tuple[Message, ...], summary: Summary | None) -> int:
    """只替换摘要实际覆盖的完整前缀，不从字数或最新 Turn 猜边界。"""
    # 1. 在快照输入边界拒绝混合 Session、重排与重复身份。
    if snapshot:
        session = snapshot[0].session_id
        if any(item.session_id != session for item in snapshot):
            raise ValueError("Context 快照只能属于一个 Session")
        if any(a.seq >= b.seq for a, b in zip(snapshot, snapshot[1:])):
            raise ValueError("Context 快照必须按 seq 严格递增")
        if len({item.message_id for item in snapshot}) != len(snapshot):
            raise ValueError("Context 快照消息身份重复")
    if summary is None:
        return -1
    count = len(summary.source_message_ids)
    if (
        tuple(item.message_id for item in snapshot[:count])
        != summary.source_message_ids
    ):
        raise ValueError("摘要覆盖范围不等于实际消息前缀")
    # 2. 不让摘要切开 provider 必须成对恢复的工具请求与结果。
    calls = {
        (item.message_id, index)
        for item in snapshot[:count]
        if isinstance(item.body, Output)
        for index, part in enumerate(item.body.parts)
        if isinstance(part, ToolCall)
    }
    results = {
        (item.body.call_ref.message_id, item.body.call_ref.part_index)
        for item in snapshot[:count]
        if isinstance(item.body, ToolResult)
    }
    if calls - results:
        raise ValueError("摘要不能覆盖尚未结算的工具调用")
    return snapshot[count - 1].seq


class ContextBuilder:
    def build(
        self,
        snapshot: Sequence[Message],
        *,
        materials: Materials,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
    ) -> ModelRequest:
        """纯函数式组装；容量不足明确报错，由调用程序取得更小视图。"""
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("输出预算必须是正整数")
        # 1. Model owner 保留自身的 call IDs 与 opaque replay，Context 不重造它们。
        snapshot = tuple(snapshot)
        cutoff = _summary_cutoff(snapshot, materials.summary)
        rendered = model.render(snapshot, after_seq=cutoff)
        if any(
            row["role"] not in {"user", "assistant", "tool"}
            for row in rendered.messages
        ):
            raise ValueError("历史投影不能产生 system/developer 权限")
        rows: list[Mapping[str, Any]] = []
        if materials.system_prompt:
            rows.append({"role": "system", "content": materials.system_prompt})
        # 2. 摘要和检索是带出处的数据，不能通过文本伪装成高权限 Prompt。
        if materials.summary is not None:
            rows.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "summary": materials.summary.content,
                            "reference": materials.summary.reference,
                            "source_message_ids": materials.summary.source_message_ids,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        rows.extend(rendered.messages)
        if materials.context:
            rows.append(
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "context": [
                                {"kind": part.kind, "value": json_value(part.value)}
                                for part in materials.context
                            ]
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                }
            )
        request = replace(
            rendered,
            messages=rows,
            tools=tools,
            system_prompt="",
            max_output_tokens=max_output_tokens,
        )
        # 3. 估算只读取完整请求，不调用模型、裁切原文或持久发布摘要。
        if model.max_tool_schemas is not None and len(tools) > model.max_tool_schemas:
            raise ValueError("可见工具数量超过模型容量")
        estimated = model.estimate(request)
        if (
            model.context_window is not None
            and estimated + max_output_tokens > model.context_window
        ):
            raise ContextOverflow(estimated, max_output_tokens, model.context_window)
        return request


CONTEXT = ServiceKey[ContextBuilder]("context.v1")


async def apply(ctx: Context, config: object) -> None:
    _ = await ctx.provide(CONTEXT, ContextBuilder())
