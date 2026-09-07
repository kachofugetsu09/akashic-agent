from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ConfigDict

from agent.plugin_composition import Context, ServiceKey
from agent.plugin_composition.models import ModelRequest
from session.message import (
    Message,
)
from session.message_codec import json_value
from plugins.context.api import ContextModel, ContextOverflow, Materials, Summary, settled_prefixes, summary_range
from plugins.context.materials import ContextMaterials, MATERIALS

api_version = 3
name = "context"
version = "1.0.0"
desc = "用固定日志和已取得材料组成请求，不执行检索或摘要"
inject = ()




class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_sources: dict[str, str] = {}
    summary_source: tuple[str, str] | None = None


def _summary_cutoff(snapshot: tuple[Message, ...], summary: Summary | None) -> int:
    """摘要代替已选择窗口的旧区间，窗口外更早历史不进入请求。"""
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
    covered = summary_range(snapshot, summary.source_message_ids)
    # 2. 不让摘要切开 provider 必须成对恢复的工具请求与结果。
    ends = settled_prefixes(snapshot[:covered.stop])
    if covered.stop not in ends or covered.start and covered.start not in ends:
        raise ValueError("摘要不能覆盖尚未结算的工具调用")
    return snapshot[covered.stop - 1].seq


class ContextBuilder:
    def build(
        self,
        snapshot: Sequence[Message],
        *,
        materials: Materials,
        model: ContextModel,
        tools: Sequence[Mapping[str, Any]] = (),
        max_output_tokens: int,
        window_start: str | None = None,
    ) -> ModelRequest:
        """纯函数式组装；容量不足明确报错，由调用程序取得更小视图。"""
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("输出预算必须是正整数")
        # 1. Model owner 保留自身的 call IDs 与 opaque replay，Context 不重造它们。
        snapshot = tuple(snapshot)
        cutoff = _summary_cutoff(snapshot, materials.summary)
        if window_start is not None:
            if materials.summary is not None:
                raise ValueError("已有摘要的请求不能重新选择首次窗口")
            identities = tuple(message.message_id for message in snapshot)
            if window_start not in identities:
                raise ValueError("首次窗口起点缺少实际消息")
            start = identities.index(window_start)
            if start and start not in settled_prefixes(snapshot[:start]):
                raise ValueError("首次窗口不能切开尚未结算的工具调用")
            cutoff = -1 if start == 0 else snapshot[start - 1].seq
            rendered = model.render(snapshot, after_seq=cutoff, fresh=True)
        else:
            rendered = model.render(
                snapshot, after_seq=cutoff,
                summary_reference=None if materials.summary is None else materials.summary.reference,
            )
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
            raise ContextOverflow(estimated, max_output_tokens, model.context_window, request=request)
        return request


CONTEXT = ServiceKey[ContextBuilder]("context.v1")


async def apply(ctx: Context, config: Config | None) -> None:
    config = Config() if config is None else config
    _ = await ctx.provide(CONTEXT, ContextBuilder())
    _ = await ctx.provide(MATERIALS, ContextMaterials(ctx, prompt_sources=config.prompt_sources, summary_source=config.summary_source))
