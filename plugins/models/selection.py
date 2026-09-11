"""模型选择的纯校验（合同层）与选择读取（实现）。

`check_selection` 是纯校验，已归位到 `agent.plugin_contracts.model_selection`；
`selection()` 会**构造** `ChatModelSelection`（组合内核的模型值类型），因此按
「合同层不依赖实现模块」的判据留在本模块。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from agent.plugin_composition.models import ChatModelSelection
from agent.plugin_contracts.message import Input, Message
from agent.plugin_contracts.model_selection import check_selection  # noqa: F401  (再导出)

__all__ = ["check_selection", "selection"]


def selection(messages: Sequence[Message]) -> ChatModelSelection | None:
    """读取给定输入范围的最后一次显式选择；None 表示没有选择事实。"""
    for message in reversed(messages):
        if isinstance(message.body, Input):
            for part in reversed(message.body.parts):
                if part.kind == "model.selection":
                    value = cast(Mapping[str, str | None], part.value)
                    return ChatModelSelection(value["model_id"], value["reasoning_effort"])
    return None
