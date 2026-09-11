"""会话模型选择的公开结构合同（纯函数）。

`check_selection` 校验 `model.selection` 内容块，`selection` 读取一段输入范围里
最后一次显式选择。两者只读取消息内容，不落库、不校验可用性（由 Model owner
在调用前验证），因此可由合同层拥有。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

# `selection()` 会**构造** ChatModelSelection（不是只做注解），因此必须是运行时 import。
# 这也是「合同层可以依赖 agent.plugin_composition.models 这个纯值原语」的白名单场景
# （与 delivery_api.py 用 ServiceKey 同理）：它零仓库内实现依赖。
from agent.plugin_contracts.message import (
    ContentPart,
    ContentReferences,
    Input,
    Message,
)


def check_selection(part: ContentPart) -> ContentReferences:
    """模型偏好是用户选择事实，实际可用性由 Model owner 在调用前验证。"""
    raw_value = part.value
    if not isinstance(raw_value, Mapping):
        raise ValueError("model.selection 必须是对象")
    value = cast(Mapping[str, object], raw_value)
    if set(value) != {"model_id", "reasoning_effort"} or any(
        item is not None and (not isinstance(item, str) or not item)
        for item in value.values()
    ):
        raise ValueError("model.selection 字段无效")
    return ContentReferences()
