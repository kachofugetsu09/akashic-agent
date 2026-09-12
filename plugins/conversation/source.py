from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from agent.model_runtime.session_selection import SessionModelSelection, write_session_model_selection
from agent.plugin_contracts import Body, Input


def update_selection(body: Body) -> Mapping[str, object | None]:
    """只从本次已验证 Input 生成选择变化，与正文同事务保存。"""
    if not isinstance(body, Input):
        raise TypeError("会话选择只能随 Input 更新")
    parts = [part for part in body.parts if part.kind == "model.selection"]
    if not parts:
        return {}
    if len(parts) != 1:
        raise ValueError("一个 Input 只能包含一次模型选择")
    value = cast(Mapping[str, str | None], parts[0].value)
    selected: dict[str, object] = {}
    write_session_model_selection(selected, SessionModelSelection(
        value["model_id"] or "", value["reasoning_effort"] or "",
    ))
    return {"model_selection": selected.get("model_selection"), "model_runtime_override": None}


