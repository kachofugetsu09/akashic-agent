from __future__ import annotations

from typing import Any, cast

from agent.plugin_composition import Context
from agent.plugin_contracts.onboarding import Ability, PreviewLine
from plugins.onboarding.projection import Group, Registry


def _registry() -> Registry:
    registry = Registry(cast(Context, object()))
    registry.groups = {
        "memory": Group("Akasha 情景记忆", Ability("记住你们聊过的事", "会想起来", (PreviewLine("你", "上次那家店？"),))),
        "models": Group("模型连接", Ability("连接一个对话模型", "先连模型", required=True)),
        "legacy": Group("旧分组", None),
    }
    return registry


def test_groups_follow_step_order_and_skip_empty_groups() -> None:
    rows: list[dict[str, Any]] = [{"group": "models"}, {"group": "memory"}, {"group": "memory"}, {"group": "orphan"}]

    groups = _registry()._group_rows(rows)

    assert [group["key"] for group in groups] == ["models", "memory", "orphan"]
    assert groups[0]["required"] is True
    assert groups[1] == {
        "key": "memory", "title": "Akasha 情景记忆", "required": False,
        "pitch": "记住你们聊过的事", "benefit": "会想起来",
        "preview": [{"speaker": "你", "text": "上次那家店？"}],
    }


def test_group_without_ability_falls_back_to_title() -> None:
    groups = _registry()._group_rows([{"group": "legacy"}, {"group": "orphan"}])

    assert groups[0] == {"key": "legacy", "title": "旧分组", "required": False, "pitch": "", "benefit": "", "preview": []}
    assert groups[1]["title"] == "orphan"
