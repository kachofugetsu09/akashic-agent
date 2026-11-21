"""启动时已装插件输入的采用规则：Manager 加载与启动迁移共用同一份决定。"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast


@dataclass(frozen=True)
class AdoptionStep:
    action: Literal["keep", "replace", "add", "disabled", "broken"]
    plugin_id: str
    ref: str | None = None
    # 该步骤最终采用的代码目录；disabled 与 broken 不被运行时加载，disabled 仍给出当前制品供迁移使用。
    code: Path | None = None
    installed: bool = True


# 按磁盘当前制品、启停选择与已选输入，决定每个已装插件启动时采用哪份代码。
def plan_installed_adoption(
    records: Sequence[tuple[str, Mapping[str, object]]],
    installed_roots: Mapping[str, Path],
    choices: Mapping[str, bool],
    broken_roots: frozenset[Path],
) -> tuple[AdoptionStep, ...]:
    steps: list[AdoptionStep] = []
    selected: set[str] = set()
    # 1. 已选输入：停用则不加载，当前制品损坏则不退回旧制品，制品不同则换成当前制品。
    for ref, record in records:
        plugin_id = cast(str, record["plugin_id"])
        selected.add(plugin_id)
        current = installed_roots.get(plugin_id)
        code = Path(cast(str, record["code"])).resolve()
        if record["source_type"] != "installed":
            steps.append(AdoptionStep("keep", plugin_id, ref, code, installed=False))
        elif choices.get(plugin_id) is False:
            steps.append(AdoptionStep("disabled", plugin_id, ref, current or code))
        elif current is None and any(root.parent == code.parent and root != code for root in broken_roots):
            steps.append(AdoptionStep("broken", plugin_id, ref))
        elif current is None or Path(cast(str, record["code"])) == current.resolve():
            # cache 缺失时保留原输入，由加载 fail loud。
            steps.append(AdoptionStep("keep", plugin_id, ref, code))
        else:
            steps.append(AdoptionStep("replace", plugin_id, ref, current.resolve()))
    # 2. 已启用但尚未进入选择的已装插件，同样按当前制品加入。
    for plugin_id, current in sorted(installed_roots.items()):
        if plugin_id not in selected and choices.get(plugin_id) is True:
            steps.append(AdoptionStep("add", plugin_id, None, current.resolve()))
    return tuple(steps)
