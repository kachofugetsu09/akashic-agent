"""技能目录的只读合同：skills 插件提供，运行时检查等消费者只读。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from agent.plugin_composition import ServiceKey


class SkillReader(Protocol):
    """技能目录只读投影的窄输入。"""

    async def list_skills(self) -> tuple[Mapping[str, object], ...]: ...

    async def list_sources(self) -> tuple[Mapping[str, object], ...]: ...


# RuntimeInspection._bind_optional 将此 key 传给 ctx.inject 的技能目录子 Fiber。
SKILL_INSPECTION = ServiceKey[SkillReader]("skills.inspection.v1")
