"""公开的技能记录原子与当前插件技能读取口。"""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Literal

from agent.plugin_composition.model import ServiceKey

SkillSource = Literal["workspace", "builtin", "plugin"]


def skill_body(content: str) -> str:
    """去掉技能 frontmatter，保留正文读取规则。"""

    if content.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if match:
            return content[match.end() :].strip()
    return content


@dataclass(frozen=True)
class SkillRecord:
    """Freeze one discovered skill without owning its loader or install state."""

    name: str
    display_name: str
    source: SkillSource
    source_id: str
    root_dir: Path
    skill_file: Path
    content: str
    description: str
    when_to_use: str
    config: dict[str, Any]
    always: bool
    available: bool
    missing: str


@dataclass(frozen=True)
class SkillIndex:
    """Immutable skill records published by one exact runtime catalog."""

    records: dict[str, SkillRecord]

    def list_records(self, *, filter_unavailable: bool) -> list[SkillRecord]:
        records = list(self.records.values())
        if filter_unavailable:
            return [record for record in records if record.available]
        return records

    def get(self, name: str) -> SkillRecord | None:
        return self.records.get(name)


SKILL_CATALOG = ServiceKey[SkillIndex]("core.skill_catalog.v1")


def plugin_records(index: SkillIndex) -> tuple[SkillRecord, ...]:
    """Return the plugin-owned records from one exact published catalog."""

    return tuple(index.records[key] for key in sorted(index.records))


__all__ = [
    "SkillIndex",
    "SkillRecord",
    "SkillSource",
    "SKILL_CATALOG",
    "plugin_records",
    "skill_body",
]
