"""技能的公开结构合同。

`SkillRecord`/`SkillIndex` 是技能目录的值词汇，`skill_body` 是纯文本处理
（去掉 frontmatter）。插件需要它们来读技能与计算正文摘要，但不需要 import
`agent/skills.py` 里的加载器实现（`SkillsLoader` 仍留在原处）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


SkillSource = Literal["workspace", "builtin", "plugin"]


def skill_body(content: str) -> str:
    """去掉技能 frontmatter，保留原有正文读取规则。"""
    if content.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
        if match:
            return content[match.end():].strip()
    return content


@dataclass(frozen=True)
class SkillRecord:
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
    records: dict[str, SkillRecord]

    def list_records(self, *, filter_unavailable: bool) -> list[SkillRecord]:
        records = list(self.records.values())
        if filter_unavailable:
            return [record for record in records if record.available]
        return records

    def get(self, name: str) -> SkillRecord | None:
        return self.records.get(name)

