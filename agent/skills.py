import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"
SkillSource = Literal["workspace", "builtin", "plugin"]


@dataclass(frozen=True)
class SkillRecord:
    name: str
    display_name: str
    source: SkillSource
    source_id: str
    root_dir: Path
    skill_file: Path
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


class SkillsLoader:
    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace = workspace
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def list_skill_records(self, filter_unavailable: bool = True) -> list[SkillRecord]:
        return self._build_index().list_records(filter_unavailable=filter_unavailable)

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts: list[str] = []
        for name in skill_names:
            content = self.load_skill_body(name)
            if content:
                parts.append(f"### Skill: {name}\n\n{content}")

        return "\n\n---\n\n".join(parts) if parts else ""

    def load_skill_body(self, name: str) -> str | None:
        record = self.load_skill_record(name)
        if record is None:
            return None
        content = record.skill_file.read_text(encoding="utf-8")
        return self._strip_frontmatter(content)

    def load_skill_record(self, name: str) -> SkillRecord | None:
        return self._build_index().get(name)

    def get_always_skills(self) -> list[str]:
        return [
            record.name
            for record in self.list_skill_records(filter_unavailable=True)
            if record.always
        ]

    def build_skills_summary(self) -> str:
        records = self.list_skill_records(filter_unavailable=False)
        if not records:
            return ""

        lines = ["<skills>"]
        for record in records:
            name = self._escape_xml(record.name)
            source = self._escape_xml(record.source)
            available = str(record.available).lower()
            desc = self._escape_xml(record.description)
            lines.append(
                f'  <skill name="{name}" available="{available}" source="{source}">'
            )
            lines.append(f"    <description>{desc}</description>")
            if record.when_to_use:
                when_to_use = self._escape_xml(record.when_to_use)
                lines.append(f"    <when_to_use>{when_to_use}</when_to_use>")
            if not record.available and record.missing:
                missing = self._escape_xml(record.missing)
                lines.append(f"    <requires>{missing}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")

        return "\n".join(lines)

    def _build_index(self) -> SkillIndex:
        records: dict[str, SkillRecord] = {}

        for record in self._scan_skills_dir(
            self.workspace_skills,
            source="workspace",
            source_id="workspace",
        ):
            records[record.name] = record

        if self.builtin_skills:
            for record in self._scan_skills_dir(
                self.builtin_skills,
                source="builtin",
                source_id="builtin",
            ):
                if record.name not in records:
                    records[record.name] = record

        return SkillIndex(records)

    def _scan_skills_dir(
        self,
        skills_dir: Path,
        *,
        source: SkillSource,
        source_id: str,
        name_prefix: str = "",
    ) -> list[SkillRecord]:
        if not skills_dir.exists():
            return []

        records: list[SkillRecord] = []
        for skill_dir in sorted(skills_dir.iterdir(), key=lambda item: item.name):
            if not skill_dir.is_dir():
                continue
            skill_file = skill_dir / "SKILL.md"
            if not skill_file.exists():
                continue
            name = f"{name_prefix}{skill_dir.name}"
            records.append(
                self._build_record(
                    name=name,
                    root_dir=skill_dir,
                    skill_file=skill_file,
                    source=source,
                    source_id=source_id,
                )
            )
        return records

    def _build_record(
        self,
        *,
        name: str,
        root_dir: Path,
        skill_file: Path,
        source: SkillSource,
        source_id: str,
    ) -> SkillRecord:
        content = skill_file.read_text(encoding="utf-8")
        meta = self._parse_frontmatter(content) or {}
        config = self._parse_skill_config(meta.get("metadata", ""))
        missing = self._get_missing_requirements(config)
        return SkillRecord(
            name=name,
            display_name=meta.get("name") or name,
            source=source,
            source_id=source_id,
            root_dir=root_dir,
            skill_file=skill_file,
            description=meta.get("description") or name,
            when_to_use=meta.get("when_to_use", ""),
            config=config,
            always=self._as_bool(config.get("always"))
            or self._as_bool(meta.get("always")),
            available=not missing,
            missing=missing,
        )

    def _parse_frontmatter(self, content: str) -> dict[str, Any]:
        if not content.startswith("---"):
            return {}
        parts = content.split("---", 2)
        if len(parts) < 3:
            return {}
        loaded = cast(object, yaml.safe_load(parts[1]) or {})
        if not isinstance(loaded, dict):
            return {}
        data = cast(dict[object, Any], loaded)
        return {str(key): value for key, value in data.items()}

    def _strip_frontmatter(self, content: str) -> str:
        if content.startswith("---"):
            match = re.match(r"^---\n.*?\n---\n", content, re.DOTALL)
            if match:
                return content[match.end() :].strip()
        return content

    def _parse_skill_config(self, raw: str | object) -> dict[str, Any]:
        if isinstance(raw, dict):
            data = cast(dict[str, Any], raw)
        else:
            try:
                parsed: Any = json.loads(str(raw))
            except json.JSONDecodeError:
                return {}
            if not isinstance(parsed, dict):
                return {}
            data = cast(dict[str, Any], parsed)
        for key in ("akashic", "skill"):
            value = data.get(key)
            if isinstance(value, dict):
                return cast(dict[str, Any], value)
        return cast(dict[str, Any], data)

    def _get_missing_requirements(self, skill_config: dict[str, Any]) -> str:
        missing: list[str] = []
        requires = skill_config.get("requires", {})
        if not isinstance(requires, dict):
            return ""
        requires_dict = cast(dict[str, object], requires)
        for binary in self._string_list(requires_dict.get("bins")):
            if not shutil.which(binary):
                missing.append(f"CLI: {binary}")
        for env in self._string_list(requires_dict.get("env")):
            if not os.environ.get(env):
                missing.append(f"ENV: {env}")
        return ", ".join(missing)

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        items = cast(list[object], value)
        return [item for item in items if isinstance(item, str)]

    def _as_bool(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "on"}
        return False

    def _escape_xml(self, value: str) -> str:
        return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
