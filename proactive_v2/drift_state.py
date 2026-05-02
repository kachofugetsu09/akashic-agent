from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.common.timekit import parse_iso
from infra.persistence.json_store import atomic_save_json, load_json

logger = logging.getLogger(__name__)


def _clip(text: str, limit: int) -> str:
    return str(text or "").strip()[:limit]


def _parse_skill_frontmatter(content: str) -> dict[str, str | list[str]]:
    if not content.startswith("---"):
        return {}
    match = re.match(r"^---\n(.*?)\n---", content, re.DOTALL)
    if match is None:
        return {}
    metadata: dict[str, str | list[str]] = {}
    lines = match.group(1).split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if ":" not in line:
            i += 1
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if value:
            metadata[key] = value
            i += 1
            continue
        # value 为空 → 检查后续行是否为 YAML 列表项 (  - item)
        list_items: list[str] = []
        j = i + 1
        while j < len(lines):
            item_match = re.match(r"^\s+-\s+(.+)", lines[j])
            if item_match is None:
                break
            list_items.append(item_match.group(1).strip().strip("\"'"))
            j += 1
        if list_items:
            metadata[key] = list_items
            i = j
        else:
            metadata[key] = value
            i += 1
    return metadata


@dataclass
class SkillMeta:
    name: str
    description: str
    last_run_at: datetime | None
    run_count: int
    status: str
    next: str
    requires_mcp: list[str]
    builtin: bool


class DriftStateStore:
    def __init__(
        self,
        drift_dir: Path,
        *,
        builtin_skills_dir: Path | None = None,
        include_builtin_skills: bool = False,
        builtin_skill_names: set[str] | None = None,
    ) -> None:
        self.drift_dir = drift_dir.expanduser()
        self.skills_dir = self.drift_dir / "skills"
        self.drift_file = self.drift_dir / "drift.json"
        self.builtin_skills_dir = (
            builtin_skills_dir.expanduser()
            if builtin_skills_dir is not None
            else None
        )
        self.include_builtin_skills = include_builtin_skills
        self.builtin_skill_names = set(builtin_skill_names or set())
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def scan_skills(self) -> list[SkillMeta]:
        skills: list[SkillMeta] = []
        seen_names: set[str] = set()
        for root, builtin in self._skill_roots():
            if not root.exists():
                logger.info("[drift_state] skills dir missing: %s", root)
                continue
            for skill_dir in sorted(root.iterdir()):
                if not skill_dir.is_dir():
                    continue
                if builtin and self.builtin_skill_names and skill_dir.name not in self.builtin_skill_names:
                    continue
                skill = self._load_skill_meta(skill_dir, builtin=builtin)
                if skill is None:
                    continue
                if skill.name in seen_names:
                    logger.info("[drift_state] skip duplicate skill=%s", skill.name)
                    continue
                seen_names.add(skill.name)
                skills.append(skill)
        skills.sort(
            key=lambda item: item.last_run_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        logger.info(
            "[drift_state] scan_skills: found=%d names=%s",
            len(skills),
            [skill.name for skill in skills[:8]],
        )
        return skills

    def valid_skill_names(self) -> set[str]:
        return {skill.name for skill in self.scan_skills()}

    def load_drift(self) -> dict[str, Any]:
        raw = load_json(self.drift_file, default=None, domain="drift_state") or {}
        recent_runs = raw.get("recent_runs")
        if not isinstance(recent_runs, list):
            recent_runs = []
        rows: list[dict[str, str]] = []
        for row in recent_runs:
            if not isinstance(row, dict):
                continue
            skill = _clip(row.get("skill", ""), 80)
            run_at = _clip(row.get("run_at", ""), 80)
            one_line = _clip(row.get("one_line", ""), 150)
            message_result = _clip(row.get("message_result", "silent"), 20)
            if message_result not in {"sent", "silent"}:
                message_result = "silent"
            if not skill or not run_at or not one_line:
                continue
            rows.append({
                "skill": skill,
                "run_at": run_at,
                "one_line": one_line,
                "message_result": message_result,
            })
        return {
            "version": 1,
            "recent_runs": rows[-10:],
            "note": _clip(raw.get("note", ""), 150),
        }

    def skill_dir_for(self, skill_name: str) -> Path | None:
        name = str(skill_name or "").strip()
        if not name:
            return None
        workspace_dir = self.skills_dir / name
        if (workspace_dir / "SKILL.md").exists():
            return workspace_dir
        if self.include_builtin_skills and self.builtin_skills_dir is not None:
            builtin_dir = self.builtin_skills_dir / name
            if (builtin_dir / "SKILL.md").exists():
                return builtin_dir
        return None

    def save_finish(
        self,
        *,
        skill_used: str,
        one_line: str,
        next_action: str,
        message_result: str,
        note: str | None,
        now_utc: datetime,
    ) -> None:
        skill_name = str(skill_used or "").strip()
        skill_dir = self.skill_dir_for(skill_name) or (self.skills_dir / skill_name)
        skill_dir.mkdir(parents=True, exist_ok=True)
        state = self._load_skill_state(skill_dir)
        logger.info(
            "[drift_state] save_finish: skill=%s next=%s note=%s",
            skill_name,
            _clip(next_action, 100),
            bool(note),
        )
        atomic_save_json(
            skill_dir / "state.json",
            {
                "version": 1,
                "last_run_at": now_utc.isoformat(),
                "run_count": max(0, int(state.get("run_count", 0) or 0)) + 1,
                "status": "in_progress",
                "next": _clip(next_action, 100),
            },
            domain="drift_state",
        )

        drift = self.load_drift()
        recent_runs = list(drift.get("recent_runs", []))
        recent_runs.append(
            {
                "skill": skill_name,
                "run_at": now_utc.isoformat(),
                "one_line": _clip(one_line, 150),
                "message_result": message_result,
            }
        )
        payload = {
            "version": 1,
            "recent_runs": recent_runs[-10:],
            "note": drift.get("note", ""),
        }
        if note is not None:
            payload["note"] = _clip(note, 150)
        atomic_save_json(self.drift_file, payload, domain="drift_state")

    def _load_skill_state(self, skill_dir: Path) -> dict[str, Any]:
        raw = load_json(skill_dir / "state.json", default=None, domain="drift_state") or {}
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _normalize_status(raw: Any) -> str:
        status = str(raw or "").strip()
        return status if status in {"idle", "in_progress"} else "idle"

    def _skill_roots(self) -> list[tuple[Path, bool]]:
        roots: list[tuple[Path, bool]] = [(self.skills_dir, False)]
        if self.include_builtin_skills and self.builtin_skills_dir is not None:
            roots.append((self.builtin_skills_dir, True))
        return roots

    def _load_skill_meta(self, skill_dir: Path, *, builtin: bool) -> SkillMeta | None:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.exists():
            return None
        metadata = _parse_skill_frontmatter(skill_file.read_text(encoding="utf-8"))
        name = str(metadata.get("name") or "").strip()
        description = str(metadata.get("description") or "").strip()
        if not name or not description or name != skill_dir.name:
            logger.info("[drift_state] skip invalid skill dir=%s name=%r", skill_dir, name)
            return None
        requires_mcp_val = metadata.get("requires_mcp")
        if isinstance(requires_mcp_val, list):
            requires_mcp = [s.strip() for s in requires_mcp_val if s.strip()]
        else:
            raw = str(requires_mcp_val or "").strip()
            requires_mcp = [s.strip() for s in raw.split(",") if s.strip()] if raw else []
        raw_state = self._load_skill_state(skill_dir)
        return SkillMeta(
            name=name,
            description=description,
            last_run_at=parse_iso(raw_state.get("last_run_at")),
            run_count=max(0, int(raw_state.get("run_count", 0) or 0)),
            status=self._normalize_status(raw_state.get("status")),
            next=_clip(raw_state.get("next", ""), 100),
            requires_mcp=requires_mcp,
            builtin=builtin,
        )
