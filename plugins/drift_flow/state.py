from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, cast

import yaml

from core.common.timekit import parse_iso
from infra.persistence.json_store import load_json

logger = logging.getLogger(__name__)


def _clip(text: str, limit: int) -> str:
    return str(text or "").strip()[:limit]


def _parse_skill_frontmatter(content: str) -> dict[str, Any]:
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
        plugin_skill_roots: tuple[Path, ...] = (),
    ) -> None:
        self.drift_dir = drift_dir.expanduser()
        self.skills_dir = self.drift_dir / "skills"
        self.db_file = self.drift_dir / "drift.db"
        self.builtin_skills_dir = (
            builtin_skills_dir.expanduser()
            if builtin_skills_dir is not None
            else None
        )
        self.include_builtin_skills = include_builtin_skills
        self.builtin_skill_names = set(builtin_skill_names or set())
        self.plugin_skill_roots = tuple(
            root.expanduser().resolve(strict=False) for root in plugin_skill_roots
        )
        self._last_saved_run_id: int | None = None
        self._last_saved_run_at: str = ""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_db()

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
        db_rows = self._load_recent_runs_from_db(limit=10)
        note = self._load_global_note()
        return {
            "version": 2,
            "recent_runs": db_rows,
            "note": _clip(note, 150),
        }

    def load_skill_continuum(self, skill_name: str) -> dict[str, Any]:
        return self._load_continuum(skill_name)

    def load_self_state(self) -> dict[str, str]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT current_skill, current_intention, last_decision,
                       decision_reason, next_tendency, updated_at
                FROM self_state
                WHERE id = 1
                """
            ).fetchone()
        if row is None:
            return {}
        return {
            "current_skill": str(row["current_skill"] or ""),
            "current_intention": str(row["current_intention"] or ""),
            "last_decision": str(row["last_decision"] or ""),
            "decision_reason": str(row["decision_reason"] or ""),
            "next_tendency": str(row["next_tendency"] or ""),
            "updated_at": str(row["updated_at"] or ""),
        }

    def save_self_choice(
        self,
        *,
        skill_name: str,
        intention: str,
        decision: str,
        reason: str,
        now_utc: datetime,
    ) -> None:
        with self._connection() as conn:
            self._upsert_self_state(
                conn=conn,
                current_skill=skill_name,
                current_intention=intention,
                last_decision=decision,
                decision_reason=reason,
                next_tendency="",
                updated_at=now_utc.isoformat(),
            )

    def load_skill_journal(
        self,
        skill_name: str,
        *,
        entry_type: str = "",
        key: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clean_skill = str(skill_name or "").strip()
        clean_type = str(entry_type or "").strip()
        clean_key = str(key or "").strip()
        if not clean_skill:
            return []
        clauses = ["skill_name = ?"]
        params: list[Any] = [clean_skill]
        if clean_type:
            clauses.append("entry_type = ?")
            params.append(clean_type)
        if clean_key:
            clauses.append("key = ?")
            params.append(clean_key)
        params.append(max(1, int(limit)))
        with self._connection() as conn:
            rows = conn.execute(
                f"""
                SELECT id, entry_type, key, payload_json, run_id, created_at
                FROM skill_journal
                WHERE {" AND ".join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                tuple(params),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in reversed(rows):
            result.append(
                {
                    "id": int(row["id"] or 0),
                    "entry_type": str(row["entry_type"] or ""),
                    "key": str(row["key"] or ""),
                    "payload": self._decode_json_object(row["payload_json"]),
                    "run_id": int(row["run_id"] or 0) if row["run_id"] is not None else None,
                    "created_at": str(row["created_at"] or ""),
                }
            )
        return result

    def load_recent_self_observations(self, *, limit: int = 12) -> list[dict[str, Any]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT skill_name, payload_json, run_id, created_at
                FROM skill_journal
                WHERE entry_type = 'self_observation'
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        return [
            {
                "skill_name": str(row["skill_name"] or ""),
                "payload": self._decode_json_object(row["payload_json"]),
                "run_id": int(row["run_id"] or 0) if row["run_id"] is not None else None,
                "created_at": str(row["created_at"] or ""),
            }
            for row in reversed(rows)
        ]

    def load_briefing(self, skills: list[SkillMeta]) -> str:
        recent_rows = self._load_recent_runs_from_db(limit=5)
        note = self._load_global_note()
        lines: list[str] = []

        lines.append("【Drift Briefing】")
        lines.append("")
        lines.append("全局前情：")
        lines.append(f"- {note}" if note else "- （空）")
        lines.append("")
        lines.append("当前可用 skill：")

        if not skills:
            lines.append("- （无）")
        for skill in skills[:8]:
            continuum = self._load_continuum(skill.name)
            status = str(continuum.get("last_status") or skill.status or "idle")
            finished_at = str(continuum.get("updated_at") or continuum.get("last_run_at") or "").strip()
            briefing = str(continuum.get("last_briefing") or "").strip()
            scratchpad = str(continuum.get("scratchpad") or "").strip()
            cursor = continuum.get("cursor")
            lines.append(f"- {skill.name}")
            lines.append(f"  运行：{skill.run_count} 次")
            if skill.builtin:
                lines.append("  来源：builtin")
            if skill.requires_mcp:
                lines.append(f"  需要：{', '.join(skill.requires_mcp)}")
            lines.append(f"  上次状态：{status}")
            lines.append(f"  上次 finish：{finished_at or 'never'}")
            lines.append(f"  上次摘要：{_clip(briefing, 160) or '（空）'}")
            if status == "completed":
                lines.append("  前情：已闭环；内部续航便签只在选中该 skill 后参考，不作为待办。")
            else:
                lines.append(f"  前情：{_clip(scratchpad, 240) or '（空）'}")
            if isinstance(cursor, dict) and cursor:
                lines.append(
                    "  cursor："
                    + _clip(json.dumps(cursor, ensure_ascii=False, sort_keys=True), 240)
                )

        lines.append("")
        lines.append("最近 Drift runs：")
        if not recent_rows:
            lines.append("- （空）")
        for row in recent_rows[-5:][::-1]:
            status = str(row.get("status") or "")
            message_result = str(row.get("message_result") or "silent")
            lines.append(
                f"- {row.get('run_at', '')[:16]}  {row.get('skill', '')} "
                f"[{status}/{message_result}] {_clip(row.get('briefing', ''), 150)}"
            )
        return "\n".join(lines)

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
        for root in self.plugin_skill_roots:
            plugin_dir = root / name
            if (plugin_dir / "SKILL.md").exists():
                return plugin_dir
        return None

    def save_finish(
        self,
        *,
        skill_used: str,
        status: str,
        briefing: str,
        message_result: str,
        scratchpad_update: str | None,
        global_note_update: str | None,
        now_utc: datetime,
        cursor_update: dict[str, Any] | None = None,
        journal_append: list[dict[str, Any]] | None = None,
        self_update: dict[str, str] | None = None,
    ) -> None:
        skill_name = str(skill_used or "").strip()
        status_value = str(status or "").strip()
        if status_value not in {"completed", "paused"}:
            raise ValueError("drift status must be completed or paused")
        logger.info(
            "[drift_state] save_finish: skill=%s status=%s note=%s",
            skill_name,
            status_value,
            bool(global_note_update),
        )

        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO runs (run_at, skill_name, status, briefing, message_result)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    now_utc.isoformat(),
                    skill_name,
                    status_value,
                    _clip(briefing, 500),
                    message_result,
                ),
            )
            run_id = int(cursor.lastrowid or 0)
            if run_id:
                self._last_saved_run_id = run_id
                self._last_saved_run_at = now_utc.isoformat()
                _ = conn.execute(
                    """
                    UPDATE run_steps
                    SET run_id = ?
                    WHERE run_id IS NULL
                      AND created_at = ?
                    """,
                    (
                        run_id,
                        now_utc.isoformat(),
                    ),
                )
            row = conn.execute(
                """
                SELECT run_count, scratchpad, cursor_json
                FROM skill_continuum
                WHERE skill_name = ?
                """,
                (skill_name,),
            ).fetchone()
            old_count = int(row["run_count"] or 0) if row is not None else 0
            old_scratchpad = str(row["scratchpad"] or "") if row is not None else ""
            old_cursor = self._decode_json_object(row["cursor_json"]) if row is not None else {}
            merged_cursor = self._merge_cursor(old_cursor, cursor_update)
            scratchpad = (
                _clip(scratchpad_update or "", 2000)
                if scratchpad_update is not None and str(scratchpad_update).strip()
                else old_scratchpad
            )
            _ = conn.execute(
                """
                INSERT INTO skill_continuum (
                    skill_name, run_count, last_run_at, last_status,
                    last_briefing, scratchpad, cursor_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_name) DO UPDATE SET
                    run_count = excluded.run_count,
                    last_run_at = excluded.last_run_at,
                    last_status = excluded.last_status,
                    last_briefing = excluded.last_briefing,
                    scratchpad = excluded.scratchpad,
                    cursor_json = excluded.cursor_json,
                    updated_at = excluded.updated_at
                """,
                (
                    skill_name,
                    old_count + 1,
                    now_utc.isoformat(),
                    status_value,
                    _clip(briefing, 500),
                    scratchpad,
                    json.dumps(merged_cursor, ensure_ascii=False, sort_keys=True),
                    now_utc.isoformat(),
                ),
            )
            self._append_journal_entries(
                conn=conn,
                skill_name=skill_name,
                run_id=run_id or None,
                entries=journal_append or [],
                created_at=now_utc.isoformat(),
            )
            if self_update is not None:
                self_state = self._load_self_state_from_connection(conn)
                self._upsert_self_state(
                    conn=conn,
                    current_skill=skill_name,
                    current_intention=str(
                        self_update.get("current_intention")
                        or self_state.get("current_intention")
                        or ""
                    ),
                    last_decision=str(self_state.get("last_decision") or ""),
                    decision_reason=str(self_state.get("decision_reason") or ""),
                    next_tendency=str(self_update.get("next_tendency") or ""),
                    updated_at=now_utc.isoformat(),
                )
            if global_note_update is not None and str(global_note_update).strip():
                _ = conn.execute(
                    """
                    INSERT INTO global_note (id, content, updated_at)
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        content = excluded.content,
                        updated_at = excluded.updated_at
                    """,
                    (
                        _clip(global_note_update, 1000),
                        now_utc.isoformat(),
                    ),
                )

    def update_last_message_result(self, message_result: str) -> None:
        run_id = self._last_saved_run_id
        if not run_id:
            return
        with self._connection() as conn:
            _ = conn.execute(
                "UPDATE runs SET message_result = ? WHERE id = ?",
                (message_result, run_id),
            )

    def append_step(
        self,
        *,
        step_index: int,
        tool_name: str,
        input_preview: str,
        output_preview: str,
        now_utc: datetime,
    ) -> None:
        created_at = now_utc.isoformat()
        run_id = (
            self._last_saved_run_id
            if created_at == self._last_saved_run_at
            else None
        )
        with self._connection() as conn:
            _ = conn.execute(
                """
                INSERT INTO run_steps (
                    run_id, step_index, tool_name,
                    input_preview, output_preview, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    max(0, int(step_index)),
                    _clip(tool_name, 120),
                    _clip(input_preview, 500),
                    _clip(output_preview, 500),
                    created_at,
                ),
            )

    def _load_skill_state(self, skill_dir: Path) -> dict[str, Any]:
        raw = cast(
            dict[str, Any],
            load_json(skill_dir / "state.json", default=None, domain="drift_state") or {},
        )
        return raw if isinstance(raw, dict) else {}

    @staticmethod
    def _normalize_status(raw: Any) -> str:
        status = str(raw or "").strip()
        return status if status in {"idle", "completed", "paused"} else "idle"

    def _skill_roots(self) -> list[tuple[Path, bool]]:
        roots: list[tuple[Path, bool]] = [(self.skills_dir, False)]
        if self.include_builtin_skills and self.builtin_skills_dir is not None:
            roots.append((self.builtin_skills_dir, True))
        roots.extend((root, False) for root in self.plugin_skill_roots)
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
            requires_mcp = [
                text
                for item in cast(list[object], requires_mcp_val)
                if (text := str(item).strip())
            ]
        else:
            raw = str(requires_mcp_val or "").strip()
            requires_mcp = [s.strip() for s in raw.split(",") if s.strip()] if raw else []
        continuum = self._load_continuum(name)
        raw_state = self._load_skill_state(skill_dir)
        last_run_at = parse_iso(continuum.get("last_run_at")) or parse_iso(raw_state.get("last_run_at"))
        run_count = continuum.get("run_count", raw_state.get("run_count", 0))
        status = continuum.get("last_status", "idle")
        return SkillMeta(
            name=name,
            description=description,
            last_run_at=last_run_at,
            run_count=max(0, int(run_count or 0)),
            status=self._normalize_status(status),
            next="",
            requires_mcp=requires_mcp,
            builtin=builtin,
        )

    def _ensure_db(self) -> None:
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            _ = conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    briefing TEXT NOT NULL,
                    message_result TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS skill_continuum (
                    skill_name TEXT PRIMARY KEY,
                    run_count INTEGER NOT NULL DEFAULT 0,
                    last_run_at TEXT,
                    last_status TEXT NOT NULL DEFAULT 'idle',
                    last_briefing TEXT NOT NULL DEFAULT '',
                    scratchpad TEXT NOT NULL DEFAULT '',
                    cursor_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS skill_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_name TEXT NOT NULL,
                    entry_type TEXT NOT NULL,
                    key TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    run_id INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_skill_journal_skill_type_key
                    ON skill_journal(skill_name, entry_type, key);

                CREATE INDEX IF NOT EXISTS idx_skill_journal_run_id
                    ON skill_journal(run_id);

                CREATE TABLE IF NOT EXISTS global_note (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    content TEXT NOT NULL DEFAULT '',
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS self_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_skill TEXT NOT NULL DEFAULT '',
                    current_intention TEXT NOT NULL DEFAULT '',
                    last_decision TEXT NOT NULL DEFAULT '',
                    decision_reason TEXT NOT NULL DEFAULT '',
                    next_tendency TEXT NOT NULL DEFAULT '',
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS run_steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    step_index INTEGER NOT NULL,
                    tool_name TEXT NOT NULL,
                    input_preview TEXT NOT NULL DEFAULT '',
                    output_preview TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                """
            )
            self._ensure_skill_continuum_columns(conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_file))
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _load_recent_runs_from_db(self, *, limit: int) -> list[dict[str, str]]:
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT run_at, skill_name, status, briefing, message_result
                FROM runs
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        result: list[dict[str, str]] = []
        for row in reversed(rows):
            result.append(
                {
                    "skill": _clip(row["skill_name"], 80),
                    "run_at": _clip(row["run_at"], 80),
                    "status": _clip(self._normalize_status(row["status"]), 20),
                    "briefing": _clip(row["briefing"], 150),
                    "message_result": _clip(row["message_result"], 20),
                }
            )
        return result

    def _load_global_note(self) -> str:
        with self._connection() as conn:
            row = conn.execute(
                "SELECT content FROM global_note WHERE id = 1"
            ).fetchone()
        return str(row["content"] or "") if row is not None else ""

    def _load_continuum(self, skill_name: str) -> dict[str, Any]:
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT run_count, last_run_at, last_status, last_briefing,
                       scratchpad, cursor_json, updated_at
                FROM skill_continuum
                WHERE skill_name = ?
                """,
                (skill_name,),
            ).fetchone()
        if row is None:
            return {}
        return {
            "run_count": int(row["run_count"] or 0),
            "last_run_at": str(row["last_run_at"] or ""),
            "last_status": self._normalize_status(row["last_status"]),
            "last_briefing": str(row["last_briefing"] or ""),
            "scratchpad": str(row["scratchpad"] or ""),
            "cursor": self._decode_json_object(row["cursor_json"]),
            "updated_at": str(row["updated_at"] or ""),
        }

    @staticmethod
    def _load_self_state_from_connection(conn: sqlite3.Connection) -> dict[str, str]:
        row = conn.execute(
            """
            SELECT current_intention, last_decision, decision_reason
            FROM self_state
            WHERE id = 1
            """
        ).fetchone()
        if row is None:
            return {}
        return {
            "current_intention": str(row["current_intention"] or ""),
            "last_decision": str(row["last_decision"] or ""),
            "decision_reason": str(row["decision_reason"] or ""),
        }

    @staticmethod
    def _upsert_self_state(
        *,
        conn: sqlite3.Connection,
        current_skill: str,
        current_intention: str,
        last_decision: str,
        decision_reason: str,
        next_tendency: str,
        updated_at: str,
    ) -> None:
        _ = conn.execute(
            """
            INSERT INTO self_state (
                id, current_skill, current_intention, last_decision,
                decision_reason, next_tendency, updated_at
            )
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                current_skill = excluded.current_skill,
                current_intention = excluded.current_intention,
                last_decision = excluded.last_decision,
                decision_reason = excluded.decision_reason,
                next_tendency = excluded.next_tendency,
                updated_at = excluded.updated_at
            """,
            (
                _clip(current_skill, 80),
                _clip(current_intention, 500),
                _clip(last_decision, 40),
                _clip(decision_reason, 500),
                _clip(next_tendency, 500),
                updated_at,
            ),
        )

    @staticmethod
    def _decode_json_object(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return cast(dict[str, Any], raw)
        try:
            data: Any = json.loads(str(raw or "{}"))
        except json.JSONDecodeError:
            return {}
        return cast(dict[str, Any], data) if isinstance(data, dict) else {}

    @staticmethod
    def _merge_cursor(
        old_cursor: dict[str, Any],
        cursor_update: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = dict(old_cursor)
        if not cursor_update:
            return merged
        for key, value in cursor_update.items():
            clean_key = str(key)
            if value is None:
                merged.pop(clean_key, None)
            else:
                merged[clean_key] = value
        return merged

    @staticmethod
    def _ensure_skill_continuum_columns(conn: sqlite3.Connection) -> None:
        rows = conn.execute("PRAGMA table_info(skill_continuum)").fetchall()
        columns = {str(row["name"]) for row in rows}
        if "cursor_json" not in columns:
            _ = conn.execute(
                "ALTER TABLE skill_continuum ADD COLUMN cursor_json TEXT NOT NULL DEFAULT '{}'"
            )

    @staticmethod
    def _append_journal_entries(
        *,
        conn: sqlite3.Connection,
        skill_name: str,
        run_id: int | None,
        entries: list[dict[str, Any]],
        created_at: str,
    ) -> None:
        for entry in entries:
            entry_type = str(entry.get("entry_type") or "").strip()
            if not entry_type:
                continue
            key = str(entry.get("key") or "").strip()
            payload = entry.get("payload")
            payload_json = json.dumps(
                payload if isinstance(payload, dict) else {},
                ensure_ascii=False,
                sort_keys=True,
            )
            _ = conn.execute(
                """
                INSERT INTO skill_journal (
                    skill_name, entry_type, key, payload_json, run_id, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (skill_name, entry_type, key, payload_json, run_id, created_at),
            )
