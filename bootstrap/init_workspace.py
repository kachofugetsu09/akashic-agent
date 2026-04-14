from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import Config
from agent.memory import MemoryStore
from core.observe.db import open_db as open_observe_db
from infra.persistence.json_store import save_json
from memory2.store import MemoryStore2
from proactive_v2.anyaction import QuotaStore
from proactive_v2.loop import ProactiveLoop
from proactive_v2.state import ProactiveStateStore
from session.store import SessionStore

_DEFAULT_NOW = """# Now

## 近期进行中

## 待确认事项
"""

_EMPTY_FILES: dict[str, str] = {
    "memory/MEMORY.md": "",
    "memory/SELF.md": "",
    "memory/HISTORY.md": "",
    "memory/RECENT_CONTEXT.md": "",
    "memory/PENDING.md": "",
}

_TEXT_FILES: dict[str, str] = {
    **_EMPTY_FILES,
    "memory/NOW.md": _DEFAULT_NOW,
    "PROACTIVE_CONTEXT.md": ProactiveLoop._PROACTIVE_CONTEXT_TEMPLATE,
}

_JSON_FILES: dict[str, object] = {
    "mcp_servers.json": {"servers": {}},
    "schedules.json": [],
    "proactive_sources.json": {"sources": []},
    "memes/manifest.json": {"categories": {}},
}


@dataclass
class InitSummary:
    created: list[Path] = field(default_factory=list)
    overwritten: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _write_text_file(path: Path, content: str, *, force: bool, summary: InitSummary) -> None:
    existed = path.exists()
    if existed and not force:
        summary.skipped.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if existed:
        summary.overwritten.append(path)
    else:
        summary.created.append(path)


def _write_json_file(path: Path, payload: object, *, force: bool, summary: InitSummary) -> None:
    existed = path.exists()
    if existed and not force:
        summary.skipped.append(path)
        return
    save_json(path, payload, domain="workspace.init")
    if existed:
        summary.overwritten.append(path)
    else:
        summary.created.append(path)


def _ensure_config(config_path: Path, *, force: bool, summary: InitSummary) -> None:
    template = Path(__file__).resolve().parent.parent / "config.example.toml"
    existed = config_path.exists()
    if existed and not force:
        summary.skipped.append(config_path)
        return
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, config_path)
    if existed:
        summary.overwritten.append(config_path)
    else:
        summary.created.append(config_path)


def _ensure_workspace_text_assets(
    workspace: Path,
    *,
    force: bool,
    summary: InitSummary,
) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    for rel_path, content in _TEXT_FILES.items():
        _write_text_file(workspace / rel_path, content, force=force, summary=summary)


def _ensure_workspace_json_assets(
    workspace: Path,
    *,
    force: bool,
    summary: InitSummary,
) -> None:
    for rel_path, payload in _JSON_FILES.items():
        _write_json_file(workspace / rel_path, payload, force=force, summary=summary)


def _ensure_workspace_db_assets(
    workspace: Path,
    *,
    memory_enabled: bool,
    summary: InitSummary,
) -> None:
    sessions_db = workspace / "sessions.db"
    sessions_exists = sessions_db.exists()
    SessionStore(sessions_db).close()
    if not sessions_exists:
        summary.created.append(sessions_db)
    else:
        summary.skipped.append(sessions_db)

    observe_db = workspace / "observe" / "observe.db"
    observe_exists = observe_db.exists()
    open_observe_db(observe_db).close()
    if not observe_exists:
        summary.created.append(observe_db)
    else:
        summary.skipped.append(observe_db)

    consolidation_db = workspace / "memory" / "consolidation_writes.db"
    consolidation_exists = consolidation_db.exists()
    MemoryStore(workspace)
    if not consolidation_exists:
        summary.created.append(consolidation_db)
    else:
        summary.skipped.append(consolidation_db)

    state_path = workspace / "proactive_state.json"
    quota_path = workspace / "proactive_quota.json"
    if not state_path.exists():
        save_json(
            state_path,
            ProactiveStateStore._empty_state(),
            domain="workspace.init",
        )
        summary.created.append(state_path)
    else:
        summary.skipped.append(state_path)
    if not quota_path.exists():
        save_json(
            quota_path,
            QuotaStore(quota_path)._state,
            domain="workspace.init",
        )
        summary.created.append(quota_path)
    else:
        summary.skipped.append(quota_path)

    if memory_enabled:
        memory2_db = workspace / "memory" / "memory2.db"
        memory2_exists = memory2_db.exists()
        MemoryStore2(memory2_db).close()
        if not memory2_exists:
            summary.created.append(memory2_db)
        else:
            summary.skipped.append(memory2_db)
    else:
        summary.notes.append("memory.enabled = false，未预创建 memory/memory2.db。")


def _ensure_fitbit_assets(*, force: bool, summary: InitSummary) -> None:
    base_dir = Path(__file__).resolve().parent.parent / "scripts" / "fitbit-monitor"
    example = base_dir / "monitor.config.example.toml"
    target = base_dir / "monitor.config.local.toml"
    if not example.exists():
        summary.notes.append("未找到 Fitbit 模板，跳过 with-fitbit。")
        return
    _write_text_file(target, example.read_text(encoding="utf-8"), force=force, summary=summary)
    summary.notes.append("Fitbit 仅初始化本地配置模板，未创建 tokens.json 等运行期文件。")


def init_workspace(
    *,
    config_path: str | Path = "config.toml",
    workspace: Path,
    force: bool = False,
    with_fitbit: bool = False,
) -> InitSummary:
    summary = InitSummary()
    config_path = Path(config_path)

    _ensure_config(config_path, force=force, summary=summary)

    config = Config.load(config_path)
    _ensure_workspace_text_assets(workspace, force=force, summary=summary)
    _ensure_workspace_json_assets(workspace, force=force, summary=summary)
    _ensure_workspace_db_assets(
        workspace,
        memory_enabled=bool(config.memory_v2.enabled),
        summary=summary,
    )

    if with_fitbit:
        _ensure_fitbit_assets(force=force, summary=summary)

    summary.notes.append(f"请检查并填写配置文件: {config_path}")
    summary.notes.append(f"工作区已初始化: {workspace}")
    return summary
