from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import Config
from agent.memory import DEFAULT_SELF_MD, MemoryStore
from bootstrap.memory import ensure_memory_plugin_storage
from infra.persistence.json_store import save_json
from proactive_v2.anyaction import QuotaStore
from proactive_v2.loop import ProactiveLoop
from proactive_v2.state import ProactiveStateStore
from session.store import SessionStore

_EMPTY_FILES: dict[str, str] = {
    "memory/MEMORY.md": "",
    "memory/HISTORY.md": "",
    "memory/RECENT_CONTEXT.md": "",
    "memory/PENDING.md": "",
}

_TEXT_FILES: dict[str, str] = {
    **_EMPTY_FILES,
    "memory/SELF.md": DEFAULT_SELF_MD,
    "PROACTIVE_CONTEXT.md": ProactiveLoop._PROACTIVE_CONTEXT_TEMPLATE,
}

_JSON_FILES: dict[str, object] = {
    "mcp_servers.json": {"servers": {}},
    "schedules.json": [],
    "proactive_sources.json": {"sources": []},
    "memes/manifest.json": {"categories": {}},
}

_DIRECTORIES: tuple[str, ...] = (
    "observe",
    "skills",
    "drift/skills",
    "mcp",
)


@dataclass
class InitSummary:
    created: list[Path] = field(default_factory=list)
    overwritten: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


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


def _ensure_workspace_directories(
    workspace: Path,
    *,
    summary: InitSummary,
) -> None:
    for rel_path in _DIRECTORIES:
        path = workspace / rel_path
        existed = path.exists()
        path.mkdir(parents=True, exist_ok=True)
        if existed:
            summary.skipped.append(path)
        else:
            summary.created.append(path)


def _ensure_workspace_db_assets(
    workspace: Path,
    *,
    config: Config,
    summary: InitSummary,
) -> None:
    sessions_db = workspace / "sessions.db"
    sessions_exists = sessions_db.exists()
    SessionStore(sessions_db).close()
    if not sessions_exists:
        summary.created.append(sessions_db)
    else:
        summary.skipped.append(sessions_db)

    consolidation_db = workspace / "memory" / "consolidation_writes.db"
    consolidation_exists = consolidation_db.exists()
    MemoryStore(workspace)
    if not consolidation_exists:
        summary.created.append(consolidation_db)
    else:
        summary.skipped.append(consolidation_db)

    proactive_db = workspace / "proactive.db"
    quota_path = workspace / "proactive_quota.json"
    proactive_exists = proactive_db.exists()
    ProactiveStateStore(proactive_db).close()
    if not proactive_exists:
        summary.created.append(proactive_db)
    else:
        summary.skipped.append(proactive_db)
    if not quota_path.exists():
        save_json(
            quota_path,
            QuotaStore(quota_path)._state,
            domain="workspace.init",
        )
        summary.created.append(quota_path)
    else:
        summary.skipped.append(quota_path)

    if config.memory.enabled:
        storage_results = ensure_memory_plugin_storage(config, workspace)
        if storage_results:
            for path, existed in storage_results:
                if existed:
                    summary.skipped.append(path)
                else:
                    summary.created.append(path)
        else:
            summary.notes.append("当前 memory engine 未声明 init 预创建逻辑，跳过语义记忆库。")
    else:
        summary.notes.append("memory.enabled = false，未预创建语义记忆库。")


def init_workspace(
    *,
    config_path: str | Path = "config.toml",
    workspace: Path,
    force: bool = False,
) -> InitSummary:
    summary = InitSummary()
    config_path = Path(config_path)

    _ensure_config(config_path, force=force, summary=summary)

    config = Config.load(config_path)
    _ensure_workspace_text_assets(workspace, force=force, summary=summary)
    _ensure_workspace_json_assets(workspace, force=force, summary=summary)
    _ensure_workspace_directories(workspace, summary=summary)
    _ensure_workspace_db_assets(
        workspace,
        config=config,
        summary=summary,
    )

    summary.notes.append(f"工作区已初始化: {workspace}")
    summary.next_steps = [
        f"1. 编辑 {config_path}，填写以下必填项：",
        "     [llm.main]  api_key = \"sk-...\"",
        "     [channels.telegram]  token = \"...\"   （或配置 QQ 频道）",
        "     [memory.embedding]  api_key = \"sk-...\"",
        "2. 运行 uv run python main.py 启动。",
        "3. 向 bot 发一条消息，确认对话正常后，可在 config.toml 开启 proactive。",
    ]
    return summary
