from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import Config
from agent.memory import DEFAULT_SELF_MD, MemoryStore
from agent.persona import VEDA_RELATIVE_PATH, read_default_veda
from infra.persistence.json_store import save_json
from session.store import SessionStore

_EMPTY_FILES: dict[str, str] = {
    "memory/MEMORY.md": "",
    "memory/PENDING.md": "",
}

_TEXT_FILES: dict[str, str] = {
    **_EMPTY_FILES,
    VEDA_RELATIVE_PATH.as_posix(): read_default_veda() + "\n",
    "memory/SELF.md": DEFAULT_SELF_MD,
}

_JSON_FILES: dict[str, object] = {
    "memes/manifest.json": {"categories": {}},
}

_DIRECTORIES: tuple[str, ...] = (
    "observe",
    "skills",
    "drift/skills",
    "mcp",
    "mcp/servers",
)


@dataclass
class InitSummary:
    created: list[Path] = field(default_factory=list)
    overwritten: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


def _write_text_file(
    path: Path, content: str, *, force: bool, summary: InitSummary
) -> None:
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


def _write_json_file(
    path: Path, payload: object, *, force: bool, summary: InitSummary
) -> None:
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
        _write_text_file(
            workspace / rel_path,
            content,
            force=force and rel_path != VEDA_RELATIVE_PATH.as_posix(),
            summary=summary,
        )


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

    if not config.memory.enabled:
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

    config = Config.load(config_path, workspace=workspace)
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
        '     [llm.main]  api_key = "sk-..."',
        '     [channels.telegram]  token = "..."   （或配置 QQ 频道）',
        '     [memory.embedding]  api_key = "sk-..."',
        "2. 运行 uv run python main.py 启动。",
        "3. 打开 http://127.0.0.1:2236 使用 Web Chat，或向 bot 发一条消息。",
        "4. 确认对话正常后，可按需启用普通 v3 插件。",
    ]
    return summary
