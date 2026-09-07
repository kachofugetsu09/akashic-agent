from __future__ import annotations

import shutil
import os
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import Config
from agent.persona import VEDA_RELATIVE_PATH, read_default_veda
from infra.persistence.json_store import save_json
from session.log import MessageLog

_TEXT_FILES: dict[str, str] = {
    VEDA_RELATIVE_PATH.as_posix(): read_default_veda() + "\n",
    "plugin-data/context-builtin/config.local.toml": (
        'prompt_sources = {default_prompt = "prompt", markdown_memory = "markdown_memory"}\n'
        'summary_source = ["compaction", "compaction"]\n'
    ),
}

_JSON_FILES: dict[str, object] = {
    "memes/manifest.json": {"categories": {}},
}

_DIRECTORIES: tuple[str, ...] = (
    "observe",
    "skills",
    "drift/skills",
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
    if existed:
        # --force 只重置配置模板，旧凭据配置必须有独立恢复文件。
        before = config_path.read_bytes()
        backup = config_path.with_name(config_path.name + ".before-init-" + uuid4().hex + ".bak")
        with os.fdopen(os.open(backup, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
            _ = output.write(before)
            output.flush()
            os.fsync(output.fileno())
        if backup.read_bytes() != before or config_path.read_bytes() != before:
            raise RuntimeError("初始化配置备份不一致或源配置已变化")
        summary.notes.append(f"原配置恢复文件: {backup}")
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
            force=False,
            summary=summary,
        )


def _ensure_workspace_json_assets(
    workspace: Path,
    *,
    force: bool,
    summary: InitSummary,
) -> None:
    for rel_path, payload in _JSON_FILES.items():
        _write_json_file(workspace / rel_path, payload, force=False, summary=summary)


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
    summary: InitSummary,
) -> None:
    sessions_db = workspace / "sessions.db"
    sessions_exists = sessions_db.exists()
    MessageLog(sessions_db).close()
    if not sessions_exists:
        summary.created.append(sessions_db)
    else:
        summary.skipped.append(sessions_db)

def init_workspace(
    *,
    config_path: str | Path = "config.toml",
    workspace: Path,
    force: bool = False,
) -> InitSummary:
    summary = InitSummary()
    config_path = Path(config_path)

    _ensure_config(config_path, force=force, summary=summary)

    _ = Config.load(config_path, workspace=workspace)
    _ensure_workspace_text_assets(workspace, force=force, summary=summary)
    _ensure_workspace_json_assets(workspace, force=force, summary=summary)
    _ensure_workspace_directories(workspace, summary=summary)
    _ensure_workspace_db_assets(
        workspace,
        summary=summary,
    )

    summary.notes.append(f"工作区已初始化: {workspace}")
    summary.next_steps = [
        f"1. 默认可用 Web；启用 Telegram/QQ 时，编辑 {config_path} 并配置同名 Sender 插件。",
        "2. 运行 uv run python main.py 启动。",
        "3. 打开 http://127.0.0.1:2236，在模型页添加连接并选择默认聊天模型。",
        "4. 需要语义记忆时，再选择默认 embedding 模型。",
        "5. 返回对话页验证消息收发；Sender 配置路径与示例见 config.example.toml。",
    ]
    return summary
