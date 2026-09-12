from __future__ import annotations

import shutil
import os
from uuid import uuid4
from dataclasses import dataclass, field
from pathlib import Path

from agent.config import Config

@dataclass
class InitSummary:
    created: list[Path] = field(default_factory=list)
    overwritten: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)


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
    workspace.mkdir(parents=True, exist_ok=True)

    summary.notes.append(f"工作区已初始化: {workspace}")
    summary.next_steps = [
        "1. 通过正式安装链安装所选插件组合，并按各包说明初始化业务配置。",
        "2. 运行 uv run python main.py 启动插件底座。",
    ]
    return summary
