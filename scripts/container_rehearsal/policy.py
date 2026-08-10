"""Own rehearsal source boundaries and exclusion policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.container_rehearsal.model import is_relative_to

EXCLUDED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".tmp",
        ".venv",
        "backups",
        "cache",
        "downloads",
        "mcp-env",
        "rebuilds",
        "runtime",
        "sandboxes",
        "subagent-runs",
        "venv",
    }
)
EXCLUDED_RUNTIME_FILES = frozenset(
    {
        ".app-server-token",
        ".instance.lock",
        ".runtime-ready.json",
        ".supervisor.lock",
        ".supervisor.pid",
        "akashic.sock",
    }
)
WEBUI_ONLY_SETTINGS: tuple[tuple[str, str, Any], ...] = (
    ("channels.chat", "enabled", True),
    ("channels.telegram", "enabled", False),
    ("channels.telegram", "token", ""),
    ("channels.qq", "enabled", False),
    ("channels.qq", "bot_uin", ""),
    ("mobile_realtime", "enabled", False),
    ("mobile_realtime", "public_url", ""),
    ("proactive", "enabled", False),
    ("proactive.target", "channel", ""),
    ("proactive.target", "chat_id", ""),
    ("proactive.drift", "enabled", False),
)


def validate_roots(
    *, source_workspace: Path, source_config: Path, plugin_home: Path, target: Path
) -> tuple[Path, Path, Path, Path]:
    """解析全部边界路径，并拒绝覆盖或递归复制。"""

    # 1. 源必须已经存在，目标必须尚未创建。
    source_workspace = source_workspace.expanduser().resolve(strict=True)
    source_config = source_config.expanduser().resolve(strict=True)
    plugin_home = plugin_home.expanduser().resolve(strict=True)
    target = target.expanduser().absolute()
    if not source_workspace.is_dir():
        raise NotADirectoryError(source_workspace)
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    if not plugin_home.is_dir():
        raise NotADirectoryError(plugin_home)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"预演目标必须是尚不存在的新路径: {target}")

    # 2. 目标不能与任何源重叠，避免读写正式状态或递归吞入副本。
    target_resolved = target.resolve(strict=False)
    for label, source in (
        ("Workspace", source_workspace),
        ("配置", source_config),
        ("插件目录", plugin_home),
    ):
        if target_resolved == source or is_relative_to(target_resolved, source):
            raise ValueError(f"预演目标不能位于源{label}内部: {target}")
        if source.is_dir() and is_relative_to(source, target_resolved):
            raise ValueError(f"源{label}不能位于预演目标内部: {source}")
    return source_workspace, source_config, plugin_home, target


def excluded_reason(relative: Path, *, is_symlink: bool) -> str | None:
    """返回排除原因；未命中时允许该路径进入隔离副本。"""

    parts = relative.parts
    name = relative.name
    if any(part in EXCLUDED_DIRECTORY_NAMES for part in parts):
        return "excluded_state_class"
    if any(part.endswith("_rebuild") for part in parts):
        return "rebuild_artifact"
    if any(part.startswith("mobile-webui-build-") for part in parts):
        return "temporary_webui_build"
    if len(parts) >= 2 and parts[-2:] in {
        ("mobile-webui", "staging"),
        ("mobile-webui", "trash"),
    }:
        return "temporary_webui_state"
    if name in EXCLUDED_RUNTIME_FILES:
        return "runtime_control_file"
    if ".corrupt." in name:
        return "forensic_corrupt_artifact"
    if name.endswith(("-wal", "-shm", "-journal")):
        return "sqlite_sidecar"
    if is_symlink and (parts[:1] == ("skills",) or parts[:2] == ("drift", "skills")):
        return "rebuildable_skill_projection"
    return None
