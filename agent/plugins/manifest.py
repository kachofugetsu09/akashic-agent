from __future__ import annotations

import json
import os
import re
from pathlib import Path



def plugins_root(plugins_home: Path | None = None) -> Path:
    if plugins_home is not None:
        return plugins_home
    configured = os.environ.get("AKASHIC_PLUGIN_HOME")
    if configured is not None:
        configured = configured.strip()
        if not configured:
            raise ValueError("AKASHIC_PLUGIN_HOME 不能为空")
        return Path(configured).expanduser().resolve(strict=False)
    return Path.home() / ".akashic-plugin"


def workspace_plugin_data_dir(
    workspace: Path,
    plugin_name: str,
    marketplace: str,
) -> Path:
    """解析 workspace 内的插件数据目录，不创建或迁移数据。"""

    # 1. 插件身份只能形成单一路径段
    for label, value in (("name", plugin_name), ("marketplace", marketplace)):
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
            raise ValueError(f"插件 {label} 不是安全路径段: {value!r}")

    # 2. 数据根始终归属于显式 workspace
    return workspace.resolve(strict=False) / "plugin-data" / f"{plugin_name}-{marketplace}"


# 实例是否已归一化的标记；状态不是 normalized 时说明归一化或还原被中断，启动必须拒绝。
LAYOUT_RECEIPT = "runtime/plugin-data-layout.json"


def read_layout_state(workspace: Path) -> str | None:
    path = workspace / LAYOUT_RECEIPT
    if not path.exists():
        return None
    state = json.loads(path.read_text(encoding="utf-8")).get("state")
    return state if isinstance(state, str) else "invalid"


# 选择记录允许的两种数据目录形态：旧形态 <name>-<marketplace>，归一化形态 <name>。
def plugin_data_labels(plugin_id: str) -> tuple[str, str]:
    name, _, marketplace = plugin_id.partition("@")
    legacy = workspace_plugin_data_dir(Path("/"), name, marketplace or "builtin").name
    return f"plugin-data/{legacy}", f"plugin-data/{name}"


# 没有选择记录时的默认目录：已归一化的实例用 <name>，否则与旧版一致用 <name>-<marketplace>。
def default_plugin_data_label(workspace: Path, plugin_id: str) -> str:
    legacy, normalized = plugin_data_labels(plugin_id)
    return normalized if read_layout_state(workspace) == "normalized" else legacy


def ensure_workspace_plugin_data_dir(path: Path, workspace: Path) -> None:
    """安全创建 workspace 内的数据目录，并拒绝中间符号链接。"""

    validate_workspace_plugin_data_path(path, workspace)
    path.mkdir(parents=True, exist_ok=True)
    validate_workspace_plugin_data_path(path, workspace)


def validate_workspace_plugin_data_path(path: Path, workspace: Path) -> None:
    """校验插件数据路径归属 workspace，且现有路径不穿过符号链接。"""

    # 1. 校验目标仍归属于 workspace
    root = workspace.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"插件数据目录越界: {path}") from error

    # 2. 逐级拒绝现有符号链接
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"插件数据目录不能穿过符号链接: {current}")


def installed_plugin_ids(plugins_home: Path | None = None) -> frozenset[str]:
    """实际安装目录拥有制品库存，不另写一份插件清单。"""
    from agent.plugins.source_resolver import resolve_plugin_sources
    return frozenset(f"{source.plugin_name}@{source.marketplace}" for source in
                     resolve_plugin_sources(installed_cache_root=plugins_root(plugins_home) / "cache"))
