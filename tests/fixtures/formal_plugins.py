"""测试用的正式插件组合安装夹具。"""

from __future__ import annotations

import shutil
import subprocess
import os
import sys
from pathlib import Path

from agent.plugins.install import PluginInstallResult, install_git_plugin
from agent.plugins.manifest import load_plugin_manifest
from agent.plugins.source_resolver import resolve_plugin_sources
from agent.plugins.static_manifest import load_static_plugin_manifest
from bootstrap.init_workspace import init_workspace


REPOSITORY = Path(__file__).resolve().parents[2]
MARKETPLACE = "fixture"

# 这些测试验证消息、回复和持久恢复。带外部 Python runtime 的入站渠道、
# 需要 Workload Controller 的 computer，以及它们的渠道构造都由专门测试覆盖。
MINIMAL_MESSAGE_PLUGINS = ("sources", "content", "models", "conversation")
FULL_RUNTIME_PLUGINS = (
    "akasha", "akashic_sender", "codex", "compaction", "content", "context",
    "conversation", "conversation_ui", "delivery", "delivery_policy", "drift",
    "eventmail", "legacy_upgrade", "markdown_memory", "message_push", "models",
    "openai_compatible", "opencode_go", "plugin_update", "programmatic", "prompt",
    "qq_sender", "react", "reply", "reply_program", "runtime_inspection", "runtime_ui",
    "scheduler", "shell_ui", "sources", "standard_tools", "standard_web", "subagent",
    "telegram_sender", "tool_search", "tools", "turn_projection", "wake", "workbench_ui",
)


def _git(repo: Path, *args: str) -> None:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"fixture Git 命令失败: {' '.join(args)}: {result.stderr}")


def install_formal_plugins(
    root: Path,
    names: tuple[str, ...],
    *,
    marketplace: str = MARKETPLACE,
    configure_materials: bool = False,
    initialize_persona: bool = False,
) -> tuple[Path, dict[str, PluginInstallResult]]:
    """把指定 checkout 插件逐个安装到临时 cache，并移走源副本。"""

    workspace = root / "workspace"
    plugin_home = root / "plugin-home"
    workspace.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.toml"
    if not config_path.exists():
        init_workspace(config_path=config_path, workspace=workspace)

    expected_ids = {
        f"{load_static_plugin_manifest(REPOSITORY / 'plugins' / name).name}@{marketplace}"
        for name in names
    }
    manifest = load_plugin_manifest(plugin_home)
    try:
        resolved = resolve_plugin_sources(
            (), installed_cache_root=plugin_home / "cache",
        )
    except (FileNotFoundError, ValueError):
        resolved = []
    existing_ids = {
        f"{item.plugin_name}@{item.marketplace}" for item in resolved
    }
    if expected_ids and expected_ids <= existing_ids and all(
        manifest.get(plugin_id, True) is True for plugin_id in expected_ids
    ):
        if configure_materials:
            _write_material_config(workspace, marketplace)
        if initialize_persona:
            _initialize_persona(workspace, plugin_home, marketplace)
        return plugin_home, {}
    if plugin_home.exists():
        shutil.rmtree(plugin_home)

    source_root = root / "formal-plugin-sources"
    if source_root.exists():
        shutil.rmtree(source_root)
    source_root.mkdir(parents=True, exist_ok=True)
    installed: dict[str, PluginInstallResult] = {}
    for name in names:
        source = source_root / name
        shutil.copytree(
            REPOSITORY / "plugins" / name,
            source,
            ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
        )
        _git(source, "init", "--quiet")
        _git(source, "config", "user.name", "formal-fixture")
        _git(source, "config", "user.email", "formal-fixture@example.invalid")
        _git(source, "add", ".")
        _git(source, "commit", "--quiet", "-m", "fixture source")
        installed[name] = install_git_plugin(
            workspace=workspace,
            source=str(source),
            marketplace=marketplace,
            plugins_home=plugin_home,
        )

    if configure_materials:
        _write_material_config(workspace, marketplace)
    if initialize_persona:
        _initialize_persona(workspace, plugin_home, marketplace, installed=installed)

    # The manager must prove it can load the installed archive without a checkout
    # source path.  Keep no second source tree that could hide accidental fallback.
    shutil.rmtree(source_root)
    return plugin_home, installed


def _write_material_config(workspace: Path, marketplace: str) -> None:
    """授予完整测试组合的真实材料 owner。"""

    data_path = workspace / "plugin-data" / f"context-{marketplace}"
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "config.local.toml").write_text(
        f"prompt_sources = {{default_prompt = \"prompt@{marketplace}\", "
        f"markdown_memory = \"markdown_memory@{marketplace}\", "
        f"skills = \"standard_tools@{marketplace}\"}}\n"
        f"summary_source = [\"compaction\", \"compaction@{marketplace}\"]\n",
        encoding="utf-8",
    )


def _initialize_persona(
    workspace: Path,
    plugin_home: Path,
    marketplace: str,
    *,
    installed: dict[str, PluginInstallResult] | None = None,
) -> None:
    """通过已安装 Prompt 包的维护命令建立测试人格文件。"""

    target = workspace / "memory" / "VEDA.md"
    if target.exists():
        return
    prompt = None if installed is None else installed.get("prompt")
    if prompt is None:
        sources = resolve_plugin_sources(
            (), installed_cache_root=plugin_home / "cache",
        )
        prompt_root = next(
            item.plugin_root for item in sources
            if item.plugin_name == "prompt" and item.marketplace == marketplace
        )
        script = prompt_root / "persona.py"
    else:
        script = prompt.installed_path / "persona.py"
    result = subprocess.run(
        [sys.executable, str(script), "--workspace", str(workspace)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONPATH": str(REPOSITORY / "sdk/python/src") + ":" + str(REPOSITORY),
        },
    )
    if result.returncode != 0:
        raise RuntimeError(f"正式 Prompt 包人格初始化失败: {result.stderr}")
