"""
交互式初始化向导。

python main.py setup
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    plugins_root,
    workspace_plugin_data_dir,
)
from agent.plugins.source_resolver import resolve_plugin_sources


def _hint(text: str) -> None:
    click.echo(click.style(f"  {text}", dim=True))


def _ok(text: str) -> None:
    click.echo(click.style(f"  ✓ {text}", fg="green"))


def _err(text: str) -> None:
    click.echo(click.style(f"  ✗ {text}", fg="red"))


def _divider() -> None:
    click.echo(click.style("─" * 40, dim=True))


def run_setup_wizard(config_path: Path, workspace: Path) -> None:
    """Write the Core config and run setup commands declared by installed plugins."""

    click.echo(click.style("\n══ akashic 初始化向导 ══\n", bold=True))
    _hint("全程按回车使用括号内的默认值")
    _hint("已安装插件会在自己的配置命令中询问凭据和其他私有设置")

    if config_path.exists():
        click.echo(f"\n已存在配置文件 {config_path}")
        if not click.confirm("覆盖并重新配置？", default=False):
            click.echo("已取消。")
            return

    _divider()
    click.echo("\n正在生成 Core 配置并运行插件配置命令...")

    _atomic_write_with_backup(config_path, _render_config(), mode=0o600)
    _ok(f"{config_path} 已生成")
    _run_declared_plugin_setups(workspace)

    _validate_config(config_path, workspace)

    from bootstrap.init_workspace import init_workspace

    _ = init_workspace(config_path=config_path, workspace=workspace)
    _ok(f"{workspace} 已初始化")

    _print_completion(workspace)


def _run_declared_plugin_setups(workspace: Path) -> None:
    """Run each plugin-owned setup entrypoint from its validated manifest."""

    builtin_root = Path(__file__).resolve().parents[1] / "plugins"
    sources = resolve_plugin_sources(
        (builtin_root,),
        installed_cache_root=plugins_root() / "cache",
        installed_selector="stable",
    )
    for source in sources:
        manifest = source.static_manifest
        if manifest is None or manifest.setup is None:
            continue
        marketplace = source.marketplace or "builtin"
        data_dir = workspace_plugin_data_dir(workspace, manifest.name, marketplace)
        ensure_workspace_plugin_data_dir(data_dir, workspace)
        plugin_root = source.plugin_root.resolve(strict=True)
        setup_path = (plugin_root / manifest.setup.entrypoint).resolve(strict=True)
        if not setup_path.is_relative_to(plugin_root) or not setup_path.is_file():
            raise RuntimeError(
                f"插件 {manifest.name} setup.entrypoint 不在 artifact 内: {setup_path}"
            )
        environment = os.environ.copy()
        environment.update(
            {
                "AKASHIC_PLUGIN_ROOT": str(plugin_root),
                "AKASHIC_PLUGIN_ID": (
                    f"{manifest.name}@{marketplace}"
                    if source.marketplace
                    else manifest.name
                ),
                "AKASHIC_PLUGIN_DATA_DIR": str(data_dir),
                "AKASHIC_SETUP_CONFIG_PATH": str(data_dir / "config.local.toml"),
            }
        )
        pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (str(plugin_root), pythonpath) if item
        )
        _ok(f"运行插件配置：{manifest.name}")
        try:
            result = subprocess.run(
                [sys.executable, str(setup_path)],
                cwd=plugin_root,
                env=environment,
                check=False,
            )
        except OSError as error:
            raise RuntimeError(
                f"插件 {manifest.name} 配置命令无法启动: {error}"
            ) from error
        if result.returncode != 0:
            raise RuntimeError(
                f"插件 {manifest.name} 配置命令失败: exit={result.returncode}"
            )


def _validate_config(config_path: Path, workspace: Path) -> None:
    """Validate the generated Core configuration before workspace initialization."""

    try:
        from agent.config import Config

        _ = Config.load(config_path, workspace=workspace)
        _ok("配置验证通过")
    except KeyError as error:
        _err(f"配置缺少必填字段：{error}")
        raise SystemExit(1) from error
    except Exception as error:
        _err(f"配置加载失败：{error}")
        raise SystemExit(1) from error


def _render_config() -> str:
    return _render_channels()


def _render_channels() -> str:
    return "\n".join(
        [
            "# Web Chat 由 Supervisor 在唯一入口 2236 提供。",
            "[channels.chat]",
            "enabled = true",
            "",
            "# 外部 channel 插件的配置由各自的 setup 声明写入 workspace/plugin-data。",
            "",
        ]
    )


def _atomic_write_with_backup(
    path: Path,
    content: str,
    *,
    mode: int = 0o644,
    backup_name: str | None = None,
) -> None:
    """备份旧文件后 fsync 并原子替换目标配置。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        backup = path.with_name(backup_name or f"{path.name}.before-setup.bak")
        shutil.copy2(path, backup)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, mode)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _print_completion(workspace: Path) -> None:
    click.echo(click.style("\n══ 配置完成 ══\n", bold=True))
    click.echo("启动 agent：")
    click.echo(click.style("  uv run python main.py", bold=True))
    _hint("启动后打开 2236 的“模型”页添加连接并选择默认模型")
    _hint(f"插件私有配置位于 {workspace / 'plugin-data'}")
