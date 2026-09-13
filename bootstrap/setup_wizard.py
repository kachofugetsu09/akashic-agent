"""
交互式初始化向导。

python main.py setup
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import click

from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    load_plugin_manifest,
    plugins_root,
    workspace_plugin_data_dir,
)
from agent.plugins.python_environment import (
    PythonEnvironments,
    read_environment_refs,
)
from agent.plugins.source_resolver import resolve_plugin_sources
from agent.plugins.static_manifest import (
    StaticPluginManifest,
    staged_python_interpreter,
)


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

    _atomic_write_with_backup(config_path, _render_config(workspace), mode=0o600)
    _ok(f"{config_path} 已生成")
    _validate_config(config_path, workspace)

    from bootstrap.init_workspace import init_workspace

    _ = init_workspace(config_path=config_path, workspace=workspace)
    _ok(f"{workspace} 已初始化")
    _run_declared_plugin_setups(workspace)

    _print_completion(workspace)


def _run_declared_plugin_setups(workspace: Path) -> None:
    """Run each plugin-owned setup entrypoint from its validated manifest."""

    plugin_home = plugins_root()
    enabled_plugins = load_plugin_manifest(plugin_home)
    cache_root = (plugin_home / "cache").resolve(strict=False)
    sources = resolve_plugin_sources(
        (),
        installed_cache_root=cache_root,
        installed_selector="stable",
    )
    for source in sources:
        manifest = source.static_manifest
        if manifest is None or manifest.setup is None:
            continue
        if source.source_type != "installed" or not source.marketplace:
            raise RuntimeError(f"插件 {manifest.name} setup 必须来自正式安装 artifact")
        if source.plugin_name != manifest.name:
            raise RuntimeError(
                f"插件 {manifest.name} installed cache identity 不一致: "
                f"{source.plugin_name}"
            )
        plugin_id = f"{manifest.name}@{source.marketplace}"
        if enabled_plugins.get(plugin_id, True) is False:
            _hint(f"跳过已禁用插件配置：{plugin_id}")
            continue
        plugin_root = source.plugin_root.resolve(strict=True)
        if not plugin_root.is_relative_to(cache_root):
            raise RuntimeError(
                f"插件 {manifest.name} setup 根不在 installed cache 内: {plugin_root}"
            )
        marketplace = source.marketplace
        data_dir = workspace_plugin_data_dir(workspace, manifest.name, marketplace)
        ensure_workspace_plugin_data_dir(data_dir, workspace)
        setup_path = (plugin_root / manifest.setup.entrypoint).resolve(strict=True)
        if not setup_path.is_relative_to(plugin_root) or not setup_path.is_file():
            raise RuntimeError(
                f"插件 {manifest.name} setup.entrypoint 不在 artifact 内: {setup_path}"
            )
        interpreter, code_root = _setup_runtime(
            workspace,
            plugin_root,
            manifest,
            setup_path,
        )
        environment = os.environ.copy()
        environment.update(
            {
                "AKASHIC_PLUGIN_ROOT": str(code_root),
                "AKASHIC_PLUGIN_ID": f"{manifest.name}@{marketplace}",
                "AKASHIC_PLUGIN_DATA_DIR": str(data_dir),
                "AKASHIC_SETUP_CONFIG_PATH": str(data_dir / "config.local.toml"),
            }
        )
        _ok(f"运行插件配置：{manifest.name}")
        try:
            result = subprocess.run(
                [
                    str(interpreter),
                    "-E",
                    "-s",
                    "-B",
                    str(code_root / manifest.setup.entrypoint),
                ],
                cwd=code_root,
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


def _setup_runtime(
    workspace: Path,
    plugin_root: Path,
    manifest: StaticPluginManifest,
    setup_path: Path,
) -> tuple[Path, Path]:
    """Open the immutable installed code and its staged setup interpreter."""

    declaration = manifest.setup
    if declaration is None:
        raise RuntimeError(f"插件 {manifest.name} 缺少 setup declaration")
    runtime = next(
        (
            item
            for item in manifest.python
            if item.runtime_root == declaration.python_runtime
        ),
        None,
    )
    if runtime is None:
        raise RuntimeError(
            f"插件 {manifest.name} setup.python_runtime 未找到: "
            f"{declaration.python_runtime}"
        )
    environment_refs = read_environment_refs(plugin_root, manifest)
    environment_ref = environment_refs.get(runtime.runtime_root)
    if environment_ref is None:
        raise RuntimeError(
            f"插件 {manifest.name} 缺少 setup Python environment reference"
        )

    environments = PythonEnvironments(workspace)
    record = environments.archive.read_descriptor(environment_ref)
    raw_input = record.get("input")
    if not isinstance(raw_input, Mapping):
        raise RuntimeError(f"插件 {manifest.name} Python environment input 无效")
    input_data = cast(Mapping[str, object], raw_input)
    code_ref = input_data.get("code")
    if not isinstance(code_ref, str):
        raise RuntimeError(f"插件 {manifest.name} Python environment code ref 无效")
    code_root = environments.archive.open(code_ref)
    archived_setup = code_root / declaration.entrypoint
    if (
        not archived_setup.is_file()
        or archived_setup.read_bytes() != setup_path.read_bytes()
    ):
        raise RuntimeError(f"插件 {manifest.name} setup.entrypoint 与已安装归档不一致")
    environment_root = environments.open(environment_ref, code_root, runtime)
    return staged_python_interpreter(environment_root, runtime), code_root


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


def _render_config(workspace: Path) -> str:
    """Render only Core settings; client settings belong to installed plugins."""

    return "\n".join(
        [
            "[runtime]",
            f"workspace = {workspace.as_posix()!r}",
            "",
            "# 本地程序化控制面；listen 留空时按 workspace 派生 Unix socket。",
            "[app_server]",
            "enabled = true",
            "listen = \"\"",
            "max_connections = 32",
            "ingress_queue_size = 128",
            "outbound_queue_size = 512",
            "",
            "# 外部 channel 与客户端插件的配置由各自的 setup 声明写入",
            "# workspace/plugin-data；Core 不读取业务配置表。",
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
    _hint("启动后使用已安装插件提供的控制面完成配置")
    _hint(f"插件私有配置位于 {workspace / 'plugin-data'}")
