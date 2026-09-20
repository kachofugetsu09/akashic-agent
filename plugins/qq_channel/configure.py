"""QQ channel's interactive first-run configuration."""

from __future__ import annotations

import os
import sys
import tomllib

from agent.plugin_composition.config_input import save_config, upgrade_config
from pathlib import Path

import click

_CONFIG_ENV = "AKASHIC_SETUP_CONFIG_PATH"


def main() -> None:
    """Collect the current QQ channel schema and publish its config file."""

    config_path = _config_path()
    if sys.argv[1:] == ["--upgrade"]:
        backup = upgrade_config(config_path.parent, lambda content: _upgrade(content, config_path.parent))
        click.echo(f"配置已升级；恢复点：{backup}")
        return
    if sys.argv[1:]:
        raise ValueError("只接受 --upgrade 或无参数的交互配置")
    click.echo(click.style("\n[QQ 频道]\n", bold=True))
    _hint("当前 QQ channel 使用 NapCat/NcatBot；配置由该插件保存和校验。")
    if not click.confirm("配置 QQ 频道？", default=False):
        _write_config(
            config_path,
            enabled=False,
            bot_uin="",
            allow_from=(),
            timeout_seconds=5.0,
        )
        return

    bot_uin = click.prompt("Bot UIN").strip()
    while not bot_uin:
        click.echo(click.style("  ✗ Bot UIN 不能为空", fg="red"))
        bot_uin = click.prompt("Bot UIN").strip()
    raw_allow_from = click.prompt(
        "允许的 QQ 用户 UIN（逗号分隔，留空允许所有人）",
        default="",
        show_default=False,
    )
    allow_from = tuple(
        item.strip() for item in raw_allow_from.split(",") if item.strip()
    )
    timeout_seconds = click.prompt(
        "WebSocket 启动超时秒数",
        default=5.0,
        type=float,
    )
    _write_config(
        config_path,
        enabled=True,
        bot_uin=bot_uin,
        allow_from=allow_from,
        timeout_seconds=timeout_seconds,
    )


def _config_path() -> Path:
    value = os.environ.get(_CONFIG_ENV, "").strip()
    if not value:
        raise RuntimeError(f"缺少 {_CONFIG_ENV}")
    return Path(value)


def _hint(text: str) -> None:
    click.echo(click.style(f"  {text}", dim=True))


def _write_config(
    path: Path,
    *,
    enabled: bool,
    bot_uin: str,
    allow_from: tuple[str, ...],
    timeout_seconds: float,
) -> None:
    """Write one complete plugin-owned config with a recoverable backup."""

    save_config(path.parent, {
        "enabled": enabled, "bot_uin": bot_uin, "allow_from": list(allow_from),
        "websocket_open_timeout_seconds": timeout_seconds,
    })
    click.echo(click.style(f"  ✓ {path} 已生成", fg="green"))


def _upgrade(content: bytes, data_dir: Path) -> dict[str, object]:
    return tomllib.loads(content.decode("utf-8"))


if __name__ == "__main__":
    main()
