"""QQ channel's interactive first-run configuration."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import click

_CONFIG_ENV = "AKASHIC_SETUP_CONFIG_PATH"


def main() -> None:
    """Collect the current QQ channel schema and publish its config file."""

    config_path = _config_path()
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

    content = "\n".join(
        [
            f"enabled = {str(enabled).lower()}",
            f"bot_uin = {json.dumps(bot_uin, ensure_ascii=False)}",
            f"allow_from = {json.dumps(list(allow_from), ensure_ascii=False)}",
            f"websocket_open_timeout_seconds = {timeout_seconds!r}",
            "",
        ]
    )
    _atomic_write(path, content)
    click.echo(click.style(f"  ✓ {path} 已生成", fg="green"))


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        shutil.copy2(path, path.with_name(f"{path.name}.before-setup.bak"))
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temp_name, 0o600)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


if __name__ == "__main__":
    main()
