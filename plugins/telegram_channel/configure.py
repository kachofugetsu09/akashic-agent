"""Telegram channel's interactive first-run configuration."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import click
import httpx

_CONFIG_ENV = "AKASHIC_SETUP_CONFIG_PATH"


def main() -> None:
    """Collect Telegram credentials and publish this plugin's config file."""

    config_path = _config_path()
    click.echo(click.style("\n[Telegram 频道]\n", bold=True))
    if not click.confirm("配置 Telegram 频道？", default=True):
        _write_config(config_path, enabled=False, token="", allow_from=())
        return

    click.echo()
    click.echo(click.style("  还没有 Telegram bot？按以下步骤创建：", dim=True))
    _hint("1. 打开 Telegram，搜索 @BotFather")
    _hint("2. 发送 /newbot，按提示给 bot 起名")
    _hint("3. BotFather 会回复一串 token，格式：123456789:AAFxxx...")
    click.echo()

    while True:
        token = _secret_prompt("Bot token")
        error = _validate_token(token)
        if error is None:
            break
        click.echo(click.style(f"  ✗ {error}，请重新输入", fg="red"))

    click.echo()
    _hint("用户名在哪里看：Telegram → 设置 → 用户名（不带 @）")
    username = click.prompt("你的 Telegram 用户名").strip()
    _write_config(config_path, enabled=True, token=token, allow_from=(username,))


def _config_path() -> Path:
    value = os.environ.get(_CONFIG_ENV, "").strip()
    if not value:
        raise RuntimeError(f"缺少 {_CONFIG_ENV}")
    return Path(value)


def _hint(text: str) -> None:
    click.echo(click.style(f"  {text}", dim=True))


def _secret_prompt(text: str) -> str:
    return (
        click.prompt(text, hide_input=True)
        .replace("\x1b[200~", "")
        .replace("\x1b[201~", "")
    )


def _validate_token(token: str) -> str | None:
    try:
        response = httpx.get(
            f"https://api.telegram.org/bot{token}/getMe",
            timeout=8,
        )
        data: Any = response.json()
        if isinstance(data, dict) and data.get("ok"):
            bot_name = data.get("result", {}).get("username", "")
            click.echo(click.style(f"  ✓ bot 验证成功：@{bot_name}", fg="green"))
            return None
        if response.status_code == 409:
            return "bot 已绑定 webhook，请先调用 deleteWebhook 删除"
        description = (
            data.get("description", response.status_code)
            if isinstance(data, dict)
            else response.status_code
        )
        return f"token 无效（{description}）"
    except Exception as error:
        return f"网络错误：{error}"


def _write_config(
    path: Path,
    *,
    enabled: bool,
    token: str,
    allow_from: tuple[str, ...],
) -> None:
    """Write one complete plugin-owned config with a recoverable backup."""

    lines = [f"enabled = {str(enabled).lower()}"]
    if token:
        lines.append(f"token = {json.dumps(token, ensure_ascii=False)}")
    lines.append(f"allow_from = {json.dumps(list(allow_from), ensure_ascii=False)}")
    _atomic_write(path, "\n".join(lines) + "\n")
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
