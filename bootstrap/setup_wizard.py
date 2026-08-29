"""
交互式初始化向导

python main.py setup
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import select
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import click
from agent.plugins.manifest import (
    ensure_workspace_plugin_data_dir,
    workspace_plugin_data_dir,
)

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class WizardAnswers:
    tg_token: str = ""
    tg_allow_from: list[str] = field(default_factory=list)
    qqbot_app_id: str = ""
    qqbot_client_secret: str = ""
    qqbot_user_openid: str = ""


def _hint(text: str) -> None:
    click.echo(click.style(f"  {text}", dim=True))


def _ok(text: str) -> None:
    click.echo(click.style(f"  ✓ {text}", fg="green"))


def _warn(text: str) -> None:
    click.echo(click.style(f"  ! {text}", fg="yellow"))


def _err(text: str) -> None:
    click.echo(click.style(f"  ✗ {text}", fg="red"))


def _section_header(step: str, title: str) -> None:
    click.echo(f"\n{click.style(f'[{step}]', bold=True)} {title}\n")


def _divider() -> None:
    click.echo(click.style("─" * 40, dim=True))


def _read_escape_sequence(fd: int) -> str:
    ready, _, _ = select.select([fd], [], [], 0.01)
    if not ready:
        return ""

    first = sys.stdin.read(1)
    if first == "[":
        seq = [first]
        while len(seq) < 5:
            ready, _, _ = select.select([fd], [], [], 0.01)
            if not ready:
                break
            ch = sys.stdin.read(1)
            seq.append(ch)
            if ch == "~" or ch.isalpha():
                break
        return "".join(seq)

    if first == "O":
        ready, _, _ = select.select([fd], [], [], 0.01)
        if ready:
            return first + sys.stdin.read(1)
    return first


def _secret_prompt(
    text: str,
    *,
    default: str | None = None,
    show_default: bool = True,
) -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        if default is None:
            return _strip_paste_markers(click.prompt(text, hide_input=True))
        return _strip_paste_markers(
            click.prompt(
                text,
                default=default,
                hide_input=True,
                show_default=show_default,
            )
        )

    try:
        import termios
        import tty
    except Exception:
        if default is None:
            return _strip_paste_markers(click.prompt(text, hide_input=True))
        return _strip_paste_markers(
            click.prompt(
                text,
                default=default,
                hide_input=True,
                show_default=show_default,
            )
        )

    suffix = ""
    if show_default and default not in (None, ""):
        suffix = f" [{default}]"
    click.echo(f"{text}{suffix}: ", nl=False)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    chars: list[str] = []
    try:
        _ = tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "\x1b":
                seq = _read_escape_sequence(fd)
                if seq in ("[200~", "[201~"):
                    continue
                if seq.startswith("[") or seq.startswith("O"):
                    continue
                chars.extend(["\x1b", *seq])
                click.echo("*" * (len(seq) + 1), nl=False)
                continue
            if ch in ("\r", "\n"):
                click.echo()
                break
            if ch == "\x03":
                raise KeyboardInterrupt()
            if ch == "\x04":
                raise EOFError()
            if ch in ("\x7f", "\b"):
                if chars:
                    _ = chars.pop()
                    click.echo("\b \b", nl=False)
                continue
            if ch < " ":
                continue
            chars.append(ch)
            click.echo("*", nl=False)
    finally:
        _ = termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    value = "".join(chars)
    if value or default is None:
        return _strip_paste_markers(value)
    return default


def _strip_paste_markers(value: str) -> str:
    return value.replace("\x1b[200~", "").replace("\x1b[201~", "")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def run_setup_wizard(config_path: Path, workspace: Path) -> None:
    click.echo(click.style("\n══ akashic 初始化向导 ══\n", bold=True))
    _hint("全程按回车使用括号内的默认值")
    _hint("频道 token 输入时会显示为 *，正常输入后回车即可")

    if config_path.exists():
        click.echo(f"\n已存在配置文件 {config_path}")
        if not click.confirm("覆盖并重新配置？", default=False):
            click.echo("已取消。")
            return

    answers = _collect_answers()

    _divider()
    click.echo("\n正在生成配置并初始化工作区...")

    toml_str = _render_config(answers)
    _atomic_write_with_backup(config_path, toml_str, mode=0o600)
    _ok(f"{config_path} 已生成")
    qqbot_config_path = _qqbot_local_config_path(workspace)
    ensure_workspace_plugin_data_dir(qqbot_config_path.parent, workspace)
    _atomic_write_with_backup(
        qqbot_config_path, _render_qqbot_config(answers), mode=0o600
    )
    _ok(f"{qqbot_config_path} 已生成")

    _validate_config(config_path, workspace)

    from bootstrap.init_workspace import init_workspace

    _ = init_workspace(config_path=config_path, workspace=workspace)
    _ok(f"{workspace} 已初始化")

    _print_completion(answers, workspace)


# ---------------------------------------------------------------------------
# 各阶段问答
# ---------------------------------------------------------------------------


def _collect_answers() -> WizardAnswers:
    a = WizardAnswers()
    _phase_telegram(a)
    _phase_qqbot(a)
    return a


def _phase_telegram(a: WizardAnswers) -> None:
    _section_header("3/5", "Telegram 频道")

    if not click.confirm("配置 Telegram 频道？", default=True):
        _hint("跳过后仍可使用 Web 或程序化调用（python main.py exec）")
        return

    # BotFather 引导
    click.echo()
    click.echo(click.style("  还没有 Telegram bot？按以下步骤创建：", dim=True))
    _hint("1. 打开 Telegram，搜索 @BotFather")
    _hint("2. 发送 /newbot，按提示给 bot 起名")
    _hint("3. BotFather 会回复一串 token，格式：123456789:AAFxxx...")
    click.echo()

    while True:
        token = _secret_prompt("Bot token")
        err = _validate_tg_token(token)
        if err is None:
            a.tg_token = token
            break
        _err(f"{err}，请重新输入")

    click.echo()
    _hint("用户名在哪里看：Telegram → 设置 → 用户名（不带 @）")
    username = click.prompt("你的 Telegram 用户名")
    a.tg_allow_from = [username]


def _phase_qqbot(a: WizardAnswers) -> None:
    _section_header("4/5", "官方 QQBot 频道（可跳过）")
    _hint("使用腾讯开放平台 WebSocket 长连接，无需 NapCat，与 Telegram 并存")

    if not click.confirm("配置官方 QQBot？", default=False):
        return

    click.echo()
    click.echo(click.style("  还没有 QQ 开放平台应用？按以下步骤创建：", dim=True))
    _hint("1. 打开 https://q.qq.com，登录腾讯开放平台")
    _hint("2. 创建机器人应用，记录 AppID 和 AppSecret")
    _hint("3. 在「开发设置」中开启「私聊」C2C 消息权限")
    click.echo()

    a.qqbot_app_id = click.prompt("AppID")
    a.qqbot_client_secret = _secret_prompt("AppSecret (client_secret)")

    err = _validate_qqbot_credentials(a.qqbot_app_id, a.qqbot_client_secret)
    if err:
        _warn(f"凭据验证失败：{err}")
        _hint("继续配置，启动后检查凭据是否正确")

    click.echo()
    click.echo(click.style("  需要获取你的 user_openid：", bold=True))
    _hint("在 QQ 中搜索你的 bot，向它发任意一条消息（比如「你好」）")
    _hint("发完回来按回车，向导会自动读取")
    click.echo()
    click.pause(info="发完消息后按回车继续...")

    openid = _fetch_qqbot_openid_with_spinner(
        a.qqbot_app_id, a.qqbot_client_secret, timeout_s=90
    )
    if openid:
        _ok(f"user_openid 已获取：{openid}")
        a.qqbot_user_openid = openid
    else:
        _warn("未收到消息，allow_from 留空")
        _hint("启动后可在 QQBot 插件的 config.local.toml 中手动填入 allow_from")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _validate_tg_token(token: str) -> str | None:
    try:
        import httpx

        resp = httpx.get(f"https://api.telegram.org/bot{token}/getMe", timeout=8)
        data = resp.json()
        if data.get("ok"):
            bot_name = data["result"].get("username", "")
            _ok(f"bot 验证成功：@{bot_name}")
            return None
        if resp.status_code == 409:
            return "bot 已绑定 webhook，请先调用 deleteWebhook 删除"
        return f"token 无效（{data.get('description', resp.status_code)}）"
    except Exception as e:
        return f"网络错误：{e}"


def _validate_qqbot_credentials(app_id: str, client_secret: str) -> str | None:
    try:
        import httpx

        resp = httpx.post(
            "https://bots.qq.com/app/getAppAccessToken",
            json={"appId": app_id, "clientSecret": client_secret},
            timeout=10,
        )
        data = resp.json()
        if data.get("access_token"):
            _ok("AppID / AppSecret 验证成功")
            return None
        return f"token 获取失败（{data}）"
    except Exception as e:
        return f"网络错误：{e}"


def _fetch_qqbot_openid_with_spinner(
    app_id: str, client_secret: str, timeout_s: int = 90
) -> str | None:
    result: list[str | None] = [None]
    done = threading.Event()

    def _run() -> None:
        try:
            result[0] = asyncio.run(
                _async_fetch_qqbot_openid(app_id, client_secret, timeout_s, done)
            )
        except Exception as e:
            _err(f"获取 user_openid 失败：{e}")
        done.set()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not done.wait(timeout=0.1):
        frame = click.style(frames[i % len(frames)], fg="cyan")
        click.echo(f"\r  {frame} 等待消息中...", nl=False)
        i += 1
    click.echo("\r" + " " * 30 + "\r", nl=False)

    thread.join()
    return result[0]


async def _async_fetch_qqbot_openid(
    app_id: str,
    client_secret: str,
    timeout_s: int,
    stop: threading.Event,
) -> str | None:
    import httpx
    import websockets

    # 1. 获取 access token
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://bots.qq.com/app/getAppAccessToken",
            json={"appId": app_id, "clientSecret": client_secret},
        )
        token_data = resp.json()
        token = str(token_data.get("access_token") or "")
        if not token:
            return None

    # 2. 获取 gateway URL
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.sgroup.qq.com/gateway",
            headers={"Authorization": f"QQBot {token}"},
        )
        gateway_url = str(resp.json().get("url") or "")
        if not gateway_url:
            return None

    # 3. 连接 WS，监听第一条 C2C 私聊消息
    try:
        async with asyncio.timeout(timeout_s):
            async with websockets.connect(gateway_url) as ws:
                async for raw in ws:
                    if stop.is_set():
                        return None
                    payload = json.loads(raw)
                    op = payload.get("op")
                    if op == 10:
                        # Hello：发送鉴权 Identify
                        await ws.send(
                            json.dumps(
                                {
                                    "op": 2,
                                    "d": {
                                        "token": f"QQBot {token}",
                                        "intents": 1 << 25,
                                        "shard": [0, 1],
                                    },
                                }
                            )
                        )
                    elif op == 0 and payload.get("t") == "C2C_MESSAGE_CREATE":
                        raw_d = payload.get("d")
                        d = (
                            cast(dict[str, object], raw_d)
                            if isinstance(raw_d, dict)
                            else {}
                        )
                        raw_author = d.get("author")
                        author = (
                            cast(dict[str, object], raw_author)
                            if isinstance(raw_author, dict)
                            else {}
                        )
                        openid = str(
                            author.get("user_openid") or d.get("user_openid") or ""
                        )
                        if openid:
                            return openid
    except TimeoutError:
        return None
    return None


# ---------------------------------------------------------------------------
# Config 验证
# ---------------------------------------------------------------------------


def _validate_config(config_path: Path, workspace: Path) -> None:
    try:
        from agent.config import Config

        _ = Config.load(config_path, workspace=workspace)
        _ok("配置验证通过")
    except KeyError as e:
        _err(f"配置缺少必填字段：{e}")
        raise SystemExit(1)
    except Exception as e:
        _err(f"配置加载失败：{e}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# TOML 渲染
# ---------------------------------------------------------------------------


def _render_config(a: WizardAnswers) -> str:
    return "\n".join(
        [
            _render_agent(a),
            _render_channels(a),
        ]
    )


def _render_agent(a: WizardAnswers) -> str:
    return f"""\
[agent]
system_prompt = "You are Akashic, a helpful AI assistant with access to tools. Always respond in the same language the user uses."
# 设为 0 表示不限制迭代轮数；长任务仍可用 /stop 中断。
max_iterations = 40
dev_mode = false

[agent.context]
[agent.context.compaction]
keep_recent_tokens = 20000

[agent.tools]
search_enabled = true
"""


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


def _render_channels(a: WizardAnswers) -> str:
    lines: list[str] = []

    lines += [
        "# Web Chat 由 Supervisor 在唯一入口 2236 提供。",
        "[channels.chat]",
        "enabled = true",
        "",
    ]

    if a.tg_token:
        allow = ", ".join(f'"{u}"' for u in a.tg_allow_from)
        lines += [
            "[channels.telegram]",
            f'token = "{a.tg_token}"',
            f"allow_from = [{allow}]",
            "",
        ]
    else:
        lines += [
            "# [channels.telegram]",
            '# token = ""',
            '# allow_from = ["your_username"]',
            "",
        ]

    lines += [
        "# QQ 频道（NapCat，如需启用，填写后取消注释）",
        "# [channels.qq]",
        '# bot_uin = ""',
        '# allow_from = ["your_qq_number"]',
        "",
        "# [[channels.qq.groups]]",
        '# group_id = ""',
        '# allow_from = ["your_qq_number"]',
        "# require_at = true",
        "",
    ]

    return "\n".join(lines)


def _render_qqbot_config(a: WizardAnswers) -> str:
    allow = ", ".join(
        f'"{user}"' for user in ([a.qqbot_user_openid] if a.qqbot_user_openid else [])
    )
    return "\n".join(
        [
            f'app_id = "{a.qqbot_app_id}"',
            f'client_secret = "{a.qqbot_client_secret}"',
            f"allow_from = [{allow}]",
            "",
        ]
    )


def _qqbot_local_config_path(workspace: Path) -> Path:
    return workspace_plugin_data_dir(workspace, "qqbot", "github") / "config.local.toml"


def _print_completion(a: WizardAnswers, workspace: Path) -> None:
    click.echo(click.style("\n══ 配置完成 ══\n", bold=True))
    click.echo("启动 agent：")
    click.echo(click.style("  uv run python main.py", bold=True))
    _hint("启动后打开 2236 的“模型”页添加连接并选择默认模型")

    if a.qqbot_app_id and not a.qqbot_user_openid:
        click.echo()
        _warn("QQBot allow_from 为空，启动后所有私聊请求会被拒绝")
        _hint("向 bot 发一条消息，日志里找到 user_openid，填入 config.toml：")
        _hint(str(_qqbot_local_config_path(workspace)))
        _hint('allow_from = ["<user_openid>"]')
