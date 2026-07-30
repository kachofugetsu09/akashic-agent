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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import click
from agent.plugins.manifest import (
    builtin_plugin_data_dir,
    ensure_workspace_plugin_data_dir,
    workspace_plugin_data_dir,
)
from plugins.default_memory.config import render_default_memory_config


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class WizardAnswers:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    auth_id: str = ""
    reasoning_effort: str = ""
    context_window: int = 0
    effective_context_percent: float = 0.9
    max_output_tokens: int = 0
    memory_window: int = 40
    enable_thinking: bool = False
    multimodal: bool = False
    use_responses_lite: bool = False
    supports_parallel_tool_calls: bool = True
    reasoning_summary: str = "none"
    vl_model: str = ""
    vl_api_key: str = ""
    vl_base_url: str = ""
    vl_auth_id: str = ""
    vl_provider: str = "openai"
    vl_context_window: int = 0
    vl_max_output_tokens: int = 0
    fast_model: str = ""
    fast_api_key: str = ""
    fast_base_url: str = ""
    fast_auth_id: str = ""
    fast_provider: str = "openai"
    fast_context_window: int = 0
    fast_max_output_tokens: int = 0
    tg_token: str = ""
    tg_allow_from: list[str] = field(default_factory=list)
    proactive_enabled: bool = False
    proactive_chat_id: str = ""
    proactive_channel: str = ""
    qqbot_app_id: str = ""
    qqbot_client_secret: str = ""
    qqbot_user_openid: str = ""
    embed_model: str = ""
    embed_api_key: str = ""
    embed_base_url: str = ""
    embed_auth_id: str = ""


# ---------------------------------------------------------------------------
# 输出工具
# ---------------------------------------------------------------------------

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
    _hint("API key / token 输入时会显示为 *，正常输入后回车即可")

    if config_path.exists():
        click.echo(f"\n已存在配置文件 {config_path}")
        if not click.confirm("覆盖并重新配置？", default=False):
            click.echo("已取消。")
            return

    answers = _collect_answers()

    _divider()
    click.echo("\n正在生成配置并初始化工作区...")

    _persist_answer_credentials(answers)
    toml_str = _render_config(answers)
    _atomic_write_with_backup(config_path, toml_str, mode=0o600)
    _ok(f"{config_path} 已生成")
    memory_config_path = _default_memory_local_config_path(workspace)
    ensure_workspace_plugin_data_dir(memory_config_path.parent, workspace)
    _atomic_write_with_backup(
        memory_config_path,
        render_default_memory_config(),
    )
    _ok(f"{memory_config_path} 已生成")
    qqbot_config_path = _qqbot_local_config_path(workspace)
    ensure_workspace_plugin_data_dir(qqbot_config_path.parent, workspace)
    _atomic_write_with_backup(qqbot_config_path, _render_qqbot_config(answers), mode=0o600)
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
    _phase_main_llm(a)
    _phase_fast_model(a)
    _phase_telegram(a)
    _phase_qqbot(a)
    _phase_memory(a)
    return a


def _phase_main_llm(
    a: WizardAnswers,
    *,
    configure_vl: bool = True,
    prompt_memory_window: bool = True,
    reuse_codex_auth: bool = False,
) -> None:
    _section_header("1/4", "主模型")

    auth_mode = click.prompt(
        "认证方式",
        type=click.Choice(["codex", "api-key"], case_sensitive=False),
        default="codex",
    )
    if auth_mode == "codex":
        _phase_codex_llm(a, reuse_existing_auth=reuse_codex_auth)
    else:
        _phase_api_key_llm(a)

    from agent.model_runtime.context_policy import recommended_context_settings

    suggested = recommended_context_settings(
        a.context_window,
        a.effective_context_percent,
    )
    a.memory_window = (
        click.prompt("历史消息窗口", type=int, default=suggested.memory_window)
        if prompt_memory_window
        else suggested.memory_window
    )
    if a.memory_window <= 0:
        raise click.BadParameter("历史消息窗口必须大于 0")

    if configure_vl and not a.multimodal:
        _phase_vl_model(a)


def _phase_api_key_llm(a: WizardAnswers) -> None:
    """收集 OpenAI-compatible API Key 模型配置。"""

    a.provider = click.prompt(
        "服务商",
        type=click.Choice(
            ["deepseek", "qwen", "openai", "opencode-go"],
            case_sensitive=False,
        ),
        default="deepseek",
    ).lower()
    if a.provider == "opencode-go":
        from agent.model_runtime.provider_profiles import OPENCODE_GO_BASE_URL

        a.base_url = click.prompt(
            "base_url（OpenAI 兼容格式）",
            default=OPENCODE_GO_BASE_URL,
        )
        a.api_key = _secret_prompt("API key")
        a.model = _choose_api_key_model(a.provider, a.base_url, a.api_key)
    else:
        a.model = click.prompt("模型名")
        a.base_url = click.prompt("base_url（OpenAI 兼容格式）")
        a.api_key = _secret_prompt("API key")
    a.auth_id = "main_default"
    a.enable_thinking = click.confirm("开启 thinking 模式？", default=False)
    a.reasoning_effort = (
        click.prompt("推理强度", default="medium") if a.enable_thinking else ""
    )
    a.context_window = click.prompt("上下文大小（tokens）", type=int, default=64000)
    if a.context_window <= 0:
        raise click.BadParameter("上下文大小必须大于 0")
    a.max_output_tokens = click.prompt(
        "最大输出 tokens（0 由 Provider 决定）",
        type=click.IntRange(min=0),
        default=0,
    )
    a.multimodal = (
        False
        if a.provider == "opencode-go"
        else click.confirm("主模型原生支持图片输入？", default=False)
    )


def _choose_api_key_model(provider: str, base_url: str, api_key: str) -> str:
    """为内建目录 provider 选择模型，其余 provider 保持手工输入。"""
    if provider != "opencode-go":
        return click.prompt("模型名")

    from agent.model_runtime.catalog.opencode_go import OpenCodeGoModelCatalog
    from agent.model_runtime.errors import AuthenticationError, TransportError

    try:
        models = asyncio.run(
            OpenCodeGoModelCatalog(api_key, base_url=base_url).list_models()
        )
    except (AuthenticationError, TransportError) as exc:
        raise click.ClickException(f"OpenCode Go 模型目录加载失败：{exc}") from exc
    if not models:
        raise click.ClickException("OpenCode Go 目录中没有可用的 Chat Completions 模型")
    slugs = [model.slug for model in models]
    return click.prompt("模型", type=click.Choice(slugs), default=slugs[0])


def _phase_codex_llm(
    a: WizardAnswers, *, reuse_existing_auth: bool = False
) -> None:
    """完成 Codex 登录并从目录选择模型能力。"""
    from agent.model_runtime.auth.codex import CodexAuthDriver
    from agent.model_runtime.auth.store import CredentialStore
    from agent.model_runtime.catalog.codex import CodexModelCatalog
    from agent.model_runtime.errors import AuthenticationError, TransportError
    import httpx

    a.provider = "codex"
    a.auth_id = "codex_default"
    a.base_url = "https://chatgpt.com/backend-api/codex"
    store = CredentialStore()
    auth = CodexAuthDriver(store, a.auth_id)
    login_required = True
    if reuse_existing_auth:
        try:
            credential = store.get(a.auth_id)
        except AuthenticationError:
            pass
        else:
            if credential.driver == "codex" and credential.access_token:
                login_required = not click.confirm(
                    "检测到已有 codex_default 登录，直接复用？",
                    default=True,
                )
    if login_required:
        code = auth.begin_device_login()
        click.echo(f"请打开 {code.verification_uri} 并输入代码：{code.user_code}")
        _ = auth.complete_device_login(code)
    try:
        models = asyncio.run(CodexModelCatalog(auth).list_models())
    except (AuthenticationError, TransportError, httpx.HTTPError) as exc:
        _err(f"Codex 模型目录加载失败：{exc}")
        if not click.confirm("显式进入手动模式？", default=False):
            raise click.ClickException("未取得模型目录，初始化已停止") from exc
        _phase_codex_manual(a)
        return
    if not models:
        raise click.ClickException("Codex 模型目录为空")
    choices = {model.slug: model for model in models}
    a.model = click.prompt("模型", type=click.Choice(list(choices)), default=models[0].slug)
    selected = choices[a.model]
    capabilities = selected.capabilities
    efforts = capabilities.supported_reasoning_efforts
    if efforts:
        default_effort = capabilities.default_reasoning_effort or efforts[0]
        a.reasoning_effort = click.prompt(
            "推理强度", type=click.Choice(list(efforts)), default=default_effort
        )
    max_context_window = capabilities.max_context_window or capabilities.context_window
    a.context_window = click.prompt(
        "上下文大小（tokens）",
        type=click.IntRange(min=1, max=max_context_window),
        default=capabilities.context_window,
    )
    a.effective_context_percent = capabilities.effective_context_percent
    a.max_output_tokens = 0
    detected_image = "image" in capabilities.input_modalities
    if not selected.input_modalities_known:
        _warn("模型目录未提供多模态元数据，请手工确认")
    a.multimodal = click.confirm("主模型支持图片输入？", default=detected_image)
    a.use_responses_lite = capabilities.use_responses_lite
    a.supports_parallel_tool_calls = capabilities.supports_parallel_tool_calls
    if capabilities.supports_reasoning_summaries:
        a.reasoning_summary = "auto"


def _phase_codex_manual(a: WizardAnswers) -> None:
    a.model = click.prompt("模型名")
    a.reasoning_effort = click.prompt("推理强度", default="medium")
    a.reasoning_summary = "auto"
    a.context_window = click.prompt("上下文大小（tokens）", type=int)
    a.max_output_tokens = click.prompt(
        "最大输出 tokens（0 由 Provider 决定）",
        type=click.IntRange(min=0),
        default=0,
    )
    a.multimodal = click.confirm("主模型支持图片输入？", default=False)


def _phase_vl_model(a: WizardAnswers) -> None:
    if not click.confirm("配置独立视觉模型？", default=False):
        return
    a.vl_model = click.prompt("视觉模型名")
    a.vl_provider, a.vl_base_url, a.vl_api_key = _phase_role_endpoint(
        a,
        allow_opencode_go=False,
    )
    a.vl_auth_id = "vl_default"
    a.vl_context_window = click.prompt(
        "视觉模型上下文大小（tokens）",
        type=click.IntRange(min=1),
        default=a.context_window,
    )
    a.vl_max_output_tokens = click.prompt(
        "视觉模型最大输出 tokens（0 由 Provider 决定）",
        type=click.IntRange(min=0),
        default=0,
    )


def _phase_fast_model(a: WizardAnswers) -> None:
    _section_header("2/4", "轻量模型（可跳过）")
    _hint("用于 memory gate / HyDE 等低延迟场景，跳过则退回主模型")

    if not click.confirm("配置独立轻量模型？", default=False):
        return

    if a.provider in {"codex", "opencode-go"}:
        a.fast_provider, a.fast_base_url, a.fast_api_key = _phase_role_endpoint(a)
        a.fast_model = _choose_api_key_model(
            a.fast_provider,
            a.fast_base_url,
            a.fast_api_key,
        )
    else:
        a.fast_model = click.prompt("模型名")
        a.fast_provider, a.fast_base_url, a.fast_api_key = _phase_role_endpoint(a)
    a.fast_auth_id = "fast_default"
    a.fast_context_window = click.prompt(
        "轻量模型上下文大小（tokens）",
        type=click.IntRange(min=1),
        default=min(a.context_window, 128_000),
    )
    a.fast_max_output_tokens = click.prompt(
        "轻量模型最大输出 tokens（0 由 Provider 决定）",
        type=click.IntRange(min=0),
        default=0,
    )


def _phase_role_endpoint(
    a: WizardAnswers,
    *,
    allow_opencode_go: bool = True,
) -> tuple[str, str, str]:
    """收集独立角色的兼容端点；API-key 主模型默认复用连接。"""
    if a.provider == "codex" or (
        a.provider == "opencode-go" and not allow_opencode_go
    ):
        providers = ["deepseek", "qwen", "openai"]
        if allow_opencode_go:
            providers.append("opencode-go")
        provider = click.prompt(
            "服务商",
            type=click.Choice(providers, case_sensitive=False),
            default="openai",
        ).lower()
        if provider == "opencode-go":
            from agent.model_runtime.provider_profiles import OPENCODE_GO_BASE_URL

            base_url = click.prompt(
                "OpenAI-compatible base_url",
                default=OPENCODE_GO_BASE_URL,
            )
        else:
            base_url = click.prompt("OpenAI-compatible base_url")
        return provider, base_url, _secret_prompt("API key")
    base_url = click.prompt(
        "base_url（回车 = 复用主模型 base_url）",
        default="",
        show_default=False,
    ) or a.base_url
    api_key = _secret_prompt(
        "API key（回车 = 复用主模型 key）",
        default="",
        show_default=False,
    ) or a.api_key
    return a.provider, base_url, api_key


def _phase_telegram(a: WizardAnswers) -> None:
    _section_header("3/5", "Telegram 频道 + Proactive")

    if not click.confirm("配置 Telegram 频道？", default=True):
        _hint("跳过后仅支持程序化调用（python main.py exec），proactive 已关闭")
        a.proactive_enabled = False
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

    click.echo()
    _hint("开启后 agent 会主动向你推送订阅内容和提醒")
    if not click.confirm("开启 proactive 主动推送？", default=True):
        a.proactive_enabled = False
        return

    a.proactive_enabled = True
    a.proactive_channel = "telegram"

    # 获取 chat_id
    click.echo()
    click.echo(click.style("  需要获取你的 Telegram chat_id：", bold=True))
    _hint("现在打开 Telegram，向你的 bot 发任意一条消息（比如「你好」）")
    _hint("发完回来按回车，向导会自动读取")
    click.echo()
    click.pause(info="发完消息后按回车继续...")

    chat_id = _fetch_chat_id_with_spinner(a.tg_token, username, timeout_s=60)
    if chat_id:
        _ok(f"chat_id 已获取：{chat_id}")
        a.proactive_chat_id = chat_id
    else:
        _warn("未收到消息，chat_id 留空")
        _hint("启动后向 bot 发 /chatid 可以随时补填")


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
        # 仅在没有 Telegram proactive 时才用 qqbot 作为 proactive 目标
        if not a.proactive_enabled and click.confirm("开启 proactive 主动推送（via QQBot）？", default=True):
            a.proactive_enabled = True
            a.proactive_channel = "qqbot"
            a.proactive_chat_id = f"c2c:{openid}"
    else:
        _warn("未收到消息，allow_from 留空")
        _hint("启动后可在 QQBot 插件的 config.local.toml 中手动填入 allow_from")


def _phase_memory(a: WizardAnswers) -> None:
    _section_header("5/5", "语义记忆（Embedding）")
    _hint("agent 用 embedding 模型将记忆转为向量，实现语义检索")
    click.echo()

    a.embed_model = click.prompt("Embedding 模型名")
    a.embed_api_key = _secret_prompt("Embedding API key")
    a.embed_auth_id = "embedding_default"
    a.embed_base_url = click.prompt("Embedding base_url")


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


def _fetch_chat_id_with_spinner(token: str, username: str, timeout_s: int = 60) -> str | None:
    result: list[str | None] = [None]
    done = threading.Event()

    def _poll() -> None:
        result[0] = _fetch_chat_id(token, username, timeout_s, done)
        done.set()

    thread = threading.Thread(target=_poll, daemon=True)
    thread.start()

    # 主线程显示等待动画
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while not done.wait(timeout=0.1):
        frame = click.style(frames[i % len(frames)], fg="cyan")
        click.echo(f"\r  {frame} 等待消息中...", nl=False)
        i += 1
    click.echo("\r" + " " * 30 + "\r", nl=False)  # 清除等待行

    thread.join()
    return result[0]


def _fetch_chat_id(token: str, username: str, timeout_s: int, stop: threading.Event | None = None) -> str | None:
    try:
        import httpx
        url = f"https://api.telegram.org/bot{token}/getUpdates"

        # 1. 清掉历史 update
        with httpx.Client(timeout=10) as client:
            resp = client.get(url, params={"offset": -1, "limit": 1})
            last = resp.json().get("result", [])
            offset = (last[-1]["update_id"] + 1) if last else 0

        # 2. 轮询
        deadline = time.time() + timeout_s
        with httpx.Client(timeout=12) as client:
            while time.time() < deadline:
                if stop and stop.is_set():
                    break
                resp = client.get(url, params={"offset": offset, "timeout": 10})
                for update in resp.json().get("result", []):
                    offset = update["update_id"] + 1
                    msg = update.get("message") or update.get("channel_post")
                    if not msg:
                        continue
                    from_user = msg.get("from", {})
                    if from_user.get("username", "").lower() == username.lower():
                        chat_id = str(msg["chat"]["id"])
                        try:
                            _ = client.get(
                                url,
                                params={"offset": offset, "limit": 1, "timeout": 0},
                            )
                        except Exception as e:
                            _warn(f"chat_id 已获取，但确认 Telegram update 失败：{e}")
                        return chat_id
    except Exception as e:
        _err(f"获取 chat_id 失败：{e}")
    return None


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


def _fetch_qqbot_openid_with_spinner(app_id: str, client_secret: str, timeout_s: int = 90) -> str | None:
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
                        await ws.send(json.dumps({
                            "op": 2,
                            "d": {
                                "token": f"QQBot {token}",
                                "intents": 1 << 25,
                                "shard": [0, 1],
                            },
                        }))
                    elif op == 0 and payload.get("t") == "C2C_MESSAGE_CREATE":
                        raw_d = payload.get("d")
                        d = cast(dict[str, object], raw_d) if isinstance(raw_d, dict) else {}
                        raw_author = d.get("author")
                        author = (
                            cast(dict[str, object], raw_author)
                            if isinstance(raw_author, dict)
                            else {}
                        )
                        openid = str(author.get("user_openid") or d.get("user_openid") or "")
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
    return "\n".join([
        _render_llm(a),
        _render_agent(a),
        _render_channels(a),
        _render_memory(a),
        _render_proactive(a),
        _render_integrations(),
    ])


def _render_llm(a: WizardAnswers) -> str:
    """把向导答案渲染为角色引用与 named runtimes。"""

    # 1. 角色只引用 runtime ID，跳过独立模型时复用 main。
    lines = [
        "[llm]",
        'main = "main"',
        f'fast = "{"fast" if a.fast_model else "main"}"',
        'agent = "main"',
    ]
    if a.multimodal:
        lines.append('vl = "main"')
    elif a.vl_model:
        lines.append('vl = "vl"')
    lines.append("")

    # 2. 主 runtime 完整声明认证、能力与上下文边界。
    main_modalities = '["text", "image"]' if a.multimodal else '["text"]'
    lines.extend([
        "[llm.runtimes.main]",
        f'provider = "{a.provider}"',
        f'auth = "{a.auth_id}"',
        f'model = "{a.model}"',
        f'base_url = "{a.base_url}"',
    ])
    if a.enable_thinking:
        lines.append("enable_thinking = true")
    if a.reasoning_effort:
        lines.append(f'reasoning_effort = "{a.reasoning_effort}"')
    if a.effective_context_percent != 0.9:
        lines.append(f"effective_context_percent = {a.effective_context_percent}")
    if a.use_responses_lite:
        lines.append("use_responses_lite = true")
    if not a.supports_parallel_tool_calls:
        lines.append("supports_parallel_tool_calls = false")
    if a.reasoning_summary != "none":
        lines.append(f'reasoning_summary = "{a.reasoning_summary}"')
    lines.extend([
        f"context_window = {a.context_window}",
        f"max_output_tokens = {a.max_output_tokens}",
        f"input_modalities = {main_modalities}",
        "",
    ])

    # 3. 独立角色使用完整 OpenAI-compatible runtime，不继承主端点。
    if a.fast_model:
        lines.extend([
            "[llm.runtimes.fast]",
            f'provider = "{a.fast_provider}"',
            f'auth = "{a.fast_auth_id}"',
            f'model = "{a.fast_model}"',
            f'base_url = "{a.fast_base_url}"',
            f"context_window = {a.fast_context_window or a.context_window}",
            f"max_output_tokens = {a.fast_max_output_tokens}",
            'input_modalities = ["text"]',
            "",
        ])

    if a.vl_model:
        lines.extend([
            "[llm.runtimes.vl]",
            f'provider = "{a.vl_provider}"',
            f'auth = "{a.vl_auth_id}"',
            f'model = "{a.vl_model}"',
            f'base_url = "{a.vl_base_url}"',
            f"context_window = {a.vl_context_window or a.context_window}",
            f"max_output_tokens = {a.vl_max_output_tokens}",
            'input_modalities = ["text", "image"]',
            "",
        ])

    return "\n".join(lines)


def _render_agent(a: WizardAnswers) -> str:
    return f"""\
[agent]
system_prompt = "You are Akashic, a helpful AI assistant with access to tools. Always respond in the same language the user uses."
max_tokens = {a.max_output_tokens}
# 设为 0 表示不限制迭代轮数；长任务仍可用 /stop 中断。
max_iterations = 40
dev_mode = false

[agent.context]
memory_window = {a.memory_window}

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


def _persist_answer_credentials(a: WizardAnswers) -> None:
    """问答全部完成后一次性持久化向导收集的 API key。"""
    from datetime import datetime, timezone
    from agent.model_runtime.auth.store import Credential, CredentialStore

    raw = {
        a.auth_id: a.api_key if a.provider != "codex" else "",
        a.fast_auth_id: a.fast_api_key,
        a.vl_auth_id: a.vl_api_key,
        a.embed_auth_id: a.embed_api_key,
    }
    credentials = {
        credential_id: Credential(
            driver="api_key",
            access_token=api_key,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        for credential_id, api_key in raw.items()
        if credential_id and api_key
    }
    if a.provider != "codex" and a.auth_id not in credentials:
        raise click.BadParameter("主模型 API key 不能为空")
    if a.embed_auth_id not in credentials:
        raise click.BadParameter("Embedding API key 不能为空")
    if a.fast_model and a.fast_auth_id not in credentials:
        raise click.BadParameter("独立轻量模型 API key 不能为空")
    if a.vl_model and a.vl_auth_id not in credentials:
        raise click.BadParameter("独立视觉模型 API key 不能为空")
    CredentialStore().put_many(credentials)


def _render_channels(a: WizardAnswers) -> str:
    lines: list[str] = []

    lines += [
        "# Web Chatbox 默认跟主进程一起启动，只监听本机。",
        "[channels.chat]",
        "enabled = true",
        'host = "127.0.0.1"',
        "port = 6322",
        'channel_name = "web"',
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
    return "\n".join([
        f'app_id = "{a.qqbot_app_id}"',
        f'client_secret = "{a.qqbot_client_secret}"',
        f"allow_from = [{allow}]",
        "",
    ])


def _qqbot_local_config_path(workspace: Path) -> Path:
    return workspace_plugin_data_dir(workspace, "qqbot", "github") / "config.local.toml"


def _render_memory(a: WizardAnswers) -> str:
    return "\n".join([
        "[memory]",
        "enabled = true",
        'engine = ""',
        "",
        "[memory.embedding]",
        f'model = "{a.embed_model}"',
        (
            f'auth = "{a.embed_auth_id}"'
            if a.embed_auth_id
            else f'api_key = "{a.embed_api_key}"'
        ),
        f'base_url = "{a.embed_base_url}"',
        "",
    ])


def _default_memory_local_config_path(workspace: Path) -> Path:
    return builtin_plugin_data_dir("default_memory", workspace) / "config.local.toml"


def _render_proactive(a: WizardAnswers) -> str:
    enabled = "true" if a.proactive_enabled else "false"
    channel = a.proactive_channel or ("telegram" if a.tg_token else "")
    return "\n".join([
        "[proactive]",
        f"enabled = {enabled}",
        'profile = "daily"',
        "",
        "[proactive.target]",
        f'channel = "{channel}"',
        f'chat_id = "{a.proactive_chat_id}"',
        "",
        "[proactive.agent]",
        "max_steps = 35",
        "content_limit = 5",
        "web_fetch_max_chars = 8000",
        "context_prob = 0.03",
        "delivery_cooldown_hours = 1",
        "",
        "[proactive.drift]",
        "enabled = false",
        "max_steps = 20",
        "min_interval_hours = 3",
        "",
    ])


def _render_integrations() -> str:
    return """\
# 可选：接入外部 Peer Agent（如 DeepResearch）
# [[integrations.peer_agents]]
# name = "DeepResearch Agent"
# base_url = "http://127.0.0.1:9404"
# launcher = ["uv", "run", "--project", "/path/to/deepresearch", "python", "-m", "app.a2a_server"]
# cwd = "/path/to/deepresearch"
# description = "对复杂问题执行多轮深度调研，生成结构化长报告。"
# startup_timeout_s = 30
# shutdown_timeout_s = 60
"""


# ---------------------------------------------------------------------------
# 完成提示
# ---------------------------------------------------------------------------

def _print_completion(a: WizardAnswers, workspace: Path) -> None:
    click.echo(click.style("\n══ 配置完成 ══\n", bold=True))
    click.echo("启动 agent：")
    click.echo(click.style("  uv run python main.py", bold=True))

    if a.proactive_enabled and not a.proactive_chat_id:
        click.echo()
        _warn("proactive 已开启，但 chat_id 未获取到")
        if a.proactive_channel == "qqbot" or (not a.tg_token and a.qqbot_app_id):
            _hint("启动后向 bot 发任意消息，日志中会出现 user_openid")
            _hint("将其填入 config.toml：")
            _hint(str(_qqbot_local_config_path(workspace)))
            _hint('allow_from = ["<user_openid>"]')
            _hint("[proactive.target]")
            _hint('channel = "qqbot"')
            _hint('chat_id = "c2c:<user_openid>"')
        else:
            _hint("启动后向 bot 发 /chatid，把返回的 id 填入 config.toml：")
            _hint("[proactive.target]")
            _hint('chat_id = "<你的 id>"')
        _hint("修改后重启生效")
    elif a.proactive_enabled and a.proactive_chat_id:
        click.echo()
        _ok("proactive 已配置，启动后会主动向你推送消息")

    if a.qqbot_app_id and not a.qqbot_user_openid:
        click.echo()
        _warn("QQBot allow_from 为空，启动后所有私聊请求会被拒绝")
        _hint("向 bot 发一条消息，日志里找到 user_openid，填入 config.toml：")
        _hint(str(_qqbot_local_config_path(workspace)))
        _hint('allow_from = ["<user_openid>"]')
