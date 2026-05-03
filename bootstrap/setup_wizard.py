"""
交互式初始化向导

python main.py setup
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class WizardAnswers:
    provider: str = ""
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    enable_thinking: bool = False
    multimodal: bool = False
    vl_model: str = ""
    vl_api_key: str = ""
    vl_base_url: str = ""
    fast_model: str = ""
    fast_api_key: str = ""
    fast_base_url: str = ""
    tg_token: str = ""
    tg_allow_from: list[str] = field(default_factory=list)
    proactive_enabled: bool = False
    proactive_chat_id: str = ""
    embed_model: str = ""
    embed_api_key: str = ""
    embed_base_url: str = ""


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


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def run_setup_wizard(config_path: Path, workspace: Path) -> None:
    click.echo(click.style("\n══ akashic 初始化向导 ══\n", bold=True))
    _hint("全程按回车使用括号内的默认值")
    _hint("API key 输入时不会显示字符，正常输入后回车即可")

    if config_path.exists():
        click.echo(f"\n已存在配置文件 {config_path}")
        if not click.confirm("覆盖并重新配置？", default=False):
            click.echo("已取消。")
            return

    answers = _collect_answers()

    _divider()
    click.echo("\n正在生成配置并初始化工作区...")

    toml_str = _render_config(answers)
    config_path.write_text(toml_str, encoding="utf-8")
    _ok(f"{config_path} 已生成")

    _validate_config(config_path)

    from bootstrap.init_workspace import init_workspace
    init_workspace(config_path=config_path, workspace=workspace)
    _ok(f"{workspace} 已初始化")

    _print_completion(answers)


# ---------------------------------------------------------------------------
# 各阶段问答
# ---------------------------------------------------------------------------

def _collect_answers() -> WizardAnswers:
    a = WizardAnswers()
    _phase_main_llm(a)
    _phase_fast_model(a)
    _phase_channels(a)
    _phase_memory(a)
    return a


def _phase_main_llm(a: WizardAnswers) -> None:
    _section_header("1/4", "主模型")

    a.model = click.prompt("模型名")
    a.base_url = click.prompt("base_url（OpenAI 兼容格式）")
    a.api_key = click.prompt("API key", hide_input=True)
    a.provider = "openai"
    a.enable_thinking = click.confirm("开启 thinking 模式？", default=False)
    a.multimodal = click.confirm("主模型原生支持图片输入？", default=False)

    if not a.multimodal:
        if click.confirm("配置独立视觉模型？", default=False):
            a.vl_model = click.prompt("视觉模型名")
            a.vl_base_url = click.prompt(
                "base_url（回车 = 复用主模型 base_url）",
                default="",
                show_default=False,
            ) or a.base_url
            a.vl_api_key = click.prompt(
                "API key（回车 = 复用主模型 key）",
                default="",
                show_default=False,
            ) or a.api_key


def _phase_fast_model(a: WizardAnswers) -> None:
    _section_header("2/4", "轻量模型（可跳过）")
    _hint("用于 memory gate / HyDE 等低延迟场景，跳过则退回主模型")

    if not click.confirm("配置独立轻量模型？", default=False):
        return

    a.fast_model = click.prompt("模型名")
    a.fast_base_url = click.prompt(
        "base_url（回车 = 复用主模型 base_url）",
        default="",
        show_default=False,
    ) or a.base_url
    a.fast_api_key = click.prompt(
        "API key（回车 = 复用主模型 key）",
        default="",
        show_default=False,
    ) or a.api_key


def _phase_channels(a: WizardAnswers) -> None:
    _section_header("3/4", "Telegram 频道 + Proactive")

    if not click.confirm("配置 Telegram 频道？", default=True):
        _hint("跳过后仅支持 CLI 模式（uv run python main.py cli），proactive 已关闭")
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
        token = click.prompt("Bot token")
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


def _phase_memory(a: WizardAnswers) -> None:
    _section_header("4/4", "语义记忆（Embedding）")
    _hint("agent 用 embedding 模型将记忆转为向量，实现语义检索")
    click.echo()

    a.embed_model = click.prompt("Embedding 模型名")
    a.embed_api_key = click.prompt("Embedding API key", hide_input=True)
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
                        return str(msg["chat"]["id"])
    except Exception as e:
        _err(f"获取 chat_id 失败：{e}")
    return None


# ---------------------------------------------------------------------------
# Config 验证
# ---------------------------------------------------------------------------

def _validate_config(config_path: Path) -> None:
    try:
        from agent.config import Config
        Config.load(config_path)
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
        _render_agent(),
        _render_channels(a),
        _render_memory(a),
        _render_proactive(a),
        _render_integrations(),
    ])


def _render_llm(a: WizardAnswers) -> str:
    lines: list[str] = [
        "[llm]",
        f'provider = "{a.provider}"',
        "",
        "[llm.main]",
        f'model = "{a.model}"',
        f'api_key = "{a.api_key}"',
        f'base_url = "{a.base_url}"',
    ]
    if a.enable_thinking:
        lines.append("enable_thinking = true")
    lines.append(f"multimodal = {'true' if a.multimodal else 'false'}")
    lines.append("")

    if a.fast_model:
        lines += [
            "[llm.fast]",
            f'model = "{a.fast_model}"',
            f'api_key = "{a.fast_api_key}"',
            f'base_url = "{a.fast_base_url}"',
            "",
        ]
    else:
        lines += [
            "# 轻量模型未配置，memory gate / HyDE 将使用主模型",
            "# [llm.fast]",
            "# model = \"\"",
            "",
        ]

    if a.vl_model:
        lines += [
            "[llm.vl]",
            f'model = "{a.vl_model}"',
            f'api_key = "{a.vl_api_key}"',
            f'base_url = "{a.vl_base_url}"',
            "",
        ]
    else:
        lines += [
            "# 视觉模型未配置",
            "# [llm.vl]",
            "# model = \"\"",
            "",
        ]

    return "\n".join(lines)


def _render_agent() -> str:
    return """\
[agent]
system_prompt = "You are Akashic, a helpful AI assistant with access to tools. Always respond in the same language the user uses."
max_tokens = 8192
max_iterations = 40
dev_mode = false

[agent.context]
memory_window = 24

[agent.tools]
search_enabled = true
"""


def _render_channels(a: WizardAnswers) -> str:
    lines: list[str] = []

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
        "# QQ 频道（如需启用，填写后取消注释）",
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


def _render_memory(a: WizardAnswers) -> str:
    return "\n".join([
        "[memory]",
        "enabled = true",
        "",
        "[memory.embedding]",
        f'model = "{a.embed_model}"',
        f'api_key = "{a.embed_api_key}"',
        f'base_url = "{a.embed_base_url}"',
        "",
        "[memory.retrieval]",
        "top_k_history = 8",
        "score_threshold = 0.45",
        "relative_delta = 0.2",
        "route_intention = true",
        "",
        "[memory.retrieval.thresholds]",
        "procedure = 0.66",
        "preference = 0.5",
        "event = 0.5",
        "profile = 0.5",
        "",
        "[memory.retrieval.inject]",
        "max_chars = 6000",
        "line_max = 600",
        "event_profile = 4",
        "",
        "[memory.gate]",
        "llm_timeout_ms = 1600",
        "max_tokens = 200",
        "",
        "[memory.hyde]",
        "enabled = true",
        "timeout_ms = 2000",
        "",
    ])


def _render_proactive(a: WizardAnswers) -> str:
    enabled = "true" if a.proactive_enabled else "false"
    channel = "telegram" if a.tg_token else ""
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
[integrations.fitbit]
enabled = false

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

def _print_completion(a: WizardAnswers) -> None:
    click.echo(click.style("\n══ 配置完成 ══\n", bold=True))
    click.echo("启动 agent：")
    click.echo(click.style("  uv run python main.py", bold=True))

    if a.proactive_enabled and not a.proactive_chat_id:
        click.echo()
        _warn("proactive 已开启，但 chat_id 未获取到")
        _hint("启动后向 bot 发 /chatid，把返回的 id 填入 config.toml：")
        _hint("[proactive.target]")
        _hint('chat_id = "<你的 id>"')
        _hint("修改后重启生效")
    elif a.proactive_enabled and a.proactive_chat_id:
        click.echo()
        _ok("proactive 已配置，启动后会主动向你推送消息")
