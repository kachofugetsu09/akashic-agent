"""启动独占的真实 Runtime，供首页弱网基准读取固定历史。"""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import time
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]


def message_digest(workspace: Path) -> str:
    """记录完整消息行，验证浏览器基准没有改变权威历史。"""
    with sqlite3.connect(workspace / "sessions.db") as db:
        rows = db.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall()
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


async def run(state: Path, port: int, switch_messages: int = 0, sessions: int = 0, chat_ready_delay_ms: int = 0,
              message_interactions: bool = False) -> None:
    """创建固定数据后启动正式应用和 Web Shell，退出时核对历史。"""
    # 1. 只接受新目录；所有配置、插件状态和消息都属于本次测试。
    state.mkdir(parents=True, exist_ok=False)
    plugins = state / "plugins"
    shutil.copytree(ROOT / "plugins", plugins, ignore=shutil.ignore_patterns("__pycache__", "static"))
    for plugin, name, entry in (("ui", "dashboard", "index.html"), ("akashic_clients", "chat", "chat-app.js")):
        assets = ROOT / "static" / name
        if not (assets / entry).is_file():
            raise RuntimeError("请先运行 npm run build")
        destination = plugins / plugin / "static"
        destination.mkdir()
        shutil.copytree(assets, destination / name)
        # 静态响应使用 Last-Modified；固定年龄，避免刚构建的候选多出启发式缓存重验证。
        for asset in (destination / name).rglob("*"):
            if asset.is_file():
                os.utime(asset, (1767225600, 1767225600))
    for key in tuple(os.environ):
        if key.startswith("AKASHIC_"):
            del os.environ[key]
    os.environ.update(HOME=str(state / "home"), XDG_CONFIG_HOME=str(state / "home/.config"),
                      AKASHIC_PLUGIN_HOME=str(state / "plugin-home"), AKASHIC_CORE_ROOT=str(ROOT),
                      AKASHIC_EXTRA_PLUGIN_DIRS=str(plugins))
    from bootstrap.init_workspace import init_workspace
    from bootstrap.app import AppRuntime
    from bootstrap.web_shell import create_web_shell_app
    from agent.config_models import Config
    from agent.plugins.bundles import set_plugin_choice
    from plugins.ledger.contract import ContentPart, Input, Output, SessionAttributes
    from plugins.ledger.log import MessageLog
    from plugins.content.plugin import check_text
    import uvicorn

    workspace, config = state / "workspace", state / "config.toml"
    init_workspace(config_path=config, workspace=workspace)
    # 隔离标记：写入型基准脚本 compare-chat-startup.mjs 在 rsync --delete 之前据此证明
    # 目标属于本次一次性 fixture。资产根是 state/plugins 下的 chat 目录，与消息 workspace
    # 不同根，因此必须分别发布；pid/port 供比较脚本独立核对活跃服务归属。
    (state / ".akashic-oneoff-fixture.json").write_text(
        json.dumps({
            "kind": "akashic-webui-oneoff",
            "state": str(state.resolve()),
            "workspace": str(workspace.resolve()),
            "chat": str((plugins / "akashic_clients" / "static" / "chat").resolve()),
            "pid": os.getpid(),
            "port": port,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    # Computer 需要独立 Workload Controller；本基准只测 Web 启动与历史阅读。
    set_plugin_choice(workspace, "computer", enabled=False)
    # web-auth 的入口门禁与登录页不属于启动性能输入；卸载后恢复无门状态。
    set_plugin_choice(workspace, "web-auth", enabled=False)
    log = MessageLog(workspace / "sessions.db")
    try:
        log.ensure_session("akashic:startup-benchmark", SessionAttributes(learning="excluded"))
        for index in range(100):
            part = ContentPart("text", f"启动基准消息 {index:03d}：固定历史用于检查页面完整可读。")
            body = Input((part,)) if index % 2 == 0 else Output((part,), "complete")
            log.writer("akashic:startup-benchmark", author="user" if index % 2 == 0 else "assistant",
                       source="benchmark", body_types=(type(body),), content={"text": check_text}).append(
                           f"startup-{index:03d}", body)
        if switch_messages:
            # 会话切换基准的长历史：混入代码块、列表与内联格式，让 markdown 渲染
            # 计入测量；内容逐条确定，保证重复运行的渲染成本一致。
            log.ensure_session("akashic:switch-benchmark", SessionAttributes(learning="excluded"))
            markdown = (
                "切换基准消息 {index:04d}：含**加粗**、`inline_code()` 与[链接](https://example.com)。\n\n"
                "- 第一项 `x = {index}`\n- 第二项\n\n"
                "```python\ndef handle_{index:04d}(payload):\n"
                "    return {{'index': {index}, 'ok': True}}  # 切换基准\n```"
            )
            for index in range(switch_messages):
                part = ContentPart("text", markdown.format(index=index))
                body = Input((part,)) if index % 2 == 0 else Output((part,), "complete")
                log.writer("akashic:switch-benchmark", author="user" if index % 2 == 0 else "assistant",
                           source="benchmark", body_types=(type(body),), content={"text": check_text}).append(
                               f"switch-{index:04d}", body)
        if sessions:
            # 目录规模基准：大量单条会话，让会话列表的分页拉取与侧栏渲染计入测量。
            for index in range(sessions):
                key = f"akashic:catalog-{index:05d}"
                log.ensure_session(key, SessionAttributes(learning="excluded"))
                part = ContentPart("text", f"目录基准会话 {index:05d}")
                log.writer(key, author="user", source="benchmark",
                           body_types=(type(Input((part,))),), content={"text": check_text}).append(
                               f"catalog-{index:05d}", Input((part,)))
        if message_interactions:
            # 独立原文在启动前固定；浏览器必须核对复制和回填全文，不能只看成功图标。
            long_input = "消息交互基准：全文不能丢失。\n" + "\n".join(
                f"第 {index:03d} 行：这是一段完整的长消息，用于检查展开收起与编辑回填。"
                for index in range(80)
            )
            long_output = "## 消息操作验收\n\n" + "\n\n".join(
                f"第 {index:03d} 段：保留 **格式** 与 `inline_{index}`，用于引用和复制核对。"
                for index in range(30)
            )
            for message_id, author, body in (
                ("interaction-long-input", "user", Input((ContentPart("text", long_input),))),
                ("interaction-long-output", "assistant", Output((ContentPart("text", long_output),), "complete")),
            ):
                log.writer("akashic:switch-benchmark", author=author, source="benchmark",
                           body_types=(type(body),), content={"text": check_text}).append(message_id, body)
            (state / "message-oracle.json").write_text(json.dumps({
                "interaction-long-input": long_input, "interaction-long-output": long_output,
            }, ensure_ascii=False))
    finally:
        log.close()
    before = message_digest(workspace)

    # 2. 不接模型或外部账号；真实 HTTP、插件目录、历史 API 和 WebSocket 仍完整运行。
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    loop = asyncio.get_running_loop()
    try:
        await app.start()
        app.plugin_watcher.stop()
        await app.plugin_watcher_task
        # 自动扫描停止后仍允许显式 SIGHUP，供独立的真实换代验收使用。
        app.plugin_watcher = None
        print(json.dumps({"origin": f"http://127.0.0.1:{port}", "messagesSha256": before}), flush=True)
        asgi = create_web_shell_app(workspace)
        if chat_ready_delay_ms:
            asgi = _ChatReadyDelay(asgi, chat_ready_delay_ms)
        server = uvicorn.Server(uvicorn.Config(asgi, host="127.0.0.1",
                                              port=port, log_level="warning"))
        # Uvicorn 退出后会重放信号；由本次服务接收，避免取消后续 Runtime 清理。
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, setattr, server, "should_exit", True)
        await server.serve()
    finally:
        try:
            await app.shutdown()
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.remove_signal_handler(sig)
            after = message_digest(workspace)
            (state / "integrity.json").write_text(json.dumps({"before": before, "after": after}))
            if before != after:
                raise RuntimeError("启动基准改变了消息历史")


class _ChatReadyDelay:
    """ready_at 之前把 /api/shell/state 的 chatReady 改写成 false：模拟真实冷启动里
    HTTP 已开始服务、聊天目录尚未就绪的窗口，让就绪轮询的调度计入测量。"""

    def __init__(self, app: object, delay_ms: int) -> None:
        self._app = app
        self._ready_at = time.monotonic() + delay_ms / 1000

    async def __call__(self, scope: dict, receive: object, send: object) -> None:
        if scope["type"] != "http" or scope.get("path") != "/api/shell/state" or time.monotonic() >= self._ready_at:
            await self._app(scope, receive, send)
            return
        messages = []

        async def capture(message: dict) -> None:
            messages.append(message)

        await self._app(scope, receive, capture)
        start = next(m for m in messages if m["type"] == "http.response.start")
        body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
        try:
            payload = json.loads(body)
            payload["chatReady"] = False
            payload["status"] = "unavailable"
            body = json.dumps(payload).encode()
        except (ValueError, UnicodeDecodeError):
            pass
        headers = [(k, v) for k, v in start["headers"] if k.lower() != b"content-length"]
        headers.append((b"content-length", str(len(body)).encode()))
        await send({"type": "http.response.start", "status": start["status"], "headers": headers})
        await send({"type": "http.response.body", "body": body})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--port", type=int, default=24137)
    parser.add_argument("--switch-session-messages", type=int, default=0,
                        help="额外灌入 akashic:switch-benchmark 长会话的消息数(默认不灌)")
    parser.add_argument("--sessions", type=int, default=0,
                        help="额外灌入的目录会话数(每会话 1 条消息,默认不灌)")
    parser.add_argument("--message-interactions", action="store_true",
                        help="在切换会话末尾追加固定长消息，用于折叠、引用、编辑和复制基准")
    parser.add_argument("--chat-ready-delay-ms", type=int, default=0,
                        help="模拟冷启动就绪窗口:该毫秒数内 /api/shell/state 回报 chatReady=false(默认 0)")
    args = parser.parse_args()
    if args.message_interactions and args.switch_session_messages <= 0:
        parser.error("--message-interactions 需要正数 --switch-session-messages")
    args_ready_delay = args.chat_ready_delay_ms
    asyncio.run(run(args.state.resolve(), args.port, args.switch_session_messages, args.sessions,
                    args_ready_delay, args.message_interactions))
