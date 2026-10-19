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
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]


def message_digest(workspace: Path) -> str:
    """记录完整消息行，验证浏览器基准没有改变权威历史。"""
    with sqlite3.connect(workspace / "sessions.db") as db:
        rows = db.execute("SELECT * FROM messages ORDER BY session_key,seq").fetchall()
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


async def run(state: Path, port: int) -> None:
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
        server = uvicorn.Server(uvicorn.Config(create_web_shell_app(workspace), host="127.0.0.1",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--port", type=int, default=24137)
    args = parser.parse_args()
    asyncio.run(run(args.state.resolve(), args.port))
