"""
调试脚本：直接把脱敏样例对话喂给 consolidation，再调一次 optimizer，
全链路验证 agent_context 的提取和保留效果。

用法（在容器内）：
  python /app/docker/debug/run_consolidation_then_optimizer.py \
    --config /sandbox/config.toml \
    --workspace /sandbox/workspace
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.config_models import Config
from bootstrap.memory import build_memory_admin_runtime
from bootstrap.providers import build_providers
from core.memory.markdown import ConsolidateRequest
from core.net.http import SharedHttpResources
from proactive_v2.memory_optimizer import MemoryOptimizer
from session.manager import Session

# 正例：浏览器已部署，助手被告知可以直接用
# 反例：NAT/CGNAT 讨论，只是诊断网络拓扑，没有实际可调用的服务
CONVERSATION: list[dict[str, str]] = [
    {
        "role": "assistant",
        "content": (
            "ExampleBrowser 不做这个事。它只解决一个问题：降低自动化浏览器特征，"
            "不解决会话持久化。\n\n"
            "你想要的是另一种东西——常住浏览器 / CDP 远程浏览器服务。"
            "在一台固定机器上跑一个 Chromium 进程，带上 --remote-debugging-port=9311 "
            "和固定的 --user-data-dir，让它一直活着。你登录一次之后，cookies 和 session "
            "就存在那个 user data 目录里。然后我这边 Playwright 不 launch()，"
            "而是 connect_over_cdp('http://localhost:9311')，连到那个常住浏览器实例上。"
        ),
    },
    {
        "role": "user",
        "content": (
            "两个问题 我不希望这个无头浏览器影响我 "
            "比如说我在服务器上 操作浏览器 他是一个单独的无头进程 一直跑的 这可以做到吗"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "完全可以。这是在服务器上跑的单独进程，跟你本地用的 Chrome/Chromium 是两码事。"
            "不影响你平时浏览。\n\n"
            "我可以直接配一个 systemd service：开机自启，挂了自动重启，"
            "开 --remote-debugging-port=9311，带固定 --user-data-dir 来持久化登录态，"
            "跑 ExampleBrowser 的改装二进制。"
            "之后我需要用的时候就 connect_over_cdp('http://localhost:9311') 连上去。"
            "你只用做一件事——启动后登录一次常用站点，后面就不用管了。"
        ),
    },
    {
        "role": "user",
        "content": (
            "你先把这个无头的建出来 然后让我帮你把东西登陆了 "
            "问题在于 你现在就是跑在我的开发机上的"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "好了，常住浏览器已经跑起来了。总结一下现在的情况：\n\n"
            "服务状态：\n"
            "- systemctl --user status example-browser.service → active (running)\n"
            "- 监听端口 9311，CDP WebSocket 已就绪\n"
            "- 开机自启，挂了自动重启\n"
            "- 数据存 /srv/example-browser/profile/，重启不丢\n\n"
            "你登一次，我后面就可以复用。你去把常用站点登了，"
            "之后我直接 connect_over_cdp('http://localhost:9311') 就能继续操作。"
        ),
    },
    {
        "role": "user",
        "content": (
            "没有直接端口映射然后内网穿透之类的方法吗 "
            "我记得以前和别人联机游戏是这么搞得"
        ),
    },
    {
        "role": "assistant",
        "content": (
            "当然有！你说的那个联机游戏的操作就是 UPnP + 端口映射，当年确实好用——"
            "但问题出在 CGNAT 上。\n\n"
            "你的光猫（10.64.12.34）\n"
            "    ↓\n"
            "运营商 CGNAT 网关（10.64.0.1）← 这里才是最外层\n"
            "    ↓\n"
            "真正的公网 IP\n\n"
            "你在光猫上做端口映射，只是把 192.168.x.x 映射到 10.64.x.x。"
            "但运营商那头还有一层 NAT，它不认得你的端口映射。"
            "所以传统 UPnP + 端口映射在 CGNAT 下没用。\n\n"
            "有几个方案：① IPv6 直连 ② FRP / Cloudflare Tunnel 中转"
        ),
    },
]


async def main(config_path: str, workspace_str: str) -> None:
    workspace = Path(workspace_str)
    config = Config.load(config_path)
    http_resources = SharedHttpResources()
    provider, light_provider, _ = build_providers(config)
    memory_runtime = build_memory_admin_runtime(
        config=config,
        workspace=workspace,
        provider=provider,
        light_provider=light_provider,
        http_resources=http_resources,
    )
    store = memory_runtime.markdown.store

    print("=== PENDING.md (before consolidation) ===")
    print(store.read_pending() or "(empty)")

    session = Session(key="probe:agent_context_test")
    for msg in CONVERSATION:
        session.add_message(msg["role"], msg["content"])

    print(f"\n>>> consolidation 开始（{len(CONVERSATION)} 条消息）...\n")
    result = await memory_runtime.markdown.maintenance.consolidate(
        ConsolidateRequest(session=session, force=True, archive_all=False)
    )
    print(f"consolidation 结果: {result}")

    print("\n=== PENDING.md (after consolidation) ===")
    pending = store.read_pending()
    print(pending or "(empty)")

    print("\n>>> optimizer 开始...\n")
    optimizer = MemoryOptimizer(
        memory=store,
        provider=provider,
        model=config.model,
        max_tokens=8192,
    )
    await optimizer.optimize()

    print("\n=== MEMORY.md (after optimizer) ===")
    print(store.read_long_term())
    print("\n=== SELF.md (after optimizer) ===")
    print(store.read_self())

    await memory_runtime.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/sandbox/config.toml")
    parser.add_argument("--workspace", default="/sandbox/workspace")
    args = parser.parse_args()
    asyncio.run(main(args.config, args.workspace))
