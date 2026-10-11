"""同一 SDK 消费者跨独立 Gateway 实现读取真实持久消息。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "sdk/python/src")]


async def run(distribution: Path, directory: Path) -> dict[str, object]:
    """正式安装默认组合，替换 Gateway 后核对相同消费结果与 Fiber 身份。"""
    from agent.config import Config
    from agent.plugins.bundles import set_plugin_choice
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from scripts.install_plugin_distribution import ensure_bundle
    from scripts.artifact_provider_scenario import repository
    from plugins.ledger.contract import MESSAGE_CATALOG, Input, ContentPart
    from plugins.content.contract import CONTENT
    from akashic_sdk import AsyncAkashic

    workspace, home, config = directory / "workspace", directory / "home", directory / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION=str(distribution),
        AKASHIC_PLUGIN_BUNDLE="headless", AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local")
    config.write_text("[runtime]\n")
    init_workspace(config_path=config, workspace=workspace)
    ensure_bundle(distribution, distribution / "bundles/headless.toml", workspace=workspace,
        plugins_home=home, config_path=config, receipt_path=workspace / "runtime/distribution-install.json")
    source = directory / "alternate"
    repository(ROOT / "examples/readonly_gateway", source)
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)

    async def read():
        endpoints = json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
        endpoint = next(row["address"] for row in endpoints if row["name"] == "gateway")
        client = await AsyncAkashic.connect(endpoint)
        try:
            return await client.session_list(), await client.message_read("akashic:kept")
        finally:
            await client.close()

    try:
        await app.start()
        host = app.core.plugin_manager
        root = host.live_root
        log = root.context.require(MESSAGE_CATALOG)._log
        log.writer("akashic:kept", author="user", source="scenario", body_types=(Input,),
            content={"text": root.context.require(CONTENT).check_text}).append("kept", Input((ContentPart("text", "unchanged history"),)))
        before = await read()
        ledger = host.generation("ledger@release").fiber
        set_plugin_choice(workspace, "gateway@release", enabled=False, distribution=distribution)
        await host.reconcile_disabled_and_drain("gateway@release")
        await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="readonly")
        await host.wait_idle()
        assert host.read_update("readonly").state == "active"
        assert await read() == before
        assert host.generation("ledger@release").fiber is ledger
    finally:
        await app.shutdown()
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    try:
        await app.start()
        assert await read() == before
    finally:
        await app.shutdown()
    assert not json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
    return {"independent_gateway": True, "same_sdk_queries": True, "ledger_fiber_stable": True,
            "restart": True, "clean_shutdown": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-second-gateway-") as directory:
        print(json.dumps(asyncio.run(run(Path(sys.argv[1]).resolve(), Path(directory)))))
