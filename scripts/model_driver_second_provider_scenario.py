"""同一 Models 和 Reply 消费者先后使用两种独立 HTTP 驱动。"""
from __future__ import annotations

import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def run(distribution: Path, directory: Path) -> dict[str, object]:
    """实际安装替代驱动，保留消费者 Fiber，经 CLI 回复后再重启回复。"""
    from agent.config import Config
    from agent.plugins.bundles import set_plugin_choice
    from agent.plugins.composable import ComposablePlugin
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from scripts.install_plugin_distribution import ensure_bundle
    from scripts.artifact_provider_scenario import repository
    from scripts.bundle_modes_scenario import headless_reply

    workspace, home, config = directory / "workspace", directory / "home", directory / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home), AKASHIC_PLUGIN_DISTRIBUTION=str(distribution),
        AKASHIC_PLUGIN_BUNDLE="headless", AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "sdk/python/src"))))
    config.write_text("[runtime]\n")
    init_workspace(config_path=config, workspace=workspace)
    ensure_bundle(distribution, distribution / "bundles/headless.toml", workspace=workspace,
        plugins_home=home, config_path=config, receipt_path=workspace / "runtime/distribution-install.json")
    source = directory / "alternate"
    repository(ROOT / "examples/text_model_driver", source)
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)

    counts = Counter()
    original = ComposablePlugin.apply
    async def counted(self, ctx):
        counts[ctx.runtime.plugin_id] += 1
        await original(self, ctx)
    ComposablePlugin.apply = counted

    async def replace():
        host = app.core.plugin_manager
        before = counts.copy()
        consumers = {name: host.generation(name + "@release").fiber for name in ("models", "reply")}
        set_plugin_choice(workspace, "openai-compatible@release", enabled=False, distribution=distribution)
        await host.reconcile_disabled_and_drain("openai-compatible@release")
        await host.install(source=str(source), marketplace="lab", ref_name="", sparse_paths=[], update_id="text-driver")
        await host.wait_idle()
        assert host.read_update("text-driver").state == "active"
        assert all(host.generation(name + "@release").fiber is fiber for name, fiber in consumers.items())
        assert all(counts[name + "@release"] == before[name + "@release"] for name in consumers)

    async def restart():
        nonlocal app
        await app.shutdown()
        app = AppRuntime(Config.load(config, workspace=workspace), workspace)
        await app.start()

    try:
        await app.start()
        await headless_reply(app, workspace, config, between_replies=(replace, restart))
    finally:
        try:
            await app.shutdown()
        finally:
            ComposablePlugin.apply = original
    assert not json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
    return {"independent_http_driver": True, "cli_replies": 3, "models_and_reply_apply_delta": 0,
            "restart": True, "clean_shutdown": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="akashic-second-driver-") as directory:
        print(json.dumps(asyncio.run(run(Path(sys.argv[1]).resolve(), Path(directory)))))
