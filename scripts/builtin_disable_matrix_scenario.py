"""逐个拔除真实发行版插件，记录依赖缺失及无关插件的 apply 次数。"""
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
    """正式安装完整 bundle，在同一宿主逐项禁用和重新安装，最后排空退出。"""
    from agent.config import Config
    from agent.plugin_composition import FiberState
    from agent.plugins.bundles import load_bundle, set_plugin_choice
    from agent.plugins.composable import ComposablePlugin
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from scripts.install_plugin_distribution import ensure_bundle

    # 1. 使用隔离目录及真实制品；计数器只观察原 apply，不替代执行。
    workspace, home, config = directory / "workspace", directory / "home", directory / "config.toml"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home),
        AKASHIC_PLUGIN_DISTRIBUTION=str(distribution), AKASHIC_PLUGIN_BUNDLE="base",
        AKASHIC_EXTRA_PLUGIN_DIRS="", AKASHIC_EXECUTION_MODE="local")
    os.environ.pop("AKASHIC_WEB_PORT", None)
    config.write_text("[runtime]\n")
    init_workspace(config_path=config, workspace=workspace)
    ensure_bundle(distribution, distribution / "bundles/base.toml", workspace=workspace,
        plugins_home=home, config_path=config, receipt_path=workspace / "runtime/distribution-install.json")
    rows = load_bundle(distribution / "bundles/base.toml")
    expected = {row.plugin for row in rows if not row.disabled}
    assert len(expected) == len(rows), "矩阵必须包含 base 的每一个启用 row"
    calls: Counter[str] = Counter()
    original = ComposablePlugin.apply

    async def counted(self, ctx):
        calls[ctx.runtime.plugin_id] += 1
        await original(self, ctx)

    ComposablePlugin.apply = counted
    report = []
    try:
        # 2. 每一项从完整已安装组合启动，依赖闭包来自实际组合图。
        for target in sorted(expected):
            app = AppRuntime(Config.load(config, workspace=workspace), workspace)
            await app.start()
            host = app.core.plugin_manager
            root = host.live_root
            assert root is not None
            assert set(host._active_generations) == expected
            assert all(item.state is not FiberState.FAILED for item in root.fibers())
            owners = root.plugin_service_owners()
            dependencies = root.plugin_dependencies()
            affected = {target}
            while True:
                following = {name for name, keys in dependencies.items()
                             if any(owners.get(key) in affected for key in keys)}
                if following <= affected:
                    break
                affected.update(following)
            removed_services = {key.name for key, owner in owners.items() if owner in affected}
            baseline_missing = {item.path: set(item.missing_services) for item in root.fibers()}
            before = calls.copy()
            fibers = {name: item.fiber for name, item in host._active_generations.items()}
            set_plugin_choice(workspace, target, enabled=False, distribution=distribution)
            async with asyncio.timeout(60):
                await host.reconcile_disabled_and_drain(target)
            assert host.generation(target) is None
            unrelated = expected - affected
            assert all(calls[name] == before[name] for name in unrelated), (target, calls - before)
            assert all(host.generation(name).fiber is fibers[name] for name in unrelated)
            waiting = []
            for item in root.fibers():
                assert item.state is not FiberState.FAILED, (target, item.path, item.error)
                if item.state is FiberState.PENDING:
                    assert set(item.missing_services) <= removed_services | baseline_missing.get(item.path, set()), (target, item.path, item.missing_services)
                    waiting.append({"fiber": item.path, "missing": item.missing_services})
            entry = {"disabled": target, "affected": sorted(affected - {target}),
                     "unrelated": len(unrelated), "unrelated_apply_delta": 0, "pending": waiting}
            report.append(entry)
            print(json.dumps(entry), flush=True)
            # 3. 禁用后的持久选择必须能冷启动；再离线发布完整 bundle。
            await app.shutdown()
            app = AppRuntime(Config.load(config, workspace=workspace), workspace)
            await app.start()
            host = app.core.plugin_manager
            assert host.generation(target) is None
            assert host.live_root is not None
            assert all(item.state is not FiberState.FAILED for item in host.live_root.fibers())
            await app.shutdown()
            assert not host._active_generations
            assert not json.loads((workspace / "runtime/endpoints.json").read_text())["endpoints"]
            entry["disabled_restart"] = True
            set_plugin_choice(workspace, target, enabled=True, distribution=distribution)
            ensure_bundle(distribution, distribution / "bundles/base.toml", workspace=workspace,
                plugins_home=home, config_path=config, receipt_path=workspace / "runtime/distribution-install.json")
    finally:
        try:
            await app.shutdown()
        finally:
            ComposablePlugin.apply = original
    result = {"source_commit": json.loads((distribution / "distribution.json").read_text())["source_commit"],
              "rows": len(rows), "cases": report, "clean_shutdown": True}
    (directory / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    distribution = Path(sys.argv[1]).resolve(strict=True)
    directory = Path(tempfile.mkdtemp(prefix="akashic-disable-matrix-"))
    print(json.dumps({"evidence": str(directory)}), flush=True)
    result = asyncio.run(run(distribution, directory))
    print(json.dumps({"rows": result["rows"], "clean_shutdown": True}), flush=True)
