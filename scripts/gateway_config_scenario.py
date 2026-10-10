"""正式 Yoyo 复制 Gateway 配置；原配置与消息始终完整保留。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def run(base: Path) -> dict[str, bool]:
    """使用真实 bundle、Yoyo、插件固定输入和 Core 重启，不修改正式状态。"""
    from agent.config import Config
    from agent.migrations.runner import MigrationRunner
    from agent.plugin_composition.config_input import CONFIG_INPUT, load_config, save_config
    from agent.plugins.manifest import builtin_plugin_data_dir
    from agent.plugins.source_resolver import ResolvedPluginSource
    from bootstrap.init_workspace import init_workspace
    from bootstrap.tools import build_core_runtime
    from core.net.http import SharedHttpResources
    from session.log import MessageLog
    from session.message import Input

    home, source = base / "home", base / "source"
    os.environ.update(HOME=str(home), AKASHIC_PLUGIN_HOME=str(home),
                      AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS="",
                      AKASHIC_EXECUTION_MODE="local")
    shutil.copytree(ROOT / "plugins/gateway", source, ignore=shutil.ignore_patterns("__pycache__"))
    bundle = ResolvedPluginSource(source, "builtin", plugin_name="gateway")
    for conflict in (False, True):
        directory = base / ("conflict" if conflict else "copy")
        directory.mkdir()
        workspace, config = directory / "workspace", directory / "config.toml"
        config.write_text('# source comment\n[runtime]\n[app_server]\nenabled = false\nlisten = "custom.sock"\n'
                          'max_connections = "7"\ningress_queue_size = 19\noutbound_queue_size = 29\nmax_message_bytes = 10101\n')
        init_workspace(config_path=config, workspace=workspace)
        before_config = config.read_bytes()
        log = MessageLog(workspace / "sessions.db")
        log.writer("saved", author="user", source="saved", body_types=(Input,), content={}).append("kept", Input(()))
        log.close()
        with sqlite3.connect(workspace / "sessions.db") as database:
            before_rows = database.execute("SELECT * FROM messages ORDER BY rowid").fetchall()
        data = builtin_plugin_data_dir("gateway", workspace)
        runner = MigrationRunner(repo_root=ROOT, config_path=config, workspace=workspace, fixed_sources=(bundle,))
        if conflict:
            save_config(data, {"listen": "unrelated.sock"})
            original = (data / CONFIG_INPUT).read_bytes()
            try:
                runner.run()
            except RuntimeError as error:
                assert "拒绝覆盖" in str(error)
            else:
                raise AssertionError("迁移覆盖了用户配置")
            assert config.read_bytes() == before_config
            assert (data / CONFIG_INPUT).read_bytes() == original
            with sqlite3.connect(workspace / "migrations.sqlite3") as database:
                assert database.execute("SELECT COUNT(*) FROM _yoyo_migration WHERE migration_id = ?",
                    ("20261010_01_gateway_config_copy",)).fetchone()[0] == 0
            # 用户在隔离场景中明确选择旧 Core 设置；原目标保存在名称清楚的恢复点。
            (data / CONFIG_INPUT).rename(data / "before-source-choice.json")
        outcome = runner.run()
        assert "20261010_01_gateway_config_copy" in outcome.migrations
        copied, _ = load_config(data)
        assert copied == {"enabled": False, "listen": "custom.sock", "max_connections": 7,
                          "ingress_queue_size": 19, "outbound_queue_size": 29, "max_message_bytes": 10101}
        copied_bytes = (data / CONFIG_INPUT).read_bytes()
        assert runner.run().state == "current"
        assert (data / CONFIG_INPUT).read_bytes() == copied_bytes
        assert config.read_bytes() == before_config
        legacy = Config.load(config, workspace=workspace)
        assert legacy.app_server.listen == "custom.sock" and legacy.app_server.max_connections == 7
        # 两次实际 boot 从固定输入加载同一个 schema；没有 source table 或配置减少。
        for _ in range(2):
            http = SharedHttpResources()
            core = build_core_runtime(legacy, workspace, http, plugin_dirs=[source])
            try:
                await core.start()
                generation = core.plugin_manager.generation("gateway")
                assert generation is not None and generation.fiber is not None
                assert generation.config_projection == copied
            finally:
                await core.stop()
                await http.aclose()
        assert config.read_bytes() == before_config and (data / CONFIG_INPUT).read_bytes() == copied_bytes
        with sqlite3.connect(workspace / "sessions.db") as database:
            assert database.execute("SELECT * FROM messages ORDER BY rowid").fetchall() == before_rows
    return {"native_yoyo_copy": True, "legacy_values_preserved": True,
            "conflict_not_applied": True, "retry_after_explicit_choice": True,
            "fixed_input_two_boots": True, "source_and_messages_unchanged": True}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="gateway-config-") as path:
        print(json.dumps(asyncio.run(run(Path(path)))))
