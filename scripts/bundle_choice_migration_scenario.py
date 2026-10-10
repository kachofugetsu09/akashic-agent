"""旧 Core 生成选择后，真实 Yoyo 迁移、文件写入失败和共享目录隔离验收。"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = '''import os, subprocess, sys
from pathlib import Path
from bootstrap.init_workspace import init_workspace
from agent.plugins.install import install_git_plugin
from agent.plugins.manifest import set_plugin_enabled
from plugins.ledger.log import MessageLog
from plugins.ledger.contract import Input, ContentPart, ContentReferences
base=Path(sys.argv[1]); home=base/'home'; home.mkdir()
os.environ['AKASHIC_PLUGIN_HOME']=str(home)
source=base/'source'; source.mkdir()
(source/'plugin.py').write_text('api_version=3\\nname="alpha"\\nversion="1.0.0"\\nasync def apply(ctx):\\n    pass\\n')
subprocess.run(['git','init','-q',str(source)],check=True)
subprocess.run(['git','-C',str(source),'add','.'],check=True)
subprocess.run(['git','-C',str(source),'-c','user.name=Scenario','-c','user.email=scenario@example.invalid','commit','-qm','source'],check=True)
for name in ('one','two'):
    state=base/name; state.mkdir(); config=state/'config.toml'; workspace=state/'workspace'
    config.write_text('[runtime]\\n[agent.plugins]\\ndisabled_builtin = ["local_only"]\\n')
    init_workspace(config_path=config,workspace=workspace)
    install_git_plugin(workspace=workspace,source=str(source),marketplace='lab',plugins_home=home)
    log=MessageLog(workspace/'sessions.db')
    log.writer('kept',author='user',source='scenario',body_types=(Input,),content={"text": lambda part: ContentReferences()}).append('before',Input((ContentPart('text','preserve complete content'),)))
    log.close()
    (workspace/'plugin-data/alpha-lab/sentinel.bin').write_bytes(b'untouched')
init_workspace(config_path=base/'one/config.toml',workspace=base/'shared')
set_plugin_enabled('alpha@lab',enabled=True,plugins_home=home)
'''


def run(old_core: Path) -> None:
    """旧值、目标值和耐久回执均从真实文件取得，不替换迁移执行函数。"""
    from agent.migrations.runner import MigrationRunner
    from agent.plugins.bundles import plugin_choices, set_plugin_choice
    from agent.config import Config

    base = Path(tempfile.mkdtemp(prefix="akashic-bundle-migration-"))
    os.environ.update(HOME=str(base / "empty-home"), AKASHIC_PLUGIN_DISTRIBUTION="", AKASHIC_EXTRA_PLUGIN_DIRS="")
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join((str(old_core), str(old_core / "sdk/python/src")))}
    subprocess.run([sys.executable, "-c", SEED, str(base)], cwd=old_core, env=environment, check=True)
    home = base / "home"
    before_manifest = (home / "manifest.toml").read_bytes()
    evidence = {}
    for name in ("one", "two"):
        state = base / name
        workspace, config = state / "workspace", state / "config.toml"
        before = {path: path.read_bytes() for path in (workspace / "sessions.db", workspace / "plugin-data/alpha-lab/sentinel.bin")}
        original_config = config.read_bytes()
        runner = MigrationRunner(repo_root=ROOT, config_path=config, workspace=workspace, plugins_home=home)
        if name == "one":
            # 1. patch 可提交而 config 目录不能原子替换；失败必须保留原 config 和未完成回执。
            # 该文件也是中断后允许存在的真实恢复材料；另一路径验证迁移自行创建它。
            saved_config = config.with_name(config.name + ".before-bundle-choices.toml")
            saved_config.write_bytes(original_config)
            saved_config.chmod(0o600)
            state.chmod(0o500)
            try:
                try:
                    runner.run()
                except RuntimeError as error:
                    assert isinstance(error.__cause__, PermissionError), repr(error)
                else:
                    raise AssertionError("只读配置目录没有阻止写入")
            finally:
                state.chmod(0o700)
            assert config.read_bytes() == original_config
            assert plugin_choices(workspace) == {"alpha@lab": True, "local_only": False}
            assert "20261011_02_bundle_choices" in runner.check()
        # 2. 同一恢复计划重试完成，原始消息文件与 plugin-data 完整字节不变。
        runner.run()
        assert plugin_choices(workspace) == {"alpha@lab": True, "local_only": False}
        Config.load(config, workspace=workspace)
        assert "disabled_builtin" not in config.read_text()
        plan = json.loads((workspace / "runtime/before-bundle-choices.json").read_text())
        assert plan["config_before"].encode() == original_config
        assert plan["legacy_manifest"].encode() == before_manifest
        for path, content in before.items():
            assert path.read_bytes() == content
        stable = (config.read_bytes(), (workspace / "bundle.patch.toml").read_bytes())
        assert runner.run().state == "current"
        assert stable == (config.read_bytes(), (workspace / "bundle.patch.toml").read_bytes())
        if name == "one":
            set_plugin_choice(workspace, "alpha@lab", enabled=False)
        evidence[name] = plugin_choices(workspace)
    # 3. 共享旧安装目录的第二个 workspace 仍从原清单迁入，不继承第一个的新禁用。
    assert evidence["one"]["alpha@lab"] is False and evidence["two"]["alpha@lab"] is True
    assert (home / "manifest.toml").read_bytes() == before_manifest
    MigrationRunner(repo_root=ROOT, config_path=base / "one/config.toml",
                    workspace=base / "shared", plugins_home=home).run()
    assert plugin_choices(base / "shared") == {"alpha@lab": True, "local_only": False}
    # 新版本新建 workspace 不继承保留给旧 workspace 的全局清单。
    from bootstrap.init_workspace import init_workspace
    from agent.plugins.manager import PluginManager
    fresh = base / "fresh"
    config = base / "fresh-config.toml"
    init_workspace(config_path=config, workspace=fresh)
    MigrationRunner(repo_root=ROOT, config_path=config, workspace=fresh, plugins_home=home).run()
    async def empty_runtime():
        manager = PluginManager([], workspace=fresh, installed_cache_root=home / "cache")
        try:
            await manager.load_all()
            assert not manager._active_generations
        finally:
            await manager.terminate_all()
    asyncio.run(empty_runtime())
    print(json.dumps({"evidence": str(base), "failure_retry": True, "shared_home_isolation": True,
                      "source_bytes_preserved": True, "choices": evidence}))


if __name__ == "__main__":
    run(Path(sys.argv[1]).resolve(strict=True))
