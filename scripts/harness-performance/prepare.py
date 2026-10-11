"""为一侧源码构建正式发行版、安装 base 组合并配置 mock 模型，产出可复制的 workspace 模板。

插件来自 --root 仓库 HEAD 提交构建的发行版（与发布镜像同一条链），核心代码直接运行 --root 工作树；
因此本地对比前要先提交插件改动。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

MOCK_PORT = 18457
WEB_PORT = 18458


@dataclass(frozen=True)
class Side:
    root: Path
    state: Path
    distribution: Path
    template: Path


def side_paths(root: Path, side: Path) -> Side:
    side = side.resolve()
    return Side(root=root.resolve(), state=side / "state", distribution=side / "distribution",
                template=side / "template")


# 测量与准备共用同一份隔离环境；HOME 与插件 home 都在 state 内，不触碰开发者目录。
def environment(side: Side) -> dict[str, str]:
    return {
        "HOME": str(side.state / "home"),
        "AKASHIC_PLUGIN_HOME": str(side.state / "home"),
        "AKASHIC_PLUGIN_DISTRIBUTION": str(side.distribution),
        "AKASHIC_PLUGIN_BUNDLE": "base",
        "AKASHIC_EXTRA_PLUGIN_DIRS": "",
        "AKASHIC_EXECUTION_MODE": "local",
        "AKASHIC_WEB_PORT": str(WEB_PORT),
        "PYTHONPATH": os.pathsep.join((str(side.root), str(side.root / "sdk/python/src"))),
    }


# 与发布镜像相同的两步：从提交构建发行版，再按目标 Python 下载依赖 wheel。
def build_distribution(side: Side) -> None:
    if side.distribution.exists():
        shutil.rmtree(side.distribution)
    # 构建只读源码提交、只写发行版目录；npm/pip 的下载缓存沿用调用者环境。
    # 构建在临时 git 仓库里提交插件源码；自动 gc/maintenance 会在后台写对象库，与临时目录清理竞争。
    env = {**os.environ, "PYTHONPATH": environment(side)["PYTHONPATH"],
           "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "gc.auto", "GIT_CONFIG_VALUE_0": "0",
           "GIT_CONFIG_KEY_1": "maintenance.auto", "GIT_CONFIG_VALUE_1": "false"}
    _ = subprocess.run([sys.executable, "scripts/build_plugin_distribution.py", "--output", str(side.distribution)],
                       cwd=side.root, env=env, check=True, stdout=subprocess.DEVNULL)
    _ = subprocess.run([sys.executable, "scripts/distribution_runtime.py", "prepare-wheels",
                        "--distribution", str(side.distribution)], cwd=side.root, env=env, check=True)


# 只通过正式安装链与模型设置命令写入状态，不直接改数据库。
async def install_and_configure(side: Side) -> None:
    from agent.config import Config
    from agent.plugin_composition.config_input import save_config
    from bootstrap.app import AppRuntime
    from bootstrap.init_workspace import init_workspace
    from plugins.gateway.contract import RpcMethod
    from scripts.install_plugin_distribution import ensure_bundle

    # 1. 新建 workspace 与 base 组合；reply 的 max_steps=0 表示不设步数上限（默认 40 跑不完 100 轮）。
    workspace, config = side.state / "workspace", side.state / "config.toml"
    (side.state / "home").mkdir()
    _ = config.write_text("[runtime]\n")
    _ = init_workspace(config_path=config, workspace=workspace)
    save_config(workspace / "plugin-data/reply-release", {"max_steps": 0})
    _ = ensure_bundle(side.distribution, side.distribution / "bundles/base.toml", workspace=workspace,
                      plugins_home=side.state / "home", config_path=config,
                      receipt_path=workspace / "runtime/distribution-install.json")

    # 2. 启动一次运行时，经 models/command 注册 mock 连接、模型与默认角色。
    app = AppRuntime(Config.load(config, workspace=workspace), workspace)
    await app.start()
    try:
        core = app.core
        assert core is not None
        method = core.plugin_manager.live_root.service_value(RpcMethod.key("models/command"))
        assert method is not None

        async def command(payload: dict[str, object]) -> int:
            result = cast(dict[str, Any], await method.call(method.params.model_validate(payload)))
            if result["status"] != 200:
                raise RuntimeError(f"models/command 失败: {result}")
            return int(result["body"]["revision"])

        revision = await command({
            "type": "add_connection", "expected_revision": 0, "connection_id": "mock", "name": "Mock",
            "driver_id": "openai-compatible", "endpoint": f"http://127.0.0.1:{MOCK_PORT}/v1",
            "auth_identity": "mock", "credential": {"api_key": "mock-key"},
            "driver_config": {"allow_unverified_manual": True},
        })
        revision = await command({
            "type": "add_model", "expected_revision": revision, "model_id": "mock", "connection_id": "mock",
            "kind": "chat", "model": "llm-mock",
            "capabilities": {"context_window": 1_000_000, "max_output_tokens": 16384, "supports_tool_calls": True},
            "capability_sources": {},
        })
        _ = await command({"type": "set_default", "expected_revision": revision, "role": "default", "model_id": "mock"})
    finally:
        await app.shutdown()

    # 3. 模板只含 workspace；每次测量从模板复制，插件 home 的宿主缓存跨次保留。
    if side.template.exists():
        shutil.rmtree(side.template)
    _ = shutil.copytree(workspace, side.template)


# 注册连接时 mock 需要在线；用一份三条纯文本的小剧本，不消耗正式剧本。
def _start_setup_mock(benchmark: Path, side: Side) -> subprocess.Popen[bytes]:
    reply: dict[str, object] = {"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
    script = side.state.parent / "setup-mock.json"
    _ = script.write_text(json.dumps({"responses": [reply] * 3}))
    log = (side.state.parent / "setup-mock.log").open("wb")
    mock = subprocess.Popen(["bun", str(benchmark / "src/server.ts"), "--script", str(script),
                             "--port", str(MOCK_PORT), "--exhausted", "error"], stdout=log, stderr=log)
    for _ in range(200):
        if mock.poll() is not None:
            raise RuntimeError("setup mock 退出，见 setup-mock.log")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{MOCK_PORT}/__mock/status"):
                return mock
        except urllib.error.URLError:
            time.sleep(0.05)
    mock.terminate()
    raise TimeoutError("setup mock 就绪超时")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--root", type=Path, required=True)
    _ = parser.add_argument("--side", type=Path, required=True)
    _ = parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    side = side_paths(Path(args.root), Path(args.side))
    if side.state.exists():
        raise SystemExit(f"state 已存在: {side.state}")
    side.state.parent.mkdir(parents=True, exist_ok=True)
    build_distribution(side)
    side.state.mkdir()
    mock = _start_setup_mock(Path(args.benchmark).resolve(), side)
    os.environ.update(environment(side))
    sys.path[:0] = [str(side.root), str(side.root / "sdk/python/src")]
    try:
        asyncio.run(install_and_configure(side))
    finally:
        mock.terminate()
        _ = mock.wait(timeout=10)


if __name__ == "__main__":
    main()
