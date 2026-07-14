from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent.config import Config
from agent.plugins.manifest import write_package_manifest
from bootstrap import app as bootstrap_app
from bootstrap.init_workspace import init_workspace
from bootstrap.tools import build_core_runtime
from core.net.http import SharedHttpResources
from proactive_v2.loop import ProactiveLoop


MEMORY_CASES = (
    ("disabled", False, ""),
    ("default", True, ""),
    ("akasha", True, "akasha"),
)
PROACTIVE_CASES = (
    ("off", False, "default", None),
    ("default", True, "default", "default-proactive"),
    ("wake", True, "wake", "wake-proactive"),
)


class _FakeServer:
    def __init__(self) -> None:
        self.should_exit = False

    async def serve(self) -> None:
        while not self.should_exit:
            await asyncio.sleep(0)


def _set_toml_value(text: str, section: str, key: str, value: str) -> str:
    """更新指定 TOML section 的单个字段。"""

    # 1. 定位 section 边界
    lines = text.splitlines()
    header = f"[{section}]"
    start = lines.index(header) + 1
    end = next(
        (index for index in range(start, len(lines)) if lines[index].startswith("[")),
        len(lines),
    )

    # 2. 更新已有字段或在 section 尾部补充
    prefix = f"{key} ="
    for index in range(start, end):
        if lines[index].startswith(prefix):
            lines[index] = f"{key} = {value}"
            break
    else:
        lines.insert(end, f"{key} = {value}")
    return "\n".join(lines) + "\n"


def _prepare_fresh_case(
    root: Path,
    *,
    memory_enabled: bool,
    memory_engine: str,
    proactive_enabled: bool,
    proactive_lifecycle: str,
    proactive_package: str | None,
) -> tuple[Path, Path, Path]:
    """从 init 产物准备一个隔离配置组合。"""

    # 1. 生成全新 HOME、配置和工作区
    home = root / "home"
    config_path = root / "config.toml"
    workspace = root / "workspace"
    home.mkdir()
    _ = init_workspace(config_path=config_path, workspace=workspace)

    # 2. 只修改组合维度
    text = config_path.read_text(encoding="utf-8")
    text = _set_toml_value(
        text,
        "memory",
        "enabled",
        str(memory_enabled).lower(),
    )
    text = _set_toml_value(text, "memory", "engine", f'"{memory_engine}"')
    text = _set_toml_value(
        text,
        "proactive",
        "enabled",
        str(proactive_enabled).lower(),
    )
    text = _set_toml_value(
        text,
        "proactive",
        "lifecycle",
        f'"{proactive_lifecycle}"',
    )
    config_path.write_text(text, encoding="utf-8")

    # 3. 显式选择互斥的 proactive 插件包
    write_package_manifest(
        {
            "default-proactive": proactive_package == "default-proactive",
            "wake-proactive": proactive_package == "wake-proactive",
        },
        plugins_home=home / ".akashic-plugin",
    )
    return home, config_path, workspace


@pytest.mark.asyncio
@pytest.mark.parametrize(("memory_name", "memory_enabled", "memory_engine"), MEMORY_CASES)
@pytest.mark.parametrize(
    ("proactive_name", "proactive_enabled", "proactive_lifecycle", "package"),
    PROACTIVE_CASES,
)
async def test_fresh_init_core_configuration_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_name: str,
    memory_enabled: bool,
    memory_engine: str,
    proactive_name: str,
    proactive_enabled: bool,
    proactive_lifecycle: str,
    package: str | None,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    home, config_path, workspace = _prepare_fresh_case(
        tmp_path,
        memory_enabled=memory_enabled,
        memory_engine=memory_engine,
        proactive_enabled=proactive_enabled,
        proactive_lifecycle=proactive_lifecycle,
        proactive_package=package,
    )
    config = Config.load(config_path)
    resources = SharedHttpResources()
    runtime = build_core_runtime(config, workspace, resources)

    try:
        await runtime.start()
        assert {
            "workspace_mcp_apply",
            "workspace_mcp_remove",
            "workspace_mcp_status",
        } <= runtime.tools.get_registered_names()
        assert runtime.memory_runtime.engine.describe().name == memory_name
        expected_lifecycles = [] if proactive_name == "off" else [proactive_name]
        assert [
            lifecycle.id for lifecycle in runtime.plugin_manager.proactive_lifecycles
        ] == expected_lifecycles
    finally:
        await runtime.stop()
        await runtime.memory_runtime.aclose()
        await resources.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("memory_name", "memory_enabled", "memory_engine"), MEMORY_CASES[1:])
@pytest.mark.parametrize(
    ("proactive_name", "proactive_enabled", "proactive_lifecycle", "package"),
    PROACTIVE_CASES,
)
async def test_fresh_init_runtime_start_stop_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    memory_name: str,
    memory_enabled: bool,
    memory_engine: str,
    proactive_name: str,
    proactive_enabled: bool,
    proactive_lifecycle: str,
    package: str | None,
) -> None:
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", lambda: home)
    home, config_path, workspace = _prepare_fresh_case(
        tmp_path,
        memory_enabled=memory_enabled,
        memory_engine=memory_engine,
        proactive_enabled=proactive_enabled,
        proactive_lifecycle=proactive_lifecycle,
        proactive_package=package,
    )
    monkeypatch.setattr(
        bootstrap_app,
        "build_dashboard_server",
        lambda **_kwargs: _FakeServer(),
    )
    monkeypatch.setattr(
        bootstrap_app,
        "build_chat_server",
        lambda **_kwargs: _FakeServer(),
    )

    async def _wait_for_cancel(_loop: ProactiveLoop) -> float:
        await asyncio.Event().wait()
        raise AssertionError("不可达")

    monkeypatch.setattr(ProactiveLoop, "_tick", _wait_for_cancel)
    runtime = bootstrap_app.build_app_runtime(Config.load(config_path), workspace=workspace)
    task = asyncio.create_task(runtime.run())
    try:
        for _ in range(100):
            if runtime._started or task.done():
                break
            await asyncio.sleep(0)

        if task.done():
            await task
        assert runtime._started
        assert runtime.memory_runtime.engine.describe().name == memory_name
        assert (runtime.proactive_loop is not None) is proactive_enabled
    finally:
        if not task.done():
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=5)

    assert not (workspace / "akashic.sock").exists()
    plugin_name = "default_memory" if memory_name == "default" else memory_name
    config_dir = home / ".akashic-plugin" / "data" / f"{plugin_name}-builtin"
    assert (config_dir / "config.local.toml").is_file()
