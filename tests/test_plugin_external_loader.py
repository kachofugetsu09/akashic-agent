"""生产 Core 默认只解析安装制品，开发目录必须显式提供。"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.config_models import Config
from bootstrap import tools as bootstrap
from core.net.http import SharedHttpResources


def test_resolve_plugin_dirs_does_not_add_checkout_plugins_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)

    roots = bootstrap._resolve_plugin_dirs(tmp_path)

    assert roots == []
    checkout_plugins = Path(bootstrap.__file__).resolve().parents[1] / "plugins"
    assert checkout_plugins not in roots


def test_resolve_plugin_dirs_accepts_only_explicit_development_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = tmp_path / "development-plugins"
    extra = tmp_path / "extra-plugins"
    monkeypatch.setenv("AKASHIC_EXTRA_PLUGIN_DIRS", str(extra))

    roots = bootstrap._resolve_plugin_dirs(tmp_path, plugin_dirs=[development])

    assert roots == [development, extra]


@pytest.mark.asyncio
async def test_core_starts_with_no_checkout_plugins_and_keeps_manager_usable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)
    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(Config(), workspace, http)
    try:
        assert core.plugin_manager.discover() == []
        assert core.plugin_manager._dirs == []
        assert core.restart_gate.boot_id != "unmanaged"
        assert core.plugin_manager.channel_generation_host.boot_id == core.restart_gate.boot_id
        await core.start()
        assert core.plugin_manager.discover() == []
        snapshot = core.plugin_manager.current_snapshot
        assert snapshot is not None and snapshot.composition_root is not None
        assert snapshot.composition_root.receipt().ready
        assert snapshot.generations == {}
        assert "identity:" in await core.inspect_modules()
    finally:
        await core.stop()
        await core.bus.aclose()
        await http.aclose()


@pytest.mark.asyncio
async def test_unmanaged_core_runtime_gets_a_new_boot_id_per_host(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """两个无 supervisor 的真实 Core host 不能共享 transport boot identity。"""

    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    first_workspace = tmp_path / "first-workspace"
    second_workspace = tmp_path / "second-workspace"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first_http = SharedHttpResources()
    second_http = SharedHttpResources()
    first = bootstrap.build_core_runtime(Config(), first_workspace, first_http)
    second = bootstrap.build_core_runtime(Config(), second_workspace, second_http)
    try:
        assert first.restart_gate.boot_id != second.restart_gate.boot_id
        assert first.plugin_manager.channel_generation_host.boot_id == first.restart_gate.boot_id
        assert second.plugin_manager.channel_generation_host.boot_id == second.restart_gate.boot_id
    finally:
        await first.stop()
        await first.bus.aclose()
        await first_http.aclose()
        await second.stop()
        await second.bus.aclose()
        await second_http.aclose()


@pytest.mark.asyncio
async def test_build_core_runtime_keeps_explicit_plugin_dirs_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    development = tmp_path / "development-plugins"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.setenv("AKASHIC_PLUGIN_HOME", str(tmp_path / "plugin-home"))
    monkeypatch.delenv("AKASHIC_EXTRA_PLUGIN_DIRS", raising=False)

    http = SharedHttpResources()
    core = bootstrap.build_core_runtime(
        Config(), workspace, http, plugin_dirs=[development]
    )
    try:
        assert core.plugin_manager._dirs == [development]
    finally:
        await core.stop()
        await core.bus.aclose()
        await http.aclose()
