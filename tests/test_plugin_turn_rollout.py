from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent.control.models import TurnStatus
from agent.plugins.turn_rollout import TurnPluginRollout


class _Manager:
    def __init__(self) -> None:
        self.promoted: list[str] = []
        self.discarded: list[str] = []
        self.annotations: list[tuple[str, dict[str, object]]] = []
        self.installed = {"fitbit@github"}
        self.staged_candidate = True

    async def install_candidate(self, **_kwargs):
        return (
            SimpleNamespace(
                plugin_name="fitbit",
                marketplace="github",
                source_revision="rev-2",
                staged_candidate=self.staged_candidate,
            ),
            {
                "candidate_generation_id": "gen-2",
                "candidate_reload_tx_id": "tx-2",
            },
        )

    def annotate_reload(self, tx_id, details) -> None:
        self.annotations.append((tx_id, details))

    async def switch_ready(self, plugin_id):
        self.promoted.append(plugin_id)
        return {"publication_state": "promoted"}

    async def drop_candidate(self, plugin_id):
        self.discarded.append(plugin_id)
        return {"plugin_id": plugin_id, "publication_state": "discarded"}

    def require_installed_plugin(self, plugin_id) -> None:
        if plugin_id not in self.installed:
            raise RuntimeError(f"插件未安装: {plugin_id}")

async def _settle() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_attached_child_freezes_candidate_and_parent_promotes_after_validation(
    tmp_path: Path,
):
    manager = _Manager()
    uninstalled: list[str] = []

    async def uninstall(plugin_id: str) -> dict[str, object]:
        uninstalled.append(plugin_id)
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="main",
        sparse_paths=[],
    )

    capability = rollout.mint_child_capability("turn-parent")
    assert capability
    reserved = rollout.child_binding(capability, False)
    binding = rollout.child_binding(capability, True)
    assert reserved == binding
    assert binding == {
        "runtime": "latest",
        "ownerTurnId": "turn-parent",
        "pluginId": "fitbit@github",
        "generationId": "gen-2",
        "sourceRevision": "rev-2",
    }
    assert rollout.child_binding(capability, True) is None

    rollout.turn_terminal(
        "turn-child",
        TurnStatus.COMPLETED,
        {
            "turnId": "turn-child",
            "_pluginRolloutOwnerTurnId": "turn-parent",
            "_pluginRolloutGenerationId": "gen-2",
            "_pluginRolloutSourceRevision": "rev-2",
        },
    )
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await _settle()

    assert manager.promoted == ["fitbit@github"]
    assert manager.discarded == []
    assert uninstalled == []
    assert "已经成功提交" in rollout.consume_fact()
    assert rollout.consume_fact() == ""


@pytest.mark.asyncio
async def test_unvalidated_or_failed_parent_discards_candidate(tmp_path: Path):
    manager = _Manager()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await _settle()

    assert manager.promoted == []
    assert manager.discarded == ["fitbit@github"]


@pytest.mark.asyncio
async def test_reserved_child_capability_expires_when_parent_seals(
    tmp_path: Path,
) -> None:
    manager = _Manager()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )
    capability = rollout.mint_child_capability("turn-parent")
    assert rollout.child_binding(capability, False) is not None
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})

    assert rollout.child_binding(capability, True) is None
    await _settle()


@pytest.mark.asyncio
async def test_completed_exact_child_needs_no_plugin_specific_evidence(
    tmp_path: Path,
) -> None:
    manager = _Manager()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )
    rollout.turn_terminal(
        "turn-child",
        TurnStatus.COMPLETED,
        {
            "turnId": "turn-child",
            "_pluginRolloutOwnerTurnId": "turn-parent",
            "_pluginRolloutGenerationId": "gen-2",
            "_pluginRolloutSourceRevision": "rev-2",
        },
    )
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await _settle()

    assert manager.promoted == ["fitbit@github"]
    assert manager.discarded == []
    terminal_annotation = manager.annotations[-1][1]
    assert terminal_annotation["candidate_checked"] is True


@pytest.mark.asyncio
async def test_child_with_wrong_candidate_identity_is_rejected(tmp_path: Path) -> None:
    manager = _Manager()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )
    rollout.turn_terminal(
        "turn-child",
        TurnStatus.COMPLETED,
        {
            "_pluginRolloutOwnerTurnId": "turn-parent",
            "_pluginRolloutGenerationId": "wrong-generation",
            "_pluginRolloutSourceRevision": "rev-2",
        },
    )
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await _settle()

    assert manager.promoted == []
    assert manager.discarded == ["fitbit@github"]


@pytest.mark.asyncio
async def test_child_with_wrong_source_revision_is_rejected(tmp_path: Path) -> None:
    manager = _Manager()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )
    rollout.turn_terminal(
        "turn-child",
        TurnStatus.COMPLETED,
        {
            "_pluginRolloutOwnerTurnId": "turn-parent",
            "_pluginRolloutGenerationId": "gen-2",
            "_pluginRolloutSourceRevision": "wrong-revision",
        },
    )
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await _settle()

    assert manager.promoted == []
    assert manager.discarded == ["fitbit@github"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [TurnStatus.FAILED, TurnStatus.CANCELLED])
async def test_noncompleted_parent_discards_checked_candidate(
    tmp_path: Path,
    status: TurnStatus,
) -> None:
    manager = _Manager()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )
    rollout.turn_terminal(
        "turn-child",
        TurnStatus.COMPLETED,
        {
            "_pluginRolloutOwnerTurnId": "turn-parent",
            "_pluginRolloutGenerationId": "gen-2",
            "_pluginRolloutSourceRevision": "rev-2",
        },
    )
    rollout.turn_terminal("turn-parent", status, {})
    await _settle()

    assert manager.promoted == []
    assert manager.discarded == ["fitbit@github"]


@pytest.mark.asyncio
async def test_same_revision_install_creates_no_pending_rollout(tmp_path: Path):
    manager = _Manager()
    manager.staged_candidate = False

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    result, _ = await rollout.install(
        "turn-parent",
        source="repo",
        marketplace="github",
        ref_name="",
        sparse_paths=[],
    )

    assert result.staged_candidate is False
    assert rollout.mint_child_capability("turn-parent") == ""
    assert rollout.child_binding("fake", False) is None
    with pytest.raises(RuntimeError, match="没有尚未提交"):
        await rollout.revert("turn-parent")


@pytest.mark.asyncio
async def test_next_turn_waits_until_parent_rollout_is_resolved(tmp_path: Path):
    manager = _Manager()
    release = asyncio.Event()

    async def uninstall(plugin_id: str) -> dict[str, object]:
        await release.wait()
        return {"plugin_id": plugin_id}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.uninstall("turn-parent", "fitbit@github")
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    waiter = asyncio.create_task(rollout.wait_for_turn_boundary())
    await asyncio.sleep(0)

    assert not waiter.done()
    release.set()
    await waiter
    assert "已卸载" in rollout.consume_fact()


@pytest.mark.asyncio
async def test_shutdown_finishes_the_only_resolution_task(tmp_path: Path) -> None:
    manager = _Manager()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def uninstall(_plugin_id: str) -> dict[str, object]:
        entered.set()
        await release.wait()
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.uninstall("turn-parent", "fitbit@github")
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await entered.wait()
    shutdown = asyncio.create_task(rollout.shutdown())
    await asyncio.sleep(0)

    assert not shutdown.done()
    release.set()
    await shutdown

    assert "已卸载" in rollout.consume_fact()
    await rollout.wait_for_turn_boundary()


@pytest.mark.asyncio
async def test_revert_is_same_turn_only_and_uninstall_stays_reversible(
    tmp_path: Path,
):
    manager = _Manager()
    uninstalled: list[str] = []

    async def uninstall(plugin_id: str) -> dict[str, object]:
        uninstalled.append(plugin_id)
        return {}

    rollout = TurnPluginRollout(
        cast(Any, manager), workspace=tmp_path, uninstall=uninstall
    )
    await rollout.uninstall("turn-parent", "fitbit@github")

    with pytest.raises(RuntimeError, match="不能回滚上一 turn"):
        await rollout.revert("turn-other")
    result = await rollout.revert("turn-parent")
    rollout.turn_terminal("turn-parent", TurnStatus.COMPLETED, {})
    await _settle()

    assert result["reverted"] is True
    assert uninstalled == []
