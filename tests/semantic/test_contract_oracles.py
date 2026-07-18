from __future__ import annotations

import asyncio
import shutil

import pytest

from tests_scenarios.contracts.oracles import assert_call_finality
from tests_scenarios.contracts.oracles import (
    assert_atomic_generation_switch,
    assert_isolated_gate_paths,
    assert_paths_retained,
    assert_rows_unchanged,
)


def test_call_finality_accepts_state_visible_before_success_returns() -> None:
    state = {"version": 1}

    async def invoke() -> dict[str, str]:
        await asyncio.sleep(0)
        state["version"] = 2
        return {"status": "completed"}

    async def scenario() -> None:
        result = await assert_call_finality(
            invoke,
            lambda: dict(state),
            expected={"version": 2},
        )
        assert result == {"status": "completed"}

    asyncio.run(scenario())


def test_call_finality_rejects_background_refresh_mutant() -> None:
    state = {"version": 1}
    release_background = asyncio.Event()
    tasks: list[asyncio.Task[None]] = []

    async def refresh_later() -> None:
        await release_background.wait()
        state["version"] = 2

    async def invoke_mutant() -> dict[str, str]:
        tasks.append(asyncio.create_task(refresh_later()))
        return {"status": "accepted"}

    async def scenario() -> None:
        with pytest.raises(AssertionError, match="承诺状态不可见"):
            await assert_call_finality(
                invoke_mutant,
                lambda: dict(state),
                expected={"version": 2},
            )
        release_background.set()
        await asyncio.gather(*tasks)

    asyncio.run(scenario())


def test_plugin_publication_oracle_rejects_early_switch_mutant() -> None:
    observations = [
        ("before", "generation-1"),
        ("candidate_ready", "generation-2"),
        ("committed", "generation-2"),
    ]

    with pytest.raises(AssertionError, match="未原子发布"):
        assert_atomic_generation_switch(
            observations,
            previous_generation="generation-1",
            next_generation="generation-2",
        )


def test_memory_oracle_rejects_derived_state_overwrite_mutant() -> None:
    authoritative_messages = [("session:0", 0, "original fact")]
    derived_rewrite = [("session:0", 0, "LLM rewritten fact")]

    with pytest.raises(AssertionError, match="既有行发生删改"):
        assert_rows_unchanged(
            authoritative_messages,
            derived_rewrite,
            state_name="sessions.db/messages",
        )


def test_workspace_oracle_rejects_official_path_mutant(tmp_path) -> None:
    sandbox = tmp_path / "gate"
    sandbox.mkdir()

    with pytest.raises(AssertionError, match="逃逸 Gate sandbox"):
        assert_isolated_gate_paths(
            sandbox=sandbox,
            workspace=tmp_path.parent / ".akashic" / "workspace",
            plugin_home=sandbox / "plugin-home",
            config=sandbox / "config.toml",
        )


def test_plugin_data_oracle_rejects_uninstall_delete_mutant(tmp_path) -> None:
    data = tmp_path / "workspace" / "plugin-data" / "feed-github"
    data.mkdir(parents=True)
    state = data / "state.json"
    state.write_text('{"keep":true}\n', encoding="utf-8")

    shutil.rmtree(data)

    with pytest.raises(AssertionError, match="越权删除持久数据"):
        assert_paths_retained([data, state], operation="普通插件卸载")
