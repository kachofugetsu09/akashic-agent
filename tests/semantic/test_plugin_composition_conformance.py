from __future__ import annotations

# pyright: reportPrivateUsage=false

from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from agent.plugin_composition import Context, EmitEventKey, FiberState
from agent.plugin_composition.context import CompositionRoot
from agent.plugin_composition.effect import Effect
from agent.plugins.manager import PluginManager
from agent.plugins.registry import plugin_registry
from tests.plugin_composition_conformance import (
    CompositionConformanceProbe,
    CompositionConformanceReceipt,
    ConformanceMismatch,
    LifecycleEvidence,
    NamespacePluginHarness,
    TurnEvidence,
    assert_conformance_equal,
)


class _Mutation(str, Enum):
    NONE = "none"
    MISSING_INJECT = "missing_inject"
    REVERSED_ORDER = "reversed_order"
    LEAKED_DISPOSER = "leaked_disposer"


@pytest.fixture(autouse=True)
def _clean_registry():
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()
    yield
    plugin_registry._handlers._handlers.clear()
    plugin_registry._classes.clear()
    plugin_registry._instances.clear()


@pytest.mark.asyncio
async def test_reference_fixture_covers_all_six_conformance_lanes(
    tmp_path: Path,
) -> None:
    receipt = await _run_fixture(tmp_path, _Mutation.NONE)

    assert receipt.identity.generation_id.startswith("plugins:")
    assert tuple(item.name for item in receipt.catalog.fibers) == (
        "a_first",
        "b_second",
        "z_provider",
    )
    assert tuple(item.dependencies for item in receipt.catalog.fibers) == (
        ("fixture.value",),
        ("fixture.value",),
        (),
    )
    assert receipt.catalog.services == (
        "core.agent_input",
        "core.plugin_assets",
        "core.skills",
        "core.timer",
        "core.tools",
        "fixture.value",
    )
    assert receipt.catalog.listeners == (
        "emit:fixture.order:a_first",
        "emit:fixture.order:b_second",
    )
    assert receipt.turn.output_by_phase == (
        ("loaded", ("first", "second")),
        ("reloaded", ("second", "first")),
        ("dependency_missing", ()),
        ("dependency_restored", ("first", "second")),
    )

    states = {item.phase: item for item in receipt.state.phases}
    assert states["loaded"].ready is True
    assert states["reloaded"].ready is True
    assert states["dependency_missing"].ready is False
    assert states["dependency_missing"].required_pending == (
        "a_first",
        "b_second",
    )
    assert states["dependency_restored"].ready is True
    assert states["disposed"].ready is False
    assert all(
        fiber.state == FiberState.DISPOSED for fiber in states["disposed"].fibers
    )

    disposed_effects = receipt.effects.phases[-1]
    assert disposed_effects.phase == "disposed"
    assert disposed_effects.effects == ()
    assert disposed_effects.writes == ()
    assert disposed_effects.external_effects == ()
    assert disposed_effects.residuals == ()
    assert receipt.lifecycle.events == (
        "load:ready",
        "cleanup",
        "load:ready",
        "cleanup",
        "load:ready",
        "cleanup",
    )
    assert_conformance_equal(receipt, receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_lane"),
    [
        pytest.param(
            _Mutation.MISSING_INJECT,
            "catalog",
            id="missing-inject",
        ),
        pytest.param(
            _Mutation.REVERSED_ORDER,
            "turn",
            id="reversed-order",
        ),
        pytest.param(
            _Mutation.LEAKED_DISPOSER,
            "effects",
            id="leaked-disposer",
        ),
    ],
)
async def test_conformance_oracle_kills_known_mutants(
    tmp_path: Path,
    mutation: _Mutation,
    expected_lane: str,
) -> None:
    expected = await _run_fixture(tmp_path / "reference", _Mutation.NONE)
    actual = await _run_fixture(tmp_path / "mutant", mutation)

    with pytest.raises(ConformanceMismatch) as caught:
        assert_conformance_equal(expected, actual)
    assert expected_lane in caught.value.lanes


async def _run_fixture(
    root_dir: Path,
    mutation: _Mutation,
) -> CompositionConformanceReceipt:
    """运行一次真实加载、依赖波动、重启、事件和释放轨迹。"""

    # 1. 生成同一组 namespace fixture，每次最多注入一个故意缺陷
    harness = NamespacePluginHarness(root_dir)
    harness.write_plugin("a_first", _first_plugin_source(mutation))
    harness.write_plugin("b_second", _second_plugin_source(mutation))
    harness.write_plugin("z_provider", _provider_plugin_source())
    manager = harness.manager()

    # 2. 经 PluginManager 加载，再观察稳定行为和一次 Fiber 重载
    await manager.load_all()
    snapshot = manager.current_snapshot
    if snapshot is None or snapshot.composition_root is None:
        raise RuntimeError("真实 namespace loader 没有发布 CompositionRoot")
    composition_root = snapshot.composition_root
    first = harness.module(manager, "a_first")
    second = harness.module(manager, "b_second")
    provider = harness.module(manager, "z_provider")
    probe = CompositionConformanceProbe(composition_root)
    turn_output: list[tuple[str, tuple[str, ...]]] = []

    probe.capture("loaded", catalog=True)
    _emit(composition_root, first, "loaded", turn_output)
    await cast(Context, first.ctx).fiber.restart()
    probe.capture("reloaded")
    _emit(composition_root, first, "reloaded", turn_output)

    # 3. 移除并恢复 provider，证明 inject 与 Effect 所有权
    await cast(Effect, provider.service_effect).aclose()
    probe.capture("dependency_missing")
    _emit(composition_root, first, "dependency_missing", turn_output)
    await cast(Context, provider.ctx).fiber.restart()
    probe.capture("dependency_restored")
    _emit(composition_root, first, "dependency_restored", turn_output)

    # 4. 释放 generation，把所有残留暴露到 effects 通道
    await manager.terminate_all()
    residuals = _residuals(manager, first, second, composition_root)
    probe.capture("disposed", residuals=residuals)
    return probe.finish(
        turn=TurnEvidence(tuple(turn_output)),
        lifecycle=LifecycleEvidence(tuple(cast(list[str], first.lifecycle))),
    )


def _emit(
    root: CompositionRoot,
    module: ModuleType,
    phase: str,
    output: list[tuple[str, tuple[str, ...]]],
) -> None:
    payload: list[str] = []
    root.context.emit(cast(EmitEventKey[list[str]], module.ORDER_EVENT), payload)
    output.append((phase, tuple(payload)))


def _residuals(
    manager: PluginManager,
    first: ModuleType,
    second: ModuleType,
    root: CompositionRoot,
) -> tuple[str, ...]:
    residuals: list[str] = []
    for name, module in (("a_first", first), ("b_second", second)):
        live_resources = cast(int, module.live_resources)
        if live_resources:
            residuals.append(f"{name}:live_resources={live_resources}")
    residuals.extend(
        f"cleanup:{item.resource}:{item.error}" for item in manager.cleanup_failures
    )
    residuals.extend(f"effect:{item}" for item in root.receipt().effects)
    residuals.extend(f"listener:{item}" for item in root.topology_view().listeners)
    return tuple(residuals)


def _first_plugin_source(mutation: _Mutation) -> str:
    inject = "()" if mutation == _Mutation.MISSING_INJECT else "(VALUE,)"
    output = "second" if mutation == _Mutation.REVERSED_ORDER else "first"
    cleanup = (
        "    await context.effect(lambda: None, label='resource:first')\n"
        if mutation == _Mutation.LEAKED_DISPOSER
        else "    await context.effect(lambda: cleanup, label='resource:first')\n"
    )
    return (
        "from agent.plugin_composition import EmitEventKey, ServiceKey\n"
        "api_version = 3\n"
        "name = 'a_first'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "ORDER_EVENT = EmitEventKey('fixture.order')\n"
        f"inject = {inject}\n"
        "ctx = None\n"
        "live_resources = 0\n"
        "lifecycle = []\n"
        "async def apply(context, config):\n"
        "    global ctx, live_resources\n"
        "    ctx = context\n"
        "    value = context.get(VALUE)\n"
        "    lifecycle.append(f'load:{value}')\n"
        "    live_resources += 1\n"
        f"    await context.on(ORDER_EVENT, lambda payload: payload.append('{output}'))\n"
        "    def cleanup():\n"
        "        global live_resources\n"
        "        live_resources -= 1\n"
        "        lifecycle.append('cleanup')\n" + cleanup
    )


def _second_plugin_source(mutation: _Mutation) -> str:
    output = "first" if mutation == _Mutation.REVERSED_ORDER else "second"
    return (
        "from agent.plugin_composition import EmitEventKey, ServiceKey\n"
        "api_version = 3\n"
        "name = 'b_second'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "ORDER_EVENT = EmitEventKey('fixture.order')\n"
        "inject = (VALUE,)\n"
        "live_resources = 0\n"
        "async def apply(ctx, config):\n"
        "    global live_resources\n"
        "    live_resources += 1\n"
        f"    await ctx.on(ORDER_EVENT, lambda payload: payload.append('{output}'))\n"
        "    def cleanup():\n"
        "        global live_resources\n"
        "        live_resources -= 1\n"
        "    await ctx.effect(lambda: cleanup, label='resource:second')\n"
    )


def _provider_plugin_source() -> str:
    return (
        "from agent.plugin_composition import ServiceKey\n"
        "api_version = 3\n"
        "name = 'z_provider'\n"
        "version = '1.0.0'\n"
        "VALUE = ServiceKey('fixture.value')\n"
        "ctx = None\n"
        "service_effect = None\n"
        "async def apply(context, config):\n"
        "    global ctx, service_effect\n"
        "    ctx = context\n"
        "    service_effect = await context.provide(VALUE, 'ready')\n"
    )
