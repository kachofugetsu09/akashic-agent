from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from agent.plugin_composition import (
    CompositionReceipt,
    ExternalEffectObservation,
    FiberView,
    TopologyFiberView,
    WriteObservation,
)
from agent.plugin_composition.context import CompositionRoot
from agent.plugins.composable import ComposablePlugin
from agent.plugins.manager import PluginManager
from bus.event_bus import EventBus


@dataclass(frozen=True, slots=True)
class IdentityEvidence:
    generation_id: str
    topology_by_phase: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CatalogEvidence:
    fibers: tuple[TopologyFiberView, ...]
    services: tuple[str, ...]
    listeners: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TurnEvidence:
    output_by_phase: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class StatePhaseEvidence:
    phase: str
    ready: bool
    fibers: tuple[FiberView, ...]
    required_pending: tuple[str, ...]
    optional_pending: tuple[str, ...]
    errors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class StateEvidence:
    phases: tuple[StatePhaseEvidence, ...]


@dataclass(frozen=True, slots=True)
class EffectPhaseEvidence:
    phase: str
    effects: tuple[str, ...]
    writes: tuple[WriteObservation, ...]
    external_effects: tuple[ExternalEffectObservation, ...]
    residuals: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EffectsEvidence:
    phases: tuple[EffectPhaseEvidence, ...]


@dataclass(frozen=True, slots=True)
class LifecycleEvidence:
    events: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CompositionConformanceReceipt:
    identity: IdentityEvidence
    catalog: CatalogEvidence
    turn: TurnEvidence
    state: StateEvidence
    effects: EffectsEvidence
    lifecycle: LifecycleEvidence


@dataclass(frozen=True, slots=True)
class LaneMismatch:
    lane: str
    expected: object
    actual: object


class ConformanceMismatch(AssertionError):
    def __init__(self, mismatches: tuple[LaneMismatch, ...]) -> None:
        self.mismatches = mismatches
        self.lanes = tuple(item.lane for item in mismatches)
        details = "; ".join(
            f"{item.lane}: expected={item.expected!r}, actual={item.actual!r}"
            for item in mismatches
        )
        super().__init__(f"插件能力回执不等价: {details}")


class CompositionConformanceProbe:
    """把一个 Root 捕获为身份、目录、状态、效果和生命周期证据。"""

    def __init__(self, root: CompositionRoot) -> None:
        self._root = root
        self._topology_by_phase: list[tuple[str, str]] = []
        self._state_phases: list[StatePhaseEvidence] = []
        self._effect_phases: list[EffectPhaseEvidence] = []
        self._catalog: CatalogEvidence | None = None

    def capture(
        self,
        phase: str,
        *,
        catalog: bool = False,
        residuals: tuple[str, ...] = (),
    ) -> None:
        """在不改变 Root 的前提下捕获一个具名观察点。"""

        # 1. 在同一个边界冻结公开拓扑和验证回执
        topology = self._root.topology_view()
        receipt = self._root.receipt()
        self._topology_by_phase.append((phase, topology.identity))
        self._state_phases.append(_state_phase(phase, receipt))
        self._effect_phases.append(
            EffectPhaseEvidence(
                phase=phase,
                effects=receipt.effects,
                writes=receipt.writes,
                external_effects=receipt.external_effects,
                residuals=residuals,
            )
        )

        # 2. 显式选择目录阶段，避免瞬态阶段覆盖权威目录
        if catalog:
            if self._catalog is not None:
                raise RuntimeError("conformance catalog 只能捕获一次")
            self._catalog = CatalogEvidence(
                fibers=topology.fibers,
                services=topology.services,
                listeners=topology.listeners,
            )

    def finish(
        self,
        *,
        turn: TurnEvidence,
        lifecycle: LifecycleEvidence,
    ) -> CompositionConformanceReceipt:
        if self._catalog is None:
            raise RuntimeError("conformance receipt 缺少 catalog phase")
        return CompositionConformanceReceipt(
            identity=IdentityEvidence(
                generation_id=self._root.generation_id,
                topology_by_phase=tuple(self._topology_by_phase),
            ),
            catalog=self._catalog,
            turn=turn,
            state=StateEvidence(tuple(self._state_phases)),
            effects=EffectsEvidence(tuple(self._effect_phases)),
            lifecycle=lifecycle,
        )


class NamespacePluginHarness:
    """通过 PluginManager 的真实 namespace 路径加载源码 fixture。"""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.plugins_dir = root / "plugins"

    def write_plugin(self, name: str, source: str) -> Path:
        plugin_dir = self.plugins_dir / name
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "plugin.py").write_text(source, encoding="utf-8")
        return plugin_dir

    def manager(self) -> PluginManager:
        return PluginManager(
            plugin_dirs=[self.plugins_dir],
            event_bus=EventBus(),
            tool_registry=None,
            workspace=self.root / "workspace",
            installed_cache_root=self.root / "home" / "cache",
        )

    @staticmethod
    def module(manager: PluginManager, plugin_id: str) -> ModuleType:
        generation = manager.generation(plugin_id)
        if generation is None or not isinstance(
            generation.instance,
            ComposablePlugin,
        ):
            raise RuntimeError(f"缺少 v3 namespace 插件: {plugin_id}")
        return generation.instance.module


def assert_conformance_equal(
    expected: CompositionConformanceReceipt,
    actual: CompositionConformanceReceipt,
) -> None:
    """比较六条公开证据通道，并一次报告所有差异。"""

    mismatches = tuple(
        LaneMismatch(lane, getattr(expected, lane), getattr(actual, lane))
        for lane in (
            "identity",
            "catalog",
            "turn",
            "state",
            "effects",
            "lifecycle",
        )
        if getattr(expected, lane) != getattr(actual, lane)
    )
    if mismatches:
        raise ConformanceMismatch(mismatches)


def _state_phase(phase: str, receipt: CompositionReceipt) -> StatePhaseEvidence:
    return StatePhaseEvidence(
        phase=phase,
        ready=receipt.ready,
        fibers=receipt.fibers,
        required_pending=receipt.required_pending,
        optional_pending=receipt.optional_pending,
        errors=receipt.errors,
    )
