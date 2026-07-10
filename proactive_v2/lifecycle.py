from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import cast

from agent.lifecycle.phase import inspect_phase, topo_sort_modules
from proactive_v2.frame import ProactiveFrame


@dataclass(frozen=True)
class ProactiveLifecycleSpec:
    id: str
    modules: tuple[object, ...] = ()
    initial_slots: tuple[str, ...] = ()
    terminal_slots: tuple[str, ...] = ()


class _CompiledModule:
    def __init__(
        self,
        module: object,
        *,
        requires: tuple[str, ...],
    ) -> None:
        self.module = module
        self.slot = str(getattr(module, "slot"))
        self.requires = requires
        self.produces = tuple(str(value) for value in getattr(module, "produces", ()))
        self.collects = tuple(str(value) for value in getattr(module, "collects", ()))

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        runner = getattr(self.module, "run")
        return await runner(frame)


class CompiledProactiveLifecycle:
    def __init__(
        self,
        spec: ProactiveLifecycleSpec,
        modules: list[_CompiledModule],
    ) -> None:
        self.spec = spec
        self._modules = modules

    async def start(self) -> None:
        for binding in self._modules:
            starter = getattr(binding.module, "start", None)
            if starter is not None:
                await starter()

    async def stop(self) -> None:
        for binding in reversed(self._modules):
            stopper = getattr(binding.module, "stop", None)
            if stopper is not None:
                await stopper()

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        for binding in self._modules:
            frame = await binding.run(frame)
        return frame

    def inspect(self) -> str:
        return f"lifecycle={self.spec.id}\n{inspect_phase(self._modules)}"

    @property
    def modules(self) -> list[object]:
        return [binding.module for binding in self._modules]


class ProactiveLifecycleBuilder:
    def build(
        self,
        spec: ProactiveLifecycleSpec,
        contributions: Iterable[object] = (),
    ) -> CompiledProactiveLifecycle:
        modules = [*spec.modules, *contributions]
        bindings = self._bind_modules(modules)
        bindings = self._expand_dependencies(bindings, spec.initial_slots)
        self._validate_terminal_slots(spec, bindings)
        ordered = cast(list[_CompiledModule], topo_sort_modules(bindings))
        return CompiledProactiveLifecycle(spec, ordered)

    def _bind_modules(self, modules: list[object]) -> list[_CompiledModule]:
        _ = self._module_slots(modules)
        bindings: list[_CompiledModule] = []
        for module in modules:
            requires = [str(value) for value in getattr(module, "requires", ())]
            bindings.append(
                _CompiledModule(
                    module,
                    requires=tuple(dict.fromkeys(requires)),
                )
            )
        return bindings

    def _module_slots(self, modules: list[object]) -> set[str]:
        slots: set[str] = set()
        for module in modules:
            slot = getattr(module, "slot", None)
            if not isinstance(slot, str) or not slot:
                raise RuntimeError(f"主动 Lifecycle 模块缺少 slot: {type(module).__name__}")
            if slot in slots:
                raise RuntimeError(f"主动 Lifecycle 模块 slot 重复: {slot}")
            slots.add(slot)
        return slots

    def _expand_dependencies(
        self,
        bindings: list[_CompiledModule],
        initial_slots: tuple[str, ...],
    ) -> list[_CompiledModule]:
        module_slots = {binding.slot for binding in bindings}
        producers = self._data_producers(bindings)
        expanded: list[_CompiledModule] = []
        for binding in bindings:
            requires = list(binding.requires)
            for required in binding.requires:
                producer = producers.get(required)
                if required not in module_slots and producer is not None:
                    requires.append(producer.slot)
            for pattern in binding.collects:
                prefix = pattern.removesuffix("*")
                requires.extend(
                    producer.slot
                    for slot, producer in sorted(producers.items())
                    if slot.startswith(prefix) and producer.slot != binding.slot
                )
            expanded.append(
                _CompiledModule(
                    binding.module,
                    requires=tuple(dict.fromkeys(requires)),
                )
            )
        self._validate_required_data(expanded, producers, initial_slots, module_slots)
        return expanded

    def _data_producers(
        self,
        bindings: list[_CompiledModule],
    ) -> dict[str, _CompiledModule]:
        producers: dict[str, _CompiledModule] = {}
        for binding in bindings:
            for slot in binding.produces:
                if slot in producers:
                    raise RuntimeError(f"主动 Lifecycle 数据 slot 多 producer: {slot}")
                producers[slot] = binding
        return producers

    def _validate_required_data(
        self,
        bindings: list[_CompiledModule],
        producers: Mapping[str, _CompiledModule],
        initial_slots: tuple[str, ...],
        module_slots: set[str],
    ) -> None:
        available = {*initial_slots, *producers}
        for binding in bindings:
            missing = [
                required
                for required in binding.requires
                if required not in module_slots
                and ":" in required
                and required not in available
            ]
            if missing:
                raise RuntimeError(
                    f"主动 Lifecycle 数据依赖不存在: module={binding.slot} "
                    f"requires={', '.join(missing)}"
                )

    def _validate_terminal_slots(
        self,
        spec: ProactiveLifecycleSpec,
        bindings: list[_CompiledModule],
    ) -> None:
        produced = {slot for binding in bindings for slot in binding.produces}
        missing = set(spec.terminal_slots) - produced - set(spec.initial_slots)
        if missing:
            raise RuntimeError(
                f"主动 Lifecycle 终点 slot 无 producer: {', '.join(sorted(missing))}"
            )
