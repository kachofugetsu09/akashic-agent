from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
import logging
from typing import NoReturn, cast

from agent.lifecycle.phase import inspect_phase, topo_sort_modules
from proactive_v2.frame import ProactiveFrame

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProactiveLifecycleSpec:
    id: str
    modules: tuple[object, ...] = ()
    initial_slots: tuple[str, ...] = ()
    terminal_slots: tuple[str, ...] = ()


type _ModuleRunner = Callable[[ProactiveFrame], Awaitable[ProactiveFrame]]
type _LifecycleHook = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class _CompiledModule:
    module: object
    slot: str
    requires: tuple[str, ...]
    produces: tuple[str, ...]
    collects: tuple[str, ...]
    runner: _ModuleRunner
    starter: _LifecycleHook | None
    stopper: _LifecycleHook | None

    async def run(self, frame: ProactiveFrame) -> ProactiveFrame:
        return await self.runner(frame)


async def _await_cleanup(
    action: _LifecycleHook,
) -> tuple[BaseException | None, asyncio.CancelledError | None]:
    """在调用方取消时等待单个清理动作完成。"""

    # 1. 让清理任务脱离调用方取消，并保留动作自身的异常
    try:
        cleanup_task = asyncio.ensure_future(action())
    except BaseException as error:
        return error, None

    external_cancellation: asyncio.CancelledError | None = None
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError as cancellation_error:
            if not cleanup_task.done() and external_cancellation is None:
                external_cancellation = cancellation_error
        except BaseException:
            break

    # 2. 消费任务结果，确保 stopper 自取消也成为可聚合错误
    try:
        await cleanup_task
    except BaseException as error:
        return error, external_cancellation
    return None, external_cancellation


def _raise_lifecycle_errors(message: str, errors: list[BaseException]) -> NoReturn:
    """按收集顺序重新抛出生命周期错误。"""

    if len(errors) == 1:
        raise errors[0]
    raise BaseExceptionGroup(message, errors)


class CompiledProactiveLifecycle:
    def __init__(
        self,
        spec: ProactiveLifecycleSpec,
        modules: Sequence[_CompiledModule],
    ) -> None:
        self.spec = spec
        self._modules = modules

    async def start(self) -> None:
        """按拓扑顺序启动模块，并在失败后完成逆序回滚。"""

        started: list[_CompiledModule] = []
        try:
            for binding in self._modules:
                started.append(binding)
                if binding.starter is not None:
                    await binding.starter()
        except BaseException as start_error:
            rollback_errors: list[BaseException] = []
            rollback_cancellation: asyncio.CancelledError | None = None

            # 1. 当前失败模块也视为已取得资源，逆序完成全部回滚
            for binding in reversed(started):
                if binding.stopper is None:
                    continue
                error, cancelled = await _await_cleanup(binding.stopper)
                if cancelled is not None and rollback_cancellation is None:
                    rollback_cancellation = cancelled
                if error is None:
                    continue
                rollback_errors.append(error)
                logger.error(
                    "主动 Lifecycle 启动回滚失败: %s",
                    binding.slot,
                    exc_info=(type(error), error, error.__traceback__),
                )

            # 2. 原始启动错误在前，回滚错误按逆序追加
            errors = [start_error, *rollback_errors]
            if rollback_cancellation is not None:
                errors.append(rollback_cancellation)
            _raise_lifecycle_errors("主动 Lifecycle 启动失败及回滚失败", errors)

    async def stop(self) -> None:
        """按逆序完成所有模块清理，并聚合全部清理错误。"""

        errors: list[BaseException] = []
        external_cancellation: asyncio.CancelledError | None = None

        # 1. 清理每个模块，调用方取消不能截断后续动作
        for binding in reversed(self._modules):
            if binding.stopper is None:
                continue
            error, cancelled = await _await_cleanup(binding.stopper)
            if cancelled is not None and external_cancellation is None:
                external_cancellation = cancelled
            if error is None:
                continue
            errors.append(error)
            logger.error(
                "主动 Lifecycle 停止失败: %s",
                binding.slot,
                exc_info=(type(error), error, error.__traceback__),
            )

        # 2. 外部取消在所有清理完成后恢复，不能覆盖清理错误
        if external_cancellation is not None:
            errors.append(external_cancellation)
        if errors:
            _raise_lifecycle_errors("主动 Lifecycle 停止失败", errors)

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
        bindings: list[_CompiledModule] = []
        slots: set[str] = set()
        for module in modules:
            binding = self._compile_module(module)
            if binding.slot in slots:
                raise RuntimeError(f"主动 Lifecycle 模块 slot 重复: {binding.slot}")
            slots.add(binding.slot)
            bindings.append(binding)
        return bindings

    def _compile_module(self, module: object) -> _CompiledModule:
        """在动态模块进入生命周期前固定并校验其内部契约。"""

        # 1. 一次性读取并校验模块声明
        slot = getattr(module, "slot", None)
        if not isinstance(slot, str) or not slot:
            raise RuntimeError(f"主动 Lifecycle 模块缺少 slot: {type(module).__name__}")
        requires = self._compile_slot_names(module, "requires")
        produces = self._compile_slot_names(module, "produces")
        collects = self._compile_slot_names(module, "collects")

        # 2. 固定运行和生命周期 hook，边界后不再动态回退
        runner = self._read_callable(module, "run", required=True)
        starter = self._read_callable(module, "start", required=False)
        stopper = self._read_callable(module, "stop", required=False)
        return _CompiledModule(
            module=module,
            slot=slot,
            requires=tuple(dict.fromkeys(requires)),
            produces=produces,
            collects=collects,
            runner=cast(_ModuleRunner, runner),
            starter=cast(_LifecycleHook, starter) if starter is not None else None,
            stopper=cast(_LifecycleHook, stopper) if stopper is not None else None,
        )

    def _compile_slot_names(self, module: object, field: str) -> tuple[str, ...]:
        value = getattr(module, field, ())
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise RuntimeError(
                f"主动 Lifecycle 模块字段不可迭代: module={type(module).__name__} "
                f"field={field}"
            )
        names = tuple(cast(Iterable[object], value))
        if any(not isinstance(name, str) or not name for name in names):
            raise RuntimeError(
                f"主动 Lifecycle 模块字段必须是非空字符串: "
                f"module={type(module).__name__} field={field}"
            )
        return tuple(cast(str, name) for name in names)

    def _read_callable(
        self,
        module: object,
        field: str,
        *,
        required: bool,
    ) -> Callable[..., object] | None:
        value = getattr(module, field, None)
        if value is None and not required:
            return None
        if not callable(value):
            requirement = "缺少或不可调用" if required else "必须可调用"
            raise RuntimeError(
                f"主动 Lifecycle 模块 hook {requirement}: "
                f"module={type(module).__name__} field={field}"
            )
        return cast(Callable[..., object], value)

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
                replace(binding, requires=tuple(dict.fromkeys(requires)))
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
