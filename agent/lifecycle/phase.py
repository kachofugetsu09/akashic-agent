from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)

I = TypeVar("I")
O = TypeVar("O")
F = TypeVar("F", bound="PhaseFrame[Any, Any]")


def _empty_slots() -> dict[str, Any]:
    return {}


def collect_prefixed_slots(
    slots: Mapping[str, object],
    prefix: str,
    *,
    reserved: Collection[str] = (),
) -> dict[str, object]:
    values: dict[str, object] = {}
    reserved_fields = set(reserved)
    for key, value in slots.items():
        if not key.startswith(prefix):
            continue
        field_name = key.removeprefix(prefix)
        if not field_name or field_name in reserved_fields:
            continue
        values[field_name] = value
    return values


def append_string_exports(target: list[str], exports: Mapping[str, object]) -> None:
    for key, value in exports.items():
        if isinstance(value, str) and value.strip():
            target.append(value)
        elif isinstance(value, list):
            items = cast(list[object], value)
            for item in items:
                if isinstance(item, str) and item.strip():
                    target.append(item)
                elif item is not None:
                    logger.warning(
                        "忽略非字符串 slot export: key=%s type=%s",
                        key,
                        type(item).__name__,
                    )
        elif value is not None:
            logger.warning(
                "忽略非字符串 slot export: key=%s type=%s",
                key,
                type(value).__name__,
            )


@dataclass
class PhaseFrame(Generic[I, O]):
    input: I
    slots: dict[str, Any] = field(default_factory=_empty_slots)
    output: O | None = None


class PhaseModule(Protocol[F]):
    """模块约定：可选 requires / produces 类属性由 Phase 启动校验读取。"""

    async def run(self, frame: F) -> F:
        ...


class SlotModule(Protocol):
    slot: str


def topo_sort_modules(modules: Sequence[object]) -> list[object]:
    slot_map: dict[str, SlotModule] = {}
    slot_order: dict[str, int] = {}
    for index, module in enumerate(modules):
        slot = getattr(module, "slot", None)
        if not isinstance(slot, str) or not slot:
            raise RuntimeError(f"模块缺少 slot 声明: {type(module).__name__}")
        if slot in slot_map:
            raise RuntimeError(f"模块 slot 重复: {slot}")
        slot_map[slot] = cast(SlotModule, module)
        slot_order[slot] = index
    active_slots = _active_module_slots(slot_map)
    slot_map = {slot: module for slot, module in slot_map.items() if slot in active_slots}

    in_degree = {slot: 0 for slot in slot_map}
    dependents: dict[str, list[str]] = {slot: [] for slot in slot_map}
    for slot, module in slot_map.items():
        for req in _module_requires(module, slot_map):
            in_degree[slot] += 1
            dependents[req].append(slot)

    queue = [slot for slot, degree in in_degree.items() if degree == 0]
    sorted_modules: list[object] = []
    while queue:
        queue.sort(key=lambda item: (_is_builtin_slot(item), slot_order[item]))
        slot = queue.pop(0)
        sorted_modules.append(slot_map[slot])
        for dependent in dependents[slot]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(sorted_modules) != len(slot_map):
        unresolved = sorted(slot for slot, degree in in_degree.items() if degree > 0)
        raise RuntimeError(f"模块循环依赖: {', '.join(unresolved)}")
    return sorted_modules


def render_dependency_tree(modules: Sequence[object]) -> str:
    sorted_modules = cast(list[SlotModule], topo_sort_modules(modules))
    slot_map: dict[str, SlotModule] = {module.slot: module for module in sorted_modules}
    children: dict[str, list[str]] = {slot: [] for slot in slot_map}
    in_degree = {slot: 0 for slot in slot_map}

    for slot, module in slot_map.items():
        for req in _module_requires(module, slot_map):
            children[req].append(slot)
            in_degree[slot] += 1

    roots = [slot for slot, degree in in_degree.items() if degree == 0]
    lines: list[str] = []
    visited: set[str] = set()
    for index, slot in enumerate(roots):
        _render_tree_node(
            lines,
            slot,
            children,
            slot_map,
            visited,
            prefix="",
            is_last=index == len(roots) - 1,
        )
    return "\n".join(lines)


def inspect_phase(modules: Sequence[object]) -> str:
    sorted_modules = cast(list[SlotModule], topo_sort_modules(modules))
    chain = "\n".join(
        f"  {index:2d}. {module.slot}"
        for index, module in enumerate(sorted_modules)
    )
    tree = render_dependency_tree(sorted_modules)
    return f"执行顺序:\n{chain}\n\n依赖树:\n{tree}"


def _module_requires(
    module: object,
    known_slots: Mapping[str, object],
) -> tuple[str, ...]:
    requires = tuple(str(slot) for slot in getattr(module, "requires", ()))
    return tuple(slot for slot in requires if slot in known_slots)


def _active_module_slots(slot_map: Mapping[str, object]) -> set[str]:
    active = set(slot_map)
    while True:
        disabled = set[str]()
        for slot, module in slot_map.items():
            if slot not in active or _is_builtin_slot(slot):
                continue
            missing = _missing_module_requires(module, active)
            if missing:
                logger.warning(
                    "Phase 模块依赖不存在，已禁用模块: module=%s requires=%s",
                    slot,
                    ", ".join(missing),
                )
                disabled.add(slot)
        if not disabled:
            return active
        active -= disabled


def _disable_modules_with_missing_module_dependencies(
    modules: Sequence[PhaseModule[F]],
) -> list[PhaseModule[F]]:
    module_slots = {
        str(slot)
        for slot in (getattr(module, "slot", None) for module in modules)
        if isinstance(slot, str) and slot
    }
    active = set(module_slots)
    disabled = set[str]()
    while True:
        current = set[str]()
        for module in modules:
            slot = getattr(module, "slot", None)
            if not isinstance(slot, str) or not slot:
                continue
            if slot not in active or _is_builtin_slot(slot):
                continue
            missing = _missing_module_requires(module, active)
            if missing:
                logger.warning(
                    "Phase 模块依赖不存在，已禁用模块: module=%s requires=%s",
                    slot,
                    ", ".join(missing),
                )
                current.add(slot)
        if not current:
            break
        disabled |= current
        active -= current
    return [
        module
        for module in modules
        if getattr(module, "slot", None) not in disabled
    ]


def _missing_module_requires(
    module: object,
    active_slots: set[str],
) -> tuple[str, ...]:
    return tuple(
        req
        for req in (str(slot) for slot in getattr(module, "requires", ()))
        if _is_module_slot(req) and req not in active_slots
    )


def _is_module_slot(slot: str) -> bool:
    return "." in slot and ":" not in slot


def _module_label(module: SlotModule) -> str:
    slot = module.slot
    tag = "[B]" if _is_builtin_slot(slot) else "[P]"
    return f"{tag} {slot}"


def _is_builtin_slot(slot: str) -> bool:
    return slot.startswith(
        (
            "before_turn.",
            "before_reasoning.",
            "prompt_render.",
            "before_step.",
            "after_step.",
            "after_reasoning.",
            "after_turn.",
        )
    )


def _render_tree_node(
    lines: list[str],
    slot: str,
    children: Mapping[str, list[str]],
    slot_map: Mapping[str, SlotModule],
    visited: set[str],
    *,
    prefix: str,
    is_last: bool,
) -> None:
    if slot in visited:
        return
    visited.add(slot)
    connector = "└── " if is_last else "├── "
    module = slot_map[slot]
    lines.append(f"{prefix}{connector}{_module_label(module)}")
    _append_tree_details(lines, module, prefix=prefix, is_last=is_last)
    child_prefix = prefix + ("    " if is_last else "│   ")
    child_slots = children.get(slot, [])
    for index, child in enumerate(child_slots):
        _render_tree_node(
            lines,
            child,
            children,
            slot_map,
            visited,
            prefix=child_prefix,
            is_last=index == len(child_slots) - 1,
        )


def _append_tree_details(
    lines: list[str],
    module: SlotModule,
    *,
    prefix: str,
    is_last: bool,
) -> None:
    indent = prefix + ("    " if is_last else "│   ")
    requires = tuple(str(slot) for slot in getattr(module, "requires", ()))
    produces = tuple(str(slot) for slot in getattr(module, "produces", ()))
    if requires:
        lines.append(f"{indent}← requires: {', '.join(requires)}")
    if produces:
        lines.append(f"{indent}→ produces: {', '.join(produces)}")


class Phase(Generic[I, O, F]):
    def __init__(
        self,
        modules: Sequence[PhaseModule[F]],
        *,
        frame_factory: Callable[[I], F],
    ) -> None:
        self._modules = _disable_modules_with_missing_module_dependencies(modules)
        self._frame_factory = frame_factory
        self._validate()

    async def run(self, input: I) -> O:
        frame = self._frame_factory(input)
        for module in self._modules:
            frame = await module.run(frame)
        if frame.output is None:
            raise RuntimeError("Phase 模块链未产生 output")
        return frame.output

    def _validate(self) -> None:
        module_slots = {
            str(slot)
            for slot in (getattr(module, "slot", None) for module in self._modules)
            if isinstance(slot, str) and slot
        }
        provided: set[str] = set()
        for index, module in enumerate(self._modules):
            requires = tuple(getattr(module, "requires", ()))
            produces = tuple(getattr(module, "produces", ()))
            for raw_slot in requires:
                slot = str(raw_slot)
                if slot in module_slots:
                    if slot != getattr(module, "slot", None) and slot not in provided:
                        logger.warning(
                            "Phase 模块依赖未满足: module=%d name=%s requires=%s",
                            index,
                            module.__class__.__name__,
                            slot,
                        )
                    continue
                if _is_module_slot(slot):
                    logger.warning(
                        "Phase 模块依赖不存在: module=%d name=%s requires=%s",
                        index,
                        module.__class__.__name__,
                        slot,
                    )
                    continue
                if slot not in provided:
                    logger.warning(
                        "Phase slot 未闭合: module=%d name=%s requires=%s",
                        index,
                        module.__class__.__name__,
                        slot,
                    )
            module_slot = getattr(module, "slot", None)
            if isinstance(module_slot, str) and module_slot:
                provided.add(module_slot)
            provided.update(str(slot) for slot in produces)
