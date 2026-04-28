from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, cast
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)

I = TypeVar("I")
O = TypeVar("O")
# TODO: 若 pyright 支持稳定，再把 F 收窄为 PhaseFrame[I, O]，避免 Phase 和 Frame 的 I/O 不匹配。
F = TypeVar("F", bound="PhaseFrame[Any, Any]")


def _empty_slots() -> dict[str, Any]:
    return {}


@dataclass
class PhaseFrame(Generic[I, O]):
    input: I
    slots: dict[str, Any] = field(default_factory=_empty_slots)
    output: O | None = None


class PhaseModule(Protocol[F]):
    """模块约定：可选 requires / produces 类属性由 Phase 启动校验读取。"""

    async def run(self, frame: F) -> F:
        ...


class Phase(Generic[I, O, F]):
    def __init__(self, modules: Sequence[PhaseModule[F]]) -> None:
        self._modules = list(modules)
        self._validate()

    async def run(self, input: I) -> O:
        frame = self._build_frame(input)
        for module in self._modules:
            frame = await module.run(frame)
        if frame.output is None:
            raise RuntimeError("Phase 模块链未产生 output")
        return frame.output

    def _build_frame(self, input: I) -> F:
        return cast(F, PhaseFrame[I, O](input=input))

    def _validate(self) -> None:
        provided: set[str] = set()
        for index, module in enumerate(self._modules):
            requires = tuple(getattr(module, "requires", ()))
            produces = tuple(getattr(module, "produces", ()))
            for slot in requires:
                if slot not in provided:
                    logger.warning(
                        "Phase slot 未闭合: module=%d name=%s requires=%s",
                        index,
                        module.__class__.__name__,
                        slot,
                    )
            provided.update(str(slot) for slot in produces)
