"""ReAct 循环的公开结构合同。

`react.v1` 是「读日志推理并逐条提交」的公开名字。消费者需要声明依赖、传预览
回调、并区分「达到模型请求上限」这一明确终态，因此 key、预览别名与异常由合同层
拥有；`react` 函数实现留在 `plugins/react/`。
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey

if TYPE_CHECKING:
    from agent.plugin_composition.models import StreamCallback

# 别名用字符串前向引用，避免合同层在运行时依赖组合内核的模型实现模块。
Preview = Callable[[str], "AbstractContextManager[StreamCallback]"]


class StepLimit(RuntimeError):
    """本次程序达到明确的模型请求上限，保留日志供来源继续控制。"""


@runtime_checkable
class ReactPort(Protocol):
    """组合上下文、模型、内容与工具的一次推理循环。"""

    async def __call__(
        self,
        reader: object,
        writer: object,
        *,
        model: object,
        context: object,
        projection: object,
        materials: object,
        content: object,
        tools: object,
        max_output_tokens: int,
        max_steps: int,
        reduce: object = None,
        preview: Preview | None = None,
        terminal_tools: frozenset[str] = frozenset(),
    ) -> object:
        """推理并逐条提交，返回最终 Output 消息。"""
        ...


REACT = ServiceKey[ReactPort]("react.v1")
