from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent.tools.base import Tool
from core.memory.engine import MemoryToolSpec

if TYPE_CHECKING:
    from core.memory.engine import MemoryEngine


class ReinforceMemoryTool(Tool):
    """加强记忆：用户纠正/强调时,把当前情境的记忆绑得更牢,影响未来召回。

    纯加强、不删除、不改写其它记忆。落地极简——本工具被调用这件事本身,
    会被会话管线记进当前轮的 tool_chain(sessions.db,源头);Akasha 重建/线上
    都读 tool_chain 检测到本次调用 → 该轮建边走定向 gain_boost。所以 execute
    不需要写任何东西,返回确认即可。
    """

    name = "reinforce_memory"
    description = "由当前 memory engine 的 tool_profile 注入工具描述。"
    parameters = {
        "type": "object",
        "properties": {"note": {"type": "string"}},
        "required": [],
    }

    def __init__(
        self,
        memory: "MemoryEngine",
        spec: MemoryToolSpec,
    ) -> None:
        self._memory = memory
        self._spec = spec
        self.description = self._spec.description
        self.parameters = self._spec.parameters

    async def execute(
        self,
        note: str = "",
        memory_ref: str | None = None,
        **extra: Any,
    ) -> str:
        # 标记来自"本次调用被记入 tool_chain"这一事实,无需在此持久化。
        # note / memory_ref 一并进入 tool_chain 的 arguments,重放时也能取到。
        return "好，我会把这条记得更牢，下次遇到类似情境会更先想起来。"
