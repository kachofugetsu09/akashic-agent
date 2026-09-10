"""重启闸门的组合内核归属。

`RESTART_GATE` 是 Core 拥有的能力 key。它按 design 从 `agent.restart`
归位到组合内核，使插件可以只依赖 `agent.plugin_composition` 与
`agent.plugin_contracts` 两个公开面声明依赖；`ServiceKey` 只按 name 相等，
因此归位不改变任何运行时语义。
"""

from agent.plugin_composition.model import ServiceKey
from agent.plugin_contracts.restart import RestartGate

RESTART_GATE = ServiceKey[RestartGate]("core.restart_gate.v1")
