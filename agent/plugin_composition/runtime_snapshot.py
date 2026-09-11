"""运行时快照访问能力的公开结构合同。

`core.runtime_snapshot.v1` 是「读取当前 task 的 runtime snapshot / lease」的公开
名字。它此前是 `agent.plugins.snapshot` 的模块级函数，任何模块 `import` 即可读取
当前 task 的作用域 —— 那是隐式全局。按参考实现（DeepSeek Harness 把运行时作用域
放在**服务对象内部持有的 AsyncLocalStorage**、依赖一律经 Context 取得）的做法，
这里把访问面变成显式注入的窄服务：

- 服务对象仍然委托同一 `ContextVar`，因此 **task-scoped 语义与返回值完全不变**；
- 插件经 `ctx.require(RUNTIME_SNAPSHOT)` 取得；
- 拿不到 `ctx` 的调用点（dashboard、工具注册、模型注册表、react 菜单）由 Core 在
  装配时**显式穿参**，而不是 import 全局。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from agent.plugin_composition.model import ServiceKey

from agent.plugin_contracts.skills import SkillIndex

if TYPE_CHECKING:
    from agent.plugin_composition.overlay import CompositionSnapshotRoot


@runtime_checkable
class RuntimeSnapshotAccessPort(Protocol):
    """读取当前 task 绑定的 runtime snapshot / lease。

    两个方法都返回 `None` 表示「当前 task 没有可用的绑定」，与原先模块级函数的
    失败语义一致；类型用不透明对象避免合同层依赖快照实现。
    """

    def composition_root(self) -> CompositionSnapshotRoot | None:
        """当前 task 绑定快照的组合 Root；没有绑定或未发布时返回 None。"""
        ...

    def current_snapshot(self) -> object | None:
        """当前 task 绑定的 runtime snapshot；没有绑定时返回 None。"""
        ...

    def plugin_skill_index(self) -> SkillIndex | None:
        """当前 task 绑定快照的插件技能索引；没有绑定或索引时为 None。

        返回合同层的 `SkillIndex`（值词汇），因此消费者不需要依赖快照实现类型。
        """
        ...

    def lease(self) -> object | None:
        """当前 task 的 runtime snapshot lease 的 fork；没有绑定时返回 None。"""
        ...


RUNTIME_SNAPSHOT = ServiceKey[RuntimeSnapshotAccessPort]("core.runtime_snapshot.v1")
