# 0155 · 组合内核只保留有消费者的派发方式

- 状态：accepted / implemented for review
- 日期：2026-11-14
- 修订：[0036](0036-plugin-composition-keeps-promotion-owner.md) 的事件与同步并发条目、[0042](0042-plugin-diagnostics-preserve-domain-owners.md) 中对 `ObserveEventKey` 的引用

0036 为组合内核定下 `emit`、`serial`、`parallel` 三种事件派发和受限的 `ExecutorService`，后续任务又加了
`transform` 与 `observe`。到 2026-11-14：

- 整个仓库（含测试）只声明过 `emit` 和 `serial` 两种 key：三个 runtime lifecycle 信号和 `source.changed.v3`。
  `ParallelEventKey`、`TransformEventKey`、`ObserveEventKey` 没有任何声明、监听或派发。
- `ExecutorService` 从未被构造，也没有人 provide `EXECUTOR_SERVICE`。于是工作线程标记永远不会置位，
  `Context` 上 39 处 `reject_executor_context_access()` 永远不会拒绝任何调用。
- hua-home 上已安装的全部外置插件（fleet 当前集合）源码里也没有这些名字。

本决定删除这三种派发、`ExecutorService`、`SyncTask`、`EXECUTOR_SERVICE` 和那组守卫，`plugin_boundary.toml`
不再把 `executor` 登记为 Core 能力。保留 `emit`、`serial` 与 `Bail`：lifecycle 信号明确拒绝 `Bail`，
这条约束仍在执行。

**为什么现在删。** 没有消费者的派发方式也要随每次改动保持取消、排空、诊断和失败隔离语义正确，
`observe` 一种就占了 `events.py` 约 130 行；守卫让 `Context` 的每个入口多一行永远不成立的检查。
此前清理特意把它们当作“设计合同”保留，但合同没有任何一方在履行。

**为什么不留着备用。** 需要并发等待多个 listener 或做纯同步并行时，由提出需求的插件带着真实用例加回，
届时按用例决定取消与失败语义，而不是维护一份没人验证过的通用实现。

**什么前提改变后值得重议。** 出现第二个需要同一派发语义的真实消费者。
