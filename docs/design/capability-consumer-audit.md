# Issue 1179：零静态消费者复核

以当前源码和 Fleet `cc1c812` 的 14 个 gitlink 为范围。目录是静态位置索引，不是
运行时依赖证明；`catalog` 现在也记录公开 key 传给局部 helper 的位置。

| key | 处理与真实消费者 |
|---|---|
| `akasha.learning.v1` | 私有耐久规则身份；Akasha `plugin.py` bind/open，恢复读取旧 learning binding |
| `akasha.recalls.v1` | 无消费者，删除服务与临时闭包；实际 RecallRecords 查询沿公开只读口 |
| `computer.control.v1` | 私有工具控制 binding；Computer freeze/open 后执行当前 scope |
| `drift.proposals.v2` | 无消费者，删除端口及变化通知；旧 proposals 仍由 DriftStore 读取和结算，不删数据 |
| `eventmail.*_source.v2` | Fleet Calendar/Feed/Fitbit/Steam 使用公开 bind 端口；主仓目录不扫描外部仓库 |
| `markdown-memory.writes.v1` | Fleet Observe 读取已提交写入历史 |
| `programmatic.v1` | Fleet Proactive Feedback/GitHub Watch 调用程序入口 |
| `subagent.program.v1` | 私有程序 binding；Tools 固定，Subagents 恢复时打开 |
| `wake.program.v1` | 私有程序 binding；Wake runtime 固定，source 恢复时打开 |
| `tools.display-name.v1` | ContentView 读取工具显示名，静态目录可见 |
| `shell.owners.v1` | 无消费者，删除服务；实际 ShellOwners 资源清理仍保留 |
| `standard_tools.skill_inspection.v1` | RuntimeInspection `_bind_optional` 装配可选读取子 Fiber，目录记录 helper 实参 |
| `message.display:*` | UI 按内容 kind 动态借用 Context/Models renderer |
| `message.result_display:content_view.read` | UI 按工具结果 kind 动态借用 ContentView renderer |
| `models.call-history.v1` | Fleet Observe 读取 provider attempt 历史 |
| `plugin.claim.embedding_memory` | 互斥角色声明；第二个 provider 在 provide 唯一性检查处冲突，不要求 reader |
| `executor` | 内核私有 worker 服务；由 Context 执行路径取得，不是插件业务能力 |

保留的公开合同在提供方 `contract.py` 标明消费路径；私有耐久 key 在所属实现声明处
标明 bind/open 路径，不为它们新增虚假的公共 SDK。删除端口的显式决定见
[0111](../decisions/0111-remove-unconsumed-business-ports.md)。

复现 `python scripts/plugin_boundary.py catalog`，结合
[第二 provider 场景](plugin-second-providers.md) 与外部源码 PR 检查；
静态零消费者既不证明死代码，也不证明可以拔除。
