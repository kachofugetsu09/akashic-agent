# 插件 TopologyView 任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 实现基线：`9eccccbfbf247ebc3f7e3ce6f1d7263fa3d7ddcf`
- 关联条款：PLG-001～PLG-014
- 关联决策：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)
- 上游设计：[Cordis 插件迁移能力等价验收](cordis-plugin-capability-parity.md)、[事件与同步执行能力](plugin-event-executor-task-contract.md)

## 1. 目标与边界

本 PR 只把当前组合 Root 冻结成内容寻址、不可变的 `TopologyView`，并让 `RuntimeSnapshot` 保存和校验这张 view。它不增加 lifecycle 接入点，不迁移插件，也不修改 `PluginManager` 的 legacy 贡献字段。

`semantic_delta: none`。旧 Plugin v2、七组 phase module、EventBus handler 和执行顺序保持逐项相同；不修改正式 workspace、plugin-data、manifest、SessionDB、memory、渠道或外部 API。

成功标准：

- `TopologyView` 包含 generation、Fiber 依赖、Service、Effect 和 listener 注册顺序；identity 是这些内容的 SHA-256，不使用持久 revision counter。
- snapshot 编译只接收 ready Root，并保存编译时 view；发布、验证封存和晋升继续拒绝 Root 拓扑漂移。
- snapshot lease 与 Root drain 行为不变。
- 组合内核、事件和 snapshot targeted tests 通过；旧插件与 lifecycle 不出现 diff。

后续 R3a 将结构 identity 收窄为 Fiber name/parent/required/dependencies、Service 与有序 typed listener，并用单调 `composition_revision` 记录同一 Root 内发生过的结构变化；generation、Fiber state 和普通 Effect 只保留在诊断视图，不参与 identity。

## 2. Owner 与后续 PR

Core publication plane 拥有 `TopologyView` 的冻结和校验。插件只产生 Root 内的 Service、Fiber、Effect 和 listener；它不能用 topology identity 宣布候选成功或自行晋升。

```text
plugin apply(ctx)
       │
       ▼
CompositionRoot ── freeze ──▶ TopologyView
                                  │
                                  ▼
                         RuntimeSnapshot
                    compile / validate / lease / drain
```

本 PR 合并后再依次提交：

1. lifecycle 领域接入点：只桥接 Prompt 与回答处理，不迁移现有插件。
2. Citation + Meme：成组迁移并做旧/新差分回放。
3. 删除这组已无消费者的 legacy slot 与贡献路径。

Job、MCP、Channel、Tool、UI 和 proactive 各自使用后续独立 Service PR，不进入本 PR。

## 3. 验证与回滚

- targeted：组合内核、类型事件和 RuntimeSnapshot。
- static：`compileall`、Basedpyright、`git diff --check`。
- Gate：`python docker/debug/gate.py run --base origin/main`。
- 停止条件：任何旧 phase 顺序、PluginManager 贡献、持久化 write set 或晋升语义发生变化。
- 回滚点：`/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-topology-lifecycle-seam-9eccccbf.bundle`。
