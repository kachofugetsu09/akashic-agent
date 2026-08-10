# Query 内 ReAct Compaction（历史设计）

- 状态：superseded by [0030 · Session context compaction ledger](../decisions/0030-session-context-compaction-ledger.md)
- 日期：2026-07-31
- 决策：[0012](../decisions/0012-query-local-compaction-is-a-persisted-projection.md)

> 本文只保留旧 query-local 方案的决策背景和迁移线索。当前 runtime、配置、SessionDB
> 持久化和验收不得引用本文作为活动合同；旧 query-local compactor、compact pair、
> `react_compaction` assistant metadata 和 `context_compact` 入口均已退役。

## 1. 历史问题

长 ReAct query 会把 tool-call/result 前缀持续放入本轮临时列表；下一次 query 又从
`messages.tool_chain` 重放完整前缀。仅替换内存列表无法表达跨 query 的持久 cursor，也
无法让 Markdown consolidation 知道自己已经处理到哪里。旧设计因此尝试在当前 query
里生成一个可重放的 compact pair，并把它作为 assistant metadata 保存。

## 2. 被取代的边界

旧方案曾约定：

- 在单次 query 的 provider 调用前按模型窗口比例估算，默认 `74%`；
- 只在已闭合 tool-call/result batch 后切点，保留当前 user anchor 与最近工具后缀；
- 摘要请求关闭 thinking、无业务工具，失败时向上暴露；
- SessionDB 只追加新的 assistant 行，既有 messages 和完整 `tool_chain` 不改写。

这些局部不变量仍由 [0030](../decisions/0030-session-context-compaction-ledger.md) 保留，
但 owner 已从“单 query 临时 projection”改为“每个 session 的 SQLite ledger +
last_consolidated cursor”。当前 Gate 以每次完整业务 payload（含动态 tools、system、
memory、检索和多模态开销）为单位，不再把 compact pair 作为持久消息格式。

## 3. 迁移后的对应关系

| 旧 query-local 概念 | 当前 0030 语义 |
|---|---|
| 当前 query 内临时 compact pair | session payload Gate 的临时 summary projection |
| assistant `react_compaction` metadata | `session_compactions` generation 与 retained tail |
| 按 query 重复压缩 | 由 session `last_consolidated`/generation lineage 驱动 |
| 消息数或 query-local 阈值 | frozen model capability 的 `context_window`、固定 74% soft、request hard input |
| 只保护当前 query 的后缀 | completed logical interaction 不可拆分，raw tail 至少 20k token |
| query 结束后再决定记忆副作用 | included exact source plan 在同一 checkpoint saga 中提交 Markdown；excluded ledger-only |

当前流程是：

```text
full business payload + dynamic tools
                 │
                 ▼
        session Context Gate
                 │ over budget
                 ▼
 prepare fence → immutable receipt → Markdown (included only)
                 │
                 ▼
       ledger INSERT + last_consolidated
```

## 4. 历史回滚边界

旧设计没有新增表，曾允许旧 runtime 忽略未知 assistant extra；这不再是当前回滚合同。
代码回滚必须使用 0030 migration 前的 config、SessionDB、memory 文件备份，不得删除
`session_compactions`、`session_compaction_prepares`、immutable receipts 或既有 messages。

## 5. 仍需阅读的权威入口

- 当前产品条款：[CTX-007](../projectneed.md#ctx-007-session-compaction-ledger-按完整-payload-和真实模型容量触发)、[MEM-011](../projectneed.md#mem-011-历史投影按不可拆分逻辑单元和-token-tail-保留)。
- 当前决策：[0030](../decisions/0030-session-context-compaction-ledger.md)。
- 当前实现设计：[Session Context Compaction Ledger](session-context-compaction-ledger.md)。
- 旧 query-local 推理历史：[0012](../decisions/0012-query-local-compaction-is-a-persisted-projection.md)。
