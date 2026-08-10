# 0012 · Query 内压缩是可持久重放的非破坏性投影（历史决定）

- 状态：superseded by [0030 · Session context compaction ledger 拥有模型窗口投影](0030-session-context-compaction-ledger.md)
- 日期：2026-07-31
- 关联条款：CTX-001～CTX-007、SES-001、SES-005、CAP-001、ERR-001

> 本记录保留 2026-07-31 的 query-local 讨论，不再定义当前 runtime。旧 query-local
> compactor、compact pair、`react_compaction` assistant metadata 和
> `context_compact` 路径均不是活动能力；当前合同、owner、预算和恢复语义以 0030 为准。

## 历史背景

长 ReAct query 会把每轮 tool-call/result 追加到临时 prompt，下一次 query 又从完整
`tool_chain` 重放前缀。只压缩函数内列表不能让后续 query 或 Markdown 维护任务知道一个
持久 cut point；因此当时提出用摘要加最近后缀构造可重放的 query projection。

## 历史决定

当时的方案曾选择：

1. 由 core model runtime 在当前 query 的 provider 调用前估算完整输入，默认 soft 74%；
2. 只在完整 tool-call/result batch 后切点，保留当前 user anchor、最近完整工具后缀和
   完整 `tool_chain` 证据；
3. 用关闭 thinking、无业务工具的 summary 请求生成 compact pair；
4. Turn 完成时只 INSERT 新 assistant 行，把 query projection 放进 `extra`，既有消息不
   UPDATE/DELETE。

这些不变量后来被重新归属到 session ledger。0030 改变了持久边界：

- 每个 session 的 `session_compactions` generation 与 `last_consolidated` cursor 成为
  checkpoint 真源；
- Gate 以每次完整业务 payload 和冻结 model capability 为单位，而非 query-local 列表；
- completed logical interaction 不可拆分，raw tail 反向累计至少 20,000 token；
- Included Markdown 只接受 cursor 到 cut point 的 exact source plan，并与
  `session_compaction_prepares`、immutable receipt、ledger INSERT 组成 crash saga；
- Excluded session 只推进 ledger，不写 Markdown/PENDING/event；
- `messages`、`tool_chain`、MEMORY/SELF/PENDING 和 Akasha 输入继续保持既有 owner 与只追加
  语义。

## 历史验收（不再作为当前 Gate）

旧方案曾验证 soft watermark 前后不生成 projection、完整工具批次切点、summary failure、
SessionDB reload 和旧消息 write set。当前验收必须改用 0030 的 full-payload Gate、frozen
generation、prepare/receipt crash recovery、pending destructive fence 与 source-plan
digest；不能用旧 assistant metadata 的存在证明当前实现正确。

## 回滚说明

本记录没有为当前运行时保留兼容 API。需要回退时，使用 0030 migration 前的 config、
SessionDB、memory 文件和 backup；不要删除新 ledger、prepare、immutable receipt 或既有
messages。
