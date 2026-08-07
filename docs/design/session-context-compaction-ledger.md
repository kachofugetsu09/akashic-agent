# Session Context Compaction Ledger

- 状态：implemented
- 日期：2026-08-08
- 决策：[0030](../decisions/0030-session-context-compaction-ledger.md)
- 关联条款：CTX-001～CTX-007、SES-001～SES-005、MEM-002、MEM-004、MEM-008、MEM-011、MIG-001、WSP-003

## 1. 目标与 owner

Core 在每次 session **业务** provider call 前拥有完整请求，因此由
`DefaultReasoner`/`ContextCompactor` 以冻结 model generation 执行 payload Gate。它只
消费 assembler 已组成的 payload、SessionDB 的只读历史单元和当前动态 tool schema；不
拥有删除消息或直接写 Markdown 的权限。`SessionCompactionRuntime` 协调 SessionStore
账本与 Markdown owner，Markdown owner 只处理被明确提交的 source plan，Akasha 只消费
completed transcript。

```text
SessionDB snapshot + system/memory/retrieval + dynamic tools
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ business-call Context    │
                 │ Gate                      │
                 │ full payload + budget     │
                 └─────────────┬────────────┘
                 < 74%          │ >= 74% / hard edge
                    │           ▼
                    │   ┌──────────────────────┐
                    │   │ session checkpoint   │
                    │   │ prepare → receipt →  │
                    │   │ Markdown → ledger    │
                    │   └──────────────────────┘
                    ▼
                 provider.chat
```

所有动态工具结果和外部效果仍留在完整 runtime/tool trace 与 SessionDB；摘要只是一份供
模型继续工作的派生投影。

## 2. 业务调用 Gate 与切点

每个 business provider call 都先对已经装配的完整 payload 估算 token，不能只估消息
列表。输入包括：system prompt、MEMORY/SELF/PENDING 等长期块、检索块、persistent
history、当前 prompt history、当前 attempt replay、多模态预算、动态 tool schema 和
协议开销。soft limit 固定为：

```text
soft_limit = floor(model.context_window * 0.74)
hard_input = model.context_window - request.max_output_tokens
```

`max_output_tokens = 0` 时不预留输出 token。`context_window`、`max_output_tokens` 和
其来源由当前 generation 的模型 capability 提供；旧 `memory_window`、
`effective_context_percent` 和 runtime compaction percent 不参与预算。

已提交 completed logical interaction（显式 `control_turn_id` 的全部 U 与最终 A）不可
拆分。当前 attempt 只能在完整闭合的 assistant tool-call 及其全部 result batch 后选择
临时切点；未闭合工具、当前 user anchor、外部效果和必要证据必须保留。raw tail 从后向
前累计至少 20,000 token，完整 logical unit 跨过目标时允许略大于 20,000。重建后的
payload 若仍超过 soft 或 hard 边界，返回可区分的 compaction failure，不切孤立消息。

工具 batch 完成后不直接假设下一次请求仍安全；下一次 provider call 重新走同一 Gate。
provider overflow 只允许一次强制 compaction/retry，第二次失败保留原始错误语义。

## 3. SessionDB ledger 与持久化

`sessions.db/session_compactions` 每次 checkpoint INSERT 一个不可变 generation，记录：

- `session_key`、`generation`、`parent_generation` 和 `source_ref`；
- `source_from_seq`、`consolidated_through_seq`、source message IDs 与 source-plan digest；
- summary format、summary、retained raw tail；
- 实际 summary runtime/model、`context_window`、soft/hard limit、before/after token、
  summary usage；
- `invalidated_at`/`invalidated_reason` 等逻辑失效字段。

`source_plan_digest` 是 `canonical_source_plan(selected_source_messages)` 的 SHA-256，
不是 selection/budget digest。Included 分支的 ledger 值必须与 immutable receipt 中的
digest 完全相等；excluded 分支不写 Markdown/receipt，但仍对同一 canonical plan 计算并
写入自己的 session ledger，reload 和幂等重放不得丢失。迁移遇到缺列的非空旧 ledger 时
不能猜测或写入空值，必须在可恢复备份后 fail-loud；只有可证明为空的预发布表可以升级到
带非空 64 位小写十六进制约束的最终 schema。

`sessions.last_consolidated` 只表示当前有效 generation，正常提交通过同一事务推进；
generation 不复用。`sessions.db/messages`、`tool_chain`、embedding 与 Akasha 输入在
压缩路径中只追加，不 UPDATE/DELETE 既有正文。

每个 session 的 source plan 使用其 `session_key + created_at` incarnation；重建或同名
session 不能复用旧 receipt/cursor。interaction 被用户显式撤销时，命中的 generation
及 descendants 逻辑失效，cursor 回到最近有效 ancestor；session 删除才可以按管理协议
cascade ledger。

## 4. Included Markdown 的 crash saga

Included checkpoint 的跨文件阶段由 durable prepare fence 保护：

```text
1. SQLite INSERT session_compaction_prepares
   └─ incarnation / generation / parent / source seq+IDs / retained tail
2. consolidation_writes.db INSERT immutable session_compaction_receipt
   └─ actual runtime/model + checkpoint + Markdown draft + digests
3. Markdown owner 幂等提交 PENDING/history/ConsolidationCommitted
4. 同一 SessionDB 事务：INSERT session_compactions
   + update sessions.last_consolidated
   + DELETE matching prepare
```

receipt 是第一个跨文件 effect。重启恢复时：

- receipt 和 prepare 都在且 source plan、digest、session incarnation 与当前 SessionDB
  一致：按 receipt 保存的 **included** 语义幂等重放 Markdown，再提交 ledger/cursor；
- 只有 prepare、没有 receipt：证明仍在 pre-effect window，私有 recovery 可以清除 orphan
  prepare，不产生 Markdown 或 ledger；
- receipt 缺 prepare、receipt schema/digest/source plan/incarnation 不一致：fail-loud，
  不能猜测或重新生成摘要。

prepare 存在时，message 撤销、interaction 删除、session cascade 和其他 destructive
mutation 必须在存储 owner 处阻断。Dashboard/管理入口返回 `409
session_compaction_pending`，同时返回稳定 audit identity；它们不得为了删除而绕过 fence。
只有正常 ledger 提交或确定性的恢复路径能清除 prepare。这样 crash 期间既不会删除
source rows，也不会让 Markdown/receipt 领先的 generation 丢失。

## 5. Summary 与模型 generation

摘要只允许 Pi-mono 六段标题：

```text
Goal
Constraints & Preferences
Progress: Done / In Progress / Blocked
Key Decisions
Next Steps
Critical Context
```

输入包含上一 generation summary 与本次 exact selected source units。Critical Context 必须
保留工具结果、路径、错误、外部 effect、execution ID、关键数值和验证证据；不得补写未
出现的事实。

当前 session 选中的模型先执行 summary；失败后使用同一个冻结 execution generation 的
configured default 模型。两者都失败、正文为空/格式错误或 summary input 不能落入其硬
边界时，业务调用明确阻断。summary provider request 使用 `tools=[]`、
`disable_thinking=True`，不递归进入 business-call Gate；它仍使用自身的硬输入边界并把
实际 runtime/model/usage 写进 checkpoint/receipt。

## 6. Markdown exact source plan

Markdown consolidation 由 checkpoint owner 在 `last_consolidated` 到新 cut point 之间
建立连续、可重放的 exact source plan。每页按照 provider 的真实 capability 估算输入，
不拆 logical unit；单个完整 unit 无法满足 Markdown provider 硬预算时阻断，不删正文。

```text
last_consolidated cursor ──► selected completed units ──► Markdown pages
          │                         │                         │
          └──── unchanged messages/tool_chain in SessionDB ◄─┘
```

included session 的 plan 才能写 `PENDING.md`、history entry payload 与
`ConsolidationCommitted`。命中 session memory exclusion 的 session 仍可推进自己的
`session_compactions`，但 ledger-only：不 prepare/receipt Markdown side effect、不写
PENDING、history 或 event。不存在按消息数、TurnCommitted 后台刷新或“先更新 recent
context 再压缩”的旁路。

## 7. 退役路径与边界

- `memory/RECENT_CONTEXT.md` 不再初始化、读取、注入或作为 proactive/Wake/Drift 输入；
  旧安装只由带备份和完整性检查的 migration 归档删除。
- 旧 `memory_window`、`keep_count`、`effective_context_percent`、手动
  consolidation/cursor API、`context_compact` 工具和 assistant `react_compaction`
  持久投影不再提供兼容写入口；旧 key 在配置边界 fail-loud 或由一次性迁移移除。
- 当前 `ContextCompactor` 只表示本合同的 session payload Gate；不再有 query-local
  internal compaction、内部 compact pair 或依赖 assistant metadata 的重放路径。
- proactive 使用自己的 VEDA/SELF/MEMORY、主动历史和 Akasha lane，不读取 session
  compaction summary；summary 不能伪装成用户事实。

## 8. 恢复、迁移与验收

Yoyo migration 在 workspace lock 下先备份并校验 config、SessionDB 和遗留
`RECENT_CONTEXT.md`，再创建 ledger/prepare schema、重置旧 cursor 并写入新 compaction
默认值；迁移阶段不调用 LLM。fresh install 直接创建新表且不生成 RECENT。代码回滚使用
对应 migration backup，不删除 ledger、prepare、receipt 或 messages。

验收至少观察以下边界：

1. assembled payload 在 74% 前后一 token、不同 model capability、输出预算（含 0）下
   触发一致；每个业务 provider call 和 tool batch 都经过 Gate。
2. 完整 logical unit、20k raw tail、source seq/IDs、incarnation 和 source-plan digest
   在 SQLite write set、重载和 crash recovery 中保持一致。
3. summary current-model → frozen-default fallback、无 tools/thinking 关闭、实际
   runtime/model/usage receipt 都可观察。
4. pending prepare 阻断 destructive mutation 并返回带 audit identity 的 409；orphan、
   receipt-without-prepare、source drift 和损坏 JSON 均保持可诊断失败。
5. included/excluded Markdown 分支、`last_consolidated` 推进和 messages/tool_chain/
   MEMORY/SELF/PENDING 的非授权写集合分别核对。
