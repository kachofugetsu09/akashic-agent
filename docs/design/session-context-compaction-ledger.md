# Session Context Compaction Ledger

- 状态：implemented
- 日期：2026-08-08
- 决策：[0030](../decisions/0030-session-context-compaction-ledger.md)
- 关联条款：CTX-001～CTX-007、SES-001～SES-005、MEM-002、MEM-004、MEM-008、MEM-011、MIG-001、WSP-003

## 1. 目标与 owner

Core 只在每次 session **业务** provider call 前调用可选的、来源无关的
`ProviderRequestProjection`。普通 `compaction` 插件拥有 `ContextCompactor`、20k tail
策略、辅助 summary 调用和 checkpoint receipt；Core 不识别插件 ID 或压缩配置。插件只
消费 assembler 已组成的 payload、Session owner 发出的不可变历史单元和 turn-scoped
可撤销 grant，不拥有删除消息或直接写 Markdown 的权限。SessionStore 独占 ledger
不变量；普通 `markdown_memory` 插件只消费仍 active 的 durable committed fact。

```text
SessionDB snapshot + system/memory/retrieval + dynamic tools
                              │
                              ▼
                 ┌──────────────────────────┐
                 │ optional request          │
                 │ projection Gate           │
                 │ full payload + budget     │
                 └─────────────┬────────────┘
                 < 74%          │ >= 74% / hard edge
                    │           ▼
                    │   ┌──────────────────────┐
                    │   │ session checkpoint   │
                    │   │ prepare → receipt v4 │
                    │   │ → ledger             │
                    │   └──────────────────────┘
                    │              │ committed
                    │              ▼
                    │      durable fact
                    │              │
                    │              ▼
                    │      Markdown plugin receipt
                    ▼
                 provider.chat
```

所有动态工具结果和外部效果仍留在完整 runtime/tool trace 与 SessionDB；摘要只是一份供
模型继续工作的派生投影。

## 2. 业务调用 Gate 与切点

每个 session business provider call 都先对已经装配的完整 payload 估算 token，不能只估消息
列表。输入包括：system prompt、MEMORY/SELF 等长期块、检索块、persistent
history、当前 prompt history、当前 attempt replay、多模态预算、动态 tool schema 和
协议开销。soft limit 固定为：

```text
soft_limit = floor(model.context_window * 0.74)
hard_input = model.context_window - request.max_output_tokens
```

`max_output_tokens = 0` 时不预留输出 token。`context_window`、`max_output_tokens` 和
其来源由当前 generation 的模型 capability 提供；旧 `memory_window`、
`effective_context_percent` 和 runtime compaction percent 不参与预算。

`context_window = 0` 表示当前 capability 未知，不表示没有上限。每次业务 provider call
仍必须完成 assembled payload 的结构校验与 token 估算，但此时跳过本地 soft/hard/force
compaction，直接调用 provider；若 provider 返回 `ContextLengthError`，保持原始错误语义，
不在该路径重试或转换。Markdown maintenance 不阻塞 Core 启动；真正执行时若窗口未知或小于
可用 input budget，必须明确失败为 `input_budget`，不得写入 ledger 或 receipt。

已提交 completed logical interaction（显式 `control_turn_id` 的全部 U 与最终 A）不可
拆分。当前 attempt 只能在完整闭合的 assistant tool-call 及其全部 result batch 后选择
临时切点；未闭合工具、当前 user anchor、外部效果和必要证据必须保留。raw tail 从后向
前累计至少 20,000 token，完整 logical unit 跨过目标时允许略大于 20,000。重建后的
payload 若仍超过 soft 或 hard 边界，返回可区分的 compaction failure，不切孤立消息。

工具 batch 完成后不直接假设下一次请求仍安全；下一次 provider call 重新走同一 Gate。
provider overflow 只允许一次强制 compaction/retry，第二次失败保留原始错误语义。

subagent 的主循环、incomplete summary、forced final summary 和 mandatory exit 都在调用前经过
一个内存态 Gate；compaction summary 本身使用独立硬输入边界且不递归 Gate。插件 jobs、
history route 和 vision 不经过本 Gate。

ledger 无 generation 时，selection 在最终业务 payload 形成前从最新消息向前按完整 logical
unit 选取约 74% 的近期窗口，同时给 20k raw tail 与 summary provider 硬边界留出空间。窗口外
更早消息不进入首次 payload、source plan 或摘要，但 SessionDB 保持完整；后续 generation
继续使用有效 cursor 到当前的增量。

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

## 4. Included Markdown 的版本化 crash saga

Included checkpoint 的跨文件阶段由 durable prepare fence 保护：

```text
1. SQLite INSERT session_compaction_prepares
   └─ incarnation / generation / parent / source seq+IDs / retained tail
2. consolidation_writes.db INSERT immutable session_compaction_receipt v4
   └─ actual runtime/model + canonical checkpoint/source plan + digests；无 Markdown draft
3. 同一 SessionDB 事务：INSERT session_compactions
   + update sessions.last_consolidated
   + DELETE matching prepare
4. ordinary Markdown plugin 消费仍 active 的 durable fact
   └─ 单次模型 projection 后，分别以独立 before-image/draft/applied receipt 发布 MEMORY/SELF
```

v4 receipt 是第一个跨文件 effect，ledger 是业务成功边界。重启恢复时：

- v4 receipt 和 prepare 都在且 source plan、digest、session incarnation 与当前 SessionDB
  一致：提交 ledger/cursor 并清除 prepare；随后 durable fact 可由 Markdown plugin 重放；
- 只有 prepare、没有 receipt：证明仍在 pre-effect window，私有 recovery 可以清除 orphan
  prepare，不产生 Markdown 或 ledger；
- v4 receipt 缺 prepare：完整校验 schema/digest/incarnation/source snapshot 后视为
  ledger 已提交的审计状态；active ledger 的 durable fact 仍可重放，任何损坏 fail-loud；
- v2 receipt 和 prepare 同时存在：先按 receipt 恢复 ledger，再把 legacy draft 作为
  durable fact 交给 Markdown plugin 的确定性 v2 路径；
- v3 receipt 仍按旧协议完成 ledger recovery，但属于已退役 PENDING/optimizer 管线，不发布
  给新 Markdown profile 插件，避免升级后重复解释历史 source plan；
- receipt schema/digest/source plan/incarnation 不一致：fail-loud，不能猜测格式或摘要。

Markdown plugin 不持有 compaction 后台队列。每个文档在自己的 SQLite receipt 下独立收敛；
失败不回滚 ledger，重启根据 before-image、draft 与 applied receipt 前向恢复。已被用户撤销
而逻辑失效的 generation 不得再次发布 durable fact。

prepare 存在时，message 撤销、interaction 删除、session cascade 和其他 destructive
mutation 必须在存储 owner 处阻断。Dashboard/管理入口返回 `409
session_compaction_pending`，同时返回稳定 audit identity；它们不得为了删除而绕过 fence。
只有正常 ledger 提交或确定性的恢复路径能清除 prepare。这样 crash 期间既不会删除
source rows，也不会把未完成的 Markdown side effect 伪装成 ledger 未提交。

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

Markdown consolidation 由 checkpoint owner 消费已提交 checkpoint 的 exact source plan。
每页按照 provider 的真实 capability 估算输入，
不拆 logical unit；单个完整 unit 无法满足 Markdown provider 硬预算时阻断，不删正文。
Markdown maintenance 是惰性执行，不得阻塞 Core 启动；只有真正执行时才按 provider capability
检查上述 unknown/过小窗口的 `input_budget` 失败。

```text
last_consolidated cursor ──► selected completed units ──► committed fact ──► profiles
          │                         │                         │
          └──── unchanged messages/tool_chain in SessionDB ◄─┘
```

included session 的 committed fact 才能触发 ordinary Markdown plugin 直接更新
`MEMORY.md` 与 `SELF.md`。命中 session memory exclusion 的 session 仍可推进自己的
`session_compactions`，但 ledger-only：不发布 Markdown committed fact。不存在 PENDING、
按消息数、TurnCommitted 后台刷新或“先更新 recent context 再压缩”的旁路。

## 7. 退役路径与边界

- `memory/RECENT_CONTEXT.md` 不再初始化、读取、注入或作为 proactive/Wake/Drift 输入；
  旧安装只由迁移 DAG 最后阶段 R06 带备份和完整性检查归档删除。
- 旧 `memory_window`、`keep_count`、`effective_context_percent`、手动
  consolidation/cursor API、`context_compact` 工具和 assistant `react_compaction`
  持久投影不再提供兼容写入口；旧 key 在配置边界 fail-loud 或由一次性迁移移除。
- 当前 `ContextCompactor` 只表示本合同的 session payload Gate；不再有 query-local
  internal compaction、内部 compact pair 或依赖 assistant metadata 的重放路径。
- proactive 使用自己的 VEDA/SELF/MEMORY、主动历史和 Akasha lane，不读取 session
  compaction summary；summary 不能伪装成用户事实。

## 8. 恢复、迁移与验收

Yoyo migration 在 workspace lock 下按
`L01 → U01 → P02 → D04 → X05 → T03 → R06` 分阶段切换：L01/U01/P02 只追加
SessionDB ledger、audit、prepare schema，D04 只把空的 legacy ledger 升级为带
`source_plan_digest` 的 final schema；X05 先只读预检 ledger/prepare 为空，再用 verified
backup 把旧 cursor 置零；T03 清理旧 trigger；R06 最后备份并校验 config 与遗留
`RECENT_CONTEXT.md`，移除 `memory_window`、旧 percent keys，归档并删除 RECENT，且不
再写 SessionDB。迁移阶段不调用 LLM。fresh install 直接创建新表且不生成 RECENT。代码
回滚使用对应 migration backup，不删除 ledger、prepare、receipt 或 messages。

验收至少观察以下边界：

1. assembled payload 在 74% 前后一 token、不同 model capability、输出预算（含 0）下
   触发一致；每个业务 provider call 和 tool batch 都经过 Gate。
2. 完整 logical unit、20k raw tail、source seq/IDs、incarnation 和 source-plan digest
   在 SQLite write set、重载和 crash recovery 中保持一致。
3. summary current-model → frozen-default fallback、无 tools/thinking 关闭、实际
   runtime/model/usage receipt 都可观察。
4. pending prepare 阻断 destructive mutation 并返回带 audit identity 的 409；orphan、v2/v3/v4
   恢复、v4 receipt-without-prepare、source drift 和损坏 JSON 均符合版本化矩阵。
5. generation 0 近期窗口、durable fact 重放/失效、included/excluded Markdown 分支、
   `last_consolidated` 推进和 messages/tool_chain/
   MEMORY/SELF 的非授权写集合分别核对。

## 9. 已知边界

以下边界为维护者已确认的可接受语义，不需要额外代码处理；新增改动不得在未先
核对本节的情况下改变它们。

1. **generation 0 窗口化只在首次 compact 前截断一次**：无任何 ledger generation 的
   session 每次组装时把 payload 截到最近 `floor(context_window * 0.74)` 完整逻辑单元
   （`window_initial_context_units`）；一旦第一次 compact 提交（generation 1），后续
   turn 走 cursor 增量，不再截断。窗口化不是 compact：不产生 summary、不推进 ledger、
   不触发 MEMORY 更新。
2. **窗口外早期历史永久退出模型视野**：首次 compact 的 source plan 只取窗口内内容；
   cursor 从窗口边界推进后，更早的 `sessions.db/messages` 不会进入任何后续摘要、ledger
   或 MEMORY 写入，仅作为只追加原始事实保留。存量安装的早期事实由旧架构时期
   consolidate 的 MEMORY.md 与迁移前 PENDING 承载；升级后新产生且落在窗口外的部分由用户接受
   为可遗忘。
3. **generation 0 且存在超窗 attempt replay 时 fail-loud**：极长单 logical interaction
   （本身超过 74% 窗口）携带 `_control_attempt_replay` 重进时，窗口化后的 replay 定位
   会错位，`run_turn` 以 `control attempt replay 未出现在完整 prompt history` 阻断。
   这是维护者接受的边界，不降级为猜测回填。
4. **Markdown plugin 采用 durable receipt 崩溃语义**：ledger 提交后 durable fact 可重发；
   MEMORY 与 SELF 各自以 draft/before-image/applied receipt 收敛。进程在任一 atomic replace
   前后退出，启动扫描都继续未完成文档；receipt 内容漂移 fail-loud。
5. **subagent 内存态 Gate 不持久化**：subagent 四个 provider 入口的 compact 投影只
   存在于内存，进程结束即丢失；`_SubagentContextGate` 从不写 session ledger。
