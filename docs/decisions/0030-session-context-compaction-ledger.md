# 0030 · Session context compaction ledger owns model-window projections

- 状态：accepted / implemented
- 日期：2026-08-08
- 取代：[0012 · Query 内压缩是可持久重放的非破坏性投影](0012-query-local-compaction-is-a-persisted-projection.md)
- 关联条款：CTX-001～CTX-007、SES-001～SES-005、MEM-002、MEM-004、MEM-008、MEM-011、MIG-001、WSP-003、TST-001～TST-006

## 背景

旧实现把 `memory_window`、Markdown consolidation 游标、全局
`RECENT_CONTEXT.md` 和 query 内的临时 compaction 混在同一条上下文路径中。窗口条数
不能表达不同模型的真实容量，临时摘要还会进入 assistant metadata，导致重放继续依赖
旧格式；Markdown 维护任务也会在后台按消息数刷新，而不是从一个明确的持久 cursor
提交。

## 决定

1. Core 在每一次 session **业务** provider 调用前，对已经组装的完整 payload 做唯一
   Gate。估算包括 system、长期记忆、检索块、persistent/prompt history、当前 attempt、
   多模态预算、动态 tool schema 和协议开销；预算来自该业务执行冻结的 model
   generation。Gate 成功后才调用 provider。subagent 的四个 provider 入口使用同样预算与
   切点规则，但只维护进程内投影，不写 session ledger。插件 jobs、history route 和视觉
   短调用不进入该 Gate，超窗保持各自既有 provider/fail-open 错误语义。
2. 软水位固定为 `floor(context_window * 0.74)`。本次请求硬输入边界为
   `context_window - request_max_output_tokens`；输出预算为 `0` 时不额外预留。旧的
   `memory_window`、`effective_context_percent` 和 runtime compaction percent 不再拥有
   配置或运行时语义。
3. 每个 session 在 `sessions.db` 的 `session_compactions` 保存不可变 generation、
   parent lineage、exact source provenance、retained tail、摘要、模型容量和 usage；
   `sessions.last_consolidated` 只保存当前有效 generation。generation INSERT、cursor
   推进和 pending-prepare 清除在同一 SQLite 事务中完成。`sessions.db/messages` 和完整
   `tool_chain` 在正常运行中只追加，压缩不得 UPDATE/DELETE 既有正文。
4. 已提交的 completed logical interaction 是不可拆分的压缩单元；当前 attempt 只可在
   完整闭合的 tool-call/result batch 后选择临时切点。raw tail 从后向前累计至少
   20,000 token，跨过完整 logical unit 可以略大于目标；没有合法切点或重建 payload
   仍越过 soft/hard 边界时，业务调用明确阻断。每个 tool batch 完成后，下一次 provider
   调用再次经过同一 Gate。
   ledger 尚无 generation 时，首次投影先从最新历史向前按完整 logical unit 取约 74%
   窗口；更早历史不进入首次 provider payload、source plan 或摘要，但 SessionDB 原始消息
   保持完整。已有 generation 后只处理有效 cursor 到当前的新单元。
5. 摘要采用 Pi-mono 的六段格式：Goal、Constraints & Preferences、Progress（Done /
   In Progress / Blocked）、Key Decisions、Next Steps、Critical Context。摘要输入保留
   上一 generation 和已淘汰的完整证据；工具结果、路径、错误、外部效果、execution
   identity 和验证状态进入 Critical Context。当前 session 选中的模型先生成摘要；失败
   后使用同一冻结 generation 中 configured default 的模型作为 fallback。摘要调用不携带
   tools、关闭 thinking，并使用自己的硬边界；receipt 保存实际 runtime/model、容量、
   usage、source plan 和 digest。
6. Markdown consolidation 只消费新 checkpoint 的 exact source
   plan，按完整 logical unit 分页并从 provider 的真实 context capability 计算输入预算。
   included session 才能写 `PENDING.md`、history payload 和 `ConsolidationCommitted`；
   excluded session 只推进 session-local ledger，不产生 Markdown/PENDING/event。不存在
   按消息数、TurnCommitted 后台刷新或独立 recent context 的第二条路径。v3 ledger 提交后，
   Markdown draft 与 PENDING/history/event 由 Runtime 持有的 per-session 有序后台任务执行；
   失败不回滚 ledger、不自动重试，重启也不补跑。优雅关闭取消并等待任务取消收束。
7. Included checkpoint 使用版本化 crash saga：先在 `session_compaction_prepares` 写入
   session incarnation、generation、parent、source seq/message IDs 和 retained tail；再以
   `source_ref` 写不含 Markdown draft 的 immutable v3 receipt；最后在一个 SessionDB 事务
   中 INSERT ledger generation、推进 cursor 并清除 prepare，随后才安排 Markdown 后台任务。
   v3 receipt 与 prepare 同时存在时只确定性完成 ledger；v3 receipt 缺 prepare 表示 ledger
   已提交，不报错、不补跑。只有 prepare、没有 receipt 才可释放 orphan。升级前已存在的
   v2 receipt 与 prepare 仍按 draft 幂等完成旧 saga；schema、source plan、digest 或
   incarnation 不一致继续 fail-loud。
8. pending prepare 是 source rows 的破坏性操作围栏。message 撤销、session cascade、
   interaction 删除和其他 destructive mutation 在 fence 存在时不得执行，管理入口返回
   `409 session_compaction_pending` 并带 audit identity；正常提交或确定性的恢复路径才
   能清除 fence。用户删除 interaction 时，命中的 generation 及 descendants 逻辑失效，
   cursor 回退到最近有效 ancestor；session 删除才 cascade SessionDB ledger 和 prepare。
   `consolidation_writes.db` 中的 immutable receipt 作为恢复/审计证据保留，除非未来另有
   名称明确的独立数据管理协议。
9. `RECENT_CONTEXT.md`、proactive/Wake 的近期摘要注入、手动 consolidation/cursor API、
   assistant `react_compaction` 持久投影和旧 query-local/internal compactor 路径全部
   退役。Akasha、MEMORY、SELF、PENDING 以及完整原始 tool chain 仍由各自 owner 管理。

```text
assembled payload + frozen model generation
                     │
                     ▼
             ┌─────────────────┐
             │ business-call   │  每次 provider.chat 前
             │ Context Gate    │  full payload + tools
             └────────┬────────┘
                      │ >= 74% / hard edge
                      ▼
             ┌─────────────────┐
             │ session ledger  │  prepare → receipt v3
             │ checkpoint      │  → ledger INSERT + cursor
             └─────────────────┘
                      │ committed
                      ▼
             background Markdown task
```

## 理由

模型窗口是 runtime 的输入预算边界，不是消息保留策略。把 payload token 估算、session
provenance、Markdown source plan 和 crash recovery 放入 Core-owned ledger，可以在不同
模型容量下保持同一语义，同时保留完整原始事实。model registry 的
`context_window`/`max_output_tokens` capability source 是预算 owner；遗留 percent 列只
为 v1 schema identity 保留，不参与配置、能力解析或 Gate。

## 影响与回滚

- 这是 context/persistence migration，按锁内 DAG 分阶段切换：
  `L01(additive ledger) → U01(audit) → P02(prepare) → D04(source-plan digest)
  → X05(cursor activation) → T03(trigger cleanup) → R06(legacy retirement)`。
  L01/U01/P02/D04 只做已校验的 SessionDB 加法（D04 只重建空 legacy 表）；X05
  预检 ledger/prepare 为空后才备份并把旧 cursor 置零；T03 清理旧 trigger；R06 最后
  备份并校验 config/`RECENT_CONTEXT.md`，删除 legacy keys、归档并删除 RECENT，且不再
  写 SessionDB。迁移阶段不调用 LLM。
- 回滚使用 migration 前的 config、SessionDB、memory 文件和 migration backup；代码回滚
  不删除 `session_compactions`、prepare、receipt 或既有 messages。

## 验收

- 相同 assembled payload 在 74% 前后一 token、不同 model capability 和输出预算下触发
  一致；`max_output_tokens=0` 不预留输出空间。
- 每一次 session 业务 provider 调用、每个工具 batch 和四个 subagent provider 入口都经过
  对应 Gate；三个明确豁免 owner 不进入该 Gate。summary 调用没有 tools、
  thinking 已关闭，当前模型失败后 fallback 到冻结 default，并记录实际 runtime/model。
- SQLite write set 可观察 prepare、receipt、ledger INSERT、cursor 推进和 crash recovery；
  pending fence 会阻断 destructive mutation 并返回带 audit identity 的 409。
- generation 0 窗口化、source plan、logical-unit 边界、20k raw tail、included/excluded
  Markdown 分支、v2/v3 恢复矩阵和后台取消语义，以及
  session incarnation 可从重载、删除和恢复测试中核对；messages/tool_chain/长期记忆
  没有非授权 UPDATE/DELETE。
- `RECENT_CONTEXT.md`、`memory_window`、manual/query compaction API、`react_compaction`
  读写和旧配置 trigger 均移除或在边界 fail-loud。
