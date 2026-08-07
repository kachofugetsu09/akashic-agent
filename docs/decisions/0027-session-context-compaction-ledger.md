# 0027 · Session context compaction ledger owns model-window projections

- 状态：accepted
- 日期：2026-08-07
- 取代：[0012 · Query 内压缩是可持久重放的非破坏性投影](0012-query-local-compaction-is-a-persisted-projection.md)
- 关联条款：CTX-001～CTX-007、SES-001～SES-005、MEM-002、MEM-004、MEM-008、MIG-001、WSP-003、TST-001～TST-006

## 背景

旧实现把 `memory_window`、Markdown consolidation 游标、全局
`RECENT_CONTEXT.md` 和 Query 内 `react_compaction` 混在同一条上下文路径中。
窗口条数不能表达不同模型的真实容量，`react_compaction` 还会让临时模型投影进入
assistant message metadata，导致重放继续依赖旧格式。Markdown 维护任务也会在
TurnCommitted 后按条数主动刷新全局近期摘要。

## 决定

1. Core 在每一次 session 的 `provider.chat` 前，使用已经组装好的完整 payload、动态
   tool schema、当前 runtime 的 `context_window` 和本次输出预算执行唯一检查点。
2. 软水位固定为 `floor(context_window * 0.74)`；硬输入边界为
   `context_window - max_output_tokens`。不再支持 `memory_window`、
   `effective_context_percent` 或 runtime 级 compaction percent。
3. `ContextCompactor` 是唯一压缩 owner。它既处理当前 attempt 的临时前缀，也处理
   committed session history 的可持久 checkpoint；压缩不会 UPDATE/DELETE messages。
4. summary 采用 Pi-mono 的六段格式：Goal、Constraints & Preferences、Progress
   （Done/In Progress/Blocked）、Key Decisions、Next Steps、Critical Context。
   工具结果、外部效果、路径、错误和 receipt 必须进入 Critical Context。
5. 每个 session 在 `sessions.db/session_compactions` 维护不可变 generation、parent
   lineage、source_ref/provenance、retained raw tail、模型/容量/usage、失效字段和
   checkpoint summary。`sessions.last_consolidated` 表示当前有效 generation，插入
   checkpoint 与推进 cursor 必须在同一事务中完成。
6. Markdown consolidation 只处理从上一个 generation 到新 cut point 的历史，不再按
   消息数或后台 TurnCommitted 刷新，也不再生成 `RECENT_CONTEXT.md`。PENDING 与
   `ConsolidationCommitted` 通过 source_ref 幂等提交，全部成功后才推进 cursor。
7. 已提交的 completed logical interaction 是不可拆分的持久压缩单元；当前 attempt
   只有已闭合 tool-call/result batch 可以作为临时压缩单元。recent raw tail 反向累计
   至少 20,000 token，跨过阈值的完整逻辑单元可以使尾部略大于 20,000；无法合法切点
   时明确阻断。
8. summary 优先使用当前模型，失败后使用 configured main/default fallback；两者失败
   则阻断业务调用。summary 请求不携带工具并关闭 thinking。
9. 删除 interaction 时，命中 source 或 retained dependency 的 generation 及全部
   descendants 逻辑失效，cursor 回退到最近有效 ancestor；session 删除 cascade
   ledger。旧 `react_compaction` 字节保留但不再读取或生成。
10. `RECENT_CONTEXT.md`、proactive 的近期摘要注入、手动 consolidation/cursor API
    和旧 QueryCompactor/context_compact 路径全部退役。Akasha、MEMORY、SELF、PENDING
    与既有完整 tool_chain 继续由原 owner 管理。

## 理由

模型窗口是 runtime 计算边界，不是消息保留策略。把 token 预算、session provenance
和 Markdown 记忆提交放入 core-owned ledger，可以同时保持完整原始事实、跨模型的容量
正确性和崩溃可恢复性；proactive 继续使用自己的 VEDA/SELF/MEMORY、主动历史与
Akasha lane，不会把模型临时摘要伪装成用户事实。

## 影响与回滚

- 这是 breaking context/persistence migration。Yoyo migration 备份并校验 config、
  `sessions.db` 和旧 `RECENT_CONTEXT.md`，把旧 cursor 重置为 0，并创建 ledger 表。
- 现有 session 采用 lazy rebuild；迁移不调用 LLM。第一次实际模型请求达到水位时生成
  generation 1。
- 回滚使用 migration 前的 config、SessionDB 和 memory 文件备份；代码回滚不得删除
  ledger 或 messages。

## 验收

- 完整 payload 在 74% 前后一 token、不同 context window 和输出预算下触发一致。
- tool-call/result 原子执行，下一次 provider 调用再次经过同一 Gate。
- ledger lineage、source_ref、retained tail、usage、cursor 同事务和删除失效可从
  SQLite write set、重载和 crash recovery 观察。
- messages、tool_chain、Akasha 固定输入、MEMORY/SELF/PENDING 字节没有非授权变化。
- RECENT、manual API、react_compaction 读写和旧配置 key 均被移除或 fail-loud。
