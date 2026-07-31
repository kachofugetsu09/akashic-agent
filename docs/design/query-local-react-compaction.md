# Query 内 ReAct Compaction 设计与任务合同

- 状态：accepted / implementation
- 日期：2026-07-31
- 决策：[0012](../decisions/0012-query-local-compaction-is-a-persisted-projection.md)
- 关联条款：CTX-001～CTX-007、SES-001、SES-005、CAP-001、ERR-001、TST-001～TST-004、TST-009

## 1. 目标和成功标准

一个 user query 可以在 `max_iterations = 0` 时执行长 ReAct。每次模型请求前，runtime 按完整 provider input 估算 token；达到模型 `context_window` 的默认 `74%` 后，在完整 tool batch 边界把旧步骤压成一对内部 compact call/result，使当前任务继续。回合完成后，完整执行事实和压缩投影共同进入 SessionDB；下一次 query 不重新展开已压缩前缀。

成功标准：

- 低于水位的普通 query 不改变 provider payload、工具行为和持久化结构。
- 长 query 可以触发一次或多次压缩，活动上下文始终只有一个 compact pair。
- 当前 user query、最近工具后缀、关键事实、决策、未完成工作和验证状态保留。
- SessionDB 仍保留完整 `tool_chain`，并在新 assistant row 中保存版本化压缩投影；旧消息 write set 不含 UPDATE/DELETE。
- Session 重载后模型只看到摘要、未压缩后缀、上轮最终回答和新 user query。
- 确定性 semantic tests、known-bad mutant、短任务回归和隔离 fault-injection 通过。

## 2. 任务合同

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - passive ReAct runtime
  - SessionDB prompt replay
runtime_patch: required
runtime_patch_reason: "只有 core 同时拥有模型预算、当前 query 工具边界和 Session prompt replay；插件或客户端实现会复制并猜测权威上下文语义。"
authoritative_state_owner: "SessionManager owns messages; DefaultReasoner owns current-query projection; LLMProvider owns request budgeting."
client_only_alternative: "not_applicable"
invariants:
  - CTX-001
  - CTX-003
  - CTX-004
  - CTX-007
  - SES-001
  - SES-005
protected_state:
  - existing sessions.db messages
  - complete persisted tool_chain
  - current user query
  - tool permission and hook semantics
  - max_output_tokens zero semantics
allowed_paths:
  - agent/model_runtime/query_compaction.py
  - agent/config.py
  - agent/config_models.py
  - agent/core/**
  - agent/lifecycle/**
  - agent/looping/ports.py
  - agent/model_runtime/**
  - agent/provider.py
  - bootstrap/tools.py
  - session/manager.py
  - session/store.py
  - tests/**
  - docs/**
  - config.example.toml
  - README.md
forbidden_paths:
  - frontend generated bundles
  - formal Akashic workspace
  - plugin cache
allowed_effects:
  - append react_compaction metadata with a newly committed assistant message
  - temporary provider request for compaction summary
forbidden_effects:
  - update or delete existing messages
  - register compact as a real tool
  - send external messages during compaction
  - alter benchmark task or verifier
validation:
  - deterministic threshold and message-pair tests
  - SessionDB reload and write-set tests
  - known-bad replay mutant
  - short-query no-op regression
  - isolated long-task fault injection
rollback: "reset consumers to backup/query-local-compaction-base-20260731; no data migration or destructive rollback is required because old runtimes ignore unknown assistant extra fields."
worktree_writer: "/mnt/data/coding/akasic-agent-worktrees/query-local-compaction"
handoff_head: ""
external_revisions: []
schema_lineages:
  - "messages.extra JSON object; no SQLite DDL change"
```

## 3. 当前调用链和 owner

```text
PromptRender
    │ initial_messages
    ▼
DefaultReasoner.run
    ├── BeforeStep
    ├── provider.chat
    ├── assistant tool calls
    ├── complete tool results
    └── AfterStep
            │
            ▼
      next provider.chat

ReasonerResult
    │ tool_chain + react_compaction
    ▼
AfterReasoning
    │ atomic user + assistant insert
    ▼
SessionDB assistant.extra
    │ Session.get_history
    ▼
next query prompt replay
```

`LLMProvider` 负责用 system prompt、消息、tool schema 和多模态块估算完整请求。`DefaultReasoner` 负责决定哪些已经闭合的当前 query 工具组进入摘要。`SessionManager` 只在完整 Turn 提交和后续重放时处理版本化字段。

## 4. 触发、切点和重复压缩

软水位：

```text
soft_limit = floor(model.context_window * compaction_trigger_percent)
default compaction_trigger_percent = 0.74
hard_limit = floor(model.context_window * effective_context_percent)
```

配置边界要求 `0 < compaction_trigger_percent < effective_context_percent <= 1`。`max_output_tokens` 不参与软水位计算。

每次 provider 调用前，优先使用上一响应的准确 input usage 加新增消息估算；没有完整 usage、tool schema 变化或压缩重建前缀时，对完整请求重新估算。达到软水位后，只能选择已经完整闭合的工具组前缀。当前 user query、当前 iteration 新注入提示和最近工具后缀不进入切点。

第一次压缩使用被淘汰的工具组生成摘要。再次压缩使用上一份摘要和新淘汰工具组生成新摘要，并替换旧 compact pair。摘要至少包含：

- Goal
- Constraints
- Progress
- Key facts and references
- Decisions
- Validation
- Unfinished work
- Next steps

## 5. 持久状态变化

| 对象 | 正常增加 | 允许原位更新 | 逻辑失效 | 物理减少 | Owner 与恢复证据 |
|---|---|---|---|---|---|
| 当前 prompt view | 每个工具组后增加临时消息 | compaction 可整体替换旧前缀投影 | 下一次 projection 取代上一份 | 回合结束即释放 | DefaultReasoner；完整 `tool_chain` 可证明事实仍在 |
| `messages` 新 assistant row | 完整 Turn 提交时 INSERT 一行 | 本功能不允许 | 后续摘要可以 supersede 模型视图，但不改变正文 | 仅用户显式删除会话或撤销 | SessionManager；SQLite row、seq 和完整 `tool_chain` |
| `assistant.extra.react_compaction` | 随新 assistant row 一次写入 | 本功能不允许 | 新 query 使用该投影；原字段仍是该 turn 的记录 | 随用户显式删除所属消息 | SessionManager；关闭重载后字段仍可解析 |
| 完整 `tool_chain` | 随新 assistant row 按既有规则写入 | 本功能不允许 | 不因 prompt 压缩失效 | 随用户显式删除所属消息 | SessionManager；数据库快照与工具 trace |

本功能不建立新表，不迁移旧行，不写正式 workspace。旧 runtime 会把 `react_compaction` 当作未知 message extra 保留，但仍按完整 `tool_chain` 重放；因此回滚代码不会损坏数据。

## 6. Case

### 短 query

完整请求始终低于 `74%`。不调用 summary，不生成 compact pair，不写 `react_compaction`。

### 第一次压缩

第 N 个工具批次结束，下一次请求估算达到水位。runtime 总结旧批次，保留当前 user query 和最近后缀，重新估算后继续调用模型。

### 重复压缩

新步骤再次达到水位。summary 输入包含上一份摘要和新淘汰步骤；活动模型输入只保留新的 compact pair。

### 下一次 user query

Session 重载读取上一 assistant row 的 `react_compaction`，跳过已压缩 `tool_chain` 前缀，投影摘要、后缀和最终回复，再追加新 user query。

## 7. Edge case 和失败语义

- 模型正在流式输出：不压缩，响应结束并形成完整工具批次后再判断。
- 并行工具部分完成：不压缩，全部 tool result 到齐后才形成候选边界。
- 初始 prompt 已达软水位但没有工具批次可淘汰：不伪造摘要；硬边界内继续原请求，provider 若报告 overflow 则保留原 `ContextLengthError`，让既有 history projection retry 接手。
- 只有一个必须保留的巨大最近批次：不得切成孤立消息；硬边界内继续，overflow 时保留原 provider 错误。
- 存在可压缩前缀，但重建后的完整 provider input 仍达到硬输入边界：返回 `compaction_insufficient`，不提交候选投影。
- summary provider 调用失败、取消、返回空白：活动 prompt 保持原值，不写半成品；错误向上暴露。
- provider 先报告 context overflow：最多执行一次强制 compaction 并重试同一请求；再次失败保留原错误。
- compaction summary 请求不携带业务工具，也不复用主 ReAct cache namespace。
- 被压缩前缀的 opaque `model_state` 不重放；压缩后的下一次真实响应建立新状态。
- tool result 在 summary 输入中按字符上限序列化，完整证据仍在 runtime `tool_chain` 和既有有界持久化结果中。
- SessionDB 中字段损坏、版本未知、切点超过 `tool_chain` 长度：在反序列化边界 fail-loud。

## 8. 验证

确定性测试覆盖 74% 前后一 token、完整批次、重复压缩、summary 失败、单个巨大后缀、Responses 投影和短 query no-op。Session 测试关闭并重新加载 SQLite，核对完整 `tool_chain`、压缩字段、下一次模型消息和旧消息 write set。

真实验证使用独立 Docker，并与生产保持相同的 `74%` 触发比例，不通过降低水位制造压缩。现有 V4 Flash 长任务尚不足以自然触发：`regex-chess` 旧峰值约为 370k/1M，`path-tracing-reverse` 约为 130k/1M；它们只验证默认配置下的 no-op 回归、容器隔离和完整任务结束。74% 边界、压缩调用和重放由可控 provider 的确定性测试覆盖；后续只有自然越过 74% 的真实任务才能作为 Docker 压缩验证。
