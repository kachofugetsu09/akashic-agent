# 程序化调用的 session 级记忆排除

- 状态：implemented（2026-07-31，PR #273）
- 确认日期：2026-07-31
- 关联条款：MEM-009、CTRL-002
- 2026-08-08 对账：上下文压缩入口由 [0030](../decisions/0030-session-context-compaction-ledger.md)
  取代旧 Markdown 后台维护；session-local checkpoint 仍可生成，但 excluded session 不得
  prepare/commit Markdown、PENDING 或 `ConsolidationCommitted`。

## 1. 问题和用户意图

程序化调用（control 协议 / SDK）目前只支持 turn 级 `skip_post_memory`：调用方在每次 `turn/start` 的 metadata 里声明该 turn 不沉淀记忆。对"整个 session 都不是学习内容"的用法，调用方必须逐 turn 重复声明，遗漏一次就把内容送进记忆沉淀。

用户希望创建 thread 时声明整个 session 标记为不沉淀：akasha 和 default memory engine（post-response worker 与 markdown 记忆）对该 session 的所有 turn 都不作为记忆沉淀。消息正文仍照常写入 `sessions.db`，已有记忆的检索不受影响；标记只改变"是否沉淀新记忆"。

另外，系统自身的定时任务会创建 `scheduler:{job.id}` 这类 session（`agent/scheduler.py` 经 `process_direct` 执行）。这类 session 是运行机制产物，不是用户对话内容，同样不应沉淀记忆。定时任务 session 作为内置排除集合，与显式标记统一生效，不再依赖调用方每次逐 turn 传递标记。

## 2. 当前调用链、状态 owner 与真实缺口

turn 级 `skip_post_memory` 现状：

```text
调用方 turn/start metadata {"skip_post_memory": true}
  → msg.metadata
  → after_turn._BuildTurnWorkModule 只读 msg.metadata
      → TurnCommitted.extra["skip_post_memory"]=true
          ├─ default_memory._on_turn_committed    不入队 TurnIngested
          ├─ akasha._on_turn_committed            不 stage
          └─ akasha builder._eligible_pairs       ✗ 当前无法观察 turn metadata，
                                                   完整重建时排除失效（缺口 B）
```

以上是 0026 前用于定位 MEM-009 的历史调用链。当前 Markdown 不再订阅
`TurnCommitted`；SessionCompactionRuntime 在提交 checkpoint 时应用相同排除谓词。

已核对代码后确认的五个缺口：

- **缺口 A：排除决策不能在 after_turn。** memory context guard 在 before_turn 执行：消息积累超过阈值时尝试 consolidation，失败则 abort 阻塞本轮。若决策点放在 after_turn，consolidate 返回 `skipped` 后 guard 判定"未释放上下文"并阻塞，excluded session 聊久了就无法继续。before_turn 已有现成逃生口 `skip_memory_context_guard`。
- **缺口 B：turn 级标记没有持久化证据。** 普通 turn 的 `skip_post_memory` 只进 `TurnCommitted.extra`，不投影到 `messages.extra`（`_PersistUserMessageModule` / `_PersistAssistantMessageModule` 不消费该 metadata）。akasha replay 只读 `messages.extra`，因此 turn 级排除在线生效、重建时丢失，MEM-009 声称的"明确标为 skip_post_memory 的 turn 不进入显式记忆图"在完整重建时不成立。
- **缺口 C：显式记忆写工具可绕过排除。** memorize / remember_memory / forget_memory 按 memory engine tool profile 注册，不受 `skip_post_memory` 控制。excluded session 内模型仍可主动调用写记忆工具。before_turn 阶段拿不到工具注册表，不能在这里枚举工具名；工具注册时也没有可查询的"memory 工具"来源身份。
- **缺口 D：metadata 缺少类型校验。** `thread/start` metadata 是任意字典，`bool("false")` 为 True，宽松判断会把非 boolean 值当作排除；`ControlService.start_thread` 在持久化前不做任何校验。
- **缺口 E：replay 排除不可观察。** akasha builder 只统计 `excluded_interrupted_turns`，普通排除直接 continue 不计数；新增 `sessions` 表依赖后，source schema 校验（当前只要求 `messages`、`message_embeddings`）也未覆盖。若直接 INNER JOIN，孤儿消息（无对应 session 行）会静默消失，看起来像被排除。

状态 owner：

- `sessions.db/sessions.metadata` 由 `SessionManager` / `SessionStore` 拥有，允许原位更新（状态地图 `sessions` 行）。
- `TurnCommitted.extra` 由 after_turn 模块链构建，消费方只读。
- akasha graph 是派生 sidecar，完整重建只读 `sessions.db` 与固定配置（MEM-009）。

## 3. 目标结构和边界

统一谓词是唯一权威规则：

```text
excludes_memory(session_key, session_metadata) =
    session_key 前缀 "scheduler:"（旧数据兼容）
    or session_metadata["skip_post_memory"] is True（严格 boolean，非 bool 值 fail-loud）
```

调用链：

```text
thread/start(metadata={"skip_post_memory": true})
  → ControlService.start_thread 先校验（非 boolean → 请求失败，不创建 session）
  → sessions.metadata（权威标记）
      │
      ▼
before_turn.acquire_session 之后（唯一决策点，只注入标记，不枚举工具名）
  ├─ msg.metadata["skip_post_memory"] = true          （赋值，不允许 turn 覆盖为 false）
  └─ msg.metadata["disable_memory_writes"] = true     （仅声明意图）
      │
      ▼
reasoner 计算 disabled_tools
  disable_memory_writes=true
  → 当前 runtime snapshot 中 source=memory 且 risk=write 的全部工具
  → disabled_tools 合并（跟随插件 generation，不携带旧工具名）
      │
      ▼
after_reasoning
  user + assistant 消息 extra 投影 skip_post_memory=true（持久证据）
      │
      ▼
after_turn 沿用现有 msg.metadata
      │
      ▼
default_memory / akasha 跳过；SessionCompactionRuntime 只写 ledger，跳过 Markdown 副作用
      │
      ▼
replay：JOIN sessions.metadata（先校验无孤儿消息），统一谓词 or 消息 extra 标记 → 排除并计数
```

边界：

- 排除资格创建时确定；excluded session 对普通 turn 是硬上界，turn 不能用 `skip_post_memory=false` 覆盖。
- 记忆写工具默认禁用：before_turn 只声明 `disable_memory_writes=true`，由 reasoner 基于当前 runtime snapshot 展开为 memory source 的 risk=write 工具名（记忆工具注册时携带 `source_type="builtin" / source_name="memory"` 来源身份）；`recall_memory`（检索）保留。是否读取已有记忆是与"是否学习本轮"正交的独立策略。
- 模型调用仍执行统一 context Gate；命中谓词时只推进 session-local ledger，不生成
  Markdown/PENDING/ConsolidationCommitted 副作用。
- 只支持创建时标记（`thread/start` metadata）；对已存在 session 补标记需要新增协议方法，不在最小范围。
- 标记写入后当前版本无撤销协议；取消标记需维护者另行确认。
- 标记不改变消息正文持久化，不删除已有记忆条目，不影响检索。

## 4. 状态变化和副作用

- 正常增加：`thread/start` 的 metadata 经现有机制合并写入 `sessions.metadata`（新键 `skip_post_memory`）；定时任务 session 不写标记，由 `scheduler:` 前缀内置规则排除；后续 turn 消息正文照常 INSERT，`messages.extra` 新增 `skip_post_memory` 投影。无迁移。
- 允许原位更新：`sessions.metadata` 行级更新协议不变；本次只新增一个键，不改变既有键语义。
- 逻辑失效：命中谓词的 session，其 turn 不再产生 `TurnIngested` 入队或 akasha stage；
  compaction checkpoint 不提交 Markdown 记忆副作用。已存在的记忆条目不因标记失效或减少。
- 物理减少：无。`sessions.db/messages` 正文不减少；akasha 派生图不包含这些 session，这正是 MEM-009 的"不进入显式记忆图"语义扩展到 session 粒度。
- 恢复证据：标记前后 `sessions.db` 备份；akasha 图可按固定输入确定性重建（MEM-009）。定时任务前缀规则是代码常量，随版本回滚即恢复。

## 5. 实施步骤

1. 新增 `session/memory_policy.py`：`excludes_memory(session_key, session_metadata)`（scheduler 前缀或 `skip_post_memory is True`）与 `validate_session_memory_metadata(metadata)`，对 `skip_post_memory` 做严格 boolean 校验，非法值 fail-loud。
2. `agent/control/service.py` `start_thread()`：在 `get_or_create` 之前调用 `validate_session_memory_metadata`，非 boolean 时请求失败、不创建 session。
3. `agent/lifecycle/phases/before_turn.py`：在 `_AcquireSessionModule` 之后注入排除策略，
   命中统一谓词时用赋值注入 `skip_post_memory=true`、`disable_memory_writes=true`；
   读取已有 `sessions.metadata` 时同样走严格校验。统一 context Gate 不绕过。
4. `agent/tools/meta/register.py` `_register_memory_tool()`：注册 memory profile 工具时携带 `source_type="builtin"`、`source_name="memory"` 来源身份；`agent/tools/registry.py` 新增按来源与 risk 查询工具名的窄方法（走 runtime snapshot 视图）。
5. `agent/core/passive_turn.py`：计算 `disabled_tools` 时，`msg.metadata["disable_memory_writes"]=true` 展开为当前 snapshot 中 memory source 的 risk=write 工具名并合并。
6. `agent/lifecycle/phases/after_reasoning.py`：`_PersistUserMessageModule` 与 `_PersistAssistantMessageModule` 在 `msg.metadata["skip_post_memory"] is True` 时把 `skip_post_memory=True` 写入 user 与 assistant 消息 dict，随 `_persist_session` 落入 `messages.extra`。这同时修复 turn 级标记的 replay 合同（缺口 B）。
7. `session/compaction_runtime.py`：命中统一谓词时仍提交 session-local checkpoint，
   但不调用 Markdown prepare/commit；receipt recovery 使用同一谓词。
8. `plugins/akasha/infrastructure/sparse_index/builder.py`：source schema 校验把 `sessions` 加入 required；先校验每条 `messages.session_key` 都有对应 `sessions` 行（孤儿消息带上下文 fail-loud，不静默消失），再 JOIN 读取 `metadata`；`_excluded_session` 改用统一谓词；build/audit 报告新增排除计数。
9. 新增/更新单元测试（见验收），运行 `docker/debug/gate.py run --base origin/main`。

不需要修改：`sessions.db` schema、memory2 / akasha schema、control 协议 schema、Python SDK 的 `thread_start`、现有在线 memory consumer。

## 6. 验收

1. `thread/start` metadata 持久化到 `sessions.metadata`，`thread/read` 可见；非 boolean 的 `skip_post_memory` 值在写入口 fail-loud、不创建 session。
2. 标记 session 的 turn 在 before_turn 注入两项策略；`TurnCommitted.extra` 带
   `skip_post_memory=true`；长会话仍经过统一 context Gate，但不沉淀 Markdown。
3. 标记 session 的 user 与 assistant 两条消息 `extra` 都带 `skip_post_memory=true`（缺口 B 回归）；从 `sessions.db` 单独导出消息后仍带排除语义。
4. 在线 default_memory / markdown / akasha 对 session 级事件与 turn 级事件行为一致，全部跳过。
5. 定时任务 session（`scheduler:{job.id}`）与显式标记同等待遇：在线不学习，
   context checkpoint 可推进但 Markdown/PENDING/event 写入为零。
6. akasha builder 完整重建：命中谓词的 session 产出 0 个学习 turn，排除计数可见；`sessions` 表缺失时 fail-loud，孤儿消息（无 session 行）带上下文 fail-loud；未命中 session 的图与改动前一致。
7. 记忆写工具在 excluded session 不可用（`disable_memory_writes` 展开为 memory source 的 risk=write 工具名），`recall_memory` 仍可用；插件 generation 更换后旧工具名不会残留。
8. 检索访问拆分验收：excluded session 的原始历史仍可通过 session history、`search_messages`、`fetch_messages` 访问；该 session 的内容不会通过 Memory2 / Akasha recall 被发现；其他已有长期记忆仍允许当前 turn 只读召回。
9. 普通 session 对照：仍正常学习。
10. 测试与 Gate 通过，报告与当前源码一致。
