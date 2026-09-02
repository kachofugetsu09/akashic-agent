# Akasha 插件 v3 迁移任务合同

- 状态：implementation candidate
- 日期：2026-08-17
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014、MEM-009～MEM-010
- 上游：[Akasha 在线与重放](akasha-v2-runtime-migration.md)、[被动回复 seam](plugin-v3-passive-response-seams-task-contract.md)、[Mobile UI seam](plugin-v3-mobile-ui-query-task-contract.md)、[持久化状态地图](persistence-state-map.md)

## 1. 目标

把 `akasha` 的生命周期、Dashboard 与 Mobile UI 从 v2 `PluginContext` 迁到 exact Root：

```text
MemoryPlugin factory ── Core boot ──► AkashaMemoryEngine
                                          │
                       narrow formal port │ candidate port rejects access
                                          ▼
AfterReasoningCtx ◄── Akasha v3 Fiber ── Mobile query
       │                                  │
       ▼                                  ▼
pending user row                    Akasha sidecars
       │                                  │
       └──── SessionStore append ─────────┘ read-only Inspector
```

`MemoryPlugin` 是 Core memory engine bootstrap protocol，不是 v2 插件壳，继续保留。删除的是
`agent.plugins.Plugin` 子类、legacy phase module、`PluginContext.memory_engine`、v2 Dashboard 三参数
注册与 v2 Mobile contribution。

## 2. exact Memory Turn port

- `MEMORY_RUNTIME` 仍只公开 frozen engine name，供 `is_active(ServiceView)` 静态决定 Akasha 是否
  投入 Root；不得向其中加入 raw engine。
- `MEMORY_TURN_RUNTIME` 只在所选 engine 实现 `MemoryTurnRuntimeApi` 且 active v3 consumer 声明
  依赖时由 Core 提供。formal port 只公开 `take_user_metadata(turn_id)` 与
  `wait_active_recall(session_key, turn_id)`；不公开 ingest、mutate、admin、SQL、Session 或 engine。
- candidate Root 获得同 ServiceKey 的拒绝代理。candidate 调用任一方法立即 fail-loud，不能消费
  正式 staged feedback 或读取正式 active recall。candidate/formal topology 因而保持同构。
- port 返回的 user metadata 深复制并冻结顶层 mapping；active recall 使用 frozen、字段有界 DTO，不返回
  `MemoryRecord`、内部图、pending ticket 或 engine 引用。

## 3. 持久化与数据边界

- remember/forget 工具只在当前进程、当前 Turn 暂存 marker。只有 completed Turn 的
  AfterReasoning listener 消费 marker，并写入 `persist_user_metadata`；SessionStore 把 marker 与
  user/assistant rows 在同一事务 append。Turn 失败或进程在提交前崩溃时没有 completed Turn，暂存
  marker 不伪装成已持久化事实。
- append 失败或取消不产生部分 message、孤立 marker 或 cache/DB 分叉；成功后即使 caller 随后
  被取消，DB 与 cache 采用相同 rows。
- `sessions.db/messages` 仍是唯一权威正文。Akasha sidecar 是可确定性重建投影；Inspector 不持有
  Session repository，也不直接打开 `sessions.db`。它需要的 tool-chain 展示字段由在线提交和离线
  rebuild 以同一规则写入 sidecar。
- candidate 的 `data_root` 与声明的 `memory` root 都是隔离副本。Dashboard/Mobile query 只读该代
  分配路径；discard、失败或取消后正式 SessionDB、Akasha sidecar 与 plugin-data 摘要不变。

## 4. UI 与生命周期

- Akasha v3 admission surface 仅包含 `api_version = 3`、静态 metadata、`is_active` 与精确
  `apply(ctx, config)`；`apply` 注册 AfterReasoning typed listener 和一个 `MobileUiDefinition`。
- active synthetic assistant 只通过 exact port 读取同 session/turn 的 pending recall；persisted
  assistant、recent 与 detail 只读 sidecar。query 不返回完整 assistant text。
- Dashboard 使用 `register(app, DashboardContext)`，只从 `data_root` 和声明的 `memory` root 建立
  reader；inactive Akasha 不进入 Dashboard/Mobile registry。
- listener、Mobile binding 与 Dashboard binding 都由 generation/Fiber scope 持有；reload、discard、
  terminate 后不得残留 listener、query lease、module 或 reader binding。

## 5. 验证与恢复范围

- 进程内：listener/append/query 的异常与取消，candidate discard，formal reload，Root drain。
- Core 进程崩溃：重开 engine/Manager 后只从已提交 Session rows 与 sidecar 恢复；未提交 staged
  marker 不出现，已提交 Inspector/recall 等价。
- 不扩展到任意断电时点或停机 checkpoint；SQLite/现有 sidecar 发布协议继续拥有自己的 durability。
- 当前候选 Gate 验证 Mobile 与公共插件边界；正式发布流程在获授权的 workspace 副本上验证
  feedback、active/persisted recall、Dashboard、sidecar hash、SessionDB append-only 与 cleanup。
  本任务不写正式 workspace。

恢复点：`backup/akasha-plugin-v3-pre-20260817`。
