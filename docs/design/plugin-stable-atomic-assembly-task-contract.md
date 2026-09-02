# 插件 stable 原子组装任务合同（R2a）

- 状态：implementing / review
- 日期：2026-08-15
- 实现基线：`66fc16c666de14a13c93208d301684ec8e2c9217`
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[v3 loader](plugin-v3-loader-task-contract.md)

## 1. 目标与边界

首次 `load_all()` 不再按扫描顺序逐个发布 generation。Core 先在不可租用的批次中完成所有 stable 插件声明、完整 v3 `CompositionRoot`、legacy v2 prepare/activate 和 runtime catalog；全部成功后只安装一个完整 `RuntimeSnapshot`。

本 PR 是 R2a，只拥有 stable boot 原子组装。R2b 另行拥有“任何 candidate（包括 v2-only candidate）不得复用 stable Root”。本 PR 不改变 promotion、lease、外部 endpoint 切换或插件领域实现。

```text
discover stable plugins
          │
          ▼
┌────────────────────────────────────────────┐
│ unpublished boot batch                     │
│ declarations ─► full v3 Root ─► settle     │
│                    │                       │
│ legacy v2 prepare/activate (migration only)│
│                    │                       │
│ runtime catalogs + event handlers          │
└────────────────────┬───────────────────────┘
                     │ all ready
                     ▼
             install one snapshot
                     │
                     ▼
               stable admission
```

`semantic_delta: compatible`：成功启动后的插件集合、v2 lifecycle 行为和 v3 topology 不变；允许变化是中间 generation 不再对外成为 stable snapshot，缺失 required Service 时不再留下部分 active owner。

## 2. 原子性合同

- 批次声明阶段可以递增 generation sequence、产生 Gate 诊断并创建 batch-local scope/catalog/root。
- 在完整 Root ready 前，`current/latest`、active generation、stable catalog、Channel/Service endpoint 和正式 admission 不变。
- required Service 缺失、Root apply 失败、legacy prepare/activate 失败或 catalog 编译失败时，Root、scope、module tree、MCP/Skill/Job/Proactive catalog 和暂存 EventBus handler 逆序清理。
- 首次创建的正式 plugin-data 目录属于未发布 batch；失败或取消时在所有 task/terminate 停止后删除，既有目录和 `.kv.json` 按原始字节恢复。
- 单个 legacy 插件失败沿用现行“记录 Gate 后跳过”的启动语义；Core 先丢弃整个未发布批次，再排除该插件重建剩余批次，禁止复用已 prepare 的对象。
- 只有完整 batch 成功后才登记 `_loaded/_scopes/_active_generations` 并安装一个 snapshot。
- 初次启动没有可用插件时保持没有 current snapshot，不制造空 generation。

原子边界只覆盖 Core-owned admission、Scope、catalog、正式 plugin-data/KV 和 Core-managed endpoint。legacy v2 `activate()` 可以绕过这些 owner 直接写文件或发外部 I/O，Core 无法把真实世界效果伪装成已回滚；本 PR 保持该行为但不为它背书。这些直写路径必须在对应插件 v3 迁移时收进 Effect/Service，之后才能删除 v2 compatibility。

## 3. v2 删除标记

v2 不是新组合平面的长期成员。本 PR 中下列代码只承担迁移期行为等价，全部是最终 v2-removal Gate 的物理删除目标：

- `_activate_stable_batch()` 中 `Plugin.prepare/activate` 和 v2 task gate；
- `_commit_stable_kv()` / `_rollback_stable_kv()` 与 `PreparedPluginKVStore.rollback_commit()`；
- `_publish_stable_batch()` 中 `_register_tools()`、`_bind_tool_hooks()`、`_publish_contributions()` 与 staged legacy EventBus；
- `_legacy_publication_counts()` / `_restore_legacy_publication_counts()`；
- `_StablePluginFailed` 的 legacy skip-and-retry 分支；
- `MetadataKind.TOOL/TOOL_HOOK/LIFECYCLE` 与 phase/job/proactive legacy contribution 收集路径。

删除条件不是“无人记得它”，而是 canonical 插件矩阵中每个 consumer 已迁移到 v3 Service/Event/Fiber/Effect，exact-commit Gate 通过，代码扫描与公开回归证明没有 v2 module/class/metadata consumer。最终不保留 deprecated alias、适配器或双注册。

## 4. 验证与停止条件

- 混合 v2/v3 stable boot 只调用一次 snapshot install，快照同时包含三类 generation 和 ready topology。
- consumer 先于 provider 扫描仍由 required Service 语义 settle，不依赖扫描顺序。
- required Service 永不出现时 fail-loud；`current_snapshot`、retained snapshot、active plugins/generations 和 scopes 都保持启动前状态。
- legacy prepare、tool catalog 或注册失败时调用 terminate/Scope cleanup，失败插件不残留，剩余插件从全新对象重建。
- 连续取消不能截断批次清理；全部 Scope、Root effect、KV rollback 和 topology Skill catalog 完成后才向调用方恢复 `CancelledError`。
- targeted：`tests/test_plugin_composition_lifecycle.py`、`tests/test_plugin_hot_reload.py`。旧 loader/manager 细分测试已在 2026-09-02 测试预算清理中移除。
- cumulative：plugin hot reload/runtime control/snapshot/composition 全量相关测试、Basedpyright、`git diff --check`、公开 change Gate。
- 停止条件：正式 plugin-data 在失败批次中改变、Core-managed endpoint 提前开放、Root 被 snapshot lease 前释放、取消后残留 task/process/catalog、失败重试复用旧 instance。
- 回滚点：Git tag `backup/plugin-atomic-assembly-r2-before-20260815`。

## 5. 后续切片

R2b 创建 candidate-owned Root/runtime/data-root，禁止任何 candidate 共享 stable Root；R3a/R3b 再引入 immutable topology identity、composition revision、Health 与 Incident。两者均不得重新扩大 v2 API。
