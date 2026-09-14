# 插件 v3 静态投影与 exact runtime 任务合同

- 状态：历史合同，已由 [0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md) 取代；不作为当前接入协议
- 日期：2026-08-15
- 关联条款：PLG-001～PLG-004、PLG-009、PLG-014、STA-001～STA-003
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[v3 包级 contribution](plugin-v3-package-contributions-task-contract.md)、[DashboardContext](plugin-v3-dashboard-context-task-contract.md)

## 1. 目标

当前实现已删除 `is_active`、`ServiceView`、`static_active` 和静态 UI/Skill 贡献。
所有注册都在 `apply(ctx)` 内，是否注册由普通分支决定；依赖只保留一份 `inject`。
下文保留旧设计缘由，不授权恢复这些入口。

让 v3 插件的静态启用状态、Skill catalog、Dashboard binding 与 composition listener
消费同一个 generation/Root 事实，不再从 original module 全局或可变
`generation.validation_workspace` 推断 candidate 状态。

```text
┌──────────────── Core frozen services ────────────────┐
│ ServiceView { MEMORY_RUNTIME, ...immutable info }    │
└──────────────────────┬───────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│ ComposablePlugin.is_active(ServiceView)              │
│ exact Fiber.static_active + Topology/Snapshot freeze │
└───────────────┬──────────────────────┬───────────────┘
                ▼                      ▼
       static Skill catalog     active generation
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                 plugin apply/Effect       DashboardContext
                 PluginRuntime.data_dir    same PluginRuntime
```

## 2. Change intent

```yaml
change_type: bugfix
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - composition plugin api v3
  - runtime snapshot compiler
  - plugin skill host
  - plugin dashboard host
runtime_patch: required
runtime_patch_reason: "Core owns candidate clones, formal rebuild and static catalogs; original module globals cannot represent an exact candidate Root."
authoritative_state_owner: "The exact composition Fiber owns evaluated static activity; RuntimeSnapshot freezes it for publication consumers."
client_only_alternative: "Not applicable; Skill and Dashboard publication happen in the server runtime."
```

## 3. 合同

- `ServiceView` 是 Core 冻结的 typed lookup，只放与 Root provider 同源的不可变描述值。
  v3 可选导出同步 `is_active(ServiceView) -> bool`；无导出默认 active，错误结果 fail-loud。
- Core 在任何 Skill catalog 或 Root apply 前绑定 static services。predicate 为 false 时不执行
  插件 `apply`，但 Fiber 本身正常 ready，并把 `static_active=false` 纳入 topology identity。
- `RuntimeSnapshot.composition_active_plugin_ids` 从 exact Root 冻结；Dashboard、Skill promotion、
  mobile/runtime inspection 等 publication consumer 不重新读取 original module 全局。
- inactive v3 generation 不进入 Skill roots、ignored roots 或重名检查；v2 的既有动态
  `is_active()` 行为保持不变，直到最终删除。
- v3 Dashboard 的 `workspace/data_root/validation` 从 snapshot exact Root 的
  `PluginRuntime` 计算。candidate Dashboard 与 listener 使用同一 attempt data root；formal
  rebuild 使用 generation 正式 data root。
- Dashboard closeable 消费 Root data，因此 candidate validation binding 必须先关闭，再
  dispose Root 和删除 attempt workspace；异常、取消、discard、promotion 均遵守此顺序。
- candidate/formal 的 `static_active` 不一致会改变 topology/snapshot identity并 fail-loud，
  不允许用相同 plugin ID 掩盖漂移。

## 4. v2 删除点

- `V2_REMOVAL(static-active)`：删除对 legacy instance `is_active()` 的动态回读，只保留
  snapshot 冻结集合。
- `V2_REMOVAL(dashboard-context)`：删除旧 Dashboard workspace 副本与三参数 ABI。
- v2 删除前不得让新 static filtering 改变其加载、错误传播或 Skill catalog 基线。

## 5. 验证

- inactive v3 与 active v3 声明同名 Skill 时，只 active owner 进入 catalog/link。
- `is_active=false` 且声明了缺失的 `inject` service 时，Fiber 不等待 runtime dependency、
  `_apply` 不执行且 Root 仍 ready；声明本身仍保留在 topology 诊断中。
- candidate clone 与 original module 不共享状态，仍可正确发布 active generation、Skill 和
  Dashboard；formal rebuild identity 相同。
- candidate listener 写入后，candidate Dashboard 从 exact same data root 立即可读；discard
  正式目录不变，promotion 后 formal binding 只读正式目录。
- Dashboard closeable 在 Root attempt 目录删除前执行；normal discard、installed promote、
  direct publish 和 snapshot drain 不留 binding、module、Effect 或 cleanup failure。
- v2 active error propagation、Dashboard hot reload、composition kernel、Manager、Skill、
  Pyright、compileall、`git diff --check` 与公开 Gate 通过。
