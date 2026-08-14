# 插件 Timer 组合 Service 任务合同

- 状态：accepted / direct-mode foundation
- 日期：2026-08-14
- 目标分支：`codex/plugin-event-catalog` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-timer-before-20260814@e42b107b`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件元能力底座与测试先行合同](plugin-meta-capability-task-contract.md)

> 2026-08-15 对账：本合同的受控时钟与 `ctx.spawn` 行为继续作为 direct-mode 单元基础；Manager Root 的生产调度、candidate 隔离与 callback 失败语义由[插件 Timer stable snapshot 调度合同](plugin-timer-snapshot-scheduling-task-contract.md)接管。

## Goal

把 Cordis Timer 的 `timeout`、`interval` 与 Fiber effect ownership 转译成 Python 组合能力，让插件声明“需要单调定时能力”，而不是声明旧 `JobService` 类别。Core 只提供调度和生命周期接入点；回调、领域状态、外部读取与后续 Agent Input 均由插件实现。

## Ownership

Timer Service 独占以下不变量：

- delay 使用秒和 event loop 单调时钟，不受系统墙钟回拨影响；
- interval 按固定 deadline 推进，慢回调期间错过的 tick 被合并，不补跑；
- 同一个 interval 的同步或异步回调串行执行，不允许自重叠；
- task 属于调用方 Fiber，手动关闭、reload 或 dispose 都取消并 join；
- callback 异常进入 Composition receipt，使候选不再 ready，而不是被吞掉。

GitHub Watcher 的轮询是已确认的后续消费者，但本 PR 只包含隔离实验插件，不修改其 canonical source。

```text
┌──────────────────┐  inject core.timer  ┌──────────────────┐
│ v3 plugin Fiber  │ ───────────────────▶ │ TimerService     │
│ owns callback    │                      │ owns cadence     │
└────────┬─────────┘                      └────────┬─────────┘
         │ ctx.spawn / Effect                      │ monotonic wait
         ▼                                         ▼
┌──────────────────┐                      ┌──────────────────┐
│ Fiber lifecycle  │ ◀── cancel + join ─ │ timeout/interval │
└────────┬─────────┘                      └────────┬─────────┘
         │ callback failure                        │ callback
         ▼                                         ▼
┌──────────────────┐                      ┌──────────────────┐
│ Core receipt     │                      │ plugin domain    │
│ readiness owner  │                      │ implementation   │
└──────────────────┘                      └──────────────────┘
```

## Public seam

- `TIMER_SERVICE = ServiceKey("core.timer")`
- `TimerService.timeout(ctx, callback, delay, *, name="timeout")`
- `TimerService.interval(ctx, callback, delay, *, name="interval")`
- 两个入口返回可等待关闭的 `TimerHandle`。

第一版不提供 `throttle`、`debounce`、Promise sleep 或 async iterator。它们在出现真实消费者和独立验收前不进入 Core。旧 `PluginJobSpec` 与 APScheduler host 本 PR 保持不变。

## Change intent

```yaml
change_type: additive
semantic_delta: none for existing plugins
capability_owner: core plugin composition
consumer_scope:
  - v3 composition plugins
runtime_patch: required
runtime_patch_reason: "Fiber-owned cancellation, candidate failure receipt and generation cleanup cannot be implemented by one plugin alone."
authoritative_state_owner: "Core owns timer lifecycle; plugin owns callback domain state and effects."
client_only_alternative: "Not applicable; this is a server plugin runtime seam."
protected_state:
  - formal workspace and plugin-data
  - existing v2 jobs and stable plugin generation
allowed_effects:
  - isolated asyncio tasks in test process
  - temporary plugin namespace and workspace fixtures
forbidden_effects:
  - formal schedule rows or plugin-data writes
  - external network, channel or GitHub effects
rollback: "Revert this adjacent PR or return to the named backup ref; v2 jobs remain unchanged."
```

## Verification

- controlled monotonic clock proves fixed cadence and missed-tick coalescing;
- slow async callback proves no overlap;
- replay-mutant runs the same fixture and produces a different wait trace;
- timeout failure appears in receipt and completed task registration self-releases;
- real namespace fixture proves Core injection and Fiber disposal cleanup;
- public plugin generation Gate binds these observations to the exact source digest.
