# 插件 Timer stable snapshot 调度任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-15
- 目标分支：`codex/plugin-parallel-sync-typing` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-timer-snapshot-scheduling-before-20260815@34eb644b`
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[插件 Timer 组合 Service 合同](plugin-timer-service-task-contract.md)、[插件递归自验证运行时设计](recursive-plugin-self-validation.md)
- 首个真实消费者：GitHub Watcher

## Goal

保留 Cordis 的“Timer 是 Fiber effect，不是 Job 类别”语义，同时补上 Akashic 独有的 stable/latest 与 snapshot lease：插件只声明 timeout/interval 回调；Core 只从当前 stable snapshot 取回调、串行执行并持有精确 lease。未晋升 latest 候选不得因计时自动产生插件数据、网络或 Agent Input。

```text
v3 Fiber                     RuntimeSnapshotStore
   │ Timer declaration              │ stable only
   ▼                                ▼
┌────────────────┐  freeze   ┌──────────────────┐
│ core.timer     │──────────▶│ snapshot.timers  │
└────────────────┘           └────────┬─────────┘
                                      ▼ lease + bind
                             ┌──────────────────┐
                             │ serial scheduler │
                             └────────┬─────────┘
                                      ▼
                              plugin callback

latest candidate ── declaration/readiness only ── no tick
```

## Ownership and invariants

- 插件拥有 callback、领域状态、远程读取和后续 Agent Input；Core 不增加 `GitHubWatchService`、`JobService` 或领域轮询接口。
- `TimerService` 在 Manager Root 中只收集 Fiber-owned 声明；Root dispose 或 handle close 把声明标成 inactive，并等待已经进入 callback 的调用结束。
- snapshot 固定 timer key、kind、delay、callback owner；重复的 `plugin_id + name` 在候选编译时 fail-loud。
- 后台 runtime 只读取 `RuntimeSnapshotStore.current`，不读取 latest；candidate 等待行为验证期间零 callback。
- 每次 callback 入队前取得精确 snapshot lease，执行期间绑定该 snapshot；同一 runtime 单 worker 串行执行，interval 的同名 tick coalesce。
- interval 保留固定单调 deadline，同 key、同 delay 的 generation 替换不补跑错过 tick；timeout 每个 snapshot 至多执行一次。
- timeout 执行结束后 handle 进入 done，过期 deadline 被移除，后台循环不会退化成短间隔空转。
- 当前 PR 复用旧 `PluginJobRuntime` 的全局队列作为 Core 内部运输层，v3 插件和公开 Timer API 不接触 `PluginJobSpec` 或 `PluginJobContext`。旧 v2 Job 全量迁移后再单独收敛内部命名和实现。
- callback 失败保持旧后台任务的 fail-loud 日志语义，不伪造成功，也不自动晋升、回滚或重试外部效果。

## Change and persistence

```yaml
change_type: fix
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - composition plugin API v3 Timer consumers
  - first planned consumer: github-watch
runtime_patch: required
runtime_patch_reason: "stable/latest selection, snapshot lease, global serialization and candidate isolation are Core-owned facts."
authoritative_state_owner: "Core owns cadence and snapshot selection; plugins own callback effects and plugin-data."
client_only_alternative: "A plugin-side validation-path check would infer Core state from filenames and cannot own promotion races."
protected_state:
  - legacy v2 Job catalog, queue, coalesce and handler behavior
  - stable/latest promotion and parent Turn authorization
  - formal workspace, plugin-data and SessionDB
  - existing Timer direct-mode unit semantics
allowed_effects:
  - in-memory timer declarations, due cursors and snapshot leases
  - temporary workspaces and callback traces in tests
forbidden_effects:
  - formal plugin install, promotion or runtime switch
  - formal plugin-data, SessionDB, channel or external API writes
rollback: "Revert this adjacent PR or return to backup/plugin-timer-snapshot-scheduling-before-20260815."
```

本 PR 不增加、更新、逻辑失效或物理减少权威持久记录。timer cursor、timeout fired set 和队列都只是进程内运行连续性；重启后重新从当前 stable snapshot 建立 cadence。

## Verification

- direct Timer 的固定 cadence、慢 callback 不重叠、取消和异常测试保持通过；
- Manager namespace Timer 在没有后台 runtime 时只声明、不 tick；
- stable runtime 启动后 interval tick，latest candidate 即使等待超过 delay 也零 tick；
- timeout 对同一 snapshot 至多一次，新的 stable generation 可以重新执行；
- reload 保留 interval key 的 coalesce、精确 snapshot lease 和 cleanup，旧 callback 未结束时新 generation 不并发执行；
- handle close 后冻结在 snapshot 中的声明不再 tick，并等待已经进入 callback 的调用结束；
- 旧 v2 Job targeted regressions、Plugin Manager、hot reload、组合 lifecycle 与公开 Gate 通过。
