# Turn committed typed event 任务合同

- 状态：accepted / implementation approved
- 日期：2026-08-15
- 目标分支：`codex/plugin-mobile-ui-slots` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-turn-committed-event-before-20260815@c828b31d`
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件事件与同步执行能力合同](plugin-event-executor-task-contract.md)

> 2026-08-15 对账：首个需要等待资源回收的真实消费者 GitHub Watch 证明同步 `emit` 无法承载异步 checkout cleanup。事件改为异步 `serial` stage；位置、payload、generation scope 和失败传播不变，`Bail` 继续 fail-loud。

## Goal

让 v3 插件从当前 Turn 冻结的 generation Root 监听已经构造完成的 `TurnCommitted` 事实。Core 只提供 phase-owned typed event；插件自己决定怎样投影、排队和持久化。现有 v2 EventBus、phase slot、SessionDB、发送与晋升行为保持不变。

```text
Session commit
      │
      ▼
┌─────────────────────┐
│ legacy EventBus     │  v2 observer 保持原顺序
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ turn.after_turn.    │  async serial / current generation
│ committed           │
└──────────┬──────────┘
           ▼
     plugin-owned logic
```

## Ownership and event contract

- `after_turn` phase 拥有 `turn.after_turn.committed` 的发生位置：旧 `TurnCommitted` fanout 完成之后、budget 日志与 `AfterTurnCtx` fanout 之前。
- payload 是同一个 `TurnCommitted` 对象，不复制字段或引入第二套 DTO。
- dispatch mode 是 `serial`：同步或异步 listener 按 generation 内注册顺序逐个完成，异常立即传播；`Bail` 不能终止 Core Turn，返回时 fail-loud。
- 当前 request 没有 composition Root 时保持 no-op。事件从 request-bound snapshot 取得 Root，不读取全局最新 generation。
- listener 是所属 Fiber 的 Effect，reload、依赖消失和 dispose 后不残留。
- Core 不新增 `LifecycleEvents`、`ObserveService`、priority、waterfall 或领域数据库接口。

## Change and persistence

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - v3 plugins observing committed passive turns
  - first async cleanup consumer: GitHub Watch
runtime_patch: required
runtime_patch_reason: "Turn commit position and request-bound generation identity are Core-owned facts; a plugin cannot reconstruct them from a global service safely."
authoritative_state_owner: "Core owns TurnCommitted and snapshot binding; each plugin owns its derived state."
client_only_alternative: "A plugin-side poller would guess commit order and generation identity."
protected_state:
  - legacy EventBus handler order and v2 plugin behavior
  - phase slots, SessionDB write set and outbound dispatch
  - stable/latest promotion and snapshot lease ownership
  - formal workspace and plugin-data
allowed_effects:
  - generation-local listener registration and dispatch
  - temporary composition roots in tests
forbidden_effects:
  - formal plugin migration or runtime switch
  - workspace, plugin-data, SessionDB or manifest writes
  - channel messages and external API calls
rollback: "Revert this adjacent PR or return to backup/plugin-turn-committed-event-before-20260815."
```

本 PR 不增加、更新、逻辑失效或物理减少权威持久记录。它只在现有 Turn commit 路径增加 generation-local 内存 dispatch。

## Verification

- 证明 legacy EventBus 先完成，composition listener 随后取得同一个 payload；
- 证明异步 listener 完成后 phase 才继续，`Bail` 被 Core 拒绝；
- 证明 listener 失败立即传播，未绑定 composition Root 时旧路径保持 no-op；
- fresh interpreter 证明公开 leaf contract 不加载 `after_turn` phase runtime；
- lifecycle、plugin generation、hot reload、turn rollout 与 public change-impact Gate 保持通过。
