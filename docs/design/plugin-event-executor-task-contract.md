# 插件事件与同步执行能力任务合同

## Role

- 负责范围：组合内核的 typed event、Fiber-owned listener/task、受限同步并发执行服务、验证回执和隔离测试。
- 当前阶段：complete
- 证据状态：historical；2026-09-02 已删除本任务的内部 events/executor/kernel 测试。E1/E2 也因随后删除的 v2 组合 API 而失效；当前合同由保留的 plugin lifecycle 与 hot reload 行为回归承担。

## Goal

在第一阶段组合内核上补齐明确的事件执行合同和四象限能力，使新式插件可以使用同步串行、异步串行、异步并发和受限同步并发，同时不改变任何现有插件行为。

```text
┌────────────── 同一 generation Root ──────────────┐
│ emit     → 同步 listener，按注册顺序串行          │
│ serial   → 可同步/异步 listener，逐个 await/Bail  │
│ parallel → 异步 listener，并发启动、全部 settle   │
│ executor → 纯同步任务，有界线程池并发              │
└───────────────────┬───────────────────────────────┘
                    │ listener / task 都是 Effect
                    ▼
             Fiber restart / dispose
                    │
                    ▼
               逆序取消并排空
```

## Success criteria

- [x] `emit` 只运行同步 listener，按稳定注册顺序执行，异常立即传播，返回 awaitable 时 fail-loud。
- [x] `serial` 逐个等待 listener，只有显式 `Bail(value)` 能短路，其他非空返回值视为合同错误。
- [x] `parallel` 并发运行异步 listener，等待全部 settle，聚合全部失败；调用方取消时取消并排空子任务。
- [x] listener 和 `spawn()` 任务都是所属 Fiber 的 Effect，provider 波动、restart 和 dispose 后不残留。
- [x] `ExecutorService.parallel_sync()` 使用有界线程池，只接受显式任务，不向工作线程暴露 Context/Fiber；结果保持输入顺序并聚合失败。
- [x] Core receipt 能观察当前事件注册与后台任务；故意错误 listener 能令候选验证失败。
- [x] 旧 Plugin v2、legacy EventBus、Phase 和正式运行时行为保持不变。

## Change intent

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: core
consumer_scope:
  - composition plugin api v3
runtime_patch: required
runtime_patch_reason: "事件分发、任务取消、线程池边界和候选回执必须由 generation Root 与 Fiber owner 统一持有。"
authoritative_state_owner: "Core owns dispatch and execution lifecycle; plugins own listener and pure task behavior."
client_only_alternative: ""
invariants:
  - 一个 dispatch 使用一个冻结 listener 列表
  - listener 和 spawned task 随所属 Fiber 逆序回收
  - emit 不接受异步 listener
  - parallel 不共享可变结果并保留全部失败
  - 同步工作线程不能取得 Context 或 Fiber
  - stable/latest 与 parent Turn 晋升语义不变
protected_state:
  - existing plugin lifecycle and contribution order
  - formal workspace and plugin-data
  - sessions, memory, channels and external APIs
  - current RuntimeSnapshot publication semantics
allowed_paths:
  - agent/plugin_composition/**
  - tests/test_plugin_composition_lifecycle.py
  - tests/test_plugin_hot_reload.py
  - tests_scenarios/contracts/impact.toml
  - tests_scenarios/contracts/coverage-baseline.json
  - docs/projectneed.md
  - docs/INDEX.md
  - docs/NOW.md
  - docs/decisions/0036-plugin-composition-keeps-promotion-owner.md
  - docs/design/cordis-plugin-capability-parity.md
  - docs/design/plugin-event-executor-task-contract.md
forbidden_paths:
  - bus/event_bus.py
  - agent/lifecycle/**
  - agent/plugins/base.py
  - external plugin canonical sources
  - frontend/**
  - migrations/**
allowed_effects:
  - run tests and static checks
  - create temporary isolated test directories
forbidden_effects:
  - modify formal workspace, manifest, cache or plugin-data
  - install, promote or unload formal plugins
  - send channel messages or call real external APIs
validation:
  - focused event, executor and composition lifecycle tests
  - existing plugin snapshot and rollout regressions
  - basedpyright and compileall for changed Python
  - change-impact Gate selected from repository contract
rollback: "/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-event-executor-fca4c23b.bundle"
worktree_writer: "/mnt/data/coding/akasic-agent-worktrees/plugin-event-executor"
handoff_head: "fca4c23bc827f5e73bbed88cd25ae55c82046454"
external_revisions:
  - "deepseek-harness@47f943859bef60e4160492346772ded9b24f765a"
schema_lineages: []
```

## Deferred

- TopologyView revision 与 RuntimeSnapshot 原子刷新进入下一张 lifecycle seam PR。
- Prompt、Turn、Tool、Job、UI、MCP 等领域 Service 随首个真实消费者逐批实现。
- Citation/Meme 与其他 Plugin v2 不在本 PR 迁移。
- 通用 `waterfall`、listener priority 和 listener dependency DAG 不进入新 API。

## Stop rules

- 若事件 API 必须修改 legacy EventBus 或现有插件顺序才能成立，停止并退回设计。
- 若同步线程任务需要访问 Context、Fiber、正式数据库或外部写接口，拒绝该能力而不扩大权限。
- 若取消或卸载后仍有无法归属的 listener/task，不发布候选。

## Final evidence

- 相邻基线：`fca4c23bc827f5e73bbed88cd25ae55c82046454`；实施 worktree：`/mnt/data/coding/akasic-agent-worktrees/plugin-event-executor`；恢复 bundle：`/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-event-executor-fca4c23b.bundle`。
- typed event、Executor 与组合生命周期聚焦回归：`49 passed`；包含旧 Plugin v2、hot-reload、snapshot、安装/卸载、MCP、Skill、Job、Channel 和移动端调度的累计回归：`398 passed`。
- 修改范围 Basedpyright：`0 errors, 0 warnings`；相关 `compileall` 与 `git diff --check` 通过。
- change-impact Gate：`passed`；最终 report、`sourceDigest` 与 `planDigest` 在提交前的验证阶段冻结。
- 未接入 legacy EventBus、Phase、正式 manifest 或外部插件 canonical source；未运行正式 Akashic workspace，未发送渠道消息或调用真实外部 API。
