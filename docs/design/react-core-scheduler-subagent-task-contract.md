# React Core、Scheduler 与 Subagent 分阶段任务合同

## Role

- 负责范围：按 Core 基建、Subagent、Subagent fixture、Timer/Scheduler、Timer fixture 五个独立阶段，建立差分 runner 并把两个来源迁成仓库内置非特权 v3 插件。
- 当前阶段：design complete；implementation not started

## Goal

用户看见的 Scheduler 与 Subagent 行为保持不变，但二者都通过公开 Core 原子能力组合同一条 `react`；Core 不再拥有两种来源的业务分支，未来 Wake 风格主动链路无需再造执行模型。

## Success criteria

- [ ] S0 建立 disposable fixture runner、scoped Turn port/handle、exact scope、Tool grant 与 typed receipt 投影；passive 零差异，`tool_loop_guard:` 零 consumer 残留删除，one-shot Timer 只冻结接口合同。
- [ ] S1 Subagent 通过正式 v3 loader 与公开 Service 运行，递归使用同一 `react`；旧路径仍作为 shadow oracle，不立即切换 owner。
- [ ] S2 Subagent fixtures 与 mutants 覆盖同步/后台/profile/容量/终态/取消/重载；等价后才切换 binding 并删除独立推理循环。
- [ ] S3 实现来源无关的 one-shot Timer，并让 Scheduler 通过正式 v3 loader 组合 Store、Timer、Turn、delivery 与 settlement；旧路径保留为 shadow oracle。
- [ ] S4 Timer/Scheduler fixtures 与 mutants 覆盖时间、恢复、投递和资源归零；等价后切换 binding、删除旧入口并运行累计全量 Gate。
- [ ] 相关验证已运行，未运行项和原因已说明。

## Evidence

- 必须先读取：`docs/INDEX.md`、`docs/WORKFLOW.md`、`docs/projectneed.md` 第 1～6、9～13 节、决策 0034/0036/0039、`persistence-state-map.md`、`recursive-plugin-self-validation.md`、`cordis-plugin-capability-parity.md`、`plugin-v3-proactive-jobs-task-contract.md` 与本设计。
- 已核对事实：被动链路由 `AgentLoop._react → PassiveTurnPipeline` 执行；Scheduler SOFT 调 `process_direct(stateless=True)` 后投递；Subagent 使用独立 `SubAgent` 循环、snapshot lease、completion event、spawn trace 和 task directory。
- 未确认事实：每个现有 lifecycle module 在三种 scope 中的精确必要性；历史生产路径是否存在与 OUT-001/0034 不一致的行为；旧入口在 installed cache 和外部插件中的全部消费者。
- 关键假设：每片先做 characterization，发现合同冲突即停，不在 `semantic_delta: none` 重构中修复。

已确认控制流边界：插件私有的“记录后 return”只能结束插件自己拥有的 tick、fire callback 或 spawn admission。普通 lifecycle listener 返回只结束 listener；现有 composition lifecycle 禁止 `Bail`，Tool authorize 的公开合同只拒绝一次工具。当前 passive/subagent Reasoner 的 `tool_loop_guard:` deny 前缀是维护者明确要求删除的失败实现；hua-home active manifest/cache/runtime 已无该插件，S0 补齐 canonical/installed 零 consumer 证明后直接删除专属分支与旧 Gate，不建立替代控制协议。任何新的“在某个 lifecycle 点结束整个 Turn”需求都必须另立 Turn 终态与 cleanup 合同，不属于 S0。

已知但未批准的候选变化：当前 subagent cancel 先发布 cancelled completion 再取消 worker；改成 child cleanup 后才发布会改变可观察顺序，不属于本合同。Scheduler SOFT 是否应裁掉当前实际运行的 passive-only hooks，也必须由 S0 回执和后续 `declared_delta` 决定。

## Change intent

```yaml
change_type: refactor
semantic_delta: none
capability_owner: mixed
consumer_scope:
  - passive turns
  - builtin scheduler plugin
  - builtin subagent plugin
runtime_patch: required
runtime_patch_reason: "Turn owner、generation lease、取消、Tool executor 权限和 terminal 是跨来源一致事实；只在插件侧实现会复制 Core 语义。"
authoritative_state_owner: "Core owns Turn execution and plugin publication; Scheduler owns schedules; Subagent owns spawn state and artifacts; Session and Channel owners remain unchanged."
client_only_alternative: "not_applicable"
invariants:
  - Message 组成 Turn，Turn 归入 Session，Loop 只表达 Message 到 react 到 Message
  - 同 session Turn 串行，不同 session 可并发
  - 一个 Turn 从 admission 到 cleanup 冻结 exact model and runtime generation
  - passive lifecycle Prompt tools memory persistence and delivery remain unchanged
  - Scheduler and Subagent have no builtin privilege or Core source branch
  - plugin-private early return cannot terminate a Core-owned Turn
  - proactive Wake and Drift are untouched
protected_state:
  - sessions.db messages turns and interaction identity
  - schedules.json schema values recovery and cancellation semantics
  - memory and Akasha write sets
  - proactive.db wake_proactive.db drift.db and proactive documents
  - spawn_trace.jsonl and subagent-runs retention
  - plugin stable latest generation publication and cleanup
  - channel delivery ordering and receipts
allowed_paths:
  - agent/looping/**
  - agent/core/passive_turn.py
  - agent/lifecycle/**
  - agent/plugin_composition/**
  - agent/plugins/**
  - agent/scheduler.py
  - agent/background/**
  - plugins/scheduler/**
  - plugins/subagent/**
  - bootstrap/**scheduler**
  - bootstrap/**subagent**
  - bootstrap/tools.py
  - tests/**scheduler**
  - tests/**subagent**
  - tests/**lifecycle**
  - tests/**plugin_composition**
  - docker/debug/**
  - tests_scenarios/contracts/**
  - docs/INDEX.md
  - docs/NOW.md
  - docs/decisions/0039-react-core-atoms-keep-sources-unprivileged.md
  - docs/design/react-core-scheduler-subagent.md
  - docs/design/react-core-scheduler-subagent-task-contract.md
forbidden_paths:
  - migrations/**
  - proactive_v2/**
  - plugins/default_proactive/**
  - plugins/wake_proactive/**
  - plugins/drift_flow/**
  - frontend/**
  - external plugin canonical sources
allowed_effects:
  - create isolated Git worktrees and recoverable Git backups
  - create run-identified disposable workspace plugin home and debug receipts
  - use fixed clocks scripted providers and recording adapters
forbidden_effects:
  - modify formal Akashic workspace plugin home cache manifest or runtime
  - read or write formal channel credentials
  - send real messages or call real external APIs
  - deploy restart promote or unload formal plugins
  - delete or migrate any authoritative state
validation:
  - focused unit and integration tests per slice
  - old versus candidate semantic receipts and write-set comparison
  - static dependency and no-special-case checks
  - lifecycle Tool grant memory cancellation HMR and cleanup mutants
  - private gate no-Turn receipt versus accepted Turn receipt comparison
  - ordinary tool deny plus tool_loop_guard zero-consumer removal mutant
  - repository change-impact Gate against current origin/main
rollback: "/mnt/data/coding/backups/akasic-agent-react-core-no-loop-guard-20260822-092ee320/spec-v2.bundle plus one Git bundle or tag per implementation slice"
worktree_writer: "/mnt/data/coding/akasic-agent-worktrees/react-core-scheduler-subagent-spec for specification only"
handoff_head: "implementation slices must record their own exact committed head"
external_revisions: []
schema_lineages: []
```

每个实施切片必须复制本合同并进一步收窄 `allowed_paths`。本 umbrella 合同不授权一次 PR 同时修改全部路径。

## Autonomy

- 可自主执行：在独立 worktree 中读取当前实现、创建可恢复备份、使用一次性 workspace 运行无真实外部效果的测试、按 S0～S4 逐阶段修改已批准范围并做只读 Review。
- 执行前需确认：任何用户可见语义差异、数据库/schema 变化、正式 workspace/plugin 状态变化、真实消息/API 调用、proactive/Wake/Drift 迁移、durable child Session 或 fork 历史。

## Tools

| 工具 | 使用时机 | 关键结果 | 空/失败如何处理 |
|---|---|---|---|
| CodeGraph | worktree 有现成索引时定位调用路径和 consumer | exact symbols 与 call path | 没有索引则使用 `rg` 和定点读取，不自行建索引 |
| `rg` / Git | 扫描 canonical source、cache 线索、diff 和消费者 | owner、入口、零 consumer 证据 | 命中不清楚则扩展到正式安装链；不能从空命中直接删除 |
| scenario runner | S0 起执行旧/新确定性场景 | identity/turn/state/effects/lifecycle/verdict | runner 或 fixture 错误是环境/测试失败，不算 candidate 通过 |
| `docker/debug/gate.py` | 每片聚焦验证后运行公开 Gate | pass/fail 与 impact report | fail 时归因实现、环境或合同冲突，不缩减 Gate |

## Output

- 交付文件或字段：S0 Core 基建与 runner；S1 Subagent 插件；S2 Subagent fixture 报告与切换；S3 Timer/Scheduler 实现；S4 Timer/Scheduler fixture 报告、切换与最终回执。
- 格式和长度：每片一个可审阅 commit/PR；实现 task contract 记录 base/head、allowed diff、验证和回滚点。
- 必须附带的证据：完整 diff、scenario identity、normalized verdict、persistent write set、external effects、resource cleanup、mutant verdict、未运行项。

## Stop rules

- 满足当前切片全部成功标准后停止，不顺手进入下一片。
- 缺少 lifecycle/module consumer、正式 cache consumer 或历史状态语义时，先做最小只读调查。
- 最多保留旧路径做一轮 shadow differential；无法证明等价时保留旧 owner并报告，不叠加兼容壳。
- 需要放宽 oracle、修改 protected state、产生真实外部效果或扩展到 Proactive 时停止并请求批准。
- 需要让 lifecycle listener、异常、共享 flag 或 event-specific Bail 获得“结束整个 Turn”的新含义时停止并另立合同。
- 发现 `tool_loop_guard` 仍有 active/canonical consumer，或删除需要减少旧 plugin-data 时停止并重新核对范围。
