# React Core、Scheduler 与 Subagent 分阶段任务合同

## Role

- 负责范围：按 Core 基建、Subagent、Subagent fixture、Timer/Scheduler、Timer fixture 五个独立阶段，建立差分 runner 并把两个来源迁成仓库内置非特权 v3 插件。
- 当前阶段：S0～S4 complete

## Goal

用户看见的 Scheduler 与 Subagent 行为保持不变，但二者都通过公开 Core 原子能力组合同一条 `react`；Core 不再拥有两种来源的业务分支，未来 Wake 风格主动链路无需再造执行模型。

## Success criteria

- [x] S0 建立 disposable fixture runner、scoped Turn port/handle、exact scope、Tool grant 与 typed receipt 投影；passive 零差异，`tool_loop_guard:` 零 consumer 残留删除，one-shot Timer 只冻结接口合同。
- [x] S1 Subagent 通过正式 v3 loader 与公开 Service 运行，递归使用同一 `react`；旧路径仍作为 shadow oracle，不立即切换 owner。
- [x] S2 Subagent fixtures 与 mutants 覆盖同步/后台/profile/容量/终态/取消/重载；等价后切换 binding 并删除独立推理循环。
- [x] S3 实现来源无关的 one-shot Timer，并让 Scheduler 通过正式 v3 loader 组合 Store、Timer、Turn、delivery 与 settlement；旧路径保留为唯一生产 owner 与 shadow oracle。
- [x] S4 Timer/Scheduler fixtures 与 mutants 覆盖时间、恢复、投递和资源归零；等价后切换 binding、删除旧入口并运行累计全量 Gate。
- [x] 相关验证已运行，未运行项和原因已说明。

## Evidence

- 必须先读取：`docs/INDEX.md`、`docs/WORKFLOW.md`、`docs/projectneed.md` 第 1～6、9～13 节、决策 0034/0036/0039、`persistence-state-map.md`、`recursive-plugin-self-validation.md`、`cordis-plugin-capability-parity.md`、`plugin-v3-proactive-jobs-task-contract.md` 与本设计。
- 已核对事实：被动链路由 `AgentLoop._react → PassiveTurnPipeline` 执行；Scheduler SOFT 调 `process_direct(stateless=True)` 后投递；Subagent 使用独立 `SubAgent` 循环、snapshot lease、completion event、spawn trace 和 task directory。
- 未确认事实：每个现有 lifecycle module 在三种 scope 中的精确必要性；历史生产路径是否存在与 OUT-001/0034 不一致的行为；旧入口在 installed cache 和外部插件中的全部消费者。
- 关键假设：每片先做 characterization，发现合同冲突即停，不在 `semantic_delta: none` 重构中修复。

S0 实施证据：代码基线 `0d1a2f97`，实现 head `74bf8303`；提交 `2b6c95b0` 删除失败的控制暗号，`42dc6b8b` 建立 scoped Turn，`0220137c` 建立 Tool grant 与 Timer 合同，`74bf8303` 修正 fleet 数量 oracle。阶段性差分 fixture 已随迁移收束删除；现行 scoped Turn 的 admission、terminal、release 和失败清理由 `tests/control/test_scoped_turn.py` 直接验证。S0 当时的定向控制面回归为 `156 passed`，最终全量 `3998 passed, 2 skipped`，Change Gate 报告 `docker/debug/reports/change-gate/20260822-210051-a6a0816a` passed。没有正式 workspace、channel、外部 API、Scheduler/Subagent owner 或 proactive 状态变化。

已确认控制流边界：插件私有的“记录后 return”只能结束插件自己拥有的 tick、fire callback 或 spawn admission。普通 lifecycle listener 返回只结束 listener；现有 composition lifecycle 禁止 `Bail`，Tool authorize 的公开合同只拒绝一次工具。S0 已删除 passive/subagent Reasoner 的 `tool_loop_guard:` deny 前缀分支、fleet lock 与专属旧 Gate，没有建立替代控制协议，也没有减少 hua-home 旧 plugin-data。任何新的“在某个 lifecycle 点结束整个 Turn”需求都必须另立 Turn 终态与 cleanup 合同，不属于 S0。

已知但未批准的候选变化：当前 subagent cancel 先发布 cancelled completion 再取消 worker；改成 child cleanup 后才发布会改变可观察顺序，不属于本合同。Scheduler SOFT 是否应裁掉当前实际运行的 passive-only hooks，也必须由 S0 回执和后续 `declared_delta` 决定。

## Change intent

```yaml
change_type: migration
semantic_delta: breaking
capability_owner: mixed
consumer_scope:
  - passive turns
  - builtin scheduler plugin
  - builtin subagent plugin
runtime_patch: required
runtime_patch_reason: "Turn owner、generation lease、取消、Tool executor 权限和 terminal 是跨来源一致事实；只在插件侧实现会复制 Core 语义。"
authoritative_state_owner: "Core owns Turn execution and plugin publication; Scheduler owns schedules; Subagent owns spawn state and artifacts; Session and Channel owners remain unchanged."
client_only_alternative: "not_applicable"
concept_gate: required
concept_gate_reason: "新增 Core Turn/Timer/lifecycle 原子，迁移两个 execution owner，并删除旧 Core source branches。"
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
  - agent/config.py
  - agent/config_models.py
  - agent/policies/**
  - bus/**
  - prompts/**
  - plugins/scheduler/**
  - plugins/subagent/**
  - bootstrap/**
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
  - docs/WORKFLOW.md
  - docs/templates/**
  - .github/pull_request_template.md
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
  - call the authorized DeepSeek model from an isolated workspace for final E2E
forbidden_effects:
  - modify formal Akashic workspace plugin home cache manifest or runtime
  - read or write formal channel credentials
  - send real channel messages or call unrelated external APIs
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
rollback: "/mnt/data/coding/backups/akasic-agent-react-core-stage0-20260822-0d1a2f97/stage0-baseline.bundle and stage0-final.bundle"
worktree_writer: "/mnt/data/coding/akasic-agent-worktrees/react-core-stage0-infra"
handoff_head: "S0 implementation head 74bf8303; later commit only reconciles project documents"
external_revisions: []
schema_lineages: []
```

| fact / invariant | sole decision/write owner | public reader/port | unrelated change propagation | static/dynamic oracle |
|---|---|---|---|---|
| Turn admission、terminal 与 exact Root lifetime | Core `ConversationRuntime` / `RuntimeSnapshotStore` | `PluginScopedTurns` | none to source plugins | scoped Turn、HMR handoff、lease-zero fixtures |
| one-shot deadline、cancel 与 receipt | Core Timer | `PluginTimers` | none to Turn/Session/memory | fixed-clock Timer fixtures |
| durable schedules 与 fire settlement | Scheduler plugin | Store + Timer + scoped Turn + Delivery | none to Subagent/Core react | restart/misfire/HMR/write-set fixtures |
| child profile、capacity、artifact 与 completion | Subagent plugin | scoped Turn + Continuation | none to Scheduler/Core react | profile/capacity/cancel/continuation fixtures |
| runtime Root start/stop order | PluginManager snapshot drain | `RUNTIME_STARTED / RUNTIME_STOPPING` | none to source identity | Bail/retry and real manager reload fixtures |
| exact Root 的 Tool runtime 路由 | Root-local `PluginToolBinding` | bound async handler | none to plugin source/module globals | shared-generation 双 Root 与 restart soak |

每个实施切片必须复制本合同并进一步收窄 `allowed_paths`。本 umbrella 合同不授权一次 PR 同时修改全部路径。

### S1 收窄合同

```yaml
stage: S1
base_head: db092ef069c6972128d82306d0d5db6492b551ef
change_type: refactor
semantic_delta: none
capability_owner: "Core owns transient Turn scope and admission; builtin v3 Subagent owns profile, task directory, lineage projection and shadow trace."
consumer_scope:
  - exact-generation builtin subagent shadow fixture
runtime_patch: required
runtime_patch_reason: "Turn scope must cross ConversationRuntime admission without persisting private Prompt, Tool grant, or memory policy into authoritative Session metadata."
authoritative_state_owner: "SessionStore remains the Turn audit owner; Subagent owns only its task directory and plugin shadow trace."
protected_state:
  - production spawn and spawn_manage Tool bindings
  - legacy SubagentManager execution and cancellation order
  - passive Prompt lifecycle memory tools and persistence
  - formal workspace plugin state and outbound channels
allowed_paths:
  - agent/control/runtime.py
  - agent/control/scoped_turn.py
  - agent/control/turn_scope.py
  - agent/core/passive_turn.py
  - agent/plugin_composition/__init__.py
  - agent/plugin_composition/scoped_turns.py
  - agent/plugins/manager.py
  - agent/tools/events.py
  - bootstrap/control_execution.py
  - bootstrap/tools.py
  - plugins/subagent/plugin.py
  - tests/control/test_scoped_turn.py
  - tests/test_subagent_v3_shadow.py
  - docs/NOW.md
  - docs/design/react-core-scheduler-subagent.md
  - docs/design/react-core-scheduler-subagent-task-contract.md
forbidden_effects:
  - publish a production subagent Tool or switch bootstrap owner
  - invoke a candidate child Turn
  - write formal workspace Session plugin-data task directories or channels
  - alter legacy cancellation completion ordering
validation:
  - real v3 loader formal shadow invocation through SCOPED_TURNS
  - candidate service rejects Session and Turn creation
  - exact scope Prompt grant memory source and cleanup assertions
  - passive and plugin composition regressions
  - pyright and change-impact Gate
rollback: /mnt/data/coding/backups/akasic-agent-react-core-subagent-stage1-20260822-db092ef0/s1-baseline.bundle
```

S1 的 shadow 入口不登记到 `TOOL_CATALOG`，因此不会出现在正式模型工具表中。S2 只有在差分 fixture 与 mutants 全部通过后，才可把公开 `spawn` binding 从 legacy manager 切到插件；这个合同不把“代码可调用”误当成 owner 已切换。

S1 实施证据：基线 `db092ef0`；`SCOPED_TURNS` 只暴露 Session 创建与 exact-snapshot scoped Turn admission，candidate facade 在任何 Session/Turn 写入前拒绝。`TurnExecutionScope` 是非持久 runtime view，冻结 Prompt hints、Tool grant、memory read/write、stateless 与 tool event source；普通被动 Turn 没有绑定 scope 时保持原路径。`plugins/subagent/plugin.py` 由真实 v3 loader 挂载，私有 shadow 入口创建 task directory、映射 research/scripting/general profile 并取得同一 `ConversationRuntime` terminal，但没有登记生产 Tool。定向回归 `215 passed`，Pyright `0 errors`，Change Gate `docker/debug/reports/change-gate/20260822-214449-34783c80` passed；没有正式 workspace、channel、旧 spawn owner 或取消顺序变化。

### S2 收窄合同

```yaml
stage: S2
base_head: 834b69ca992c6b31d2eecf4f3d5b20fe4966ae2c
change_type: refactor
semantic_delta: none
capability_owner: "Core owns scoped Turn, transient tool overrides, exact snapshot and continuation admission; builtin v3 Subagent owns spawn admission, profiles, task artifacts, completion and trace."
protected_state:
  - sessions.db message retention and schema
  - schedules.json and proactive Wake Drift state
  - cancellation completion-before-interrupt ordering
  - formal workspace channels and external APIs
allowed_paths:
  - agent/control/turn_scope.py
  - agent/core/passive_turn.py
  - agent/plugin_composition/**
  - agent/plugins/manager.py
  - agent/tools/registry.py
  - agent/background/**
  - agent/subagent.py
  - agent/tools/spawn.py
  - bootstrap/tools.py
  - bootstrap/toolsets/meta.py
  - plugins/subagent/plugin.py
  - tests/**subagent**
  - tests/test_shell_tool.py
  - tests/test_plugin_hot_reload.py
  - tests/test_plugin_packages.py
  - tests/semantic/test_react_core_contract.py
  - docs/NOW.md
  - docs/design/react-core-scheduler-subagent*.md
forbidden_effects:
  - use formal workspace plugin home or channel credentials
  - send real messages or call real external APIs
  - change scheduler or proactive owners
validation:
  - exact profile grants and task-local Tool instances
  - sync and background terminal receipts
  - capacity rejection before child Session creation
  - completion exactly once and cancel before interrupt
  - no late success and snapshot lease count returns to zero
  - full pytest pyright and change-impact Gate
rollback: /mnt/data/coding/backups/akasic-agent-react-core-subagent-stage2-20260822-834b69ca/s2-baseline.bundle
```

S2 已把 `spawn`/`spawn_manage` 正式 binding 切到 builtin v3 插件；bootstrap 的旧 `spawn` toolset 只保留无状态 wiring slot，`spawn_enabled=false` 通过通用 builtin activation filter 禁用该插件。`TurnExecutionScope.tool_overrides` 只冻结已授权名称对应的 Turn-local Tool 实例；文件根、shell cwd 和网络限制仍由原 Tool 边界 owner 强制，Core 无 profile/source 特判。新 fixture 杀死权限串值、重复 completion、cancel 后迟到 success 和 lease 残留四类 mutant。旧 `SubAgent`、`SubagentManager`、profile builder、background runner 和 legacy Tool adapter 已物理删除。

S2 实施证据：基线 `834b69ca`；正式 Tool description 与 JSON Schema 保持 legacy Prompt surface，生产调用改由插件私有 runtime 组合 scoped Turn、continuation 与 Turn-local Tool。定向回归 `90 passed`，最终全量 `3972 passed, 2 skipped`，Pyright `0 errors`，Change Gate `docker/debug/reports/change-gate/20260822-224014-ce93accb` passed。验证只使用一次性 workspace/provider/channel recorder，没有启动正式 runtime、写入正式 SessionStore 或向真实渠道投递。

### S3 收窄合同

```yaml
stage: S3
base: 787dbfcb
change_type: compatible
semantic_delta: none
capability_owner: Core owns one-shot Timer, committed Delivery, scoped Turn admission and declared workspace-file projection; Scheduler plugin owns jobs, recurrence, misfire, fire, settlement and shadow lifecycle
consumer_scope: repository builtin scheduler shadow only; legacy SchedulerService remains sole production owner
runtime_patch: true
runtime_patch_reason: a plugin-neutral Timer and narrow Delivery boundary cannot be implemented inside the Scheduler plugin without duplicating Core time and Channel custody
authoritative_state_owner: legacy SchedulerService for production in S3; scheduler plugin only in disposable fixture or dormant formal mount
client_only_alternative: not applicable
allowed_changes:
  - one Timer registration settles exactly once as fired or cancelled
  - plugins may submit one complete logical message through a narrow Delivery Service
  - a plugin may declare one top-level product file and receive an isolated candidate projection
  - Scheduler v3 shadow may explicitly load an isolated schedules.json and arm one Timer per active job
protected_state:
  - production schedules.json and legacy Scheduler binding remain unchanged
  - candidate mount creates no Session, wait, Turn, delivery or formal schedule write
  - no proactive, Wake, Drift, memory or formal channel state changes
allowed_side_effects:
  - repository files and disposable fixture workspace only
verification:
  - Timer fire/cancel/cleanup receipts and zero residual wait
  - instant/every/SOFT/delivery-failure/cancel/dispose Scheduler shadow fixtures
  - real v3 formal/candidate loader fixture with deliberately invalid dormant store
  - legacy Scheduler/Tool/runtime-inspection regression, pyright and Change Gate
rollback:
  - revert the S3 commit or restore the named S3 bundle
```

S3 实施证据：基线 `787dbfcb`；Core `TIMERS` 不知道 cron、job、source 或 delivery，`DELIVERIES` 只提交完整逻辑消息并把非 delivered receipt 暴露为失败，`workspace_files` 只投影插件声明的顶层普通文件。Scheduler v3 正式挂载保持 dormant，candidate 使用隔离文件投影且不会读取损坏 store；显式 shadow fixture 才组合 wait、SOFT scoped Turn、delivery 与 settle。Timer/Scheduler/legacy/loader 定向回归 `209 passed`，聚焦 Pyright `0 errors`；未连接正式 workspace、真实 provider 或 channel。

### S4 收窄合同与实施证据

S4 以 `e7844474` 为基线，`semantic_delta: breaking`，仅对应下文明确列出的配置键迁移。Core 只新增来源无关的 formal Runtime Root start/stop 生命周期，并用 snapshot lease 跟随 stable Root 热重载；它不知道 Scheduler、cron、misfire 或 job。Scheduler 插件私有 runtime 成为 `schedules.json`、recurrence、misfire、wait、SOFT Turn、delivery 和 settlement 的唯一生产 owner，旧 `SchedulerService`、`agent/tools/schedule.py` 和 bootstrap runtime binding 已物理删除；移动端只读投影通过同一个严格 `JobStore` schema 读取，不获得执行或删除能力。

确定性 fixture 覆盖 instant/SOFT、one-shot/every/cron、capacity-before-write、misfire grace/expired、restart、delivery rejection、空 SOFT terminal、cancel/dispose、disabled no-work 和真实 v3 candidate/publish 热重载；旧 Root wait 归零后新 Root 恰好挂载一个 wait。S4 定向回归为 `244 passed`，聚焦 Pyright 为 `0 errors`。最终全量 pytest 与 Change Gate 证据只记录在 PR，避免文档回填改变 Gate source digest。未连接正式 workspace、真实 model/provider/channel，也未修改 Proactive/Wake/Drift 任何状态或 owner。

S4 累计概念审查发现并修正四项方向性问题。第一版把 runtime lifecycle 绑定在长期 snapshot lease 上，热更新会与 admission drain 互相等待；最终 lifecycle supervisor 只短暂租用 stable Root，旧 Root 在 snapshot drain 中先 `RUNTIME_STOPPING`，排空后新 Root 才 `RUNTIME_STARTED`。第二版曾尝试把 start/stop 塞进可回滚 publication participant，但 stop 对任意插件不保证可逆，因此撤回；旧 SOFT fire 若在交棒窗口尚未获得 Turn admission，只收到通用 `TurnAdmissionRetiredError`，不结算 durable job，新 Root 从未改写的 `schedules.json` 重挂。第三，清理阶段证明旧 `SpawnCompletionItem/Event`、Core delegation/profile prompt、spawn/schedule no-op wiring 与 bootstrap shutdown 均无生产构造者，已物理删除；profile、并发 admission 与 child prompt 归入 `plugins/subagent/`，Core/Bootstrap 不再按来源 ID 分支。第四，restart soak 证明未变化的 builtin generation 会同时挂进旧、新两棵 Root，模块级 `runtime` 会把两个房间的按钮接到同一个玩具上；最终复用已有 `PluginToolBinding` 保存本 Root 的 bound handler，Tool、lifecycle 和 cleanup 都捕获同一个插件私有 runtime，不新增 PluginState、root-token 字典或 Scheduler/Subagent 专属 Core 分支。

配置迁移是本阶段唯一 breaking delta：`spawn_enabled=false` 已 fail-loud 退役，替代为 `[agent.plugins] disabled_builtin = ["subagent"]`。这是通用 builtin activation projection；Scheduler 与 Subagent 的工具、生命周期和状态仍由各自插件拥有。

最终独立 Concept Integrity Gate 由 `gpt-5.6-terra` xhigh 执行并给出 PASS、零 must-fix。新增真实 fixture 同时运行未变化的 Scheduler/Subagent 与独立 MCP v1，在旧 Root Turn lease 下准备 MCP v2，并在发布交棒窗口证明两棵 Root 共享同一插件 module、却拥有不同的 Root-local handler；旧 Root cleanup 后，新 exact ToolRegistry 仍可执行 `spawn` 和 `list_schedules`。该夹具不把 Scheduler/Subagent 名称引入 Core，只在测试端按产品能力核对结果。

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
