# Akashic Agent 项目阅读索引

这份文件是每个新会话进入仓库后的第一站。它只回答三件事：先读什么、什么情况下继续读哪份文件、读完后去哪里核对真实实现。

它不是需求规范，也不保存临时进度。修改仓库文件时按 [`WORKFLOW.md`](WORKFLOW.md) 执行；长期语义以 [`projectneed.md`](projectneed.md) 为准，当前接手点以 [`NOW.md`](NOW.md) 为准，决策理由以 [`decisions/`](decisions/README.md) 为准。

## 1. 先分清 Git worktree 与 Akashic workspace

本项目同时使用两个容易混淆的“工作区”，它们不是一回事：

```text
┌──────────────────────────────┐
│ Git repository / worktree    │  源码、测试、项目工作手册、Git diff
└──────────────────────────────┘

┌──────────────────────────────┐
│ Akashic <workspace>          │  会话、记忆、附件、调度、主动流程、
│                              │  plugin-data、能力投影和运行状态
└──────────────────────────────┘
```

文档中的裸词 `workspace` 一律指第二种：由 `--workspace`、`AKASHIC_WORKSPACE` 或 `config.toml` 选中的 Akashic 运行数据根。要表达代码副本时必须写 `Git worktree`、`repository` 或 `checkout`。代码 worktree 可以随时重建；正式 Akashic workspace 含用户和 agent 的持续数据，不能随代码清理、切分支或重构一起变化。

## 2. 新会话固定入口

无论任务看起来多简单，进入仓库后的第一个主动读取动作都是本文件。本地 `AGENTS.md` 与 `CLAUDE.md` 由 coding agent 运行环境提供，不属于版本化项目文档；本索引负责把会话带到任务需要的项目事实。

按下面的顺序读取：

1. **先读本索引全文。** 确认任务类型、状态 owner、必读材料和停止条件。
2. **确认执行顺序。** 会修改仓库文件时读取 [`WORKFLOW.md`](WORKFLOW.md)；只读问答和调查按用户授权停在对应阶段。
3. **建立公共理解。** 非简单任务读取 [`projectneed.md`](projectneed.md) 第 1～6 节；简单、纯局部任务至少核对与改动直接相关的条款。
4. **确认当前接手点。** 读取 [`NOW.md`](NOW.md)，只把仍未完成的事项带入当前任务。
5. **按任务路由展开。** 使用第 4 节的表，只读相关领域、决策和设计，不批量装填全部历史。
6. **最后检查真实证据。** 读取当前分支上的代码、配置、日志、数据库 schema 和测试。文档说明目标与理由，代码证明当前实际行为；两者冲突时先报告，不自行改写其中一方。

不要用下面几种方式开工：

- 从旧会话摘要、自动记忆或某个搜索命中直接推导项目意图。
- 为了“上下文完整”一次读入全部 `_handbook/`、全部决策和全部历史设计。
- 只看代码能做什么，就反推用户原本想要什么。
- 只看文档目标，不检查当前实现、当前分支和真实数据路径。

## 3. 文档骨架与权威边界

```text
┌─────────────────────┐
│ docs/INDEX.md       │  新会话入口，只负责阅读路由
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ docs/projectneed.md │  长期需求、禁止事项、语义不变量
└──────┬────────┬─────┘
       │        │
       ▼        ▼
┌────────────┐  ┌──────────────────┐
│ NOW.md     │  │ decisions/       │
│ 当前未完成 │  │ 决策理由与勘误   │
└──────┬─────┘  └────────┬─────────┘
       │                 │
       └────────┬────────┘
                ▼
       ┌──────────────────┐
       │ design/          │  问题级调用链、状态地图、迁移与验收
       └────────┬─────────┘
                ▼
       ┌──────────────────┐
       │ 代码/配置/数据/测试│  当前实现证据
       └──────────────────┘
```

文件职责如下：

| 文件或目录 | 回答的问题 | 读取策略 |
|---|---|---|
| [`WORKFLOW.md`](WORKFLOW.md) | 修改仓库文件时怎样从接手任务走到提交评审 | 每个修改任务读取 |
| [`projectneed.md`](projectneed.md) | 系统必须保持什么 | 公共章节先读，再按领域展开 |
| [`NOW.md`](NOW.md) | 当前还有什么没做 | 每个非简单任务读取；完成项不应存在 |
| [`decisions/README.md`](decisions/README.md) | 哪些重要选择已经作出 | 先查索引，只展开相关记录 |
| [`design/`](design/) | 某个问题的真实链路、方案和验收 | 任务命中时读取，不把 proposed 设计当已实现事实 |
| [`writing-rules.md`](writing-rules.md) | 文档应该写到哪里、怎样避免漂移 | 新增或修改文档时读取 |
| [`templates/`](templates/) | 怎样写任务合同、变更声明和交接 | 复杂或高风险任务按需复制 |
| `_handbook/` | 历史专题说明和操作材料 | 只作线索；必须用当前代码和本索引复核 |

冲突时按下面的顺序处理：

1. 用户当前明确指令拥有本次任务最高优先级，但不能被扩大解释。
2. `projectneed.md` 规定长期目标和不变量。
3. accepted 决策记录解释当前选择；后续勘误优先于被取代记录。
4. `NOW.md` 说明当前未完成工作，不能重定义长期语义。
5. 代码、配置、数据库和测试证明“现在是什么”，不自动证明“本来就应该这样”。
6. 旧 handbook、旧会话和历史记忆只提供调查线索。

如果第 2～5 项互相冲突，先写明冲突对象、当前行为、目标行为和可能影响，再向维护者核对。不得挑一个最方便实现的版本继续。

## 4. 按任务选择阅读路径

| 任务 | 必读顺序 | 随后检查的真实入口 |
|---|---|---|
| 任何会修改仓库文件的任务 | 本索引 → [`WORKFLOW.md`](WORKFLOW.md) → 下方对应领域 | 当前分支、目标分支、完整 diff、验证报告 |
| Web/Mobile 共享 Session、Akashic Channel 或客户端 adapter | `projectneed` AKC-001～AKC-003 → [0044](decisions/0044-akashic-channel-uses-web-and-mobile-adapters.md) → [Akashic Channel 与客户端 Adapter 规格](design/akashic-channel-client-adapters.md) → [持久化状态地图](design/persistence-state-map.md) | `session/`、`infra/channels/`、`infra/mobile_realtime/`、`bootstrap/app.py`、`bootstrap/chat_api.py`、`plugins/scheduler/`、`plugins/wake/`、`plugins/akasha/`；实现仍需独立授权 |
| 其他 Akashic 产品路线、Project Session、Project Akasha 或大型 tool result | [未来方向与 Issue 拆分草案](design/akashic-future-roadmap-issue-drafts.md) → 草案中对应领域的现行条款、决策与设计 | `session/`、`agent/scheduler.py`、`plugins/eventmail/`、`plugins/wake/`、`plugins/drift/`、`plugins/akasha/`、`agent/tools/message_push.py`；草案未提升为现行合同前不得直接实现 |
| Prompt、人格、上下文窗口、历史裁切、重试 | `projectneed` 第 5～7、13 节 → [Veda 人格设计](design/veda-persona.md) → [0002](decisions/0002-context-reduction-is-a-nondestructive-projection.md) → [0030](decisions/0030-session-context-compaction-ledger.md) → [Session compaction ledger](design/session-context-compaction-ledger.md) → [上下文事故设计](design/project-workbook-and-semantic-safety.md) → [Wake 最近主动消息上下文](design/wake-recent-delivery-context.md) | `agent/persona.py`、`agent/core/prompt_block.py`、`agent/core/passive_turn.py`、`agent/prompting/`、`session/manager.py`、`session/store.py` |
| 会话、消息、turn、同 Turn 输入、打断、附件、删除或恢复 | `projectneed` 第 6～7、11～13 节 → [持久化状态地图](design/persistence-state-map.md) → [Codex 式同 Turn 输入需求](design/codex-style-same-turn-input-requirements.md) → [Codex 式同 Turn 输入设计](design/codex-style-same-turn-input.md) → [0025](decisions/0025-codex-style-same-turn-input.md) | `agent/control/runtime.py`、`agent/core/passive_turn.py`、`bootstrap/passive_worker.py`、`session/`、`infra/channels/base.py`、`bootstrap/channels.py`、`bootstrap/chat_api.py` |
| Markdown 记忆、退役 Memory2 归档、Akasha | `projectneed` 第 6、8、11～13 节 → [0041](decisions/0041-turn-effects-and-memory-plugins-are-orthogonal.md) → [0006](decisions/0006-akasha-v2-is-the-canonical-explicit-memory-engine.md) → [Akasha V2 在线与重放](design/akasha-v2-runtime-migration.md) → [持久化状态地图](design/persistence-state-map.md) | `bootstrap/memory.py`、`core/memory/markdown.py`、`plugins/akasha/` |
| 主动流程、Wake、Drift、调度 | `projectneed` 第 6、9、12～13 节 → [持久化状态地图](design/persistence-state-map.md) → [Wake 最近主动消息上下文](design/wake-recent-delivery-context.md) → [Content / Wake 现有原子能力与第一阶段](design/content-wake-existing-atoms-first-stage.md) → [Content / Wake / Proactive 分层任务合同](design/content-wake-proactive-migration-task-contract.md) → [0040](decisions/0040-wake-duty-gate-lives-in-scoped-react.md) → [0048](decisions/0048-eventmail-keeps-three-mail-lifecycles.md) | `plugins/eventmail/`、`plugins/wake/`、`plugins/drift/`、`plugins/scheduler/`、`plugins/subagent/`、`agent/plugin_composition/timers.py`、`agent/plugin_composition/background_jobs.py` |
| React Core 原子能力、Scheduler/Subagent 非特权插件 | `projectneed` 第 6、9～13 节 → [0034](decisions/0034-turn-is-the-logical-work-unit.md) → [0039](decisions/0039-react-core-atoms-keep-sources-unprivileged.md) → [React Core 与 Scheduler/Subagent 设计](design/react-core-scheduler-subagent.md) → [分阶段任务合同](design/react-core-scheduler-subagent-task-contract.md) → [持久化状态地图](design/persistence-state-map.md) | `agent/looping/core.py`、`agent/core/passive_turn.py`、`agent/lifecycle/`、`agent/scheduler.py`、`agent/background/subagent_manager.py`、`agent/plugin_composition/`、`agent/plugins/snapshot.py` |
| 正式启动、Supervisor、自重启、停止信号 | `projectneed` RUN-001～RUN-004 → [Linux Supervisor 安全自重启提议](design/linux-supervisor-safe-self-restart.md) → [`docker/debug/README.md`](../docker/debug/README.md) | `main.py`、`agent/supervisor.py`、`agent/restart.py`、`agent/tools/agent_restart.py`、`scripts/stop-runtime.sh`、restart Gate 报告 |
| 容器、云主机运行适配、Host Bridge、hua-home迁移 | `projectneed` RUN-013～RUN-015、WSP-005、SH-001～SH-003 → [0032](decisions/0032-host-bridge-preserves-host-equivalent-execution.md) → [容器与 Linux 主机运行适配设计](design/akashic-container-cloud-runtime-adaptation.md) → [Core 与 Host Bridge 安装设计](design/akashic-core-bridge-installer.md) → [非迁移实验合同](design/akashic-container-host-bridge-experiment-contract.md) → [Unified Shell Execution 设计](design/unified-shell-execution.md) → [持久化状态地图](design/persistence-state-map.md) | exact-commit 安装、执行后端、runtime identity、插件安装链、Supervisor 与隔离实验；正式迁移前先运行 plan-only 清单并取得独立批准 |
| Provider、模型角色、运行时切换、usage、首次配置、模型普通插件化 | `projectneed` RUN-005～RUN-012、ONB-001、CTX-001 → [0050](decisions/0050-model-revision-lives-in-ordinary-plugin.md) → [模型普通插件与 Provider 组合规格](design/model-plugin-ordinary-capability-spec.md) → [0027](decisions/0027-runtime-models-use-generation-leases.md) → [0028](decisions/0028-model-credentials-live-with-workspace-connections.md) → [现行实现与历史验收基线](design/runtime-model-registry-and-onboarding.md) → [持久化状态地图](design/persistence-state-map.md) | `agent/model_runtime/`、`agent/provider.py`、`bootstrap/providers.py`、`bootstrap/settings_api.py`、`bootstrap/tools.py`、`agent/plugin_composition/`、`plugins/akasha/`、`plugins/scheduler/`、`plugins/subagent/`、`plugins/wake/`、`frontend/chat/src` |
| 插件安装、热重载、自验证、Cordis 迁移、plugin-data、Skill、Drift skill、MCP | `projectneed` 第 6、9～13 节 → [0008](decisions/0008-plugin-runtime-publishes-only-committed-snapshots.md) → [0024](decisions/0024-plugin-self-validation-uses-stable-and-latest.md) → [0026](decisions/0026-plugin-rollout-is-owned-by-the-parent-turn.md) → [0036](decisions/0036-plugin-composition-keeps-promotion-owner.md) → [0038](decisions/0038-operator-trust-can-publish-offline-plugin-batches.md) → [0042](decisions/0042-plugin-diagnostics-preserve-domain-owners.md) → [0046](decisions/0046-plugin-candidate-validation-is-incremental.md) → [插件 install/uninstall/revert turn 边界发布合同](design/plugin-install-uninstall-turn-boundary-rollout.md) → [插件递归自验证运行时设计](design/recursive-plugin-self-validation.md) → [Cordis 插件迁移能力等价验收](design/cordis-plugin-capability-parity.md) → [插件 v3 最终迁移地图](design/plugin-v3-final-migration-map.md) → [插件 v3 生产替代清单](design/plugin-v3-production-readiness-checklist.md) → [插件 v3 admission/lifecycle 收口合同](design/plugin-v3-admission-lifecycle-closeout-task-contract.md) → [插件 v3 generation metadata 收口合同](design/plugin-v3-generation-metadata-task-contract.md) → [插件 v3 Runtime Inspection 合同](design/plugin-v3-runtime-inspection-task-contract.md) → [插件 v3 committed command catalog 合同](design/plugin-v3-command-catalog-task-contract.md) → [插件 v3 MCP/managed process 合同](design/plugin-v3-mcp-managed-process-task-contract.md) → [插件组合内核第一阶段任务合同](design/plugin-composition-kernel-task-contract.md) → [插件事件与同步执行能力任务合同](design/plugin-event-executor-task-contract.md) → [插件 TopologyView 任务合同](design/plugin-topology-view-task-contract.md) → [插件 lifecycle 接入点任务合同](design/plugin-lifecycle-seam-task-contract.md) → [Turn committed typed event 合同](design/plugin-turn-committed-event-task-contract.md) → [插件 v3 generation loader 任务合同](design/plugin-v3-loader-task-contract.md) → [插件 stable 原子组装任务合同](design/plugin-stable-atomic-assembly-task-contract.md) → [插件 candidate Root 隔离任务合同](design/plugin-candidate-root-isolation-task-contract.md) → [插件组合结构身份与 revision 任务合同](design/plugin-composition-revision-task-contract.md) → [插件组合 Health/Incident/Validation 任务合同](design/plugin-composition-health-incident-task-contract.md) → [插件 Transform/Observe 事件任务合同](design/plugin-transform-observe-task-contract.md) → [插件 generation 数据根任务合同](design/plugin-data-root-task-contract.md) → [插件 Tool 组合事件任务合同](design/plugin-tool-composition-events-task-contract.md) → [插件 Tool v3 迁移组合 Gate 任务合同](design/plugin-tool-v3-migration-gate-task-contract.md) → [Citation + Meme 纯 v3 组合 Gate](design/plugin-passive-composition-v3-gate-task-contract.md) → [持久化状态地图](design/persistence-state-map.md) | `agent/plugins/base.py`、`agent/plugins/install.py`、`agent/plugins/manager.py`、`agent/plugins/snapshot.py`、`agent/plugins/reload_journal.py`、`agent/plugins/turn_rollout.py`、`agent/plugins/skill_links.py`、`agent/control/runtime.py`、`agent/looping/core.py`、`agent/mcp/host.py`、`agent/mcp/client.py`、`agent/plugins/service_host.py`、`agent/plugin_composition/context.py`、`agent/plugin_composition/effect.py`、`bootstrap/app.py`、`utils/process_group.py` |
| Core 领域 Observe 事件 | [Core 领域 Observe 事件合同](design/plugin-domain-observe-events-task-contract.md) → [插件 Transform 与 Observe 事件任务合同](design/plugin-transform-observe-task-contract.md) | `agent/turn_events/observe.py`、`agent/lifecycle/composition.py`、`bus/event_bus.py`、`agent/retrieval/default_pipeline.py` |
| 插件 v3 包级 Skill/Drift skill/Dashboard 声明 | [插件 v3 generation loader 任务合同](design/plugin-v3-loader-task-contract.md) → [插件 v3 包级 contribution 任务合同](design/plugin-v3-package-contributions-task-contract.md) | `agent/plugins/composable.py`、`agent/plugins/manager.py`、`agent/plugins/generation.py` |
| 插件只读既有 Session 投影 | [持久化状态地图](design/persistence-state-map.md) → [插件 Session Read 组合能力任务合同](design/plugin-session-read-service-task-contract.md) | `agent/plugin_composition/session_read.py`、`agent/plugins/manager.py`、`session/manager.py` |
| 插件 v3 Dashboard 注册与数据边界 | [插件 v3 包级 contribution 任务合同](design/plugin-v3-package-contributions-task-contract.md) → [v3 DashboardContext 任务合同](design/plugin-v3-dashboard-context-task-contract.md) | `agent/plugin_composition/dashboard.py`、`agent/plugins/dashboard_host.py` |
| 插件 v3 静态投影与 exact Root runtime | [v3 DashboardContext 任务合同](design/plugin-v3-dashboard-context-task-contract.md) → [静态投影与 exact runtime 任务合同](design/plugin-v3-static-projection-runtime-task-contract.md) | `agent/plugin_composition/model.py`、`agent/plugins/composable.py`、`agent/plugins/snapshot.py`、`agent/plugins/dashboard_host.py` |
| Akasha v3、feedback、Inspector 与 Mobile recall | [持久化状态地图](design/persistence-state-map.md) → [Akasha 在线与重放](design/akasha-v2-runtime-migration.md) → [Akasha v3 迁移任务合同](design/akasha-plugin-v3-migration-task-contract.md) | `plugins/akasha/`、`core/memory/plugin.py`、`agent/plugin_composition/runtime_services.py`、`agent/plugins/manager.py` |
| Citation/Meme v3 被动回复组合接入点 | [持久化状态地图](design/persistence-state-map.md) → [lifecycle 接入点任务合同](design/plugin-lifecycle-seam-task-contract.md) → [candidate Root 隔离任务合同](design/plugin-candidate-root-isolation-task-contract.md) → [v3 被动回复组合接入点任务合同](design/plugin-v3-passive-response-seams-task-contract.md) → [纯 v3 组合 Gate](design/plugin-passive-composition-v3-gate-task-contract.md) → [WebUI E2E Gate](design/plugin-passive-webui-v3-e2e-task-contract.md) | `agent/lifecycle/types.py`、`agent/lifecycle/phases/after_reasoning.py`、`agent/plugin_composition/model.py`、`agent/plugins/composable.py`、`agent/plugins/manager.py`、`agent/plugins/dashboard_host.py`、`bootstrap/chat_api.py`、`bootstrap/web_shell.py` |
| 插件 v3 prepared context、只读 Memory runtime 与显式 interaction 撤销 | [插件 lifecycle 接入点任务合同](design/plugin-lifecycle-seam-task-contract.md) → [context-prepared 与 Memory capability 任务合同](design/plugin-context-prepared-memory-capability-task-contract.md) → [Plugin Undo v3 与 interaction 撤销协调合同](design/plugin-v3-interaction-undo-task-contract.md) | `agent/lifecycle/composition.py`、`agent/lifecycle/phases/before_turn.py`、`agent/plugin_composition/runtime_services.py`、`agent/plugin_composition/interaction_undo.py`、`agent/plugins/interaction_undo.py`、`agent/plugins/manager.py`、`session/store.py` |
| 插件 v3 background job、Agent Work 与 LLM capability | [插件 v3 Proactive / background job 历史合同](design/plugin-v3-proactive-jobs-task-contract.md) → [插件 v3 Agent Work 能力合同](design/plugin-v3-agent-work-capability-task-contract.md) → [插件 v3 MCP/managed process 合同](design/plugin-v3-mcp-managed-process-task-contract.md) → [持久化状态地图](design/persistence-state-map.md) | `agent/plugin_composition/background_jobs.py`、`agent/plugins/generation_job_host.py`、`agent/tools/registry.py`、`agent/plugins/manager.py`、`agent/control/`、`bootstrap/control_execution.py` |
| 插件 v3 inbound/outbound channel capability | [插件 v3 Channel capability 合同](design/plugin-v3-channel-capability-task-contract.md) → [插件 v3 Channel 附件持久化合同](design/plugin-v3-channel-attachment-task-contract.md) → [插件 v3 committed command catalog 合同](design/plugin-v3-command-catalog-task-contract.md) | `agent/plugin_composition/`、`agent/plugins/manager.py`、`agent/plugins/snapshot.py`、`agent/tools/message_push.py`、`agent/looping/core.py`、`agent/core/passive_turn.py`、`agent/lifecycle/phases/after_turn.py`、`agent/turns/orchestrator.py`、`agent/turns/outbound.py`、`bootstrap/app.py`、`bootstrap/channel_host.py`、`bootstrap/channels.py`、`bootstrap/passive_worker.py`、`bus/queue.py`、`bus/events.py`、`infra/channels/base.py`、`infra/channels/contract.py`、`infra/channels/delivery.py`、`infra/channels/telegram_channel.py`、`infra/channels/qq_channel.py`、`infra/channels/web_chat_channel.py`、`infra/mobile_realtime/channel.py`、`session/manager.py`、`session/store.py`、`/mnt/data/coding/akashic-plugin/feishu`、`/mnt/data/coding/akashic-plugin/qqbot` |
| 移动端查看 Markdown、定时任务、插件、Skill、MCP | `projectneed` 第 6、10～13 节 → [移动端运行时检查](design/mobile-runtime-inspection.md) → [v3 Mobile UI/query capability](design/plugin-v3-mobile-ui-query-task-contract.md) → [持久化状态地图](design/persistence-state-map.md) | `infra/mobile_realtime/runtime_inspection.py`、`infra/mobile_realtime/protocol.py`、`infra/mobile_realtime/channel.py`、`agent/plugins/mobile_ui.py` |
| Workspace、配置、凭据、迁移、备份 | `projectneed` 第 6、11～13 节 → [持久化状态地图](design/persistence-state-map.md) → [0021](decisions/0021-yoyo-workspace-ledger-defines-migration-origin.md) → [Yoyo 迁移维护手册](design/git-migration-authoring.md) | `main.py`、`bootstrap/init_workspace.py`、`agent/config.py`、`agent/migrations/`、`migrations/yoyo/`、`agent/model_runtime/auth/store.py`、`scripts/rolling_backup.py` |
| 高风险 refactor、语义不变重构、CI oracle | `projectneed` 第 4～6、13、15 节 → [综合重构账本](refactor/clean-code-ledger.md) → [上下文事故设计](design/project-workbook-and-semantic-safety.md) → 相关决策 | 改动前后的完整 diff、semantic tests、write set、故障注入 |
| 变更影响 Gate、跨仓库插件契约 | `projectneed` 第 10、13、15 节 → [0004](decisions/0004-cross-repository-evidence-is-an-immutable-combination.md) → [移动端与跨仓库 Gate](design/mobile-cross-repository-semantic-gate.md) → [Gate 总体设计](spark/2026-07-16-change-impact-contract-gate.md) → [持久化状态地图](design/persistence-state-map.md) | `tests_scenarios/contracts/`、`docker/debug/gate.py`、`private_runtime/` |
| Companion 安全、容量和长时运行 Edge Case | `projectneed` SEC-001～SEC-010 → [0017](decisions/0017-one-person-companion-security-boundary.md) → [Companion 安全边界与 Edge Case 实施设计](design/security-scan-edge-cases.md) → [持久化状态地图](design/persistence-state-map.md) | 相关 D1～D9 owner、`tests_scenarios/contracts/`、`docker/debug/gate.py` |
| Harness benchmark、独立 runtime trial、证据驱动优化 | `projectneed` 第 8～13、15 节 → [V4 Flash Harness Benchmark 设计](spark/2026-07-30-v4flash-harness-benchmark-design.md) → [Benchmark 诊断循环设计](spark/2026-07-30-agent-benchmark-diagnostic-loop-design.md) → [0010](decisions/0010-provider-default-output-and-benchmark-diagnostics.md) → [0011](decisions/0011-benchmark-concurrency-six.md) → [实验 ledger](benchmark/v4flash-harness-experiment-ledger.md) → [运行审计](benchmark/terminalbench-2.1-run-audit-2026-08-05.md) → [逐题 CSV](benchmark/terminalbench-2.1-case-results-2026-08-05.csv) → [持久化状态地图](design/persistence-state-map.md) | `benchmark/harbor_v4flash/`、`agent/control/`、`bootstrap/control_execution.py`、`docker/debug/`、独立 artifact store 与 experiment ledger |
| Shell、长任务、PTY、进程续接或轮询 | `projectneed` SH-001、RUN-002～RUN-003、ERR-001 → [0014](decisions/0014-shell-uses-unified-execution.md) → [Unified Shell Execution 设计](design/unified-shell-execution.md) | `agent/tools/shell.py`、`agent/tools/unified_exec.py`、`agent/tools/meta/register.py`、`agent/background/subagent_profiles.py`、`bootstrap/tools.py` |
| 移动端、客户端协议、跨仓库 runtime patch 或 stacked PR 评审 | `projectneed` MOB-001～MOB-008、AKC-001～AKC-003、GOV-001～GOV-005、TST-001～TST-008 → [0003](decisions/0003-core-capability-ownership-is-semantic.md) → [0004](decisions/0004-cross-repository-evidence-is-an-immutable-combination.md) → [0035](decisions/0035-mobile-protocol-delivery-is-phased.md) → [0044](decisions/0044-akashic-channel-uses-web-and-mobile-adapters.md) → [移动端与跨仓库 Gate](design/mobile-cross-repository-semantic-gate.md) → [移动端投影审计](design/mobile-projection-audit.md) → [`templates/review-contract.md`](templates/review-contract.md) | 每层 `base..head`、最终累计 diff、所有 schema lineage、协议 source、runtime/provider/scenario identity 和设备隔离证据 |
| 新增或修改项目文档 | 本索引 → [`writing-rules.md`](writing-rules.md) → 目标文档的权威上游 | 所有相对链接、重复规则、过时入口和 Git diff |
| Dashboard、Chat UI | `projectneed` 公共合同、WEBUI-001～WEBUI-007 → [0018](decisions/0018-chat-webui-has-one-source-and-two-adapters.md) → [0022](decisions/0022-mobile-webui-uses-server-selected-generations.md) → [0029](decisions/0029-main-gateway-reconciles-mobile-webui-stable.md) → [0043](decisions/0043-paper-brand-tokens-replace-material-visual-semantics.md) → [纸张品牌系统](design/akashic-paper-brand-system.md) → [共享对话 WebUI](design/shared-chat-webui.md) → [Mobile Browser Lab](design/mobile-browser-lab.md) → [WebUI 交互性能与组件边界优化](design/webui-interaction-optimization.md) → [服务端发布的移动 WebUI OTA](design/server-published-mobile-webui.md) → `NOW.md` 对应事项 | `frontend/**/src`、真实构建和渲染结果 |
| 2236 顶层页面、导航与递归插件 UI | `projectneed` WEBUI-001～WEBUI-007、PLG-001～PLG-016 → [0051](decisions/0051-web-ui-composes-ordinary-plugin-modules.md) → [0050](decisions/0050-model-revision-lives-in-ordinary-plugin.md) → [模型普通插件与 Provider 组合规格](design/model-plugin-ordinary-capability-spec.md) → [2236 WebUI 插件组合设计](design/web-ui-plugin-composition.md) → [共享对话 WebUI](design/shared-chat-webui.md) → [服务端发布的移动 WebUI OTA](design/server-published-mobile-webui.md) | `frontend/dashboard/src`、`frontend/chat/src`、`agent/plugin_composition/`、`agent/plugins/`、`plugins/models/`、`plugins/openai_compatible/`、`plugins/codex/`、`plugins/opencode_go/` |

任务同时命中两行以上、会修改持久数据或会产生外部不可逆效果时，读取 `projectneed.md` 全文。执行阶段可以收窄材料，评审阶段必须展开所有相关 diff、状态变化和证据。

Skill/MCP 任务固定从插件安装链进入：插件 source → `skill_roots` / `drift_skill_roots` / `mcp_servers` → 安装 staging 与 runtime 准备 → generation catalog → workspace 投影。独立 workspace MCP 声明、watcher/admin 和手工 MCP 目录不再是运行时入口；不能从已删除的旧 API 反推新的能力安装设计。

## 5. 持久化任务的强制前置读取

只要任务中出现下列任一对象或动作，先读 [`design/persistence-state-map.md`](design/persistence-state-map.md)：

- `sessions.db`、`memory2.db`、`akasha.db`、`proactive.db`、`wake_proactive.db`、`drift.db`。
- `MEMORY.md`、`SELF.md`、`PENDING.md`、`PROACTIVE_CONTEXT.md`。
- 附件、plugin-data、插件 Skill/MCP、旧 workspace MCP/skill 兼容路径、调度、quota、凭据或 workspace 迁移。
- 裁切、压缩、清理、归档、替换、重建、同步、恢复、删除、卸载或备份。

读取后先回答六个问题。答案必须描述数据怎样变化，不能只列文件路径：

1. 被操作的是权威事实、运行连续性、派生索引、诊断证据，还是临时控制文件？
2. 正常运行通过什么事件增加哪些行、记录或文件？
3. 哪些字段可以原位更新，哪些变化只是 supersede、消费或终态等逻辑变化？
4. 什么用户动作、提交协议或进程生命周期事件才允许物理删除、覆盖或 cascade；当前调用者为什么拥有该权限？
5. 这次允许改变哪些行、文件和外部效果，哪些必须保持逐项相同？
6. 失败、取消或进程重启后，哪个恢复点能证明数据仍然存在？

会话任务还要先使用一个固定判断：`sessions.db/messages` 在正常运行中只 INSERT 新消息。只有用户主动撤销消息或删除会话，独立的数据管理命令才可以减少既有正文。当前代码虽然存在 `update_message`，但旧消息编辑是否允许原位 UPDATE 仍要按状态地图向维护者核对。`sessions` 元数据、`turns` 状态和派生索引有各自的更新协议，不能用它们可更新这一事实推翻消息正文的只追加合同。

状态地图中的“代码事实”可以直接用于定位。“意图推断”和“待确认问题”必须由维护者确认后，才能写入 `projectneed.md`、删除策略、迁移脚本或备份排除规则。

## 6. 开工时要形成的最小任务合同

普通局部任务只需在脑内或一段简短说明中回答；复杂任务使用 [`templates/agent-task-contract.md`](templates/agent-task-contract.md)：

- 目标：用户最终能观察到什么结果。
- 完成标准：什么证据出现后可以停止。
- `change_type` 与 `semantic_delta`。
- 允许变化：本次明确授权改变的行为和状态。
- 受保护状态：即使实现更方便也不能改变的对象。
- 允许副作用：文件、数据库、进程、网络和消息发送范围。
- 关键未知：哪些歧义会改变持久化结果、权限或兼容性。
- 验证与回滚：怎样独立判断成功，失败后恢复到哪里。

这一结构来自 [OpenAI · Prompting guidance for GPT-5.6](https://developers.openai.com/api/docs/guides/prompt-guidance-gpt-5p6) 的结果优先、完成标准、批准边界、工具前置条件和停止规则，并按本项目的持久化风险收窄。不要把指南全文复制进任务 prompt；只保留会改变当前任务行为的约束。

## 7. 调查、设计、实现和验收不能自动跨层

先判断用户授权的是哪一层：

| 当前层 | 默认可做 | 不自动做 |
|---|---|---|
| 调查 | 读代码、配置、日志、schema、Git 和只读数据 | 改实现、改数据库、发布或发送 |
| 设计 | 写问题定义、owner、数据流、方案、风险和验收 | 把 proposed 方案当已批准语义 |
| 实现 | 修改已授权范围内的本地文件并做非破坏性验证 | 外部发布、破坏性迁移、显著扩展范围 |
| 评审 | 检查 diff、测试、write set、权限和副作用 | 顺手重构被评审代码 |
| 外部协调 | 按明确授权提交、推送、发消息或部署 | 替用户作未授权决定 |

长任务只在主要阶段变化时汇报。上下文压缩只在调查完成、设计确定、实现完成或验证完成等里程碑进行，并使用 [`templates/context-handoff.yaml`](templates/context-handoff.yaml) 保留已核对事实和引用。压缩摘要是任务状态，不是新的需求来源。

## 8. 当前工作手册文件树

```text
docs/
├── INDEX.md
├── WORKFLOW.md
├── projectneed.md
├── NOW.md
├── writing-rules.md
├── decisions/
│   ├── README.md
│   ├── 0001-project-workbook-is-shared-reality.md
│   ├── 0002-context-reduction-is-a-nondestructive-projection.md
│   ├── 0003-core-capability-ownership-is-semantic.md
│   ├── 0004-cross-repository-evidence-is-an-immutable-combination.md
│   ├── 0005-git-cursor-drives-one-shot-migrations.md
│   ├── 0006-akasha-v2-is-the-canonical-explicit-memory-engine.md
│   ├── 0007-mobile-plugin-control-and-data-planes-are-explicit.md
│   ├── 0008-plugin-runtime-publishes-only-committed-snapshots.md
│   ├── 0009-akasha-mobile-recall-preserves-semantic-lanes.md
│   ├── 0010-provider-default-output-and-benchmark-diagnostics.md
│   ├── 0011-benchmark-concurrency-six.md
│   ├── 0012-query-local-compaction-is-a-persisted-projection.md
│   ├── 0013-linux-supervisor-uses-one-boot-guardian.md
│   ├── 0014-shell-uses-unified-execution.md
│   ├── 0015-cleanup-does-not-own-turn-or-restart-finality.md
│   ├── 0016-channel-delivery-uses-complete-logical-messages.md
│   ├── 0017-one-person-companion-security-boundary.md
│   ├── 0018-chat-webui-has-one-source-and-two-adapters.md
│   ├── 0019-mobile-long-messages-use-bounded-events.md
│   ├── 0020-mobile-history-content-uses-authenticated-http-ranges.md
│   ├── 0021-yoyo-workspace-ledger-defines-migration-origin.md
│   ├── 0022-mobile-webui-uses-server-selected-generations.md
│   ├── 0023-akashic-tokens-own-material-3-semantics.md
│   ├── 0024-plugin-self-validation-uses-stable-and-latest.md
│   ├── 0025-codex-style-same-turn-input.md
│   ├── 0026-plugin-rollout-is-owned-by-the-parent-turn.md
│   ├── 0027-runtime-models-use-generation-leases.md
│   ├── 0028-model-credentials-live-with-workspace-connections.md
│   ├── 0029-main-gateway-reconciles-mobile-webui-stable.md
│   ├── 0030-session-context-compaction-ledger.md
│   ├── 0031-stable-matching-head-allows-gateway-restart.md
│   ├── 0032-host-bridge-preserves-host-equivalent-execution.md
│   ├── 0033-local-agent-instructions-are-not-project-documents.md
│   ├── 0034-turn-is-the-logical-work-unit.md
│   ├── 0035-mobile-protocol-delivery-is-phased.md
│   ├── 0036-plugin-composition-keeps-promotion-owner.md
│   ├── 0037-plugin-runtime-is-pure-v3.md
│   ├── 0038-operator-trust-can-publish-offline-plugin-batches.md
│   ├── 0039-react-core-atoms-keep-sources-unprivileged.md
│   ├── 0040-wake-duty-gate-lives-in-scoped-react.md
│   ├── 0041-turn-effects-and-memory-plugins-are-orthogonal.md
│   ├── 0042-plugin-diagnostics-preserve-domain-owners.md
│   ├── 0043-paper-brand-tokens-replace-material-visual-semantics.md
│   ├── 0044-akashic-channel-uses-web-and-mobile-adapters.md
│   ├── 0045-akashic-direct-messages-commit-before-notify.md
│   ├── 0046-plugin-candidate-validation-is-incremental.md
│   ├── 0047-provides-may-bind-one-tool.md
│   ├── 0048-eventmail-keeps-three-mail-lifecycles.md
│   ├── 0049-wake-content-is-a-decaying-eventmail-pool.md
│   ├── 0050-model-revision-lives-in-ordinary-plugin.md
│   └── 0051-web-ui-composes-ordinary-plugin-modules.md
├── design/
│   ├── akasha-v2-runtime-migration.md
│   ├── akashic-channel-client-adapters.md
│   ├── akashic-paper-brand-system.md
│   ├── akashic-future-roadmap-issue-drafts.md
│   ├── akashic-container-cloud-runtime-adaptation.md
│   ├── akashic-container-host-bridge-experiment-contract.md
│   ├── linux-supervisor-safe-self-restart.md
│   ├── mobile-cross-repository-semantic-gate.md
│   ├── mobile-long-message-delivery.md
│   ├── mobile-browser-lab.md
│   ├── mobile-projection-audit.md
│   ├── codex-style-same-turn-input-requirements.md
│   ├── codex-style-same-turn-input.md
│   ├── cordis-plugin-capability-parity.md
│   ├── plugin-composition-kernel-task-contract.md
│   ├── plugin-event-executor-task-contract.md
│   ├── plugin-topology-view-task-contract.md
│   ├── plugin-lifecycle-seam-task-contract.md
│   ├── plugin-v3-loader-task-contract.md
│   ├── project-workbook-and-semantic-safety.md
│   ├── query-local-react-compaction.md
│   ├── runtime-model-registry-and-onboarding.md
│   ├── server-published-mobile-webui.md
│   ├── shared-chat-webui.md
│   ├── unified-shell-execution.md
│   ├── veda-persona.md
│   ├── web-ui-plugin-composition.md
│   ├── webui-interaction-optimization.md
│   ├── persistence-state-map.md
│   ├── programmatic-session-memory-exclusion.md
│   ├── recursive-plugin-self-validation.md
│   ├── react-core-scheduler-subagent.md
│   ├── react-core-scheduler-subagent-task-contract.md
│   └── wake-recent-delivery-context.md
├── spark/
│   ├── 2026-07-16-change-impact-contract-gate.md
│   ├── 2026-07-21-web-settings-provider-switching-design.md
│   ├── 2026-07-21-git-backed-one-shot-migrations-design.md
│   ├── 2026-07-30-v4flash-harness-benchmark-design.md
│   └── 2026-07-30-agent-benchmark-diagnostic-loop-design.md
├── refactor/
│   ├── clean-code-ledger.md
│   └── 20260808-context-compaction-fixes.md
└── templates/
    ├── agent-task-contract.md
    ├── change-intent.yaml
    ├── context-handoff.yaml
    ├── decision-record.md
    ├── review-contract.md
    └── semantic-oracle-checklist.md
```

新增文件前先判断现有文件能否承担该职责。必须新增时，把它放进上面的骨架，更新本索引和所有入站链接；不要再创建第二个“总说明”“最新状态”或“完整需求”。

## 9. 索引维护验收

修改项目工作手册后至少检查：

1. 所有索引路径存在，相对链接可以解析。
2. 新会话能从本文件找到执行工作流、需求、当前事项、决策、相关设计和代码入口。
3. 索引没有复制需求正文，也没有出现与 `projectneed.md` 竞争的规则版本。
4. `NOW.md` 没有已完成流水账。
5. proposed 设计、代码事实和维护者已确认意图有明确标签。
6. 历史 handbook 如果与当前实现冲突，入口处有醒目提示或已经完成勘误。
