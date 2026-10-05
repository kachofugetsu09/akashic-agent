# Akashic Agent 项目阅读索引

每个新会话先读本文件，再沿相关入口核对需求、决定和真实实现。索引只负责路由；执行步骤见 [WORKFLOW.md](WORKFLOW.md)。

## 1. 固定入口

1. 每个任务必读 [设计原则](projectneed.md#设计原则每个任务必读)，先理解项目允许改变什么。
2. 修改任务读 [WORKFLOW.md](WORKFLOW.md)；非简单任务读 [NOW.md](NOW.md)，只带入相关未完成事项。
3. 按下表选择相关需求与设计，并在 [决策索引](decisions/README.md) 按行为、数据对象和 owner 查找决定。读取命中的记录，跟随勘误和替代关系；不能只按准备修改的文件名选择材料。
4. 核对真实调用路径、状态读写和消费者。改动跨 owner 时展开所有受影响边界；不因命中多行而机械读取全部历史。

例如“缩小模型窗口”必须读 CTX-001、SES-005 和 [0002](decisions/0002-context-reduction-is-a-nondestructive-projection.md)，即使任务没有出现“数据库”一词。删除源码也要查持久数据、动态入口和插件消费者，不能由“没有静态调用”推出可删除。

用户当前明确指令决定本次授权范围；`projectneed.md` 定义长期语义，accepted 决策说明理由，后续勘误优先于被替代记录。代码证明现状，不能反推用户意图。发现冲突先说明当前行为、既定要求与影响，不自行挑选或改写规格。proposed 设计、历史会话与 `_handbook/` 只提供线索。

历史设计中的评审模型、Gate、单测和任务表格不再定义开发流程；按 [0083](decisions/0083-short-workflow-preserves-design-intent.md) 使用当前 WORKFLOW。历史产品约束仍须按其有效决策核对。

## 2. 文档各自回答什么

| 入口 | 用途 |
|---|---|
| [projectneed.md](projectneed.md) | 设计原则、长期需求和禁止事项 |
| [WORKFLOW.md](WORKFLOW.md) | 怎样理解边界、实现验证和交付 PR |
| [NOW.md](NOW.md) | 尚未完成的工作与接手限制 |
| [decisions/README.md](decisions/README.md) | 为什么这样做、为什么不选另一方案 |
| [design/](design/) | 问题级调用链、迁移、失败与验收；先看状态和勘误 |
| [writing-rules.md](writing-rules.md) | 文档如何维护；改文档时读取 |
| [templates/](templates/) | 复杂任务与交接按需使用，不是每次必填表 |

## 3. 按行为选择阅读路径

下表给出当前入口；继续沿需求和决定中的相关链接展开，不把所有旧任务合同当作开工清单。

| 涉及的行为或状态 | 需求与设计入口 | 真实实现入口 |
|---|---|---|
| 窗口、摘要、裁切、历史加载、重试 | CTX、SES-005 → [0002](decisions/0002-context-reduction-is-a-nondestructive-projection.md) → [0030](decisions/0030-session-context-compaction-ledger.md)、[上下文设计](design/session-context-compaction-ledger.md) | `plugins/context/`、`plugins/compaction/`、`session/` |
| Prompt 人格与主动消息上下文 | CTX、PRM → [人格设计](design/veda-persona.md)、[Wake 最近送达](design/wake-recent-delivery-context.md) | `agent/prompting/`、`plugins/wake/` |
| 长工具结果折叠、原文回读 | CTX-008 → [0081](decisions/0081-content-views-keep-original-messages.md) | `plugins/content/`、`plugins/context/` |
| Message、Turn、来源、回复和送达 | SES、OUT → [消息设计](design/0902-reviewed-v4.md)、[故障恢复](design/interrupt-and-fault-model.md)、[回复预算与截断](decisions/0089-reply-output-budget-includes-reasoning.md) | `session/`、`plugins/sources/`、`plugins/reply/`、`plugins/delivery/` |
| 同 Turn 输入、打断、撤销 | SES、CTRL → [0025](decisions/0025-codex-style-same-turn-input.md)、[同 Turn 设计](design/codex-style-same-turn-input.md) | `plugins/conversation/`、`plugins/turn_projection/` |
| 元数据与旧执行恢复 | SES-009 → [0060](decisions/0060-message-plugin-metadata.md)、[0061](decisions/0061-archive-stopped-legacy-executions.md) | `session/message.py`、`agent/migrations/` |
| Akasha、Project scope、学习与重建 | MEM、SES-010 → [0073](decisions/0073-session-scope-routes-akasha-graphs.md)、[在线与重放](design/akasha-v2-runtime-migration.md)、[成本优化](design/akasha-memory-cost.md) | `plugins/akasha/`、`plugins/projects/` |
| Markdown 记忆与 consolidation | MEM → [0052](decisions/0052-compaction-and-markdown-memory-are-ordinary-plugins.md)、[插件化设计](design/compaction-markdown-memory-plugin-task-contract.md) | `plugins/markdown_memory/`、`plugins/compaction/` |
| 插件安装、卸载、热更新、generation | PLG → [0072](decisions/0072-single-graph-local-plugin-updates.md)、[单图设计](design/issue-750-plugin-publication-simplification.md) | `agent/plugins/`、`agent/plugin_composition/` |
| 能力、owner、Core 与插件边界 | PLG、CAP → [0065](decisions/0065-plugin-boundary-checks-do-not-grant-core-ownership.md)、[能力手册](design/plugin-v3-capabilities.md)、[Issue 766](design/issue-766-orthogonal-capabilities.md) | `agent/plugin_contracts/`、`agent/plugin_composition/`、`plugin_boundary.toml` |
| Skill、Drift skill、MCP、进程和 Workload | PLG-017 → [普通资源 provider](design/plugin-resource-providers.md)、[0053](decisions/0053-plugins-declare-managed-workloads.md) | 插件源码与正式安装链；`agent/plugin_composition/` |
| Wake、Drift、Scheduler、Subagent | PRO、SCH、SES → [0039](decisions/0039-react-core-atoms-keep-sources-unprivileged.md)、[React Core 设计](design/react-core-scheduler-subagent.md)、[Content/Wake](design/content-wake-existing-atoms-first-stage.md) | `plugins/wake/`、`plugins/drift/`、`plugins/scheduler/`、`plugins/subagent/` |
| EventMail、alert、内容消费 | PRO → [0048](decisions/0048-eventmail-keeps-three-mail-lifecycles.md)、[分层合同](design/content-wake-proactive-migration-task-contract.md) | `plugins/eventmail/`、`plugins/wake/` |
| Channel、持久接纳、Host boot 身份 | RUN-003、AKC → [Channel 归属](design/channel-resource-ownership.md)、[durable inbound](design/plugin-v3-durable-inbound-host-contract.md) | `bus/`、`infra/channels/`、`plugins/channels/` |
| 模型配置、选择、凭据、onboarding | RUN-005～RUN-012、ONB → [0078](decisions/0078-plugin-config-and-onboarding-ownership.md)、[引导设计](design/plugin-onboarding-projection.md)、[模型选择](design/model-user-disable.md) | `plugins/models/`、Provider 插件、`bootstrap/settings_api.py` |
| Web 或插件 UI、Android Shell | WEBUI、MOB、AKC → [frontend/design.md](../frontend/design.md)（修改前必读）、[0076](decisions/0076-android-shell-retires-legacy-mobile-stack.md)、[Shell 合同](design/android-shell-experiment.md) | `frontend/**/src`、`plugins/akashic_clients/`；不编辑生成 bundle |
| Web 窄屏、布局、导航和插件组合 | WEBUI → [窄屏设计](design/web-narrow-reading.md)、[Web 组合](design/web-ui-plugin-composition.md)、[纸张品牌](design/akashic-paper-brand-system.md) | `frontend/**/src`、`plugins/conversation_ui/` |
| 启动、停止、自重启 | RUN-001～RUN-004 → [Supervisor 设计](design/linux-supervisor-safe-self-restart.md)、[产品启动](design/product-startup.md) | `main.py`、`agent/supervisor.py`、`agent/restart.py` |
| Project 目录、Session cwd、AGENTS | SES-011、SH-004、CTX-009 → [0085](decisions/0085-project-default-and-session-working-directory.md)、[目录设计](design/project-working-directory.md) | `plugins/projects/`、`plugins/standard_tools/`、`agent/plugin_composition/messages.py` |
| Shell、PTY、进程续接 | SH → [0014](decisions/0014-shell-uses-unified-execution.md)、[Shell 设计](design/unified-shell-execution.md) | `plugins/standard_tools/`、`agent/tools/unified_exec.py` |
| 容器、Host Bridge、Computer | RUN-013～RUN-016、PLG-017 → [0084](decisions/0084-computer-keeps-identity-without-an-idle-desktop.md)、[0088](decisions/0088-computer-streams-h264-over-dashboard-websocket.md) → [显示验收](design/computer-h264-display.md)、[0075](decisions/0075-host-bridge-runtime-recovery.md)、[Bridge 协议](design/host-bridge-protocol-v2.md)、[Computer 合同](design/computer-plugin-workload-task-contract.md) | `docker/`、`agent/plugin_composition/`、正式 Controller |
| 部署、升级、备份、恢复 | MIG、BAK → [0082](decisions/0082-distribution-owned-plugin-composition.md)、[部署手册](design/operator-deployment.md)、[hua-home 事实入口](design/hua-home-plugin-runtime-source-of-truth.md) | `scripts/install-akashic.sh`、`scripts/akashic_release/` |
| Workspace、配置、迁移与数据清理 | STA、WSP、MIG → [状态地图](design/persistence-state-map.md)、[0066](decisions/0066-yoyo-current-baseline.md)、[Yoyo 手册](design/git-migration-authoring.md) | 相应状态 owner、`migrations/`、`bootstrap/init_workspace.py` |
| 事件循环、执行资源与阻塞 | ERR、RUN → [执行边界](design/event-loop-isolation.md) | 实际 owner 的调用路径与运行证据 |
| 安全边界、benchmark | SEC、TST → [安全设计](design/security-scan-edge-cases.md)、[benchmark 诊断](spark/2026-07-30-agent-benchmark-diagnostic-loop-design.md) | 对应真实边界与隔离场景 |
| 新产品方向 | [路线草案](design/akashic-future-roadmap-issue-drafts.md) → 对应现行需求；草案不是实现授权 | 按已批准范围定位 |

## 4. 数据任务的额外入口

涉及消息、记忆、附件、配置、凭据、调度、plugin-data，或裁切、压缩、重建、同步、迁移、覆盖、卸载、删除时，先读 [持久化状态地图](design/persistence-state-map.md) 的相关对象及其勘误，再按 STA-003 核对本次增、改、减与恢复方式。地图中的推断和未知不能充当删除依据。

Git worktree 保存源码、测试和项目文档；Akashic `<workspace>` 保存运行数据。切分支、删源码或清理 worktree 不授权改变后者。插件 cache 和 workspace 软链接不是源码编辑入口；外置插件只以 [fleet 当前清单](https://github.com/kachofugetsu09/akashic-fleet) 为维护范围，再回到对应源码仓库并通过正式安装链验证；本地历史目录和旧部署快照不能扩充范围。取证规则见 [hua-home 事实入口](design/hua-home-plugin-runtime-source-of-truth.md)。

## 5. 维护本索引

新增、移动或删除工作手册文件时更新相关入口和入站链接；检查相对链接、决策状态和路由是否仍能带到真实实现。无需维护第二份完整文件树。历史事故分析保留在 [语义安全设计](design/project-workbook-and-semantic-safety.md)，不能用已退役的流程覆盖当前工作手册。
