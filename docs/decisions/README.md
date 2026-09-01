# 决策记录

这个目录保存 Akashic Agent 已经作出的重要工程决策和后续勘误。新会话先按任务关键词查找相关记录，不需要一次读完全部文件。

## 索引

| ID | 状态 | 主题 | 关联条款 |
|---|---|---|---|
| [0001](0001-project-workbook-is-shared-reality.md) | superseded | 项目工作手册是协作共享现实 | WBK-001～WBK-006、COM-001～COM-004 |
| [0002](0002-context-reduction-is-a-nondestructive-projection.md) | accepted | 上下文缩减是非破坏性投影 | CTX-001～CTX-005、SES-003 |
| [0003](0003-core-capability-ownership-is-semantic.md) | accepted | 核心能力归属由权威语义决定 | MOB-001、GOV-001～GOV-005 |
| [0004](0004-cross-repository-evidence-is-an-immutable-combination.md) | accepted | 跨仓库证据绑定不可变组合 | GOV-005、MOB-002～MOB-004、TST-006～TST-008 |
| [0005](0005-git-cursor-drives-one-shot-migrations.md) | superseded | Git cursor 驱动一次性兼容迁移 | MIG-001、MIG-002、WSP-003、BAK-001 |
| [0006](0006-akasha-v2-is-the-canonical-explicit-memory-engine.md) | accepted | Akasha V2 是显式记忆的唯一算法实现 | MEM-009、SES-003、GOV-005、TST-002、TST-005 |
| [0007](0007-mobile-plugin-control-and-data-planes-are-explicit.md) | accepted | 移动插件控制面与查询数据面显式分离 | MOB-001、MOB-003、MOB-006、PLG-003、PLG-011、TST-006～TST-008 |
| [0008](0008-plugin-runtime-publishes-only-committed-snapshots.md) | superseded | 插件运行时只发布已提交快照 | PLG-001～PLG-008、GOV-005、TST-006～TST-008 |
| [0009](0009-akasha-mobile-recall-preserves-semantic-lanes.md) | accepted | Akasha 移动卡片完整保留有界召回 lane | MOB-006、PLG-011、TST-006～TST-008 |
| [0010](0010-provider-default-output-and-benchmark-diagnostics.md) | accepted | Provider 默认输出边界与 Benchmark 诊断边界 | RUN-006、TST-009 |
| [0011](0011-benchmark-concurrency-six.md) | accepted | Benchmark 隔离实例并发上限提高到六 | TST-009、WSP-004、SH-001 |
| [0012](0012-query-local-compaction-is-a-persisted-projection.md) | superseded | Query 内压缩是可持久重放的非破坏性投影 | CTX-001～CTX-007、SES-001、SES-005、CAP-001 |
| [0013](0013-linux-supervisor-uses-one-boot-guardian.md) | accepted | Linux Supervisor 每个 boot 只使用一个 Guardian | RUN-001～RUN-004、WSP-001～WSP-004 |
| [0014](0014-shell-uses-unified-execution.md) | accepted | Shell 采用统一可续接执行句柄 | SH-001、RUN-002、RUN-003、ERR-001 |
| [0015](0015-cleanup-does-not-own-turn-or-restart-finality.md) | accepted | Cleanup 不拥有 turn 与重启终态 | SH-002、RUN-003、RUN-004、OUT-001、ERR-001 |
| [0016](0016-channel-delivery-uses-complete-logical-messages.md) | accepted | 渠道投递使用完整逻辑消息 | OUT-001～OUT-003、MOB-001、MOB-005、SES-005～SES-006 |
| [0017](0017-one-person-companion-security-boundary.md) | accepted | 单一 Companion 的安全、容量与可恢复失败边界 | SEC-001～SEC-010、TST-001～TST-006 |
| [0018](0018-chat-webui-has-one-source-and-two-adapters.md) | accepted | 对话 WebUI 使用一个源码真源和两个平台适配器 | WEBUI-001～WEBUI-003、MOB-001、TST-007～TST-008 |
| [0019](0019-mobile-long-messages-use-bounded-events.md) | accepted | Mobile 长消息使用有界正文事件和紧凑终态 | MOB-001、MOB-003、MOB-005、MOB-007、SES-001 |
| [0020](0020-mobile-history-content-uses-authenticated-http-ranges.md) | accepted | Mobile 历史长正文使用认证 HTTP Range 恢复 | MOB-001、MOB-003、MOB-006、MOB-007、SES-001、TST-005、TST-008 |
| [0021](0021-yoyo-workspace-ledger-defines-migration-origin.md) | accepted | Yoyo workspace 账本定义迁移原点 | MIG-001、MIG-002、WSP-003、BAK-001 |
| [0022](0022-mobile-webui-uses-server-selected-generations.md) | accepted | 移动 WebUI 使用服务端选择的不可变 generation | WEBUI-001～WEBUI-006、MOB-001～MOB-004、TST-006～TST-008 |
| [0023](0023-akashic-tokens-own-material-3-semantics.md) | superseded | Akashic Token 拥有 Material 3 设计语义 | WEBUI-001～WEBUI-007 |
| [0024](0024-plugin-self-validation-uses-stable-and-latest.md) | superseded | 插件自验证使用 stable/latest 与 session 级并发 | RUN-007、OUT-004、PLG-013、CTRL-003、TST-001～TST-006 |
| [0025](0025-codex-style-same-turn-input.md) | accepted | 中断后的新 Attempt 续接同一 Logical Interaction | SES-007～SES-008、MEM-010～MEM-011、RUN-008、OUT-005 |
| [0026](0026-plugin-rollout-is-owned-by-the-parent-turn.md) | accepted | 插件发布由父 Turn 在终点统一授权 | PLG-010、PLG-012、PLG-013、RUN-007、CTRL-003、ERR-001、TST-001～TST-006 |
| [0030](0030-session-context-compaction-ledger.md) | accepted / implemented | Session context compaction ledger 拥有模型窗口投影 | CTX-001～CTX-007、SES-001～SES-005、MEM-002、MEM-004、MEM-008、MEM-011、MIG-001、WSP-003、TST-001～TST-006 |
| [0027](0027-runtime-models-use-generation-leases.md) | accepted / partially superseded by 0050 | 运行时模型切换使用 execution generation lease | RUN-009～RUN-012、ONB-001、CTX-001、PLG-003 |
| [0028](0028-model-credentials-live-with-workspace-connections.md) | accepted | 模型凭据随 workspace connection 保存 | RUN-009～RUN-012、ONB-001、WSP-001、BAK-001 |
| [0029](0029-main-gateway-reconciles-mobile-webui-stable.md) | accepted | main Gateway 对账移动 WebUI Stable | WEBUI-004～WEBUI-006、GOV-005、TST-006～TST-008 |
| [0031](0031-stable-matching-head-allows-gateway-restart.md) | accepted / implemented | Stable 与本地 HEAD 一致时允许 Gateway 重启 | WEBUI-004～WEBUI-006、GOV-005、TST-006～TST-008 |
| [0032](0032-host-bridge-preserves-host-equivalent-execution.md) | accepted | Host Bridge 保留宿主等价执行能力 | RUN-013～RUN-014、WSP-005、SH-001～SH-003 |
| [0033](0033-local-agent-instructions-are-not-project-documents.md) | accepted | 本地 Agent 指令不属于版本化项目文档 | WBK-001～WBK-006、COM-001～COM-004 |
| [0034](0034-turn-is-the-logical-work-unit.md) | accepted | Turn 是逻辑工作单元 | CTX-003、SES-007、SES-008、MEM-011、OUT-001、OUT-004、SCH-003 |
| [0035](0035-mobile-protocol-delivery-is-phased.md) | accepted | 移动协议交付按变更性质分阶段 | MOB-008、MOB-006、TST-007、GOV-002 |
| [0036](0036-plugin-composition-keeps-promotion-owner.md) | accepted | 插件组合内核保留现有晋升 owner | PLG-001～PLG-013、WSP-001～WSP-005、ERR-001、TST-001～TST-007 |
| [0037](0037-plugin-runtime-is-pure-v3.md) | accepted / implementing | 插件运行时收敛为 pure v3 | PLG-001～PLG-014、WSP-001～WSP-005、ERR-001、TST-001～TST-008 |
| [0038](0038-operator-trust-can-publish-offline-plugin-batches.md) | accepted | Operator 信任可以离线发布 exact 插件批次 | PLG-013、RUN-015、ERR-001 |
| [0039](0039-react-core-atoms-keep-sources-unprivileged.md) | accepted | React 原子能力留在 Core，来源保持非特权 | RUN-001～RUN-003、RUN-007～RUN-009、OUT-001～OUT-004、PLG-014、SCH-001～SCH-003、PRO-001、SEC-005、SEC-007 |
| [0040](0040-wake-duty-gate-lives-in-scoped-react.md) | accepted | Wake duty gate 属于 Wake scoped react | RUN-003、RUN-007～RUN-009、OUT-001～OUT-003、PLG-014、PRO-001～PRO-002 |
| [0041](0041-turn-effects-and-memory-plugins-are-orthogonal.md) | accepted / implementing | Turn 副作用与 Memory 插件保持正交 | SES-001、SES-007～SES-008、MEM-002、MEM-009～MEM-011、PLG-001～PLG-014、RUN-003、RUN-007～RUN-009 |
| [0042](0042-plugin-diagnostics-preserve-domain-owners.md) | accepted / implementing | 插件诊断保留领域 owner | OBJ-002、PLG-003、PLG-006、PLG-014～PLG-015、ERR-001 |
| [0043](0043-paper-brand-tokens-replace-material-visual-semantics.md) | accepted | 纸张品牌 Token 取代 Material 视觉语义 | WEBUI-001～WEBUI-007 |
| [0044](0044-akashic-channel-uses-web-and-mobile-adapters.md) | accepted | Akashic Channel 使用 Web 与 Mobile 两个 Adapter | AKC-001～AKC-003、MOB-001～MOB-008、SES-001～SES-008 |
| [0045](0045-akashic-direct-messages-commit-before-notify.md) | accepted | Akashic 主动消息先提交 Session 再通知客户端 | AKC-001～AKC-003、MOB-002、MOB-005、MOB-008、OUT-001、OUT-003～OUT-004 |
| [0046](0046-plugin-candidate-validation-is-incremental.md) | accepted / implemented | 插件候选只重建依赖闭包 | PLG-001～PLG-004、PLG-008～PLG-010、PLG-014 |
| [0047](0047-provides-may-bind-one-tool.md) | accepted / implementing | 一个 provide 可以绑定一个 Tool | PLG-001～PLG-014、PRO-001～PRO-002 |
| [0048](0048-eventmail-keeps-three-mail-lifecycles.md) | accepted / implemented | EventMail 统一信封并保持三类生命周期 | PLG-014～PLG-016、PRO-001～PRO-005 |
| [0049](0049-wake-content-is-a-decaying-eventmail-pool.md) | accepted / implemented | Wake Content 是 EventMail 中的衰减池 | PRO-004～PRO-006、PLG-014～PLG-016 |
| [0050](0050-model-revision-lives-in-ordinary-plugin.md) | accepted | 模型 revision 由普通插件拥有 | RUN-005～RUN-012、ONB-001、PLG-003、PLG-014、PLG-016、WSP-001 |
| [0051](0051-web-ui-composes-ordinary-plugin-modules.md) | accepted / implementing | WebUI 由普通插件递归组合 | WEBUI-001～WEBUI-007、PLG-001～PLG-016、ONB-001、MOB-001 |
| [0052](0052-compaction-and-markdown-memory-are-ordinary-plugins.md) | accepted / implementing | Compaction 与 Markdown 记忆是普通插件 | CTX-007、MEM-001～MEM-011、PLG-001～PLG-014、SES-003～SES-005 |
| [0053](0053-plugins-declare-managed-workloads.md) | accepted / implementing | 插件声明受管 Workload | RUN-016、PLG-017、WEBUI-008、WSP-006 |

## 新增规则

1. 使用四位递增编号和短英文文件名。
2. 写明状态、日期、背景、决定、理由、影响、验收和关联条款。
3. 旧决定被推翻时保留原文件，新记录声明 `supersedes`，旧记录补 `superseded by`。
4. 没有形成选择的讨论不进入这里；未完成动作写入 `NOW.md`。
