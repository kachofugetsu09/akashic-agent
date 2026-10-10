# 决策记录

这个目录保存 Akashic Agent 已经作出的重要工程决策和后续勘误。新会话先按任务关键词查找相关记录，不需要一次读完全部文件。

## 索引

| ID | 状态 | 主题 | 关联条款 |
|---|---|---|---|
| [0108](0108-bundles-are-composition-inputs.md) | proposed / modes implemented, user choice migration pending | Bundle 分层整行替换，选择仍唯一 | PLG、RUN、MIG |
| [0107](0107-restore-missing-vectors-before-reading-memory.md) | proposed / implemented for review | 按原模型空间补齐丢失的派生向量 | STA、MEM、MIG |
| [0106](0106-derived-vectors-use-separate-storage.md) | proposed / implemented for review | 派生向量独立存储，复制保留原表 | STA、MIG、MEM |
| [0105](0105-ledger-owns-business-storage.md) | proposed / implemented for review | Ledger 插件拥有业务持久化与入站交接 | PLG、STA、SES、MIG |
| [0092](0092-model-generation-recovers-until-output.md) | accepted | 模型生成持续恢复，工具效果按原回执处理 | OBJ-005、ERR-001、RUN-012 |
| [0091](0091-skill-sources-are-layered-providers.md) | proposed | Skill 来源是分层 Provider，本地目录不必打包成插件 | PLG-009、PLG-014、PLG-016、CTX-004 |
| [0092](0092-computer-viewing-and-control.md) | proposed | Computer 多目标观看与人工接管分开 | PLG-017、WEBUI-008 |
| [0088](0088-computer-streams-h264-over-dashboard-websocket.md) | proposed | Computer 经现有 Dashboard WebSocket 传输 H.264 | PLG-017、RUN-016、WEBUI-008 |
| [0086](0086-session-soft-delete-and-akasha-replay.md) | accepted | 会话软删除：逻辑失效可恢复，Akasha 学习与重放继续参与 | SES-003、SES-005、STA-003、MEM-009、WEBUI-009 |
| [0087](0087-session-title-override.md) | proposed | 会话标题覆盖：显式管理状态，空值回到推导标题 | SES-003、WEBUI-009 |
| [0085](0085-project-default-and-session-working-directory.md) | accepted | Project 固定默认目录，Session 独立当前目录与请求规则 | SES-011、SH-004、CTX-009 |
| [0084](0084-computer-keeps-identity-without-an-idle-desktop.md) | accepted | Computer 保存身份、按需唤醒与空闲回收 | PLG-017、STA-003、BAK-001、ERR-001 |
| [0083](0083-short-workflow-preserves-design-intent.md) | accepted | 精简流程、短 PR、行为验证与按风险备份，保留设计意图 | WBK、GOV、BAK-001、TST-001～TST-006 |
| [0079](0079-parallel-tool-calls-commit-in-model-order.md) | accepted | 并行工具调用重叠执行，按模型顺序提交 | PRM-004、SES-003～SES-005、RUN-003、STA-001 |
| [0077](0077-trust-installed-runtime-inputs.md) | accepted | 信任已安装的运行材料，删除运行时摘要复验 | PLG-001、PLG-002、PLG-009、ERR-001 |
| [0073](0073-session-scope-routes-akasha-graphs.md) | accepted / implemented（首版） | Session scope 宽键路由 Akasha 物化图 | SES-010、MEM-009、MEM-013、CTRL-003 |
| [0076](0076-android-shell-retires-legacy-mobile-stack.md) | accepted / implementing | Android Shell 取代旧 Mobile 协议与 OTA | MOB-001、WEBUI-001～WEBUI-004、AKC-001～AKC-002 |
| [0075](0075-host-bridge-runtime-recovery.md) | accepted | Host Bridge 运行期按故障范围恢复 | RUN-013、RUN-015、SH-001～SH-003、ERR-001 |
| [0074](0074-deployment-policy-belongs-to-operator.md) | accepted | 部署者选择备份、插件映射与迁移 | MIG-001、MIG-002、BAK-001、PLG-013、WSP-003 |
| [0072](0072-single-graph-local-plugin-updates.md) | accepted / 设计已确认，实现未完成 | 单张运行图与局部插件换代 | PLG-001～PLG-018、RUN-007、RUN-009、RUN-016、CTRL-003 |
| [0071](0071-plugin-composition-and-whole-runtime-updates.md) | accepted / implementing；整图换代部分 superseded by 0072 | 插件底座只解释组合与整体换代 | PLG-001～PLG-018 |
| [0001](0001-project-workbook-is-shared-reality.md) | superseded | 项目工作手册是协作共享现实 | WBK-001～WBK-006、COM-001～COM-004 |
| [0002](0002-context-reduction-is-a-nondestructive-projection.md) | accepted | 上下文缩减是非破坏性投影 | CTX-001～CTX-005、SES-003 |
| [0003](0003-core-capability-ownership-is-semantic.md) | accepted | 核心能力归属由权威语义决定 | MOB-001、GOV-001～GOV-005 |
| [0004](0004-cross-repository-evidence-is-an-immutable-combination.md) | accepted | 跨仓库证据绑定不可变组合 | GOV-005、TST-006～TST-008 |
| [0005](0005-git-cursor-drives-one-shot-migrations.md) | superseded | Git cursor 驱动一次性兼容迁移 | MIG-001、MIG-002、WSP-003、BAK-001 |
| [0006](0006-akasha-v2-is-the-canonical-explicit-memory-engine.md) | accepted | Akasha V2 是显式记忆的唯一算法实现 | MEM-009、SES-003、GOV-005、TST-002、TST-005 |
| [0008](0008-plugin-runtime-publishes-only-committed-snapshots.md) | superseded | 插件运行时只发布已提交快照 | PLG-001～PLG-008、GOV-005、TST-006～TST-008 |
| [0010](0010-provider-default-output-and-benchmark-diagnostics.md) | accepted | Provider 默认输出边界与 Benchmark 诊断边界 | RUN-006、TST-009 |
| [0011](0011-benchmark-concurrency-six.md) | accepted | Benchmark 隔离实例并发上限提高到六 | TST-009、WSP-004、SH-001 |
| [0012](0012-query-local-compaction-is-a-persisted-projection.md) | superseded | Query 内压缩是可持久重放的非破坏性投影 | CTX-001～CTX-007、SES-001、SES-005、CAP-001 |
| [0013](0013-linux-supervisor-uses-one-boot-guardian.md) | accepted | Linux Supervisor 每个 boot 只使用一个 Guardian | RUN-001～RUN-004、WSP-001～WSP-004 |
| [0014](0014-shell-uses-unified-execution.md) | accepted | Shell 采用统一可续接执行句柄 | SH-001、RUN-002、RUN-003、ERR-001 |
| [0015](0015-cleanup-does-not-own-turn-or-restart-finality.md) | accepted | Cleanup 不拥有 turn 与重启终态 | SH-002、RUN-003、RUN-004、OUT-001、ERR-001 |
| [0016](0016-channel-delivery-uses-complete-logical-messages.md) | accepted | 渠道投递使用完整逻辑消息 | OUT-001～OUT-003、MOB-001、SES-005～SES-006 |
| [0017](0017-one-person-companion-security-boundary.md) | accepted | 单一 Companion 的安全、容量与可恢复失败边界 | SEC-001～SEC-010、TST-001～TST-006 |
| [0021](0021-yoyo-workspace-ledger-defines-migration-origin.md) | accepted | Yoyo workspace 账本定义迁移原点 | MIG-001、MIG-002、WSP-003、BAK-001 |
| [0023](0023-akashic-tokens-own-material-3-semantics.md) | superseded | Akashic Token 拥有 Material 3 设计语义 | WEBUI-001～WEBUI-007 |
| [0024](0024-plugin-self-validation-uses-stable-and-latest.md) | superseded | 插件自验证使用 stable/latest 与 session 级并发 | RUN-007、OUT-004、PLG-013、CTRL-003、TST-001～TST-006 |
| [0025](0025-codex-style-same-turn-input.md) | accepted | 中断后的新 Attempt 续接同一 Logical Interaction | SES-007～SES-008、MEM-010～MEM-011、RUN-008、OUT-005 |
| [0026](0026-plugin-rollout-is-owned-by-the-parent-turn.md) | accepted / 晋升协议 superseded by 0072 | 插件发布由父 Turn 在终点统一授权 | PLG-010、PLG-012、PLG-013、RUN-007、CTRL-003、ERR-001、TST-001～TST-006 |
| [0030](0030-session-context-compaction-ledger.md) | accepted / implemented | Session context compaction ledger 拥有模型窗口投影 | CTX-001～CTX-007、SES-001～SES-005、MEM-002、MEM-004、MEM-008、MEM-011、MIG-001、WSP-003、TST-001～TST-006 |
| [0027](0027-runtime-models-use-generation-leases.md) | accepted / partially superseded by 0050 | 运行时模型切换使用 execution generation lease | RUN-009～RUN-012、ONB-001、CTX-001、PLG-003 |
| [0028](0028-model-credentials-live-with-workspace-connections.md) | accepted | 模型凭据随 workspace connection 保存 | RUN-009～RUN-012、ONB-001、WSP-001、BAK-001 |
| [0032](0032-host-bridge-preserves-host-equivalent-execution.md) | accepted | Host Bridge 保留宿主等价执行能力 | RUN-013～RUN-014、WSP-005、SH-001～SH-003 |
| [0033](0033-local-agent-instructions-are-not-project-documents.md) | accepted | 本地 Agent 指令不属于版本化项目文档 | WBK-001～WBK-006、COM-001～COM-004 |
| [0034](0034-turn-is-the-logical-work-unit.md) | accepted | Turn 是逻辑工作单元 | CTX-003、SES-007、SES-008、MEM-011、OUT-001、OUT-004、SCH-003 |
| [0036](0036-plugin-composition-keeps-promotion-owner.md) | accepted / 晋升 owner superseded by 0072 | 插件组合内核保留现有晋升 owner | PLG-001～PLG-013、WSP-001～WSP-005、ERR-001、TST-001～TST-007 |
| [0037](0037-plugin-runtime-is-pure-v3.md) | accepted / implemented | 插件运行时收敛为 pure v3 | PLG-001～PLG-014、WSP-001～WSP-005、ERR-001、TST-001～TST-008 |
| [0038](0038-operator-trust-can-publish-offline-plugin-batches.md) | accepted / 旧提交协议部分按 0072 调整 | Operator 信任可以离线发布 exact 插件批次 | PLG-013、RUN-015、ERR-001 |
| [0039](0039-react-core-atoms-keep-sources-unprivileged.md) | accepted | React 原子能力留在 Core，来源保持非特权 | RUN-001～RUN-003、RUN-007～RUN-009、OUT-001～OUT-004、PLG-014、SCH-001～SCH-003、PRO-001、SEC-005、SEC-007 |
| [0040](0040-wake-duty-gate-lives-in-scoped-react.md) | accepted | Wake duty gate 属于 Wake scoped react | RUN-003、RUN-007～RUN-009、OUT-001～OUT-003、PLG-014、PRO-001～PRO-002 |
| [0041](0041-turn-effects-and-memory-plugins-are-orthogonal.md) | accepted / implementing | Turn 副作用与 Memory 插件保持正交 | SES-001、SES-007～SES-008、MEM-002、MEM-009～MEM-011、PLG-001～PLG-014、RUN-003、RUN-007～RUN-009 |
| [0042](0042-plugin-diagnostics-preserve-domain-owners.md) | accepted / implementing | 插件诊断保留领域 owner | OBJ-002、PLG-003、PLG-006、PLG-014～PLG-015、ERR-001 |
| [0043](0043-paper-brand-tokens-replace-material-visual-semantics.md) | accepted | 纸张品牌 Token 取代 Material 视觉语义 | WEBUI-001～WEBUI-007 |
| [0045](0045-akashic-direct-messages-commit-before-notify.md) | accepted | Akashic 主动消息先提交 Session 再通知客户端 | AKC-001～AKC-002、OUT-001、OUT-003～OUT-004 |
| [0046](0046-plugin-candidate-validation-is-incremental.md) | accepted / implemented | 插件候选只重建依赖闭包 | PLG-001～PLG-004、PLG-008～PLG-010、PLG-014 |
| [0047](0047-provides-may-bind-one-tool.md) | partially superseded | 一个 provide 可以绑定一个 Tool | PLG-001～PLG-014、PRO-001～PRO-002 |
| [0048](0048-eventmail-keeps-three-mail-lifecycles.md) | accepted / implemented | EventMail 统一信封并保持三类生命周期 | PLG-014～PLG-016、PRO-001～PRO-005 |
| [0049](0049-wake-content-is-a-decaying-eventmail-pool.md) | accepted / implemented | Wake Content 是 EventMail 中的衰减池 | PRO-004～PRO-006、PLG-014～PLG-016 |
| [0050](0050-model-revision-lives-in-ordinary-plugin.md) | accepted | 模型 revision 由普通插件拥有 | RUN-005～RUN-012、ONB-001、PLG-003、PLG-014、PLG-016、WSP-001 |
| [0051](0051-web-ui-composes-ordinary-plugin-modules.md) | accepted / implementing | WebUI 由普通插件递归组合 | WEBUI-001～WEBUI-007、PLG-001～PLG-016、ONB-001、MOB-001 |
| [0052](0052-compaction-and-markdown-memory-are-ordinary-plugins.md) | accepted / implementing | Compaction 与 Markdown 记忆是普通插件 | CTX-007、MEM-001～MEM-011、PLG-001～PLG-014、SES-003～SES-005 |
| [0053](0053-plugins-declare-managed-workloads.md) | accepted / implementing | 插件声明受管 Workload | RUN-016、PLG-017、WEBUI-008、WSP-006 |
| [0054](0054-model-sync-refreshes-public-capabilities.md) | accepted / implementing | 模型同步刷新公共能力目录 | RUN-011、ONB-001、WSP-001 |
| [0055](0055-host-bridge-uses-typed-protobuf.md) | accepted | Host Bridge 使用 typed Protobuf V2 | RUN-013～RUN-015、SH-001～SH-003 |
| [0056](0056-plugin-update-crashes-return-to-stable.md) | accepted / implementing；崩溃回旧 stable superseded by 0072 | 插件更新中进程死亡时恢复旧指针，不续跑候选 | PLG-010、PLG-013、RUN-007 |
| [0057](0057-internal-source-messages.md) | accepted | Subagent 与 Wake 保留完整内部消息 | SES-003～SES-005、MEM-001～MEM-002 |
| [0058](0058-scheduler-keeps-internal-messages.md) | accepted | Scheduler 保留可恢复的内部消息 | SCH-001～SCH-003、SES-003～SES-005、MEM-001～MEM-002 |
| [0059](0059-abandon-settles-tool-calls.md) | accepted | 明确放弃结算工具调用，不等待物理清理 | RUN-003、RUN-008、SES-003～SES-005、SH-002 |
| [0060](0060-message-plugin-metadata.md) | accepted | 插件附加信息使用普通 Message metadata | SES-001、SES-003～SES-006、SES-009 |
| [0061](0061-archive-stopped-legacy-executions.md) | accepted | 已停止旧执行完整归档，不自动续跑 | SES-001、SES-003、WSP-001、BAK-001 |
| [0062](0062-tools-flow-through-provider-views.md) | accepted / implemented | 工具通过 provider view 流向消费者 | CTX-004、CTX-007、PLG-003、PLG-008、PLG-009、PLG-014、PLG-016、PLG-018 |

| [0063](0063-execution-failures-have-terminal-results.md) | accepted | 执行失败明确收尾，恢复依据原回执 | Tools、Delivery、Wake、Models |
| [0064](0064-plugin-boundary-is-machine-enforced.md) | superseded by 0065 | 插件边界静态门的初始设计 | PLG-001～PLG-017、GOV-001～GOV-005、TST-001～TST-008 |
| [0065](0065-plugin-boundary-checks-do-not-grant-core-ownership.md) | accepted | 边界检查不授予 Core 归属，按外置与替换验收 | PLG-014、PLG-016、STA-001、CAP-001、TST-003 |

| [0066](0066-yoyo-current-baseline.md) | accepted | 保留 Yoyo，以当前基线退役历史兼容脚本 | MIG-001、MIG-002、WSP-003 |

| [0067](0067-clients-are-ordinary-plugin.md) | accepted | Web Chat 由普通插件拥有，Core 只提供中立原子能力 | AKC-001、AKC-002、PLG-001、WSP-003 |

| [0068](0068-compaction-uses-one-recent-window.md) | accepted | Compaction 每代只摘要一个近期窗口 | CTX-001～CTX-007、MEM-011～MEM-012、SES-003～SES-005 |
| [0069](0069-bindings-follow-selected-runtime-scope.md) | accepted / implementing | Binding 跟随调用已选的 runtime scope | PLG-003、PLG-004、PLG-009、PLG-013、PLG-018、RUN-008～RUN-009、ERR-001 |

| [0070](0070-plugins-own-persisted-data.md) | accepted | 插件系统负责依赖与切换，插件负责自己的持久化数据 | PLG-018、RUN-008～RUN-009、ERR-001 |

| [0078](0078-plugin-config-and-onboarding-ownership.md) | accepted | 配置由业务插件拥有，普通引导只组合当前状态 | ONB-001、PLG-003、PLG-014、PLG-016 |

| [0080](0080-retire-qq-runtime-support.md) | accepted | 退役全部 QQ 运行支持，保留旧数据与历史证据 | ONB-001、PLG-003、STA-001～STA-003、SEC-001、SEC-002 |

| [0081](0081-content-views-keep-original-messages.md) | superseded（折叠策略） | 长结果折叠复用原消息与实际展示回执 | CTX-008、STA-001、CAP-001 |

| [0082](0082-distribution-owned-plugin-composition.md) | accepted | 内置代码随部署、外置选择保留，沿同一普通插件图提交 | ONB-002、PLG-007、PLG-013、PLG-016、MIG-001 |

| [0089](0089-reply-output-budget-includes-reasoning.md) | accepted | 回复预算包含推理，长度截断不作为完成或工具执行 | ERR-001、RUN-005、SES-001、STA-002 |

| [0090](0090-tool-results-stay-visible.md) | accepted | 移除自动折叠，工具原文持续可见，保留旧回读 | CTX-001、CTX-008、STA-002 |

| [0095](0095-context-compaction-uses-settled-batches.md) | accepted | 请求摘要按安全工具批次切分，20K 为目标，失败原因对用户可见 | CTX-001、CTX-003、CTX-007、MEM-011 |
| [0096](0096-context-reminders-keep-their-first-position.md) | proposed | 不变提醒保留首次位置，变化材料追加，实时材料不回放 | CTX-004、CTX-009 |

| [0097](0097-request-deltas-belong-to-one-react-run.md) | proposed | 冻结请求增量只引用本次执行的已提交记录，保持 JSON 类型与准确重放 | CTX-001、STA-002、SES-005 |
| [0098](0098-host-bridge-reuses-protobuf-socket.md) | accepted | Host Bridge 在复用的 Unix socket 上传输 Protobuf | RUN-013～RUN-015、SH-001～SH-003 |

| [0099](0099-ledger-commits-use-wal-normal.md) | accepted | Session 与 Models 账本采用进程崩溃级保证，接受宿主故障丢失尾部提交 | STA-001～STA-003、CAP-002、SES-001～SES-002 |

| [0100](0100-message-tool-calls-recover-at-turn-granularity.md) | accepted | 默认消息工具未结调用报告未知，ReAct 恢复与显式重试使用当前材料发新请求 | ERR-001、CAP-002、SES-001 |

| [0101](0101-first-message-session-title.md) | accepted | 首条输入自动命名，复用 title 并以当前空值条件写入 | SES-012 |

| [0102](0102-plugin-owned-services-and-public-contracts.md) | accepted target / implementation | 服务与公共合同归提供方，Core 只保留组合与宿主机制 | PLG、CAP、STA |

| [0103](0103-processes-follow-provider-lifetime.md) | accepted | 进程集合随 HostExecution provider 关闭，取消等待保留本代续接 | PLG-017、SH |

## 新增规则

1. 使用四位递增编号和短英文文件名。
2. 写明状态、日期、背景、决定、理由、影响、验收和关联条款。
3. 旧决定被推翻时保留原文件，新记录声明 `supersedes`，旧记录补 `superseded by`。
4. 没有形成选择的讨论不进入这里；未完成动作写入 `NOW.md`。

- [0057 · Subagent 与 Wake 保留完整内部消息](0057-internal-source-messages.md)：独立内部 Session 的保存、展示、投递和学习边界。

| [0091](0091-shell-self-deployment.md) | accepted | Shell 提交宿主部署，正常回合与排空之后更新 Docker | RUN-004、SH-001、OUT-001、MIG-001 |

| [0094](0094-plugin-runtime-uses-installed-files.md) | accepted | 插件使用安装文件与当前配置，取消强快照归档 | PLG-002、PLG-003、PLG-010、PLG-013、STA-003 |
| [0093](0093-computer-reuses-anonymous-browsers.md) | proposed | 匿名实例稳定 ID、错误保留进度与闲置回收 | PLG-017 |

| [0104](0104-public-codecs-without-ui-runtime.md) | proposed / implemented for review | 公共只读编码与类型同寿命，headless 不要求 UI runtime | PLG、CAP |

| [0109](0109-workspace-owns-plugin-choices.md) | proposed / patch API implemented, activation pending | 启停归 workspace patch，库存取实际制品，配置保留单一 owner | PLG、MIG、STA |
