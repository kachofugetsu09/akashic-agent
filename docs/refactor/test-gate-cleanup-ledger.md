# 测试与 Gate 清理账本

> **已被 [`orthogonality-test-baseline.md`](orthogonality-test-baseline.md) 取代。** 现行 `tests/` 保留清单、新增规则和 CI 收敛以该文件为准。下文只保留历史清理记录。

本账本记录测试与 Gate 的永久收敛。数量只是历史观察指标，不是删除依据；取舍按用户可观察失败、持久化与安全边界、并发 finality、恢复能力和插件 v3 生命周期排序。

## 2026-09-23：Issue 750 绿色集成 R1 的测试入口迁移

本次固定基线是 `33bd6d41fc06ca8c39c250b0f384666b4d381fa5`，比较的 main 是
`ae444d47281b5b25bb467a76dd25eab02c95ec08`。恢复材料位于
`/mnt/data/issue750-green-r1-20260923.68KJfw/recovery-T-e69982-sybk8bhb`；
隔离命令、源码摘要、JUnit、退出码位于同任务目录的 `evidence/`。以下迁移只改
测试执行入口，不减少 Message、归档、选择或 cleanup 的保护。

| 旧 oracle 或入口 | 当前执行边界与保留断言 | 定点结果 |
| --- | --- | --- |
| `test_plugin_latest_models.py` 经已删除的 `RuntimeModelControl`/candidate registry 读设置 | 真实 Models provider Context、live Root、RPC 与持久 ModelsStore；保留选择、metadata、凭据、绑定和失败原子性 | 4/4 |
| restart、reply follow、runtime smoke、UI/Dashboard、MCP、Channel、Akasha、Source、Commands 等测试从 `current_snapshot` 或旧 fixture 读服务 | 读取 Manager `live_root`、准确 provider/consumer Context、`runtime_scope` 与实际 Fiber；错误注入仍在原不可信边界，保留原业务断言 | 按各文件 JUnit；最终全量结果另记 |
| `test_plugin_hot_reload.py` 的 compile/identity/module tree/bytecode/asset 准备候选 | `reconcile_changed`、source diagnostic、同一 Root 的 generation 与 selected archive；坏源码不改选择，修复后新 generation 可观察，资产读取固定归档 | 迁移前 13 红；迁移后文件 77/77 |
| installed latest/promote/discard 指针与旧 candidate 失败 | 本地 Git 源经公开 `PluginManager.install` 提交选择，旧指针漂移不自动发布；B accepted、ACTIVE 或 FAILED 分离，失败 B 保持 selected，显式 retry 读精确 B 归档；workspace skill 不被改写 | 新四项 4/4，包含于 77/77 |
| `test_snapshot_admission_waits_while_current_is_quiesced` | `test_live_admission_waits_for_held_owner_call_during_update`：旧 Fiber 已接纳调用阻止排空，关接纳后拒绝旧 activation，释放后 B 才 active | 1/1，包含于 77/77 |
| `test_runtime_snapshot_lease_commit_and_abort` | `test_live_selection_compile_abort_then_commit`：编译失败保留 A/selection；有效 B 只在 CAS 后进入同一个 Root | 1/1，包含于 77/77 |
| `test_runtime_snapshot_latest_closes_before_fresh_formal_publication` 的公开 Gate 入口 | `test_local_retry_reloads_selected_archive_after_start_failure`：B 启动失败后原归档和诊断仍可见，retry 使用 B 新 owner，不复活 A | 1/1，包含于 77/77 |
| `test_runtime_snapshot_discard_keeps_stable_and_waits_for_latest_lease` 的公开 Gate 入口 | 上述 live compile/commit 与 admission 两项共同证明提交前保留 A 和真实调用排空；旧 latest/discard 双图没有现行执行者，测试主体删除 | 2/2，包含于 77/77 |
| Computer `current_snapshot`/snapshot lease、Content duplicate schema 的异常抛出假设 | Computer 用真实 Root、Tool owner scope、隔离虚拟环境 Python 执行原 MCP 进程；Content 检查失败 Fiber 的原始 `ValueError`；保留实际 CallRef、follower、重复 kind 拒绝 | Computer 端到端和重启各 1/1；Content 20/20 |

### 旧 candidate 故障用例与现行 owner 的逐组对账

下表只退役已经没有执行者的 candidate/validation/promote/discard 入口。旧用例中提交前不改选择、提交后不虚报回退、物理 cleanup 失败保留 owner、重启读取已选归档等安全断言，仍由右列的真实 local install、Root、journal 或外部接纳用例执行。测试数量变化不作为依据。

| 旧用例或故障注入 | 现行执行边界及保留断言 | 已运行证据 |
| --- | --- | --- |
| `test_plugin_business_validation.py` 的真实模型回复、独立消息/Task、MCP 失败、验证宿主构造/关闭，以及 `test_validation_host_owners.py` 的 validation lease 与取消建 Root | 双图 validation host 已退役；正式 `plugin_install` 的消息与送达由 `test_plugin_update_source.py::test_real_plugin_update_watcher_reports_through_delivery` 验证，外部 API/Tool 执行与拒绝由 `test_plugin_external_acceptance.py` 验证；Root 资源关闭失败由 `test_plugin_composition_lifecycle.py::test_terminate_joins_untransferred_root_without_generations` 验证。Mobile 的两个实际 Root 隔离测试保留。 | update source 3/3；external acceptance 8/8；composition lifecycle 23/23；validation owner 1/1 |
| `test_plugin_candidate_init_failure.py` 的 reconcile/init 失败、journal 失败、双故障与 cleanup/discard 重试 | 旧候选 registry、latest/discard 回退和 validation 清理已退役。`test_plugin_hot_reload.py` 的 `test_live_pre_fiber_error_stages_project_one_failed_generation`、`test_selected_pre_fiber_failure_retains_cleanup_owner_until_explicit_retry`、`test_selected_pre_fiber_cleanup_cancellation_waits_for_scope_gate`、`test_public_install_retains_failed_b_until_real_selected_retry` 证明已选 B 的错误可查询、失败 owner 不被丢弃、显式 retry 才换代；`test_plugin_scope.py::test_generation_disposal_keeps_failed_owner_and_module` 保留物理失败与模块句柄。 | hot reload 77/77；scope 文件定点通过 |
| 同文件的候选数据空目录、live old host、SIGKILL supervised/unsupervised、旧 cleanup 表迁移 | `test_plugin_candidate_data.py` 验证正式 plugin-data 与不相关 DB 隔离；`test_plugin_retired_activity_recovery.py` 拒绝旧 owner journal 且未知 selection 不伪造恢复；`test_plugin_stable_runtime.py::test_boot_settles_exact_transition_without_resuming_or_rolling_back_install` 保留 exact selection 与未知结果；`test_plugin_update_input_ref_migration.py` 检查当前唯一迁移列、备份和严格 schema。已删除的 candidate_cleanup 表没有现行读写者。 | candidate data 4/4；retired activity 文件定点通过；stable runtime 10/10；input ref migration 6/6 |
| `test_plugin_discard_retained_validation.py`、`test_plugin_drop_update.py` 的关闭候选、指针/manifest 回写、外部漂移与重复 discard | discard/自动 rollback 不再授权修改 selected B。`test_plugin_uninstall_root_drain.py` 走公开 uninstall 排空并移除 selected target；`test_plugin_updates_api.py` 检查 accepted、busy、失败、显式 retry；`test_plugin_update_rollback.py` 检查不确定写入、重启和安装 settlement。 | uninstall 文件定点通过；updates API 9/9；plugin suite 中 rollback 通过 |
| `test_plugin_fresh_root.py` 的候选/正式两次挂载与旧快照替换 | 新用例 `test_live_update_replaces_only_target_fiber_in_one_root` 证明同一 Root 中只更换目标 Fiber、peer 身份不变；`test_plugin_hot_reload.py::test_local_compile_failure_keeps_selection_and_old_owner` 保留预提交失败保护，`test_local_retry_reloads_selected_archive_after_start_failure` 保留失败 B 的新 generation retry。 | fresh root 1/1；hot reload 77/77 |
| `test_plugin_latest_authorization.py`、`test_plugin_latest_control.py`、`test_plugin_publication_admission.py` 的 latest 读取、promote、revert、candidate call lease 与旧控制工具 | 现行工具只有 `plugin_install`；`test_plugin_updates_api.py`、`test_plugin_update_operation_lease.py` 与 `test_plugin_hot_reload.py::test_live_admission_waits_for_held_owner_call_during_update` 验证实际调用许可、一次操作接纳和局部排空；历史未知外部效果不重跑由 `test_plugin_update_source.py` 的 durable receipt 检查。 | updates API 9/9；plugin suite 中 operation lease 通过；hot reload 77/77；update source 3/3 |
| `test_plugin_stable_runtime.py` 的 `_replace_formal_root`/snapshot callback、`test_plugin_composition_lifecycle.py` 的 prepublication/candidate Root 与旧 seal 回调 | 前者保留完整归档重启、null 初始选择和五种 boot selection 结算（10/10）；后者保留 23 个真实 Root/Fiber/Effect/EventBus oracle，包括失败 STOPPING 重试、未移交 Root 并发关闭、绑定冻结和跨 Task Channel 许可拒绝。CAS 冲突、取消、失败 B 与 selected archive 由 hot reload/operation lease 覆盖。 | stable runtime 10/10；composition lifecycle 23/23；hot reload 77/77 |

`recursive_plugin_self_validation_contract` 替换上表三个旧 snapshot nodeid；
`plugin_uninstall_drain_finality` 的三个旧 nodeid 分别改为 live MCP 更新、
Socket accepted/排空，并以现行清理重试和隔离 host grant 补足已退役 candidate
只读验证的责任。该场景现有六个 nodeid 已在隔离环境 6/6 通过。
两个场景的 `requirements`、`observes`、`mutants` 和场景总数均保留。

其余尾段迁移也保留具体失败路径：`test_stable_view_plugin.py` 的真实 Manager
仅投影已选插件，缺席的 MCP 不再伪造一行 unavailable；MCP unavailable 格式化
由同文件独立用例保留。Tool binding 改从 Manager `BINDINGS` 和实际 Tools
OwnerCall 固定，源码修改后由显式 `reconcile_changed` 或 disable 才换已选归档；
旧 binding 的授权名称、schema 和历史保持可查。Subagent 的受控远端回复丢失
验证不重付与恢复回传，Web 测试验证旧回调排空与新 binding 同时存在。
`test_workload_boot_cleanup.py` 以同一 Root 的局部 generation 换代代替旧
`_replace_formal_root`/candidate 清理调用，boot cleanup 的未知结果和坏 receipt
仍 fail loud。Yoyo 退役测试补上其声称的旧 manifest fixture，生产迁移文件未改。

Models 的先前删除映射已被 T-71d1a1 拒绝：frozen baseline `683b4791`
只有仍存活的 `agent/plugins/manager.py`，不能用它证明后来新增、此次删除的
`model_catalog.py`、`model_control.py`。现以普通 `paths` 中的
`agent/plugin*/model*.py` 表示模型能力的变更归属：当前匹配
`agent/plugin_composition/model.py`、`model_settings_http.py`、`models.py`，
固定 main 还匹配两个已退休的 `agent/plugins/model_*.py` wrapper；
不匹配 Manager、manifest、mobile_ui。原 composition-model 命中集保留；
P0 `model_owner_contract`、要求与依赖未减。这是普通影响映射，不是 frozen base
上的删除证明。
`coverage-baseline.json` 只更新 `catalogDigest`，由现有 Gate 算法按
`impact.toml`、`state-contracts.toml`、`scenarios.toml` 的文件名和原始字节顺序
计算，修正后值 `327e454ed750ac5a941d22f6cda6fc21ae385368b189dc117c40b002d652d6f7`；
`acceptedGaps`、`coveredP0`、`base`、purpose、version 不变。Gate audit 与 27 场景
plan 通过；首轮 run 为 26/27，唯一红项是上述三个旧 nodeid 无法收集。
最终 run 的实际结果须由同轮独立报告记录，plan 不算执行。

完整 Python 顺序执行还暴露 Wake 确定性探针的退出责任：settlement 故意抛
`RuntimeError` 后，依赖 Fiber 按合同保留失败 owner，探针却在同一测试事件循环
内模拟进程重启。`RuntimeStack.abort_after_fault` 现在只取消该隔离栈启动后创建
的任务并等待结束；fixture source 收到取消后不重挂 Timer。故障注入类型、耐久
恢复与 ACK 断言保持原样，`test_wake_gate_contract.py` 定点 4/4 且 pytest
正常退出。完整套件和公开 Gate 的最终结果仍以本轮回执与实际报告为准。

## 2026-09-06：Message 输入接纳替换旧回复队列

本项属于已批准的 Message/plugins 栈第 08 层：Channel 在 Input 提交后返回，回复由独立消费者运行。`publish_channel_inbound` 的 BUS → LANE → LOOP 接纳已被删除；不为中间 PR 恢复兼容队列。原测试备份：`/tmp/message-plugins-pr08-inbound-recovery-backup-20260906/tests/test_message_bus_admission.py`，Git 相邻基线 `79fcc358` 也可恢复旧测试。

以下对应旧 `tests/test_message_bus_admission.py` 的 19 项用例。新边界均在 `tests/test_channel_input.py`；独立回复在 `tests/test_default_reply.py`。删除不依据测试数量或失败本身。

| 旧测试范围（共同前缀 `test_v3_`） | 处置与继续保护的行为 |
| --- | --- |
| `channel_inbound_transfers_bus_lane_loop_and_closes_once`、`channel_inbound_bus_close_releases_queued_exact_lease`、`channel_inbound_blocked_at_lane_is_closed_by_concurrent_bus_close`、`channel_bus_close_cancellation_drains_every_queued_lease`、`channel_inbound_release_cancellation_clears_lane_before_return` | 删除已移除队列/lane 的实现合同；新边界证明 Input 无排队、无模型或发送，取消关闭 exact lease，关闭后的 lock waiter 不能提交。 |
| `channel_worker_preserves_exact_binding_through_terminal_delivery`、`channel_worker_holds_session_admission_until_terminal`、`channel_worker_cancel_closes_running_and_lane_queued_leases` | 删除旧 worker 持有到最终回复的合同。新接纳只持有到 Input ACK；重启仍取 current exact binding，独立回复取消与 drain 由 Task 和默认回复测试保护。 |
| `mobile_inbound_reserves_before_bus_queue_and_deletes_after_terminal`、`mobile_delete_retry_retains_exact_and_session_owners` | 替换为 Input 提交前耐久预留、提交后清理失败不推翻接纳；交接行成功删除前保留 exact lease 与 Session admission。 |
| `mobile_handoff_recovers_through_current_exact_binding`、`mobile_recovery_redelivers_existing_turn_without_duplicate` | 重启对尚未/已经提交的 Input 都走实际 Channel ingress；原 Message 身份/seq 不变，只收束传输，不重跑模型或发送。 |
| `mobile_restart_missing_session_keeps_visible_handoff`、`mobile_same_process_recovery_does_not_duplicate_live_owner` | 缺失 Session 不复活、不删除原行；分页越过 live owner 后继续处理，失败不遗留阻止重试的 claim。 |
| `mobile_bus_close_retains_durable_handoff_for_next_boot`、`mobile_mark_pending_race_with_close_cannot_queue_after_shutdown` | 提交前取消/关闭保留附件与交接行，释放进程资源；durable lock 等待者不能在关闭后提交 Input。 |
| `mobile_delete_failure_then_bus_close_keeps_durable_row`、`mobile_completion_cancellation_waits_for_exact_cleanup` | 已提交 Input 不撤销；取消等待收束，失败后关闭仍保留下一次启动的恢复证据。 |
| `channel_worker_projects_and_closes_attachment_lease` | 新用例使用真实 ArtifactStore 导入与 Message yoyo，经过实际 Channel ingress 核对 artifact_ref、数据库引用、文件保留与 read lease 关闭。 |

保留的 Session override 拒绝测试改走 `prepare_channel_input`；仍保护 durable handoff 与 envelope 的 Session 一致性。公开 `companion_mobile_receipt_contract` 保留原 Mobile storage/channel、有效 bus 测试和 mutant，再加入新输入与独立回复测试，没有缩减公开场景。完整生产启动、实时客户端和 Delivery 由后续层累计验收。

## 2026-09-04：移除固定测试预算门槛

1080 项 Python、62 项 Web 和 72 个 Python 测试文件是 2026-09-02 清理的历史快照，不再是当前合同。删除固定数量检查、Python 保留清单和 Web 数量断言；CI 继续运行仓库实际存在的 Python 测试，Web runner 自动发现源目录下的 `.test.mjs` 文件。后续测试只按用户可观察回归、非平凡不变量、边界或具体 bug 保留，新增或删除不因数量本身失败。

本批次恢复点：`/mnt/data/akasic-agent-backups/test-cleanup-followup-20260904-before/pre-cleanup.bundle`，SHA-256 `3af27673dc5b20ee969ab16cb7a7d32154bed9ef1f32d9df5976e1a55988a6f7`。

## 2026-09-02：保留最高价值的三分之一

### 结果

| 范围 | 清理前 | 保留 | 删除 | 预算结果 |
| --- | ---: | ---: | ---: | --- |
| Python | 3239 项 / 250 文件 | 1080 项 / 72 文件 | 2159 项 / 178 文件 | `ceil(3239 / 3) = 1080` |
| Node | 194 项 / 34 文件 | 62 项 / 4 文件 | 132 项 / 30 文件 | 低于三分之一 |
| PR CI job | 8 | 2 | 6 | 低于三分之一 |

当时的 Python 1080 是仓库完整收集数，不是从完整套件中挑出的 PR 子集。旧实现中的 `scripts/check_test_budget.py` 同时固定数量和文件集合；任何未列入 `tests_scenarios/contracts/retained-test-files.txt` 的新测试都会使 CI 失败。Node 当时只保留 mobile message state、pairing response schema、Web transport 和 Akasha mobile UI 四个行为边界，由唯一命令 `npm run test:web` 执行。

删除的精确路径以本次提交的 delete diff 为准。Python 删除清单 SHA-256 为 `e807c64144b4693959d85edd23bea2832832ad138e662cab752cd55c8a967785`，Node 删除清单 SHA-256 为 `9b6cc344774d16dbd7d4f9a4e2bc154c1c7285ef5434aaccfb905d786b1c01d1`；摘要基于排序后的仓库相对路径，每行一个。

### 保留理由

保留清单不是按文件大小或覆盖率生成。每个文件至少拥有以下一种高价值失败：

- `tests/semantic/**`：P0 mutant/oracle、非破坏历史、模型 owner、递归插件验证和 change-impact Gate 自身的 fail-closed 合同。
- `tests/control/**`、`test_session_store.py`、`test_message_bus_admission.py`：Turn admission、同 session 排他、跨 session 并发、中断、重放、终态一次性和消息只追加。
- `test_plugin_hot_reload.py`、`test_plugin_install.py`、`test_plugin_generation_job_host.py`、`test_plugin_managed_process_host.py`、`test_plugin_runtime_control.py`、`test_plugin_turn_rollout.py`：插件 v3 generation、lease、promotion、rollback、卸载、进程清理和崩溃恢复。
- `mobile_realtime/**`、`test_web_chat_channel.py`、`test_channel_attachment_store.py`、`test_durable_deliveries.py`：真实入口的认证、附件、游标、持久交付、跨客户端身份和 exactly-once/finality。
- `test_context_compaction_contract.py`、`test_session_compaction_runtime.py` 及迁移测试：历史正文不得因裁切或迁移减少，迁移链必须 append-only 且可从旧状态恢复。单项迁移测试数量小，但保护不可逆数据变换。
- `test_agent_restart.py`、`test_mcp_process_recovery.py`、`test_rolling_backup.py`、`test_runtime_smoke.py`：监听器归属、子进程 epoch、备份恢复和跨层启动/关闭失败语义。
- `test_shell_tool.py`、`test_unified_exec.py`、`test_tool_executor.py`：外部进程、权限、取消和输出 finality 的信任边界。
- `mobile-message-state.test.mjs`、`mobile-pairing.test.mjs`、`web-chat-transport.test.mjs`、`test_akasha_mobile_ui.mjs`：用户真正看到的消息身份、外部 pairing 响应校验、流式终态、草稿/阅读锚点和 Akasha 查询边界。

最后一次等额调整用 138 项更高价值边界替换 138 项内部覆盖：加入 rolling backup、MCP process recovery、attachment store、durable delivery、真实 Web ingress 和 runtime smoke；移出 MCP slot、turn pipeline、composition wiring、reload journal 以及重复的 mobile adapter/publisher 组合。数量不变，但对灾难恢复、进程恢复、权威附件、交付 finality 和真实入口的保护更强。

独立复审又完成两次等额交换：用 Web pairing 的外部响应 schema 边界替换一项通知文案字面测试；用 2 项正式 credential/ref 冻结与原始配置 revision drift 测试替换 2 项 injected requester wiring 测试。它们分别保护不可信网络输入和插件 secret/config 的 TOCTOU 边界，优先级高于展示字符串与依赖注入接线。

### 删除理由

被删除测试按主要理由归入以下类别。一个文件可能同时符合多项；删除仍有取舍，不声称它们完全没有价值。

| 删除类别 | 主要路径示例 | 为什么在 1080 预算外 |
| --- | --- | --- |
| 实现镜像与分层重复 | `test_agent_core_p*.py`、`test_plugin_composition_*.py`、`test_*_modules.py` | 固定 helper、slot、wiring、字段转发或显然控制流；同一可观察合同已在 runtime、control、generation 或 semantic 边界保留。 |
| 字面量、schema 与 catalog 枚举 | `test_plugin_static_manifest.py`、`test_plugin_config_schema.py`、`test_model_catalog_reader.py`、theme/module-boundary Node 测试 | 主要镜像常量、映射、导出列表或静态形状；真实加载、安装、协议拒绝或 UI 行为边界优先。 |
| 重复 adapter/client 组合 | `test_channel_base.py`、`test_channel_clients.py`、`test_core_channel_adapter.py`、mobile gateway/pairing/publisher 测试 | 相同身份、鉴权、交付和 publication 语义已由 Web/mobile 真实入口及持久存储边界覆盖。 |
| 已移除或历史过渡面 | `test_workspace_mcp_removed.py`、`test_plugin_v3_only_surface.py`、shadow/legacy migration 辅助面 | 仅证明旧入口不存在或过渡实现仍在；没有持续的公共 absence 合同则不占长期预算。真正不可逆的数据库迁移仍保留。 |
| 宽矩阵与低增量排列 | provider/model 普通安装组合、plugin composition 各 slot 组合、UI state 细分 Node 文件 | 多个用例沿同一路径只替换插件、provider 或状态枚举；保留最能穿过公共边界和失败路径的代表。 |
| benchmark、性能与部署演练 | `tests/benchmark/test_harbor_*.py`、WebUI performance `.test.mjs`、container/release rehearsal 测试 | 它们是专项测量或环境验收，不是每次源码变更都必须固定的核心回归；正式性能或发布验收应由独立、带真实环境证据的流程拥有。 |
| 被更高层 finality 覆盖 | `test_turn_pipelines.py`、`test_turn_effects.py`、`test_content_store.py`、部分 wake/drift 与 support 测试 | 保留 ConversationRuntime、SessionStore、durable delivery、semantic mutant 和 wake durable 边界，避免在下游重复验证同一 owner。 |

主动放弃的检测粒度包括：每种 provider/plugin 的对称安装排列、每个 composition slot 的内部快照、全部桌面 UI 小状态、benchmark controller 细节，以及部分旧 CLI/部署 helper。若这些区域以后发生具体生产 bug，应优先在现有公共边界补一个回归，并从 1080 预算中移出更低价值测试，而不是扩大总数。

### Gate 清理

- 普通 PR 从 8 个 job 收敛到 `check-and-test` 与 `change-impact-gate` 两个。2026-07-18 引入的统一 Gate 已能按 diff 选择 P0 mutant/oracle 并对未知映射 fail closed，因此保留；它是当前 Core 变更的单一语义 owner。
- 2026-07-14 的 control 三连跑和 restart soak、2026-08-18 的 static fleet、2026-08-15～16 的旧 composition 不再进入每个 PR。control/restart 已合并为一个每周 lifecycle job；旧 composition 已由当前 plugin lifecycle、hot reload 和可观察插件边界覆盖。
- 手动候选 workflow 从 4 个 job 收敛到 1 个，只运行 fleet completeness、Mobile 和公共 WebUI。它们分别固定全部 18 个锁定插件的来源/v3-only/retired 排除、用户可见 Mobile ABI，以及 Citation/Meme 的真实公共 WebSocket 行为；不重复 1080/62 回归。
- 2026-08-18 引入的 E1/E2 在 2026-09-02 Core 删除 v2 compatibility 后失效：锁定的 Emotion 仍导入已删除的 `CoreEvent`，Calendar 仍导入已删除的 `PROACTIVE_COMPONENTS`。这两条失败固定的是历史 API，不是当前可观察回归，因此删除 `plugin_v3_e1_gate.py` 与 `plugin_v3_e2_gate.py`，不通过升级外部插件来维持 Gate。E4 硬依赖 E1 报告和仓库中从未存在 runner 的 E3 报告，不能执行其发布合同，也删除 `plugin_v3_e4_gate.py`。恢复方式是 revert 本清理提交；若未来需要正式发布 rehearsal，应以当时的 Core、锁定插件和真实部署输入重新建立合同。
- Terra 复审发现 static fleet 对旧锁会假阳性；因此 fleet lock 前移到 Calendar `048c8e8`、Emotion `d828fd7`、Observe `09214c2`、Feed `dccbcd9`、Fitbit `e0eda11` 与 Steam `d2ddd1b` 的当前正式 main，Mobile lock 同步其中的 Emotion、Observe 和 Fitbit。Fleet Gate 新增对所有生产源码 `from agent.* import ...` 的当前 Core export 检查，已删除符号会 fail closed；它只声称 source/API compatibility，不冒充完整运行加载或正式部署。
- 2026-08-15 的 `plugin_composition_v3_gate.py` 只有历史文档调用者，并重复固定 Tool/plugin snapshot 排列，因此物理删除。`plugin_passive_composition_v3_gate.py` 不再作为独立 CI Gate，但公共 WebUI runner 真实复用它的 exact source、装配和摘要 helper；clean-head 验证暴露这一动态模块依赖后已恢复，避免为了删文件复制同一套逻辑。
- `programmatic-control-nightly.yml` 改为每周唯一的 full-process lifecycle job，顺序运行 failure matrix、100-turn resource soak 与 restart soak；进程级 SIGTERM/crash、workspace lock 和资源泄漏因此仍有明确 owner，但不阻塞每个 PR。
- semantic scenario 与 Content/Wake lock/H5 manifest 都只引用仍保留的测试；已删除的 slot、pipeline、gateway、shadow 和 support 测试不再被 Gate 间接复活。
- 正式 workspace 演练不再由缺失前置报告的仓库脚本占位；未来需要时由拥有部署输入的发布流程重新建立可执行合同。本次清理不伪造发布通过。

### 恢复与验证

清理前恢复点：`/mnt/data/akasic-agent-backups/test-gate-one-third-20260902-before-clean/pre-hard-budget-71b27f5b.bundle`，SHA-256 `78c213310dc94c8ee5a16da65f8dd25c4dc0078aab7bb965cb772b91001ed7f5`。更早的完整测试归档为同目录 `test-and-gate-surface.tar.gz`。

本地验证：预算检查为 `python_files=72 python_tests=1080 node_files=4`；最终等额交换后的 Python 全量为 `1075 passed, 5 skipped`（155.87 秒），Node 为 `62 passed`。Python/测试/SDK Pyright、TypeScript、control schema、Yoyo append-only、SDK 11 项测试、workflow YAML、Gate audit 和 `git diff --check` 均通过；受保护合同变化触发的 27 个公开场景也通过。Terra xhigh 独立复审提出的 fleet coverage、pairing/credential swap、full-process lifecycle owner 和活跃文档悬空引用均已修正，代码与文档 P0/P1 清零。提交后仍需远端 CI 对精确 head 验证。

## 2026-09-07：旧执行图与插件文档清理

### 清理范围与保留结果

本批次只清理当前候选中已无生产消费者的旧执行图和与其绑定的文档入口：

| 删除或退役的图 | 原因 | 当前承接 |
|---|---|---|
| `agent/core/passive_turn.py`、`agent/core/passive_support.py`、`agent/core/prompt_block.py`、`agent/core/response_parser.py`、`agent/core/runtime_support.py` | 被 Message → source → reply/react 的插件链替代，旧固定 Passive Turn 编排已无当前调用入口 | `plugins/sources/`、`plugins/conversation/`、`plugins/reply/`、`plugins/react/`、`plugins/context/`、`plugins/content/` |
| `agent/lifecycle/` composition/facade/phase/types 及 `agent/plugin_composition/turn_lifecycle.py` | 删除固定 Before/After lifecycle 与 Turn 业务身份，避免第二套执行控制流 | `agent/plugin_composition/` 的 Context/Fiber/Task/Effect 与插件自身 Service |
| `agent/retrieval/events.py`、`agent/retrieval/protocol.py`、旧 `agent/turn_events/observe.py` 接入 | retrieval/observe 旧事件不是当前 Message/Turn owner，也没有新 Core 发布证据 | `plugins/turn_projection/`、`plugins/akasha/` 及各自 typed signal |
| Mobile 旧 stop/interrupt 注入、生产 `_Bus` 假设及其过渡辅助 | 真实 ChannelRuntimePorts、MessageCatalog 和 recoverer 已承接输入与恢复；旧队列模型会误导新入口 | `infra/mobile_realtime/`、`session/`、Channel/来源插件 |
| 旧 Memory2/runtime helper 路由 | 退役能力不再进入当前插件组合；其历史数据仍按状态地图和恢复合同保留 | `plugins/compaction/`、`plugins/markdown_memory/`、`plugins/akasha/` |

当时的核对结果：`core.tool_catalog` 不是兼容残留：`agent/plugin_composition/tool_catalog.py` 的 `TOOL_CATALOG` 仍由 manager 提供，并在 snapshot generation、freeze、activation 和 lease 路径消费；它是 Core 内部组合服务。插件公开工具合同是 `plugins.tools` 的 `TOOLS`（`tools.v1`），不是这个内部 key。当前普通出站 owner 是 `plugins.delivery`，提供 `DELIVERY_SENDERS`、`DELIVERY`、`FINAL_OUTPUT_DELIVERY` 和 `DELIVERY_READ`；`agent/plugin_composition` 的 `DELIVERIES` / `DURABLE_DELIVERIES` 仍由 manager/static candidate view 提供，但当前生产插件不再 import，只有 manager、导出和测试保留它们。`AFTER_TURN_COMMITTED` 仍有当前定义与测试残留，外部 Observe/Proactive Feedback stable artifact 的迁移仍待完成；这些旧事件和兼容导出不能证明新 Core 会发布对应业务事实。

**2026-09-13 更新：** 上述工具保留依据已由后续实际迁移解除：Emotion 与 GitHub Watch 源码改用普通 `tools.v1`，Core 的旧 `TOOL_CATALOG`、注册器及 snapshot wiring 已删除。PF 源码改用 Message 跟随与 Turn projection，`SESSION_READ` 也已删除。历史数据库和 settlement 保护路径没有随接口删除而减少；最终外部运行验收见[插件边界地基](../design/plugin-boundary-foundation.md)。本节旧 revision 和保留判断只描述 2026-09-07 的候选。

### 写入集与恢复

本次文档修正未修改数据库、yoyo、workspace、plugin-data、Android 源码、外部插件 checkout、安装 cache 或生成 bundle。此前 stacked code commits 可能包含各自 owner 的 yoyo 迁移；本句只描述本次文档写入集。删除代码不减少既有 Message、学习、附件、receipt 或 plugin-data。第 10 层已有分范围的 programmatic smoke、MC01 和 G5 证据；完整 MessageLog 启动与每周 lifecycle/restart probe 已由下方 clean run 验收，Android 配套、外部插件迁移和正式切换仍未验收。

代码清理的实际历史是：`1492ce5f` 为 `7340e5a0` 的父提交，`7340e5a0` 删除旧 lifecycle/passive graph，随后 `576e8add` 删除旧 event/retrieval leaves。`5e1b1b93` 是 Message 行为与 Gate 元数据的候选基线，不是这些代码删除的恢复点；如需回放文档/行为候选，可将其作为参考提交，不能据此恢复已删除代码。文档修改前的逐文件恢复副本位于 `/tmp/akasic-agent-backups/docs-cleanup-20260907-before-edit/`，包含本批次涉及的六份文档。恢复时先停用候选运行时，再按 Git worktree 和文档副本逐项回放，不能触碰正式 workspace。验证至少包括 `git diff --check`、相对链接检查、旧入口搜索，以及与当前实现直接相关的文档/API路径核对。

## 2026-09-07：MessageLog 第 10 层 Gate 分层

- 当前候选已分别完成 programmatic control smoke、MC01 memory-context 和 G5 programmatic Message soak。它们各自验证来源接纳、MessageLog 追加/投影和受控失败边界；随后 clean run 又完成了完整 MessageLog 启动与 lifecycle/restart probe 验收。
- MC01 在 admission 时显式传 `persist_memory=true`，并核对 `learning=eligible`；G5 soak 使用默认 `persist_memory=false`，并核对 `learning=excluded`。两种 Session 资格是有意分开的合同，不能把 G5 的默认排除写成 MC01 结果。
- 当时 `agent/plugin_composition/tool_catalog.py` 的 `core.tool_catalog` 仍被 manager、snapshot generation、freeze、activation 和 lease 消费；active `content-source-interop.lock` 的 `emotion` revision `2bb332b7` 仍以该边界验收，公开插件/工具合同是 `tools.v1`，因此该 leaf 不能按名称相似或局部无调用就删除。
- FrameBook 断线时移除 active route，将原始 `ConnectionError` 留给 live claim；RestartWatcher 先 `gate.prepare`，等待 Turn complete 后在 claim drain 与普通 delivery 之间分支，`gate.commit` 成功后才消费 programmatic claim。详细 owner 合同见 [消息日志设计](../design/0902-reviewed-v4.md) 与 [Linux 自重启设计](../design/linux-supervisor-safe-self-restart.md)。
- clean Core `7189f19fb7ba385e37932e2d08c4c9a56c86942e` 的 run `20260907-182121-3efdcba6` 已通过完整启动与每周 lifecycle/restart probe：primary `103/103`（20 轮）、`unsupervised=true`、5 个 failure mode 全通过，`inside`/`unsupervised`/`failures`/`cleanup` 均为 0，残留 containers/networks/volumes 为空，`repositoriesUnchanged=true`。报告位于 `/mnt/data/coding/akasic-agent-worktrees/message-plugins-10-restart-probe/docker/debug/reports/restart/20260907-182121-3efdcba6/`，其中 `sourceDigest=sandboxAppDigest=a6de60971668fdabc0efc8a732050335f774d8e65c61177ab128f38330c44fbd`、`fileCount=1395`；完整 change-impact Gate 的最终提交与源码摘要由对应报告和 PR 记录。Android 原生配套、外部插件源码迁移、旧 workspace/历史效果转换和正式 workspace 演练仍是正式切换前提。
