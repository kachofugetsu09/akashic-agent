# 正交化测试基线：保留清单、补充清单与 Gate 削减

- 基线：`main@c1eb108`
- 范围：`tests/`（232 个文件、1941 个测试函数、约 9.9 万行）、`tests/semantic/`、`tests_scenarios/contracts/`、`docker/debug/gate.py`、`.github/workflows/`
- 用途：按本清单保留，清理其余全部测试与 Gate 项。本清单是 `tests/` 与 CI 的权威保留依据。
- 关联：[#766](https://github.com/kachofugetsu09/akashic-agent/issues/766)、[`projectneed.md`](../projectneed.md) PLG/SES/CTX、[测试与 Gate 清理账本](test-gate-cleanup-ledger.md)（已被本文件取代）

## 1. 选择标准

保留一个测试，必须同时满足：

1. 它守护一条**概念不变量**，而不是某个功能是否正确。概念来源只有三处：
   - **O（正交定义）**：改变一个能力时，无关能力不变（删 Memory 仍能回复、换 Channel 不动模型、换 `react` 不动 Memory……）。
   - **C1–C5（Core 只守五件事）**：C1 插件接线/卸载/generation 切换；C2 Message 怎样进入 Turn；C3 Turn 身份、取消与终态；C4 Session 追加不可改写事实；C5 外部发送留回执、结果未知时保持 `uncertain`。Core 不含 passive、Wake、Scheduler 等来源的业务分支。
   - #766 的正交化方向：名字无特权、声明图等于使用图、执行归属由框架统一、可选能力解耦。
2. 它在**真实组合**上断言：真实 `CompositionRoot`、真实 `PluginManager` 安装链或真实 `MessageLog`。玩具插件可以，手工构造的假状态不行。
3. 同一条不变量只留一到两个最直接的测试。一个在内核层（玩具插件），必要时再加一个在安装链层（真实安装）。

以下情况一律不保留：

- 功能正确性，例如 Mobile 协议、Shell 执行、MCP 进程树、WebUI 发布、Markdown Memory 抽取、Scheduler 容量、Dashboard 渲染。
- 许可机制的实现细节，例如 retain 顺序、capture/enter/close 状态机、各种 cancel 竞态路径。它们描述的是"现在怎么实现"，#766 方向 B 落地时会整体改写。
- **自证式变异测试**：`tests/semantic/test_contract_oracles.py`、`test_companion_contract.py`，以及 `test_context_history_contract.py` 的 mutant 用例。它们在手工构造的假状态上检验 oracle 函数会不会报错，没有调用任何产品代码。例如 `test_plugin_data_oracle_rejects_uninstall_delete_mutant` 是测试自己先 `rmtree`，再断言 oracle 报错。这类测试只证明检查器会报错，不证明系统没有越界。
- Gate 元治理测试：`tests/semantic/test_change_gate.py` 整个文件（路径映射、超时、catalog 完整性）。
- 静态检查器自身的单元测试：`test_plugin_boundary.py`、`test_yoyo_migration_append_only.py`。检查器脚本本身留在 CI 中直接运行（见 §4）。

## 2. 保留清单（41 个测试节点）

### 2.1 插件内核：接线、局部更新、失败局部化、清理（C1）

这些测试用玩具插件跑在真实 `CompositionRoot` 上，是"正交能力 + 插件生命周期"最底层的保证。

| # | 测试节点 | 守护的概念 | 保留原因 |
|---|---|---|---|
| 1 | `tests/test_plugin_local_lifecycle.py::test_local_replace_only_affects_changed_and_consumers` | O：正交定义本身；PLG-001、PLG-005 | 直接断言替换 provider 时，只有它和它的消费者重启，无关插件的 apply 次数为 0。这是"改一个能力，无关能力不变"在内核上的原型。 |
| 2 | `tests/test_plugin_local_lifecycle.py::test_optional_child_restarts_without_host_restart` | PLG-014 可选能力；#766"可选能力解耦" | 可选子能力更新时宿主不重启。#766 场景 2（UI 更新不牵连 models）就是它在真实插件上的实例，这是那条修复的内核底座。 |
| 3 | `tests/test_plugin_local_lifecycle.py::test_loading_owner_not_consumable_until_active` | PLG-012/013 accepted 不等于 active | 处于加载中的 owner 不能被消费。防止"已提交"被误当成"已生效"。 |
| 4 | `tests/test_plugin_local_lifecycle.py::test_failed_new_apply_does_not_restart_old_instance` | PLG-004 失败局部化 | 新版本 apply 失败时，旧实例不被连带重启，失败不外溢。 |
| 5 | `tests/test_plugin_local_lifecycle.py::test_late_provide_failure_is_local_to_required_downstream` | PLG-004 失败只影响硬依赖分支 | 失败只传播到硬依赖它的下游，旁支不受影响。这是"依赖图决定影响范围"的直接证据。 |
| 6 | `tests/test_plugin_local_lifecycle.py::test_missing_conflict_and_cycle_are_observable` | PLG-008 冲突、缺依赖、循环必须显式失败 | 缺依赖、重复提供者、循环依赖三类组合错误必须显式可见，不能静默挑一个。依赖图正确性的底线。 |
| 7 | `tests/test_plugin_local_lifecycle.py::test_cleanup_failure_keeps_owner_and_explicit_retry_succeeds` | PLG-006 清理失败保留 owner | cleanup 失败时不能假装已卸载，owner 保留到显式重试成功。 |
| 8 | `tests/test_plugin_local_lifecycle.py::test_inflight_call_blocks_affected_cleanup_only` | PLG-003、PLG-005 | 进行中的调用只阻塞受影响插件的 cleanup，不冻结整棵树。局部更新和正交性在并发下的形式。 |
| 9 | `tests/test_plugin_local_lifecycle.py::test_inflight_call_reads_own_activation_deps_during_drain` | PLG-003 在途调用绑定其 activation | 旧调用在 drain 期间读到的是自己 activation 的依赖，不会中途串到新版本。 |
| 10 | `tests/test_plugin_local_lifecycle.py::test_stale_context_loses_capabilities_after_reload` | PLG-003 旧引用不静默指向新实例 | 重载后旧 Context 失去能力，防止旧句柄悄悄操作新 generation。 |
| 11 | `tests/test_plugin_local_lifecycle.py::test_runtime_owner_uses_local_owner_call_and_rejects_inherited_or_foreign_context` | PLG-003 子任务和外来上下文不继承归属 | 守护"执行归属不能被继承或冒用"。**注意**：#766 方向 B（执行归属由框架统一）落地时，按新语义重写这个测试，不要删除它守护的概念。 |
| 12 | `tests/test_task_scopes.py::test_raw_child_does_not_inherit_parent_plugin_scope` | PLG-003 无主后台任务不得意外继承绑定 | 裸 `create_task` 出来的子任务不继承父插件作用域。和第 11 条一起定义执行归属的边界；方向 B 落地时同样按新语义重写。 |
| 13 | `tests/test_plugin_composition_lifecycle.py::test_failed_consumer_cleanup_keeps_provider_until_explicit_retry` | PLG-006 反向依赖清理 | 消费者没清理干净之前，provider 不能先被拆掉。清理顺序跟随依赖图。 |
| 14 | `tests/test_plugin_composition_lifecycle.py::test_pending_initial_dependency_resolves_then_frozen_teardown_closes_consumers_first` | PLG-006、PLG-014 | 依赖晚到时消费者等待而不是失败；拆除时消费者先关。一个测试覆盖依赖等待和反向拆除两件事。 |
| 15 | `tests/test_plugin_composition_lifecycle.py::test_emit_event_listener_failure_is_fail_loud` | PLG-014 事件派发合同（emit） | 五种派发语义里唯一现有的直接测试：emit 的 listener 失败必须传播，不能被吞掉。 |
| 16 | `tests/test_plugin_composition_lifecycle.py::test_event_bus_does_not_bridge_into_plugin_composition` | 同一事实一个 owner | 旧 `EventBus` 不能成为绕过插件组合的第二条通道。按 #766 删除 `EventBus` 时，这个测试一起删。 |

### 2.2 安装链：内置等于外部、名字无特权、局部更新与卸载（C1，PLG-016）

这些测试走真实 `PluginManager` 或安装链，证明内核的保证在真实插件上依然成立。

| # | 测试节点 | 守护的概念 | 保留原因 |
|---|---|---|---|
| 17 | `tests/test_plugin_external_acceptance.py::test_legal_subsets_run_separately_and_accept_differently_named_provider` | O：正交；PLG-016；#766 场景 9 | 最重要的一个：两个独立安装根各自跑合法子集；一个异名 provider（`alternate`）只按公共合同编写，不需要原包名和源码，就能被同一个 consumer 使用。这是"everything is plugin + 名字无特权"的端到端证明。 |
| 18 | `tests/test_plugin_external_acceptance.py::test_business_composition_writes_reads_and_replaces_provider_from_new_generation` | PLG-016、PLG-005 | 外部安装的插件在新 generation 中替换 provider，业务读写保持连续。证明局部更新对外部插件同样成立。 |
| 19 | `tests/test_plugin_fresh_root.py::test_live_update_replaces_only_target_fiber_in_one_root` | PLG-005；决策 0072 单 live Root | 真实安装链上只替换目标 Fiber，对等插件保留原 owner，不重建 Root。 |
| 20 | `tests/test_plugin_hot_reload.py::test_cold_selected_import_failure_retains_failed_generation_and_peer` | PLG-004、PLG-012、PLG-013 | 选中版本导入失败时，失败被记录，对等插件继续运行。安装链层的失败局部化。 |
| 21 | `tests/test_plugin_hot_reload.py::test_live_selection_compile_abort_then_commit` | PLG-007 发现不等于提交 | 编译中止时 selection 不变，之后提交才生效。selection 是唯一持久输入这件事的直接证据。 |
| 22 | `tests/test_plugin_uninstall_root_drain.py::test_uninstall_accepts_before_target_owner_drains_and_preserves_peer` | PLG-012 accepted 不等于 active；PLG-001 | 卸载请求先被 accepted，目标 owner 随后 drain，期间对等插件不受影响。 |
| 23 | `tests/test_plugin_uninstall_root_drain.py::test_uninstall_cleanup_failure_keeps_owner_and_explicit_retry_finishes` | PLG-006 | 卸载路径上 cleanup 失败保留 owner，显式重试后完成。卸载是独立的概念操作，和第 7 条不重复。 |
| 24 | `tests/test_plugin_install.py::test_plugin_enable_disable_and_uninstall_preserve_data` | PLG-010 卸载保留 plugin-data | 插件生命周期操作不能删除持久数据，属于持久化硬边界。 |
| 25 | `tests/test_migration_runner.py::test_core_only_cli_restarts_after_creating_runtime_data` | C：Core 不含业务；PLG-016 | Core-only 可以启动和重启，并且不会隐式发现仓库里的插件源码。证明 Core 离开任何插件都能独立存在。 |
| 26 | `tests/test_commands_provider.py::test_core_only_manager_does_not_supply_commands` | C：Core 不隐式提供业务能力 | 没装插件时 `COMMANDS` 就是 `None`，Core 不偷偷兜底。 |
| 27 | `tests/test_commands_provider.py::test_binding_keeps_selected_provider_handler_and_dependency_closure` | 名字无特权 | 把 commands 插件改名为 `human_actions` 后，它仍是 `COMMANDS` 的合法提供者。能力按服务合同绑定，不按包名。 |
| 28 | `tests/test_plugin_rpc.py::test_plugin_method_cannot_replace_host_management` | 非特权插件的另一面：插件也不能获得特权 | 插件不能注册 `plugin/install` 这类宿主管理方法来冒充控制面。 |
| 29 | `tests/test_tool_bindings.py::test_prepare_and_authorize_follow_exact_registration_identity` | PLG-018 工具按引用而非名字 | 同名的新工具不能继承旧注册的参数转换或限制。授权跟随注册身份，不跟随名字。 |

### 2.3 Message、Turn 与 Session：来源正交、Turn 终态、只追加事实、外部发送回执（C2–C5）

| # | 测试节点 | 守护的概念 | 保留原因 |
|---|---|---|---|
| 30 | `tests/test_default_reply.py::test_installed_default_reply_is_an_independent_log_consumer` | C：Core 无 passive 分支 | 默认回复只是 `MessageLog` 的普通消费者，Core 里没有"被动回复"特权路径。 |
| 31 | `tests/test_delivery_policy.py::test_real_input_reply_and_archived_delivery_are_independent_consumers` | O：换 Channel 或 Delivery 不动回复 | 回复与投递归档是两个独立消费者，互不依赖。 |
| 32 | `tests/test_content_protocols.py::test_meme_and_citation_are_independent_of_registration_order` | O：对等插件互不感知 | 两个对等内容协议的结果与注册顺序无关，证明对等插件之间没有隐式耦合。 |
| 33 | `tests/test_message_push_plugin.py::test_push_completes_while_target_turn_is_active_and_appends_one_output` | AGENTS：message_push 只是 Message 来源 | 目标 Turn 正在运行时 push 仍能完成，且只追加一条输出。证明 message_push 没有复制出第二套执行模型。 |
| 34 | `tests/test_conversation_source.py::test_interrupt_inputs_survive_and_old_output_cannot_commit` | C3 Turn 取消与终态 | 中断后输入不丢，被取消 Turn 的旧输出不能再提交。Turn 终态唯一。 |
| 35 | `tests/semantic/test_context_history_contract.py::test_real_message_reply_preserves_history_embeddings_and_restart_seq` | CTX-001 上下文裁切是非破坏投影；SES-005 只追加 | 真实回复走完后，历史消息、embedding 和重启后的 seq 都不变。这个文件里只保留这一个节点，mutant 节点删除。 |
| 36 | `tests/test_message_log.py::test_concurrent_writers_allocate_one_sequence_per_fact` | SES-002 seq 单调唯一 | 并发写入时每条事实恰好一个序号。 |
| 37 | `tests/test_message_log.py::test_missing_resource_rolls_back_message_and_sequence` | SES-001 原子追加 | 资源缺失时消息和序号一起回滚，不留半条事实。 |
| 38 | `tests/test_message_metadata.py::test_unknown_metadata_survives_restart_history_and_follow_without_plugins` | O：删除插件后核心事实照常可读；SES 公共 Message | 写入元数据的插件不在场时，历史、重启和 follow 仍能完整读回。Session 事实不依赖插件存在。 |
| 39 | `tests/test_durable_deliveries.py::test_provider_receipt_precedes_one_append_only_session_projection` | C5外部发送留回执 | 先有 provider 回执，再有唯一一条只追加的 Session 投影。 |
| 40 | `tests/test_durable_deliveries.py::test_provider_started_sigkill_recovers_uncertain_without_resend` | C5结果未知时保持 `uncertain` | 发送中途被 SIGKILL 后恢复为 `uncertain`，不自动重发。外部效果不能被"恢复内存指针"伪装成已回滚。 |

### 2.4 静态边界（不是 pytest 节点，但属于保留项）

| # | 项 | 守护的概念 | 保留原因 |
|---|---|---|---|
| 41 | `scripts/plugin_boundary.py check`（plugin-boundary workflow） | R1–R5：Core 不导入插件、插件之间不互相导入、插件只用公开表面、ServiceKey 有角色 | 这是 import 层面的正交性，用 AST 静态检查最便宜也最可靠。它的单元测试 `tests/test_plugin_boundary.py` 可以删；检查器本身保留。 |

### 2.5 保留测试依赖的 helper（不能删的非测试代码）

清理时只删 `test_*` 函数，下列模块里的 helper 要保留，否则保留的测试会 import 失败：

- `tests/conftest.py`、`tests/fixtures/plugin_workspace.py`、`tests/fixtures/durable_delivery_crash/`（第 40 条要用）
- `tests/test_plugin_install.py`（被第 19、22、23、28 条 import）、`tests/test_plugin_fresh_root.py`（被第 22、23 条 import）、`tests/test_python_environment.py`（被 `test_plugin_install` import）
- `tests/test_default_reply.py`、`tests/test_delivery_bindings.py`、`tests/test_message_delivery.py`（被第 31、33 条 import）
- `tests/test_message_react.py`（被第 35 条 import）
- `tests_scenarios/contracts/oracles.py`：第 35 条要用。建议把用到的函数内联进测试文件，再删除整个 `contracts/` 目录。
- `docker/debug/plugin_external_acceptance.py`（第 17、18 条要用）

更干净的做法：把这些 helper 抽到 `tests/support/`，被引用的测试文件只留保留节点。清理完运行 `pytest --collect-only -q tests`，确认收集到 40 个节点。

## 3. 需要补充的测试（10 个，全部来自 #766 验收）

这些不变量现在**没有**任何测试守护，或者守护方式与 #766 的目标语义相反。每一项都在 #766 对应 PR 里落地，并先证明在当前 `main` 上失败（或按当前语义通过、按新语义失败）。

| # | 待补测试 | 守护的概念 | 为什么现有测试不够 |
|---|---|---|---|
| A1 | 不装 UI 插件时，models 和 embedding 仍能启动并完成一次调用 | O：删除无关能力，其余不变；#766 场景 1 | 当前 `plugins/models/plugin.py` 把 `UI` 写成硬依赖，这条会失败。现有测试全部在装了 UI 的组合下运行。 |
| A2 | UI 插件更新时，models 不重新 activation | PLG-005 局部更新；#766 场景 2 | 第 2 条只在内核玩具插件上证明，真实插件的依赖声明错了，没有测试能发现。 |
| A3 | 读取未声明的服务被拒绝，或必须显式借用 | #766"声明图等于使用图" | `Context.get` 目前会回退到全局 providers，没有任何测试断言"不声明就拿不到"。 |
| A4 | 异步 listener 运行到一半其 owner 被卸载：listener 被 drain，新调用被拒绝 | PLG-014 listener 随 Fiber 回收；#766 方向 B | `_listener_boundary` 目前只用于诊断，不持有执行归属。 |
| A5 | 插件不写任何手动 scope 代码，`spawn` 出的任务和跨 owner 调用照样被正确 drain | #766 执行归属由框架统一 | 现有测试（第 11、12 条）锁的是"手动 scope + 不继承"，这条补的是"框架代管"。方向 B 合入后，第 11、12 条按此重写。 |
| A6 | 两个插件各自有名为 `worker` 的子 Fiber，互不冲突 | 名字无特权；命名空间正交 | `context.py` 目前做全局重名检查。 |
| A7 | Binding 从 V1 切到 V2 时，在途调用和新调用分别遵循选定语义 | PLG-003；#766 binding 语义 | `bindings.py` 的 `open` 目前读的是当前 provider，语义没有被显式测试锁定。 |
| A8 | 禁用 `standard_web` 后，wake 仍然 active | O：删除无关能力，其余不变；#766 场景 8 | 当前存在跨插件的隐式依赖，没有测试覆盖这种子集组合。 |
| A9 | 五种派发（emit、serial、parallel、transform、observe）各有一个合同测试：谁能中断、失败如何传播、顺序是否确定 | PLG-014 五种派发合同 | 现在只有 emit fail-loud（第 15 条）和 lifecycle bail。另外四种语义没有测试，重构时最容易被悄悄改掉。 |
| A10 | 同一个 Turn 原子被 message_push、scheduler 和 spawn 三种来源复用，Turn 的身份、父子关系、取消和终态一致 | C3；#766 Turn 原子 | 第 33、34 条各自只覆盖一种来源；Turn 原子统一后，需要一个跨来源的等价性测试。 |

每一项只写一个测试。断言写在可观察结果上（服务是否可得、activation 次数、终态），不断言内部字段。

## 4. Gate 与 CI 削减

### 4.1 保留（6 项）

| 项 | 位置 | 保留原因 |
|---|---|---|
| `pytest -q tests`（此时只剩上面 40 个节点） | `ci.yml` "Run Python regressions" | 概念底线的唯一执行入口。40 个节点预计一分钟内跑完。 |
| `scripts/plugin_boundary.py check --base` | `plugin-boundary.yml` | import 层正交性（R1–R5）。 |
| `scripts/check_yoyo_migrations.py --base origin/main` | 从 `change-impact-gate` job 挪到 `check-and-test` | 迁移历史只追加，是持久化硬边界。它是脚本，不依赖 Gate 机制。 |
| `pyright --level error`（主工程和 tests 两份配置） | `ci.yml` | 类型即接口合同，成本低；插件与 Core 的 Protocol 边界靠它兜底。 |
| `generate_host_bridge_protocol.py --check`、`generate_control_schema.py --check` | `ci.yml` | 公共协议 schema 与生成物一致。这是对外边界，不是功能测试。 |
| `npm run typecheck` | `ci.yml` | 前端类型边界，成本低。 |

### 4.2 移除

| 项 | 位置 | 移除原因 |
|---|---|---|
| 整个 `change-impact-gate` job（`gate.py plan/run`） | `ci.yml` | 按路径选场景、Docker 隔离、报告摘要都是为 27 个功能场景服务的治理机制。保留测试只有 40 个，直接全跑，不需要影响分析。 |
| `docker/debug/gate.py` 的 change-gate 部分，以及 `docker/debug/docker-compose.change-gate.yml` | `docker/debug/` | 同上。`gate.py` 共 1300 行，其中只服务于 change gate 的部分一起删。 |
| `tests_scenarios/contracts/`：`impact.toml`（31 个 group）、`scenarios.toml`（27 个场景、约 38 条 mutant 映射）、`state-contracts.toml`、`coverage-baseline.json`、`oracles.py` | `tests_scenarios/contracts/` | 1780 行映射与元数据，只为 change gate 服务。`oracles.py` 先内联进第 35 条再删。 |
| 27 个 Gate 场景（contract_catalog、workspace_bootstrap、shell_*、host_bridge_*、mobile_*、mcp_*、plugin_generation、plugin_workload、model_owner、plugin_domain_observe_events、plugin_uninstall_drain_finality、memory_persistence、recursive_plugin_self_validation、companion_* 共 9 个、content_wake_delivery、context_history_nondestructive） | `scenarios.toml` | 其中和概念相关的节点已经逐个挑进 §2；其余是 Shell、Mobile、MCP、Companion 等功能正确性检查。 |
| `tests/semantic/` 除第 35 条以外的全部内容 | `tests/semantic/` | 自证式变异测试和 Gate 元治理测试，理由见 §1。`test_model_owner_contract.py` 的 import 约束已被 plugin-boundary R1 覆盖。 |
| `npm run test:web` | `ci.yml` | 前端功能单元测试，不守护插件正交性。 |
| `sdk/python` 的 `pytest -q tests` | `ci.yml` | SDK 功能测试。对外协议已由 `generate_control_schema.py --check` 和 SDK pyright 守住。SDK pyright 保留。 |
| `plugin-v3-candidate-gates.yml` | `.github/workflows/` | 手动触发，服务于已被 0072 单 live Root 取代的 candidate 模型；fleet、Mobile、WebUI 都是功能端到端检查。 |
| `programmatic-control-nightly.yml` | `.github/workflows/` | 每周 failure-matrix 和 soak 属于稳定性和功能压测，不是概念底线。需要时手动跑脚本即可。 |

`computer-image.yml` 是镜像发布流程，不是 Gate，不在本清单范围内。

### 4.3 清理后需要同步修改的文档与规格

这些文档把现行 Gate 写成了硬性要求，不同步修改就会和新基线矛盾：

- `docs/WORKFLOW.md` §5 Gate、Verify 行（第 56 行）、Deliver 行（第 59 行）和第 162 行：把"change-impact Gate"改为"概念基线 pytest + 静态检查"，删除 `gate.py` 命令与报告摘要要求。§6 Review 模式里的"只读概念 Gate"（架构 PR 的独立审查）是人工评审，不是自动化测试，保留。
- `docs/projectneed.md`：TST-003（用已知错误验证验收器）、TST-006（变更影响由版本化 Gate 决定）与新基线直接冲突，需要维护者改写或废止。TST-001（oracle 独立于实现）仍然成立，本清单的第 2 条选择标准就是按它执行的。
- `docs/refactor/test-gate-cleanup-ledger.md`：标记为已被本文件取代。

## 5. 以后的新增规则

1. 默认不写单元测试。功能是否正确，靠真实运行和 scenario 验证。
2. 只有两种测试可以进入 `tests/`：
   - 某条概念不变量的回归复现：先在出错的提交上失败，再在修复后通过。
   - §3 清单里的补充项。
3. 新增测试的 PR 必须写明守护的是 §1 里的哪一条概念，并说明为什么现有 41 项守不住。说不清就不收。
4. 重构改变了某个保留测试的**实现假设**（例如第 11、12 条），按新语义重写，不删除它守护的概念。
