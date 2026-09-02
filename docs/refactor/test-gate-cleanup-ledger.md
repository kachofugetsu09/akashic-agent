# 测试与 Gate 清理账本

本账本记录测试与 Gate 的永久收敛。数量只是预算，不是删除依据；取舍按用户可观察失败、持久化与安全边界、并发 finality、恢复能力和插件 v3 生命周期排序。

## 2026-09-02：保留最高价值的三分之一

### 结果

| 范围 | 清理前 | 保留 | 删除 | 预算结果 |
| --- | ---: | ---: | ---: | --- |
| Python | 3239 项 / 250 文件 | 1080 项 / 72 文件 | 2159 项 / 178 文件 | `ceil(3239 / 3) = 1080` |
| Node | 194 项 / 34 文件 | 62 项 / 4 文件 | 132 项 / 30 文件 | 低于三分之一 |
| PR CI job | 8 | 2 | 6 | 低于三分之一 |

Python 的 1080 是仓库完整收集数，不是从完整套件中挑出的 PR 子集。`scripts/check_test_budget.py` 同时固定数量和文件集合；任何未列入 `tests_scenarios/contracts/retained-test-files.txt` 的新测试都会使 CI 失败。Node 只保留 mobile message state、pairing response schema、Web transport 和 Akasha mobile UI 四个行为边界，由唯一命令 `npm run test:web` 执行。

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
