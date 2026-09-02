# 测试与 Gate 清理账本

本账本记录测试和 Gate 的收敛。数量不是删除证明：只有明确没有独占现实可观察回归、非平凡不变量、信任边界、生命周期、持久化或具体 bug 的测试，才可以永久删除。

## 记录格式

每次清理记录基线、运行范围、永久删除、删除理由、保留边界、验证和恢复点。新增或删除测试都必须说明它固定或放弃的可观察失败。

## 2026-09-02：普通 Pull Request 回归与插件 v3 Gate 分层

- 基线：`origin/main@1b889e01`；完整 Python 为 3239 项、完整 Node 为 194 项，Pull Request CI 为 8 个 job。
- 普通 PR：运行 1080 项 Python，即完整数量向上取整后的三分之一；运行 62 项 Web；CI 收敛为 `check-and-test` 与 `change-impact-gate` 两个 job。
- 完整资产：3239 项 Python、194 项 Node 及原始 semantic scenario、impact mapping、Content/Wake manifest、插件 v3 Gate 脚本全部保留。change-impact Gate 可以按生产 diff 选择普通 PR 清单外的边界测试；候选或发布里程碑继续运行完整回归与对应集中 Gate。
- 永久删除：无。第一次实现曾按数量批量删除 205 个测试文件；复核发现其中包含 ConversationRuntime finality、Web ingress、attachment durable publication、backup restore、MCP recovery 和 plugin lifecycle 等独占边界，无法证明安全，因此全部恢复。
- 收敛理由：过重来自“每个 PR 总是运行全部回归和多个候选/发布 Docker Gate”，不是测试文件存在本身。最终只减少常态执行频率，不销毁可在影响 Gate、候选或发布阶段调用的证据。
- Python 清单：`tests_scenarios/contracts/pr-regression-files.txt` 明确列出普通 PR 文件；`scripts/check_pr_regression_budget.py` 在 CI 中校验路径存在且唯一，并用 pytest 真实收集数固定为完整集向上取整后的三分之一。复审后以 50 项 mobile protocol schema/枚举覆盖换入 19 项 ConversationRuntime finality 和 31 项 channel-to-runtime durable handoff；前者在完整套件保留，并在 `infra/mobile_realtime/**`、protocol 或 schema 变化时由 `mobile_realtime_contract` 运行整个 `tests/mobile_realtime`，后两者此前没有 scenario 直接覆盖，固定 late failure、exactly-once delivery 与 durable handoff。清单还覆盖 Session 只追加、迁移/backfill/retirement、认证、附件、generation 发布、进程清理、provider 错误、确定性并发、中断恢复和 Content/Wake 互操作。
- Web 清单：`npm run test:web:pr` 保留移动消息状态、Web transport 和 Akasha 移动端行为；原有 navigation、theme、plugin web module、mobile state 和 performance 命令保持可运行。
- Docker Gate：普通 PR 不再总是重复完整回归、static fleet、Mobile、composition、programmatic-control 与 restart-soak job；手动 `Plugin v3 Candidate Gates` workflow 是候选与发布阶段运行完整 Python/Node、这些 Gate 脚本并上传 commit-bound evidence 的 owner。依赖正式来源 workspace 副本的 E4 仍按发布手册显式执行，不进入无正式状态的 GitHub runner。
- 发布缺口：E3 Fleet/Channel/Proactive 是既有发布合同，但仓库尚无 runner。本次不伪造或删除该合同；E4 会对缺失或身份不匹配的 E3 报告 fail closed，因此 E3 落地前只能合并本次 CI 分层，不能宣称完整插件 v3 发布 Gate 已完成。
- 并发：保留 WebSocket generation 发布用例的 5 秒 fail-loud timeout。1012 close 只协调客户端关闭，测试退出 TestClient WebSocket context 完成 close handshake，让 middleware 拥有的 snapshot lease 可以释放，再等待 publication；lease drain 没有内部 deadline，删除外层 timeout 会把回归退化成 CI 级无界挂起，因此不属于可安全清理项。
- 恢复点：初始清理前归档位于 `/mnt/data/akasic-agent-backups/test-gate-one-third-20260902-before-clean/test-and-gate-surface.tar.gz`；安全返工前 bundle 位于同目录 `pre-safety-rework-fab050bf.bundle`，SHA-256 为 `bd64973eea6291bd6934a3fda14e4284de4026011e0d2a590a6ddda49c511d3e`；独立 Review 修复前 bundle 为 `pre-terra-fixes-7806cf79.bundle`，SHA-256 为 `c46560c496492f906ee243badcfab98917060603a7976dcb9102990366fdeb83`。
- 验证：最终普通 PR Python 清单实跑为 `1075 passed, 5 skipped`（472.22 秒）；完整 Python 为 `3233 passed, 6 skipped`（810.76 秒）；完整 Node 为 `194 passed`，候选 workflow 的相同 Node 发现命令也为 `194 passed`。WebSocket lease 顺序修复后目标用例连续运行 10 次全部通过。TypeScript、生产/测试 Pyright、控制协议 schema、Yoyo 只追加检查、Gate audit 与差异检查均通过；最终 change-impact Gate 通过，报告为 `docker/debug/reports/change-gate/20260902-215329-121a9138`。独立 Review 首轮发现清单预算与候选 Gate owner 缺少机器约束，以及 publication timeout 被错误移除；上述三项已返工，复审结论随后记录。
