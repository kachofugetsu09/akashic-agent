# 测试与 Gate 清理账本

本账本记录测试和 Gate 的收敛，不把“删得多”当作目标。清理必须说明被删内容没有独占现实可观察的回归、非平凡不变量、信任边界或具体 bug；仍有这些职责的覆盖必须保留或先转移到已有行为边界。

## 记录格式

每次清理记录基线与结果、删除内容、删除理由、保留边界、验证和恢复点。后续新增测试也必须说明它固定的可观察失败，而不能只引用代码变化或覆盖率。

## 2026-09-02：Core Pull Request 测试与 v3 Gate 收敛

- 基线：`origin/main@9f30b079`；Python 为 3239 项、250 个文件，Node 为 194 项、34 个文件，Pull Request CI 为 8 个 job。
- 结果：Python 为 1080 项、76 个文件，即原数量向上取整后的三分之一；Node 为 62 项、3 个文件；Pull Request CI 收敛为 2 个 job。
- 删除：只复述字面量、映射、显然控制流、helper 内部步骤或布局 token 的单元测试；已经删除能力的兼容测试；被更高层 v3 install、hot reload、turn rollout、generation 和真实 channel 行为重复覆盖的 catalog、slot 与 adapter 测试。
- 理由：这些测试没有独占用户可观察行为或边界，主要锁定实现形状。它们使普通改动承担重复维护和容器启动成本，却不能比已有行为边界更早、更准确地发现回归。
- 保留：Session 正文只追加、迁移/backfill/retirement、协议与认证、附件、生命周期与 generation 发布、进程清理、provider 错误、确定性并发、中断恢复、Wake/Content 互操作，以及拒绝合同 mutant 的语义测试。
- Web：保留移动消息状态、Web transport 和 Akasha 移动端真实行为；删除 parser、CSS/token、性能阈值与内部 adapter 的重复镜像。
- Gate：保留版本化 change-impact 选择、公开语义 mutant 和 Content/Wake v3 互操作；删除已经没有对应测试文件的 `tests/test_drift_*.py` impact 规则，Drift 源码仍由 `plugins/drift/**` 选择同一行为场景；从普通 Pull Request 移除重复的 static fleet、Mobile、composition、programmatic-control 与 restart-soak job。插件 v3 的 static fleet、领域组合和 E1～E4 集中 E2E 仍用于候选或发布里程碑。
- 并发：移除 WebSocket generation 发布用例中重复的 5 秒外层竞速。1012 close 已是确定性 lease 释放协调，publish 路径拥有自己的 drain deadline；外层 wall-clock timeout 在全量高负载下造成过一次假失败，却不增加可观察合同。
- 恢复点：pull 前仓库 bundle 与未跟踪文件归档位于 `/mnt/data/akasic-agent-backups/test-gate-cleanup-20260902-pre-pull/`；清理前测试与 Gate 表面归档位于 `/mnt/data/akasic-agent-backups/test-gate-one-third-20260902-before-clean/test-and-gate-surface.tar.gz`，SHA-256 为 `3125f0944364b68daefec731069a2f30f683592fe66bfc6e0dfd8b4041374900`。
- 验证：Web 回归 `62 passed`；TypeScript、生产/测试 Pyright、控制协议 schema 与 Yoyo 只追加检查通过。Python 全量首次为 `1074 passed, 5 skipped, 1 failed`，唯一失败是上述重复 5 秒竞速；该用例独立复跑通过，所在 hot reload 文件为 `43 passed`。移除外层竞速后，最终全量为 `1075 passed, 5 skipped`，change-impact Gate 也重新验证对应场景；最终 Gate 报告见本次交付说明。
