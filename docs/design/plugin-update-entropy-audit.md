# 插件自更新复杂度审查

## 1. 结论

插件自更新不需要新的特权插件，也不需要第二套 Root 切换系统。现有链路已经拥有安全迁移所需的状态和边界。本次让实现重新符合决策 0026 的通用 child 完成合同，并删除没有业务调用者的新设计。

## 2. 现有链路

```text
source ── immutable artifact ── candidate generation ── latest snapshot
                                                        │
stable snapshot ── parent Turn ── attached child ───────┘
       │                                      │
       └──────── 旧请求继续使用 lease          └── 检查 exact candidate

parent 完成 ── stable 指针切换 ── journal 完成 ── 旧 lease 排空
```

DeepSeek Harness 提供了复杂度基线。它用普通配置行装配插件，依赖可用性决定 Fiber 激活顺序（`packages/bundle/base/cordis.patch.yml`）；Service 绑定当前 Fiber，Fiber 卸载时逆序清理 Effect（`vendor/cordis/src/reflect.ts`、`vendor/cordis/src/fiber.ts`）。`agent-loop` 也只是依赖 agents、sessions、llm、tools、systemPrompt 与 sessionProjections 的普通 Service 插件（`packages/core/agent-loop/src/index.ts`）。

DSH 没有跨进程旧 Root 续跑：reload 会停止旧 Fiber，进程崩溃会把 open Turn 修成 interrupted，再用当前插件世界开始新 Turn（`packages/core/session/src/repair.ts`、`packages/session/session-persistence/src/coordinator.ts`）。Akashic 保留 snapshot lease 来让同进程旧请求排空，但不额外发明跨进程旧世界复活协议。

状态 owner 没有变化：

| 事实 | Owner |
|---|---|
| artifact 内容与 source revision | 插件安装器 |
| stable/latest snapshot 与 generation | `PluginManager` |
| 一个请求使用哪份运行时 | `RuntimeSnapshot` lease |
| candidate 是否提交 | parent `TurnPluginRollout` |
| 崩溃后读取哪个 stable | stable 指针与 reload journal |

## 3. 删除证据

2026-09-02 对 Core、仓库内插件、外部插件源码和 hua-home 正式安装 cache 做了静态消费者检查：

| 新接口 | 生产消费者 | 结论 |
|---|---:|---|
| `ServiceCall` | 0 | 与已有 Service 调用重复 |
| `TaskControl` | 0 | 没有插件取得或调用 |
| `RootSwitch` / `SwitchPart` | 0 个 part | 与 stable/latest 和现有领域切换重复 |
| `SwitchInput` | 0 | 没有输入来源 |
| `ServiceHold` / `ctx.hold` | 0 | 与 snapshot lease 和进程 owner 重复 |

“存在接入点”没有算作消费者。检查同时覆盖动态注册、manifest 声明、插件 cache 和 hua-home 当前运行 revision。当前没有外部迁移义务，因此直接删除，不保留 deprecated 名称或兼容壳。

## 4. 保留资产

- 不可变 artifact 和 source hash 检查。
- stable/latest 与候选 generation。
- candidate Root 和 plugin-data 隔离。
- RuntimeSnapshot lease 与旧 generation 排空。
- attached child 的 parent/generation/source binding。
- parent Turn 的自动提交、丢弃和 reload journal 恢复。
- operator trusted batch 的独立信任边界。

## 5. 修正的问题

原实现把候选检查写成“成功使用候选拥有的 Tool 或 Skill”。这让 UI、Channel、模型、Job 等普通插件被类别歧视，也迫使 Core 理解业务证明。

修正后，Core 只确认 exact attached child 正常完成。Agent 在旧 stable 的 parent 中决定 child 做什么检查；检查失败时先执行 `plugin-revert`，没有 revert 且 parent 正常完成就是提交授权。检查使用候选公开的普通能力，新增插件类别不需要修改 Core。

## 6. 未改动和未知

- 本次没有改变进程崩溃时的 Turn 恢复语义；崩溃中的 Turn 仍是中断，不伪装成外部效果回滚。
- 本次没有部署 hua-home；对它的检查只是消费者盘点。
- operator 控制入口、Activity/Channel 切换和候选恢复仍有复杂度，是否多余要继续用生产消费者、历史原因和失败测试证明，不能从名字直接删除。

## 7. 第二轮熵审查

第一轮提交后又用监管、反向、值班、极限成本和十岁视角独立发散，再按可落地性、影响和风险收敛。脑暴只产生候选，删除仍以真实消费者和失败边界为准。

### 7.1 本批删除

| 重复面 | 真实消费者 | 处理 |
|---|---:|---|
| `begin_publish(admission_gated=...)` | 0 次读取 | 删除形参与两处无效传值 |
| `TurnPluginRollout._tasks` | 只有同一个 `_resolution_task` | 删除集合；shutdown 取消并等待唯一 task |
| rollout terminal `items` | Tool/Skill evidence 删除后为 0 | 从内部 callback 删除，不改变 Turn items 的保存和输出 |
| `RuntimeSnapshotStore.stable` | Core 仅四个测试断言 | 统一使用 `current`，不保留别名 |

2026-09-02 12:54 CST 对 hua-home 做了只读检查。证据来自 `akashic-core.service`
的 `AKASHIC_RUNTIME_CHECKOUT`、release manifest 和正式 plugin-home cache，不是主机上某个
开发 worktree。实际运行 release 为 `376556c616de39d43af528f8fbdde15a0db83e7f`，
两个 Core 容器均 healthy。在
`/srv/data/services/akashic/state/plugin-home/cache` 内，`admission_gated`、
`quiesce_current` 和 `TurnPluginRollout` 均为零命中；只有 shell-restore 和
shell-safety 的测试实例化 `RuntimeSnapshotStore`，都没有读取 `.stable`。
本次没有修改该主机。

### 7.2 暂缓删除

| 候选 | 暂缓理由 | 删除前证据 |
|---|---|---|
| `plugin/promote`、`plugin/discard`、`plugin/status` | 已进入 CLI 与 app-server v1 schema | 真实 control 日志、受支持客户端和跨仓协议 Gate 均证明零消费者 |
| 无 owner 的 install/uninstall/disable | 仍是公开协议与 operator 行为 | 明确新的 operator 合同和 breaking migration 顺序 |
| 泛化 `ActivityHost` | 当前只有 Background Job child，但它拥有 admission、drain、rollback 与 recovery | 单 child 全盘点和逐故障等价回放 |
| Channel boot transaction 与 candidate recovery 分支 | journal 和外部 owner 的恢复语义尚未证明重复 | 冷启动、pointer 前后和外部 owner 残留故障矩阵 |

暂缓项没有标 deprecated，也没有增加兼容层。它们保持现状，等独立证据证明后再整块删除。

## 8. 验收

- 代码净差异不再包含上述五组新接口。
- exact child 检查不依赖插件类别或 TurnItem。
- 相关插件安装、热重载、候选隔离、lease 与 journal 测试通过。
- PR 记录第二轮熵审查发现、保留理由和实际删除项。
