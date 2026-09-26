# NOW

这份文件只保存 Akashic Agent 当前仍未完成的工作。事项完成后删除，不保留“已完成”记录。

## P1 · Akasha 学习图一次性重放

`plugins/akasha` 已收口为单一学习实现（重建 = 空图 + 无切换上界重放同一个
`MessageConsumer`），旧的稀疏索引、`legacy_prefix`、frozen history、离线 CLI 与 repair
通道退役。待完成：

- 用插件自有 migration bundle 登记的一次性重放完成线上切换，并按
  `docker/debug/akasha_replay_compare.py` 核对历史前缀（本机副本实测：legacy 复现
  5549/5581、相对顺序无逆序、交集身份与时间瞬间零差异）。
- 重放会丢掉旧图里 26 条早期 `remember` 事件。维护者已确认它们不是 Akasha 现行反馈
  通道的产物（属于很早期的 tool result），接受该偏差，不另立 carry-over 迁移。
- `memory/akasha-v2-index.db`（约 511MB）退役后的物理删除单独授权。

## P0 · 插件正交化实施与最终验收

按 [0065](decisions/0065-plugin-boundary-checks-do-not-grant-core-ownership.md) 与
[阶段验收](design/plugin-boundary-foundation.md#7-验收标准) 完成已授权的 stacked PR 实施。
当前优先完成全部本地插件的行为回归、默认组合与无 checkout 的分发产物验收。
维护者已明确将第三方外部插件的进一步迁移与验收后置；已提交改动保留，
不将第三方迁移完成作为本地插件阶段的退出条件。
当前维护者要求只交付 stacked Draft PR；仅做静态检查和独立只读评审，
不运行测试、Gate 或 CI。真实运行验收须另行获得授权，不因实现完成自动开始。
验收必须覆盖 Core-only CLI/AppRuntime、独立子集和异名 provider、generation/归档生命周期、
实际 Message 与持久送达闭环；不能以 import 数量清零代替这些证据。

## P0 · Akashic Channel 与 Web/Mobile Adapter 实现

[Akashic Channel 与 Web/Mobile Adapter 规格](design/akashic-channel-client-adapters.md) 已确认
一个 `akashic` Channel、两个薄 adapter 和一次 breaking rekey。渠道归属按
[0067](decisions/0067-clients-are-ordinary-plugin.md) 修订为普通插件。实现已获授权，当前核对
Session/Message 全身份迁移、配置、Akasha 和 Android 强制全量同步；不得直接迁正式 workspace。

## P1 · 移动端主题 token 边界

[`移动端投影审计 D2`](design/mobile-projection-audit.md) 已确认原生壳 Compose 色板与 Core WebUI CSS token 是两个渲染层的表示，不是重复 owner。仍需决定色值一致性由构建期产物还是显式 token 边界保证。

- 移动端用户 checkout 存在未提交 Theme diff（Theme.kt 等 5 个文件）；D2 决策（原生壳与 WebUI token 边界）完成前不得合入。

## P0 · 单图插件系统与局部换代（Issue 750）

按 [0072](decisions/0072-single-graph-local-plugin-updates.md) 与
[单图设计](design/issue-750-plugin-publication-simplification.md) 继续完成：

- **外部组合验收**：Content/H5 的历史 lock 不匹配当前 Core 祖先与单参数 `apply(ctx)`
  入口。固定兼容的外部源码和 artifact-owned Python 后，按原完整场景重新验收；
  不改 lock 指向任意 HEAD、删 case 或降低验收覆盖取绿。
- **真实资源验收**：本地 Workload 协议测试不代表 Docker 写入挂载、跨 boot 资源恢复
  或显示代理已验证；这些结果须由对应真实 Controller 与客户端提供。
- **发行与恢复**：已有 state 的 `akashic-release install` 按[部署操作手册](design/operator-deployment.md)
  使用显式插件/迁移清单、可选备份、完整 selection 与 live Fiber 核对；远端 CI、正式发布、真实运行数据的隔离
  恢复演练和 final enable 仍需独立交付。普通 restart/ensure_profile 不自动采用新 bundle；
  通用 distribution 夹具不能替代真实 Akasha 发行输入、重放完成及功能验收，历史未决资源
  也不能用源码通过代为结算。

既有 Subagent Tools owner、DeliveryPolicy readiness 和旧 oracle 的历史阻塞不再作为
当前缺陷；本地退役、停止期回退和分层证据见单图设计的“本地收口与运行验收边界”一节。
本地开发与验证不授权 PR 合并、正式数据写入或部署。

## P1 · 外部插件发布兼容

核对外部已安装插件的显式依赖声明、公共合同导入与真实分发组合。
[Issue 766 本地实现](design/issue-766-orthogonal-capabilities.md) 已完成合同和宿主装配收敛；
内置插件验证不能代替外部安装、正式发布或真实 Mobile 客户端验收。

## P0 · 正交化概念基线

- [正交化测试基线](refactor/orthogonality-test-baseline.md) 的覆盖扩展仍待维护者另行授权。#766 按本次明确要求不新增或改写单元测试，保留既有概念验证；唯一例外为既有夹具补充 TASKS 依赖声明。
- 建立轻量 `change-intent` 校验，检查实际 diff、允许路径、受保护状态和副作用是否超出声明。

## P1 · 工作流扩展

- 按 [`容器与 Host Bridge 非迁移实验合同`](design/akashic-container-host-bridge-experiment-contract.md) 完成 mise/锁文件前置、本机 Local/Bridge/容器分层验证和 hua-home 隔离候选运行时；在 capability matrix、Supervisor 故障注入、OpenCode V4 Flash High 与正式状态零写入证据齐全前，不启动正式 workspace 迁移。
- 由维护者继续确认 [`design/persistence-state-map.md`](design/persistence-state-map.md) 的 INT-009、INT-010、INT-012～INT-014，以及旧消息编辑和 turns retention；INT-001～INT-008、INT-011 已提升为 projectneed 条款。
- 把已确认的持久化状态地图转成机器可读备份 manifest，补齐目录快照、global companion state 和隔离恢复演练；确认 snapshot 能启动只读 runtime，并读取会话、记忆、调度、插件数据和主动流程连续性。

## P2 · PTY resize 既有规格差距

SH-003 写有 resize，但当前 ShellProcessManager 和 Host Bridge 尚无 resize 入口。
Protocol V2 保留已有 PTY 输入、输出和 stop，不把协议升级视为 resize 验收通过；后续独立确认实现范围。
