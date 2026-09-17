# NOW

这份文件只保存 Akashic Agent 当前仍未完成的工作。事项完成后删除，不保留“已完成”记录。

## P1 · Akasha 学习图一次性重放

`plugins/akasha` 已收口为单一学习实现（重建 = 空图 + 无切换上界重放同一个
`MessageConsumer`），旧的稀疏索引、`legacy_prefix`、frozen history、离线 CLI 与 repair
通道退役。待完成：

- 用插件自有 migration bundle 登记的一次性重放完成线上切换，并按
  `docker/debug/akasha_replay_compare.py` 核对历史前缀（本机副本实测：legacy 复现
  5549/5581、相对顺序无逆序、交集身份与时间瞬间零差异）。
- 重放会丢掉旧图里 26 条 `remember` 反馈事件（旧 `history.transcript` 轨迹不再恢复成
  独立 ToolResult），需要确认接受该偏差或另立 carry-over 迁移。
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

## P0 · 插件普通调用与晋升

- 按 [0071](decisions/0071-plugin-composition-and-whole-runtime-updates.md) 调整为调用程序拥有验证、provider 拥有隔离资源、底座拥有整体 stable 提交。旧 `validation_port_env`、双指针与 attached child 的 Core 特例不再是目标合同。
- 按 [latest 普通调用](design/plugin-latest-programmatic.md) 完成非阻塞调用的过程/最终结果可见性、发起者撤销和原 owner 清理；正常完成默认请求晋升，不引入后台裁判或批准 JSON。继续核对隔离宿主的最小职责与累计消费者。
- 普通 latest 仍需后续运行证据；模型 owner 接续现有设置和凭据、新组合使用自己的 driver、调用账写在本次调用环境的源码已通过独立静态审查。真实插件链的测试已写未跑，不能据此声明实际模型请求与晋升验收通过。
- 独立 Fitbit source 的候选 listener 与正式资源隔离仍待该仓库交付，本轮不修改外部插件。
- 提交前后崩溃、排空失败和真实恢复的行为证据尚缺；本轮用户要求只提 PR，不执行 Gate/CI，不能将代码交付视作这些验收已完成。

## P0 · 独立语义验收

- 将 CTX-001 当前的 trace、完整状态快照和 fixture `DELETE` pilot 升级为 SQLite authorizer 与一次性候选真实 retry seam mutant；导入失败、fixture 失败或超时不得计为 mutant kill。
- 建立受保护路径 policy：`semantic_delta: none` 的普通实现改动不能同时修改 P0 oracle、mutant 或 coverage baseline 来获得全绿。
- 建立轻量 `change-intent` 校验，检查实际 diff、允许路径、受保护状态和副作用是否超出声明。

## P1 · 工作流扩展

- 按 [`容器与 Host Bridge 非迁移实验合同`](design/akashic-container-host-bridge-experiment-contract.md) 完成 mise/锁文件前置、本机 Local/Bridge/容器分层验证和 hua-home 隔离候选运行时；在 capability matrix、Supervisor 故障注入、OpenCode V4 Flash High 与正式状态零写入证据齐全前，不启动正式 workspace 迁移。
- 把 `projectneed.md` 中其他 P0 不变量逐步迁入可执行契约，优先处理 MEM-001、MEM-002、OUT-001、PLG-001、PLG-004、WSP-001 和 BAK-001。
- 为高风险 refactor 增加 base/candidate 差分回放，核对持久 write set、事件、外部调用和错误分类。
- 由维护者继续确认 [`design/persistence-state-map.md`](design/persistence-state-map.md) 的 INT-009、INT-010、INT-012～INT-014，以及旧消息编辑和 turns retention；INT-001～INT-008、INT-011 已提升为 projectneed 条款。
- 把已确认的持久化状态地图转成机器可读备份 manifest，补齐目录快照、global companion state 和隔离恢复演练；确认 snapshot 能启动只读 runtime，并读取会话、记忆、调度、插件数据和主动流程连续性。

## P2 · PTY resize 既有规格差距

SH-003 写有 resize，但当前 ShellProcessManager 和 Host Bridge 尚无 resize 入口。
Protocol V2 保留已有 PTY 输入、输出和 stop，不把协议升级视为 resize 验收通过；后续独立确认实现范围。
