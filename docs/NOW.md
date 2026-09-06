# NOW

这份文件只保存 Akashic Agent 当前仍未完成的工作。事项完成后删除，不保留“已完成”记录。

## P0 · Akashic Channel 与 Web/Mobile Adapter 实现

[Akashic Channel 与 Web/Mobile Adapter 规格](design/akashic-channel-client-adapters.md) 已确认
一个 Core `akashic` Channel、两个薄 adapter 和一次 breaking rekey。实现已获授权，当前核对
Session/Message 全身份迁移、配置、Akasha 和 Android 强制全量同步；不得直接迁正式 workspace。

## P1 · 移动端主题 token 边界

[`移动端投影审计 D2`](design/mobile-projection-audit.md) 已确认原生壳 Compose 色板与 Core WebUI CSS token 是两个渲染层的表示，不是重复 owner。仍需决定色值一致性由构建期产物还是显式 token 边界保证。

- 移动端用户 checkout 存在未提交 Theme diff（Theme.kt 等 5 个文件）；D2 决策（原生壳与 WebUI token 边界）完成前不得合入。

## P0 · 插件递归自验证

- 在独立 Fitbit canonical source 变更中让 monitor 与 MCP 读取同一个 `validation_port_env`，再以一次性 workspace 验收真实隔离 listener、child tool trace、正式切换和旧 listener 恢复；不得在 Core 中添加 Fitbit 特判。
- 补充 turn-boundary rollout 的进程崩溃注入矩阵，覆盖 terminal 封口后、候选服务停止后、正式 endpoint 切换后和 pointer 提交前；恢复失败必须保持 degraded 可见，不能只恢复 pointer。

## P0 · 独立语义验收

- 将 CTX-001 当前的 trace、完整状态快照和 fixture `DELETE` pilot 升级为 SQLite authorizer 与一次性候选真实 retry seam mutant；导入失败、fixture 失败或超时不得计为 mutant kill。
- 建立受保护路径 policy：`semantic_delta: none` 的普通实现改动不能同时修改 P0 oracle、mutant 或 coverage baseline 来获得全绿。
- 建立轻量 `change-intent` 校验，检查实际 diff、允许路径、受保护状态和副作用是否超出声明。

## P1 · Message 日志与回复链插件化

- 用户已批准[完整设计与分层合同](design/0902-reviewed-v4.md)：Message 独立保存，Turn 由普通无状态插件投影，Akasha 是消费者；完整回复链由非特权插件组合，替换现行 logical interaction / attempt 执行模型。
- 仓库内重构以 stacked draft PR 交付，持久变化的 yoyo 脚本与引入变化的 PR 同步；禁止灰度、shadow、双 writer 和旧 hook 兼容壳。已核实冗余的删除记录在既有账本，代码删除不授权减少历史消息、学习、附件和插件数据。
- hua-home 上 Citation、Meme、反馈、诊断、命令和工具检查的功能按设计第 15 节保留并重组。外部插件源码迁移后续交付；其实际安装与功能验收是正式切换前提，不阻塞仓库内新接口实现。正式 workspace 尚未迁移。

## P1 · 工作流扩展

- 按 [`容器与 Host Bridge 非迁移实验合同`](design/akashic-container-host-bridge-experiment-contract.md) 完成 mise/锁文件前置、本机 Local/Bridge/容器分层验证和 hua-home 隔离候选运行时；在 capability matrix、Supervisor 故障注入、OpenCode V4 Flash High 与正式状态零写入证据齐全前，不启动正式 workspace 迁移。
- 把 `projectneed.md` 中其他 P0 不变量逐步迁入可执行契约，优先处理 MEM-001、MEM-002、OUT-001、PLG-001、PLG-004、WSP-001 和 BAK-001。
- 为高风险 refactor 增加 base/candidate 差分回放，核对持久 write set、事件、外部调用和错误分类。
- 由维护者继续确认 [`design/persistence-state-map.md`](design/persistence-state-map.md) 的 INT-009、INT-010、INT-012～INT-014，以及旧消息编辑和 turns retention；INT-001～INT-008、INT-011 已提升为 projectneed 条款。
- 把已确认的持久化状态地图转成机器可读备份 manifest，补齐目录快照、global companion state 和隔离恢复演练；确认 snapshot 能启动只读 runtime，并读取会话、记忆、调度、插件数据和主动流程连续性。

## P2 · PTY resize 既有规格差距

SH-003 写有 resize，但当前 ShellProcessManager 和 Host Bridge 尚无 resize 入口。
Protocol V2 保留已有 PTY 输入、输出和 stop，不把协议升级视为 resize 验收通过；后续独立确认实现范围。
