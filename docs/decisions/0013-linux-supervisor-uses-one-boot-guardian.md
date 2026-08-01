# 0013 · Linux Supervisor 每个 boot 只使用一个 Guardian

- 状态：accepted
- 日期：2026-08-01
- 关联条款：RUN-001～RUN-004、WSP-001～WSP-004
- supersedes：无
- superseded by：[0015](0015-cleanup-does-not-own-turn-or-restart-finality.md)（仅勘误旧 boot 空集前置条件）

## 背景

Supervisor 的产品目的，是让 Agent 修改 Gateway/Core 代码后，在当前回复已经持久化并实际送达的前提下安全重启。改造前实现在启动线程后使用 `preexec_fn`，以 20ms 周期轮询 child 和 readiness 文件，并且只在 Supervisor 存活时拥有 boot 清理责任。固定 15 秒还把完整串行启动压成一个不可诊断的等待阶段。

跨平台完整实现会引入无法在当前环境真实验证的进程树语义。直接 `python main.py` 又不能依赖 systemd、可写 cgroup 或 root 权限。

## 决定

Linux 正式入口保留 Supervisor。每个 boot 只增加一个轻量 Boot Guardian：Guardian 是 Gateway 的父进程和 Linux child subreaper，通过 Supervisor lease、pidfd、现有进程组和 boot identity 收束当前 boot。Supervisor 也注册为 subreaper，但只在 Guardian 异常时收割内核转交的孤儿进程。Gateway 通过继承的私有单向 pipe 发布 `stage`、`ready` 和 `commit`；workspace readiness 文件只作诊断投影。

只有当前 boot 已 ready、正式回复已送达并提交有效 commit、Gateway 以 75 退出且旧 boot 已验证为空时，Supervisor 才创建下一代。普通退出、崩溃和 owner 故障只清理并失败，不自动重启。

非 Linux 默认进入明确警告的 unmanaged gateway，不提供 Supervisor 配套能力；显式 `supervise` 直接拒绝。settings server 固定监听 `127.0.0.1`。

## 理由

一个 Guardian 已能覆盖 Supervisor lease 丢失、Gateway 退出、double-fork/`setsid` 后代和现有 MCP/service 进程组，不需要每服务 guardian、通用 spawn RPC 或 cgroup 正确性前提。pidfd 提供稳定进程身份，私有事件避免轮询 workspace 文件；总体 deadline 与阶段事件分别承担失败上限和诊断，不用 heartbeat 延长启动。

## 影响

- 正面影响：移除多线程 `preexec_fn` 和常驻 20ms polling；Supervisor、Guardian、Gateway 单 owner 故障有明确清理路径。
- 兼容性：Linux 默认入口与 `supervise` 保持；非 Linux 失去完整 Supervisor、settings restart 和 `agent_restart`。
- 数据和迁移：业务持久状态不迁移、不减少；正式 supervised 启动只由 Supervisor 执行一次 Git cursor 迁移检查。
- 失败与回滚：Guardian 或协议异常时清理当前 boot 后非零退出；可按实现提交整体回滚，workspace 权威状态保持不变。

## 验收

- [x] 私有 ready/commit、非法 75、stale readiness、设置回滚与 Linux/非 Linux 分流通过测试。
- [x] Supervisor、Guardian、Gateway 与 double-fork 后代故障注入证明旧 boot 清空且未知 PID 不被杀。
- [x] 20 轮 soak 证明 FD、线程、RSS、zombie、端口和非终态 turn 没有超出门禁。

## 未决问题

- 启动总体 deadline 的默认值由冷启动与热启动阶段 profile 决定；当前不以任意放大数值替代测量。
