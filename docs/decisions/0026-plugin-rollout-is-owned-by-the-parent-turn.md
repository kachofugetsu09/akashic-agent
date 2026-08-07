# 0026 · 插件发布由父 Turn 在终点统一授权

- 状态：accepted
- 日期：2026-08-08
- supersedes：[0024](0024-plugin-self-validation-uses-stable-and-latest.md) 的 Agent 显式 latest/promote/discard 接口；保留其内部双快照与 session 并发机制
- refines：[0008](0008-plugin-runtime-publishes-only-committed-snapshots.md)、[0015](0015-cleanup-does-not-own-turn-or-restart-finality.md)
- 关联条款：PLG-010、PLG-012、PLG-013、RUN-007、CTRL-003、ERR-001、TST-001～TST-006

## 背景

0024 证明了父 turn 保持 stable、独立 programmatic child 租用 latest 可以形成真实递归验证。但让 Agent 手工编排 status、promote、discard、uninstall、restart，把 snapshot、lease、manifest、独占 endpoint 和恢复等 Core 状态泄漏给调用者。2026-08-07 Fitbit 更新因此出现调用方绕过 `endpoint_coexistence` Gate，形成 snapshot 与正式 listener 分裂。

## 决定

1. Agent 只使用 `plugin-install`、`plugin-uninstall`、`plugin-revert`。latest、promote/discard 和 drain 继续作为 Core 内部机制，不进入 Agent 快路径。
2. install/uninstall 必须绑定 active parent turn。revert 只撤销同一 turn 最近一次尚未封口的操作，不能跨 turn 回滚。
3. 父 turn 从 admission 到 terminal 始终使用原 stable。它创建的 attached programmatic child 按 `owner_turn_id + generation_id + source_revision` 自动冻结候选；detached 或其他 turn 不继承。
4. install 只有在当前候选至少一个 attached child 正常完成、没有 revert 且 parent 正常结束时才提交。无验证、失败、取消、超时、身份不一致或 parent 非正常终结时自动丢弃。
5. parent terminal 只完成封口。Core 在 lease 释放后异步执行排空、endpoint 切换、pointer/manifest 提交和清理，不让调用栈等待自己的 lease。
6. 改变 managed service 的候选必须声明 `validation_port_env`。Core 复制 plugin-data 到隔离目录、分配临时 loopback 端口、将同插件 MCP 路由到候选服务并验证 readiness；缺少声明或服务不读取变量时拒绝候选。
7. Channel candidate 不复制正式 ownership。正式提交按 old Channel stop、service switch、new Channel start 作为同一代际事务；任一步失败恢复旧代。`stop()`/`start()` 返回分别是 release/ready 的确认边界。
8. install/uninstall/revert 返回自然语言，明确已发生、未发生、turn 后会发生什么和 Agent 下一步。Core 不创建伪对话；最终结果写入一个可消费的 runtime fact，下一用户 turn 以自然语言获知。
9. uninstall 在 parent turn 内不改 manifest、不停 endpoint、不删代码；提交后删除 installed code 和能力投影，保留 plugin-data、SessionDB、memory、journal 与 canonical source。

## 理由

- Agent 只表达意图和领域判断，Core 拥有资源状态机，操作面更小且每个动作含义完整。
- 因果继承避免把 latest 设为全局默认，也不需要暴露 candidate token。
- 隔离端口和隔离 plugin-data 让 Fitbit 类固定 listener 能真实自验证，同时正式端口仍由旧 stable 持有。
- turn terminal 是清楚的提交授权；lease 释放后的后台事务消除 self-wait。

## 影响

- 旧的显式 `--runtime latest`、plugin-status/promote/discard 文档不再是 Agent 合同。
- managed service 作者需要让服务和 MCP 读取 `validation_port_env`；不支持者安全拒绝在线候选。
- `runtime/plugin-rollout-fact.json` 是单条可消费的派生反馈，不进入 SessionDB 或 memory。
- stable/latest pointer 和 reload phase 继续存在，用于内部隔离、审计与 crash recovery。

## 验收

- 父 turn 保持 S0，attached child 自动绑定 S1，其他 turn 和 detached child 保持 stable。
- 没有正确 child、revert 或异常 parent 时 S1 不发布。
- 独占 service 的 candidate listener 与正式 listener 同时可辨识，正式切换失败时恢复 S0。
- uninstall 在 turn 内可 revert，turn 后才停止、移除代码且保留 plugin-data。
- CLI 返回和下一用户 turn 的 runtime fact 均能说明真实结果与唯一下一步。
