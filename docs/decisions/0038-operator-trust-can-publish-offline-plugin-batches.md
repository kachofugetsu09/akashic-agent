# 0038 · Operator 信任可以离线发布 exact 插件批次

- 状态：accepted
- 日期：2026-08-22
- refines：[0026](0026-plugin-rollout-is-owned-by-the-parent-turn.md)、[0037](0037-plugin-runtime-is-pure-v3.md)
- 关联条款：PLG-013、RUN-015、ERR-001

## 背景

0026 约束 Agent 自改进：`plugin-install` 只能创建 candidate，必须由同一 parent 的 attached programmatic child 验证后提交。最终 pure-v3 fleet 合并时，另有一类操作来自已经完成代码审计、真实验证或因外部效果难以重复而由用户明确承担判断的 operator。继续让这类离线批次伪装成 Agent 自验证，会重复调用外部服务、拉长维护停机，并把“人工信任”错误记录成“模型验证”。

## 决定

1. 新增独立 shell 入口 `plugin-install-trusted-batch`；它不进入 Agent tool、runtime control 或 candidate API。
2. 入口只接受 JSON batch 中的完整 40 位 commit SHA，并继续使用正式 installer 校验 pure-v3 static manifest、路径、依赖与 immutable artifact。
3. Runtime 从启动到停止独占 plugin-home 的 `.publication.lock`。trusted batch 执行前按 workspace `.supervisor.lock` → `.instance.lock` → plugin-home `.publication.lock` 的顺序非阻塞取得三把锁；即使另一个 workspace 共享该 plugin-home，任一生命周期或消费 owner 存活也会 fail-loud，因此 batch 不能与 Turn、Watcher、generation publication 或 restart 并发。
4. operator 一次确认信任整个 batch；命令检测到 active-turn provenance 时拒绝。该区分是操作入口与生命周期证明，不宣称能对拥有同等宿主权限的进程做身份认证。
5. 每个插件直接把 stable/latest 指向同一 exact artifact；失败保留已完成项并明确报告其列表，不声称整个批次原子成功。执行 owner 必须在调用前创建 recovery point。
6. JSON 回执固定声明 `mode=operator_trusted_offline_batch` 与 `programmaticValidation=bypassed_by_operator_trust`。这表示信任已在入口外建立，不是行为 Gate 通过。
7. 好验证但尚无证据的插件不得放入 trusted batch，仍走普通 candidate 与不写长期记忆的 attached programmatic call。

## 影响

- Agent 自改进语义不变；0026 的 child 验证与 turn 后提交继续成立。
- operator 可以按已审计证据批量更新 pure-v3 fleet，而不为难以重复的外部效果制造假验证。
- batch 不是跨插件原子事务；部署恢复点和回执拥有失败后的恢复与对账责任。
- runtime 仍只消费 committed stable，插件 main、fleet lock、实际 artifact SHA 和验证证据必须分别记录。
