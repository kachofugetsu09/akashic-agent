# 0015 · Cleanup 不拥有 turn 与重启终态

- 状态：accepted
- 日期：2026-08-01
- 关联条款：SH-002、RUN-003、RUN-004、OUT-001、ERR-001
- supersedes：0013 的旧 boot 空集重启前置条件；0014 的 cleanup 失败传播语义
- superseded by：无

## 背景

一次 Shell 命令成功完成并生成最终回复后，回复已经写入 SessionDB 并发布 `TurnCommitted`。随后 AgentLoop 在外层 `finally` 回收 execution；进程组中只剩由 Boot Guardian 收养的其他 UID zombie 时，`killpg` 返回 `EPERM`。工具层已经把第一次错误转换给 Agent，但外层 cleanup 再次抛出，控制面因此把已提交 turn 标记为 failed，移动端只显示通用错误。

Linux zombie 不能由 `kill` 消灭，只能由父进程或 subreaper `wait`。不同 UID 只决定 `killpg` 是否显式暴露 `EPERM`，不是 zombie 泄漏的必要条件。

## 决定

Turn Pipeline 独占回复提交和终态；ShellProcessManager 与 Guardian 独占 execution cleanup。cleanup 在提交后只能产生 execution 状态和诊断，不能回滚 turn 或合法 restart commit。

```text
Turn Pipeline ── persist + TurnCommitted + dispatch ──▶ completed
                                                       │
Execution owner ── cleanup ────────────────────────────┤
                  ├─ clean / zombie reaped             │ 不反向改写
                  └─ live residual ── retain + log ─────┘
```

- Guardian 在运行期间持续响应 `SIGCHLD` 并 `wait` adopted zombie。
- cleanup 未确认时 execution 继续留在 manager，当前 runtime 隔离同 owner 的新 Shell spawn；普通对话不受影响。
- boot cleanup 失败写结构化诊断，但已具备合法 ready、commit 和退出码证据的重启继续进入下一代。
- execution 与隔离状态仍不进入 SessionDB，不跨 runtime 恢复。

## 理由

外部命令的副作用在 cleanup 前已经发生，回滚 turn 会诱导客户端或用户重试，造成重复副作用。完全吞掉 cleanup 并删除注册项虽然接近 Codex 的 best-effort 行为，但会静默遗留活进程。保留 ownership 与诊断可以同时保持用户回复终态和运维可观察性。

## 影响

- cleanup 权限错误不再成为用户可见 turn failure 或重启 blocker。
- 无法清理的活进程可能跨 Gateway 代际存在；日志必须包含 boot/session/turn/execution/error 信息，运维按真实权限处理。
- runtime 重启后不恢复 execution quarantine；需要跨 runtime 强制隔离时应引入 cgroup/Job Object owner，不能写入 SessionDB 冒充进程事实。

## 验收

- 注入提交后的 `EPERM`，turn 仍 completed、最终回复保留且只派发一次。
- zombie 被 Guardian 在 Gateway 运行期间实际 `wait`，最终从 `/proc` 消失。
- 活残留保留 execution 记录并隔离同 owner 新 spawn；cleanup 成功后解除。
- 注入 boot cleanup 失败，结构化日志存在且合法 restart 仍创建下一代。
- 两个 P0 mutant 分别证明 Gate 能拒绝“cleanup 反杀 turn”和“未确认即忘记 execution”。
