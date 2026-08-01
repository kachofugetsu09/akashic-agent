# 0014 · Shell 采用统一可续接执行句柄

- 状态：accepted
- 日期：2026-07-31
- 关联条款：SH-001、RUN-002、RUN-003、ERR-001、TST-009
- superseded by：[0015](0015-cleanup-does-not-own-turn-or-restart-finality.md)（仅勘误 cleanup 失败传播语义）

## 背景

现有 Shell 把命令拆成前台、自动转后台、显式后台和累计日志轮询四套行为。长任务返回 `background_task_id` 后，模型只能反复调用 `task_output`；每次读取都会重发累计输出，单次等待又被限制为 30 秒。Benchmark 中的长训练任务因此产生大量无信息轮询，工具协议本身占用了步骤和上下文。

Codex 的 unified exec 只保留一种执行状态：命令先进入进程表，短时间内完成就返回结果；尚未完成就返回执行句柄，后续由 `write_stdin` 等待、读取增量输出或向 PTY 写入。Codex 在工具协议中称该句柄为 `session_id`，但 Akashic 已把 session 用作持久对话概念。用户已确认本项目按 Codex 语义改造，同时为避免两种生命周期混淆，将工具边界字段改名为 `execution_id`。

## 决定

Shell 采用与 Codex unified exec 对齐的状态机，并在 Akashic 工具边界使用 `execution_id`：

```text
┌──────────────────┐
│ shell(command)   │
└────────┬─────────┘
         │ 先注册进程，再等待 250ms～30s
         ├─────────────── 已退出 ───────────────▶ exit_code + 新输出
         │
         └─────────────── 仍运行 ───────────────▶ execution_id + 新输出
                                                       │
                                      ┌────────────────┴───────────────┐
                                      ▼                                ▼
                              write_stdin                         task_stop
                         增量读取 / PTY 输入                   确认终止执行边界
```

- 删除 `run_in_background`、`auto_promote` 和 `background_task_id`；删除 `task_output`，新增 `write_stdin`。
- 命令默认使用 passwd 中的用户 shell 和 login 语义；模型可通过当前接口的 `shell`
  与 `login` 选择 Codex 支持的 shell。manager 直接执行解析后的 argv，不再隐式调用
  `/bin/sh -c`。旧默认、旧参数、旧工具名和兼容开关均不保留。
- `execution_id` 是 manager 内的 opaque integer handle，不等于 OS PID，也不进入 SessionDB。
- 进程必须在初始等待前进入 manager。取消当前等待只停止等待，不终止已注册进程。
- 空输入轮询等待 5～300 秒；带输入等待 250 毫秒～30 秒。普通非 PTY 执行拒绝写入，唯独 `Ctrl-C` 转换为进程中断。
- 每次返回只消费新增输出。内存输出缓冲固定为 1 MiB，溢出时保留等量首尾并明确标出省略字节；完整输出追加写入临时日志。
- manager 最多持有 64 个进程，回收时保护最近 8 个执行，优先移除已退出执行，否则终止最久未使用且未受保护的进程。
- Akashic 保留比 Codex 更严格的 owner 和硬超时：执行只能由创建它的 `owner_session_key` 继续操作；硬超时、显式 stop、当前 query 结束或 runtime shutdown 必须确认终止平台执行边界。
- 该边界与 Codex 一致：Unix 是初始 process group，Windows 是 `taskkill /T` 可见的后代集合。显式 `setsid`、daemonize 或外部服务管理器会脱离该边界；需要覆盖它们时必须使用拥有 cgroup/Job Object 的受控 runner，不能把 `killpg` 描述成任意后代回收。
- 第一版不增加完成事件、自动回传或持久化队列。

## 理由

- 一个协议覆盖短命令、长命令和交互命令，模型不需要预测命令会运行多久。
- 增量输出避免重复把历史日志塞回上下文，长时间无输出时一次等待可以替代多次 30 秒轮询。
- 注册先于等待让 turn 取消与进程生命周期解耦，下一次工具调用仍可续接同一执行。
- `execution_id` 不与 Akashic 持久对话 session 或 OS PID 争夺语义。
- owner、硬超时和 runtime shutdown 让可续接不等于孤儿进程。

## 影响

- 工具 schema 和长任务调用方式发生 breaking change；旧 prompt 或调用方必须改用 `execution_id` 与 `write_stdin`。
- Shell 临时日志仍是诊断证据，不是会话权威状态；runtime shutdown 后执行句柄不恢复。
- 此改动不改变命令安全校验、网络限制或容器隔离责任。
- 默认 shell 从 Python 隐式 `/bin/sh` 改为 Codex 风格的用户 shell，是已确认的
  行为语义改变；显式未知或缺失 shell fail-loud，不静默换成另一种语义。
- Benchmark 只用于诊断协议是否减少无信息轮询，不以单题提分覆盖真实失败或改变 verifier。

## 验收

- 短命令一次返回退出码；长命令在初始等待后返回整数 `execution_id`。
- 初始等待被取消后，进程仍可通过同一 `execution_id` 续接。
- 连续 `write_stdin` 返回互不重复的输出片段，进程退出后最后一次返回退出码且不再返回 `execution_id`。
- PTY 可输入；非 PTY 普通输入明确失败，`Ctrl-C` 能中断进程。
- stop、硬超时、当前 query 结束、容量回收和 runtime shutdown 均确认终止平台执行边界。
- 复跑既往高轮询案例时，任务和 verifier 不变，并记录工具调用数、轮询次数、token、耗时和最终结果。
