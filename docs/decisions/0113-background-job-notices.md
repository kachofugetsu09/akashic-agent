# 0113 · 后台任务完成通知：忙时注入下一步，闲时唤醒原会话

- 状态：proposed / implemented for review
- 日期：2026-10-11
- 补充：[0096](0096-context-reminders-keep-their-first-position.md) 的可回放提醒；Shell 进程 owner 分区。
- 参考：deepseek-harness `tool-jobs` 的 inject / followup 投递语义与 `maxConsecutiveWakes`。

## 决定与理由

Agent 需要把长任务（例如 `codex exec`、`claude -p`）派发出去，期间继续和用户对话或空闲，
任务结束时由运行时回调，而不是模型轮询或子任务同步阻塞等待。

1. **生产方只登记等待入口。** `jobs.notices.v1` 只有 `watch(JobWatch)`；`settled()`
   在结算时返回 `JobOutcome`，返回 `None` 表示结果已被模型收走或任务被显式停止。
   注册表与生产方种类无关，Shell 后台命令是第一个生产方。
2. **Shell 增加 `run_in_background`。** 后台命令进入 `(会话来源, "background")` 分区，
   回合结束和放弃时的 `shell_cleanup` 只回收回合分区，后台进程由 `task_stop`、硬超时或宿主停止回收。
   `write_stdin` / `task_stop` 先查回合分区，再查后台分区。
3. **结算判定不消费输出。** 进程运行时新增 `execution_states(owner)`：已退出且没有调用方
   正在等待它的终态才算结算。首个窗口和 `write_stdin` 等待期间都登记为等待者，
   因此模型自己收走的终态不会再通知（对应 DSH 的 `awaited`）。Host Bridge 增加同名短 RPC，
   不使用会阻塞 manager 排空的长轮询。
4. **忙时注入：** jobs 插件注册 `context` 材料。react 每一步都会重新准备材料，
   当前用户输入领取未读结算，以 `replay=True` 提醒进入请求；同一输入的后续步骤保持同一组提醒，
   新输入出现后旧领取退出。不修改 AgentCore 或 react。
5. **闲时唤醒：** 结算后经 `conversation.complete.v2` 等待原会话槽空闲，在槽内再次核对：
   已被某个回合领取的结算不唤醒；否则以 `job:<id>` 来源运行一轮回复，并按最近输入的渠道显式投递。
6. **自激约束：** `max_consecutive_wakes`（默认不封顶）按用户输入连续计数，耗尽后留给下一次
   用户输入注入；`completion_delivery: quiet` 只注入不唤醒。

```text
shell(run_in_background) ──► watch(JobWatch) ──► settled()
                                                   │
               ┌───────────────────────────────────┴──────────────────┐
     回合运行中：下一步材料领取并注入            会话空闲：complete 槽内核对 → 唤醒一轮 → 渠道投递
```

## 生命周期与持久状态

与 DSH `jobs-local` 同等复杂度：通知状态只在进程内，宿主停止时后台进程随 manager 一起回收，
未投递的结算随之消失。已经产生的唤醒回复是普通 Message，照常持久化。不新增表、迁移或恢复指针。

## 已知限制

- 结算落在模型最后一步生成期间时，本回合看不到它，回合结束后由唤醒补上。
- 回合领取后若该回合失败且没有后续步骤，下一次用户输入会让这条领取退出。
- 后台子任务（subagent）仍使用自己的持久回传路径；本决定不改变它。
- 内置 `codex-delegate` skill 改为主会话直接后台 Shell 启动 `codex exec`（bypass 模式），
  不再包一层阻塞等待的 subagent；续聊用 `codex exec resume <thread_id>`，同一会话串行。

## 验收

`tests/test_background_jobs.py` 用真实 CompositionRoot 和夹具模型验证：忙时第二步请求携带通知
且不再唤醒；闲时唤醒一轮并经测试 sender 投递；真实 `sleep` 后台 Shell 越过回合清理并以
`exit_code=0` 唤醒原会话。`tests/test_process_execution.py` 验证结算判定与 Bridge 编解码。
