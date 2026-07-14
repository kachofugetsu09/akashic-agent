# Plan 002: 从 AgentLoop 提取 ConversationRuntime

> **Executor instructions**：执行前确认 Plan 001 已 DONE。逐步验证，遇到 STOP condition
> 停止。完成后更新 `advisor-plans/README.md`。
>
> **Drift check**：
> `git diff --stat 6b8f438d..HEAD -- agent/looping agent/core bus bootstrap session tests`

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH（被动 turn 主调用链与 shutdown）
- **Depends on**: `advisor-plans/001-thread-turn-contract.md`
- **Category**: tech-debt
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

app-server 必须调用一个稳定 application service，而不是伪造 channel message。当前
`AgentLoop` 同时拥有 MessageBus consumer、turn task、interrupt snapshot、核心执行和
错误转 outbound；该对象无法被程序化调用安全复用。本计划提取唯一的 turn owner，
让 channel worker 和未来 app-server 成为同级 adapter。

## Current state

- `agent/looping/core.py:376-428` 的 `run()`/`_run_inbound_turn()` 绑定 MessageBus。
- `agent/looping/core.py:507-543` 以 session key 管理 interrupt。
- `agent/looping/core.py:652-726` 包含真正可复用的 admission + pipeline 调用。
- `agent/looping/core.py:728-765` 的 direct API 只返回字符串。
- `agent/core/passive_turn.py:242-253` 已能在 `dispatch_outbound=False` 时返回
  `OutboundMessage`，可作为提取起点，但最终 API 应返回 Plan 001 的 `TurnResult`。
- `bootstrap/app.py:282-286` 把 `agent_loop.run()` 与 bus dispatcher/scheduler 一起作为
  primary tasks 启动。

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Runtime tests | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_conversation_runtime.py tests/test_turn_pipelines.py tests/test_runtime_smoke.py` | all pass |
| Race tests | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_runtime_races.py tests/test_channel_host.py` | all pass or exact existing equivalent |
| Typecheck | `/mnt/data/coding/akasic-agent/.venv/bin/pyright agent/control agent/looping bootstrap bus` | exit 0 |

## Scope

**In scope**:

- `agent/control/runtime.py`（新增）
- `agent/control/ports.py`（新增，定义 executor/event stream ports）
- `agent/looping/core.py`
- `bootstrap/passive_worker.py`（新增）
- `bootstrap/app.py`
- `bus/events.py` / `bus/queue.py` 仅限 adapter 所需改动
- `tests/test_conversation_runtime.py`（新增）
- `tests/test_runtime_races.py`（新增）
- `tests/test_turn_pipelines.py`、`tests/test_runtime_smoke.py`

**Out of scope**:

- JSON-RPC、socket、SDK、TUI 删除
- Telegram/QQ/Web 内部重写
- 移除全局 `_passive_runtime_lock`
- 同 thread 并行或同轮 steer

## Git workflow

- Commit: `refactor(runtime): extract conversation control service`
- 不 push，不开 PR。

## Steps

### Step 1: 建立 ConversationRuntime API

实现 transport-neutral API：

- `start_turn(request) -> TurnHandle`
- `read_turn(thread_id, turn_id) -> TurnRecord`
- `interrupt_turn(thread_id, turn_id) -> TurnRecord`
- `subscribe(turn_id) -> async iterator[TurnEvent]`
- `shutdown()`

`TurnHandle` 至少提供 `id`、`result()`、`events()`、`interrupt()`。启动应先持久化 queued
并立即返回 handle；task 在后台推进状态。

**Verify**：fake executor tests 证明 start 不等待完成，result 等待 terminal。

### Step 2: 迁移 active task 与 interrupt 所有权

将 `_active_tasks`、`_active_turn_states`、`_interrupt_states` 从 AgentLoop 移到
ConversationRuntime，并以 `turn_id` 为主键、thread id 为辅助索引。interrupt 必须
同时匹配 thread 和 turn。保留中断后续跑的现有领域能力，但把 snapshot 关联到明确的
被中断 turn。

**Verify**：测试旧 turn id 不能中断同 thread 的新 turn；重复 interrupt 幂等返回
terminal/idle 语义，不取消错误 task。

### Step 3: 建立有界事件流

每个 active turn 使用有界 event buffer。慢 observer 不得阻塞 pipeline；buffer 满时应
终止该 subscriber 并给出明确 `SlowConsumerError`，不能丢失 terminal event 后假装正常。
TurnRecord 是最终真相，断线客户端可用 `turn/read` 补状态。

内部 EventBus 仍用于 plugin hooks/observers；ConversationRuntime 只投影 Plan 001 定义的
稳定 TurnEvent，不把任意内部 dataclass 暴露成协议。

**Verify**：两个 subscriber 中一个卡住时另一个和 turn 正常完成；卡住者明确失败。

### Step 4: 把 MessageBus consumer 降为 adapter

新增 `PassiveMessageWorker`：消费 `InboundItem`，构造 `TurnRequest`，调用
ConversationRuntime，等待结果，再把 completed/failed/interrupted 映射为现有
OutboundMessage 或用户提示。它负责 `complete_inbound()`，并保持取消期间的 lane 计数
收束语义。

删除 AgentLoop 内的 bus-consumer 主循环；如果暂留 `AgentLoop` 名称，它只能作为核心
pipeline facade，不能再拥有 transport lifecycle。优先在本计划末将其重命名为表达职责
的内部对象，避免新代码继续依赖旧名字。

**Verify**：现有 channel smoke tests 的可观察结果不变。

### Step 5: 集成 AppRuntime 生命周期

`AppRuntime.start()` 构建唯一 ConversationRuntime；primary tasks 启动
PassiveMessageWorker、bus outbound dispatcher 和 scheduler。shutdown 顺序：停止新 ingress
→ interrupt/await active turns（按明确 policy）→ 停 worker → 关 event subscriptions → 关
core/store。

**Verify**：startup 任一步失败和外部 cancellation 都无悬挂 task、未完成 lane 或丢失
原始异常。

## Test plan

- 新增 `tests/test_conversation_runtime.py`，覆盖成功、失败、中断、错误 turn id、双订阅、
  慢订阅、shutdown。
- 复用 `docker/debug/runtime_race_probe.py` 体现的 admission 场景，但 unit tests 不启动
  真实 provider。
- 保留 Plan 001 状态机 tests。

## Done criteria

- [ ] 程序化调用不需要 MessageBus 或伪造 channel writer
- [ ] ConversationRuntime 是 active turn 和 cancel 的唯一 owner
- [ ] 每个 event 都有 thread/turn id
- [ ] MessageBus worker 只是 adapter
- [ ] 慢 subscriber 不阻塞 turn
- [ ] targeted tests 与 pyright 全通过

## STOP conditions

- plugin hot-reload snapshot lease 无法在 background turn task 中保持当前语义。
- scheduler/process_direct 的调用无法迁移到同一 ConversationRuntime 而必须保留第二执行链。
- MessageBus lane 的完成只能由已删除的 AgentLoop finally 保证且无法在 adapter 复现。
- shutdown 需要吞掉非 cancellation 异常才能通过测试。

## Maintenance notes

以后新增 transport 只能依赖 ConversationRuntime。评审应搜索 `process_direct` 和
`_process_with_runtime_admission` 的外部调用；若仍有绕过 service 的调用，本计划未完成。
