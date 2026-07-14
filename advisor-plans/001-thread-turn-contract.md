# Plan 001: 固化现状并建立 thread/turn 持久契约

> **Executor instructions**：逐步执行，每一步验证通过后再继续。发现 STOP condition 时
> 停止并报告，不要自行改协议。完成后更新 `advisor-plans/README.md` 状态。
>
> **Drift check**：
> `git diff --stat 6b8f438d..HEAD -- session bus agent/core agent/looping tests`
> 若 in-scope 文件已变化，先逐项核对本计划的 Current state。

## Status

- **Priority**: P1
- **Effort**: L
- **Risk**: HIGH（数据库 schema 和公共事件类型变化）
- **Depends on**: none
- **Category**: tech-debt / tests
- **Planned at**: commit `6b8f438d`, 2026-07-14

## Why this matters

正式协议不能建立在 `session_key + 最终字符串` 上。先把 thread、turn、item、usage 和
失败状态变成可测试、可持久化的领域契约，后续 app-server 才不会把当前实现细节冻结
成 public API。

## Current state

- `session/store.py:228-248` 仅创建 `sessions` 和 `messages` 表。
- `bus/events_lifecycle.py:35-50` 的 turn/stream 事件没有 `turn_id`。
- `agent/core/passive_turn.py:449-466` 的 turn id 仅用于 diagnostic context。
- `agent/core/passive_turn.py:580-603` 将 provider 异常转成成功形态的普通回复。
- `agent/core/passive_turn.py:2067-2075` 已聚合 model usage，可进入正式 result，不需要
  重新从日志计算。

项目约定：信任边界严格校验，内部违反契约 fail-fast；非平凡函数使用短中文 docstring
和真实的编号阶段注释。参考 `session/store.py:33-94` 的数据库 JSON 边界校验风格。

## Commands you will need

| Purpose | Command | Expected |
|---|---|---|
| Target tests | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_session_store.py tests/test_turn_contract.py` | all pass |
| Typecheck | `/mnt/data/coding/akasic-agent/.venv/bin/pyright session bus agent/control agent/core agent/looping` | exit 0 |
| Regression | `/mnt/data/coding/akasic-agent/.venv/bin/pytest -q tests/test_turn_pipelines.py tests/test_agent_core_p7_commit.py` | all pass |

## Scope

**In scope**:

- `agent/control/models.py`（新增）
- `agent/control/ids.py`（新增）
- `session/store.py`
- `session/manager.py`
- `bus/events_lifecycle.py`
- `agent/looping/core.py`、`agent/core/passive_turn.py`、
  `agent/lifecycle/phases/after_turn.py` 中的直接 event producers
- 上述事件构造签名涉及的现有 tests/plugin fixtures
- `tests/test_turn_contract.py`（新增）
- `tests/test_session_store.py`（新增）

**Out of scope**:

- socket/JSON-RPC/SDK/CLI
- `MessageBus` 消费方式
- 不实现 concurrent turns 或 `turn/steer`
- 不搬迁现有 session key，不批量重写历史 message

## Git workflow

- Branch: `refactor/programmatic-control-api`
- 每个计划一个逻辑 commit；建议本计划使用
  `refactor(control): establish persisted turn contracts`
- 未经操作者指示不要 push 或开 PR。

## Steps

### Step 1: 用 characterization tests 固定现有行为

新增 tests 覆盖：普通成功、reasoner/provider 失败、after-reasoning 持久化失败、中断、
同 session 连续两轮、usage 存在与缺失。先写现状断言，再开始重构。

**Verify**：运行 target tests；新增测试在未改生产代码前应精确暴露缺失的 turn contract，
不要用宽泛 `raises(Exception)`。

### Step 2: 定义 transport-neutral 领域类型

在 `agent/control/models.py` 定义：

- `ThreadRecord`
- `TurnStatus`：`queued|in_progress|completed|interrupted|failed|cancelled`
- `TurnRecord`
- `TurnItem` discriminated union
- `TurnUsage`
- `TurnError`
- `TurnRequest` / `TurnResult`

这些类型不能 import socket、JSON-RPC、Channel 或 SDK。时间统一使用 UTC aware datetime，
序列化为 RFC 3339。`ids.py` 负责 thread/turn/item id 生成；若 Python 3.12 环境没有
标准 UUIDv7，使用 `uuid4`，不要新增仅为 UUIDv7 的依赖。

**Verify**：model tests 覆盖状态枚举、UTC 序列化和 ID 唯一性。

### Step 3: 持久化 turn 状态机

在 `SessionStore._init_schema()` 增加 `turns` 表和必要索引：至少包含 `id`、
`session_key`、`status`、`input_json`、`items_json`、`usage_json`、`error_json`、
`created_at`、`started_at`、`completed_at`。增加 typed encode/decode helpers，复用当前
JSON 损坏时 fail-loud 的模式。

所有状态更新必须是 compare-and-set：例如只有 `queued` 可进入 `in_progress`，只有
terminal 之前可进入 terminal。非法跳转抛领域异常，不静默覆盖。

**Verify**：SQLite tests 覆盖合法路径、非法转换、进程重开后恢复和损坏 JSON。

### Step 4: 给内部生命周期事件添加 turn identity

`TurnStarted`、`StreamDeltaReady`、`TurnCommitted`、`ToolCallStarted`、
`ToolCallCompleted` 必须携带 `turn_id`。由一次 turn 的创建者生成，沿执行上下文传递，
不要在每个事件 producer 重新生成。

更新所有事件实例化点、插件 fixture 和 tests。允许 plugin consumer 读取新字段，但本
阶段不做 protocol projection。

**Verify**：`rg "TurnStarted\(|StreamDeltaReady\(|TurnCommitted\(|ToolCallStarted\(|ToolCallCompleted\("`
逐个确认 producer 提供真实同一 `turn_id`。

### Step 5: 分离领域失败与 channel 文案

让核心 pipeline 的 provider/reasoner 失败进入 `TurnResult(status=failed, error=...)` 或抛
明确 typed exception；不再返回伪成功 `OutboundMessage`。保留现有日志 traceback。
暂时由当前 MessageBus worker 把 typed failure 映射成原有用户文案，保证渠道回归不变。

**Verify**：失败测试同时断言领域 result 为 failed、channel 用户仍得到安全中文文案。

## Test plan

- 新增至少 12 个 contract/store tests。
- 模式参考 `tests/test_turn_pipelines.py` 的 fake core 和 `tests/test_dashboard_api.py` 的
  临时 `SessionStore`。
- 必须包含进程重开数据库测试，不能只测 in-memory object。

## Done criteria

- [ ] 所有生命周期事件可关联到稳定 `turn_id`
- [ ] turn terminal 状态和 usage 可从 SQLite 重读
- [ ] 非法状态转换 fail-loud
- [ ] provider 失败不再伪装成 completed 文本
- [ ] target、regression、pyright 全通过
- [ ] 无 socket/JSON-RPC/SDK 改动

## STOP conditions

- 现有 plugin API 明确承诺 lifecycle dataclass 构造签名长期稳定。
- turn 状态和消息提交需要跨两个独立数据库文件才能保持一致。
- usage 在成功路径无法从 `TurnCommitted.react_stats` 或等价 typed result 获得。
- 实现需要为缺失数据造默认 usage 或吞掉数据库损坏。

## Maintenance notes

后续协议 schema 只能投影这些领域类型，不得另建第二套 turn status。评审重点检查状态
转换是否原子、失败是否仍可能落成 completed，以及 turn id 是否在一次执行中保持一致。
