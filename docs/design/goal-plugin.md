# 目标（goal）插件任务合同

- 状态：draft（按首轮评审修订）
- 日期：2026-11-02
- 目标分支：`feat/goal-state`（stacked）
- 上游：Codex `codex-rs/ext/goal`、Claude Code `src/commands/goal*`

## Goal

让 Akashic 在同一个 Session 里拥有一个可持久化的目标：用户设置目标后，模型在目标未完成时自动继续下一轮，直到目标完成、被阻塞或被用户暂停。目标状态、续跑次数、空转次数与 token 用量都由插件拥有；续跑输入和状态推进在同一个 Ledger 事务里提交。

这一版只做“目标生命周期 + 自动续跑 + 安全熔断”，不引入新的调度器、模型或工具协议。

## 先对齐命名

两边的实现并不完全同名：

- Codex：`ThreadGoal` + `ext/goal`，工具是 `get_goal` / `create_goal` / `update_goal`，状态为 `active / paused / blocked / usage_limited / budget_limited / complete`。线程空闲时自动开始下一轮（`continue_if_idle`）。
- Claude Code：交互命令是 `/goal <condition>`，不是 `/loop`。`/loop` 是另一个“按 cron 重复执行 prompt”的技能。`/goal` 通过一个 Stop hook 在模型准备停止时检查条件；未满足就把 hook 反馈当成续跑指令。

Akashic 采用 Codex 的持久状态模型，吸收 Claude Code 的“停止前检查”语义：**每次目标轮次结束后，由目标运行时判断是否继续；继续时在同一个对话来源上追加下一条目标输入。**

Akashic v1 不实现 Codex 的 `usage_limited`；模型/宿主用量错误会表现为 `Control("failure")`，计入空转并在阈值后转 `blocked`。

## 能力边界

### 属于本插件

- 每个 Session 至多一个目标；目标有 objective、status、创建/更新时间、续跑次数、连续空转次数、token 预算与用量、结算游标。
- `/goal` 命令：设置、查看、暂停、恢复、清除、设置预算。
- 工具：`get_goal`、`create_goal`、`update_goal`。模型只能把目标改成 `complete` / `blocked` / `paused`；`active` 与预算状态由用户或系统控制。
- 自动续跑：目标为 `active` 且当前对话来源空闲时，在**同一个 `conversation` 来源**追加一条 `author="goal"` 的内部 Input，由默认回复链路执行下一轮。
- 熔断：连续 3 个目标轮次没有任何活动 → `blocked`；单目标续跑次数上限 → `paused`；预算耗尽 → `budget_limited`。
- 用量：扫描目标轮次内所有 Output 的 `model.facts.call_record_id`，从 `MODEL_CALL_STATS` 累计，并持久化已结算调用 ID。

### 不属于本插件

- 不新增模型、工具协议、Turn 类型或第二套消息账本。
- 不改变默认回复链路的 `max_steps`、工具授权、上下文压缩与投递规则。
- 不把目标当成更高优先级指令：objective 始终是用户数据，续跑提示要明确这一点。
- 首版目标轮次与普通对话共用工具目录，不额外复制一份工具授权/禁用逻辑；是否在自动轮次禁用记忆工具留作后续配置项。

## 数据模型

`plugins/goal/state.py` 定义纯状态机；状态存进 Ledger 的 `OWNER_STATE`，不再使用独立 JSON 文件。

```python
class GoalStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    BUDGET_LIMITED = "budget_limited"
    COMPLETE = "complete"

@dataclass(frozen=True)
class Goal:
    session_id: str
    goal_id: str
    objective: str
    status: GoalStatus
    token_budget: int | None
    tokens_used: int
    continuations: int
    consecutive_empty: int
    settled_through_seq: int
    last_settled_message_id: str | None
    continuation_id: str | None
    usage_coverage: Literal["exact", "partial", "unavailable"]
    created_at: datetime
    updated_at: datetime
    last_reason: str | None = None
```

- `objective` 长度 1..4000，去首尾空白后保存；空目标拒绝。
- `token_budget` 只接受正整数；`None` 表示不限额。
- 终态：`complete`、`budget_limited`。`blocked` 与 `paused` 不会自动续跑，只有用户显式恢复才会回到 `active`。
- 同一 Session 同时只有一个目标；`create_goal` 只在没有目标或上一个目标为 `complete` 时成功。
- `settled_through_seq` 是最后一个已结算目标轮次的边界；状态推进与续跑追加在同一个事务里提交，因此不需要额外的已计费调用列表。

### 持久化

状态存进 `OWNER_STATE`，与消息写入同一个 SQLite authority：

- `goal:<session_id>`：当前目标记录（含 `settled_through_seq`、`last_settled_message_id`、`continuation_id`）。
- `goalop:<kind>:<session_id>:<operation_id>`：工具/命令的幂等回执，含请求 digest、响应文本、目标快照。
- 记录用 `OwnerTransaction.save(..., expected_version=...)` 做 CAS；状态推进和续跑 Input 的 `append` 在同一个事务里完成。
- 操作回执按最近 1024 条保留；超限时在同一事务里删除最旧的 `goalop:` 记录。

## 状态机

```text
(no goal) --create--> active
active --update complete--> complete
active --update blocked--> blocked
active --update paused--> paused
active --3 次空转--> blocked
active --续跑上限--> paused
active --预算耗尽--> budget_limited
active --用户新输入抢占--> active（目标轮次被同来源取消，记录 interrupted，不自动续跑直到用户轮次结束）
paused --/goal resume--> active
blocked --/goal resume--> active
complete --create--> active（新目标，新 goal_id）
任意状态 --/goal clear--> (no goal)
```

模型侧限制：

- `update_goal(status="complete")` 只允许在 `active` 下调用；完成是“逐项证据已核对”的声明，不能因为预算快用完或想停下就调用。
- `update_goal(status="blocked")` 只允许在 `active` 下调用；提示要求同一阻塞连续出现至少 3 个目标轮次后才调用。
- `update_goal(status="paused")` 只允许在用户明确要求暂停时调用；模型不得自行暂停。
- 模型不能把目标改回 `active`，也不能设置 `budget_limited`。

完整转移表（actor / 前置状态 / 结果 / 计数重置）：

| actor | 前置 | 结果 | 计数 |
| --- | --- | --- | --- |
| `/goal <objective>` | 无目标或 complete | active，新 goal_id | 全部归零，`settled_through_seq` = 当前来源 head |
| `create_goal` | 无目标或 complete | active，新 goal_id | 同上 |
| 模型 `update_goal` | active | complete / blocked / paused | 保留计数 |
| 运行时空转 | active | blocked | 连续空转 ≥ 3 |
| 运行时续跑上限 | active | paused | `continuations` ≥ 上限 |
| 运行时预算 | active | budget_limited | `tokens_used` ≥ `token_budget` |
| 用户新输入 | active 且目标轮次运行中 | 目标轮次取消，status 仍 active | `settled_through_seq` 推进到用户 Input 之前 |
| `/goal pause` | active | paused | 保留计数 |
| `/goal resume` | paused / blocked | active | 连续空转清零 |
| `/goal clear` | 任意 | 删除目标 | — |
| `/goal budget <n>` | active / paused / budget_limited | 设置预算；`n > tokens_used` 时从 budget_limited 回到 active，否则保持 budget_limited | 保留计数 |
| `/goal budget clear` | active / paused | `token_budget=None` | 保留计数 |

## 命令与工具

### `/goal`

| 输入 | 行为 |
| --- | --- |
| `/goal <objective>` | 创建或替换（仅上一个目标 complete 时）目标，状态 `active`；命令结束后由运行时开始第一轮。 |
| `/goal` | 显示当前目标、状态、续跑次数、用量、最近原因与已结算调用数。 |
| `/goal pause` | `active` → `paused`；命令输入走同一来源，接纳时取消当前模型循环。已开始的工具会照常结算；需要立即中断工具时用聊天停止按钮。 |
| `/goal resume` | `paused` / `blocked` → `active`，清空连续空转计数，等待运行时续跑。 |
| `/goal clear` | 删除目标；命令输入走同一来源，接纳时取消当前模型循环。已开始的工具会照常结算。 |
| `/goal budget <n>` | 设置 token 预算；budget_limited 下加预算可继续。 |
| `/goal budget clear` | 清除预算；budget_limited 下清除后回到 active。 |

命令以 `(kind="command", session_id, message_id)` 作为幂等键；恢复时返回同一条结果文本。

### 工具

- `get_goal`：返回状态、objective、预算、已用/剩余 token、续跑次数、最近原因、usage coverage。
- `create_goal(objective, token_budget?)`：仅在无目标或上一个目标 `complete` 时成功；否则返回明确错误，提示用 `update_goal` 只改状态。
- `update_goal(status)`：只接受 `complete` / `blocked` / `paused`，并校验当前状态。

工具 `prepare` 固定目标快照，`invoke` 与 `OWNER_STATE` 写入同一次事务；`query(key)` 返回 `None`，重放由 `invoke` 的 OWNER_STATE 幂等事务完成（操作 key 不含 Session，不做跨会话查询）。

## 自动续跑

### 同来源追加，不新增来源

目标输入追加到 `conversation` 来源，`author="goal"`：

- 与用户消息共享同一个 `SourceSession`，`(session_id, source)` 串行化，不存在两个来源并行推理。
- 用户新输入到达时，`SourceSession.accept` 会取消正在运行的目标轮次；目标保持 `active`，用户轮次结束后运行时再续。
- 默认回复链路不需要认识 goal 来源；它只看到一条普通的待回复 Input。
- UI 通过 `author="goal"` 识别系统输入，不把它当用户发言。

续跑输入只有一个 `text` part：自然语言续跑提示，开头明确“系统自动继续目标，不是用户的新发言”，并在 `<objective>` 数据块里带上目标原文与预算快照。

### 触发与结算

`plugins/goal/runtime.py` 用一个单 worker 队列处理 Session：

1. `catalog.follow(wake_on=(Output, Control), poll_interval=1.0)` 只作为唤醒；目标状态与结算游标是唯一事实来源。
2. 每次唤醒，用 `OWNER_STATE` 扫描有 active 目标的 Session，入队。
3. `advance(session_id)` 在同一个 `OwnerTransaction` 里完成“结算上一轮 + 追加下一轮”：
   - 读目标记录与当前 `conversation` 来源 head。
   - 若没有目标或不是 active → 返回。
   - 找 `settled_through_seq` 之后第一条 `goal-continuation:<goal_id>:<n>` Input：
     - 没有 → 追加第一条续跑 Input。
     - 有 → 扫描它之后的消息，直到第一个终态 Output/Control，或第一个非 goal 的 Input（用户抢占）。
   - 计算活动与用量：
     - 活动：至少一个 `ToolResult.outcome == "success"`，或至少一段非空正文。
     - 用量：收集区间内所有 `model.facts.call_record_id` 后查 `MODEL_CALL_STATS`；结算游标保证同一轮不会重复计费。
   - 若用户抢占：`settled_through_seq` 推进到用户 Input 之前，不追加；等用户轮次结束后再续。
   - 若终态是 `Control("failure")`：计为空转，原因写 Control.reason。
   - 若连续空转 ≥ 3 → blocked；续跑上限 → paused；预算耗尽 → budget_limited。
   - 否则 `save` 新目标记录，并 `transaction.append` 下一条续跑 Input，`expected_source_head` 固定为事务开始时读到的 head。
   - 事务内 `transaction.source_changed(reader, "conversation", expected_head)` 必须为 false；用户输入在这期间提交会让整个事务重试，不会出现“检查完空闲又被插队”的续跑。
4. 事务冲突或 head 变化时，重新读取并重试（最多 3 次）；仍失败就等下一次唤醒。

### 为什么不用 Stop hook 直接续在同一轮

Akashic 的 `react` 在模型没有工具调用时提交最终 Output 并结束 Turn；要在同一 Turn 内继续，需要改 `react`/`REPLY_EXECUTE` 的停止语义。目标插件选择“同来源的新一轮”：

- 不需要改 Core 的 Turn 关闭规则；
- 停止、取消、插话、压缩、工具授权全部沿用现有来源语义；
- 每个目标轮次在账本里可审计，崩溃后由来源待回复状态恢复。

代价是每个目标轮次会多一条 `author="goal"` 的 Input/Output，聊天里会看到一条“系统自动继续目标”的提示。这与定时任务的现有 UX 一致。

## 安全与熔断

- **空转熔断**：连续 3 个目标轮次没有成功工具结果、也没有非空正文 → `blocked`。任何成功工具结果或非空正文都会清零计数。`denied` / `error` / `interrupted` 的工具结果不算活动。
- **续跑上限**：默认 100 轮；达到后 `paused`，不自动恢复。
- **用户抢占**：同来源新输入取消目标轮次；目标保持 `active`，用户轮次结束后再续。不把抢占写成 `paused`。
- **预算**：`tokens_used` 累计目标轮次内所有已结算调用；只计目标自身烧的钱，用户轮次与缓存命中的输入不计入。`usage_coverage` 记录 exact/partial/unavailable。有限预算下 coverage 不是 exact 时，保守地 `paused`，原因“用量无法确认”，不继续花钱。
- **停止语义**：用户停止目标轮次（`pause`）→ 目标 `paused`；模型失败（`failure`）→ 计入空转；`abandon` → `paused`。
- **不重复计费**：结算游标与 token 增量在同一个事务里提交；事务失败时两者都不生效，重启不会重复计费。
- **不写记忆**：首版不额外禁用记忆工具；目标轮次与普通对话共用工具目录。需要禁用时再单独加 source-specific tool view，不在本版偷改回复链路。

## 插件边界

`plugins/goal` 注入：

- `PLUGIN_CONFIG`、`MESSAGE_CATALOG`、`MESSAGE_WRITERS`、`OWNER_STATE`、`CONTENT`
- `TOOLS`、`ALL_TOOLS`、`COMMANDS`、`TASKS`
- `MODEL_CALL_STATS`
- `REPLY_STATUS`（只作为“默认回复存在”的依赖；不直接调用回复程序）

提供：

- `GOAL_STATUS`：只读状态查询，供 UI/其他插件使用。
- `input_part_key("goal.continuation")`：如果未来允许客户端携带目标片段，由本插件校验。

不修改 `plugins/react`、`plugins/reply_program`、`plugins/conversation` 的合同；只通过 `MESSAGE_WRITERS` 在 `conversation` 来源追加一条内部 Input。

## 测试计划

### 单元

- 状态机：合法/非法转移、objective 长度、预算校验、终态不可自动恢复、抢占不改变 status。
- OWNER_STATE：记录 schema、版本 CAS、操作幂等、过期 `goal_id` 的结算被拒绝、操作回执保留上限。
- 命令：设置/查看/暂停/恢复/清除/预算、重复 message_id 返回同一结果。
- 工具：create 只在无目标或 complete 时成功；update 只接受三个状态；get 返回剩余预算与 coverage。

### 运行时

- 目标轮次结束且目标 active → 同事务追加下一条续跑 Input。
- 连续 3 个空轮次 → blocked；中间有一次成功工具结果 → 计数清零。
- `update_goal(complete)` → 不再续跑。
- `pause` Control → paused；`failure` Control → 计入空转。
- 用户新输入 → 目标轮次取消，`settled_through_seq` 推进到用户 Input 之前；用户轮次结束后再续。
- 重启：已有 pending goal Input 时只由默认回复链路接手，不重复写 Input；结算游标不重复计费。
- 续跑上限 → paused。
- 有限预算 + coverage 非 exact → paused，不继续。

### 集成

复用 `tests/test_default_reply.py` 的组合夹具，加载 `goal` 插件与假模型：

1. `/goal ship the feature` → 账本出现 author=goal 的 Input，模型被再次调用。
2. 假模型第二轮调用 `update_goal(complete)` → 目标 complete，后续不再出现 goal Input。
3. 假模型连续返回空响应（`EmptyResponseError` → Control failure）→ 3 轮后 blocked。
4. 用户在目标轮次中发送新消息 → 目标轮次被同来源取消，用户轮次完成后继续。

### Monkey test

`tests/test_goal_monkey.py`：固定随机种子，生成 200 组操作序列（set/resume/pause/clear/complete/blocked/empty/activity/restart/preempt），每组 50 步，覆盖：

- 同一 Session 最多一个目标；
- 终态不会自动回到 active；
- `continuations` 在单个 goal_id 内单调不减；
- 空转计数只在无活动轮次增长，活动轮次清零；
- 任何操作序列后 OWNER_STATE 记录都能 round-trip 通过 schema 校验；
- 重复 key 的操作不会改变第二次结果；
- 旧 goal_id 的结算不会写到新目标；
- 用户 Input 与续跑追加的竞争：事务冲突时不会同时产生两个 pending Input。

## 变更与持久化

```yaml
change_type: feature
semantic_delta: compatible
capability_owner: plugins/goal
consumer_scope:
  - 默认回复链路（conversation 来源新增一种内部 Input）
  - 聊天 UI（按 author=goal 识别系统输入）
runtime_patch: none
authoritative_state_owner: plugins/goal
protected_state:
  - 现有 Session 消息与 Turn 投影
  - 默认回复的停止、取消与工具授权
```

## 验收

分层验收：第 1 层只跑 state/store/tools/command；runtime/integration 在第 2 层，monkey 在第 3 层。

- `pytest tests/test_goal_state.py tests/test_goal_store.py tests/test_goal_tools.py tests/test_goal_command.py tests/test_goal_runtime.py tests/test_goal_monkey.py`
- 集成测试里至少一条真实 `/goal` → 续跑 → `complete` 的账本证据。
- `python -m pytest tests/test_default_reply.py -q` 不回归。
- PR 按 stacked 顺序合并：状态/命令/工具 → 续跑/熔断 → 预算/展示/monkey。

## 评审记录

- 第一轮 codex review（gpt-5.6-sol）：reject。指出独立 JSON 状态与 Ledger 追加非原子、跨来源准入竞态、token 计费不完整。
- 修订：目标状态迁入 `OWNER_STATE`；续跑输入回到 `conversation` 来源；结算与追加同事务；按整个目标轮次扫描 `model.facts` 计费。
- 第二轮 codex review：reject。指出抢占游标卡死、续跑上限未执行、跨会话 `query`、partial coverage、续跑提示快照过期、回执保留越界。
- 修订：抢占游标推进到用户输入；上限在 `resume` / `mark_continuation` / `settle_round` 三处执行；工具 `query` 返回 `None`；partial/unavailable 都暂停；提示由结算后的目标渲染；回执与写入同事务。
- 第三轮 codex review：reject。指出 `resume` 可绕过上限、空 `call_ids` 被当成 exact 0、回执保留事务不安全。
- 修订：`resume` 拒绝耗尽目标；空 `call_ids` 返回 `unavailable`；保留按 `reserve=1` 计算并在事务内删除，冲突时整体重试。
- 第四轮 codex review：**approve**，remaining blockers: none。
- Akashic review bot 复核：指出子命令首词误匹配、budget_limited 死路、`/goal` 缺 recover、抢占后 pause 被续跑覆盖、monkey 覆盖偏弱。已修：子命令改为整词匹配；budget_limited 可加预算恢复；命令注册 recover；抢占分支识别 pause/abandon 并暂停目标；monkey 增加 model.facts、工具轮次、终态规则与更均衡的动作分布。
- 最终验证：237 个目标/默认回复测试通过；`pyright plugins/goal` 与 `pyright --project pyrightconfig.tests.json --level error` 均 0 errors；`plugin_boundary.py check` 通过。
