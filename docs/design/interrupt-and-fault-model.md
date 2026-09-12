# 中断、工具结算与宿主故障模型（概念设计）

- 状态：proposed
- 日期：2026-09-12
- 关联决策：[0063 执行失败终态](../decisions/0063-execution-failures-have-terminal-results.md)
- 关联设计：[持久化状态地图](persistence-state-map.md)、[Codex 式同 Turn 输入设计](codex-style-same-turn-input.md)
- 参考实现：`codex@553df1c691fe8bf7747e50da22f1342984495ae0`；Durable AgentHarness 设计草案（harness-v2）

## 1. 问题与意图

本文只讲概念：当前实现的问题在哪里、目标语义应该是什么样、每个改动解决什么用户场景。不包含实现方案与迁移步骤，确认方向后另行立实施合同。

用户可见的问题集中在一句话：**"执行协议内"的错误处理已经很严，但"协议外"的基础设施失败没有统一语义**。具体表现为三类事故：存储故障时被当成业务错误无限重跑且每轮重新付费；单个 session 的底层异常杀死全部 session 的回复能力；一次可重试的 provider 错误（429、断流）就把整轮回复变成需要人工 resume 的失败。

本文借两个参考模型校准目标语义：

- harness-v2（Durable AgentHarness）的贡献是**故障与业务错误的分层**：storage write 失败是 `HarnessFault`，不是 operation error；fault 使 lane 停摆而不是触发另一次付费尝试。
- codex 的贡献是**中断生命周期与错误分类的具体形态**：`CodexErr{details, retry_delay}` 用显式 `is_retryable()` 分类承载"可重试"判断；中断时先把模型可见的 interrupted marker 持久化并 flush，再发出 `TurnAborted` 终态事件；pending 审批在任务观察到取消之后才清除；fork/resume 一个被中断的会话时追加 interrupt boundary。

## 2. 当前真实链路与 owner

```text
用户/渠道 Input, Control
        │
        ▼  admission 回调内排序（lane mutation 点）
Conversation (plugins/conversation/source.py)
   accept / control / pause / resume
        │  SOURCE_CHANGED 同步通知
        ▼
reply follow watcher (plugins/reply/follow.py)
   catalog.follow() 追赶；无 cursor，无回复队列
   drive() 每 session 独立；start() → run() 包裹失败写 Control("failure")
        │
        ▼
react (plugins/react/plugin.py)
   _pending_calls / _steps 从日志归约；先结算已提交调用再发模型请求
   Output append 带 expected_source_head CAS
        │
        ├─► ToolExecution (plugins/tools/execution.py)
        │     owner record: requested→prepared→started→done
        │     check_start 闸门；started 后被取消记 interrupted
        │     follow_abandon (plugins/tools/abandon.py) 结算被放弃调用
        │
        └─► BoundChatModel.complete (plugins/models/state.py)
              start_call 先于 provider I/O；finish_call 记成功/失败
```

进程级生命周期由 `agent/restart.py`（permit + drain）与 `agent/supervisor.py`（lifecycle frame、commit、exit code）负责；supervisor 的重启是请求驱动的 drain→commit→exit 协议，**不做健康自动重启**。

## 3. 已经成立的语义（不重复建设）

以下部分已符合 harness-v2 的核心要求，本文不复述其合理性，只作为缺口讨论的基线：

- 工具调用是 durable intent：四阶段 owner record、调用身份从持久消息推导、`started` 后取消记 `interrupted` 且不声称无效果、abandon 区分 `denied`/`interrupted` 并警告"可能已产生效果，不能据此重跑"。
- 恢复从日志归约：`_pending_calls`、`_steps`、abandon 边界都从 append-only 消息计算；没有平行的 Attempt 状态表。
- 取消等待真实结算：`_settle` 的 shield 循环保证普通取消先等已开始工具落终态；subagent 取消先持久化意图（pause Control）再撤权。
- 模型调用两段式记账：`start_call` 先于 I/O，`except BaseException` 连取消也记 `finish_call`，记账失败 chain 进原错误。
- CAS 提交：`expected_source_head` 保证失败草稿不触发工具；`check_source` 在新 Input/Control 已接纳时禁止新效果，不依赖取消信号及时送达。

## 4. 问题在哪里

按用户可感知场景列出；每条给出当前真实行为与错误原因。

### S1. 存储故障下的无界付费重跑

**场景**：sessions.db 所在磁盘满、锁竞争、或 `changed` 回调抛异常，导致 `Control("failure")` 落不了地。

**当前行为**：`Conversation.start` 的 `run` 包裹里，`program` 抛出的任何 `Exception` 都走 `failed` 回调写 failure Control；这次写入本身失败时异常上抛 → task 失败 → `drive` 记日志并把 `wake.changed = True` → `needs_reply` 仍为真（failure 没落地）→ 立即再 `start` → `program` 重跑 → 再次调用 `model.complete`。**无退避、无上限、每轮都是一次付费 provider 请求。**

**错误原因**：把宿主故障（authority write 失败）当成了 operation 的业务失败。harness-v2 的对应语义是：storage write 失败 fault 的是 harness 本身，lane 应当停摆等待外部恢复信号，而不是再付一次钱去撞同一堵墙。

### S2. 单个 session 的底层异常杀死全部回复

**场景**：某个 session 的 `reader.snapshot`/`head` 抛 sqlite 错误，或 `_tasks.admit`、`restart_gate.acquire` 失败。

**当前行为**：`drive` 只 catch `task.join()` 的异常；`source.open(session_id).start(program)` 的异常逃出 `drive` → `TaskGroup` 传播 → `follow` watcher 整体退出。`ctx.spawn` 把任务失败记入 health（`task:reply` unhealthy → ready=False），但没有自动重启路径接管。**所有 session 静默停止回复，直到进程重启。**

**错误原因**：per-item 失败（一个 session）与 watcher 级故障（整个能力）没有隔离；且 "watcher 死 → health 红 → 无人拉起" 是比 "watcher 死 → 自动重启" 更差的中间态。

同构问题存在于 `follow_abandon`（`plugins/tools/abandon.py`）：`abandon_call` 的 try 只覆盖 `LegacyReplyIdentityUnavailable`，`reply()` 构造、`reader.snapshot` 和其余异常都会杀死 watcher，此后所有 abandon 控制不再被结算，直到进程重启靠 `seen` 重扫恢复。被放弃调用将永远没有终态结果。

### S3. retryable 模型错误没有 turn 级重试

**场景**：provider 返回 429、网络瞬断、空响应。

**当前行为**：模型错误分类已经把 `RateLimitError`/`TransportError`/`ModelTimeoutError`/`EmptyResponseError` 标为 `retryable=True`（供调用方使用），但 react 主循环不消费这个标记——driver 有界内部尝试用尽后，任何错误直接让整轮回复记 failure Control，**用户必须显式 resume 才能续上**。

**对照**：codex 在 turn 级做 `is_retryable()` 判断，按 provider 的 `stream_max_retries` 有界重试，带 `retry_delay`（服务端 Retry-After）与 `backoff`，用尽后落到 `RetryLimitReached` 终态。对 IM 助手，一次 429 就要人工干预是明显的体验缺口。

### S4. abandon 晚到结果打挂新 run

**场景**：用户 abandon 一段含工具调用的工作，随后立即 resume 或发新消息；`follow_abandon` 还在给旧段写 `interrupted` ToolResult。

**当前行为**：react 用 `expected_source_head=head` 做 CAS。abandon 消费者以同 source 追加 ToolResult 会移动 head，若落在 `model.complete` 飞行期间，新 run 的 append 冲突 → `MessageConflict` → failure Control。**旧段的结算结果不改变新 run 的决策基础，却被当成抢占。**

**错误原因**：CAS 冲突只分"冲突/不冲突"，没分"决策基础是否真的变了"。delta 全是旧段 ToolResult 时应重算 head 重试；含新 Input/Control/Output 才是真的被抢占。

### S5. 中断对模型不可见

**场景**：用户 abandon 后 resume；或运行中发新输入打断生成。

**当前行为**：projection 把 abandoned 的 Output 整体跳过，abandoned 调用的 ToolResult 也跳过，`Control` 一律不进入模型上下文（`plugins/models/projection.py`）。结果：**用户在聊天里看得见"刚才那段被放弃了"，模型完全看不见**——模型可能把已提交的调用当成没发生过，或者对突然多出的新 Input 缺少"上一段被打断"的因果。

**对照**：codex 在中断时持久化一个模型可见的 interrupted marker（先 flush 再发 `TurnAborted`），fork/resume 被中断会话时也追加 interrupt boundary。akasic 不需要新增消息类型：abandon/pause Control 已是 durable 事实，projection 可以**派生**出 marker——归约层的事，不是新 owner。

### S6. 崩溃悬挂的 model_calls 无对账

**场景**：进程在 `start_call` 之后、provider 响应或 `finish_call` 之前硬崩溃。

**当前行为**：`model_calls` 行永远停在 `state='started'`，没有任何 sweep。同时 crash-after-request 意味着 provider 侧可能已计费，本地只有悬挂记录，usage 永远缺失。

**错误原因**：两段式写入缺了 reconciler 一端。对 `model_calls` 自己这张表做启动对账（把早于本进程启动的 `started` 标为 `error`）是 owner 内部的原位更新，不违反 messages 的 append-only 边界。

### S7. failure reason 无结构化分类

**场景**：conversation 的 failure Control 只有 `str(error)`。

**当前行为**：resume、UI、运维都分不出"值得重试的 429"与"重试也没用的 defect"；S1 的 fault 分类也没有落点。

**对照**：`plugins/wake/request.py` 的 `WakeFailure.retryable`（JSON 编码进 reason）已经是正确模式；codex 的 `CodexErrKind` 把低基数语义标签与诊断载荷分离。目标是把这一模式推广到 conversation failure，而不是发明第二套。

### S8. 结算等待无界

**场景**：工具插件内部死锁或不可取消的阻塞调用，使 tool task 永远不结束。

**当前行为**：`_settle` 的 shield 循环、`drive` finally 的 `while not task.done: join()`、`Tasks.close` 的排空都无界等待——**取消路径和插件停止都会被一个 hang 死的工具卡住**。

**目标**：结算看门狗。宽限期后由 owner 给调用写 `interrupted` 终态（契约已允许：interrupted 不声称无效果）+ `report_incident`，durable 状态闭合，不再等 asyncio 层的僵尸任务。这只关闭持久语义，不假装僵尸线程被杀死。

### S9. 次要观察（不要求本概念处理，登记待核）

- `react._complete` 在 `ContextLengthError` 重试路径上对同一个 `ExitStack` 先 `close()` 再 `enter_context`——当前 CPython 恰好允许，语义上是已关闭栈的复用，应按每 attempt 新栈处理。
- `agent/model_runtime/execution_history.py` 的 `_TERMINAL_STATUSES` 仍含 `"unknown"`——需确认该 shell 执行协议是否属于 0063 约束的公共终态词汇。
- `ToolExecution.deny_call` 目前没有生产消费者，能力已建成但无接入方。

## 5. 目标语义

### 5.1 错误三层分类（核心）

所有执行路径上的失败归入三层，三层各有不同的合法响应：

| 层 | 含义 | 合法响应 |
|---|---|---|
| 业务终态 | operation 的终态结果（工具 `error`/`denied`/`interrupted`，来源 `failure`/`pause`） | 持久化终态，不重跑；恢复走显式 resume |
| 可重试瞬态 | `retryable` 分类的 provider/transport 错误 | lane 内有界重试（计数 + 退避 + 服务端 `retry_delay`），用尽降级为业务终态 |
| 宿主故障 | authority write、admission、snapshot 求值、writer/reader 失败 | lane 标记 faulted 并停摆 + `report_incident`；不得再发起付费/外部效果；由 head 变化或显式操作解除 |

判据：**下一次重跑是否需要重新支付外部代价**。需要付费或产生外部效果的重试，只能由"瞬态"或"显式 resume"发起；宿主故障永远没有资格触发它。

### 5.2 lane fault 与 watcher 隔离

- lane（session × source）faulted 是内存判定，不需要新持久类型：`drive` 给每个 lane 记连续基础设施失败计数，超过阈值后停摆并 `report_incident`；catalog head 变化或显式 resume 清除。
- 所有"跟随日志的后台 owner"（reply follow、follow_abandon、delivery_policy follow、subagent watcher）遵守同一条规则：**per-item 异常 → incident + 跳过/退避该项，不杀死 watcher**。watcher 自身死亡继续由 `ctx.spawn` 记 health 失败；是否接 fiber restart 留给实施合同，但不得停在"红了没人管"的当前态。

### 5.3 react 的有界瞬态重试

- `model.complete` 抛出的 `retryable` 模型错误在 turn 内重试有限次（含退避，尊重服务端 Retry-After 若可得）；用尽后按业务终态记 failure。
- 每次重试前仍过 `check_source`：新输入到达立即停，重试不得屏蔽中断。
- `EmptyResponseError`（空响应）纳入同一瞬态路径。
- 计数默认内存即可（崩溃重置的代价是有界的少量重复请求）；若要防崩溃烧 token，用 owner record 按 `(source, latest Input seq)` 记 attempt——属于可选增强，不是本概念的必需项。

### 5.4 CAS 冲突区分

`writer.append` 冲突后重读 snapshot：delta 全是 ToolResult（含 abandon 消费者的晚到结算）→ 重算 head 重试一次；delta 含 Input/Control/同段 Output → 维持现状按抢占/取消处理。

### 5.5 中断的模型可见性

projection 从 durable `Control`（abandon/pause）边界**派生**模型可见的中断标记，替代现在"abandoned 段整体消失"的做法。被放弃调用的合成 ToolResult（`interrupted`，已带"可能有效果"警告）保留在投影内。不新增消息类型，不改变 append-only 语义——标记是归约产物。

### 5.6 记账对账

models store 启动时把 `started` 且早于此进程启动的 `model_calls` 标为 `error`（`failure='interrupted'` 语义），usage 保持未知而不是记零。若 provider 后续提供按 request id 查 usage 的能力，以 `BoundTool.query` 同形的可选 `query_call` 接入——接口预留，不承诺各 provider 都实现。

### 5.7 结构化失败 reason

conversation 的 failure Control 采用 `WakeFailure` 同形的 JSON reason：低基数 `kind` + `retryable` + 诊断 message。S1 的 fault 停摆以 `kind` 区分，resume/UI/运维可以区分"可重试的瞬态终态"与"defect"。

## 6. 解决什么场景的问题

| 场景 | 现状 | 目标 |
|---|---|---|
| 磁盘满/DB 锁时用户发消息 | 无限循环重跑，每轮付费请求打向坏存储 | lane faulted，incident 可见，不再付费；head 变化后恢复 |
| 某 session 的 reader 抛错 | 全部 session 静默停止回复 | 只有该 lane 退避；其余不受影响 |
| provider 429/断流/空响应 | 整轮 failure，需人工 resume | 有界自动重试；用尽才落终态 |
| abandon 后立即 resume | CAS 冲突概率性把新 run 记 failure | ToolResult delta 重试，正常续跑 |
| 用户中断后继续对话 | 模型看不到"上一段被打断/放弃" | 投影派生中断标记，模型有因果 |
| 崩溃在 model 调用中 | `started` 行悬挂，usage 永远缺 | 启动对账闭合，标注为中断 |
| 工具 hang 死 | 取消、停止、卸载全部被卡住 | 看门狗落 `interrupted` 终态 + incident |
| 排查 failure | 只有 `str(error)` | 结构化 kind + retryable + 诊断 |

## 7. 明确不引入的

- **不引入 operation/lane 持久记录表**：Input 接纳、pause/failure/resume/abandon Control、`_pending_calls` 归约已经承载等价语义；第二套表违反"同一事实只有一个 owner"。
- **不引入统一 `Result<T, E>` 包装或跨 owner 错误中间件**：错误分类留在各 owner；收益集中在模型调用一处，由 §5.3 局部解决。
- **不引入事件 replay 模型**：`catalog.follow` + heads 已是"不重放、重连拿新 snapshot"的等价物。
- **不改变 messages 的 append-only 与 owner 边界**：中断标记是投影派生物；`model_calls` 对账是该表自身的两段写闭合；lane fault 是内存态。
- **不在本轮决定 watcher 死亡是否自动重启 fiber**：先把 per-item 隔离与 incident 做实，重启策略由实施合同决定。

## 8. 分阶段方向（概念层）

1. **故障分层**：failure-Control 写失败/求值失败 → lane faulted + incident；回复驱动不再对故障重跑。解 S1、S2。
2. **watcher 隔离**：reply follow、follow_abandon、delivery_policy follow、subagent watcher 的 per-item 异常隔离与 incident 上报。解 S2 的其余半边。
3. **瞬态重试与结构化 reason**：react 内 `retryable` 有界重试；failure reason 采用 `WakeFailure` 同形结构。解 S3、S7。
4. **结算与投影细节**：CAS delta 重试、abandoned 段的中断标记投影、`model_calls` 启动对账。解 S4、S5、S6。
5. **无界等待与登记项**：结算看门狗、ExitStack 修正、`"unknown"` 归属确认、`deny_call` 接入或标注。解 S8、S9。

## 9. 验收边界

概念接受与否按以下可观察判据衡量（实施阶段各自细化）：

- 用故障注入让 `controls.append` 失败：lane 停摆、产生 incident、不再发起 `model.complete`；head 变化后恢复。
- 用故障注入让单个 session 的 reader 抛错：其余 session 正常回复，该 session 有 incident。
- 对 `model.complete` 注入一次 `RateLimitError`：同轮内自动重试成功，不产生 failure；连续注入至上限：落 failure 且 reason 可解析出 `retryable`。
- abandon + 立即 resume + 延迟落地的旧段 ToolResult：新 run 正常完成，不记 failure。
- abandon 后 resume：模型上下文里能看到该段被放弃的标记与被放弃调用的 interrupted 结果。
- 进程在 `start_call` 后崩溃重启：`model_calls` 不留永久 `started`。
- 工具不响应取消：看门狗后调用有 `interrupted` 终态，回复/停止路径退出。
