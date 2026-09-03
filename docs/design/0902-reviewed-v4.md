# Akashic v4：从 Message 到 `react` 的插件架构

- 文档版本：`0902-reviewed-v4`
- 日期：2026-09-03
- 状态：设计提案，等待批准
- 当前代码基线：`47896b4200731183a54081e2eca77602a0881a0a`
- 需求来源：2026-09-02 Codex 设计会话（私有原始记录未提交）
- 输入：`0902-reviewed-v3.md`、`0902-02.md`、当前项目合同与真实代码
- 本设计不授权：实现、数据库迁移、正式 workspace 写入、删除或部署

## 结论

v4 只保留一句话：

> Core 不决定 Agent 怎样思考；Core 只保证一次被接受的交流有一个 Turn，每次实际运行只看一个 Root 和一份权限，完成时把该进入 Session 的 Message 整批原子写入。

Agent 怎样思考，由普通 `REACT` 插件决定。被动消息怎样进入系统，由普通 `MESSAGE_HANDLER` 插件决定。默认的 Prompt、上下文裁切、模型调用、Tool 选择和 `while` 循环全部属于 `default-react` 插件，不再属于 Core。

v4 的目标结构是：

```text
原始渠道输入
    │
    ▼
Channel Adapter ── 只管协议、去重、ACK
    │
    ▼
Incoming ── exact Root ── MESSAGE_HANDLER
    │ accept()
    ▼
Turn ── Run 1 ── interrupted
  │      Run 2 ── failed
  │      Run 3 ── REACT ── Draft Messages
  │
  ▼ complete()，一个 SessionDB 事务
Messages + seq + Turn outcome + SessionReceipt
    │                              │
    │                              └── 普通插件各自追 cursor
    ▼
Delivery Owner ── Channel Adapter ── 外部平台
```

最终只新增两个可替换的行为 Service：

1. `MESSAGE_HANDLER`：处理一次原始入站消息。
2. `REACT`：读取一个 Run，产生一组待提交 Message。

其余名字都必须拥有独立事实，不能只是换名包装：

- `Message`：一条有稳定身份的内容。
- `Turn`：用户能理解的一次完整交流。
- `Run`：Turn 内部一次实际执行，只是内部运行坐标。
- `Session`：已经提交的 Message 顺序。
- `Root`：一次运行看到的插件世界。
- `TurnGrant`：这个 Turn 允许触及的能力上界。

不再保留公共 `Attempt`、`AgentProgram`、`RunLock`、通用 `CommitPlan`、通用 `CommitIntent`、通用 `DerivedStore` 或“Session 本身就是另一条变更日志”这些概念。

---

## 一、这份 v4 要满足的真实要求

这些要求来自原始 Codex session，而不是从 v3 文案反推：

1. 先承认已经实现的 v3 插件底座，不把已有能力重新设计一遍。
2. 当前被动回复仍是硬编码特权链；目标是让它由普通插件组合出来。
3. AgentLoop 的本质要比 DeepSeek Harness 更简单，而不是再做一个更大的框架。
4. Tool、LLM、Prompt、Memory、Scheduler、Proactive 和来源插件都不能靠 Core 产品分支获得特权。
5. ToolSearch 自己必须是普通插件；“展示给模型”不能伪装成“授予权限”。
6. Akasha 已经是普通 Memory 插件，不得重新塞回 Core。
7. `U1 → interrupt → U2 → interrupt → U3 → A` 是一个 Turn，而不是三个用户交流。
8. Proactive、schedule fire、wake 和 spawn completion 在真正产生交流时，也能用同一个 Turn 模型。
9. 概念尽可能少，词尽可能普通；每个概念只能拥有一个变化轴。
10. 先做减法。旧设计没有证明价值的层、DTO、事件和兼容桥必须退出目标态。

### 判定标签

本文固定区分三类内容：

- **F（事实）**：当前代码、schema 或 accepted 文档已经证明。
- **T（目标）**：v4 推荐的最终设计，尚未实现。
- **U（未知）**：必须由维护者批准或在迁移前补证据。

T 不能写成当前事实，U 不能被实现者自行猜成需求。

---

## 二、当前真实起点

v4 不是从空白开始。

| 结论 | 状态 | 证据 |
|---|---|---|
| Root、Fiber、Effect、generation、stable/latest 和 snapshot lease 已存在 | F | `agent/plugins/snapshot.py:876`、`agent/plugins/snapshot.py:1567`、`agent/plugin_composition/effect.py:16` |
| 当前 Core 仍直接创建 `DefaultReasoner` 和 `PassiveTurnPipeline` | F | `agent/looping/core.py:335` |
| 当前被动链固定为 BeforeTurn → BeforeReasoning → Reasoner → AfterReasoning → AfterTurn | F | `agent/core/passive_turn.py:355` |
| ToolSearch 开关、Tool 展示顺序、Prompt phase 和 step phase 仍在 `DefaultReasoner` 内 | F | `agent/core/passive_turn.py:958` |
| completed 被动 Turn 已能把多条 user input 与最终 assistant 作为一批准备 | F | `agent/lifecycle/phases/after_reasoning.py:257` |
| `TurnCommitted` 目前仍同步 fanout；插件失败可以卡住后续 dispatch | F | `agent/lifecycle/phases/after_turn.py:243` |
| Session message 与 metadata 已在一个 SQLite 事务中追加 | F | `session/manager.py:642` |
| 当前 `message_id` 仍由 `session_key:seq` 生成 | F | `session/store.py:4744` |
| 当前 `turns` 表保存的是 execution attempt 进度，不是逻辑 Turn | F | `session/store.py:2392`、决策 0034 |
| completed interaction 已有整组删除、备份、embedding 删除和 compaction 失效流程 | F | `session/store.py:5385` |
| Akasha 已按普通插件边界拥有 Prompt、Tool 与自己的投影 | F | 决策 0041 |
| Core 尚无 `MESSAGE_HANDLER`、`REACT`、`TOOL_SELECTOR` 或 `SessionReceipt` | F | 当前代码符号盘点 |

### DeepSeek Harness 值得借，但不能照抄

DeepSeek Harness 做对了三件事：

1. 模型 adapter、Tool registry、Session log 和 Agent loop 都是插件。
2. 插件注册是可逆 effect，卸载会撤销自己拥有的注册。
3. profile 是插件树，默认 loop 可以被替换。

但它当前的具体 `agent-loop` 仍直接：

- 开关 Turn 和 Step；
- 组装 system prompt；
- 派生模型消息；
- 调 LLM；
- 执行 Tool；
- 追加 Session event。

这适合作为“默认 loop 也是插件”的证据，不适合作为 Akashic 的最终拆分。Akashic v4 再向前一步：Core 只给运行和提交端口，默认 `while` 循环整体搬进 `default-react`。

---

## 三、最小领域模型

### 3.1 Message：身份与顺序分开

```text
Message = (MessageId, role, content, source, placement)
placement = session(SessionId) | run-only
```

规则：

1. `MessageId` 是不透明、稳定、全局唯一的字符串。
2. `MessageId` 不从 `seq`、`TurnId`、ordinal、时间或正文计算。
3. 旧 `session_key:seq` 值继续作为合法的不透明旧 ID；迁移不改写它。
4. 新 Message 推荐使用 UUIDv7，但 UUIDv7 的时间位没有业务语义。
5. ordinal 只表示 Message 在 Turn 内的位置。
6. `seq` 只表示 Message 在 Session 内的位置。
7. 客户端同步继续只需要 `message_id + seq`；不得按正文或时间猜对应关系。
8. `run-only` Message 是一份真实、稳定的 Turn 输入，但不进入正式 Session，因此永远没有 seq。
9. 外部输入的 `source` 带经过 Channel 校验的稳定 source_ref；同一 source_ref 最多接纳成一个 MessageId。

`placement` 不是让插件临时挑选的提交计划。来源在 Turn 第一次接纳时就固定它：

- 普通 user/assistant Message 放进目标 Session；
- proactive 或 schedule 的内部 Task Message 可以是 `run-only`；
- proactive 最终 Assistant Message 放进目标 Session；
- PromptPart、stream delta、thinking、interrupt 和未封口 Tool call 不是 Message。

不可信 envelope、模型输出和 Handler 都不能自报 placement。Channel/Source adapter 只能在自己的受信任 admission API 中提出来源类型，ConversationStore 按该入口的固定规则写入 placement；REACT 返回的 DraftMessage 不带 SessionId，RunHost 只允许它进入 Turn 已固定的目标 Session。

因此下面三件事互不替代：

```text
MessageId                 谁
position(turn, message)   在这个 Turn 的第几个
seq(session, message)     在这个 Session 的第几个
```

`(TurnId, ordinal)` 可以定位关系，但不能取代 Message 身份。附件绑定、引用、重试、跨客户端同步和显式删除都需要稳定 MessageId。

### 3.2 Turn：一次完整交流

目标定义：

```text
Turn T = (
    turn_id,
    session_id,
    cause,
    grant,
    messages = <message_id_1, ..., message_id_n>,
    outcome
)

n >= 1
```

字段只各管一件事：

- `turn_id`：这次交流的稳定身份。
- `session_id`：它最终归入哪个 Session。
- `cause`：谁发起，如 user、schedule、wake、spawn 或 outbound call。
- `grant`：整次交流的能力上界。
- `messages`：属于这次交流的有序 MessageId。
- `outcome`：这次交流是否已经结束，以及怎样结束。

更精确地写，Turn 不是一份复制正文的新大对象：

```text
H_t = (turn_id, session_id, cause, grant)       # 一次写下的 header
M_t = <m | member(turn_id=t, message=m, ordinal)> # 有序 Message 引用
O_t = open | one terminal outcome               # 唯一结果

T_t = project(H_t, M_t, O_t)
```

存储只需要拥有 header、Message membership 和 outcome；`TurnView` 从它们投影，不复制 Message body、Session rows、Run facts 或 Delivery facts。这样 Turn 可以被查询和恢复，却不会成为第二份对话真相。

对已经进入 Session 的 Message，再定义：

```text
m_i ~ m_j  当且仅当  turn(m_i) = turn(m_j)
```

同一个等价类就是 Session 中可见的一个 Turn。组内按 seq 排，Turn 之间按每组最小 seq 排。由于一个 completed `session_batch(T)` 原子追加，同一 Turn 的可见 Message 必须形成连续块，中间不能夹进另一 Turn。

Turn 不拥有 Root，不拥有模型，不拥有当前运行 task，也不拥有 Delivery 状态。这些事实都可以在 Turn 不变时独立变化。

`TurnGrant` 保存的是稳定 capability key、scope 和限制，不保存 secret、连接对象或具体插件实例。真正的 credential handle、模型 adapter 和 Tool 实现由每个 Run 的 exact Root 冻结。授权流程可以在 grant 已允许的上界内批准一次具体调用；它不能在同一 Turn 中引入 grant 之外的新能力。

目标 outcome 只保留用户能理解的终态：

```text
open
completed(reply_message_id | no_reply)
superseded(by_turn_id)
abandoned(reason)
```

`interrupted` 和一次 provider error 不直接成为 Turn outcome，因为它们只结束一次 Run。显式重试仍可继续同一个 Turn。用户明确放弃，或失败后普通新输入选择 fresh，才关闭旧 Turn。

### 3.3 Run：内部运行坐标，不是第二个领域对象

```text
RunRef = (TurnId, run_seq)
run_seq = 1, 2, 3, ...
```

Run 只回答：“这个 Turn 第几次真的开始执行？”

它需要持久化，是因为中断、崩溃恢复、Tool 幂等和精确诊断都需要区分多次运行；但它不进入用户公共领域词，也不叫公共 `AttemptId`。

运行时可以给客户端一个短期 `interrupt_token`。它只授权中断当前 Run：

- Run 终结后 token 立即失效；
- token 不能作为 Turn、Message 或重试身份；
- UI 显示 TurnId，不显示内部 `run_seq`；
- Core 用 token 精确找到当前 Run，不需要创造另一个公共对象。

每个 Run 有自己的终态：

```text
succeeded(draft_ids)
interrupted(cause)
failed(error_code)
```

`succeeded` 也不等于 Turn 已完成。只有 ConversationStore 的原子提交成功，Turn 才能变成 `completed`。

一个 `open` Turn 不一定正在运行。它可以在某个 Run interrupted/failed 后安静等待下一次继续。UI 的 `running / interrupted / failed` 来自 `latest_run_state` 投影，不是假装成 Turn outcome；只有显式继续、替代或放弃才改变 Turn 本身。

### 3.4 Session：已提交 Message 的唯一顺序

```text
Session S = <(seq_1, message_id_1), ..., (seq_n, message_id_n)>

seq_1 < ... < seq_n
```

定义 Turn 对某个 Session 的投影：

```text
session_batch(T) =
    <m in T.messages | m.placement = session(T.session_id)>
```

Session 不是 `Commit/Delete` event 的 fold，也不是旁边一条日志的投影。权威内容仍是 `sessions.db/messages`。`session_batch(T)` 只是从 Turn 选择应进入该 Session 的 Message，不产生第二份正文。

规则：

1. 正常完成只 INSERT Message。
2. 同一 Session 的 `seq` 在提交事务内单调增加且不复用。
3. completed Turn 的整个 `session_batch(T)` 一次提交；不能出现半个可见批次。
4. Prompt 裁切、compaction、索引或插件重载无权改写正文。
5. 只有用户显式撤销或删除可以减少正文，并必须走单独的数据管理协议。

普通被动 Turn 的 batch 是 `U1...Un+A`；proactive/schedule 可以是 `A`，内部 Task Message 留在 Run/Turn 事实中。这样既保留“Message 组成 Turn”，也不把机器任务伪装成用户对话。

### 3.5 SessionReceipt：提交后的引用通知

普通插件需要知道 Session 新增或移除了什么，但不能因此重定义 Session。v4 增加一个很窄的引用 feed：

```text
SessionReceipt = (
    feed_seq,
    receipt_id,
    kind,                 # messages_appended | interaction_removed
    session_id,
    session_version,
    turn_id,
    message_ids,
    final_run_ref?,
    audit_ref?
)
```

它有五条硬规则：

1. 与非空 `session_batch(T)` 的提交或显式删除在同一个 SessionDB 事务中写入。
2. 只保存引用和版本，不复制正文、Prompt、Tool trace 或插件 payload。
3. 插件按 MessageId 从窄只读接口取自己需要的事实。
4. 每个插件在自己的 plugin-data 中拥有 cursor 和幂等 receipt。
5. Session 读取永远不通过 Receipt 反推正文；Receipt 丢失是损坏，不是“Session 为空”。
6. v4 首版不自动裁掉 Receipt；未来 retention 必须先有可证明的 snapshot/watermark 协议，不能猜所有插件都追上了。

它不是通用 EventBus，不负责插件执行，也不保证“所有插件都已经处理”。它只消除“Session 已提交但进程在通知插件前崩溃”的丢通知窗口。

### 3.6 六条代数不变量

```text
I1  MessageId 稳定；seq 与 ordinal 都不承担身份。

I2  一个 Message 最多属于一个 Turn；一个 completed Turn 的
    session_batch 在 Session 中保持同一顺序。

I3  同一 Turn 同时最多一个 active Run；同一 Session 同时最多一个
    active conversation Turn。

I4  一个 Run 只看一个 exact Root；同一 Run 的模型、Prompt contributor、
    Tool catalog 和执行端口不跨 generation 漂移。

I5  effective authority = TurnGrant ∩ Root capabilities ∩ call permit；
    每一层只能缩小，不能放大。

I6  Turn 只完成一次；非空 Session 批次、Turn outcome 和 SessionReceipt
    要么一起提交，要么都不提交。session_batch 为空时只提交 Run/Turn 终态，
    不伪造 SessionReceipt。
```

### 3.7 不存在 ε Turn

没有 Message，就没有 Turn。

- 一个被过滤的垃圾输入在 `Incoming.accept()` 前被忽略：没有 Turn。
- `/stop` 只是精确中断当前 Run：没有新 Message，也没有新 Turn。
- wake tick 只检查状态后决定不交流：它是 wake 插件自己的 tick，不是 Turn。
- wake 或 schedule 真正调用 `react` 时，先创建一条明确的 `run-only` Task Message，再创建 Turn。
- 一次维护 job 没有交流内容时，就是 job，不借 Turn 记账。

这让 Turn 永远保持“交流的最小单元”，不再兼任所有后台工作的统一盒子。

### 3.8 典型场景

#### 被动中断续接

```text
Turn T
├── Message U1
├── Run (T,1) ── interrupted
├── Message U2
├── Run (T,2) ── interrupted
├── Message U3
├── Run (T,3) ── succeeded
└── Message A ── complete ── Session batch [U1,U2,U3,A]
```

#### 显式 retry 与普通 fresh

```text
Run failed
├── retry(original MessageId) ── same Turn, new Run
└── ordinary new input        ── old Turn superseded, new Turn
```

#### Schedule

```text
Schedule tick ── private due/misfire rules
    ├── skip：只写 schedule 自己的状态
    └── run：run-only Task Message → Turn → REACT
                                      └── Assistant Message → target Session
```

#### Proactive

```text
observation → private gate
    ├── skip：不是 Turn
    └── speak：run-only Task Message → Turn → REACT
                                             └── Assistant Message → Delivery
```

来源只负责“何时创建 Message、把结果送到哪里”。一旦进入 `react`，不再有 passive、schedule 或 proactive 专用 Loop。

---

## 四、每个事实只有一个 owner

| Owner | 唯一拥有 | 明确不拥有 |
|---|---|---|
| Channel Adapter | 外部协议解析、envelope custody/重投/ACK、一次真实 provider 调用的协议映射 | Turn、Session、durable delivery 状态、Prompt、Tool 权限 |
| ConversationStore | 已接纳 source_ref 的唯一性、Message、Turn、内部 Run 记录、Session 顺序、SessionReceipt 的原子事务 | LLM、Tool 执行、渠道调用、插件 cursor |
| RunHost | session lane、active task、interrupt token、exact Root lease、短期资源清理 | Session 正文、长期插件状态、外部效果最终状态 |
| Plugin Runtime | artifact、Root、Fiber、Effect、generation、stable/latest、发布与 drain | Turn outcome、Delivery outcome、插件业务数据 |
| `default-react` | 默认 Prompt/Context/Tool/LLM `while` 算法和自己的 step 状态 | Session commit、Delivery、权限授予、Root 发布 |
| Tool plugin | Tool schema、参数边界、执行、自己的外部效果 ledger 和幂等 | Turn 总控制、其他 Tool、Session 任意写入 |
| Delivery Owner | durable envelope、dedupe、provider receipt、uncertain/rejected/delivered/settled 状态 | 渠道协议实现、Session 正文所有权 |
| Memory/Projection plugin | 自己的索引、cursor、receipt 和 rebuild | 原始 Message 保留、Turn commit、其他插件 readiness |

### Channel 与 Delivery 的精确分工

二者不能都声称“拥有投递”。

```text
Delivery Owner                    Channel Adapter
──────────────────────────────    ─────────────────────────
创建 durable envelope             把 envelope 映射成平台 API
决定 dedupe/idempotency           执行一次真实调用
记录 provider_started             返回 provider 原始 receipt
记录 uncertain/rejected/success   不保存长期状态机
恢复未完成操作                     不自行猜是否该重试
```

外部调用的 durable 真相只在 Delivery Owner。Channel Adapter 是协议 driver。

### live 资源与 durable 外部效果不能共用一个领域词

Root/Fiber/Effect 只管理本进程短期资源：listener、timer、连接、临时 task、注册项。它们随 Root 卸载逆序清理。

已经可能发生的外部操作由真实领域 owner 管理：

- 消息发送属于 Delivery Owner；
- Tool 写操作属于该 Tool 插件；
- schedule 状态属于 Scheduler 插件；
- Memory 投影属于 Memory 插件。

旧 Root 不需要为了一个长期 pending delivery 永远活着。插件必须先把 durable operation 交给稳定 owner，才能释放 Run lease。

---

## 五、最小公共 API

以下代码只说明可观察合同，不是最终 Python 语法。

### 5.1 一个来源无关的 Turn 入口

Channel、Scheduler、Proactive 和 Spawn 都需要同一扇窄门，但它不是另一套 Loop：

```python
class TurnPort(Protocol):
    async def start(self, request: TurnStart) -> Run: ...

class TurnStart:
    session_id: SessionId
    message: NewMessage
    placement: "session" | "run-only"
    cause_ref: CauseRef
```

调用者不提供 TurnId、RunRef、TurnGrant、Root、模型或 Tool。TurnPort 根据已认证来源、调用插件 scope、目标 Session 和部署 policy 生成这些事实，并在 durable admission 成功后才返回 Run。

它是 Core 的原子能力，不是行为策略：

- `Incoming.accept()` 是一个已经填好并限制为最多调用一次的 TurnPort 请求；
- Scheduler/Wake 用 scoped TurnPort 接纳 `run-only` Task Message；
- Subagent 用自己的 Session 和父子 cause_ref；
- 来源不能借它取得任意 Session repository 或扩大 grant。

### 5.2 两个行为 Service

```python
MESSAGE_HANDLER: ServiceKey[MessageHandler]
REACT: ServiceKey[React]

class MessageHandler(Protocol):
    async def handle(self, incoming: Incoming) -> None: ...

class React(Protocol):
    async def react(self, run: ReactRun) -> DraftBatch: ...
```

没有 `AGENT_PROGRAM`。`react` 已经是项目接受的普通动词：输入 Message，产生输出 Message。再增加一个 Program 只会建立第二套执行模型。

### 5.3 `Incoming` 是一次性 capability

```python
class Incoming:
    view: IncomingView

    async def accept(self) -> Run: ...
```

规则：

1. `view` 是经过 Channel 边界校验的只读数据。
2. `accept()` 最多成功一次。
3. session、source、cause、lane key 和最大 TurnGrant 在创建 Incoming 时已经由 host 固定；Handler 不能扩权或换 Session。
4. Handler 返回时从未调用 `accept()`，表示忽略。
5. Handler 调用 `accept()` 后，必须让 Run 到达内部 terminal；否则 Host 记录合同违反并让 Run failed，不静默完成。
6. 不存在 `decide() → plan payload → handle(plan)` 两段协议。

`accept()` 在同一个 admission 事务里固定 source_ref → MessageId → TurnId → RunRef。Channel 因 ACK 丢失重投同一 envelope 时，ConversationStore 返回同一接纳结果，不增加第二个 Message；Channel Adapter 仍独自拥有外部 custody 和何时 ACK 的协议状态。

Dispatcher 在调用 Handler 前先租用 exact Root。`accept()` 把同一份 lease 转成 Run lease，因此 Handler 与后续 `REACT` 不会跨 generation。

`/stop` 不进入 Handler。Channel Adapter 在协议边界识别它，使用当前 `interrupt_token` 调 RunHost；它不创建 Message，也不触发 Prompt 或 Memory hook。

默认 passive handler 本身也是普通插件，它可以直接写成：

```python
async def handle(incoming: Incoming) -> None:
    if await inbound_filters.ignore(incoming.view):
        return

    run = await incoming.accept()
    command_reply = await commands.try_handle(run, run.current_input)
    if command_reply.handled:
        await run.complete(command_reply.drafts)
        return

    drafts = await run.react()
    await run.complete(drafts)
```

`inbound_filters` 和 `commands` 可以是这个插件依赖的普通 registry；Core 不知道有哪些 filter 或 command。顺序由默认 Handler 明确拥有，不能再散成无 owner 的 waterfall。替换整个 Handler 就能替换这套入口策略。

### 5.4 `Run` 只暴露必要端口

```python
class Run:
    ref: RunRef
    turn: TurnView
    current_input: MessageView
    runtime_history_view: RuntimeHistoryView
    signal: CancelSignal
    stream: StreamPort
    resources: RunResources

    def bind_model(self) -> ModelExecution: ...
    def tool_port(self) -> ToolPort: ...
    async def react(self) -> DraftBatch: ...
    async def complete(self, drafts: DraftBatch) -> CompletedTurn: ...
```

这些字段都有独立 owner：

- `turn` 和 `runtime_history_view` 来自 ConversationStore 的冻结只读视图；
- `bind_model()` 第一次调用时从 exact Root 内普通 models 插件冻结一份 `ModelExecution`，重复调用返回同一份；
- `tool_port()` 只暴露 `TurnGrant ∩ exact Root`，不返回全局 registry；
- `stream` 只发 live delta，不建立持久真相；
- `resources` 只登记必须在本 Run 结束前关闭的短期资源；
- `signal` 只表示当前 Run 是否还活着。

Run 明确没有：

- `history` 这种无修饰名字；
- 任意 `metadata` 或 `extra` 袋子；
- 任意 SQL 或全功能 Session repository；
- `loop_input`；
- 通用 `commit_intents`；
- 任意 Root lookup；
- 改 TurnGrant 的方法。

`run.react()` 只调用本 Run exact Root 中的 `REACT`。`run.complete()` 只把经过结构校验的 DraftBatch 交给 ConversationStore；REACT 插件不能自己写 Session。

模型和 Tool 端口按使用绑定，缺失时在第一次调用处 fail-loud，而不是在 Run admission 时伪造空实现。于是 echo Handler 可以完全不安装 LLM/Tool；一个要用模型的 REACT 则必须声明并验证自己的 models 依赖。

### 5.5 `DraftBatch`

```python
DraftBatch = tuple[DraftMessage, ...]
```

对 `REACT` 的返回：

1. 至少有一条输出 Message。
2. 恰好一条 terminal assistant。
3. 每个 DraftMessage 在交给外部 effect 前由 RunHost 获得稳定 MessageId。
4. Draft 不是 Session 事实；只有 `complete()` 事务成功后才进入 Session。
5. stream delta、thinking 和未封口 Tool 调用都不是 DraftMessage。
6. DraftMessage 不能指定 SessionId、placement、TurnId 或 TurnGrant。

命令可以由 Handler 直接构造 DraftBatch 后 `complete()`，不必调用 `react`。如果一个已经有输入 Message 的命令明确选择不回复，Handler 可以调用 `complete(())`，Turn 结束为 `completed(no_reply)`；空输入加空输出仍被拒绝。这不需要 Core 的 `Skip`、`Enter` 或 `Return` 控制对象。

### 5.6 Run 内部状态机

```text
created
  │
  ├── start ── active ── react succeeded ── completing ── succeeded
  │                    ├── interrupt ───────────────────── interrupted
  │                    └── error ───────────────────────── failed
  └── admission error ──────────────────────────────────── failed
```

ConversationStore 只有在 `completing` 时接受一次 complete。重复 complete 必须按同一 draft identity 幂等；内容漂移 fail-loud。

### 5.7 Session lane 是 Host 细节

同一 Session 的 conversation Turn 串行，不同 Session 可并发。用于排队的 key 是稳定服务 namespace 加 SessionId，例如：

```text
("conversation", SessionId)
```

它不能包含 generation、provider、模型或 Handler 实例身份。否则热换后旧、新 Root 会各拿一把锁，导致同一 Session 意外并发。

`RunLock` 不进入公共 API。插件只看见 `accept()` 成功、busy 或取消，不持有锁对象。

---

## 六、`default-react` 普通插件

### 6.1 它完整拥有默认算法

```text
runtime history view
       │
       ▼
CONTEXT_VIEW ── prompt history
       │
PROMPT_PARTS ── immutable prompt parts
       │
TOOL_SELECTOR ─ visible tool names
       │
       ▼
build one frozen model request
       │
       ▼
Model stream
       │
       ├── no tool call ── terminal DraftBatch
       │
       └── tool calls ── ToolPort ── append run-local closed facts
                                      │
                                      └── next model call
```

伪代码：

```python
async def react(run: ReactRun) -> DraftBatch:
    local = DefaultReactState()
    model = run.bind_model()
    tools = run.tool_port()

    while True:
        prompt_history = context_view.project(
            runtime_history_view=run.runtime_history_view,
            closed_run_facts=local.closed_facts,
        )
        prompt_parts = await prompt_parts_registry.collect(run, local)
        tool_view = await tool_selector.select(run, local)
        request = freeze_request(
            model=model,
            prompt_parts=prompt_parts,
            prompt_history=prompt_history,
            tools=tools.schemas(tool_view.names),
        )
        reply = await model.stream(request, run.stream, run.signal)

        if not reply.tool_calls:
            return seal_terminal_reply(reply)

        results = await tools.execute_visible(
            calls=reply.tool_calls,
            visible=tool_view,
            signal=run.signal,
        )
        local.append_closed(reply, results)
```

Step 是这个算法内部的循环计数，不进入 Core 领域模型。另一个 REACT 插件可以一次模型调用就结束、完全不用 Tool，或使用 plan/execute；Core 都不需要新增分支。

### 6.2 Prompt 是不可变贡献

`PROMPT_PARTS` 是 `default-react` 依赖的普通 registry，不是 Core 的第三条控制轴。

每个贡献是：

```text
PromptPart = (key, kind, order, content, source_ref, trust)
kind = instruction | context
```

规则：

1. 插件只能返回自己的新 PromptPart，不能取得共享可变 Prompt。
2. 一次模型调用收集完后整体冻结。
3. 排序键只保证重放稳定，不证明两个 instruction 在语义上可交换。
4. 重复 key、互斥 slot、越过字节预算和非法 trust 提升在 candidate Gate fail-loud。
5. 当前 user Message 永远独立，不被 context 插件改写。
6. Memory、skill 和检索内容必须带来源与 trust，不能伪装成用户原话。

稳定排序可以使用：

```text
(kind_order, order, plugin_id, key)
```

但 candidate profile 必须额外证明语义兼容。不能用“排序是确定的”替代“组合是合理的”。

### 6.3 `CONTEXT_VIEW` 只产生 prompt history

输入与输出必须写全名：

```text
persistent history     Session 中完整已提交正文
runtime history view   当前 Run 可读取的冻结工作视图
prompt history         本次模型调用真正使用的投影
```

`CONTEXT_VIEW` 只能：

- 从 runtime history view 选择完整逻辑单元；
- 做通用 token 预算与 compaction；
- 保留当前 user anchor、成对 Tool 事实与外部效果证据；
- 返回带来源的 prompt history。

它不能：

- UPDATE/DELETE Session Message；
- 改当前输入；
- 改 Prompt instruction；
- 改 ToolGrant 或 ToolView；
- 改模型绑定；
- 用空列表掩盖损坏数据。

Akasha 继续是普通 PromptPart/Tool/SessionReceipt consumer，不成为 CONTEXT_VIEW 内部分支。

### 6.4 ToolGrant 与 ToolView 完全分开

```text
TurnGrant                    ToolView
─────────────────────────    ─────────────────────────
安全能力上界                 本次模型调用展示哪些 schema
Turn 建立时冻结              每次模型调用可变化
只能被交集缩小               可以搜索、排序、分页
由 admission/policy 拥有     由 TOOL_SELECTOR 拥有
执行时必须检查               只影响模型可发现性
```

真正执行一项 Tool 必须同时满足：

```text
tool ∈ TurnGrant
tool ∈ exact Root catalog
call permit accepts arguments
```

Tool 是否显示不能授予权限。Tool 没显示但模型凭名字猜到时，`default-react` 返回明确 `tool_not_visible`；这只是模型协议错误，不是安全拒绝。Tool 不在 grant 时返回 `tool_not_allowed`，两者不能混成一句字符串。

ToolSelector 收到的候选集已经是 `TurnGrant ∩ exact Root catalog`。它不能枚举、搜索或泄露 grant 之外的 Tool metadata；`tool_search` 本身也必须同时在 grant 和初始 ToolView 中，profile 缺少这条闭环时在加载期失败。

### 6.5 ToolSearch 是完整普通插件

`tool-search` 插件拥有：

1. `TOOL_SELECTOR` provider；
2. `tool_search` 这个普通 Tool；
3. 搜索索引与排序；
4. 当前 Run 的 selection state；
5. provider schema 上限下的展示策略。

它不拥有：

- ToolGrant；
- Tool registry；
- Tool 执行；
- Agent 重启授权；
- Core metadata flag；
- 跨 Run 的隐式已解锁集合。

`tool_search` 返回选择结果并改变该插件自己的 Run-local selection state。新 Tool 只在下一次模型调用进入 ToolView。卸载它后，可以换成一个 `show-all-selector` 或 `fixed-selector`，不改 Core 和 REACT 接口。

目标态删除当前这些耦合：

- `tool_search_enabled`；
- `always_on/preloadable/requires_turn_search` 驱动的 Core 分支；
- Turn ContextVar 中的 search grant；
- ToolSearch 与 `agent_restart` 授权绑定；
- Reasoner 内部 provider schema 挤压特判。

必要的“默认总要展示哪些 Tool”由 selector 配置表达，不再成为 Tool 的安全属性。

### 6.6 三种 retry 不能混在一起

| retry | 身份 | owner |
|---|---|---|
| 同一 HTTP/model call 的网络重试 | 同一 Run、同一模型调用 | Provider plugin |
| context overflow 后重新投影并再次请求 | 同一 Run、新 model call | `default-react` + `CONTEXT_VIEW` |
| failed Run 后用户显式 retry | 同一 Turn、新 Run | ConversationStore/入口协议 |

任何 retry 都不能恢复隐藏思维，也不能重放结果不明的外部 Tool effect。

### 6.7 不强制永久保存完整模型请求

每次模型调用必须留下足以问责的 binding receipt：

```text
RunRef + call_seq
Root logical digest + root instance token
ModelExecution identity
PromptPart refs/digests
prompt history source refs
ToolView names/schema digests
compaction generation
provider outcome/usage/request id
```

完整 Prompt 和完整 provider payload 只在显式诊断开关下保存，并且必须有容量、保留期、权限和脱敏规则。它不是 v4 的永久事实要求。

Root 使用两个身份：

- logical digest：证明插件拓扑和 catalog 内容相同；
- instance token：区分两个物理 Root 实例，避免把重建后的对象当成原对象。

---

## 七、提交、恢复与外部效果

### 7.1 被动 Turn 的正常完成

```text
1. Incoming.accept
   └── durable Turn/input facts + RunRef；尚未进入 Session

2. REACT
   └── run-local model/tool facts；Session 正文不变

3. Delivery Owner 可先 prepare 一个不执行外部调用的 envelope

4. Run.complete，单个 SessionDB 事务
   ├── 验证 Turn open 且 Run 是当前运行
   ├── 给 session_batch(T) 分配连续 seq
   ├── INSERT session_batch(T) messages
   ├── 写 Turn completed(reply_message_id | no_reply)
   ├── 写 Run succeeded
   ├── 更新 session metadata/head
   └── session_batch 非空时 INSERT SessionReceipt

5. 事务提交后发布 Receipt wakeup

6. Delivery Owner 根据 prepared envelope / Receipt 发送
```

第 5 步只是唤醒优化。进程在第 4、5 步之间崩溃，插件和 Delivery Owner 仍能按 `feed_seq` 追上。

如果第 4 步失败：

- Session 不出现任何本 Turn Message；
- Turn 不变成 completed；
- Draft MessageId 仍只属于 pending/run 事实；
- 外部发送不能开始。

pending 正文与 Session 正文不能长期成为两个 owner。目标存储在同一事务中先校验并 INSERT `session_batch(T)`，再把对应 pending slot 收敛为 `MessageId + digest` 引用；事务失败时 pending 原文保持不变。`run-only` Task Message 不做这次 handoff，它的正文与 retention 继续由创建它的来源/Turn 事实 owner 管理。具体表结构与旧数据处理必须在持久化 ADR 中批准。

### 7.2 中断或失败

中断只原子结束当前 Run：

```text
active Run → interrupted
Turn        → open
Session     → unchanged
```

失败同理。下一条输入如何处理由已接受规则决定：

- interrupted 后普通输入：同 Turn、新 Message、新 Run；
- retryable failed 后显式 retry：同 Turn、复用原 Message、新 Run；
- failed 后普通 fresh：旧 Turn `superseded`，新 Turn。

Run terminal 必须先落 durable control fact，再释放 Root lease。不能先把内存指针清空后假装已经终结。

每条 active Run 还记录创建它的 runtime boot identity。进程重启时，ConversationStore 在取得唯一恢复 owner 后，把不存在 live owner 的 active Run 原子终结为 `interrupted(process_lost)`，保持 Turn open，再释放 session lane。它绝不自动重跑模型、Tool 或 Delivery；这些领域分别按自己的 receipt 对账。

### 7.3 Tool 外部效果

不建立通用 `CommitIntent`。

一个可能写外部世界的 Tool 自己保存：

```text
(turn_id, run_seq, call_id, operation_key, state, domain_receipt)
```

规则：

1. Tool owner 在调用前固定 operation key。
2. started/succeeded/rejected/uncertain 由 Tool 自己的 durable store 表达。
3. Run 重试先查这个 owner 的 receipt，不盲目重放。
4. Turn 完成后，该 Tool 如需消费 SessionReceipt，使用自己的 cursor。
5. Core 不理解支付、发信、文件写入、重启或日历事件 payload。

这既避免外部效果丢失，也避免 Core 变成第二个工作流引擎。

### 7.4 SessionReceipt consumer

```text
SessionReceipt feed
   ├── Akasha cursor/data
   ├── Markdown memory cursor/data
   ├── FTS/embedding cursor/data
   └── UI/notification cursor/data
```

每个 consumer 独立：

- 自己决定是否适用；
- 自己保存 cursor；
- 自己保证同 receipt 幂等；
- 自己报告 lag/failure；
- 一个 consumer 失败不回滚 Session，也不阻止其他 consumer。

新装或升级后的 consumer 不从“第一条 Receipt”盲扫全历史。它使用 ConversationStore 的窄只读操作在一个一致性读点取得：

```text
canonical Message snapshot + current feed_seq watermark
```

插件先从 snapshot 建自己的投影并原子保存 cursor=watermark，再消费更大的 feed_seq。candidate 只能在隔离数据上演练这套过程；正式 promotion 后的新 Fiber 才继续 production cursor。这样插件是否安装、何时更新，与 Session 写入保持正交。

因此删除同步 `TurnCommitted` 大 fanout 后，Memory 故障不会卡住用户回复的外部 dispatch。

### 7.5 两种 Delivery 顺序

#### 被动回复与 Akashic/Web/Mobile 主动消息：Session first

```text
prepare envelope → complete Session → send/notify → settle delivery
```

发送失败不回滚 Session。Web/Mobile 以 Session head 和连续最大 seq 恢复，通知只负责更快看到更新。

#### 外部渠道 proactive：provider first

```text
prepare envelope → provider call → durable provider receipt
                 → idempotent Session append → settle delivery
```

只有完整 provider success 才追加 Session。部分成功或结果不明停在 Delivery Owner 的结构化状态，不能靠返回文案猜成功，也不能自动重复发送。

这两种顺序属于 Delivery 领域的明确分支，不升级成通用 `CommitPlan` 平台。

### 7.6 显式删除

删除不是普通 Session 生命周期事件。它必须继续满足：

1. 用户主动发起名称明确的撤销/删除操作；
2. 精确到 completed Turn/interaction；
3. 执行前 SQLite online backup 与 `integrity_check`；
4. 在 session lane 和 compaction fence 下串行；
5. messages、embedding、compaction invalidation、audit 与 `interaction_removed` Receipt 原子提交；
6. Message seq 不复用；
7. 派生插件失败时不能继续提供撤销前的陈旧结果。

目标态 completed Turn/Run 只保留被删 Message 的 identity、digest 和非正文终态，不保留第二份正文。迁移前必须额外盘点当前 `turns.items/input`、tool trace、delivery envelope 和诊断文件中是否复制了内容；是否随“撤销对话”一起减少要逐类写进数据管理合同，不能由 v4 文档猜测。

`SessionReceipt` 只能通知启用中的 consumer。永久删除某插件的 opaque plugin-data 需要单独的 Data Governance inventory：列出 owner、数据位置、备份、清除动作与证明。v4 不声称一个通用 receipt 能自动发现和清掉所有未来插件数据。

---

## 八、Root、热更新与自我更新

### 8.1 一个 Run 一个 exact Root

Root 在 Handler 开始前选择，并保持到 Run terminal 与短期资源 cleanup 完成。

同一个 Turn 的下一 Run 可以使用更新后的 committed Root：

```text
Turn T
├── Run 1 → Root G10 → interrupted
└── Run 2 → Root G11 → completed
```

TurnGrant 不随 Root 更新而扩大。Run 2 的有效能力仍是：

```text
TurnGrant(T) ∩ capabilities(Root G11)
```

如果新 Root 缺少 Turn 继续所需的 Service，admission fail-loud；不能偷偷回退到全局单例或旧插件。

### 8.2 candidate 只验证变化面

沿用当前增量 candidate 原则：

1. 以 stable snapshot 为基线。
2. 从变化插件和它真实依赖的 provider closure 建 candidate Root。
3. 未变化插件不重启，不复制正式数据库，不运行正式 listener。
4. candidate 使用只读或隔离的 plugin-data、临时端口和 staged event bus。
5. 重复 Service、缺依赖、catalog 冲突、manifest 不一致在 latest-ready 前失败。

不同变化面跑不同 probe：

| 变化面 | 最小真实 probe |
|---|---|
| `MESSAGE_HANDLER` | 原始 envelope → accept/ignore/command；无模型假成功 |
| `REACT` | 固定 Message → 真实或可重放 model/tool trace → DraftBatch |
| PromptPart | 最终 part refs、顺序、digest、预算和冲突 |
| Tool/Selector | catalog、grant/view 分离、真实调用、参数拒绝 |
| Delivery | prepared/provider_started/receipt/uncertain 恢复 |
| Memory consumer | SessionReceipt replay、cursor、幂等、删除屏障 |

“能 import”“manifest healthy”或“plugin-doctor 通过”都不能替代行为 probe。

### 8.3 自我安装与 promotion

推荐流程：

```text
owner Turn
  │
  ├── staged install → immutable artifact → latest-ready
  ├── programmatic child 用 exact latest 做真实 probe
  ├── owner Turn 回复并完成 Session/Delivery
  └── 释放自己的 stable Root lease
          │
          ▼
runtime-owned promotion operation
  ├── fence 新 admission
  ├── drain 旧 Run leases
  ├── 停旧 formal Root
  ├── 启新 formal Root
  ├── commit stable pointer/snapshot
  └── 失败则按 journal 重建旧 stable Root
```

不能在安装自己的 Turn 内同步等待旧 Root drain；该 Turn 自己正持有旧 lease，会形成环形等待。

### 8.4 drain 与 cleanup

1. 指针切换后旧 Root 不接新 Run。
2. 旧 Run 继续使用旧 Root，直到内部 terminal。
3. Run 内短期资源全部逆序 cleanup 后才释放 lease。
4. cleanup 失败时 Root 保持 `cleanup_failed/degraded` drain blocker，并保留全部错误。
5. 不能像当前 `Effect._close()` 一样在 cleanup 报错后仍把 owner 移除并标为 closed。
6. durable external operation 已交给领域 owner 后，不再阻塞旧 Root；没有 handoff 的 operation 必须阻塞。

### 8.5 Python 插件仍是可信代码

Root 隔离解决生命周期、状态污染和发布一致性，不是恶意代码沙箱。被安装的 Python 插件仍可执行宿主权限下的代码。

因此自我安装还必须保持：

- 明确来源与 source revision/tree digest；
- immutable artifact；
- 用户或既有授权链；
- candidate 不接正式 secret/write port；
- 安装失败可恢复；
- 正式 promotion 有 journal 和旧代重建证据。

---

## 九、从 v3 和 `0902-02` 再做一次减法

| 旧概念或提议 | v4 处理 |
|---|---|
| 公共 `Attempt` / `AttemptId` | 删除；保留内部 `RunRef=(TurnId, run_seq)` 和短期 interrupt token |
| `AGENT_LOOP` | 改为普通 `REACT`；Loop 只是默认插件的一种实现 |
| `AgentProgram` / `ProgramOutput` | 删除；与 `react` 重复 |
| `MESSAGE_HANDLER.decide()+handle()` | 压成一次 `handle(Incoming)`；`accept()` 是一次性 capability |
| public `RunLock` | 删除；session lane 由 RunHost 内部拥有 |
| `TurnRules` | 收敛为 immutable `TurnGrant`；不放 model、delivery、memory switch |
| `AgentRun` 大袋子 | 改成固定窄端口；删除 extra、metadata、loop_input、任意 repository |
| Core `TOOL_VIEW` 第三选择轴 | 降为 `default-react` 依赖的普通 `TOOL_SELECTOR` |
| `(TurnId, ordinal)` 取代 MessageId | 拒绝；它只表示成员位置 |
| Session = `Commit/Delete(turn)` fold | 拒绝；Session 仍是 canonical Message rows + seq |
| 通用 `SessionChange` | 收窄为同事务 `SessionReceipt` 引用 feed |
| `DERIVED_STORES` registry | 删除；每个插件拥有自己的 cursor/data/rebuild |
| 通用 `CommitIntent` / evidence bag | 删除；Tool/Delivery/Memory 各自保存领域 pending record |
| 强制永久 `FinalModelRequest` | 删除；只强制 binding receipt，完整 payload 是有界诊断 |
| Fiber Effect 与外部效果同名 | 分开；Effect 只管 Root 资源，外部效果归领域 owner |
| ChannelHost 与 DeliveryHost 都拥有 delivery | 修正；Delivery Owner 持久化，Channel Adapter 执行协议调用 |
| 通用 phase/EventBus 主链 | 迁到具体 Service/registry/Receipt；只保留有独立领域意义的事件 |
| `post_commit` 业务开关袋 | 用来源明确的 Session/Delivery 流程替代；迁移期兼容，目标态删除 |
| passive/schedule/proactive 各有 loop | 来源只创建 Message/Turn；共用 `REACT` |

### 现有生命周期接线的归宿

旧 phase 不能整包换名后继续存在。逐项归位：

| 当前环节 | v4 owner |
|---|---|
| BeforeTurn 的 Session/load/lane | TurnPort + ConversationStore + RunHost |
| 入站过滤、命令与是否进入回复 | 普通 `MESSAGE_HANDLER` 插件 |
| BeforeReasoning 的模型输入准备 | `default-react` 的 PromptParts、ContextView、ToolSelector |
| PromptRender mutable frame | 删除；改成 immutable PromptPart 收集与冻结 |
| BeforeStep/AfterStep | `default-react` 私有循环；有独立领域意义的 Tool/Model 事件留在各自 owner |
| 模型回复 parse/seal | Provider adapter + `default-react` 输出校验 |
| assistant attachment import | Attachment owner 在 complete 前返回 immutable refs；ConversationStore 同事务绑定 |
| AfterReasoning 的 Session 写入 | ConversationStore 的唯一 `complete()` |
| TurnCommitted 同步 fanout | 删除；换成 durable SessionReceipt + wakeup |
| Memory、embedding、presence | 各自的 SessionReceipt consumer |
| AfterTurn dispatch | Delivery Owner；Channel Adapter 只执行协议调用 |
| budget/log/trace | 诊断 owner，不能阻塞 Session 与 Delivery |
| success/failure extras | 回到产生该事实的领域插件；无独立 owner 的删除 |

迁移原则不是“每个 phase 变成一个新 Service”。只有跨插件需要替换、并且拥有独立规则的能力才成为 Service；其余逻辑内联到真实 owner。

### 被动链特殊规则的最终 owner

| 当前特殊规则 | v4 归宿 |
|---|---|
| command catalog 短路 | default Handler 依赖的 ordinary command registry |
| compaction 水位、完整 Turn 切点、overflow 重试 | `CONTEXT_VIEW`/compaction 插件；ModelPort 只执行已冻结请求 |
| Tool schema 预载、数量上限、`tool_search` 解锁 | `TOOL_SELECTOR` 与 tool-search 插件 |
| 空回复 retry | `default-react` 的模型输出规则；不是 Core retry |
| terminal Tool deadline | Tool 声明终结语义，`default-react` 拥有循环 deadline |
| provider continuation | Provider adapter 给出 typed finish，`default-react` 决定是否再调一次 |
| attachment/media import | Attachment owner；Message 只绑定 immutable ref |
| error response 文案 | default Handler 的 source-facing error mapper；RunHost 只返回 typed error |
| Citation | Citation 插件的 PromptPart/Tool/Receipt consumer，不进 REACT 分支 |
| Meme | Meme 插件的 Tool/PromptPart/独立数据 owner，不进 REACT 分支 |
| Akasha/Markdown post-memory | 各自的 SessionReceipt consumer |
| `agent_restart` | runtime-operation Tool + TurnGrant；与 ToolSearch 展示完全分开 |
| delivery、partial、uncertain、ACK | Delivery Owner 与 Channel Adapter 的明确分工 |
| stream/thinking/status | live StreamPort/UI projection；不能反写 Session 正文 |

这张表必须在迁移中逐行对账。某项旧逻辑没有新 owner 时不能删；某项已经有 owner 时不能再保留旧 phase 的第二个 writer。

### 目标态 Core 中不应出现的产品名字

```text
Akasha
ToolSearch
Scheduler
Proactive
Wake
Spawn profile
Telegram / QQ / Mobile
OpenAI / Anthropic / DeepSeek
Markdown memory
Meme / Citation
```

它们可以作为普通插件、Provider 或 Channel Adapter 存在，但 Core 不按这些名字分支。

---

## 十、目标架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                           Akashic Core                              │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌─────────────────────┐  │
│  │ Plugin Runtime │  │ConversationStore│  │      RunHost        │  │
│  │ Root/Fiber     │  │Message/Turn     │  │lane/cancel/lease    │  │
│  │ Effect/gens    │  │Session/Receipt  │  │narrow live ports    │  │
│  └────────────────┘  └────────────────┘  └─────────────────────┘  │
│                                                                     │
│  Core 只查两个行为 key：MESSAGE_HANDLER、REACT                       │
└─────────────────────────────────────────────────────────────────────┘
                 │ exact Root                         │ receipts
                 ▼                                    ▼
┌────────────────────────────────┐      ┌─────────────────────────────┐
│      ordinary behavior plugins │      │ ordinary state plugins      │
│                                │      │                             │
│ passive-message-handler        │      │ Akasha                      │
│ default-react                  │      │ Markdown memory             │
│ simple-react / no-tool-react   │      │ Embedding / FTS             │
│ prompt-parts                   │      │ UI projection               │
│ context-view                   │      │                             │
│ tool-search-selector           │      │ each owns cursor + data     │
└────────────────────────────────┘      └─────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ ordinary source/domain plugins                                     │
│ Channel · Scheduler · Proactive · Spawn · Delivery · Tools · Models│
└─────────────────────────────────────────────────────────────────────┘
```

### 最小启动证明

Core 在没有默认插件时也应能启动：

```text
Plugin Runtime + ConversationStore + RunHost
```

此时收到普通消息应明确返回 `MESSAGE_HANDLER unavailable`，而不是偷偷走旧被动链。

装一个十几行的 `echo-message-handler` 后，可以完成：

```text
Incoming → Turn → "echo" DraftMessage → Session
```

它不需要 LLM、Tool、Prompt、Memory 或 `default-react`。这个 Gate 是“被动能力真的不再特权”的最强证明。

---

## 十一、迁移路线

这不是一次大改。每阶段都必须可回滚，并先保留当前行为 oracle。

### Phase 0：批准语义变化，冻结证据

先做：

1. 把本文与 accepted 决策的差异列成正式 ADR：公共 Attempt 降为内部 Run、MessageId 与 seq 解耦、一个 Run 一个 Root。
2. 录制当前被动、interrupt/retry、ToolSearch、Memory、Delivery、插件自更新 fixtures。
3. 保存真实 SessionDB、tool trace、delivery ledger、snapshot identity 和 client payload 作为 oracle。
4. 不改正式 workspace。

回滚点：无代码和数据变化。

### Phase 1：先扶正身份与原子提交

目标：只改 ConversationStore，不改现有 Reasoner 行为。

1. 增加逻辑 Turn 与内部 Run 的明确存储表示。
2. 新 MessageId 与 seq 解耦；旧 ID 原样保留。
3. 增加 `SessionReceipt` 表和 global `feed_seq`。
4. 把 completed batch、Turn outcome、Run terminal、session head 和 Receipt 收进同一事务。
5. 现有 API 用 adapter 继续返回旧字段，禁止双写漂移。
6. Realtime/Mobile 同时引入稳定 TurnId 与短期 interrupt token；旧“`turn_id` 实际指一次 attempt”的字段只在客户端迁移期保留，不能继续污染新存储。

旧 Message 只有在已有 `control_turn_id` 等明确证据时才能建立新 Turn membership。缺少身份的历史行继续由版本化 legacy reader 读取，不能按角色邻接、正文、时间或 seq 距离猜 Turn。历史 backfill 算法要作为单独的数据迁移合同批准。

持久化迁移前必须：完整 SQLite backup、`integrity_check`、row count/digest、隔离恢复 smoke。不得直接复用当前 `turns` 表名改变旧行语义；推荐新表承接 v4，旧表只读保留到 parity Gate 通过。

回滚点：旧读路径仍是 authority；新表/Receipt 只是 shadow，不能驱动外部效果。

### Phase 2：把当前算法整体包进 `REACT`

1. 新建普通 `default-react` 插件。
2. 第一版内部仍调用现有 `DefaultReasoner`，只建立接口边界。
3. `PassiveTurnPipeline` 通过 exact Root 的 `REACT` 调它。
4. 比较回复、tool trace、Prompt digest、usage、Session batch 和错误终态。

回滚点：切回旧 Reasoner adapter，不迁数据。

### Phase 3：把默认算法内部拆成普通依赖

按这个顺序：

1. immutable `PROMPT_PARTS`；
2. `CONTEXT_VIEW`；
3. `TOOL_SELECTOR`；
4. ToolSearch 完整插件；
5. model binding 与 model-call receipt；
6. 删除 Reasoner 内对应开关、ContextVar 和 phase。

每删一条旧路径，先证明新插件能独立卸载、替换、热换并恢复。

回滚点：每个 capability 保留一个短期 adapter，但同一时刻只能有一个 writer。

### Phase 4：建立 `MESSAGE_HANDLER`，切被动入口

1. 原始渠道 envelope 在边界校验后进入 exact Root Handler。
2. `/stop` 旁路 Handler，只调 interrupt。
3. 默认 passive handler 调 `accept → react → complete`。
4. 命令、过滤和直接回复迁到 Handler 或各自普通 Service。
5. 删除 Core 对 `PassiveTurnPipeline` 和 `DefaultReasoner` 的直接构造。

回滚点：按 generation 原子切回旧 handler，不能按单请求随机双跑。

### Phase 5：迁移提交后 consumer 与 Delivery

1. Akasha、Markdown memory、embedding、UI 改为各自消费 SessionReceipt。
2. Delivery Owner 用 prepared envelope 消除 commit 后 crash gap。
3. 外部渠道 proactive 验证 provider-first 恢复。
4. 同步 `TurnCommitted` fanout 只剩诊断后删除。
5. `post_commit`、通用 after-turn extra 和旧 response patch 退出。

回滚点：consumer 可从自己的 cursor 重放；不得回滚 Session 消息。

### Phase 6：删桥并收窄 Core

只有全部 Gate 通过后才删除：

- 旧 passive phase bundle；
- `DefaultReasoner` 的 Core 接线；
- public Attempt/RunLock/AgentProgram 草案类型；
- ToolSearch Core 开关与授权 ContextVar；
- 通用 CommitIntent/DerivedStore/SessionChange 试验代码；
- 无 consumer 的 lifecycle event 和 DTO。

删除前再次核对外部插件源码、安装 cache、动态 consumer、测试和运行日志。cache 不是 canonical source，不能因为静态 `rg` 无命中就直接删接入点。

---

## 十二、验收 Gate

### 12.1 Turn 与身份

- [ ] `U1 → interrupt → U2 → interrupt → U3 → A` 只产生一个 Turn、三个内部 Run。
- [ ] completed Session batch 恰好为 `[U1,U2,U3,A]`，顺序与 Turn ordinal 一致。
- [ ] schedule/proactive 的 `run-only` Task Message 有稳定身份但没有 seq；目标 Session 只出现最终 A。
- [ ] failed 后显式 retry 复用原 MessageId；普通 fresh 产生新 MessageId 和新 Turn。
- [ ] MessageId 不依赖 seq；旧 ID 不被改写。
- [ ] 同 Session seq 单调且不复用。
- [ ] `/stop` 和 ignored inbound 不创建 ε Turn。
- [ ] 同一外部 source_ref 在 accept 前后任意 crash/重投都只产生一个 MessageId、TurnId 和 RunRef。
- [ ] 同一 Turn 只能完成一次；重复相同 complete 幂等，内容漂移失败。

### 12.2 并发与中断

- [ ] 同一 Session 同时最多一个 active conversation Turn。
- [ ] 不同 Session 可以并发。
- [ ] 热换 generation 不改变 lane key。
- [ ] interrupt token 只能结束它绑定的当前 Run；旧 token 不能杀新 Run。
- [ ] active Run 只接受 interrupt，普通输入明确 busy。
- [ ] 重启把失去 live owner 的 active Run 收束为 `interrupted(process_lost)`，不自动重跑外部效果。
- [ ] UI 用 TurnId 分组，用 latest_run_state 显示 running/interrupted/failed，不把 RunRef 当 TurnId。

### 12.3 Root 与权限

- [ ] Handler、REACT、Prompt、Tool schema、Tool execute 和 model binding 来自同一个 exact Root。
- [ ] Run 1 在 G10 中断、Run 2 在 G11 恢复时，TurnGrant 不扩大。
- [ ] hidden Tool、not-allowed Tool 和 invalid arguments 有不同结构化错误。
- [ ] ToolSearch 改 ToolView，不改 TurnGrant。
- [ ] ToolSelector 无法看到或泄露 grant 之外的 Tool metadata。
- [ ] candidate Root 不能写正式 Session、Memory、plugin-data 或外部服务。
- [ ] cleanup 失败保留 drain blocker 和全部错误。

### 12.4 Prompt、Context 与 Tool

- [ ] PromptPart 输入相同则顺序与 digest 相同。
- [ ] 两个插件不能改写彼此的 PromptPart。
- [ ] duplicate/exclusive Prompt key 在 candidate Gate 失败。
- [ ] `persistent history`、`runtime history view`、`prompt history` 在 API 和日志中不混名。
- [ ] ContextView 不写 Session，也不裁开完整 logical Turn 或 Tool pair。
- [ ] 卸载 ToolSearch 后换 selector 不改 Core。
- [ ] 无 Tool 的 REACT 能正常工作。

### 12.5 提交与恢复

- [ ] 在 complete 事务每个语句前后注入 crash，都只得到“全有”或“全无”。
- [ ] 非空 batch 的 Message、seq、Turn outcome、Run terminal、session head、SessionReceipt 同事务一致。
- [ ] run-only Turn 无 Session 变化时不生成假的 SessionReceipt。
- [ ] commit 后、wakeup 前 crash，consumer 能从 feed_seq 追上。
- [ ] consumer 失败不回滚 Session、不阻止其他 consumer。
- [ ] consumer 重放相同 Receipt 幂等，内容漂移失败。
- [ ] 存储损坏 fail-loud，不变成空 Session 或 cache miss。

### 12.6 Delivery 与外部效果

- [ ] 被动回复 Session commit 后发送失败，Session 内容仍存在且客户端可补尾。
- [ ] Akashic/Web/Mobile 通知不产生第二份 durable 正文。
- [ ] 外部 proactive 只有 provider full success 后才追加 Session。
- [ ] partial/uncertain 不自动重发，不伪装成功。
- [ ] Tool effect 在 Run retry 时按 owner receipt 对账，不重复执行。
- [ ] Run/Root cleanup 不声称回滚已经发生的外部效果。

### 12.7 热更新与自我更新

- [ ] ordinary run 只租 stable；validation child 可显式租 latest。
- [ ] candidate 行为 probe 证明真实调用，不只证明 import/manifest。
- [ ] owner Turn 完成并释放旧 lease 后才开始 promotion drain。
- [ ] promotion 失败保持 latest candidate 事实并恢复旧 stable formal Root。
- [ ] 旧 Run 不跨 Root；新 Run 不再进入退休 Root。
- [ ] durable operation 完成 handoff 后才允许旧 Root drain。

### 12.8 复杂度与非特权证明

- [ ] Core 不 import `default-react`、ToolSearch、Akasha、Scheduler、Proactive 或具体 Provider。
- [ ] Core 不按来源名、插件 ID、模型名或渠道名分支。
- [ ] 无 `MESSAGE_HANDLER` 时明确 unavailable，不回退旧被动链。
- [ ] echo handler 可在没有 LLM/Tool/Prompt/Memory 时完成 Turn。
- [ ] `simple-react` 可替换 `default-react`，不修改 Core。
- [ ] 目标公共词中没有 Attempt、AgentProgram、RunLock、CommitPlan、CommitIntent、DerivedStore。
- [ ] 新 helper、DTO、event 若没有独立 owner 或第二 consumer，删除或内联。

---

## 十三、需要批准的决定

### 推荐直接批准

1. **公共 Attempt 改为内部 Run。** 保留内部持久坐标，不保留第二套用户领域身份。
2. **保留 opaque MessageId，并与 seq 解耦。** 不采用 `(TurnId, ordinal)` 作为身份。
3. **使用 `REACT`，不引入 `AgentProgram`。** 默认 while loop 整体是普通插件。
4. **Session 保持 canonical Message rows。** 只增加同事务、无正文的 SessionReceipt feed。
5. **ToolSearch 只拥有 ToolView。** 权限永远由 TurnGrant 和执行边界拥有。
6. **一个 Run 一个 exact Root。** 同 Turn 的后续 Run 可以使用新 committed Root。
7. **外部效果归领域 owner。** 不建立通用 CommitIntent/EffectAttempt 平台。

### 实现前仍需单独确认

1. v4 物理表名与旧 `turns` 表怎样长期归档；不能原地解释旧行。
2. 新 MessageId 的具体编码；本文推荐 UUIDv7，但只要求 opaque 和稳定。
3. model-call binding receipt 的保留期，以及完整 payload 诊断默认关闭多久。
4. plugin-data 的全局数据删除 inventory 与阻断语义；它不是本轮 SessionReceipt 的职责。
5. 外部 proactive、`message_push` 和 BackgroundJobs 各自迁移批次；不能借被动链改造顺手改产品语义。
6. accepted 决策和 `projectneed` 中 `Attempt`、snapshot execution unit 的措辞如何升级；批准前仍以当前合同为准。

---

## 十四、证据索引

### 输入与恢复点

| 文件 | SHA-256 |
|---|---|
| `0902-reviewed-v3.md` | `d3e5e9e1ecc09cdea60a532357da31ec87207ffd68f5f6d26f6e913021bd54a0` |
| `0902-02.md` | `7c1dee104a8706f8a2030de1b22b4a00407a8ab5c74e2ef06e5485a7dd0207af` |
| `.0902-reviewed-v3.pre-v4-20260903-111415.bak` | 与 v3 相同 |
| `.0902-02.pre-v4-20260903-111415.bak` | 与 `0902-02.md` 相同 |

### 用户要求

- 2026-09-02 Codex 设计会话；原始记录保留在本地，不随公开 PR 提交。

### 当前项目合同

- `docs/projectneed.md:371`：不得使用无修饰的 history。
- `docs/projectneed.md:426`：completed Turn 持久化全有或全无。
- `docs/projectneed.md:430`：seq 单调且不复用。
- `docs/projectneed.md:434`：破坏性删除只接受用户显式意图。
- `docs/projectneed.md:444`：Session 正文正常只追加。
- `docs/projectneed.md:452`：未完成 logical interaction 的续接规则。
- `docs/projectneed.md:458`：一个 completed interaction 拥有全部输入和唯一最终回复。
- `docs/projectneed.md:560`：同 Session 串行，不同 Session 并发。
- `docs/projectneed.md:564`：active execution 只接受精确中断。
- `docs/projectneed.md:568`：每个执行单元冻结模型执行绑定。
- `docs/projectneed.md:638`：被动 Session-first 与 Akashic 主动消息语义。
- `docs/projectneed.md:648`：外部渠道按完整逻辑消息提交。
- `docs/projectneed.md:658`：硬终止只关闭 execution attempt。
- `docs/projectneed.md:664` 起：candidate、snapshot、cleanup 与插件发布不变量。
- `docs/design/persistence-state-map.md`：持久对象 owner、增改减、备份与恢复边界。
- 决策 0034：Turn 是逻辑工作单元，Attempt 是当前内部执行概念。
- 决策 0039：Core 原子能力来源无关，`react` 是唯一控制流。
- 决策 0041：Turn effect 与 Memory 插件正交，Akasha 是普通插件。
- 决策 0045：Akashic 主动消息先提交 Session，客户端使用 `message_id + seq`。
- 决策 0008、0036、0046：Root/generation/snapshot lease、增量 candidate 与 drain。

### 当前代码

- `agent/looping/core.py:335-369`：Core 直接组 `DefaultReasoner` 与被动 pipeline。
- `agent/core/passive_turn.py:355-435`：固定 phase 主链。
- `agent/core/passive_turn.py:958-1075`：Reasoner 内 ToolSearch、Prompt 和 step phase。
- `agent/lifecycle/phases/after_reasoning.py:257-333`：多 user input 的提交准备。
- `agent/lifecycle/phases/after_turn.py:243-281`：同步 committed fanout。
- `session/manager.py:642-738`：Session message 批次事务。
- `session/store.py:2392` 起：当前 `turns` execution record。
- `session/store.py:4744`：当前 MessageId 从 session/seq 生成。
- `session/store.py:5385-5539`：completed interaction 显式删除与恢复证据。
- `agent/plugins/snapshot.py:876-988`、`1567-1665`：exact snapshot lease 与 drain。
- `agent/plugin_composition/effect.py:16-148`：Effect setup/rollback/LIFO cleanup。

### DeepSeek Harness 参考

- `deepseek-harness/docs/architecture.md:11-29`：所有组成件都是插件、profile 与 live patch reload。
- `deepseek-harness/docs/architecture.md:51-95`：默认 core package、Turn flow 与 durable Session event。
- `deepseek-harness/packages/core/agent-loop/src/agent.ts:234-359`：默认 loop 直接拥有 pre-step、Turn、Prompt、LLM 与 Tool 驱动。
- `deepseek-harness/packages/core/system-prompt/src/index.ts:424-536`：Prompt contribution 通过 effect 注册并确定性组装。
- `deepseek-harness/vendor/cordis/src/fiber.ts:405-560`：effect 注册、逆序 disposer 与卸载行为。

---

## 最终判断

v3 的方向是对的：让 AgentLoop、ToolSearch、Memory 和来源能力回到插件。但它仍把“为了迁移而看得见的运行细节”抬成了长期领域对象，又用通用 Commit、Change、DerivedStore 和 Effect 平台去包住所有失败。

`0902-02.md` 做对了主要减法，却又删掉了不能删的稳定 MessageId，并把 Session 误写成变更日志，还让 Channel 与 Delivery 重复拥有同一个事实。

v4 的改进不是增加一个更精巧的总框架，而是把边界缩到刚好够用：

```text
Message 组成 Turn
Turn 中该进入 Session 的 Message 整批原子提交
Run 只是 Turn 的内部执行
每个 Run 只看一个 Root
REACT 只是普通插件
外部效果回到自己的 owner
```

如果一个新能力不能用这些积木直接表达，它必须先证明自己拥有新的权威状态、不变量、控制流或生命周期；否则不进入 Core。
