# Akashic v4：Message WAL、Turn Projection 与普通插件组合

- 文档版本：0902-reviewed-v4
- 日期：2026-09-03
- 状态：设计提案，等待维护者批准
- 当前代码基线：47896b4200731183a54081e2eca77602a0881a0a
- DSH 参考基线：49a606bc5b5934603f22a26957a07dc799ab0291
- 本文不授权：实现、数据库迁移、正式 workspace 写入、删除、部署或合并

## 结论

v4 同时完成两件彼此正交的事：

1. **对话事实只剩 Session 与 Message。** Session 是一条只追加的 Message WAL；用户输入、
   Agent 输出、Tool 调用和 Tool 结果只要被接纳，就立即成为 Message，不再等一个 Turn
   完成后批量补写。
2. **对话行为由普通插件组合。** 当前固定的被动回复大链被拆成 Turn projection、
   `passive-conversation`、可替换 `AGENT_PROGRAM`、受保护 `ToolHost` 和普通行为插件；
   Core 不认识 passive、proactive、Wake、Scheduler 或某个 Agent 算法。

`Turn` 仍然是有用的用户概念，但它只是一个普通插件从 Message 因果图算出的 projection：
不落库、没有稳定 TurnId，也不决定 Message 能否存在。不同 Turn 插件可以改变分组规则，
而不迁移 Session WAL。

最重要的例子是：

~~~text
真实提交顺序

seq 1  Human U1
       interrupt                 # 只取消短命执行，不写 Message
seq 2  Human U2
       interrupt
seq 3  Agent P                   # 与 U1/U2 无关的 proactive A
seq 4  Agent A1 responds_to(U1, U2)

Turn projection

Turn H = {U1, U2, A1}
Turn P = {P}
~~~

`P` 插在物理时间线中间，不会让 A1 失效，也不会把自己塞进 Turn H。A1 的正确性取决于
它选择的 causes 是否仍未被结算，而不是 Session head 是否完全没变。

### 把我当六岁

Session 是唯一一本作业本，Message 是已经用墨水写下的一行。

- U1 写进作业本以后，按下“停”只让机器人停笔，不会把 U1 撕掉。
- 后来写入 U2，它也马上留在本子里。
- 机器人最后写 A1，并明确说“我在回答 U1 和 U2”。
- 中间另一个机器人写了一条主动消息 P，也不会把 U1、U2 或 A1 挤走。
- Turn 插件只是拿彩笔画圈：把 U1、U2、A1 圈在一起，把 P 单独圈起来。
- 换一个画圈插件，圈法可以改变；本子里的字和顺序完全不变。

模型重试、半截 token、取消信号和正在运行的 Python object 都像草稿纸：可以丢失。
只有完整 Message append 成功以后，才算真正写进了作业本。

## 一、唯一事实模型

### 1.1 Session

~~~text
Session {
  session_id
  messages: Message[]  # 按 seq 连续排列
}
~~~

这里的 WAL 是领域层的 append-only Message log，不是再套一张 `SessionEvent` 表。底层
可以使用 SQLite WAL，但 `sessions.db/messages` 才是产品真源。

Session 不拥有当前 Turn、Run、Attempt、投递状态、模型请求或投影缓存。

### 1.2 Message

~~~text
Message {
  message_id      # append 前由可信边界产生；全局稳定、不透明
  session_id
  seq             # WAL commit 时分配；Session 内单调且不复用
  author          # Human | Agent | Tool | Source(plugin_key)
  responds_to[]   # 结构化 cause 边；不是正文 block
  content[]       # 完整、类型明确的正文或治理内容
}
~~~

初始内容 union 只保留当前设计必须证明的类型：

~~~text
text         { text }
artifact     { artifact_ref, media_type }
tool_call    { name, arguments, tool_binding }
tool_result  { resolves: call_message_id, outcome, output }
no_reply     {}
tombstone    { seed_message_ids[] }
~~~

新增 image、audio、citation 等能力时扩展 typed content，不新增平行 Message 表，也不加
通用 `meta/context/intent` 袋子。

`responds_to` 不放进 `content[]`，因为它不是聊天正文，而是这条 Agent Message 结算了
哪些 cause。它没有 block 顺序，不参与 citation span，也不会与渠道的原生 reply-to 混淆。
默认只有 Agent Message 可以写 `responds_to`；Tool result 用 `resolves` 结算 exact
`tool_call`。所有 cause/ref 必须指向同一 Session 中更早、真实存在的 Message/block；重复、
跨 Session、未来引用或错误 block 类型都在 append 边界 fail-loud，因此因果图天然无环。

### 1.3 持久 author 不是 provider role

旧的 `role=system|user|assistant|tool` 把两个不同问题绑在了一起：

~~~text
谁产生了这条事实？
模型应当给它多高的指令权限？
~~~

v4 只持久化 author。模型请求 projection 再做映射：

| Message author/content | provider request view |
|---|---|
| `Human` text/artifact | user input |
| `Agent` text/tool_call | assistant |
| `Agent` no_reply | 不产生 provider body；只保留已生成的 cause settlement |
| `Tool` tool_result | tool |
| `Source(plugin_key)` | untrusted task/context frame，绝不自动成为 system instruction |
| tombstone | 不进入模型请求 |

system/developer instruction 只来自 Agent Program 拥有的 Prompt assembler。Scheduler、
Wake 或第三方 source plugin 即使能向 Session 写 Source Message，也不能借 `system` role
绕过 Prompt 权限。

### 1.4 `message_id` 与 `seq` 仍然都需要

两者不是重复身份：

| 字段 | 唯一职责 |
|---|---|
| `message_id` | 识别“是不是同一条 Message”、幂等重试和引用 |
| `seq` | 表示一个 Session 内的提交顺序和客户端 cursor |

如果只用 seq，ACK 丢失的 producer 在 commit 前不知道最终序号，无法证明重试的是同一条
Message。如果只用 message_id，客户端仍不知道顺序和缺口。

但 canonical `message_id` 不能直接信任外部客户端。可信 Channel boundary 必须：

- 从已认证 account/device/session namespace 与 provider inbound reference 确定性映射；或
- 在发送前向客户端签发/验证属于该 namespace 的最终 ID。

外部 nonce/provider ref 只是 Channel staging 的传输输入，不进入 conversation schema，
也不恢复 `client_message_id + message_id` 双重身份。

Agent/Tool/Source/Tombstone 路径则由对应可信 writer 在完整内容 sealed 后、append 前一次性
mint server-namespace message_id；append ACK 丢失时复用它，进程在 append 前崩溃则什么
事实都没有留下。

初始 v4 每条 Agent Message 最多包含一个 tool_call，因此那条 Message 的 `message_id` 就是
调用身份。Tool result 直接 `resolves: call_message_id`，不再增加 ToolCallId、`call_ref` 或
block-index identity。将来若要并行 tool_call，应先证明值得扩大这个合同。

## 二、唯一写协议

### 2.1 输入先写 WAL，再开始 Agent

任何被接纳的 Human 输入都立即 append：

~~~text
Channel 收到 U1
    │ 认证、附件 durable、canonical message_id
    ▼
append Human U1 ── durable ──▶ ACK inbound
    │
    └──▶ 唤醒普通插件
~~~

因此 `U1 → interrupt → U2 → interrupt → A1` 的 Session 从来不会暂存 U1/U2 到某个
Attempt row，最后再批量复制正文。interrupt 只取消当前内存 Reaction；U1/U2 已经是事实，
不能因为模型没答完而丢失。

这里 interrupt 只表示“停掉这张草稿纸”，不表示“永远不要回答这条 Message”。后者是
持久产品意图，必须由一条被接纳的 Message 表达，再由行为插件依照 Turn policy 结算，
不能藏在已消失的 cancel flag 里。

这会有意替换当前 SES-001、SES-007、SES-008 和 decision 0025 中“最终 A 出现时再批量
提交完整 transcript”的合同。旧合同是迁移原点，不是 v4 的正确性来源；只有 Phase 0
批准并勘误长期条款后才能实现。

写入事务保持最小：

1. 验证 typed writer 与 Message 结构；
2. 若同 `message_id` 已存在，比较不可变内容：完全相同则返回原 receipt，不同则
   fail-loud；
3. 检查该 writer 的 typed precondition；
4. 分配下一个 seq 并 INSERT；
5. commit 后发布 feed。observer 失败不能回滚 Message。

### 2.2 Cause CAS 是基础；head CAS 只是可选隔离级别

先把不可变事实和可替换 policy 分开：

~~~text
Settled(m, S) = 存在 committed Agent Message a，且 m ∈ a.responds_to

CauseOpen(m, S) =
  m 存在于 S and not Settled(m, S) and not Hidden(m, S)

PendingCause_T(S) = {
  m | TurnPolicy_T.Reactable(m, S) and CauseOpen(m, S)
}
~~~

`CauseOpen` 是 Session owner 能在 append 临界区检查的结构事实；`Reactable` 和怎样把 open
cause 分组属于可替换 Turn plugin。这样换 Turn policy 不会把任意插件代码塞进 WAL 事务。

Agent 输出 A 针对明确 cause set C 时，基础提交条件是：

~~~text
CanAppend(A, C, S) =
  C 是 writer 绑定的 exact cause set
  and C 中每个 Message 仍满足 CauseOpen(m, S)
  and writer authority 仍有效
  and 可选 read condition 仍成立
~~~

AgentReplyWriter 只能由受保护 factory 针对 `PendingCause_T` 中一个 projected group 的
**全部 open causes** 签发；
普通插件不能自己拼 cause ID。Session 任意 append 不再自动使 A 失效。调用它的行为插件
再按工作语义选择一个很小的 typed read condition：

| 条件 | 用途 |
|---|---|
| 无额外 read condition | command、独立 cause；只使用必选 CauseOpen CAS |
| `NoNewHumanInputAfter(source_seq)` | 默认被动回答；新 Human 对话输入加入本组，tombstone/Source/主动 Agent 不干扰 |
| `HeadEquals(source_seq)` | 极少数确实要求读取最新完整 Session 前缀的算法 |

这些条件由 Session owner 在同一个 append 临界区检查，不持久化 token/revision。首版只
提供封闭的 typed 条件，不接受插件传任意 predicate 或可变配置袋。
`HumanInput` 由 author + conversation content 的结构类型判断，不把 Human tombstone 当成
新对话输入。

#### U2 会打断旧草稿，proactive P 不会

~~~text
seq 1  Human U1
       Reaction R0 读取到 source_seq=1，writer 条件 NoNewHumanInputAfter(1)

seq 2  Human U2
       R0 append A-old ──▶ condition conflict；A-old 丢弃

       Reaction R1 causes={U1,U2}，source_seq=2
seq 3  Agent P，独立主动消息
       R1 append A1(responds_to U1,U2) ──▶ success
seq 4  Agent A1
~~~

如果 A-old 的事务先于 U2 commit，它就是对 U1 的合法完整回答，随后 U2 成为新 pending
cause；如果 U2 先 commit，A-old 必须失败。这是唯一需要定义的竞态顺序。

feed 收到 P 可以触发一次无害 reconcile，但 `passive-conversation` 只在 projected cause
group 改变时取消当前 Reaction；P 没有改变 `{U1,U2}`，所以无需判断 `is_proactive`，也不
取消 R1。

### 2.3 不公开通用 `MessageWriteGrant`

一个不断增加 `role/mode/variant/cause/call/expiry` 字段的通用 grant 会变成权限配置袋。
v4 让类型直接表达用途：

~~~text
InboundWriter.append_human(content)                # exact Session + authenticated boundary
AgentReplyWriter.append(content)                   # exact Session + causes + read condition + Root
ToolResultWriter.complete(outcome, output)         # exact call Message
SourceWriter.append(content)                       # exact Session + Source(plugin_key)
StandaloneAgentWriter.append(content)              # exact source job + target Session, no causes
TombstoneWriter.hide(seed_message_ids)             # exact user-approved management request
~~~

每个 writer 都是不可伪造、不可序列化、短命且窄用途的 object capability：

- 调用者不能再传 `author`、另一个 Session、另一个 cause set 或另一个 call Message；
- `AgentReplyWriter` 的 `responds_to` 自动等于 exact causes，不能只消费宽授权的一部分；
- `ToolResultWriter` 只能完成一个 call；相同结果幂等，冲突结果 fail-loud；
- `SourceWriter` 只能写绑定 plugin_key 的低信任输入；`StandaloneAgentWriter` 只能为绑定
  source job 写无 cause 的完整 Agent Message；
- `TombstoneWriter` 只能写纯 tombstone Message，不能混入聊天正文；
- 底层仍只有一个 Session WAL owner 和一个 internal append，不形成多个事实 owner。

### 2.4 短命 receipt 不会创造第二份真相

“durable 结果只看 WAL”不等于所有函数都必须返回 `None`。调用者和测试可以得到只引用
既有事实的短命 receipt：

~~~text
AppendReceipt(message_id, seq, already_existed)

ReactionReceipt =
  Appended(message_id)
  | Conflict

ToolReceipt =
  InFlight(call_message_id)
  | Result(result_message_id, already_existed)
~~~

receipt 不落盘、不携带正文、不授权后续写入，也不能覆盖 WAL。丢掉它以后重新 fold
Session 仍得到相同答案。`InFlight` 只描述当前进程眼前的 task，随时可能过期；`Result`
只引用已经存在的 terminal Tool Message。ToolReceipt 不是可恢复 Tool 状态，恢复仍只 fold
call/result Message。

### 2.5 模型 retry 发生在输出 append 前

~~~text
读取 immutable Reaction view
        │
        ├── provider 断网：重试，Session 不变
        ├── partial token：只给当前 UI preview，崩溃可丢
        └── 完整输出 A
                  │
                  ▼
          AgentReplyWriter.append(A)
~~~

重试耗尽时，产品可以生成一条完整 Agent error Message；没有生成 Message，就不声称
Agent 说过。commit 成功但 ACK 丢失时，用同一个 canonical `message_id` 重试并拿到原 seq。

## 三、Turn 是普通插件拥有的 projection

### 3.1 为什么保留 Turn 这个词

用户、历史窗口、Akasha 和 UI 都需要“哪些 Message 属于一件完整事情”的视图。因此
公开一个有真实多消费者的普通 capability：

~~~text
turn-projection plugin
  provides TURN_VIEW
  injects  SESSION_READ
~~~

`TURN_VIEW` 可以返回 `member_message_ids`、完成/等待的派生状态和 UI 所需顺序，但：

- 不创建 Turn row、TurnId、open/seal/abort 状态；
- 不拥有 Message commit；
- 不被 retry、Tool 或 Delivery 当作寻址身份；
- 缓存只能带 `source_seq + projection_version`，可删除、可重建；
- UI 需要 key 时复用 root/first `message_id`，不生成 ProjectionId。

这让 Turn 插件可以真正“自己定义 Turn”。替换它会改变分组、上下文切点和学习样本，
但不会迁移 Message。

### 3.2 默认分组使用因果图，不使用连续行

默认 Turn projection 使用两类不可变边：

~~~text
cause Message ──responds_to──▶ Agent Message
tool_call block ──resolves───▶ Tool result Message
~~~

它先形成因果连通分量，再应用以下默认 policy：

默认 `Reactable` 只有 Human conversation input、Source input 和 terminal Tool result；Agent
输出与 tombstone 不会仅因“排在最后”而自动变成新 cause。

1. 当前尚未被回答的有序 Human Message 组成一个 pending human group；interrupt 不切断它；
2. Agent A 明确 `responds_to(U1,U2,...)` 后，A 与这些 Human cause 属于同一 Turn；
3. Agent tool_call、对应 Tool result、继续回答通过边的传递闭包归入同一 Turn；
4. 一个 Human Turn 尚有 unresolved call 时，后来进入的 Human input 先成为该 Turn 的
   deferred open cause；TURN_VIEW 只标记 deferred，`passive-conversation` 在 Tool result
   到达前不得为它单独申请 Reaction；
5. Source 触发的 Agent 工作单独成 Turn，不与同时未完成的 Human group 混合；
6. 没有 Human cause 的 standalone Agent Message，例如直接 proactive/message_push 输出，
   自己成为一个 Turn；
7. tombstone 等治理 Message 不成为聊天 Turn。

因此 Turn member 不要求 seq 连续：

~~~text
seq 10  Human U1      ┐
seq 11  Human U2      ├── Turn H
seq 12  Agent P       │                 Turn P = {P}
seq 13  Agent A1      ┘  responds_to(U1,U2)
~~~

这里 Turn H 是 `{seq10, seq11, seq13}`。投影不能因为 seq12 插入就拆开 H，也不能因为 H
跨三次短命 Reaction 就增加 Attempt identity。

Tool 也不制造第二种 Turn 规则。默认插件可以把等待 Tool 期间到来的新 Human cause 接到
同一条因果链：

~~~text
seq 20  Human U1                                  ┐
seq 21  Agent M1 responds_to(U1), tool_call C1   │
seq 22  Human U2                                  │ Turn H
seq 23  Agent P                         Turn P    │
seq 24  Tool T1 resolves(C1)                      │
seq 25  Agent A2 responds_to(T1,U2)               ┘
~~~

这里 M1 已结算 U1；最终 continuation 只结算仍 open 的 T1 与 U2，传递闭包仍得到
`{U1,M1,U2,T1,A2}`。如果另一个 Turn plugin 选择把 U2 分开，它可以改变这个 projection，
但不能改 WAL 或伪造另一条 Message。

中间快照必须明确：到 seq22 时 group 是
`members={U1,M1,U2}, unresolved={C1}, deferred={U2}`，所以不签发 `Reaction(U2)`；seq24
到达后才得到 `open_causes={T1,U2}` 并签发唯一 continuation。否则先写 A(U2) 会把 U2
结算掉，破坏上图承诺的默认 grouping。

### 3.3 Turn 插件只 fold 视图，不执行 action

Turn 插件对完整 WAL 计算：

1. 先收集全部 `responds_to` 与 tool `resolves` settlement 边；
2. 再应用 tombstone 得到 Hidden set；隐藏旧回复不能让已结算 cause 复活；
3. 标出未结算 tool_call 及等待它的 Turn；
4. 按 policy 投影 exact pending cause groups；默认不混合 Human 与 Source group；
5. 没有 pending cause 或 call 的 Turn 标为 complete。

projection 不决定哪个 group 先执行，不取消 Reaction，不执行 Tool、不调用模型、不 append。
`passive-conversation` 等行为插件只能从完整 projected group 申请 Reaction；ToolHost 则直接从
WAL 找 unresolved call。重复 fold 必须得到相同结果。

#### 分组不改写时间顺序

`TURN_VIEW` 返回的只是 `root_message_id → member_message_ids[]`。root 复用该组最早的
Message ID，不是新的 TurnId。WAL 顺序仍只由 seq 决定：时间线 UI 仍按 `U1,U2,P,A1`
显示，并给 U1/U2/A1 标相同 group；它不能为了做一张连续 Turn card，把 A1 移到 P 前面。
需要整组样本的 Akasha/compaction 可以按 member IDs 收集非连续成员。

### 3.4 其他全部是 Message projection

| projection | 从 Session Message 得到什么 |
|---|---|
| Chat | 按 seq 的可见内容与 Turn group 标记 |
| Model context | author/content 到 provider request 的有界映射 |
| Tool status | call + result 得到 pending/success/error/unknown |
| Web/Mobile | `seq > cursor` 的 Message 与按需 Turn view |
| Memory/Search/Akasha | 按 TURN_VIEW 与允许学习规则建立派生项 |
| Live stream | 尚未 commit 的内存 preview；刷新可丢 |

客户端最小同步仍是：

~~~text
request:  session_id, after_seq
response: Message[]
cursor:   最后完整应用的 seq
~~~

客户端用 message_id 去重、seq 排序。Turn plugin 是分组规则的唯一 owner；服务端按
`source_seq + projection_version` 返回可丢的 group membership。客户端离线时退化为纯 seq
时间线，不复制一套可能漂移的 Turn 算法，也不要求另一条权威 cursor。

## 四、被动回复大链变成普通插件组合

### 4.1 当前真正需要替换的固定链

当前基线仍是：

~~~text
PassiveMessageWorker
  ├── inbound custody / attachment / per-session lane
  ▼
ConversationRuntime
  ▼
AgentLoop._react()
  ▼
PassiveTurnPipeline
  ├── command short-circuit
  ├── BeforeTurn / BeforeReasoning
  ├── reasoner + BeforeStep / AfterStep
  ├── AfterReasoning：parse + persistence + outbound
  └── AfterTurn：event + dispatch / ACK
~~~

代码证据：

- `bootstrap/passive_worker.py:96`：worker 拥有准入、lane 与结果 task；
- `agent/looping/core.py:556`：`AgentLoop._react()` 转入固定 pipeline；
- `agent/core/passive_turn.py:355`：固定四段 phase；
- `agent/core/passive_turn.py:440`：command 专门短路；
- `agent/core/passive_turn.py:524`：默认被动回复由固定 `run()` 统管。

只换成 Message WAL 而保留这条大链，仍然不能替换 Agent 算法，也会继续给 Core 加
Citation、Meme、Tool Search、proactive 等分支。

### 4.2 目标拓扑

~~~text
turn-projection plugin
  provides TURN_VIEW

passive-conversation plugin
  owns     startup reconcile + SESSION_FEED subscription
  private  reconcile(session_id)
  injects  TURN_VIEW, COMMANDS, AGENT_PROGRAM, REACTION_FACTORY

default-agent plugin
  provides AGENT_PROGRAM
  injects  PROMPT_PARTS, CONTEXT_VIEW, TOOL_SELECTOR, CHAT_MODELS

protected substrate
  Session WAL + typed writers
  ReactionFactory + exact Root lease
  ToolHost + effect-start gate
  Channel ingress + Delivery effect port
~~~

不再公开 `MESSAGE_REACTOR`。它只有 `passive-conversation` 自己一个 consumer，没有独立
owner 或生命周期，所以只是插件内部函数。真正可替换的公共行为边界只有：

~~~text
AGENT_PROGRAM.react(reaction) -> ReactionReceipt
~~~

### 4.3 Reaction 只是短命 capability，不是 Run

`passive-conversation` 从 TURN_VIEW 选择 exact causes 后，让受保护 factory 创建：

~~~text
Reaction {
  causes: immutable MessageView[]
  history: immutable SessionView(source_seq)
  model: scoped ModelPort
  tools: scoped ToolPort
  reply: AgentReplyWriter
  stream: StreamPreview
  cancellation
  resources
}
~~~

Reaction 没有 ID、表、序列化、恢复或状态机；函数结束就失效。它只是把当前调用可用的
几扇小门绑在一起：

- exact Root lease 在一次 `react()` 和其中的模型 retry 内不变；
- Agent Program 看不到任意 Session repository、裸 SQL 或通用 append；
- `reply` 已绑定 exact Session、cause set、author、read condition 和一次性预算；
- `tools` 只能把已 commit 的 call Message ID 交给受保护 ToolHost；
- 崩溃后重新 fold WAL，用新 Reaction 恢复，不恢复旧 Reaction。

一次 Reaction 最多 append 一条 Agent Message。若它包含 tool_call，Reaction 在 commit 后
结束；ToolHost 独立结算并 append Tool result，feed 再为该 result 创建一张新 Reaction。
所以 continuation 靠 Message 因果边恢复，不靠跨 Tool 等待存活的 Run 或 Reaction；新
Reaction 可以在升级后使用新的 Root，已提交 call 的 exact Tool binding 仍保持不变。

### 4.4 `SESSION_FEED` 只唤醒，但启动必须无缺口

仅订阅未来 callback 会漏掉“Message commit 后、subscriber 收到前进程崩溃”的 Session。
每个必须追赶 WAL 的 owner 都使用同一个 snapshot-to-feed handoff：

~~~text
1. 打开 subscription，取得短命 feed watermark W
2. 在一致 snapshot 中枚举截至 W 的 Session heads
3. 对每个 Session fold/reconcile
4. 消费 W 之后的通知；扫描与通知重复时按 message_id/因果 fold 去重
5. 追到当前 head 后才把该 consumer 标为 ready
~~~

watermark 是 feed 实现游标，不进入 Session schema，也不成为恢复身份。崩溃后从头重复
上述 handoff。`passive-conversation`、ToolHost recovery 和 Delivery projector 都必须遵守；
feed 丢一条唤醒不能让已 committed Message 永久无人处理。
若 subscription buffer 溢出或 watermark 不再可读，consumer 必须放弃 ready、从新 W 重做
handoff，不能用“可能追上了”继续运行。

### 4.5 固定 phase 的能力去向

| 当前固定行为 | v4 owner |
|---|---|
| channel envelope、附件导入 | Channel/Artifact adapter；artifact ready 后使用 InboundWriter |
| inbound handoff / ACK | Channel adapter；Human Message durable 后结算 |
| Turn 分组、pending causes | 普通 `turn-projection` 插件 |
| command catalog 与短路 | `passive-conversation` 私有 reconcile + `COMMANDS` |
| Session/history 准备 | immutable SessionView + `CONTEXT_VIEW` |
| policy、identity、task、context Prompt | Agent Program assembler + 封闭 `PROMPT_PARTS` slots |
| context 裁切、摘要、compaction retry | `CONTEXT_VIEW` / provider request projection |
| tool schema preload / Tool Search | `TOOL_SELECTOR`，只决定模型可见集合 |
| Tool 授权、唯一执行、恢复 | 受保护 `ToolHost` |
| Tool schema/query/execute | 普通 Tool 插件的 exact generation |
| 默认 ReAct、provider retry、empty reply、terminal policy | `AGENT_PROGRAM` |
| model/provider binding | 普通 model/provider 插件，通过 Reaction scope 冻结 |
| Citation | Agent Program 根据 Tool/Prompt 直接产生 typed citation content |
| Meme/媒体 | 普通 Tool 产生 artifact；Agent Program 决定是否写入最终 Message |
| Agent/Tool 写入、幂等、cause CAS | typed writers + Session WAL owner |
| Memory、Akasha、compaction 派生项 | committed Message/Turn observers |
| partial stream | `StreamPreview`，可丢且不能 append 半条 Message |
| error/no_reply | Agent Program 产生普通完整 Message |
| delivery/ACK | Delivery effect 与 source state 各自拥有，见第七节 |
| generation、取消、资源清理 | Root lease + Reaction resource scope |

Command 若消费了一个 cause group，也必须通过同一个 AgentReplyWriter append text、artifact
或 no_reply；插件内部的 `handled=true`、日志或返回值不能把 Message 结算。带持久或外部
副作用的 command 还必须调用拥有该状态的 typed domain/effect port，不能借 Turn projection
藏一套恢复状态。

### 4.6 删除通用 `ASSISTANT_TRANSFORMS`

v4 不把旧 `PassiveTurnPipeline` 换皮成一串共享正文 mutator。最终输出只有：

~~~text
Agent Program 产生 final typed content
        ──▶ validate
        ──▶ AgentReplyWriter.append
~~~

Citation、Meme 或媒体必须通过 Prompt、Tool 和 typed content 显式进入 Agent Program 的
决定；不允许插件 A 改文字、插件 B 改附件、插件 C 再声明必须排在 A 后面。append 后的
observer 更无权回来改 Message。

Prompt contribution 也不是靠 plugin_id、数字 priority 或词法排序解决语义冲突。首版只
提供 assembler 定义的封闭层：

~~~text
policy → identity → task → context
~~~

每个层的 owner、是否允许多 contribution 和合并规则由 Agent Program 明确声明；重复
exclusive owner、循环依赖或无法解释的冲突在 candidate 阶段 fail-loud。

## 五、Tool：意图和结果是 Message，安全 owner 不是普通插件

### 5.1 正常链路

~~~text
Agent Message M10
  responds_to(U1,U2)
  tool_call(search, args, exact binding)
              │ commit 后 ToolHost 才能执行
              ▼
Tool Message M11
  tool_result(resolves=M10, success, output)
              │
              ▼
Agent Message M12
  responds_to(M11)
  text(final answer)
~~~

`outcome` 只有 `success | error | unknown`。Tool 调用可能已经发生但无法确认时必须写
`unknown`，不能猜成功，也不能盲重试。

初始 v4 的 Agent Message 最多包含一个 tool_call，且同一 Turn 有
unresolved call 时不启动 continuation。于是等待期间进入的 U2 可以在 result 到达后与
该 result 一起成为下一张 Reaction 的全部 open causes，不需要为了并行 Tool 再引入 work
identity。未来若要并行 tool_call，必须先给出同组 late result 的 typed read condition，
不能退回全局 head CAS。
provider 一次返回多个 call 时，Agent Program 在 append 前请求修正或报错；不能偷偷拆成
多条已经“说过”的 Message。

### 5.2 Protected ToolHost 与普通 Tool plugin 分开

受保护 `ToolHost` 独占这些不变量：

- 从 call Message ID 读取原 Agent Message，验证唯一 tool_call 和 exact binding；
- 重新检查 Tool 权限；模型可见性不能扩大真实授权；
- 为 exact call 签发 `ToolResultWriter`；
- 保证一个 call 只有一个 terminal result；
- 管理 effect-start gate、运行中 generation lease、恢复扫描和 `unknown`；
- 决定旧 generation artifact 何时可以物理回收。

普通 Tool 插件只拥有：

- schema 与参数解析；
- 领域校验；
- provider query/idempotency；
- 实际执行与领域结果格式。

因此 Agent Program 使用 `reaction.tools.execute(call_message_id)`；它不能注入一个可由任意普通
插件替换安全 owner 的全局 `TOOL_EXECUTOR`。替换 Tool plugin 可以改变领域能力，不能
绕过 grant、binding、唯一结果或删除闸门。

### 5.3 即时执行和 crash recovery 是同一条 ToolHost 路径

两种唤醒都调用：

~~~text
ToolHost.execute_or_recover(call_message_id) -> ToolReceipt
~~~

1. Agent Program commit tool_call 后立即请求；
2. ToolHost 按第 4.4 节启动扫描并订阅 feed，发现 unresolved call 后请求。

重复请求以 call Message ID、本地互斥和 result append 幂等收敛。provider 能 query 就先 query；
支持相同幂等键才可安全重试；两者都不支持就写 `unknown`。

停用 `passive-conversation` 或 `AGENT_PROGRAM` 不会遗弃 committed call。ToolHost 仍结算，
result 留在 WAL；以后启用的 Turn 插件会看到它。管理员停用某个 Tool generation 前必须
先拒绝新 binding，并处理全部 WAL 引用。

### 5.4 generation 的持久可达性来自 WAL

内存 lease 只能保护当前进程正在执行的调用，不能保护“tool_call 已 commit、observer
尚未运行就崩溃”的窗口。generation GC 必须直接读取 WAL projection：

~~~text
CanRetire(g) =
  不存在任何尚无 terminal tool_result 的 call c，
  且 BindingGeneration(c) == g
~~~

即使 call 已被 tombstone 隐藏，只要它仍可能已经执行而尚无 result/unknown，generation
仍不可删除。启动时先从完整 WAL 重建这些引用，再允许清理旧 artifact。exact generation
意外缺失时记录 incident 并写 `unknown`，绝不换用“差不多”的新 generation。

### 5.5 Tombstone 与 effect start 只有一个先后顺序

Tool 和 Delivery 在跨过真实外部 I/O 起点前取得 per-session shared **start permit**；
TombstoneWriter commit 前取得同一闸门的 exclusive permit。permit 没有 ID、不落盘，只
负责把“允许开始”与“治理 Message 先提交”排成先后。

~~~text
effect 先拿 shared permit
  └── 在 permit 内重查 pending/Hidden/authority，固定 generation 并真正发起 I/O
      └── tombstone 随后可提交；effect 继续结算，晚到 result 自动被 projection 隐藏

tombstone 先拿 exclusive permit
  └── commit seed
      └── 后来的 effect 看见 Hidden，不发起 I/O
          ├── Tool append terminal tool_result(error: hidden_before_start)
          └── Delivery 写 suppressed
~~~

permit 不需要等整个 provider 调用完成，只保持到请求已被 provider 接纳，或本地 effect
已经完成；如果 adapter 无法把 start 与 await-result 分开，必须在有界 timeout 内保持。
运行中 exact generation lease 则持续到 result/unknown。

Tool 没有另建 durable `started` row。若进程在 start 之后、result 之前崩溃，重启从 call
Message query/idempotency/unknown。若此时已有 tombstone，绝不新执行，只 query；不能确认
就写 unknown。这是没有 RunId 时必须诚实接受的保守边界。

## 六、Proactive 与其他 source 不再是 Core 特判

Core 不认识 proactive、Wake、Scheduler、Drift、`message_push` 或 subagent。source plugin
先在自己的边界决定是否有内容真正进入对话：

~~~text
no_due / 未选择 / 尚未生成完整输出  ──▶ 不写目标 Session Message
需要可恢复的 Agent 工作输入          ──▶ Source Message
Agent 生成完整主动输出                ──▶ Agent Message
模型对已进入 Session 的 cause quiet   ──▶ Agent no_reply Message
~~~

Source Message 永远是低信任 context，不是 system instruction。一个 source-triggered
工作可形成 `{Source, tool_call, tool_result, Agent A}` 的 Turn projection；Chat 可以只显示
A。直接 `message_push` 的 standalone Agent A 自己形成一个 Turn。

各 source 仍有不能从 Session 反推的 durable state：

| source owner | 自己保留的 state | 进入 Session 的边界 | 结算 |
|---|---|---|---|
| Channel adapter | inbound provider ref/staging、认证 route | `InboundWriter` 写 Human Message | Message durable 后 ACK |
| Scheduler | schedule、enabled、next fire、missed tick | 需要 Agent 时写 Source Message；无事不写 | job state 自己推进 |
| EventMail | Content/Alert/Context envelope 与各自 transition | 被选择且需要对话时写 Source Message | supersede/expiry/selection 仍归 EventMail |
| Wake | watermark、receipt、reservoir、hazard、pending ACK、cooldown | 真正开始 Agent 工作时写 Source Message | 观察 Delivery receipt 后推进自己的 ACK |
| Drift | cursor、journal、continuum、下一轮选择 | 需要 Agent 时写 Source Message | cursor/journal 自己恢复 |
| `message_push` | parent call 的 Tool 意图/结果 | target Session 写 standalone Agent Message | target Delivery 与 parent Tool result 分开 |
| subagent | job/process continuity 与 parent call | parent Session 写 tool_call/result；必要时 child 用独立 Session | 按 call Message 恢复 |

`proactive.db`、`wake_proactive.db`、`drift/drift.db`、`schedules.json`、quota、EventMail
transition 和 pending/ACK/hazard/reservoir state 不能因为 Core 不再识别 proactive 就清除。
迁移前逐项标记 preserve、replace 或 retire，并提供 owner handoff 和恢复证据。

### 6.1 用户指出的交错 case

~~~text
seq 1  Human U1
       stop R1
seq 2  Human U2
       stop R2
seq 3  Agent proactive P
seq 4  Agent A1 responds_to(U1,U2)
~~~

必须同时满足：

- U1、U2 在各自 ingress 时已经 durable，不因 stop 或失败模型请求消失；
- P 没有 Human cause，不能触发 `NoNewHumanInputAfter` 冲突；
- A1 cause CAS 只结算 U1/U2；
- TURN_VIEW 输出 `{U1,U2,A1}` 和 `{P}` 两个 Turn，即使成员不连续；
- Akasha、compaction、history 和 UI 使用同一个 TURN_VIEW 版本，不能各自按邻接角色猜。

这里没有 `proactive=true`、ProactiveTurn 或特殊 writer。P 之所以单独成组，只因为它没有
U1/U2 的 cause 边；相同结构的 command push、subagent notice 或普通 standalone Agent
Message 得到同样结果。

### 6.2 hua-home 历史只提供 fixture，不提供正确答案

2026-09-03 对 hua-home 私有 `sessions.db` 的只读窄查询只取 role 和布尔标记，不取正文、
Session ID 或私人时间戳：共看到 6 条 interrupted marker、分布在 2 个 Session；有 15 条
proactive row 位于“此前 50 seq 内至少有两次 interrupt”的窗口。脱敏相对序列也确实呈现
`interrupt → 后续 user → interrupt → 后续 user → proactive → 后续普通 assistant` 的
交错。它只能证明这种压力形态真实发生过，不能证明旧写法正确。

当前 `session/manager.py:190-230` 仍按**连续** `control_turn_id` 分组，并在
`session/manager.py:213-226` 显式检查 `proactive`；当前
`agent/lifecycle/phases/after_reasoning.py:257-332,413-447` 仍在 reasoning 后把暂存 Human
与 Agent rows 批量 append。v4 有意同时替换这两点：每个 U 在 ingress 单独 durable，P
不再有特判，专门的 Turn plugin 只按 Message 因果与自己的 policy 投影。

这段历史应转成脱敏 fixture：输入只有 `U1/interrupt/U2/interrupt/P/A1`，断言 WAL 是六个
动作中的四条完整 Message、Turn groups 是 `{U1,U2,A1}` 与 `{P}`；旧 marker、attempt 和
`control_turn_id` 都不是 expected output。

## 七、Delivery 是按 Message 与 sink 建立的外部效果 projection

一条 Message 可以同时送到 Telegram、邮件和手机通知，不能为了不同目的地复制三条
对话 Message。Delivery key 是：

~~~text
DeliveryKey = (message_id, sink_id)
~~~

Delivery scan 不表示“看到 Message 就发送”。初始 eligibility 是纯 projection：

~~~text
Deliverable(m, S) =
  m.author == Agent
  and m 有 user-visible final content
  and m 不含 tool_call/no_reply/tombstone
  and not Hidden(m, S)
~~~

因此 Human、Source、Tool result、tombstone、no_reply 和带 tool_call 的中间 Agent Message
都不创建 DeliveryEffect；command reply、user-visible error、standalone proactive P 与普通
最终 Agent reply 使用同一规则。这里没有 `deliverable/proactive` Message 字段。具体送到
哪些 sink 再由 Delivery plugin 的 route/user setting projection 决定；没有 sink 就没有
effect record。

`sink_id` 是 Delivery plugin 自己的稳定地址引用，不是对话身份，也不是随机 DeliveryId。
每个 effect record 不保存正文：

~~~text
DeliveryEffect[(message_id, sink_id)] = {
  immutable sink/provider binding,
  state: prepared | provider_started | delivered | rejected | unknown | suppressed,
  provider_receipt?
}
~~~

发送时只按 message_id 从不可变 Session WAL 读取正文；effect record 不复制 seq、正文或
digest。Message 缺失/损坏是 incident，不能用 effect 中的副本降级。状态单调：

~~~text
prepared ──fsync──▶ provider_started ──▶ delivered
    │                         ├─────────▶ rejected
    │                         ├─────────▶ unknown
    │                         └─────────▶ suppressed  # query 证明尚未产生效果且已 tombstoned
    └──────────────────────────────────▶ suppressed
~~~

- `provider_started` 必须在 provider I/O 前 durable，并在 shared start permit 内提交；
- 永久拒绝写 `rejected`，它是终态，永不重试；
- 只有 provider 能证明未产生效果时，才可在同一 record、同一 message_id/sink/binding 上
  安全重试；无法确认就写 `unknown`；
- Message 已 tombstoned 且 record 尚不存在时，recovery 直接创建 `suppressed`，不经过
  `prepared`；
- effect 已先开始时，tombstone 不伪装成撤回；worker query 后写 delivered/unknown，或在
  证明未发送时写 suppressed；
- unresolved record 固定 exact channel generation；GC 同样受 durable effect 引用约束。

Delivery 不再有 `settled`。`delivered` 只说明 provider effect；Wake 的 ACK、dedupe、
cooldown 或 Scheduler 的 run state 是另一个 owner 的状态：

~~~text
DeliveryEffect.delivered
          │ typed receipt
          ▼
Wake 自己推进 ACK/cooldown
~~~

Wake 失败不能倒退或改写 Delivery，Delivery 也不解释 Wake 是否完成。

`SessionRoute` 可以是 Channel 的默认 sink discovery 输入，但不是“一 Session 只能有一个
目的地”的限制。广播增加 sink effect，不增加 Message。

## 八、逻辑删除仍由 Message 承载，但只保存 seed

本设计不增加 `SessionRecord = Message | Tombstone` 第三种权威记录。用户明确要求隐藏时，
`TombstoneWriter` append 一条只有 tombstone content 的治理 Message：

~~~text
seq 20  Human  tombstone(seed_message_ids=[M7])
~~~

它不是聊天气泡，也不进入模型上下文；但它确实是“用户向 Session 提交了一项治理要求”，
因此仍由唯一 Message carrier 保存。writer 的认证和权限证明不放进 Message 字段。

不再把完整 descendant closure 展开进 tombstone。projection 每次从不可变因果边计算：

~~~text
Hidden(S) = DescendantClosure(NormalizeSeeds(S), CausalEdges(S))
~~~

边包括 `responds_to` 与 `resolves`。seed 归一化只有一个向上规则：如果用户点中 Tool
result，server 把 seed 改成包含对应 tool_call 的 Agent Message；然后只向后展开。这样
不会留下看似 pending 的 call，也不需要保存所有未来 result ID。

归一化可能扩大可见影响，所以顺序必须是：可信 Data Management 边界先算 normalized
seeds 和当前可见影响预览，用户确认这组 exact seeds 后，才签发一次性 TombstoneWriter。
append 只保存确认过的 seeds；预览和当时的 closure 都不成为第二份持久真相。
seed 只能指向 conversation Message；指向 tombstone、未来/跨 Session Message 或重复 seed
都在签发前拒绝。

晚到的 Tool result 或 Agent continuation 会自然成为已隐藏 call 的 descendant。Next
action 先用完整历史 settlement 边判断 cause/call 已处理，再应用 Hidden；删除可见回复
不会让旧输入或 Tool 复活。

Tombstone 与 effect start 按第 5.5 节只争夺一个短临界区，不等待整个远程 effect 才让 UI
隐藏。若 effect 已先开始，它诚实结算并被 projection 隐藏；若 tombstone 先提交，effect
永不开始。

### 8.1 物理擦除不属于初始 v4

初始 v4 只提供 append-only logical tombstone，不 UPDATE/DELETE 既有 Message body。
物理 purge 需要另外回答 projection 枚举、artifact 引用、外部 receipt、备份、恢复、旧
重试和法规证明，不能作为被动链插件化的阻塞依赖，也不能在本文假装已经闭合。

以后若批准 purge，必须另立 Data Purge 设计和操作 Gate；上下文裁切、容量优化、插件
卸载或普通 migration 永远无权触发。

## 九、Thin Core 与信任边界

Core/受保护 substrate 只保留来源无关且不能安全交给产品算法的原子能力：

1. **Plugin composition**：ServiceKey、provide/require/inject、candidate/stable、exact Root
   lease、Fiber/Effect 清理、health/incident；
2. **Session Message WAL**：read/feed/internal append、message identity、seq、幂等、cause
   precondition 与 typed writers；
3. **短命执行安全**：Reaction factory、取消、timeout、有界资源、per-session admission 和
   effect-start permit；
4. **真实外部边界**：Model transport、ToolHost、Channel ingress 与 Delivery effect port；
5. **类型化观察**：只发布 committed Message 与短命 receipt，observer 无权修改事实。

Core 不出现：

~~~text
passive / proactive / Wake / Scheduler / command
compaction / memory / Citation / Meme / Tool Search
某个 Agent、model、provider、tool、channel 或 plugin 名称
~~~

Turn projection、Agent Program、Prompt policy、source state 和 delivery selection 都由普通
插件组合。

“普通插件”不表示任意 Python 代码天然安全。当前 in-process plugin 是用户 install 时
授权的可信代码；object capability 减少误用，不能充当恶意代码 sandbox。不可信第三方
能力必须进入受 OS 文件系统、网络和进程权限限制的 MCP/Workload/out-of-process host。

公共 `PluginRuntime` 还需在迁移前盘点并移除 broad workspace/私有 store/SQL 暴露。正式
Gate 必须从仓库外源码走 install → candidate → stable → hot reload → uninstall，并测试
typed writer、ToolHost、workspace path 和 candidate production effect 拒绝。

## 十、完整目标链路

~~~text
Channel adapter
  │ artifact durable + authenticated canonical message_id
  ▼
InboundWriter ───────────────▶ Session Message WAL ───────────────┐
                                      │ committed feed             │
                     ┌────────────────┼───────────────┐             │
                     ▼                ▼               ▼             │
              TURN_VIEW fold     ToolHost scan   Delivery scan     │
                     │                │               │             │
                     ▼                │               ▼             │
         passive-conversation         │       Effect[message,sink] │
         private reconcile            │               │             │
                     │                │        provider I/O         │
                     ▼                │                             │
           create short Reaction      │                             │
                     │                │                             │
                     ▼                │                             │
               AGENT_PROGRAM          │                             │
          prompt / model / tool choice│                             │
                     │                │                             │
                     ▼                │                             │
           AgentReplyWriter ────────┴────────────────────────────▶│
                     │ tool_call committed                          │
                     ▼                                              │
              scoped ToolPort                                       │
                     ▼                                              │
               protected ToolHost                                   │
                     │ ToolResultWriter                              │
                     └──────────────────────────────────────────────▶│
                                                                    ▼
                                                       next committed feed
~~~

这条链只有 Message commit 能改变对话事实。Turn、Reaction、stream、feed watermark 和
receipt 都可以丢失或重建。

## 十一、从 DSH 借什么

检查的 DSH 基线是
`/mnt/data/source-code/deepseek-harness@49a606bc5b5934603f22a26957a07dc799ab0291`。

只借三条原则：

1. 一份 ordered Session log 是 history 与 projection 的共同来源；
2. immutable Message representation 跨 history、model request 和 delivery 复用；
3. runtime scope、service injection、Fiber/Effect 是短命执行结构，不进入 Session。

对应源码证据：

- `packages/llm/llm/src/message.ts:130`：immutable Message 跨消费者复用；
- `packages/llm/llm/src/message.ts:175`：Message 在 publication 前有稳定 ID；
- `packages/core/session/src/index.ts:628`：Session 单调 append；
- `packages/core/session/src/index.ts:772`：model messages 从 Session surface 派生；
- `packages/session/session-projection/src/index.ts:40`：projection 按 seq fold。

不照抄 DSH 的 SessionEvent/turn/step runtime。Akashic 的 Message 自己就是 WAL record。

## 十二、迁移按纵向切片，不一次切全部 owner

本文是总设计，不是一个实现 PR。正式实现必须拆成独立任务合同；物理 purge 另立设计。
每一刀都保持一个生产 writer，不长期 dual-write。

### Phase 0：批准破坏性语义

- 将“每条 Human 输入立即进入 Session”取代 completed transcript batch；
- 将 Turn 从持久 identity 改成普通 `TURN_VIEW` projection；
- 批准 author + responds_to schema、cause CAS 与默认 `NoNewHumanInputAfter`；
- 批准 typed writers、ephemeral Reaction/receipt 和 trusted message ID namespace；
- 批准 protected ToolHost、WAL generation reachability 与 multi-sink Delivery key；
- 批准 tombstone seed-only closure，物理 purge 延后；
- 分别批准各 source state 的 preserve/replace/retire；
- 逐项勘误或 supersede projectneed、0025、0034、0039、0050；核对 0052 的 compaction/
  Markdown ordinary-plugin owner 与新 TURN_VIEW/CONTEXT_VIEW 边界，不用“等”掩盖冲突；
- 建立全量备份与 schema lineage。

### Slice A：新 Message WAL 在旧 Agent 算法下运行

- 新 schema 支持 author、responds_to、typed content 和 canonical message_id；
- Channel 用 InboundWriter 立即 append Human U；旧 pipeline adapter 只读 WAL，不再批量补写 U；
- 旧 Agent 输出通过 AgentReplyWriter append；同一时刻只有新 writer 生效；
- 用 `U1/stop/U2/stop/P/A1` fixture 证明物理顺序与正文不丢。

### Slice B：Turn projection shadow 接管所有分组消费者

- `TURN_VIEW` 从 responds_to/resolves 构图，并 shadow 对比 UI、history、compaction、Markdown
  和 Akasha；
- 新数据不再依赖 control_turn_id；旧数据只在迁移 adapter 中读取已有 ID；
- 非连续 `{U1,U2,A1}`、standalone P、tool chain 与 incomplete group 均有 fixture；
- 全部消费者切换后删除各自的邻接角色/control ID 猜测。

### Slice C：ToolHost 与 Message recovery

- tool_call 先 commit，ToolHost 后执行；result 只由 ToolResultWriter 写；
- startup snapshot-to-feed、provider query/idempotency/unknown 和 generation reachability
  先通过 crash fixture；
- 再删除旧 Attempt/tool item 作为恢复真源的路径。

### Slice D：被动大链插件化

- 先冻结 command、Prompt、compaction retry、Tool Search、empty reply、terminal tool、
  continuation、Citation/Meme、stream、Memory、delivery/ACK 的行为 oracle；
- 抽出普通 AGENT_PROGRAM 与短命 Reaction；
- 接入 `passive-conversation` 的 private reconcile 和 no-gap feed handoff；
- 用仓库外 Agent Program 完成真实回复，证明无需修改 Core/WAL/Channel；
- 最后删除 `PassiveMessageWorker → ConversationRuntime → AgentLoop._react →
  PassiveTurnPipeline` 的固定业务编排和无消费者 phase/hook；原链拥有的 ingress custody、
  attachment durable、per-session admission、cancel 和资源清理由前三个 slice 明确迁到
  Channel/受保护 substrate 后才可删，不能随类名一起丢掉。

### Slice E：Delivery 与 source handoff

- 迁移到 `(message_id,sink_id)` effect，不复制正文、不保留 settled；
- 冻结 terminal/command/error/proactive eligibility oracle；证明 Human/Source/Tool/no_reply/
  tombstone/tool-call intermediate 都不被默认投递；
- Channel、Scheduler、Wake、Drift、EventMail、message_push、subagent 逐项 handoff；
- 只有旧 owner 的 pending/ACK/cursor/journal 全部可恢复后才发布新 generation；
- 真实 provider 验证多 sink、ACK loss、unknown、rejected 和 tombstone race。

### Slice F：逻辑 tombstone

- TombstoneWriter 只保存 normalized seeds；
- Chat、TURN_VIEW、Context、Memory、Delivery 与 ToolHost shadow 计算同一 Hidden closure；
- 切换后删除旧 interaction-delete 依赖，但不物理擦除正文；
- Data Purge 等维护者另行批准后再设计、实现和演练。

## 十三、验收 Gate

### 13.1 事实与 schema Gate

- Core conversation schema 只有 Session 与 Message；
- 没有 TurnId、RunId、AttemptId、SessionEvent、ReactionId、DeliveryId 或 ProjectionId；
- Message 使用 author，不持久化 provider `system` role；
- responds_to 是结构字段，不在 content blocks 中；
- tool_call 身份直接复用所在 Agent message_id，没有 ToolCallId/call_ref/block-index identity；
- canonical message_id 由认证 namespace 产生/验证，跨 Session/账号碰撞 fail-loud；
- 正常路径只 INSERT Message；初始 v4 没有物理 purge。

### 13.2 并发与 Turn Gate

- U1、U2 在 ingress 时立即 durable，interrupt 不减少或延后它们；
- U2 先 commit 时，`NoNewHumanInputAfter` 拒绝只读过 U1 的旧 A；
- proactive P 先 commit 时，不拒绝 A1(responds_to U1,U2)；
- TURN_VIEW 把非连续 U1/U2/A1 投影成一个 Turn，把 P 投影成独立 Turn；
- 时间线仍按 seq 显示 U1/U2/P/A1，Turn grouping 不重排 Message；
- Tool 等待期间插入 U2/P 时，默认 projection 得到 `{U1,M1,U2,T1,A2}` 与 `{P}`；
- seq22 的 U2 在 C1 unresolved 时是 deferred cause，不会先得到 A(U2)；T1 后只签发
  `Reaction(T1,U2)`；
- cause 已被另一 Agent Message 结算时，旧 writer conflict；
- 只有显式选择 `HeadEquals` 的算法才因任意 head 变化重算；
- 替换 Turn plugin 不迁移或改写任何 Message。

### 13.3 Capability 与插件 Gate

- 没有公开 `MESSAGE_REACTOR` 或通用 MessageWriteGrant 配置袋；
- Agent Program 只拿短命 Reaction，不拿裸 SessionStore、author、cause 或任意 append；
- Inbound/AgentReply/ToolResult/Source/StandaloneAgent/Tombstone writer 的越权调用均
  fail-loud；
- receipt、Reaction、Root lease、watermark 和 permit 均无 ID、不可序列化；
- ToolReceipt 只短命引用 call/result Message，不含正文或可恢复 Tool 状态；
- Core/Bootstrap 不再固定构造默认 Agent 算法或 PassiveTurnPipeline；
- Core 不按 passive/proactive/Wake/Scheduler/Citation/Meme/provider/plugin 名称分支；
- 仓库外 Agent Program 经正式 install/candidate/stable 链完成真实回复；
- trusted in-process 与 out-of-process sandbox 在报告中明确区分。

### 13.4 Feed 与恢复 Gate

- Message commit 后、subscriber callback 前 crash，重启扫描仍会 reconcile 该 Session；
- scan 与实时 feed 重叠只重复唤醒，不重复 Message/effect；
- consumer 追到 snapshot watermark 和当前 head 前不报告 ready；
- ToolHost 从 WAL 恢复 unresolved call，不依赖 Agent Program 存活；
- generation GC 在 call commit、observer 前 crash 时仍拒绝删除 exact binding；
- provider 无 query/idempotency 时写 unknown，不盲重试。

### 13.5 Prompt、Tool 与行为 Gate

- Source Message 永不映射为 system/developer instruction；
- Prompt slots 封闭，重复 owner/语义冲突 fail-loud，不用任意 priority 排序；
- 不存在通用 ASSISTANT_TRANSFORMS；Citation/Meme/媒体由 Agent Program + Tool 显式产生；
- Tool visibility 不能扩大 ToolHost authorization；
- 初始 v4 每条 Agent Message 至多一个 tool_call，unresolved 时不启动 continuation；
- command、attachment、compaction retry、Tool Search、empty reply、terminal、continuation、
  stream、error/no_reply 与 Memory 都有目标 owner 和 fixture；
- 停用 passive/Agent Program 后输入同步仍工作，committed Tool 仍由 ToolHost 结算。

### 13.6 Delivery、source 与 tombstone Gate

- 同一 Message 可向两个 sink 发送而不复制正文或 message_id；
- `(message_id,sink_id)` 同 binding 幂等，binding 冲突 fail-loud；
- Delivery eligibility 只由 author/content/Hidden 投影；中间和非 Agent Message 不发送；
- rejected 永不重试，unknown 不伪装 exactly-once；
- Delivery delivered 后不等待或写 Wake settled；Wake 自己推进 ACK/cooldown；
- tombstone 先于 effect start 时 provider 不被调用；effect 先行时结果诚实结算并被隐藏；
- 只选 Tool result 时 seed 归一化到 call Message；晚到 result 自动进入 Hidden closure；
- 隐藏 Agent reply/tool result 不会让旧 cause/call 重跑；
- Scheduler/Wake/Drift/EventMail/Channel/subagent/message_push 的 state 都有 owner、备份和
  handoff 证明。

### 13.7 迁移 Gate

- 每个 slice 只有一个正式 Message writer，没有窗口式 dual-write；
- Phase 1 行为账单逐项标成 preserve、intentional replacement 或 proven old bug；
- 旧 control_turn_id/attempt/delivery ledger 的每条数据都有转换、保留或人工阻塞结论；
- WAL、Turn view、ToolHost、Delivery、source state、client cursor 和 generation artifact
  都从备份完成恢复演练；
- 删除旧链前核对静态 import、动态插件、cache、测试、数据库、日志和运行进程消费者；
- Gate 证据绑定 exact clean commit/tree；concept Gate 的 must-fix 清零。

## 十四、明确接受的代价与仍需批准的问题

v4 明确接受：

- Session 不审计失败 provider attempt；
- 不恢复半截 token；
- interrupt/cancel 不拥有持久身份；
- Turn grouping 可以随 Turn plugin 版本改变；
- 无幂等/query 的外部效果只能得到 unknown；
- 初始 v4 只有逻辑 tombstone，没有物理擦除；
- in-process Python plugin 是可信代码，不冒充 sandbox。

维护者仍需决定：

- 默认 Turn plugin 是否永远不混合 Human 与 Source causes；
- multi-sink 的 sink discovery、用户配置 owner 与未来 eligibility 扩展流程；
- 各 source plugin state 的 preserve/replace/retire 清单；
- author 是否在未来多 Agent/多人 Session 中增加 stable principal ref；当前 v4 不预建。

## 十五、相对 v3 与前一版 v4 的最终减法

| 删除/替换 | 原因 | v4 归宿 |
|---|---|---|
| Turn/Run/Attempt 持久实体 | 重复 Message 已能表达的事实与恢复 | 普通 TURN_VIEW + 短命 Reaction |
| completed transcript batch | 会让 U1/U2 在完成前不属于 canonical Session | ingress 立即 append 每条 Human Message |
| 全局 mandatory head CAS | 把无关 append 与回答有效性绑死 | cause CAS + typed optional read condition |
| 公开 MESSAGE_REACTOR | 只有插件自己一个 consumer | `passive-conversation` 私有 reconcile |
| 裸 `respond(session_id,cause_ids)` | 反复传可漂移身份并给程序宽能力 | `AGENT_PROGRAM.react(Reaction)` |
| 通用 MessageWriteGrant | 会长成 mode/role/variant 配置袋 | 分型短命 writer |
| persistent system role | 混淆来源与模型指令权限 | author + provider request projection |
| reply content block | 混淆正文与 cause settlement | Message.responds_to |
| client_message_id + message_id | 两份身份会让重试和引用分叉 | 认证边界产生一个 canonical message_id；seq 只管顺序 |
| ToolCallId / `(message_id, block index)` call_ref | 每条 Agent Message 只有一个 call 时是重复身份 | 直接复用 call Message 的 message_id |
| 普通插件拥有 TOOL_EXECUTOR 安全边界 | grant/binding/唯一结果不可随算法替换 | protected ToolHost + ordinary Tool plugins |
| ASSISTANT_TRANSFORMS | 共享可变正文和顺序耦合 | Agent Program 直接产出 final typed content |
| DeliveryEffect[message_id] 单 route | 对话事实与投递拓扑耦合 | DeliveryEffect[(message_id,sink_id)] |
| Delivery 中复制 seq/body digest | immutable Message 已拥有顺序和正文 | effect 只引用 message_id |
| Delivery settled | 跨 owner 解释 Wake/Scheduler 是否完成 | Delivery terminal receipt → source 自己结算 |
| 完整 delete closure manifest | 阻塞删除且遗漏未来 descendant | tombstone normalized seeds + dynamic closure |
| v4 内物理 purge | projection/artifact/恢复 owner 尚未闭合 | 独立 Data Purge 设计 |

最终只剩四句话：

~~~text
Session = Ordered(Message)
Pending = Fold(Messages)
Turn = PluginProjection(Messages)
Commit = CausesStillOpen + AuthorityValid + OptionalReadCondition
~~~
