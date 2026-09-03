# Akashic v4：Session 是账本，其他都是投影

- 文档版本：0902-reviewed-v4
- 日期：2026-09-03
- 状态：设计提案，等待维护者批准
- 当前代码基线：47896b4200731183a54081e2eca77602a0881a0a
- 输入：0902-reviewed-v3.md、0902-02.md、Codex 历史会话、当前项目合同、真实运行案例、DeepSeek Harness
- 本文不授权：实现、数据库迁移、正式 workspace 写入、删除、部署或合并

## 结论

v4 只设一个消息真源：

~~~text
Session = 一份只追加的事实账本

一条完整 Message 的正文只在 Session 中记一次
Turn / Run / Transcript / Model History / Mobile / Memory 都从 Session 折出来
Delivery 只引用 Message，不复制 Message
~~~

前一版 v4 的 `MessageBody + SessionEntry + SessionReceipt + ProjectionClaim` 仍然造了多条路，应该删除。

最小主线变成：

~~~text
Envelope / Draft
       │ 只有接纳或封口时才成为 Message
       ▼
Session.append(events)
       │
       ├── Turn projection
       ├── Run projection
       ├── Transcript projection ──▶ Web / Mobile
       ├── ModelHistory projection ──▶ REACT
       ├── Memory projection
       └── Delivery projection ──▶ provider
~~~

Core 不认识 passive、proactive、Wake、Scheduler、Spawn、Content、Drift 或插件名。它只保证 Session 事件合法、有序、可恢复，以及当前 Run 才能提交结果。

### 把我当六岁

把 Session 想成唯一一本不能偷偷改页码的作业本：

~~~text
作业本 Session
┌──────────────────────────────────────────┐
│ 10 这件事开始了                         │
│ 11 花月哥哥说：U1                       │
│ 12 第一次尝试开始                       │
│ 13 第一次尝试被打断                     │
│ 14 花月哥哥补充：U2                     │
│ 15 第二次尝试开始                       │
│ 16 Akashic 回答：A                      │
│ 17 第二次尝试完成                       │
│ 18 这件事完成                           │
│ 19 把 U1 放进聊天页                     │
│ 20 把 U2 放进聊天页                     │
│ 21 把 A 放进聊天页                      │
└──────────────────────────────────────────┘
         │             │             │
         ▼             ▼             ▼
      Turn 卡片      手机聊天页      模型上下文
~~~

作业本里的事实只有一份。Turn 卡片、手机聊天页和模型上下文只是用不同彩笔画出的视图，丢了可以再从作业本画出来。

- Message 是作业本里真正写下的一句话。
- Turn 是把“同一件事”的页码圈在一起，不是另一本本子。
- Run 是这件事实际试了第几次，也不是另一本本子。
- Transcript 是用户看到的聊天页。
- Delivery 是邮差的工作状态，只记“送哪一句、送到哪、送成没有”，不再抄一遍信。

U1 后被打断，再收到 U2，仍是同一个 Turn，只多一个 Run。Wake 检查后没事，连作业本都不用动；Wake 已经开始工作后决定不说话，则账本里有一个完成但没有可见回复的 Turn。

### 本版五个决定

| 问题 | v4 决定 |
|---|---|
| Message 在哪里 | 只在 Session 的 `message/append` 事件里一次 |
| Turn 和 Run 是什么 | Session 事件的稳定分组投影，不是第二套正文存储 |
| 是否新增随机 MessageId | 不新增；新 MessageRef 由 SessionId 与 message event seq 派生 |
| 手机如何同步 | 仍只认不透明 `message_id`、单调 `seq` 和已有 cursor |
| proactive 是什么 | 来源插件的一种触发故事，不是 Core 类型、字段或状态机 |

---

## 一、先确定什么是真源

### 1.1 唯一消息载体

下列规则是本设计的起点：

1. 还没进入 Session 的输入叫 `Envelope`，不是 Message。
2. 还没封口的模型输出叫 `Draft`，不是 Message。
3. Message 一旦创建，身份与元数据不可变；正文只存在于该 Session event 的可擦除 body slot。
4. 任何其他模块只能保存 `MessageRef` 或可重建 projection，不能保存第二份正文。
5. 模型看见过的 Message、Tool 结果和请求绑定都必须能从 Session 重建。
6. 正常运行只追加；只有用户明确删除时，Data Management 才能按受控协议擦除正文。

这里的“一次”指一条已经组装完成的 canonical Message。网络分片和模型 token delta 只能短暂直播，不能写入 Session、trace、request 记录或第二张流式表；crash 后丢掉未封口 Draft，不能拿半截 token 冒充 Message。Tool call 参数是另一类事实，只在 `tool/call` 写一次；Tool result 正文只在它对应的 `message/append` 写一次。

这与“Session 是用户当前看见的聊天列表”不同。Session 是完整账本；聊天列表只是其中一个 projection。未完成输入、内部 task、Tool 结果和投递状态可以在账本里，但不必出现在聊天页。

### 1.2 不是所有事实都是 projection

“其他都是 projection”只适用于 Message 的各种读法，不能抹掉两个真实外部边界：

- 来源插件在 admission 前拥有自己的 trigger attempt、due、cursor 和业务去重。`no_due` 时根本没有 Session Message。
- Tool 或 provider 已经在外部世界产生的效果由对应 owner 的 effect ledger 最终确认。Session 只记录请求与观察到的结果，不能把 Git 回滚或内存回滚说成外部效果没发生。

这两类事实都不能复制 Message 正文。它们只在成功进入 Session 后保存稳定 EventRef。

### 1.3 当前行为只当反例库

hua-home 私有历史里的多 Run、interrupt、retry、Wake `no_due`、model skip 和 shared 只用于构造场景。当前表名、字段和分支不自动升级成正确设计。

若当前行为与以下模型冲突，应登记为 migration delta，再由维护者决定修改 accepted 合同或修改本提案。

---

## 二、从 DSH 借什么，不借什么

DeepSeek Harness 当前源码提供了四个重要证据：

1. `Session` 是 typed `SessionEvent` 的 append-only log，是完整交互历史唯一真源。
2. 模型历史由 `deriveMessages()` 从 log 的 surface 投影，不单独保存。
3. `turnBoundary` 和 `turnOutline` 都是对 `turn/start`、`turn/end` 与消息事件的纯 fold。
4. 纯 projection checkpoint 只保存 `(session, key, version, source seq, value)`；版本不符或越过 log 末尾就丢弃重建。

v4 借这条骨架：

~~~text
durable Session events
        └── pure / ordered folds
             ├── runtime state
             ├── model surface
             └── client views
~~~

但不照抄三个 DSH 选择：

| DSH 当前选择 | Akashic v4 选择 | 原因 |
|---|---|---|
| `createMessage()` 先生成随机 UUID | Message 进入 Session 时由 event coordinate 得到身份 | Akashic 不需要让未接纳 Envelope 冒充 Message |
| durable inbox splice 可再次携带整条 Message | Message 正文只写一次，后续事件只引用 MessageRef | 避免同一 log 内也复制正文 |
| 单一 turn 顺序足够 | 多个 Turn 可以交错，但每个事件显式引用 TurnRef | message_push/task 不应占住会话的 history-reading lane |
| Session fork 是现成能力 | v4 不引入 durable fork | 避免在未定义 lineage、删除和 retention 前制造跨 Session Message graph |

如果未来确实需要 durable inbox，入队动作本身就追加 `message/append`；后续 `turn/input` 只引用它，不再携带正文。先证明这个消费者存在，再增加该事件。

DSH 在这里是“单日志 + pure fold”的参考，不是 Akashic 多 Turn 并发、异步外部 projection、删除或外部 effect 正确性的证明。尤其 DSH 的 projection contract 是同步纯函数；有网络 I/O 的 materializer 必须另外解决 started-but-unknown，而不能套一个 checkpoint 就宣称安全。

---

## 三、最小权威模型

### 3.1 SessionEvent 是唯一坐标系

~~~text
SessionEvent = (
    session_id,
    event_seq,
    type,
    data
)

EventRef = (SessionId, event_seq)
~~~

规则：

- `event_seq` 在一个 Session 内从 0 连续增加，永不复用。
- Session Store 只接受原子 `append_batch`；同一批事件先整体校验，再一起提交。
- 插件可以扩展事件词汇，但必须声明 producer、consumer、重放规则和未知 reader 是否可忽略。
- 会改变重建语义的未知事件默认 fail-loud，不能静默跳过。
- EventRef 是位置，不是 capability；读、写、删除仍检查 Session 与调用者权限。
- 写入某个 Session event 的所有 `TurnRef`、`MessageRef`、`RunRef`、`StepRef`、`ToolCallRef` 和 `DeliveryRef` 必须指向同一个 Session。Store 集中拒绝跨 Session ref。
- 只有 source/effect ledger 可以从外部保存一个 Session EventRef；Session 内部不会反向引用另一个 Session 的事件。

所有新领域引用都只是 EventRef 的类型化名字：

~~~text
TurnRef      = ref of turn/open event
MessageRef   = ref of message/append event
RunRef       = ref of run/open event
StepRef      = ref of step/open event
ToolCallRef  = ref of tool/call event
DeliveryRef  = ref of delivery/prepared event
~~~

因此 v4 不新增随机 TurnId、MessageId、RunId、AttemptId、StepId、ReceiptId 或 DeliveryId。SessionId 已经提供全局作用域，event_seq 提供 Session 内唯一位置。

### 3.2 Message 只出现一次

~~~text
message/append = {
    turn: TurnRef,
    role: user | assistant,
    body: Live {
        content: ContentBlock[],
        attachments: AttachmentRef[]
    } | Gone { redaction: EventRef },
    producer:
        Admission { admission_token: AdmissionToken }
      | Run { run: RunRef, step: StepRef,
              output: model_output | tool_result { call: ToolCallRef } },
    private_slot:
        AdmissionOnly(
            Live { source: SourceRef, reply_target?: ReplyTarget }
          | Gone { redaction: EventRef }
        ) | NoneForRun
}
~~~

MessageRef 就是这条 `message/append` 事件的 EventRef，不在 payload 里再写一次 id。初次 append 只允许 `Live`；`Gone` 只可能由第八章的显式 redaction 原子转换得到。

规则：

- human 输入、插件 task 输入、模型 assistant、Tool result 都用同一个不可变 Message 结构。
- Admission 只能由 admission capability 写，不能携带 RunRef；它把 source-scoped admission 与可选回复目标固定下来。原始 source/target 是可擦 privacy slot，immutable admission token 只是 keyed digest，不能反解地址。
- Admission 不是绕过 Run fence 的旁门：初始 Turn 遵守 3.4 的同批规则；已有 open Turn 的新 Admission 只允许作为 5.2 continuation batch 的一部分，不能裸 append。
- 模型输出与 Tool result 只能由 current Run capability 写，必须携带同 Session、同 Turn 的 RunRef 和 StepRef。Store 在同一写锁内验证 Run 仍 active；旧 Run 的迟到结果没有可绕过的字段。
- Tool result 对模型仍是 user-role Message，但 `producer.kind=tool_result`，对应 `tool/call` 只保存参数和这条 MessageRef，不再保存 result body。
- stream chunk、thinking、ACK、Prompt section、turn/run 边界和普通 debug span 不是 Message，也不得持久化 Message 正文。直播 chunk 只存在于当前进程和连接；封口后客户端改读 canonical Message。
- 输入附件在同一个 append batch 里绑定；输出附件在 Message 提交前只是 staging object。
- Message 只能属于一个 Session；向该 Session 之外发送由 Delivery 引用它，不复制它。

如果相同正文出现两次，就是两条 Message。content hash 不是身份。

### 3.3 wire 上仍叫 message_id

内部 MessageRef 永远只有一种形状：

~~~text
MessageRef = ref of message/append EventRef
~~~

wire `message_id` 是边界表示，不是第二种内部身份：

~~~text
WireMessageIdOut = encode_v4(MessageRef)
WireMessageIdIn  = encode_v4(MessageRef) | LegacyWireMessageId

identity/legacy-token = {
    message: MessageRef,
    token: (key_version, HMAC(key_version,
                              "legacy" + session_scope
                              + canonical LegacyWireMessageId))
}
~~~

- 新消息只输出版本化、不透明的 `encode_v4(MessageRef)`，不再有独立随机 UUID。
- 迁移旧 Message 时，在同一 append batch 追加一条一对一 token mapping event；Session 不保存 raw old id。LegacyIdResolver projection 可由 token events 重建。
- v4 full snapshot 对旧消息也输出 `encode_v4(MessageRef)`。旧 id 只在 API 入站兼容期被接受：边界用 active/retained identity keys 计算候选 token，解析成 MessageRef；Core 之后看不到 Legacy 类型。
- resolver 必须先从已经认证的 route/session context 取得 `session_scope`，再计算 token；禁止拿裸 legacy id 做跨 Session 全库查找。
- 同一 legacy token 映到两个 MessageRef、两个 token 映到同一 MessageRef或跨 Session 使用都 fail-loud；不得按角色、时间、正文或相邻 seq 猜。
- 映射所用 identity key version 与 AdmissionToken 遵守相同 keyring/退休规则；SessionGone 前仍能把旧 id 解析为原 MessageRef 或 Gone，却无法从 token 反推出 chat identity。
- 客户端不能拆 message_id 获得权限或业务含义。
- v4 feedback、附件、删除和引用都用输出的同一 message_id，不再增加 alias；旧链接只走入口 resolver。
- 来源 transport 的 `client_message_id` 只用于 admission/retry，不是第二个领域 MessageId。

为什么不能只用 wire `seq` 代替 message_id？因为两者回答不同问题：

~~~text
message_id   指向 Session 中那一条不可变 Message
seq          指向 Transcript 最近一次 add/remove 变化的顺序
~~~

删除或重新投影后，MessageRef 不变，但客户端看到变化的 seq 会前进。因此保留两个字段是正交，不是重复。

v4 不定义 durable Session fork。复制到新 Session 会创建新的 Message event 和新的 MessageRef；可记录非权威 provenance，但不得跨 Session 引用正文。若以后要保留 fork identity，必须先用单独 ADR 定义 lineage、父 Session 删除、retention、权限和 wire 语义，不能暗中塞进 EventRef。

### 3.4 Turn 是事件分组，不是 row

~~~text
turn/open  ────────────────────────────── turn/close
    │            │             │                │
 message      run/open      message          outcome
   U1          Run 1          U2/A
~~~

`turn/open` 的 EventRef 就是 TurnRef。属于该 Turn 的 Message、Run、Delivery 和 policy event 显式引用 TurnRef，因此不同 Turn 可以安全交错。

`turn/open` 不是一个可以空放的壳。Store 强制同一个初始 append batch 按 `turn/open → 至少一条 Admission Message → 一个 run/open` 排序；Message 用 batch local Turn handle，Run 只能引用本批已经验证的 Message。没有 Admission Message/Run 的 Turn、先开 Run 后补输入、以及纯内部空 Turn 都被拒绝。`no_due` 留在 source ledger，不能用空 Turn 代替。

Turn projection 只回答：

- 这次逻辑工作有哪些输入和最终输出；
- 它是否仍 open；
- 它最后 completed、superseded、abandoned 或 failed；
- 它包含哪些 Run。

Turn 不拥有 Message body、Root、Delivery、Session 可见性或 projection cache。

`turn/open` 的 TurnGrant 是否包含 `finish_without_output` 也是可重放事实。`turn/close(completed)` 是否需要 final assistant Message 由这个 grant 和事件结构共同决定：

- grant 不含 `finish_without_output` 时必须提供唯一 terminal assistant；
- grant 包含它时，只有先追加一个由 DomainDecisionPermit 授权的 `turn/decision`，才可零 assistant 完成；
- 不保存 `turn_kind` 或 `no_reply` boolean。

### 3.5 Run 是一次执行占用

`run/open` 的 EventRef 就是 RunRef。一个 Turn 可以有多个 Run，一个 Run 可以有多个 Step；一个 Step 是一次模型请求及其 Tool 调用。

~~~text
Turn T
├── Message U1
├── Run R1 ── interrupted
├── Message U2
├── Run R2
│   ├── Step S1 ── tool calls
│   └── Step S2 ── final answer
└── completed
~~~

Turn 打开时一次性固定最大 `TurnGrant`；后续 Run 只能缩权，不能重写这个上限。Run 事件只记录不可推导的执行事实：

- exact plugin Root/generation binding；
- 若读取 Session history，则记录 exact history cut；
- 本 Run 固定的 `reply_to: MessageRef`（没有发送能力时为空）；
- close outcome：completed、interrupted 或 failed。

同一 Turn 的第一次 history-reading Run 固定 base `TranscriptThrough(cut)`；continuation 和 retry 必须复用这个 cut，只增加本 Turn 的新输入与新 Run facts。这样 background transcript change 不会在 U1 与 U2 之间悄悄改变同一次工作的旧历史。fresh Turn 才取得新的 cut。

第一次真实模型请求再追加 exact `request/bound`，并强制引用 current RunRef 与 StepRef。它固定 provider artifact、model、connection、CredentialHandle、roles、CallPermit、prompt artifact refs 与 Tool schema refs，但不复制 history body、rendered Message 或 Tool result。Root 加这些引用必须足以确定地重新构造请求；无法重构的动态 context 必须先作为自己唯一的 Session fact 写一次，再由 request 引用。同一 Run 不漂移；下一 Run 可以使用新的已提交 Root 和 binding。

RunRef 本身就是 fence：

- 每个 Turn 同时最多一个 active Run；
- 同一 Session 同时最多一个带 `history_cut` 的 active Run；
- interrupt、append model-visible facts 和 seal 都必须证明自己仍是当前 RunRef；
- 新 RunRef 永不复用，所以旧 Run 的迟到 seal 必然失败，没有 ABA；
- 不再新增一个内容相同的 fence UUID。

StepRef 用同一条规则 fence Run 内的迭代：

- 每个 active Run 同时最多一个 current StepRef；`step/open` 只在没有 current Step 时成功。
- `request/bound`、`tool/call`、Tool result Message 和 final assistant Message 必须同时匹配 current RunRef 与 current StepRef。
- 一个 Step 可以并行发出多个 ToolCallRef，但每个 call 只有一个 terminal Tool-result Message；有 pending/started call 时，必须先得到真实 result，或提交 structured `outcome_unknown` result，才能 `step/close`。外部 effect ledger 的 uncertain 可继续存在。
- final assistant Message 与 `step/close` 在同一 batch 提交；普通 Tool Step 必须等全部 call terminal 后才能 close。
- close 后 current Step 清空；S2 打开后，S1 的迟到 callback 即使 Run 仍 active也只能命中已有幂等结果或 fail-loud，不能追加新 Message。
- `step/open` 的 EventRef 已经不可复用，不增加 Step fence UUID。

不读取 history 的 task/message_push Run 可以与 conversation Turn 共存，但所有 Session append 仍由 single-writer 事务串行。

### 3.6 Tool 与模型事实也进同一本账

下一 Run 恢复所需的内容直接是 Session events：

~~~text
step/open(S1)
├── request/bound(run, S1, refs only)
├── tool/call(run, S1, arguments once)
├── message/append(tool result, run, S1)
└── step/close(run, S1)
step/open(S2)
├── request/bound(run, S2, refs only)
└── one batch: message/append(final assistant, run, S2)
               + step/close(run, S2)
~~~

`assistant/chunk` 不在 durable vocabulary 中。Tool call 参数只在 `tool/call`；Tool result 正文只在 Message，其 producer 直接引用 ToolCallRef，不再需要 result-link 或 `ToolFact` 第二张表。ModelHistory projection 只使用完整、已经闭合的 call/result 对；未闭合调用在 crash repair 中变成明确 `outcome_unknown`，不能静默重放。

unknown 之后的外部查询只追加无正文状态事实：`tool/effect-resolved { call, happened | not_happened, receipt_ref? }`。显式覆盖则追加 `tool/uncertain-override { call, UserPermitRef }`。二者都不改写原 unknown Message；TurnState fold 用它们清除或受审计地越过 unresolved-effect fence。

Tool 插件自己的 effect ledger 仍独占外部副作用 finality。Session 记录模型见到什么，effect ledger 记录外部世界实际发生什么，两者不能互相冒充。

---

## 四、所有读模型都是 projection

### 4.1 一个统一 projection 规则

每个 projection 都是：

~~~text
State(n + 1) = apply(State(n), SessionEvent[n + 1])
~~~

每个 projection owner 注册：

- 稳定 `projection_key`；
- 初始状态；
- 纯、确定性的 `apply`；
- state schema version；
- 可选 wire view；
- 若持久化，最后成功应用的 `source_event_seq`。

cache/checkpoint 只是一条捷径：

~~~text
(session_id, projection_key, state_version, source_event_seq, value)
~~~

server-side checkpoint 的 `value` 只保存 refs、状态和必要的 derived value，不保存原始 MessageBody；需要展示正文时按 MessageRef 从 Session 读取。schema 版本不符、cursor 越过 Session 末尾或校验失败时，直接丢弃并从 Session 重建。禁止用 cache 反写 Session。

客户端 Room/浏览器状态可以持有用于离线阅读的 materialized body，但它明确是可替换 projection：不能回传覆盖 Session，必须应用 remove/SessionGone。外部 provider 在发送过程中收到正文也是外部效果边界，不因此取得 Akashic Message authority。

### 4.2 必需的 projection

| Projection | 输入 | 输出 | 可否删除重建 |
|---|---|---|---|
| TurnState | turn/run/message events | open Turn、Run、outcome | 是 |
| ModelHistory | transcript cut、当前 Turn message、tool/request events | 下一次模型输入 | 是 |
| Transcript | transcript add/remove + MessageRef | 用户聊天历史 | 是 |
| Mobile/Web | Transcript + TurnState + domain projections | 已提交 wire snapshot/delta | 是 |
| Memory/Embedding | eligible transcript + policy events | 长期检索结构 | 必须可对账、可清理、可重建 |
| DeliveryState | delivery events | pending/uncertain/settled | 是；外部效果仍以 provider receipt 为证 |

Turn 和 Run 因此仍是重要领域词，但不是独立真源。删掉 projection cache 不会删掉 Turn；重新 fold 同一 Session 会得到同一 Turn。

partial Draft 的 token stream 不是 Message projection，也不是 durable history。它是 Run Host 到当前连接的短暂 signal，必须带 RunRef，旧 Run signal 在客户端和服务端都被丢弃。crash 后不能恢复半截 Draft；一旦 seal，所有 UI 都切到 Session 中唯一的 final Message。

#### ModelHistory 的选择规则

`run/open` 固定一种 `HistorySpec`：

~~~text
HistorySpec = TranscriptThrough(session_event_seq) | NoPriorTranscript
~~~

每个 Step 的模型输入只解析 refs，不复制 body：

1. `TranscriptThrough(cut)` 取 cut 时已经 visible 且未 remove 的 Message；`NoPriorTranscript` 取空。
2. 再按 Session event order 重放当前 Turn、当前 Step 之前的 model-visible surface：所有 Admission Message，以及所有更早 Run/Step 已闭合的 `tool/call → tool result Message` 对。
3. replay 跨 Run 保留原因果顺序。例如 R1 tool result、U2、R2 tool result 必须仍按这个顺序出现，不能把全部输入和全部 Tool 结果分成两堆。
4. 未闭合 Tool call 在 interrupt/crash repair 时，先由 current Run capability append 一个 canonical structured `outcome_unknown` Tool-result Message，再关闭原 Step/Run；于是 pair 可按原序编入 ModelHistory，模型明确看见“外部可能已发生”。没有 terminal Message 的 pair 不能 close、不能编入历史。
5. 忽略其他 open Turn 的内部 Message；它们只有先通过 `transcript/add`，并且 event seq 不大于 cut，才会进入后续 history。

因此同 Session 的 background task 可以和 conversation 交错，却不会因为“恰好写在前面”污染聊天模型。Task output 是否成为未来 history 只由 `transcript/add` 决定，不由 source 名、role 或 policy 反推。带 `TranscriptThrough` 的 Run 仍独占 Session history-reading lane；`NoPriorTranscript` 的 Run 只读自己 Turn，可以并发。

TurnState 还从 `outcome_unknown` 派生 unresolved-effect fence。fence 存在时，CallPermit 默认拒绝新的 effectful Tool；无法可靠区分 read-only/effectful 时就拒绝全部 Tool。只有 owner query 后追加明确 resolution，或用户用专门 permit 追加 `tool/uncertain-override`，才可再次产生外部效果。模型自己说“重试”不构成授权。

### 4.3 Transcript 才是“聊天页”

Message 被写入 Session，不等于用户已经看见。可见性由引用型事件表达：

~~~text
transcript/add    { message: MessageRef }
transcript/remove { message: MessageRef, reason }
~~~

这些事件不复制正文。每条 transcript 变化各占一个 Session event seq；conversation seal 可以在一个 append batch 中连续写 U1、U2、A 的三个 add。

Transcript fold 强制每个 Message 只能 `unseen → visible → removed`：不能重复 add、不能 remove unseen、不能在 remove 或 Message Gone 后重新 add。add/remove 与被引用 Message 必须同 Session。

手机和 Web 使用：

~~~text
message_id = referenced MessageRef 的不透明编码
seq        = transcript/add 或 transcript/remove 的 event_seq
cursor     = 客户端已经扫描完成的 Session event high-water
~~~

于是：

- `changes(after=cursor)` 扫描更高 Session events，只返回其中的 transcript changes，并返回本次扫描到的 `next_cursor`；即使没有 delta，cursor 也能前进；
- transcript `seq` 是稀疏但严格递增的 change watermark，客户端不得要求 `seq + 1` 连续；
- upsert/remove 仍按 message_id；
- U1…Un+A 在同一事务一起发布；
- 删除用更大的 seq 到达，不会被旧 cursor 漏掉；
- 不需要 projection_id、row alias 或第三个持久同步身份。

### 4.4 Projection policy 留在 Session，不另建 Grant 系统

“同样的 user + assistant 形状，是否允许写长期记忆”是一个真实独立轴，不能由来源名猜，也不能继续塞进全局 `post_commit` boolean。

最小表达是一条 Session event：

~~~text
projection/policy = {
    turn: TurnRef,
    allow: ProjectionKey[]
}
~~~

- event_seq 本身就是 policy revision，不再生成 grant id。
- policy owner 在 `turn/open` 同一 batch 写入；来源和模型不能扩权。
- 未知 key 默认拒绝。
- 普通 human conversation、内部 validation、user-visible task 可以拥有不同 key 集合，但按“允许什么”表达，不按 Wake/Scheduler 名字表达。
- 撤销时追加更高 seq 的 policy event；对应 projector 顺序应用并清理。

v1 建议只定义当前有真实 consumer 的 key：

| key | 允许的输出 |
|---|---|
| `message_embedding` | 为已发布 Message 写检索 embedding |
| `long_term_memory` | 把 conversation 纳入长期事实图 |
| `profile_memory` | 更新 SELF/MEMORY 类用户画像 |

Transcript 本身不受这张表控制；`transcript/add/remove` 已经是明确可见性事实。Compaction 也不受它控制；compaction 只是 ModelHistory 的无损替换 projection，不能删原文。

不再需要独立 `ProjectionGrant` relation，也不需要 Core 全局 `ProjectionClaim`。纯 fold 只需 source cursor；真正会向 Session 外写入的 materializer 必须在自己的 effect 边界解决不确定性。

### 4.5 异步 projection 如何不迟到写回

先分两类，不能拿一种协议冒充另一种：

1. **纯 fold / 同库 materialization**：apply 无 I/O；若结果与 cursor 能在同一个 SQLite transaction 提交，`last_applied_seq` CAS 足够。
2. **外部 materialization**：embedding、图数据库或远程索引已经越过本地事务边界；owner 必须有窄的 durable effect journal。

每个 effectful projector 自己保存：

~~~text
ProjectionEffect = (
    projection_key,
    subject_ref,              # same-Session MessageRef / TurnRef, or the SessionId itself
    source_event_seq,
    desired_version,
    effect_key,
    state = prepared | started | applied | cleanup_started | cleaned | uncertain
)
~~~

这不是 Core 的通用 Claim，也不保存正文。它只回答一个无法从 Session 推导的事实：“远端这次写到底发生了没有”。协议是：

1. 同一 subject_ref 串行推进 `desired_version`；effect journal 的 `prepared` 必须在网络 I/O 前持久化。
2. 外部 object key 由 `(projection_key, subject_ref)` 派生；operation key 再加 `desired_version`。provider 必须支持幂等写、按 key 查询或条件版本中的至少一种。
3. I/O 前写 `started`；crash 后先 query/retry same key。无法判断时进入 `uncertain`，不能假装没写。
4. 更高 seq 的 remove/revoke/redaction 先提高 desired version，阻止旧 `prepared` 开始；已经 `started` 的旧写必须先被查询或收口，再执行 cleanup。
5. cleanup 使用相同 object key 和更高 desired version。只有远端确认 cleaned、或查询证明对象不存在，owner 才推进 cleanup cursor。
6. provider 不提供幂等、查询、条件版本或可靠删除时，不得承载需要可证明删除的 projection；既有 uncertain 会阻塞删除完成并交给人工处置。
7. Data Management 只有在所有相关 owner 越过 deletion seq，且没有 `started/uncertain` 后才报告完成。

所以迟到写只有三种可诚实处理的结局：先完成再被 cleanup、被高版本 fence 拒绝，或进入明确 uncertain 并阻塞完成。一个本地 CAS 不能证明远端没有发生；v4 不再作这个错误承诺。

---

## 五、Turn 怎样工作

### 5.1 第一次普通输入

Channel 先校验 wire Envelope。MESSAGE_HANDLER 可以拒绝或处理命令；只有 accept 才追加 Session：

~~~text
一个 Session append batch
├── turn/open(TurnGrant)              → TurnRef
├── projection/policy
├── message/append(U1, Admission {
│       source, reply_target })        → MessageRef
└── run/open(history_cut, Root,
             reply_to=U1)              → RunRef
~~~

逻辑 `AdmissionKey = (source_owner, source_ref)` 必须包含 owner 所需的 channel/session scope，不能假设裸 client id 全局唯一。持久索引不保存 raw key，而保存 `AdmissionToken = (key_version, HMAC(key_version, "admission" + canonical AdmissionKey))`，由 Session Store 在边界计算并在 append 时强制唯一。固定 domain tag 防止两类 token 互相碰撞。

相同 AdmissionToken 与相同已校验 payload 重投时返回原 MessageRef；同 token 不同 payload fail-loud。Message 已被用户删除时，token tombstone 返回 Gone，不能重新创建正文。不能把可逆 source_ref 偷放进 token。

key rotation 也属于 identity contract：Store 对 raw AdmissionKey 用 active 与尚未退休的 key versions 计算候选并一次查询；任一候选命中就返回原 MessageRef/Gone，只有全部未命中才用 active version 新建。只要某版本仍有 live token 或 tombstone，其 HMAC key 就必须保留在受备份的 lookup keyring；整个 Session 已进入 SessionGone 且该版本再无引用后才能退休。轮换前后 redelivery 必须命中同一结果，不能借换 key 复活 Message。

若 source transport 必须 durable 地保管 Envelope，handoff 采用前向恢复：先以 AdmissionKey append Session，再把 source attempt 原子替换成 `admitted(MessageRef)` 并 ACK/擦除 Envelope body。若两步之间 crash，重试第一步只返回同一 MessageRef，再完成 cleanup；source copy 在此期间只是不可投影、不可寻址的 transport recovery buffer，不是第二个 Message。它不能在 admitted 后按自己的正文重新驱动 Turn。

### 5.2 interrupted continuation

~~~text
Turn T
├── message U1
├── Run R1 interrupted
├── message U2
├── Run R2 interrupted
├── message U3
├── Run R3 completed
├── message A
└── Turn completed
~~~

U2 到达时，在一个 batch 里用精确 RunRef 关闭 R1、在 U2 的 Admission 元数据捕获新 ReplyTarget、在同一个 TurnRef 下追加 U2，并打开 R2。R2 的 `reply_to=U2`；即使 U1、U2 来自不同 thread，也不会到 seal 时猜。新 Message、新 Run，不新建 Turn。

这是 Store invariant，不是调用约定：

- 调用者提交 `expected_current_run`；single writer 在锁内 CAS 当前值。
- 有 active Step/Run 时，batch 必须先把每个未 terminal ToolCall 变成真实 result 或 canonical `outcome_unknown` result，再合法关闭 exact Step/Run，随后 append 新 Admission，最后 open 后继 Run。
- 没有 active Run 时，只有 latest Run 已明确 `interrupted` 且 Turn 仍 open，才可声明 `expected_current_run=None + expected_latest_run`，并把 Admission 与后继 `run/open` 同批提交。
- 后继 Run 的 Turn 必须相同；新 Admission 带 ReplyTarget 时，`reply_to` 必须是这条新 MessageRef。
- idempotent redelivery 只返回既有 MessageRef，不重复 interrupt/open；两个真正的新输入竞争时由 CAS 串行成 R2、R3。
- bare Admission、错误 expected ref、只 append 不开后继 Run、或让后继继续 reply_to 旧 Message 都 fail-loud。
- latest Run 是 failed 时，普通 Admission 不得留在旧 Turn；它走 5.3 fresh，原子 close(superseded) 并创建新 Turn。只有显式 retry 可以在旧 Turn 无新 Message地 open 新 Run。

因此 R1 的 ModelHistory 永远不会在运行中突然看到 U2；R1 的旧 callback 也会同时被 Run/Step fence 拒绝。

最终 seal 的一个 batch 追加：

~~~text
message/append(A, run=R3, step=Sfinal)
run/close(R3, completed)
turn/close(T, completed)
transcript/add(U1)
transcript/add(U2)
transcript/add(U3)
transcript/add(A)
delivery/prepared(A, reply address, exact binding)
~~~

整个 batch 成功或失败，不会出现半个聊天 Turn。

### 5.3 retry 与 fresh

~~~text
latest Run failed
├── explicit retry(prior_client_message_id, command_id)
│   ├── 不创建 Message
│   └── 同 Turn 追加一个新 run/open
└── ordinary fresh(new source_ref)
    ├── 旧 Turn close(superseded)
    └── 新 Turn、新 Message、新 Run
~~~

- prior_client_message_id 只在原 channel/session identity 下解析到 AdmissionKey，再解析到最后一个 human MessageRef。
- unknown、cross-scope、not-last 与 Gone 分别 fail-loud。
- command_id 只让 retry 命令幂等，不成为 MessageId。
- 正文相同不等于 retry。

retry 与 continuation 的区别不靠 `run_reason` enum：两次 Run 之间有新 Message 就是 continuation；有已接纳 retry command 而没有新 Message 就是 retry。

### 5.4 seal 与迟到完成

RunWork 只能用 current RunRef/StepRef seal。Session Store 在同一个写锁内 fold 当前 TurnState 并检查：

- Turn 仍 open；
- active Run 正是调用者 RunRef；
- active Step 正是 final Draft 的 StepRef；
- history-reading lane 仍属于该 Run；
- final Draft 合法且只有一个 terminal assistant；
- Session 未进入 deletion state；
- Run 打开时固定的 `reply_to` Message 仍是 Live，且其 ReplyTarget 可由 Delivery owner 解析成 exact binding。

seal 请求由 `RunRef + final content digest` 幂等识别。完全相同的重试返回同一 MessageRef；同一 RunRef 带不同 digest fail-loud。新 Run 已开始后，旧 RunRef 永远不能封口。

所有运行期 append 都走同一条检查：`request/bound`、`tool/call`、Tool result Message、`step/close` 和 final assistant Message 必须携带 RunRef 与 StepRef。Store 验证它们属于同 Session/Turn，且二者在该事件提交前都是 current，或在同一 batch 中合法 close。只带 TurnRef、只带 RunRef或带旧 StepRef 的迟到回调都没有写权限。

任何 RunWork 都可以 append assistant Message 后完成。只有 TurnGrant 含 `finish_without_output` 时，Core 才另发一个窄 `QuietCloser`；它还必须收到已提交领域 decision 的 DomainDecisionPermit，才可完成而不追加 assistant Message。

普通 listener 的 return 只结束 listener，不能关闭 Turn。

### 5.5 crash repair

- unmatched current Step 先为无 terminal result 的 ToolCall append structured `outcome_unknown` Tool-result Message，再在一个 repair batch 关闭 Step 与 `run/close(interrupted: process_lost)`；不自动重跑模型、Tool 或 provider。
- conversation Turn 保持 open，等待显式 continuation、retry、fresh 或 abandon。
- task Turn 的恢复由 source owner 根据自己的 durable decision 选择继续或关闭。
- 下一 Run 从 Session 按原序重放本 Turn 所有既有 Run 的已闭合 Tool call/result，包括明确标为 unknown 的 result；unresolved-effect fence 与 Tool effect ledger 共同阻止盲重试。
- crashed Run 的旧 Root 不跨进程复活；下一 Run 绑定当前已提交 generation。

---

## 六、Proactive 从 Core 消失

### 6.1 两个 quiet path

~~~text
source timer fires
    │
    ├── source ledger 写 attempt
    │
    ├── no_due / admission rejected
    │   └── 关闭 source attempt；Session 完全不变
    │
    └── due
        └── 在目标 Session 追加普通 Turn + plugin-source Message + Run
                                      │
                                      ├── domain decline / skip
                                      │   ├── source/domain decision event
                                      │   ├── run/close
                                      │   └── turn/close；无 assistant、无 transcript、无 delivery
                                      │
                                      └── share
                                          ├── assistant Message
                                          ├── turn/close
                                          └── Delivery
~~~

第一个 quiet 是“没有开始一次 Session 工作”；第二个 quiet 是“一次工作正常结束，但没有可发布 Message”。它们不需要一个共同 `proactive_skip` 字段。

### 6.2 来源私有的仍归来源

Wake 继续拥有 TimerAttempt、due、watermark、Content pool、Drift 顺序、业务 decision 与 ACK。Scheduler、Spawn 和 message_push 也各自拥有自己的 trigger ledger。

Core 只看到普通 event vocabulary：

~~~text
turn/open → message/append → run/open → ... → turn/close
~~~

没有 `proactive`、`wake`、`scheduler`、`spawn profile` 或 source enum 改变 Turn/Run 状态机。

### 6.3 可见、记忆和发送各看自己的事实

| 问题 | 唯一依据 |
|---|---|
| 用户是否看见 | transcript/add/remove |
| 模型是否看见 | ModelHistory projection 规则 |
| 是否写长期记忆 | projection/policy + Memory 自己的 eligibility |
| 是否应发送 | delivery/prepared 与 destination contract |
| 为什么开始 | message source → source ledger |

这样 assistant-only 主动消息不是特殊 Message。它只是一个没有 human transcript input、但有 assistant transcript/add 的 completed Turn。

---

## 七、Delivery 也不携带 Message

### 7.1 ReplyTarget 与 exact binding

回复位置也只捕获一次，但它不是 Message 正文，也不需要自己的 ID。Channel Host 或 source adapter 用受限 admission capability，把 opaque ReplyTarget 写进输入 Message 的可擦 privacy slot：

~~~text
ReplyTarget = {
    logical_channel,
    logical_address,
    credential_handle
}
~~~

ReplyTarget 不固定实现 generation，不包含 secret，也不复制输入 Message。v4 的每个 Turn 必须由至少一条 Admission Message 开始；`run/open.reply_to` 只能选择当前 Turn 内、带已授权 ReplyTarget 的 Admission Message。普通 conversation 选择触发本 Run 的最新输入；retry 复用原输入。Task/Wake/Scheduler 也由自己的 source permit 在 Admission Message 上写 target，不能让模型构造地址；没有 `reply_to` 的 Run 只能完成而不发送。

seal 前，Delivery owner 从 `reply_to` Message 的 metadata 解析当前可用的 exact target，并取得短 lease；seal batch 才追加：

~~~text
delivery/prepared = {
    turn: TurnRef,
    message: MessageRef,
    reply_to: MessageRef,
    exact_binding_slot:
        Live {
            exact_artifact,
            exact_generation,
            adapter_contract_version,
            destination_mapping,
            credential_handle
        } | Gone { redaction: EventRef }
}
~~~

DeliveryRef 就是该 event 的 EventRef。调用者不提供 idempotency key；worker 只在 prepared 已提交后，从 `DeliveryRef` 派生 provider operation key 再开始 I/O，因此没有“先知道自己的 event seq”分配环。若 batch 失败就释放临时 lease；提交后 lease 转给 durable Delivery lifecycle。

这比 admission 时长期钉住 adapter 更合理：一个 Turn 可以经过多次 Run 和 generation reload；真正产生发送义务时才冻结实现。一旦 prepared，恢复必须用 exact binding，找不到就 fail-loud，不能换成当前 adapter 猜。

### 7.2 状态是 delivery events 的 projection

~~~text
delivery/prepared
    ├── delivery/canceled
    └── delivery/provider_started
          ├── delivery/rejected
          ├── delivery/uncertain
          │      ├── delivery/rejected（query 证明未发送）
          │      ├── delivery/delivered（query 证明已发送）
          │      └── delivery/abandoned_uncertain（用户明确接受未知）
          └── delivery/delivered
                 └── transcript/add(U1…Un+A，同一 batch；provider-first 时）
                       └── delivery/settled
~~~

每次 transition 追加一个引用 DeliveryRef 的事件。DeliveryState fold 拒绝非法跳转；Session single writer 和当前 state CAS 防止两个 worker 同时推进。

规则：

1. prepared 引用 MessageRef、完整附件集合和 exact target binding，不复制正文。
2. provider_started 必须在网络 I/O 前提交。
3. provider 没有幂等键或查询能力时，crash 后只能 uncertain，不能盲发。
4. delivered 保存去掉正文、preview 和附件内容的 provider receipt；它不能因后续 Session projection 失败而回滚。
5. provider-first 只有拿到 receipt 后，才在一个 Session batch 追加 delivered 与该 completed Turn 的全部 `transcript/add(U1…Un+A)`；不能先露出半个 Turn。
6. session-first 在 seal batch 已追加 transcript/add，通知失败不能撤销 Message。
7. settled 只表示本地与来源 ACK 已前向收口，不改写 provider 历史。
8. prepared 到 settled/rejected/canceled/abandoned_uncertain 之间，exact target 不能退役。
9. uncertain 只能由 provider query/recovery 进入 delivered 或 rejected；没有证据时，只有显式用户授权才能追加 abandoned_uncertain。该事件只结束本地等待，审计仍记“外部可能已发生”，不能声称回滚或未发送。

顺序由 destination contract 决定，不由 passive/proactive 名字决定：

~~~text
canonical Akashic client: Session-first
external provider:          Provider-first
~~~

### 7.3 deletion 与 delivery

- prepared 且尚未 I/O：先追加 canceled；同一 maintenance transaction 再把 exact binding slot 转 Gone，之后才删除正文。
- provider_started 或 uncertain：删除阻塞，直到 provider 查询确认结果，或用户明确授权 `abandoned_uncertain` 并接受外部副本可能仍存在。
- delivered 但尚未 transcript/add：删除流程追加 `settled(projection_gone)`，保留 delivered 事实，不再把 Message 放进 Transcript。
- settled/rejected/canceled/abandoned_uncertain：Data Management 把 exact binding privacy slot 转 Gone，只保留无正文、无地址的效果 identity、最小 receipt 和必要 uncertainty audit。

Delivery 发现 Message tombstone 时返回 Gone，不得重建或另存正文。

---

## 八、删除与 projection 收口

### 8.1 正常路径只追加

- Session event envelope、坐标和非内容元数据正常只追加。
- Projection cache、索引、Mobile Room、Web state 和 compaction 可以随时删除重建。
- context 裁切、容量优化、插件 reload 和 cache cleanup 无权改写 Session Message。
- source ledger 与 provider effect ledger 按自己的 retention contract 管理，不由 Session cache cleanup 删除。

为同时满足“正常只追加”和“用户可以真正撤销正文”，Session Store 明确定义一个、也只定义一个例外：内容型 event 的 payload 是 Session-owned erasable slot。

~~~text
StoredMessageEvent = {
    immutable: event_seq + turn + role + producer refs / admission_token,
    body_slot:    Live(MessageBody) | Gone(redaction_event_seq),
    private_slot: Live(SourceRef + ReplyTarget) | Gone(redaction_event_seq)
}
~~~

普通 writer 只能创建 `Live`，永远不能 UPDATE。只有 Data Management maintenance transaction 能把 content/privacy slot 的 `Live → Gone`；不能反向恢复，也不能换成另一段正文或地址。`read_events` 和 `get_message` 先解析 slots：历史位置已经 Gone 时，重放从该位置直接得到 tombstoned Message，不会先遇到一个缺 `content` 的坏事件。

### 8.2 删除一个 Turn 的 Message

用户明确撤销后：

~~~text
建立可验证备份与影响预览
        │
关闭相关 Run / Delivery；query 或显式覆盖 unknown Tool effect
        │
一个 Session maintenance transaction
├── transcript/remove × N
├── redaction/apply × N
├── turn/redacted
└── 同事务把对应 Message body/private 与
    terminal Delivery exact-binding slots: Live → Gone(redaction seq)
        │
清理 attachment object 与各 durable materializer
并让 source owner 擦除 Envelope/recovery payload
        │
append redaction/verified；全部 ack 后才报告完成
~~~

通用 redaction event 只命名要擦的 owner 与 slot：

~~~text
redaction/apply = {
    targets: [
        { owner: MessageRef,  slot: body | private },
        { owner: ToolCallRef, slot: arguments },
        { owner: DeliveryRef, slot: exact_binding },
        { owner: EventRef,    slot: declared_dynamic_context }
    ],
    reason,
    audit_principal_token
}
~~~

event 不保存旧正文、旧地址、附件内容或默认 content digest。event vocabulary 为每种 event 声明允许擦除的 slot；任意字符串、任意 SQL 或不属于 Data Management scope 的 target 都 fail-loud。`get_message` 在 body Gone 后返回 Gone，不伪装成 NotFound。

Session Store 在事务锁内先分配 redaction seq，再同时追加 redaction event 和写入 `Gone(redaction_seq)`。crash 在 commit 前，两者都不存在；crash 在 commit 后，两者都存在。integrity check 强制每个 Gone 指向同 Session、更高 seq、已提交且精确列出 `(owner_ref, slot)` 的 redaction event；反向也强制每个 target slot 已 Gone 且指回该 event。full replay 因而始终确定。

原始 event envelope 与非正文 Run/Delivery finality 可以保留；Session content/privacy slot 的 `Live → Gone` 是 append-only 的唯一破坏性例外。raw SourceRef、ReplyTarget、Delivery exact address/credential handle、Tool call 参数、一次性 dynamic context 和其他含用户内容的 event 都使用这种 erasable slot，分别变成 Gone；不存在可供删除的持久 `assistant/chunk` 或 rendered Prompt 副本。防重只保留不可逆 AdmissionToken，不保留 raw AdmissionKey。

迟到写不能复活正文：旧 Run 因 RunRef fence 不能 append；同库 projection 被更高 cursor 拒绝；外部 materializer 则按 4.5 先 drain `started/uncertain`、再 cleanup。仅仅推进本地 CAS 不算外部清理完成。

### 8.3 删除整个 Session

`delete_session` 不是循环删行：

1. 建立 SQLite backup、附件清单与 projection/effect 影响预览。
2. 关闭新 admission，停止 history-reading lane。
3. interrupt active Run，等待其 durable close。
4. cancel 未 I/O Delivery；provider_started/uncertain 明确阻塞，除非 query 收口或用户显式接受 abandoned_uncertain。
5. 对 delivered Delivery 保存外部 finality，并关闭未完成 transcript projection。
6. 在 maintenance transaction 追加 session deletion/redaction、全部 transcript remove，并把 Message/Tool argument/dynamic context content slots，以及 SourceRef/ReplyTarget/Delivery binding privacy slots，原子转为 Gone。
7. 擦除 attachment object、staging object、source Envelope/recovery payload、受控日志副本和其他已盘点用户内容；Request 从未保存 rendered Prompt 或 history body。
8. 等所有同库 projection cursor 越过 deletion seq，并按各 owner journal 收口外部 materializer cleanup。
9. append deletion/verified，只保留 SessionId、wire cursor shell、AdmissionToken/LegacyToken Gone 映射、最小非正文 effect finality 与删除 audit。
10. 校验引用图、cursor high-water、backup retention 与 SQLite integrity 后才报告成功。

SessionId、EventRef、AdmissionToken 和 LegacyToken 永不复用。删除后的 retention 窗口内保留无正文 Session shell，使离线客户端收到更高 seq 的 remove 和 terminal `session_gone`；shell 物理 GC 后，最小 Session tombstone 仍让旧 cursor 得到 `SessionGone` 并清空该 Session，而不是当作一个从未存在的新 Session。服务端不能声称已经擦除离线设备无法控制的本地副本。

---

## 九、公共 API 应该很小

### 9.1 Session Store 原子能力

~~~python
class SessionWriter(Protocol):
    async def append_batch(self, build: EventBatchBuilder) -> CommittedBatch: ...
    async def read_events(self, after: int, limit: int) -> list[SessionEvent]: ...
~~~

调用者不能发明 event_seq 或自行构造任何 typed ref。它只能使用 capability 已绑定的 committed ref，或 Builder 为本批较早事件分配的 typed local handle；Store 在锁内分配坐标，提交后返回真正 EventRef。

Store 在边界集中校验 JSON、event vocabulary、所有内部 ref 同 Session且指向正确类型、`turn/open` 初始 batch 至少一条有序 Admission Message且恰有一个后置 run/open、已有 Turn Admission 的 interrupt→append→successor 原子结构、运行产物的 current RunRef/StepRef、Turn/Transcript outcome、`reply_to` 指向本 Turn 的已授权 Admission Message、delivery transition、AdmissionKey uniqueness 和 deletion state。

### 9.2 一个 writer，加一个可选 quiet capability

~~~python
class RunWork:
    async def react(self) -> FinalDraft: ...
    async def interrupt(self) -> None: ...
    async def seal(self, draft: FinalDraft) -> MessageRef: ...

class QuietCloser:
    async def finish_without_output(self, decision: DomainDecisionPermit) -> None: ...
~~~

它们只是限制合法 append batch 的一次性 capability，不是持久 Turn 类型：

- 默认 RunWork 不能无输出 completed。
- QuietCloser 只在 TurnGrant 明确允许时签发；没有已提交 decision 也不能使用。
- 输入或输出是否进入 Transcript 只看 transcript events，不由 writer 类型决定。
- 模型只产生 Draft，不选择 Session、event seq、message_id 或 destination；target 已由 Run capability 固定。
- DomainDecisionPermit 只证明 source/domain decision 已提交，不是 Core 的 source enum。

### 9.3 插件边界

默认 REACT 插件拥有 Prompt、Context、LLM、ToolView、ToolSearch、Tool loop、stream 和 Draft。Core 提供 Session append/read、Root lease、TurnGrant、CallPermit 与 writer capability。

~~~text
ToolView       模型看见哪些 Tool
TurnGrant      这个 Turn 最多可做什么
CallPermit     这一次调用真正获准什么
~~~

三者不能合并。隐藏 Tool 不等于撤权，展示 Tool 不等于授权。

---

## 十、每个事实只有一个 owner

| Owner | 唯一拥有 | 不拥有 |
|---|---|---|
| Session Store | event log、event_seq、batch 原子性、引用校验、redaction | Prompt、provider、projection 业务 |
| Source plugin | pre-admission attempt、due、cursor、业务 decision | Message body、Turn/Run 状态、Delivery |
| Channel Host / source adapter | wire 校验、source_ref、受限 target capture、ACK | Session truth、exact binding、模型算法、provider finality |
| Run Host | live task、per-Turn lease、history-reading lane、interrupt token、Root lease | Message body、外部 effect |
| Plugin Runtime | artifact、Root、generation、publish、drain | Session/Delivery outcome |
| REACT plugin | Prompt/Context/LLM/Tool 算法与 Draft | Message identity、Session commit、授权授予 |
| Tool plugin | schema、调用、自己的 effect ledger | 全局 loop、任意 Session 写入 |
| Projection Registry | fold 驱动、key、schema version、checkpoint/cursor | Session event、领域 projection 内容 |
| Projection plugin | 自己的 state、wire view、外部 effect journal、cleanup receipt | 原始 Message、Core 全局 claim |
| Delivery owner | delivery event producer、exact binding、provider receipt、恢复 | Message body、source ACK 业务 |
| Data Management | backup、maintenance mode、redaction、tombstone、cleanup 协调 | 正常 compaction、provider 回滚 |

物理上同一个 SQLite 文件可以让 append batch 原子；逻辑 owner 仍通过窄端口分开。

---

## 十一、v3/v4 再做一次减法

| 旧概念 | 新处理 |
|---|---|
| 独立 MessageBody store | 删除；Message body slot 是 Session message event 的一部分 |
| SessionEntry relation | 删除；Transcript add/remove 是 Session events |
| SessionReceipt outbox | 删除；Session log 本身就是 cursor/change feed |
| ProjectionGrant table | 删除；policy 是 Session event |
| Core ProjectionClaim/跨插件 lease | 删除；纯 fold 用 cursor，外部写由 owner 的窄 effect journal 收口 |
| Turn table | 删除；TurnState fold turn events |
| Run/Attempt table与随机 ID | 删除；RunState fold，RunRef 是 run/open EventRef |
| ToolFact table | 删除；request/tool/message events 已在 Session |
| Delivery row随机 ID | 删除；DeliveryRef 是 prepared EventRef，state 由 events fold |
| pending body 与 committed body | 删除双份；Envelope/Draft 在提交前不是 Message |
| placement | 删除；Transcript event 是唯一可见性事实 |
| proactive boolean/kind | 删除；source attempt 与普通 Turn 足够 |
| storage=durable/in_memory | 删除全局轴；Session fact durable，cache 可重建 |
| post_commit allow/suppress | 拆成明确 projection keys 的 policy event |
| 额外 fence UUID | 删除；当前 RunRef 就是不可复用 fence |

仍必须保留的真实轴：

- SessionId：事实作用域；
- event_seq：Session 内唯一顺序；
- MessageRef：正文身份；
- transcript seq：用户可见变化顺序，直接复用 transcript event_seq；
- ReplyTarget metadata 与 `run.reply_to`：逻辑发送位置在输入处捕获，Run 只选择已有 MessageRef，exact binding 后置；
- source attempt：可能先于 Session 或根本没有 Session；
- exact Root 与 request binding：同一 Turn 的不同 Run 可以独立换代；
- projection policy：相同消息形状可以有不同长期影响许可；
- provider / materializer effect state：外部世界不会跟 Session 事务一起回滚；
- tombstone：删除后的重投不能复活正文；
- Legacy token mapping：旧入站身份必须在不保存 raw id 的前提下确定地解析到唯一 MessageRef/Gone。

“foolish and simple”不是把事实硬挤成一个字段，而是让所有事实只沿一条 Session 时间线出现一次，其他东西都能丢掉重算。

---

## 十二、目标架构

~~~text
┌──────────────────────── Sources ─────────────────────────┐
│ Channel · Wake · Scheduler · Spawn · message_push        │
│ own: attempt / due / cursor / source_ref                 │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────── Session append log ──────────────────┐
│ turn · message · run · step · request · tool             │
│ transcript · policy · delivery · redaction               │
│ one SessionId + one event_seq line                       │
└───────────────┬──────────────────────────┬────────────────┘
                ▼                          ▼
┌──────────────────────────┐    ┌───────────────────────────┐
│ REACT / ModelHistory     │    │ Pure Projection Registry  │
│ reads refs, emits Draft  │    │ Turn · Transcript · UI    │
└──────────────────────────┘    └─────────────┬─────────────┘
                                             ▼
                                effectful materializer owner
                                Memory / index journal + I/O
~~~

目标 Core schema/API/test base 不出现：

~~~text
proactive
wake
content duty
drift duty
scheduler
spawn profile
provider family
memory plugin name
~~~

这些词可以留在 owning plugin 的事件和 projection 中，不能改变通用 Turn/Run append rules。

---

## 十三、迁移路线

### Phase 0：先批准语义

本提案若获批，先同步修改 projectneed 与 accepted decisions：

1. Session 从“已完成聊天 rows”提升为完整 append log；Transcript 成为 projection。
2. Turn/Run 从独立权威 row 改成 Session events 的 projection。
3. 全局 post_commit 改成 projection policy events。
4. proactive 从 Core schema/API 删除。
5. internal MessageRef 只有 EventRef；wire boundary 通过 v4 encoding 或 legacy mapping 解析，不引入随机 v4 MessageId。
6. v4 不提供 durable Session fork；跨 Session 复制产生新 Message identity。

批准前本文只是 proposed target，不能覆盖现行合同。

### Phase 1：建立版本化 Session log

1. 用 SQLite backup API 备份 sessions.db，并记录 schema、hash 与 integrity_check。
2. 新建带 format version 的 Session event storage、erasable content slot、append_batch 和 single-writer handle。
3. 写 event envelope、同 Session typed refs、Run fence、unknown-event fail-loud、redaction replay 与 projection fold tests。
4. 先迁移只读 projection，不切生产 writer。

### Phase 2：一次迁移旧事实

1. 每条旧 Message 生成一个 message event，并在同批追加 `identity/legacy-token(message, versioned_hmac(old_id))`；验证后删除 raw old id，LegacyIdResolver 由 token events 重建。
2. 每条用户可见旧 Message 生成 transcript/add event；新 transcript seq 直接使用新 Session event_seq，不伪造旧 seq 映射。
3. 每个迁移 Turn 也必须用 `turn/open + 首条 Admission Message` 的初始 batch。旧 assistant-only 工作只有在 durable source evidence 能确定地产生非可见 Admission Message 时才能迁；否则整次迁移 fail-loud，不能造空 Turn。
4. 只有已有明确 interaction/attempt evidence 时才生成 Turn/Run links；任何 Message 无法确定所属 Turn 时整次迁移 fail-loud，交维护者处理，不能按角色、时间或相邻 seq 猜。
5. allow/suppress 按持久事实迁成 policy event；非法/冲突/未知值 fail-loud。
6. 差分验证消息数、legacy resolver 一对一、顺序、正文 hash、附件、删除边界和 projection 结果。

迁移完成后删除旧 writer 和旧 schema reader；不长期 dual-write 或双读。Legacy resolver 只接受旧入站值、从不输出 raw old id，是兼容期唯一边界索引，不是第二个内部 Message model。

### Phase 3：切 Turn/Run 与 REACT

1. 实现 per-Turn current RunRef/StepRef fence、同 Session ref 校验与 Session history lane。
2. 用事件 fold 取代 turns/attempt rows。
3. 把当前 default reasoner 整体包成 REACT，再逐步把 Prompt、Context、Tool 与模型 binding 变成普通依赖。
4. fixture 比较每个请求的 model-visible messages、Tool facts、final Draft 和 Session events。

### Phase 4：切 Transcript 与客户端

1. Session 完成 batch 写 transcript/add；provider-first 在 delivered 后写。
2. Web/Mobile 使用 message_id + 稀疏 transcript event seq + Session high-water cursor。
3. cursor 做明确断代，不猜映射：旧 cursor 首次请求得到 `reset_required`；客户端用 v4 full snapshot 原子替换该 Session 本地投影，再保存响应的 `cursor_v4=session_high_water`。
4. shadow 对比旧 Session history 与新 Transcript projection。
5. 证明 empty delta 也推进 cursor、reconnect、重复 frame、乱序 frame、删除 tombstone 和 interrupted continuation。

### Phase 5：迁移来源与 Delivery

按 Scheduler → Spawn → Wake → message_push 分批：

- 每批只替换 source admission 和 output handoff；
- Core event vocabulary 不增加来源名；
- Wake 同时证明 no_due 无 Session event、domain skip 有 completed Turn 无 Transcript；
- Delivery 用真实 provider、exact binding 和 crash injection 证明前向恢复。

### Phase 6：删除旁路

只有所有动态 consumer、插件 cache、正式 generation 和真实 DB 差分都通过后，才删除：

- MessageBody/SessionEntry 双模型；
- SessionReceipt/outbox；
- turns/attempt 权威 rows；
- global post_commit；
- proactive 字段与分支；
- 旧 schema/cursor 兼容 reader；只保留 legacy wire resolver。

### 回滚

- 每个 schema/data 阶段前建立名称清楚、可校验的 backup。
- v4 writer 启用后，旧 binary 不得直接打开正式库。
- 回滚要么运行版本化前向转换，要么恢复切换前 backup 并明确放弃其后的新事实。
- Git 回滚不能撤销 provider 或 Tool 外部效果；Delivery/effect ledger 继续收口。

---

## 十四、验收 Gate

### 14.1 单一真源

- [ ] 新 Message 正文只存在于一个 Session message event 的 body slot；durable chunk/request/tool link 不复制它。
- [ ] Turn、Run、Transcript、ModelHistory、Mobile 与 Memory 可从 Session 重建。
- [ ] 删除所有 projection cache 后，同一 Session 得到相同视图。
- [ ] Delivery、Tool ledger、日志和 trace 不复制 Message body。
- [ ] model-visible Message 与 Tool result 都有 Session event 证据。

### 14.2 身份与同步

- [ ] 新 MessageId、TurnId、RunId、DeliveryId 不使用独立随机 UUID。
- [ ] typed refs 都指向正确类型的 EventRef，调用者不能自己填写 seq。
- [ ] 所有内部 refs 同 Session；v4 无 durable fork 和跨 Session Message graph。
- [ ] raw legacy message_id 不持久化、不输出；一对一 versioned token 仍把旧入站值解析到原 MessageRef/Gone，Core 内部只有 EventRef。
- [ ] wire message_id 不透明，授权不依赖不可猜性。
- [ ] transcript seq 稀疏、单调、不复用；empty delta 也用 next_cursor 推进 Session high-water。
- [ ] 旧 cursor 只走 reset + full snapshot，不猜新旧 seq 映射。
- [ ] client_message_id 只做 source-scoped admission/retry。
- [ ] identity HMAC key rotation 前后，同一 admission redelivery 或 legacy id 命中同一 MessageRef/Gone；有引用的旧 key version 不退休。

### 14.3 Turn、Run 与原子性

- [ ] U1 → interrupt → U2 → interrupt → U3 → A 是一 Turn、三 Run。
- [ ] `turn/open` 初始 batch 缺 Admission/Run、或 run/open 排在首条 Admission 前时，Store fail-loud。
- [ ] retry 是同 Turn 新 Run且无新 Message；fresh 新建 Turn。
- [ ] existing Turn 的 U2 不能裸 append；必须 CAS exact current Run，并同批 close R1→append U2→open R2(reply_to=U2)。
- [ ] 两个并发 continuation 由 current Run CAS 排成两个 Run；旧 callback 与只 append 不开 Run 都 fail-loud。
- [ ] latest Run failed 后 ordinary input 必须 supersede + 新 Turn；只有 explicit retry 可无新 Message留在旧 Turn。
- [ ] request/tool/运行产物 Message 都强制携带 current RunRef/StepRef，拒绝旧 Run 的迟到 append/seal。
- [ ] S1 close、S2 open 后，S1 的迟到 Tool result 被拒绝；pending call 只有真实 result 或 canonical outcome_unknown result 后才能 close。
- [ ] U1→non-idempotent tool started→interrupt/U2→provider unknown 时，R2 history 含 unknown，unresolved-effect fence 阻止自动重复调用。
- [ ] 同 Session history-reading Run 串行；无 history Run 可以并发。
- [ ] ModelHistory 取 cut 时 Transcript，并按原序重放当前 Turn 当前 Step 前、跨所有 Run 的 Admission 与 closed Tool pair；不吸入交错 task 私有 Message。
- [ ] conversation seal 原子追加 final Message、Run/Turn close、全部 transcript/add 和 session-first Delivery。
- [ ] 每个 Run 固定 exact Root；每个 request 固定 ModelExecution/header。
- [ ] closed Tool pair 可重放；unknown effect 不盲重试。

### 14.4 projection

- [ ] 每个 projection 是确定 fold，有 state version 和 source cursor。
- [ ] cache mismatch 直接丢弃重建，不反写 Session。
- [ ] 同库 materialization 用同事务 cursor；外部 materializer 在 I/O 前写 owner-local effect journal。
- [ ] started/uncertain 先 query/drain，cleanup 用更高 desired version；本地 CAS 不冒充远端完成。
- [ ] validation 与普通 conversation 同形状时，policy 仍能禁止长期影响。
- [ ] policy key 不使用 source/plugin 名字。
- [ ] Transcript visibility 只读 transcript events，不读 proactive 或 Session 前缀。

### 14.5 proactive

- [ ] no_due/reject 有 source attempt、零 Session event。
- [ ] due 只使用普通 Turn/Message/Run events。
- [ ] domain skip 是 completed Turn，无 assistant、Transcript 或 Delivery。
- [ ] share 只产生一条 final assistant Message，其他模块引用它。
- [ ] Core schema/API/test base 没有 proactive/wake/source enum。

### 14.6 Delivery

- [ ] prepared 引用 MessageRef，不复制正文。
- [ ] Run 打开时 `reply_to` 已固定同 Turn、带授权 ReplyTarget 的 Admission Message；U1/U2 不同 target 不会在 seal 时猜。
- [ ] prepared 冻结 exact target；找不到 exact binding 时 fail-loud。
- [ ] provider operation key 由 committed DeliveryRef 派生，prepared payload 不自引用未分配 seq。
- [ ] provider_started 在 I/O 前提交。
- [ ] crash 后不盲发；uncertain 可查询、可人工收口。
- [ ] prepared→canceled 与 uncertain→delivered/rejected/abandoned_uncertain 都是合法显式 transition；人工关闭仍保留 unknown audit。
- [ ] provider-first 只有 delivered 后才 transcript/add。
- [ ] session-first notification 失败不撤销 Transcript。
- [ ] target lease 只从 prepared 保持到 terminal，不从 admission 长期钉住。

### 14.7 删除

- [ ] 删除前有 backup 与影响预览。
- [ ] redaction/remove 使用更高 Session seq；MessageRef 进入 Gone。
- [ ] Gone body slot 与更高 redaction event 同事务提交；full replay 不会遇到半个 Message。
- [ ] raw SourceRef、ReplyTarget 与 terminal Delivery binding 都进入 privacy slot；删除后只留 AdmissionToken 与无正文 finality。
- [ ] 单 Turn 删除先 cancel 尚未 I/O 的 Delivery，并在同一 maintenance transaction 擦除 canceled exact binding。
- [ ] source ledger 只留 token/ref/finality，Envelope 与 recovery payload 已擦除；unknown Tool effect 已 query 或由用户明确覆盖。
- [ ] projection cleanup、started/uncertain materializer 或 uncertain Delivery 未闭合时不报告删除成功。
- [ ] 迟到 Run、retry、source replay、Delivery 和 projection 都不能复活正文。
- [ ] delete_session 覆盖 Run、Delivery、附件、Tool result、Tool args、dynamic context、Mobile/Memory projection、cursor shell 与 tombstone。

### 14.8 概念 Gate

- [ ] 每个权威事实只在一个 Session event 或一个明确外部 ledger 中出现。
- [ ] 每个 cache 删除后可由权威事实重建。
- [ ] 删除任一留下的字段会破坏一个已命名不变量；否则继续删除。
- [ ] 没有按来源名复制 Turn/Run/Delivery 状态机。
- [ ] 没有永久 dual-write、dual-read 或 guessed migration。

---

## 十五、需要维护者批准的决定

推荐整组批准：

1. Session append log 是唯一 Message 真源，Transcript 只是 projection。
2. 完整 Message body 只写入一个 Session-owned erasable slot；durable chunk、rendered Prompt 和第二份 Tool result 不存在。
3. Turn、Run、Step 和 Delivery identity 由 typed EventRef 派生，不新增随机 ID。
4. internal MessageRef 只有 `(SessionId, message event seq)`；raw legacy id 不持久化/不输出，只通过 versioned HMAC token 在入站 boundary 解析。
5. wire seq 使用 transcript add/remove event seq，cursor 是 Session scan high-water；旧 cursor 通过 reset + full snapshot 断代。
6. Turn/Run 权威 rows、SessionEntry、SessionReceipt、ToolFact 和 Core ProjectionClaim 从目标模型删除。
7. current RunRef 同时承担 execution fence，不再增加 fence UUID。
8. projection policy 是 Session event，不是 global boolean 或独立 Grant relation。
9. proactive 从 Core 完全删除；no-turn 与 completed-without-output 分开。
10. Delivery state由 Session delivery events fold；provider、Tool 和 effectful materializer 的外部 finality 仍归各 owner journal。
11. logical ReplyTarget 是 Admission Message 元数据，Run 复用 MessageRef 选择 `reply_to`，exact binding 在 prepared 时冻结；不新增 target ID。
12. 显式删除是 append-only 的唯一内容 slot 变更，以同事务 Gone/redaction、外部 drain 和高版本 cleanup 收口。
13. v4 不支持 durable Session fork；跨 Session 复制产生新 MessageRef。

实现任务仍需单独确定：

- event envelope 与 wire encoding 的具体字节格式；
- SQLite 表、索引、format version 和 append_batch API；
- abandoned/completed-without-output Turn 的 retention；
- 每个外部 provider 的幂等键与查询能力；
- 哪些 durable projections 必须阻塞删除完成。

---

## 十六、证据索引

### DSH 当前源码

检查基线：`/mnt/data/source-code/deepseek-harness` commit `49a606bc5b5934603f22a26957a07dc799ab0291`。以下证据只支持单日志与 pure fold，不证明 Akashic 的异步 I/O、删除、并发或 fork 语义。

- `/mnt/data/source-code/deepseek-harness/docs/architecture.md:74`：Turn 是零个或多个 Step；turn/step/message/tool 都写 Session events。
- `/mnt/data/source-code/deepseek-harness/docs/architecture.md:103`：Session log 是 model context、fork、resume、transcript、telemetry 和 persistence 的源。
- `/mnt/data/source-code/deepseek-harness/packages/core/session/src/types.ts:255`：SessionEventMap 是 append-only truth。
- `/mnt/data/source-code/deepseek-harness/packages/core/session/src/index.ts:628`：seq 等于 log length，append 后才发布。
- `/mnt/data/source-code/deepseek-harness/packages/core/session/src/index.ts:772`：deriveMessages 从 surface projection 重建。
- `/mnt/data/source-code/deepseek-harness/packages/core/agent-loop/src/index.ts:55`：turnBoundary 是纯 projection。
- `/mnt/data/source-code/deepseek-harness/packages/session/session-turn-outline/src/projection.ts:84`：Turn outline 从 committed events fold。
- `/mnt/data/source-code/deepseek-harness/packages/session/session-projection/src/index.ts:40`：projection unit 是纯同步 fold，并以 state version/source seq checkpoint。
- `/mnt/data/source-code/deepseek-harness/packages/llm/llm/src/message.ts:130`：同一个 Message representation 供 delivery、history 和 model request 使用。
- `/mnt/data/source-code/deepseek-harness/packages/llm/llm/src/message.ts:175`：DSH 当前为入 Session 前的 Message 生成 UUID；本设计有意不照抄。

### Akashic 当前合同与代码

- `docs/projectneed.md`：SES-001～SES-008、RUN-001～RUN-009、OUT-001～OUT-005、PRO-001～PRO-006。
- `docs/decisions/0039-react-core-atoms-keep-sources-unprivileged.md`。
- `docs/decisions/0040-wake-duty-gate-lives-in-scoped-react.md`。
- `docs/decisions/0041-turn-effects-and-memory-plugins-are-orthogonal.md`。
- `docs/design/persistence-state-map.md`。
- `session/store.py:2392`：当前 turns row 仍表达 attempt；目标态改成 Session event projection。
- `session/store.py:4744`：当前 message_id 由 session_key:seq 生成；证明 derived identity 已有先例。
- `session/store.py:5385`：当前 completed interaction 删除边界。
- `infra/mobile_realtime/protocol.py:188`：client_message_id 是 transport/admission identity，不是 Session MessageId。

### 用户要求与运行案例

- 2026-09-02/03 Codex 会话：要求回到基本原理、减少概念、正确解释 Turn、消除 proactive 特判，并质疑新 MessageId。
- 2026-09-03 修正：当前行为不等于正确设计；Session 应是唯一消息载体，其他消息视图都是 projection，并参考 DSH。
- hua-home 私有历史只用于 interrupt/retry/Wake 场景覆盖；原始内容不提交，也不定义目标语义。

---

## 最终判断

真正简单的模型不是：

~~~text
MessageBody → Turn row → SessionEntry → Receipt → ProjectionClaim → UI
~~~

而是：

~~~text
Session events（唯一事实）
        ├── Message 只出现一次
        ├── Turn / Run 只是 fold
        ├── Transcript / Model / Mobile / Memory 只是 fold
        └── Delivery 只引用 MessageRef
~~~

新 Message 仍需要一个稳定引用，但不需要一个新的随机 UUID。`(SessionId, message event seq)` 已经足够；wire 继续叫 message_id，只是不再拥有第二套身份。

proactive 也不需要被“优化成更聪明的特判”。它应该从 Core 消失：开始前是 source attempt，开始后就是普通 Session events；是否可见、是否学习、是否发送分别由 Transcript、policy 与 Delivery 事实决定。

这版比前一版更少，也更正交：少的不是可靠性，而是第二份事实。
