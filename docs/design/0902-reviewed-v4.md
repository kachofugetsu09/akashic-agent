# Akashic v4：Message WAL 与普通插件组合

- 文档版本：0902-reviewed-v4
- 日期：2026-09-03
- 状态：设计提案，等待维护者批准
- 当前代码基线：47896b4200731183a54081e2eca77602a0881a0a
- DSH 参考基线：49a606bc5b5934603f22a26957a07dc799ab0291
- 本文不授权：实现、数据库迁移、正式 workspace 写入、删除、部署或合并

## 结论

v4 不是只有一次数据库减法。它同时做两件彼此正交的事：

1. **事实层做减法**：对话事实只剩 Session 与 Message；
2. **行为层做拆分**：原来固定在 Core/Bootstrap 的被动回复大链路，变成普通插件组合。

两件事缺一不可。只做第二件，会让插件继续围着 Turn/Run 等重复状态转；只做第一件，
则只是把新 WAL 塞回旧的 `PassiveMessageWorker → AgentLoop → PassiveTurnPipeline`
巨型流水线，并没有得到可替换的 Agent。

Akashic 的对话事实层只保留两个名词：

~~~text
Session = 一条只追加的 Message WAL
Message = 一个已经完整产生并被 WAL 接纳的事实
~~~

没有第三种权威对象。没有 Turn、Run、Step、Attempt、Delivery row，也没有一层
`SessionEvent` 再包住 Message。

这里的“只有”限定在**对话事实层**。附件字节、credential、插件 generation、
scheduler 配置和 provider 外部状态仍由各自边界拥有；它们不是对话实体，也不能复制
Message。它们若要影响或进入对话，只能被 Message 的类型化内容引用或报告。

~~~text
┌──────────────────── Session ────────────────────┐
│ seq 1  Message：用户说 U1                      │
│ seq 2  Message：用户补充 U2                    │
│ seq 3  Message：助手请求调用 tool              │
│ seq 4  Message：tool 返回 success              │
│ seq 5  Message：助手回答 A                     │
└─────────────────────────────────────────────────┘
          │              │               │
          ▼              ▼               ▼
       聊天视图       模型上下文       手机增量同步
~~~

网络重试、模型重试和未完成 token 都发生在对应 Message 写入以前。它们没有产出
Message，因此不进入 Session。重启后可以丢失这些执行过程；不能丢失的只有已经
提交的 Message。

被动回复也不是第三个事实对象。它只是一个普通插件提供的函数：读到已提交的
Message，调用其他普通插件，最后再产生 Message。Core 不认识 passive、proactive、
Wake 或某一种 Agent 算法。

~~~text
对话事实                         普通插件行为
────────────────────            ─────────────────────────────
Session                          committed Message
  └── Message WAL ─────────────▶ MESSAGE_REACTOR
         ▲                         ├── COMMANDS
         │                         └── AGENT_PROGRAM
         │                               ├── Prompt contributions
         └──────── append Message ◀──────┼── Context projection
                                         ├── Tool selection
                                         └── Model / Tool ports

同一份 Message WAL ──▶ Chat / Model context / Memory / Sync / Delivery projection
~~~

`MESSAGE_REACTOR`、`AGENT_PROGRAM` 和各项依赖是运行时 capability，不是实体、日志
记录或另一种消息载体。替换 Agent Program 不改变 Session schema；停用自动回复插件
也不会删除已经收到的 user Message。

### 把我当六岁

Session 是唯一一本作业本。Message 是已经用墨水写完的一行字。

- 小朋友先在草稿纸上写，写错可以重来很多次。
- 只有一句话写完整，才抄进作业本。
- 抄进去以后，这一行有自己的编号 `message_id`，也有所在页码 `seq`。
- 聊天页、模型看到的上下文和“这一轮”的括号，都只是拿彩笔从作业本里画出来。
- 擦掉彩笔，作业本没有少东西；换一种画法，也不用迁移事实。

旁边还有一个会读作业本、再写新行的机器人，它就是插件：

- 换一个机器人，只是换回答办法，作业本格式不变；
- 关掉机器人，孩子写下的 user Message 仍在，只是不再自动回答；
- 机器人找模型、挑工具、重试网络，都是它工作时的动作，不是作业本里的新东西；
- 机器人只有把一句话写完整并交给作业本，才算 Agent 真的说过。

所以 U1 后模型断网三次，最后生成 A，作业本仍只有两行：

~~~text
1  user       U1
2  assistant  A
~~~

那三次断网不是三次对话，也不是三条事实。它们只是草稿纸上的失败。

## 一、两个权威对象

### 1.1 Session

Session 只拥有身份和按 `seq` 排列的 Message。它不拥有当前执行、轮次、主动模式、
投递状态或投影视图。

~~~text
Session {
  session_id
  messages: Message[]  # 按 seq 连续排列
}
~~~

这里的 WAL 指领域层的 append-only message log，不是再增加一张 event 表。底层可以
使用 SQLite WAL，但 `sessions.db/messages` 本身才是产品真源。

### 1.2 Message

~~~text
Message {
  message_id   # 在 append 前生成；全局稳定、不透明
  session_id
  seq          # WAL 原子提交时分配；Session 内单调连续
  role         # system | user | assistant | tool
  content[]    # 完整、类型明确的内容块
}
~~~

初始合同不提供通用 `meta` 袋子。以后若要加字段，必须先证明它拥有一个不能由
`role`、`content`、Session 配置或 projection 表达的独立事实。

最小内容块是：

~~~text
text        { text }
artifact    { artifact_ref, media_type }
reply       { target_message_id }
tool_call   { name, arguments, tool_binding, provider_token? }
tool_result { call_ref, outcome, output }
no_reply    {}
delete      { target_message_id }
~~~

这不是所有产品内容的封闭枚举，而是本次设计必须验证的最小集合。新增 image、audio、
citation 等内容时继续扩展 `content` 的 typed union，不新增平行 Message 表或通用
metadata 袋子。`artifact_ref` 和 `tool_binding` 指向各自边界已有的不可变对象；它们
不是新的对话身份。

`outcome` 只有 `success | error | unknown`。失败不是缺一条成功记录，而是一条内容
明确为 `error` 的完整 tool Message。外部结果无法确认时必须写 `unknown`，不能猜成
成功，也不能盲目重试。

`call_ref` 不引入新的随机身份。它由发出 `tool_call` 的
`(assistant message_id, content block index)` 得到。provider 自己要求的 token 只是
协议内容，不能升级成 Core 的 ToolCallId。

### 1.3 “Message 存在”究竟证明什么

Message 存在只证明两件事：

1. producer 已经产生一个完整 Message；
2. Session WAL 已经持久接纳它。

它不自动证明别的事情：

- assistant Message 存在，不代表手机或邮件已经收到它；
- `tool_call` 存在，不代表工具成功；
- `tool_result(outcome=success)` 才表示工具 owner 确认成功；
- `tool_result(outcome=unknown)` 表示外部效果可能发生，但现在无法确认。

这样“有没有说出来”和“外部事情有没有做成”是两条正交事实，不再由一个 Run
状态含糊地同时代表。

## 二、为什么仍需要 message_id 和 seq

这不是两个 Message 身份，而是两个不同问题的答案：

| 字段 | 回答什么 | 何时得到 |
|---|---|---|
| `message_id` | 这是不是同一条 Message | append 前 |
| `seq` | 它在这个 Session 的第几个位置 | commit 时 |

只用 `seq` 会遇到一个无法消失的问题：客户端发送 U1，服务端可能已经提交，但 ACK
在网络中丢了。客户端重发时还不知道 U1 的 `seq`。如果没有预先存在的稳定身份，
服务端无法区分“同一条 U1 重试”和“又说了一次 U1”。

因此正确的减法不是删除 `message_id`，而是让它成为唯一身份：

- 客户端创建完整 user Message 时生成 `message_id`；重试始终复用它；
- 模型或工具产生完整内容以后、append 以前生成 `message_id`；
- 删除 `client_message_id`、`retry_of_client_message_id`、TurnId、RunId、StepId 和
  Core DeliveryId；
- 老数据已有的 message ID 原样保留并视为不透明值，不建立 alias 或映射系统；
- 新 ID 的具体编码只是实现选择，合同只要求唯一、稳定、不可从业务含义推断。

如果改成 `(session_id, seq)` 派生 `message_id`，仍要再造一个 pre-commit retry key。
那会把一个身份重新拆成两个，反而更复杂。

## 三、唯一写协议

### 3.1 Append 是 commit 点

所有 producer 都走同一条路：

~~~text
在内存中产生完整内容
        │
        ▼
封好不可变内容，并分配 message_id
        │
        ▼
Session.append(message, expected_head_seq?)
        │
        ├── 校验 Message 与原子前置条件
        ├── 分配 seq
        ├── durable commit
        └── commit 后才 ACK / 发布给 projection
~~~

WAL 必须保证：

1. `message_id` 唯一；
2. 同一 Session 的 `seq` 唯一且连续；
3. 同 ID、同完整内容重试时返回原 `seq`，不再追加；
4. 同 ID、不同内容时 fail-loud，不能覆盖或悄悄归一化；
5. ACK 只能发生在 durable commit 以后；
6. projection 失败不能把已提交 Message 变回未提交。

append 前的内容与 ID 只是调用参数和内存值；只有 commit 返回的带 `seq` 记录才是
Session 中的 Message。Message 引用的 artifact 必须已经 durable/ready，不能先写半条
Message 再补附件。

Message 正常路径不可变。唯一例外是用户明确删除后的受控正文擦除，见第九节。

### 3.2 模型重试发生在 append 以前

~~~text
read Session at head H
        │
        ▼
provider 请求 / 断网 / 重试 / token stream    ← Message 内容只在内存
        │
        ▼
完整输出成为 assistant Message M
        │
        ▼
append(M, expected_head_seq=H)
~~~

- provider 断网：丢掉未完成输出，重试；Session 不变。
- token 只生成一半：可以直播给当前界面，但不能 append；崩溃后丢掉。
- 重试耗尽：产品可以产生一条完整 error Message 再 append；若没有产生 Message，
  Session 就不声称助手说过什么。
- WAL commit 成功但 ACK 丢失：用同一个 `message_id` 重试 append，得到原 `seq`。

这里不需要 durable attempt、Run 或 Step。若计费和排障需要看失败尝试，telemetry
只记录时间、错误码、用量和 provider request ID 等运行数据，不持久复制未提交正文、
tool 参数或结果。telemetry 不能决定对话事实，也不能反向补写 Session。

### 3.3 新输入打断旧输出

假设模型正在根据 U1 生成答案，Session head 是 1；这时 U2 先提交成 seq 2：

~~~text
seq 1  U1 ──▶ 生成旧草稿 A-old
seq 2  U2
               append(A-old, expected_head_seq=1) ──▶ conflict
               丢掉 A-old，读取 U1 + U2，重新生成 A
seq 3  A
~~~

`expected_head_seq` 就是足够小的并发栅栏。两个 worker 同时从同一个 head 生成答案，
也只有第一个能提交；另一个看到 conflict 后丢弃结果。无需保存“现在是哪一个 Run”。

## 四、工具调用也只是 Message

### 4.1 正常调用

~~~text
seq 10  assistant  tool_call(search, {...})
                           │  commit 后才能执行
                           ▼
                    调用 / 查询 / 安全重试
                           │
                           ▼
seq 11  tool       tool_result(call_ref, success, {...})
seq 12  assistant  根据结果回答
~~~

顺序是故意的：请求调用工具的 assistant Message 先写入 WAL，外部执行才开始。这样
崩溃后总能从 `tool_call` 内容算出同一个 `call_ref`。

`tool_call` append 前，边界 adapter 必须按本次模型可见的 schema 校验名称与参数，
并封入 exact immutable `tool_binding`。执行时再检查该 binding 仍被当前权限允许；若
已经撤权，产生 `outcome=error` 的 tool Message。模型看见什么工具是配置的投影，
能否真的调用由工具权限 owner 决定，不能从模型输出反推授权。

工具 owner 的规则：

1. 同一时刻只启动一个该 `call_ref` 的本地 executor；这是可丢失的调度约束，不是
   durable 对话状态；
2. 把 `call_ref` 用作 provider 幂等键；
3. 网络错误时先按 provider 的 query/idempotency 能力确认，再决定是否重试；
4. 得到确定结果以后，产生一个完整 tool Message，再 append；
5. tool Message 的同一 `call_ref` 只能有一个终态结果；重复相同结果幂等成功，冲突
   结果 fail-loud；
6. 最终 assistant Message 重新读取包含 tool result 的最新 Session，再用 head CAS
   提交。

### 4.2 崩溃与未知外部效果

`tool_call` 已提交而 tool result 不存在时，“pending”只是从 WAL 算出的视图。

- provider 能按 `call_ref` 查询：查询真实结果，产生 tool Message；
- provider 支持相同幂等键：可以安全恢复调用；
- 两者都不支持：不能盲目再执行，产生 `outcome=unknown` 的 tool Message。

这承认一个不能被数据模型消灭的边界：Session 回滚不了已经发出的付款、邮件或 Git
操作。正确做法是保留调用意图、复用一个幂等地址并诚实记录 `unknown`，不是再加一套
Run/effect 状态机假装获得 exactly-once。

## 五、旧 Turn 和 Run 去哪里

它们都不进入新领域模型、schema、公共 API 或持久化合同。

如果聊天 UI 想把若干行圈成“一次交互”，可以临时画括号：

~~~text
┌─ 对话分组视图 ───────────────────────┐
│ seq 1  user       U1                 │
│ seq 2  user       U2                 │
│ seq 3  assistant  tool_call          │
│ seq 4  tool       success            │
│ seq 5  assistant  A                  │
└──────────────────────────────────────┘
~~~

这个括号是 projection，不叫 Turn 对象，没有 ID，没有 open/seal/abort 状态，也不被
重试、删除、计费或插件引用。分组算法升级后重新画即可。

同样，开发者界面可以把当前 token、provider 延迟和重试次数画成“执行中”视图；它读
内存与 telemetry，不叫 Run，也没有恢复业务语义。进程重启后这个视图消失是允许的。

本设计明确接受以下损失：

- 无法从 Session 重建失败过几次 provider 请求；
- 无法恢复半截 token stream；
- 无法给一次执行尝试分配稳定身份；
- 无法承诺 UI 的对话分组永远不变。

这些都不是用户或 Agent 已经说出的 Message，因此不值得污染权威模型。

### 5.1 用 hua-home 历史反推，而不是照抄现状

检查过的私有历史只提供反例场景，原始内容不进入本文，旧字段也不定义目标模型：

- **补充输入**：U1 后生成尚未完成，用户又发 U2。目标态是 `U1, U2, A`；旧草稿
  因 head 冲突丢弃，不留下 interrupt/attempt 记录。
- **provider 重试**：一次回复前出现连接错误，后来生成 A。目标态仍是 `U1, A`；
  错误只进短期 telemetry。
- **Wake 检查**：`no_due` 没有进入对话的内容，因此不写 Message；已经产生输入后，
  模型无话可说则写 `[no_reply]`，有话可说则写普通 assistant text。
- **tool 不确定**：外部调用 ACK 丢失时，目标态从已提交 `tool_call` 恢复并查询；不能
  确认时写 `tool_result(unknown)`，不把旧 Run 状态当答案。

这些 case 证明需要的是 WAL、幂等和 CAS，不是 Turn/Run 实体。当前行为若与此冲突，
它是 migration delta，不是保留旧设计的理由。

## 六、proactive 不再是特判

Core 不认识 `proactive`、Wake、Scheduler 或某个插件名。

~~~text
scheduler 检查 no_due       ──▶ 不产生 Message
scheduler 判断需要处理      ──▶ 产生普通 system/user Message
agent 判断 quiet             ──▶ assistant Message [no_reply]
agent 生成内容               ──▶ 普通 assistant Message [text/...]
~~~

所以主动说话和用户问答走完全相同的 append、模型上下文、客户端同步与投递路径。
没有 `is_proactive` 字段，也没有 proactive 状态机。

一旦输入 Message 已经进入 Session，模型产生的 quiet 判断也必须成为完整 assistant
Message：`content=[no_reply]`。Chat projection 隐藏它，恢复 projection 用它判断此前
缀已经处理。否则重启只看到一条没有后续的输入，会无限重复同一次判断。这里记录的
不是一个空 Turn，而是 Agent 确实生成的结果。

Scheduler 可以在插件边界拥有自己的时间表和 due cursor；那是调度器的运行配置，不是
第三种对话事实。若“由计划任务触发”本身必须进入对话历史，就由 producer 把它写成
普通、类型明确的 Message 内容，而不是给 Core 增加来源分支。

## 七、其余对话状态全部是视图

“全部是视图”说的是**持久对话状态**：除了 Session 与 Message，不再保存一个平行
状态来解释对话进行到哪里。插件行为本身不是 projection，也不是事实；它是读取
Message、调用 capability、再产生 Message 的短命函数。

| 视图 | 从 Session Message 得到什么 |
|---|---|
| Chat | 选择 user/assistant text，应用 delete |
| Model context | 映射 role/content，裁切只发生在本次请求 |
| 对话分组 | 为 UI 临时圈住相邻 Message |
| Tool status | `tool_call` 加对应 `tool_result` 得到 pending/success/error/unknown |
| Next action | 从末尾 Message 判断执行 pending tool、生成回复或 idle |
| Web/Mobile | 按 `seq > cursor` 返回 Message |
| Memory/Search | 从允许学习的 Message 建索引 |
| Live stream | 展示尚未提交的内存输出；刷新或崩溃可丢失 |

projection 可以缓存 `(session_id, projection_version, source_seq, value)`，但缓存必须：

- 可删除、可重建；
- 只读到某个 `source_seq`，不能宣称看见未来；
- 不获得覆盖、重排或补造 Message 的能力；
- 版本不匹配时重建，不迁移成第二份事实；
- 上下文裁切、摘要和 token budget 永不 UPDATE/DELETE Session Message。

客户端协议只需要：

~~~text
request:  session_id, after_seq
response: Message[]
cursor:   最后完整应用的 seq
~~~

客户端用 `message_id` 去重和引用，用 `seq` 排序与追赶。projection 晚到、重复或重建
都不能改变 WAL。

恢复时也不读 Run 状态。`Next action` projection 依次判断：

1. 有尚无终态 result 的 `tool_call`：查询或安全恢复该工具；
2. 否则，末尾是需要反应的 user/system/tool Message：从最新 head 生成 assistant；
3. 最新 assistant 没有 unresolved tool call（包括 text 或 `no_reply`）：idle。

assistant Message 通过 head CAS 紧跟它实际读取的前缀，因此“哪个前缀已经处理”也能
由 seq 推出，不需要额外 cursor、Turn outcome 或 Run receipt。

## 八、投递是 Message 之后的外部效果

canonical Web/Mobile 直接读 Session WAL。邮件、推送或第三方聊天渠道则由投递 worker
把“哪些 assistant Message 需要送出”算成队列视图。

这类 retry 发生在 Message append 以后，但它不是“重新生成 Message”：

~~~text
assistant Message 已提交
        │
        ├── provider 发送成功
        ├── provider 明确失败，可按策略重试同一个 message_id
        └── ACK 不明，先查询；不能确认时标记外部状态 unknown
~~~

- worker 始终引用原 `message_id`，不复制正文，不生成 DeliveryId；
- queue/cache 只保存 `message_id` 和短期运行数据，发送时从 WAL 读取正文；
- provider 支持幂等键时直接使用 `message_id`；
- provider 不支持幂等或查询时，系统不能承诺 exactly-once；
- receipt、重试次数和延迟属于 provider telemetry/worker cache，不进入 Core schema；
- 若将来产品真的要求“送达结果也成为可回忆事实”，它只能作为一条新 system Message
  进入同一 WAL，不能新建平行账本。

因此，Message 存在的标准仍是“Agent 已经完整说出并提交”，而不是“每个外部渠道都
已经收到”。

## 九、删除仍只使用 Message

正常路径永远只追加。用户明确撤销一条 Message 时，先追加一条普通 delete Message：

~~~text
seq 20  system  delete(target_message_id = M7)
~~~

所有 projection 读到 seq 20 后都不再展示、送入模型或学习 M7。删除没有 Redaction、
Tombstone row 或另一套版本号；delete 只是 Message 的一个内容块。

delete 的效果单调：普通 delete 不能以“删除 delete Message”的方式恢复旧正文。已删除
ID 的旧 append 重试只返回原 `seq` 与 gone 结果，永远不比较、覆盖或恢复原 content。

“撤销刚才那组对话”先让当前 UI projection 解析出一组明确的 `message_id`，再为每个
目标 append delete Message。执行清单固定后不随分组算法变化，也不需要一个 TurnId。
删除整个 Session 则是名称明确的 Data Management 操作，不伪装成普通对话写入。

若用户还要求物理擦除正文，Data Management 必须在 delete Message durable 以后：

1. 建立名称清楚的恢复点；
2. 列出目标 Session、Message 和受影响 projection；
3. 只擦除目标 Message 的 content payload，保留 `message_id`、`session_id` 和 `seq`；
4. 重建所有持久 projection 并做前后完整性检查；
5. 保证旧请求重试不能让 M7 复活。

这是 append-only 的唯一例外，必须由用户明确的数据管理操作触发。上下文压缩、容量
优化、迁移和插件都无权调用。

## 十、从 DSH 借什么

检查的 DSH 基线是
`/mnt/data/source-code/deepseek-harness@49a606bc5b5934603f22a26957a07dc799ab0291`。

值得借用的只有三个原则：

1. 一份 ordered Session log 是 history 与 projection 的共同来源；
2. 一个 immutable Message representation 跨 history、model request 和 delivery 复用；
3. Message 在 publication/append 前已经拥有稳定 ID，pure projection 可随时重建。

本设计不照抄 DSH 的 `SessionEvent`、turn/step 事件或完整 runtime。Akashic 再做一步
减法：Message 自己就是 WAL record，不需要事件壳。

对应源码证据：

- `packages/llm/llm/src/message.ts:130`：一个 immutable Message 供 history、model
  request 和 delivery 共同使用；
- `packages/llm/llm/src/message.ts:175`：Message 在发布前创建稳定 UUID；本设计只借
  pre-append identity，不规定 UUID 格式；
- `packages/core/session/src/index.ts:628`：Session 以单调 seq append；
- `packages/core/session/src/index.ts:772`：model messages 从 Session surface 派生；
- `packages/session/session-projection/src/index.ts:40`：projection 是按 seq fold 的
  versioned cache。

## 十一、v4 的另一半：把被动回复大链路变成普通插件

### 11.1 现在真正需要替换的链路

当前基线的被动回复不是普通插件组合，而是一条固定 owner 链：

~~~text
PassiveMessageWorker
  ├── 入站 custody / attachment / per-session lane
  ▼
ConversationRuntime
  ▼
AgentLoop._react()
  ▼
PassiveTurnPipeline
  ├── command short-circuit
  ├── BeforeTurn
  ├── BeforeReasoning
  ├── reasoner + BeforeStep / AfterStep
  ├── AfterReasoning：parse + persistence + outbound
  └── AfterTurn：事件 + dispatch / ACK
~~~

代码证据也显示这些责任仍被固定装配：

- `bootstrap/passive_worker.py:96`：`PassiveMessageWorker` 拥有消息准入、lane task 和
  结果 task；
- `agent/looping/core.py:556`：`AgentLoop._react()` 只把请求转给固定 pipeline；
- `agent/core/passive_turn.py:355`：`PassiveTurnPipeline` 构造固定的四段 phase；
- `agent/core/passive_turn.py:440`：command 在 Session/model 准入前走专门短路；
- `agent/core/passive_turn.py:524`：默认被动回复仍由一个固定 `run()` 入口统管。

这会让插件只能在旧流水线上挂 hook，而不能真正替换业务。新增语音 Agent、无工具
Agent、plan/execute Agent 或另一种回复策略时，要么继续给 Core 加分支，要么复制整条
pipeline。单独换成 Message WAL 并不会消除这个问题。

### 11.2 目标：`Message → react → Message`

默认产品提供一个普通 `passive-conversation` 插件。它与第三方插件经过同一套加载、
依赖解析、candidate 校验、generation 发布和生命周期清理；Core 不给它后门。

这不是再建第二套插件框架。当前代码已经有可复用的骨架：

- `agent/plugin_composition/model.py:28` 已定义类型化 `ServiceKey`；
- `agent/plugin_composition/context.py:148` 已提供绑定 exact Root 的短命 runtime scope；
- `plugins/models/plugin.py:58` 已用普通 `provide()` 发布 model services；
- `plugins/compaction/plugin.py:468` 已用普通插件发布 provider request projection；
- `plugins/markdown_memory/plugin.py:67` 已把 Prompt 与 post-commit memory 行为接入普通
  plugin lifecycle。

~~~text
passive-conversation plugin
  provides MESSAGE_REACTOR
  injects  SESSION_READ, SESSION_FEED, SESSION_APPEND
           COMMANDS, AGENT_PROGRAM

default-agent plugin
  provides AGENT_PROGRAM
  injects  SESSION_READ, SESSION_APPEND
           PROVIDER_REQUEST_PROJECTION
           PROMPT_PARTS
           TOOL_SELECTOR
           ASSISTANT_TRANSFORMS
           CHAT_MODELS
           TOOL_EXECUTOR
           STREAM_PREVIEW
~~~

这些大写名字都是普通 `ServiceKey`，不是 Core 固定 slot，更不是新领域对象：

- `SESSION_READ` 只提供 read/head，`SESSION_FEED` 只发布 committed Message，
  `SESSION_APPEND` 只签发绑定 Session、role 与 CAS/typed precondition 的 writer；三者
  共用一个 WAL owner，但权限彼此独立，都没有任意 SQL、原位改写或删除能力；
- `MESSAGE_REACTOR` 读一个已经 committed 的输入 Message，按最新 WAL projection 判断
  `idle / command / respond / recover tool`，再选择 command 或 `AGENT_PROGRAM`；它不
  保存自己的 outcome；
- `AGENT_PROGRAM` 拥有默认模型/工具算法，包括 provider retry、Tool Search、空回复
  修正、terminal tool deadline 和继续生成；
- 其他 key 只是 `default-agent` 自己的依赖。不使用工具的 Agent 不需要提供假的
  `TOOL_SELECTOR`，Core 也不维护一张“所有 Agent 都必须有”的选择表。

接口只传已有身份，不发明 ReactionId、ProgramId 或通用 context 袋子：

~~~text
MESSAGE_REACTOR.react(session_id, cause_message_id) -> None
AGENT_PROGRAM.respond(session_id, cause_message_id, scoped_messages) -> None
~~~

`None` 只表示函数已经结束；可观察结果只能是 WAL 中新增了哪些 Message。异常、取消、
provider retry 和临时资源留在当前 Fiber/telemetry。它们不能通过另一个 result record
偷偷变成第二份对话事实。

不保留 v3 的 `decide() → handle()` 双阶段，也不新增 ReactionPlan。能否继续只由最新
Message 前缀算出；同一 `cause_message_id` 被重复唤醒时，projection 已经是 idle 就直接
结束，否则最终仍由 append 的 CAS/typed precondition 仲裁。

`passive-conversation` 自己用普通 Effect 订阅 committed Message，并在自己的 Fiber
里调用 `MESSAGE_REACTOR`。因此 Core 甚至不需要知道这个 ServiceKey。产品 profile
要求自动回复时，candidate 必须恰好解析出一个 provider；依赖缺失或重复 provider
在发布前 fail-loud。停用这个插件以后：

- 渠道仍可把 user Message 写入 WAL；
- Chat、同步和历史仍正常；
- 不再自动产生 assistant Message；
- UI projection 可以显示“Agent 未启用”，Core 不偷偷启用 legacy fallback。

这里的 `passive` 只是默认插件包名，不是 Message 字段、Session 模式或 Core 分支。
Scheduler、Wake、Channel 和 subagent 都只产生普通 Message，或者显式依赖同一个
`AGENT_PROGRAM`；来源不会复制一套执行模型。

### 11.3 固定 phase 不原样搬家

插件化不是把 `PassiveTurnPipeline` 整块移动到 `plugins/`。现有每项能力先找到唯一
owner；没有独立不变量或真实消费者的 phase/hook 直接删除：

| 当前固定行为 | v4 owner |
|---|---|
| channel envelope、附件导入 | Channel/Artifact adapter；artifact ready 后才 append user Message |
| 入站 durable handoff / ACK | Channel adapter；user Message durable 后结算 |
| command catalog 与短路 | `passive-conversation` + 注入的 `COMMANDS` |
| Session/history 准备 | `SESSION_READ` + pure Session projection |
| system prompt、skills、memory、profile | 有序、不可变的 `PROMPT_PARTS` contributions |
| history 裁切、摘要与 compaction retry | `PROVIDER_REQUEST_PROJECTION`；沿用普通 compaction plugin |
| tool schema preload / Tool Search 解锁 | `TOOL_SELECTOR`，由 `default-agent` 使用 |
| Tool 展示 | `TOOL_SELECTOR`；只能缩小当前可见集合 |
| Tool 授权与真实执行 | 受保护的 `TOOL_EXECUTOR`；调用边界重新校验 |
| 默认 ReAct、provider retry、空回复重试、terminal deadline | `AGENT_PROGRAM` |
| model/provider 绑定 | models/provider 插件；从当前 exact Root 注入 |
| Citation、Meme、最终文本/媒体改写 | append 前的有序、不可变 `ASSISTANT_TRANSFORMS` |
| assistant/tool 写入、幂等、seq、CAS | `SESSION_APPEND` 签发的 scoped writer |
| Memory、Akasha、compaction 派生数据 | committed Message observers / projections |
| partial token 展示 | `STREAM_PREVIEW`；可丢失且无权 append 半条 Message |
| error reply、quiet | reactor/program 产生普通 assistant `text` 或 `no_reply` Message |
| continuation、crash 后下一步 | 从 WAL 重建的 `Next action` projection |
| assistant 对外发送、重试、provider ACK | 独立 Delivery projection/effect，只引用 `message_id` |
| generation 固定、取消、资源清理 | 通用 plugin Root lease + Fiber/Effect 生命周期 |

Command 若只读或只生成回复，可以直接产生 assistant Message；若会付款、发信、改 Git
等产生外部副作用，必须先生成普通 `tool_call` Message，再走同一 Tool 协议。不能因为
它叫 command，就在 WAL 看不见的短路里执行一次可能重复的外部动作。

Prompt 与 assistant transform 的顺序必须确定，但“顺序确定”不表示插件可以任意叠加。
每个 contribution 不可变，只有 composition owner 能形成最终序列；candidate Gate 要拒绝
重复 owner、冲突位置和循环依赖。transform 必须在 assistant Message append 前结束；
post-commit observer 无权回来改正文。

### 11.4 Tool 可见性不是 Tool 权限

`default-agent` 可以用 `TOOL_SELECTOR` 决定本次把哪些 schema 给模型看，但它不能扩大
当前授权。模型产生 `tool_call` 后：

1. 按本次可见 schema 校验并把 exact immutable `tool_binding` 写进 assistant Message；
2. `TOOL_EXECUTOR` 在真实副作用边界重新检查当前授权；
3. 已撤权时不执行，append 一个明确的 `tool_result(error)`；
4. 未知外部结果按第四节写 `unknown`，不能由 Agent Program 猜成 success。

因此替换 `AGENT_PROGRAM` 只能改变算法，不能绕过工具权限 owner。

## 十二、Thin Core 最终保留什么

Core 只保留来源无关、产品算法无法安全拥有的原子能力：

1. **Plugin composition**：`ServiceKey`、`provide/require/inject`、依赖冲突校验、exact
   committed Root lease、candidate/stable、Fiber/Effect 清理、health/incident；
2. **Session Message WAL**：append、read、subscribe、head、幂等、seq、CAS 和受限
   writer；用户删除 Session/Message 的 Data Management 是另一个显式管理入口；
3. **短命执行安全**：取消、超时、per-session 串行准入和有界资源 scope；这些可以
   丢失，不分配持久身份；
4. **真实外部边界**：模型调用、Tool 授权/执行、stream preview、channel ingress/ACK
   和 delivery effect 的窄 port；具体 provider 与策略仍由普通插件提供；
5. **类型化观察**：只发布 committed Message；observer 失败不能回滚、覆盖或补造
   WAL 事实。

Core 不认识下面这些产品词：

~~~text
passive / proactive / Wake / Scheduler / command
compaction / memory / Citation / Meme / Tool Search
某个 model/provider 名称 / 某个 channel 名称
~~~

如果 Core 源码需要按这些名字分支，说明 capability owner 仍没有拆干净。反过来，
`ServiceKey`、Root lease 和 Fiber 也不进入 Session；它们是让插件安全工作的机器零件，
不是对话事实。

## 十三、完整目标链路

~~~text
Channel adapter
  │  先让 artifact durable
  │  append user Message(message_id)
  │  durable 后 ACK inbound
  ▼
┌──────────────────────── Session Message WAL ────────────────────────┐
│ user / system / assistant(tool_call|text|no_reply) / tool / delete │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ committed feed
                               ▼
                    passive-conversation Effect
                               │ exact Root lease
                               ▼
                       MESSAGE_REACTOR
                      ┌────────┴────────┐
                  known command    AGENT_PROGRAM
                                      │
                  ┌───────────────────┼────────────────────┐
                  ▼                   ▼                    ▼
        provider request view   prompt/tool selection   model retry
                                                          │ complete
                                                          ▼
                                            append assistant Message
                                            ├── text/no_reply ─────┐
                                            └── tool_call          │
                                                   │ commit first  │
                                                   ▼               │
                                             TOOL_EXECUTOR         │
                                                   │ complete      │
                                                   ▼               │
                                            append tool Message    │
                                                   └── generate ───┘

committed Message feed
  ├──▶ Chat / Web / Mobile cursor
  ├──▶ Model context / Memory / Search projections
  └──▶ Delivery projector ── provider send/retry/ACK
~~~

这里有两个故意分开的 commit 点：

- **输入事实**：channel 在 user Message durable 后就可以 ACK。Agent 后来是否回答，
  不能反过来决定“用户有没有说过”；
- **输出事实**：assistant Message append 表示 Agent 已经说出。外部渠道是否送达由
  Delivery effect 继续处理，不能回滚或复制 assistant Message。

这会有意替换当前将入站 custody、回复终态和 delivery 结算绑在同一大链中的行为。
它不是漏迁移，而是由“Message 是唯一消息载体”直接推导出的新边界。

## 十四、重启、热更新与并发不需要 Run

### 14.1 活着时固定一个 exact Root

一次 `react()` 调用开始时拿到 exact committed Root lease，到函数完成或取消才释放。
其中的 model retry、Prompt、tool selection 和 transform 都看同一 Root。热更新可以发布
新 Root，但不能在半次调用中偷换依赖。

Fiber、取消 token、超时和 per-session lane 都是内存资源。U2 到来时可以尽早取消旧
生成；即使取消来不及，旧输出也会被 `expected_head_seq` 拒绝。正确性来自 WAL CAS，
不是来自一个持久 Run 状态。

### 14.2 崩溃后只从 Message 恢复

- assistant Message append 前崩溃：没有事实；重启后可用最新 Root 重新生成；
- assistant `tool_call` 已 append：从其 `call_ref` 与 `tool_binding` 查询或恢复；
- tool Message 已 append：下一次从最新 WAL 继续生成 assistant Message；
- assistant text/`no_reply` 已 append：该前缀已经有结果，projection 得到 idle；
- delivery 中崩溃：继续用同一 `message_id` 查询或重试，不重新生成正文。

`tool_binding` 中的 generation identity 是 Message 内容的一部分，不需要 RunId。由 WAL
派生的 unresolved-tool projection 为所引用的插件 generation 加 retention lease；进程
启动时先重建这些 lease，再允许清理旧 generation。若安全 owner 撤销权限，就写明确
的 tool error Message 并释放 lease，不能执行过期授权。

这只保证一段仍活着的函数内部使用同一 Root。崩溃后的新函数可以使用新 Root；唯一
必须保持的是已提交 tool binding 的执行身份。为了跨崩溃保存整套旧算法而新增 durable
Run，成本大于它保护的事实，本设计明确不做。

### 14.3 Candidate 与外部插件边界

- 内置插件与外部插件使用同一 loader、manifest、依赖图和生命周期 API；
- 缺少依赖、重复 Service provider、贡献顺序冲突在 candidate 阶段 fail-loud；
- candidate 使用隔离 Session feed，不得订阅生产 Session、发送真实 delivery 或执行
  高风险 Tool；
- 普通插件不得 import Core 私有模块、获得完整 workspace/repository/SQL 或删除能力；
- generation 下线前必须 drain 自己的 Effects、listeners、tasks 与 leases；
- 一个仓库外 fixture 必须能提供替代 `AGENT_PROGRAM`，不改 Core 就完成真实回复。

## 十五、迁移路线：WAL 与插件化一起完成

当前 `projectneed`、decisions、代码和数据库仍有 Turn/Run/attempt 合同。本文是有意的
替代设计；当前行为是调查证据，不是正确性的来源。批准前不实现，批准后也不能把
旧名词藏进 adapter 永久保留。

### Phase 0：批准新合同

- 批准 Session/Message 是唯一对话事实，失败 attempt 不进入 Session；
- 批准单一 pre-commit `message_id`、head CAS、tool `unknown` 和外部 exactly-once
  边界；
- 批准 inbound ACK 与 Agent reply/delivery 解耦；
- 以新条款 supersede `projectneed` 和相关 decisions 中的 Turn/Run 合同；
- 建立数据库、附件引用、插件 generation 和客户端 cursor 的可恢复备份与影响清单。

### Phase 1：先冻结完整行为账单

在改动 owner 前，用真实 Session fixture 和受控 provider/tool 记录当前大链的：

- 输入与附件、command、Prompt sections、model request、tool schemas 与调用顺序；
- compaction retry、Tool Search 解锁、空回复 retry、terminal tool、continuation；
- Citation/Meme/媒体变换、partial stream、error/no_reply；
- Message 写集、Memory/Akasha 观察、Web/Mobile cursor、delivery 与两侧 ACK；
- 热更新时的 exact generation、取消和资源清理。

每个差异必须先标成“保留能力”“按 v4 有意替换”或“已证明的旧 bug”。oracle 不要求
盲目复制现状；它要求任何消失的能力都有明确决定。旧 phase 名称本身不是能力，没有
消费者的 hook 不迁移。

### Phase 2：建立目标 WAL 与窄 capability

- 从现有数据只读重建目标 Message WAL；已有 message ID 原样保留；
- 无法证明顺序、角色、tool pairing 或 outcome 的记录 fail-loud；
- 实现正交的 `SESSION_READ`、`SESSION_FEED`、`SESSION_APPEND` 与 scoped CAS writer；
- Chat、Model context、Tool status、Next action、Memory 和 Mobile 先 shadow rebuild；
- 还未切换生产 writer，不改正式 workspace。

### Phase 3：先抽出普通 `AGENT_PROGRAM`

- 把默认 Reasoner/ReAct、provider retry、Tool Search、空回复和 terminal policy 从
  `AgentLoop`/pipeline 抽成 `default-agent` 插件；
- 把 Prompt、context、tool selection 和 assistant transforms 变成普通依赖/contribution；
- 用临时窄 adapter 接回旧入口，比对 Phase 1 oracle；adapter 只存在于迁移期并登记
  删除 commit；
- 用一个无工具替代 Agent Program 证明 Core 与默认算法已解耦。

### Phase 4：接入 `passive-conversation` 并 shadow

- 插件订阅 shadow Message feed，调用 `MESSAGE_REACTOR → AGENT_PROGRAM`；
- command、附件、error/no_reply、tool loop、continuation 与输出 transform 逐项对账；
- shadow 禁止真实 append、Tool 副作用和 delivery，只比较计划产生的 Message 与
  外部调用；
- candidate/stable 切换期间验证 exact Root 和所有 Effect 均可排空。

### Phase 5：一次切换唯一 writer

- Channel 先 append user Message，再 ACK inbound；
- user、assistant、tool、scheduler producer 全部只走同一个 WAL append；
- 只启用 `passive-conversation` 的生产 subscriber，旧 worker 变为不可达；
- Delivery projector 只消费 committed assistant Message；
- projection 完成 cursor reset/shadow 对账后接管读取；不长期 dual-write 两套事实。

### Phase 6：删除旧链和旧权威结构

- 删除 `PassiveMessageWorker → ConversationRuntime → AgentLoop._react →
  PassiveTurnPipeline` 固定业务链；
- 删除 Before/After phase DAG、proactive/source 分支和无消费者的兼容 hook；
- 删除旧 Turn/Run/Step/attempt/delivery Core rows、API 与双重 message identity；
- 卸掉所有迁移 adapter，确认没有插件 cache、动态消费者或恢复任务仍引用它们；
- 重建 projection，核对每个 Session 的 Message 数、ID、seq、tool pairing、delete、
  plugin generation retention 与客户端 cursor。

迁移前后都不能改写正式 workspace，除非另有明确授权、恢复点和执行前后完整性检查。

## 十六、验收 Gate

### 16.1 概念 Gate

- Core schema 和领域 API 只有 Session、Message；
- 不存在 TurnRef、RunRef、StepRef、AttemptRef、DeliveryRef、ProjectionClaim 或
  SessionEvent 壳；
- `MESSAGE_REACTOR`、`AGENT_PROGRAM`、Root、Fiber 和 projection 不可序列化成第三种
  对话事实；
- tool call 只用 Message 内可派生的 `call_ref`；
- proactive 不是类型、字段、状态机或特殊 commit 路径；
- 没有 `metadata/context/intent` 通用可变袋子绕过 typed Message 与 capability。

### 16.2 WAL Gate

- 同 `message_id` 同内容重试只得到同一 `seq`；
- 同 `message_id` 不同内容 fail-loud；
- commit 前崩溃没有 Message，commit 后 ACK 丢失不会重复 Message；
- 每个 Session 的 seq 唯一连续，projection/observer 失败不影响 commit；
- 正常路径没有 UPDATE/DELETE Message；受控物理擦除只走 Data Management；
- U2 抢先提交时，基于旧 head 的 A 被拒绝；两个 worker 也只有一个能提交。

### 16.3 插件组合 Gate

- Core/Bootstrap 不再 import 或构造 `PassiveMessageWorker`、`AgentLoop` 默认算法和
  `PassiveTurnPipeline`；
- Core 源码不按 passive/proactive/Wake/Scheduler/Citation/Meme/compaction/provider
  名称分支；
- 停用 `passive-conversation` 后，输入 Message、历史与同步仍工作，只停止自动回答；
- 替换 `AGENT_PROGRAM` fixture 无需修改 Core、WAL schema 或 channel adapter；
- 缺失/重复 Service 和 contribution 冲突在 candidate 发布前失败；
- Prompt 与 transform 次序确定，post-commit 插件不能修改 Message；
- 仓库外测试插件经正式安装链提供 `AGENT_PROGRAM`，不使用 Core 私有 import；
- generation 下线后无遗留 task、listener、subscription、Root lease 或 tool binding lease。

### 16.4 行为 Gate

- command short-circuit、附件、Prompt、compaction、Tool Search 和模型绑定都有目标 owner；
- provider 断网、retry 与 partial stream 不写 Session；
- `tool_call` 先提交再执行，crash 后按 `call_ref/tool_binding` 查询或安全恢复；
- Tool 可见性不能扩大真实授权，撤权产生明确 tool error；
- 不可确认的外部效果产生 `unknown`，不会盲目重复；
- 空回复 retry、terminal deadline、continuation、error reply 和 transform 有受控 fixture；
- `no_due` 不产生 Message；quiet 产生普通 `no_reply` assistant Message，Chat 隐藏；
- user Message durable 后即可 ACK，reply/delivery 失败不会抹掉输入；
- assistant Message durable 后 Delivery 独立重试同一 `message_id`，不会重新生成正文；
- Web/Mobile 只用 `message_id + seq + cursor` 完成重复、断线和追赶；
- delete Message 到达后所有 projection 一致隐藏目标，旧 retry 不会复活它。

### 16.5 迁移 Gate

- Phase 1 每项行为都有 preserve/replace/bug 分类和可复跑 fixture；
- WAL shadow 重建与旧读取逐 Session 对账；无法转换的数据有显式阻塞清单；
- 切换点只有一个生产 writer、一个自动回复 subscriber，没有窗口式 dual-write；
- 数据迁移、客户端 cursor reset 和 generation retention 均从备份做过恢复演练；
- 删除清单逐项证明静态 import、插件源码/cache、测试、数据库、日志和运行进程都无消费者。

### 16.6 明确接受的非目标

- 不从 Session 审计失败 provider attempt；
- 不恢复半截 token；
- 不给执行尝试分配稳定 ID；
- 不跨崩溃固定整套旧 Agent Program；只保留已提交 tool binding；
- 不保证外部 provider exactly-once，除非 provider 提供幂等或查询合同；
- 不让 UI 分组成为持久身份。

## 十七、对 v3 和前一版 v4 的最终减法

| 删除的东西 | 原因 | 现在由什么承担 |
|---|---|---|
| MessageBody / SessionEntry 两层 | 同一正文两种 owner | Message WAL record |
| SessionEvent 壳 | Message 已经是日志记录 | Message |
| Turn / Run / Step / Attempt | 把短命执行升级成事实 | head CAS + Fiber + telemetry + projection |
| client/retry message IDs | 同一 Message 多个身份 | 唯一 `message_id` |
| ToolCallId | 可由调用所在位置确定 | `message_id + block index` |
| proactive/source 执行分支 | 来源不应改变执行语义 | 普通 input Message + 同一 reactor |
| DeliveryId / conversation delivery rows | 投递不能复制消息身份 | `message_id` + provider projection/effect |
| ProjectionClaim | projection 不配拥有事实 | version + source seq cache |
| Core 中的默认 AgentLoop | 产品算法不可替换 | 普通 `default-agent` 插件 |
| `PassiveTurnPipeline` 固定 phase DAG | 把业务、持久化和外部效果绑死 | `MESSAGE_REACTOR` + 普通依赖/contributions |
| `PassiveMessageWorker` 大 owner | 同时拥有 ingress、reply 和 delivery | Channel adapter + plugin Effect + Delivery projector |
| Citation/Meme/ToolSearch 等 Core 特判 | 产品能力侵入基础设施 | 普通插件 contribution/service |
| 通用 metadata/context 袋子 | 隐藏第二套协议和 owner | typed Message content + 窄 capability |

最终不是“只做 WAL”，也不是“把旧 pipeline 移进插件目录”，而是三句可以独立验证的
规则：

~~~text
事实：Session = ordered Message WAL
行为：ordinary plugins read Message and append Message
视图：projection = fold(Messages up to source_seq)
~~~

Core 只保护 append、权限、外部边界和插件生命周期。它不再决定 Agent 怎样回答，
也不再为回答过程发明另一套可持久化名词。
