# Akashic v4：只有 Session 与 Message

- 文档版本：0902-reviewed-v4
- 日期：2026-09-03
- 状态：设计提案，等待维护者批准
- 当前代码基线：47896b4200731183a54081e2eca77602a0881a0a
- DSH 参考基线：49a606bc5b5934603f22a26957a07dc799ab0291
- 本文不授权：实现、数据库迁移、正式 workspace 写入、删除、部署或合并

## 结论

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

### 把我当六岁

Session 是唯一一本作业本。Message 是已经用墨水写完的一行字。

- 小朋友先在草稿纸上写，写错可以重来很多次。
- 只有一句话写完整，才抄进作业本。
- 抄进去以后，这一行有自己的编号 `message_id`，也有所在页码 `seq`。
- 聊天页、模型看到的上下文和“这一轮”的括号，都只是拿彩笔从作业本里画出来。
- 擦掉彩笔，作业本没有少东西；换一种画法，也不用迁移事实。

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

## 七、其余全部是视图

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

## 十一、目标写入路径

~~~text
Sources                         Session Message WAL
──────────────────────          ─────────────────────────────
Mobile user Message ───────┐    ┌──────────────────────────┐
Scheduler Message ─────────┼───▶│ append + seq + durability│
Assistant complete output ─┤    └────────────┬─────────────┘
Tool result Message ────────┘                 │
                                             ├──▶ Chat
                                             ├──▶ Model context
                                             ├──▶ Web/Mobile cursor
                                             ├──▶ Memory/Search
                                             └──▶ Delivery queue
~~~

公开原子能力保持很小：

~~~text
SessionStore.append(message, expected_head_seq?) -> committed Message
SessionStore.read(session_id, after_seq?, limit?) -> Message[]
SessionStore.head(session_id) -> seq
~~~

工具关联、删除应用、上下文选择和 UI 分组都是对 `read()` 结果的函数。插件只得到完成
任务所需的窄能力；projection 不持有任意 SQL 或删除权限。

## 十二、迁移路线

当前 `projectneed`、代码和数据库仍有 Turn/Run/attempt 合同。本文是替代设计，不允许
借普通重构偷偷改变线上语义。批准后按以下顺序迁移：

### Phase 0：冻结目标合同

- 批准“失败 attempt 不进入 Session”的信息损失；
- 批准单一 pre-commit `message_id`；
- 批准 stale assistant output 由 head CAS 丢弃；
- 批准 tool `unknown` 与不支持幂等 provider 的边界；
- 修改 `projectneed` 和相关 decisions，明确旧合同被替代。

### Phase 1：只读转换器

- 从现有 `sessions.db/messages` 和有必要的 tool 数据重建目标 Message WAL；
- 现有 message ID 原样保留；只有缺少 ID 的历史数据才在一次性迁移中分配；
- 对无法证明顺序、角色或 tool outcome 的记录 fail-loud，不能猜默认值；
- 固定真实 Session fixture，比对 Chat、Model context、Tool status 和 Mobile 输出。

### Phase 2：切换唯一 writer

- 所有 user、assistant、tool 和 scheduler producer 改走同一个 append；
- 输入 transport 直接复用 `message_id`，删除平行 retry identity；
- 模型 retry 保留在 pre-append 内存流程；
- projection 先 shadow rebuild，证明读取等价后再接管消费者；
- 不长期 dual-write 两套事实。

### Phase 3：删除旧权威结构

- 在可恢复备份和影响清单上删除旧 Turn/Run/Step/attempt/delivery Core rows 与 API；
- 删除 proactive 分支，让 scheduler 只产生普通 Message；
- 若 seq 发生变化，客户端执行一次明确 snapshot/cursor reset；
- 重建 projection，并核对每个 Session 的 message 数、ID 唯一性、seq 连续性、tool
  配对与 delete 结果。

迁移前后都不能改写正式 workspace，除非另有明确授权和恢复方案。

## 十三、验收 Gate

### 13.1 概念 Gate

- Core schema 和领域 API 只有 Session、Message；
- 不存在 TurnRef、RunRef、StepRef、DeliveryRef、ProjectionClaim 或 SessionEvent 壳；
- tool call 只用 Message 内可派生的 `call_ref`；
- proactive 不是类型、字段、状态机或特殊 commit 路径；
- 新增任何第三个权威名词时，必须先证明 Session/Message 无法表达其独立不变量。

### 13.2 WAL Gate

- 同 `message_id` 同内容重试只得到同一 `seq`；
- 同 `message_id` 不同内容 fail-loud；
- commit 前崩溃没有 Message，commit 后 ACK 丢失不会重复 Message；
- 每个 Session 的 seq 唯一连续，projection 失败不影响 commit；
- 正常路径没有 UPDATE/DELETE Message。

### 13.3 行为 Gate

- 模型断网和 partial stream 不写 Session；
- U2 抢先提交时，基于旧 head 的 A 被拒绝并重新生成；
- 两个并发 worker 只有一个 assistant Message 能提交；
- tool_call 先提交再执行，crash 后按 `call_ref` 查询或安全恢复；
- 不可确认的外部效果产生 `unknown`，不会盲目重复；
- `no_due` 不产生 Message；quiet 产生普通 `[no_reply]` assistant Message，Chat 隐藏；
- Web/Mobile 只用 `message_id + seq + cursor` 完成重复、断线和追赶；
- delete Message 到达后所有 projection 一致隐藏目标，旧 retry 不会复活它。

### 13.4 明确接受的非目标

- 不从 Session 审计失败 provider attempt；
- 不恢复半截 token；
- 不给执行尝试稳定 ID；
- 不保证外部 provider 的 exactly-once，除非 provider 提供幂等或查询合同；
- 不让 UI 分组成为持久身份。

## 十四、对 v3 和前一版 v4 的最终减法

| 删除的东西 | 原因 | 现在由什么承担 |
|---|---|---|
| MessageBody / SessionEntry 两层 | 同一正文两种 owner | Message WAL record |
| SessionEvent 壳 | Message 已经是日志记录 | Message |
| Turn / Run / Step | 把临时执行升级成事实 | head CAS + projection |
| Attempt 状态 | 失败重试发生在 append 前 | 内存 + telemetry |
| client/retry message IDs | 同一 Message 多个身份 | 唯一 `message_id` |
| ToolCallId | 可由调用所在位置确定 | `message_id + block index` |
| proactive 分支 | 来源不应改变执行语义 | 普通 input/output Message |
| DeliveryId / delivery rows | 投递不能复制消息身份 | `message_id` + provider 能力 |
| ProjectionClaim | projection 不配拥有事实 | version + source seq cache |

最终模型不是“把许多事件放进一份账本”，而是更直接：

~~~text
Session
  └── Message
        ├── user text
        ├── assistant text / tool_call
        ├── tool_result
        └── delete

除此以外，Core 内都只是读取这本账的方式。
~~~
