# Content / Wake 现有原子能力盘点与第一阶段设计

- 状态：accepted architecture / staged implementation
- 日期：2026-08-23
- 基线：`origin/main@9586a931fda5d1266d0449e44bcd569d9103d6fa`
- 关联决策：[0039 React 原子能力留在 Core，来源保持非特权](../decisions/0039-react-core-atoms-keep-sources-unprivileged.md)
- 语义修订：[0040 Wake duty gate 属于 Wake scoped react](../decisions/0040-wake-duty-gate-lives-in-scoped-react.md)
- 关联设计：[React Core、Scheduler 与 Subagent](react-core-scheduler-subagent.md)
- 本文权限：维护者已批准按本文分层实现、隔离 E2E 与最终删除旧 proactive island；正式 hua-home runtime 切换仍是独立 activation gate

## 1. 问题与用户意图

第一版讨论曾把“定时获取 Content”描述成 Content 邮箱自己的能力。这没有先充分使用已经合并的 v3 原子能力，也把三个不同问题放进了一个对象：

1. 来源什么时候读取 Fitbit、Calendar、Feed 或 GitHub；
2. 读到的条目怎样进入一个可恢复的邮箱；
3. 什么时候值得启动一次 Wake `react`。

修订后的判断是：**Content 本身不需要 Timer。** 来源插件与 Wake 插件分别使用同一个 Core one-shot Timer 原子，但各自只安排自己拥有的 deadline。Content 只拥有 durable inbox 与条目状态。

```text
┌──────────────────┐   poll    ┌──────────────────┐
│ Source plugin    │──────────▶│ external source  │
│ owns poll Timer  │◀──────────│ cursor / events  │
└────────┬─────────┘           └──────────────────┘
         │ submit
         ▼
┌──────────────────┐   snapshot / CAS / settle
│ Content plugin   │◀────────────────────────────┐
│ owns inbox facts │                             │
└────────┬─────────┘                             │
         │ non-authoritative hint                │
         ▼                                       │
┌──────────────────┐   optional scoped Turn      │
│ Wake plugin      │──────────────────────┐      │
│ owns Wake Timer  │                      ▼      │
└──────────────────┘                ┌──────────┐ │
                                    │  react   │─┘
                                    └────┬─────┘
                                         │ delivery receipt
                                         ▼
                                    user-visible message
```

这不是“Content 里放一个 Timer”的改名。两种 Timer 的 deadline owner 不同：

- 来源插件知道上游 cursor、限流、采样频率和 retry-after，所以它决定下一次 poll；
- Wake 知道 Content/Drift 何时值得再次检查，所以它决定下一次 admission；
- Content 不知道外部采样协议，也不决定什么时候执行 `react`。

## 2. 当前已经存在的 v3 积木

本节只记录基线代码已经提供的能力。每个后续方案必须先回答“为什么这些积木不能表达”，才能新增 Core 接口。

| 现有原子 | Core 唯一拥有的事实 | 插件能做什么 | 明确不能做什么 |
|---|---|---|---|
| `Context` / `Fiber` / `Effect` | Root 内注册、生命周期归属、逆序 cleanup | 注册 service、listener、task 和 cleanup | 不能把 cleanup 当成外部效果回滚 |
| `ServiceKey` + `provide/require/inject` | exact Root 内 capability 解析与依赖激活 | 提供和消费窄 typed service；缺依赖 fail-loud | 配置扫描顺序不承担依赖语义 |
| typed events | `emit/serial/parallel/transform/observe` 的 dispatch 与 listener owner | 表达有合同的通知、串行判断、变换和观察 | 普通 listener `return` 不能结束整个 Turn |
| `RUNTIME_STARTED / RUNTIME_STOPPING` | formal Root 开始接纳工作和停止排空的顺序 | 恢复持久 deadline、arm Timer、取消并等待本 Root task | candidate Root 不能启动真实 timer 或外部工作 |
| `TIMERS` | 一个带时区 deadline 的 one-shot wait、cancel、receipt 和 cleanup | 等一次；到点后由插件决定下一步并重新 arm | 不知道 recurrence、job、source、retry、Content 或 Wake |
| `SCOPED_TURNS` | exact snapshot lease、Turn admission、活进程 handle terminal、interrupt 和 cleanup | 递交普通 Message，配置临时 Prompt/Tool/Memory scope，复用同一 `react` | 当前不能在重启后按 accepted receipt 读取 durable Turn terminal，也不能显式要求 fresh logical interaction |
| `TurnExecutionScope` | 本次 Turn 的临时执行边界 | 限定 tool grant、精确预加载已授权 Tool、memory read/write、stateless、来源 | 临时 scope 不得改变全局 Tool 定义或反向改写 Session 保留语义 |
| `DELIVERIES` | 当前 Root 的完整逻辑消息发送边界和 provider receipt | 向正式 channel 发送一条完整消息并取得 delivered receipt | 当前没有 caller-supplied stable logical id 或跨崩溃 settlement ledger |
| `CONTINUATIONS` | 向既有 parent flow 投递普通 continuation Message | Subagent 完成后通知父流程 | 不是 durable mailbox，也不等于长期记忆 |
| `TOOL_CATALOG` | exact Root 的 Tool definition 与 bound handler | 普通插件登记 Tool；handler 捕获本 Root 私有 runtime | Core 不按插件名字寻找 handler |
| plugin data/workspace projection | generation/plugin 的窄持久路径与声明文件 | 插件保存自己的 cursor、inbox、trace、job state | 不能取得任意 workspace 或 Session repository |
| `BACKGROUND_JOBS` | committed generation job admission、durable outcome/retry ledger、可选 LLM/programmatic Turn | 运行需要这些完整语义的 interval/Core-event job | 不应成为简单 `poll → submit → re-arm` 的默认包装 |

### 2.1 已经发生的组合例子

Scheduler v3 没有取得 Core 特权。它组合：

```text
schedules.json
      + TIMERS
      + SCOPED_TURNS（仅 SOFT job）
      + DELIVERIES
      + RUNTIME_STARTED / STOPPING
      = Scheduler plugin
```

它自己拥有 cron、misfire、recurrence、`run_count` 和 job settlement。`TIMERS` 只等一次；Scheduler 在结算后计算下一次时间，再 arm 新的 one-shot。

Subagent v3 组合：

```text
spawn admission / profile / task dir
      + SCOPED_TURNS
      + TurnExecutionScope / ToolGrant
      + CONTINUATIONS
      = Subagent plugin
```

它没有第二个 ReAct loop。child 是同一个 `react` 递归产生的普通 scoped Turn。

这两个生产例子已经证明：recurrence 和递归执行不需要加入 Core 来源枚举。Content/Wake 第一阶段必须沿用同一种设计语言。

## 3. 修订后的 owner

| 事实 | 唯一 owner | 其他组件只获得什么 |
|---|---|---|
| 上游 poll cursor、next due、backoff、retry-after | 每个 source plugin | Content 只见已经规范化的 item |
| inbox item、revision、eligibility、selection、delivery/settlement 状态 | Content plugin | source 可 submit/ack；Wake 可读 snapshot 和请求 CAS transition |
| Content snapshot high-watermark | Content plugin | ContentGate 取得冻结只读 view |
| Wake debounce、下一次 admission deadline、Content/Drift 选择顺序 | Wake plugin | Core Timer 只等 deadline；Gate 只返回 proposal/decline |
| Drift durable due、cursor 和领域状态 | Drift plugin | Wake 读取窄 due/proposal capability |
| Turn admission、exact Root、`react`、terminal 和 cleanup | Core | Wake 只调用 `SCOPED_TURNS` |
| queued/in-progress Turn 的崩溃收敛与 durable terminal | Core ConversationRuntime / SessionStore | 当前普通插件没有按 accepted receipt 读取的窄口 |
| 当前用户可见 send 与 provider receipt | Core/Channel 的通用 delivery owner | Wake 提交完整消息并读取本次 attempt receipt |
| 跨崩溃 delivery identity、投影与 settlement 恢复 | 当前尚无完整 owner | 先由 S2 fixture 固定差距，再单独批准来源无关合同 |
| 上游 ACK 尝试、provider receipt、uncertain 状态 | source plugin | Content 保留 delivered/unsettled，直到 source 报告完成 |

Core 中不得出现 `if content`、`if wake`、`if fitbit`、Content 数据库 schema、source cursor 或 Wake gate 顺序。

## 4. Source 使用现有 Timer

每个来源插件只运行自己的一条串行 one-shot 链：

```text
RUNTIME_STARTED
      │ read durable cursor / next_due / ACK state
      ▼
arm TIMERS.schedule(next_due)
      │ fired
      ▼
drain delivered + unsettled ACK work
      │
      ▼
poll upstream with current cursor
      │
      ▼
Content.submit(stable event id + revision + payload)
      │ submit committed
      ▼
persist source cursor / next_due / backoff
      │ persisted
      ▼
arm the next one-shot Timer
```

顺序是不变量：

1. poll 返回带稳定 upstream event id/revision 的批次；
2. Content 幂等提交全部条目；
3. submit 成功后，source 才推进 cursor 并持久化 `next_due/backoff`；
4. source 状态持久化成功后，才 arm 下一次 Timer。

submit 后、cursor 前崩溃会重复 poll，但 Content 以 `source_id + item_id + revision` 吸收重复。cursor 先推进、submit 后执行的实现必须被 mutant 杀死，因为它会永久漏掉事件。

来源没有稳定 event id 时，由来源插件从上游 canonical payload 与 cursor 生成带版本的稳定 id；Content 不猜测外部身份。

简单采集默认直接使用 `TIMERS`。只有出现下列已经存在的需求时，才考虑 `BACKGROUND_JOBS`：

- 需要 Core-owned durable job outcome/retry ledger；
- 需要 invocation-scoped LLM；
- 需要 job 自己创建 programmatic Turn；
- 需要 typed Core-event admission；
- 需要跨进程恢复一个已经正式接纳的 job invocation。

仅仅“每五分钟读一次 Fitbit Monitor 并 submit”不满足这些条件。

## 5. Content 是没有时钟的邮箱

Content 的最小能力按消费者分窄，不向 source 暴露 Wake selection writer，也不向 Gate 暴露 ACK writer：

```text
CONTENT_SUBMIT       source → submit(items)
CONTENT_WAKE_READ    Wake/Gate → snapshot(), due()
CONTENT_SOURCE_ACK   source-bound view → unsettled(), ack(ref)
CONTENT_TRANSITION   Wake/delivery settlement → select/defer/delivered
```

这些是 Content 插件提供的普通 `ServiceKey` capability，不是 Core capability。source-bound ACK view 只读取注册 source 自己的 unsettled rows，不能枚举其他来源。是否必须拆成四个 public key，要先用跨插件 fixture 验证当前 loader 的 `inject`、exact Root 和热重载语义；若更少的窄 view/command port 已能保持相同权限边界，不再增加 facade。

### 5.1 条目状态

```text
submit
  │
  ▼
pending ──CAS select_batch(≤100 items, turn_id)──▶ selected batch
  │                                                │
  ├─ defer(not_before)                             ├─ completed + share(1..5) ─▶ ready_for_delivery
  │                                                ├─ completed + skip ───▶ release all to pending
  │                                                ├─ missing/conflict ───▶ deferred
  ├─ await_change                                  ├─ known retryable ────▶ deferred / pending
  └─ invalidated                                   └─ known nonretryable ─▶ invalidated / abandoned

ready_for_delivery ──generic delivery settled──▶ delivered ──source ACK──▶ settled
```

其中 Content 的 completed 分支必须进一步满足 typed decision：

```text
completed + share_content(message, items[1..5]) ──▶ one delivery; cited items only
completed + skip_content(reason)                 ──▶ release batch to pending
completed + missing/conflict       ──▶ deferred（零发送）
```

- `ContentGate` 只读冻结 snapshot，返回 proposal 或 decline；它不改数据库。
- proposal 后由 Content owner 用冻结页的 `item_ref + snapshot_seq/revision + accepted Turn receipt` 做一次批次 CAS selection，最多包含 100 个候选。CAS 冲突时本 Turn quiet abort，不进入 reasoner。
- `selected batch` 不是“已经消费”。Content selection ledger 保存 selection token、accepted Turn receipt、冻结成员与顺序。只有 durable Turn items 中恰好一个成功的 `share_content(message, items)` 才推进 `ready_for_delivery`；`items` 必须是冻结页内 1～5 个不重复 candidate id，整批只对应一条 logical delivery。投影成功后只有被引用成员进入 delivered/settled，未引用成员保持 pending。`skip_content(reason)` 释放整批到 pending，不 ACK、不发送。缺失、冲突或越界引用 defer。普通 `final_response` 只是内部诊断，不能进入 delivery。模型失败、Tool 失败、取消或明确 rejected 均由 Wake 根据 terminal receipt 向 Content 提交 retry/defer/invalidated；结果 unknown 时保持 selected 或对应 delivery uncertain 并进入可观察恢复，不得猜测超时后再选一次。
- `selected` 阶段冻结整页，避免同一候选同时进入另一 Turn；进入 `ready_for_delivery` 后只继续锁定实际引用的 1～5 个成员。provider `uncertain` 时这些引用成员不能重选或二次发送，未引用成员仍可在下一轮选择。
- Wake admission 的已见事实按稳定 source/item/revision identity 持久化。一次抽签只标记当时已经 due 的新条目，不能用全局 snapshot watermark 顺带吞掉 future `not_before` 条目。
- 兼容重构前行为时，Core 的只读 conversation semantic service 从最近 256 个完整非 proactive Turn 生成 prototype。Wake 对 due 候选合成 `1-(1-preprocess)*(1-semantic)`，同一份增强后的冻结页同时进入 hazard 与候选排序；主动投影不作为 prototype，空正文或无 embedding runtime 的语义分为零。
- 进程崩溃后的 selection 只按 durable Turn terminal 与 delivery settlement forward-complete。S1 必须先固定现有 Control recovery 的真实查询入口；若没有插件可用的窄查询能力，先用失败 fixture 证明，再评审来源无关的 Turn terminal read port。不得在 Content 中复制 Turn ledger。
- decline 不是一句日志。Content owner 必须提交 `defer(not_before)`、`await_change` 或 `invalidated`，再重算 `wake_needed` 与 `earliest_not_before`。
- `wake_needed` 是由 eligible 条目推导并持久化的领域事实，不是 Timer 状态。并发 submit 与重算必须以事务/CAS 防止丢更新。
- Timer fired、Turn admitted 或 provider completed 都不能提前清除 `wake_needed`；只有条目状态变化后才能重算。

Content snapshot 使用冻结 high-watermark。snapshot 建立后到达的新 item 留给下一轮，不能改变正在进行的选择。

### 5.2 持久状态的增、改、减

| 对象 | 正常增加 | 允许原位更新/逻辑失效 | 物理减少 | owner 与恢复证据 |
|---|---|---|---|---|
| inbox item/revision | `submit` INSERT 新 item/revision | pending、selected、ready-for-delivery、deferred、await-change、delivered、settled；invalidated/abandoned/expired 是逻辑失效 | 第一阶段无自动减少协议 | Content；完整 row、revision、snapshot_seq、accepted Turn receipt、事务 receipt |
| Content selection batch | selection 时 INSERT batch/member ledger | selected、released、ready-for-delivery、delivered/settled | 无自动物理减少协议 | Content；selection token、accepted Turn、冻结成员、引用集合、settlement receipt |
| Wake admission seen set | due 新条目完成一次 hazard 抽签后 INSERT stable identity | v1 watermark 只作旧状态兼容；新条目逐 identity 追加 | 无自动物理减少协议 | Wake；schema migration、SQLite integrity、future-due fixture |
| wake state | submit/transition 更新 `wake_needed`、earliest deadline | 只由 Content 根据 eligible rows 重算 | 不适用；是单例状态 | Content；重启扫描与 invariant query |
| source cursor/next_due | 成功 submit 后推进 | backoff/retry-after/last result 更新 | 第一阶段不自动减少 | source plugin；source 私有状态与 poll receipt |
| source ACK record | Content delivered 后 source 查询并建立 pending | provider_acked、content_settled、uncertain | 第一阶段不自动减少 | source plugin；provider receipt 与 Content ack receipt |
| Content delivered/unsettled | 通用 delivery committed 后更新 | source ACK 完成后 settled | 第一阶段不自动减少 | Content；stable delivery id、settlement_ref |

## 6. Wake 使用现有 Timer 和同一条 react

Content submit 提交后可以发布一个允许丢失、允许重复的 typed hint。hint 只降低延迟，不是队列，也不拥有恢复语义。

Wake 的工作是：

1. 在 `RUNTIME_STARTED` 读取 Content 的 `wake_needed/earliest_not_before` 与 Drift 的 durable due；
2. 取最早 deadline，用 `TIMERS` arm 一个 one-shot；
3. 新 hint 到达时，在 Wake 私有 runtime 中取消/清理旧 handle，再按最新事实重新 arm；Core Timer 不需要稳定业务 key；
4. Timer 到点后先重新读取 Content/Drift due；两者都不再 due 时只记录、重算 deadline 并 return，不创建 Turn；
5. 仍有 due fact 时，创建一个带 Wake `TurnExecutionScope` 的 scoped Turn；
6. 在 `turn.context_prepared` 的一个 Wake listener 内，固定先读纯 `ContentGate`，未命中再读纯 `DriftGate`；
7. proposal 经领域 owner CAS 成功后才让共享 reasoner/tools 继续；两者都 decline 时由 Wake 记录领域 transition，并使用现有 before-turn abort 路径安静结束；
8. Content proposal 的 Wake scope 精确预加载 `share_content` 与 `skip_content`；结算只读取 durable Tool call，绝不把普通模型正文当成用户消息；
9. fixture 必须证明 `skip_content` 即使伴随“过滤、不推”正文也产生零 outbound/零用户 Session projection，`share_content` 只发送其 `message` 参数，重启恢复仍读取同一 durable decision；
10. fixture 必须先证明 abort 不产生空 outbound、不写错误 Session Message、不触发不适用的 memory/after hooks。若做不到，停止并回到 Turn 合同，不新增 Core `Skip`。

```text
Wake Timer fired
      │
      ▼
recheck durable due ── none ──▶ record + re-arm + return
      │ due
      ▼
SCOPED_TURNS.start(Message, Wake scope)
      │
      ▼
turn.context_prepared
      │
      ├─ ContentGate proposal ──CAS select──▶ shared react
      │
      ├─ ContentGate decline ──▶ DriftGate proposal ──CAS──▶ shared react
      │
      └─ both decline ──domain transition + existing abort──▶ quiet terminal
```

外层 due recheck 仍是不调用 Turn port 就可以 return 的 admission gate；它只判断是否存在到期事实，不做内容选择。进入 Turn 后的 Content/Drift duty gate 属于普通 `react` 的 before-turn lifecycle。两者不重复拥有同一判断。

这项放置由 0040 明确修订 0039 中“全部价值判断都在 Turn port 外”的旧约束。Wake 是一个完整 scoped `react`；`before` 是它内部的 lifecycle 阶段，不是第二套 proactive loop。

当前 `TurnExecutionScope.tool_source` 只进入 Tool 调用归因，不会投影到 `BeforeTurnCtx`。但 `SCOPED_TURNS.start(channel=...)` 已把 channel 投影到 lifecycle context；Scheduler 也用自己的 channel 启动 scoped Turn。因此 Wake 使用 `channel="wake"` 分流 listener，`tool_source` 继续只负责工具归因，本阶段不新增 Core origin 字段。只有未来出现“同一 channel 内还必须区分 exact execution source”的真实案例，才重新评估来源无关的不可变 Turn origin。

配置 delivery target 后，Wake Turn 使用目标 conversation Session，而不是另建 `wake:*` 用户历史。其 scope 固定为 `stateless + session_history_read + memory_read + memory_write=false`：react 能读目标会话最近历史和 Core 记忆，但临时 Wake input、reasoning 与普通 `final_response` 不进入 Session messages 或 Akasha。只有 durable provider delivery 完成后的 proactive assistant projection 才追加到目标 Session；因此连续 20 条未回复 proactive 会保留为 20 个独立主动 Message，随后 `u → a` 仍只形成一个普通 Akasha interaction，不把前面的主动消息并入该 interaction。

另一个已证明的缺口与 origin 无关：`BeforeTurnCtx` 看不到当前 durable `turn_id`，而 Content/Drift selection 必须绑定 Core 已经接受的 Turn；活进程 handle 丢失后，`SCOPED_TURNS` 也不能按 receipt 读取 SessionDB 中已经收敛的 terminal。最小 Core 修补仍属于同一个 Turn 聚合：lifecycle 投影当前 `turn_id`，并让 `SCOPED_TURNS.read(accepted_receipt)` 返回 immutable durable Turn view。插件不得获得 SessionStore 或完整 ControlService。

Control 启动时会在插件启动前把遗留 queued/in-progress Turn 收敛为 cancelled/interrupted。owner 启动扫描据此 forward-complete selected：active 不重选；completed 重新读取 durable Turn items，合法 `share_content` 才进入 delivery，`skip_content` 释放整批到 pending，缺失/冲突决策 defer；cancelled/interrupted/明确 retryable failure 原子 defer；明确 non-retryable failure进入 invalidated/abandoned；receipt 指向缺失 Turn 时保持 orphaned 并发出 Incident。任何分支都不以经过多少秒猜测 terminal，也不重新解释 `final_response`。

固定 Wake session 还暴露了 fresh interaction 缺口：失败或中断后的下一次 Control start 当前会自动续接旧 logical interaction。如果下一次已经选择另一个 Content/Drift proposal，这会错误继承旧 attempt replay。因此同一 Core 修补必须让 scoped programmatic start 直接表达独立 Turn 边界，不能靠随机换 session 绕过并发 owner。

fresh admission 不是“忽略旧 predecessor”。Core 在创建新 Turn 的同一 SessionStore 事务中，把被替代的 recoverable interaction identity 作为新 Turn 的 append-only supersession edge 持久化，然后才发布新的 interaction identity。重启后的 continuation 解析必须识别这条 edge，不再续接已经被关闭的旧 interaction。该选项只属于 `SCOPED_TURNS.start` 的程序化边界；普通 passive `turn/start` 的 failed/interrupted 自动续接保持不变。没有消费者时不新增 service、manager、来源枚举或可变全局模式。

现有 abort 的真实边界也已查清：Wake listener 设置空 `abort_reply` 后，不进入 reasoner、Tool、after-reasoning、after-turn 或 channel delivery，Session messages 保持为空；Control runtime 仍保存一个 completed Turn、输入 item 和空 assistant item。这是 Turn 执行诊断，不是用户记忆。fixture 必须固定这两类状态，不能把“无 Session Message”误写成“什么都没记录”。

## 7. Delivery 与 ACK：现有积木暴露出的真实缺口

当前 `DELIVERIES.send(channel, chat_id, content)` 会发送并返回 receipt，但调用者不能提供稳定 logical delivery id，也没有一个公开 durable settlement owner 封住下面的窗口：

```text
provider 已经收到消息
      │ process crash
      ▼
Content 尚未提交 delivered
```

因此第一阶段**不能**把“消息只发送一次”列为现有能力，也不能通过在 Content 内增加一个布尔值伪装解决。故障 fixture 应先稳定复现这个差距。

真实 Wake 实现前需要单独批准一个 source-neutral delivery settlement 合同，至少表达：

```text
prepared → delivered → projected → settled
```

- caller 提供 stable logical delivery id；
- Core/Channel 通用 owner 保存可恢复 receipt；
- delivered 后再把 assistant Message 追加投影到 Session；
- settlement 重启只向前补做投影/领域通知，不重复发送；
- 目标 provider 不支持幂等且结果未知时进入 `uncertain`，不得自动重发。

这个合同只能使用 Message、Turn、delivery、receipt、projection 和 settlement 等来源无关词，不能出现 Content/Wake/Drift 特判。本文不批准实现该缺口。

### 7.1 上游 ACK

通用 delivery owner 先持久化 stable settlement event；Content 再按 `settlement_ref` 幂等 forward-complete 为 `delivered && unsettled`。两者是不同 owner、不同持久库，不宣称跨库原子事务。source plugin 在启动和每次 timer fire 时查询自己的 unsettled rows；hint 只负责加速。

source ACK 状态机是：

```text
pending
   │ call upstream ACK
   ├─ confirmed ──persist provider receipt──▶ provider_acked
   │                                             │ local Content.ack only
   │                                             ▼
   │                                       content_settled
   └─ unknown and non-idempotent ─────────────▶ uncertain
```

- `provider_acked` 持久化后，重启只重试本地 `Content.ack()`，不得再次调用上游。
- 上游支持幂等时，source 使用 `settlement_ref` 作为幂等身份重试。
- 上游不支持幂等且调用结果未知时进入 `uncertain`，保留诊断和人工协调入口，不盲目重试。
- ACK 失败或 uncertain 不会把 Content 从 delivered 倒退，也不会再次发送用户消息。

## 8. 修订后的第一阶段

第一阶段不再叫“给 Content 建 timer 基建”。它只做现有能力盘点与 characterization fixture：

### S0 · 文档与合同

- 固定本文的 owner、状态机、已确认能力和未知项；
- Core 预期 diff 为零；
- 不接触正式 workspace、旧 proactive DB、外部插件 canonical source 或 channel。

### S1 · 现有积木组合 fixture

在一次性 workspace 中安装三个最小普通插件：`fixture_content`、`fixture_source`、`fixture_wake`。只使用当前公开 v3 接口，验证：

1. `ServiceKey provide/require/inject` 不依赖扫描顺序，三者绑定同一个 exact Root；
2. `channel="wake"` 是否真实进入 `turn.context_prepared`；其他 channel 的 listener 领域写入为零；
3. source `poll → submit → cursor persist → re-arm`，重启重复 poll 由 Content 幂等吸收；
4. source-bound ACK view 不能读取或提交其他 source 的 settlement；
5. Content proposal 必须 CAS select；竞争失败不进入 reasoner；
6. decline 提交 defer/await-change/invalidated 后不会 hot-loop；
7. 丢掉全部 hint 后，Wake 仅靠 durable `wake_needed` 恢复恰好一次 admission；
8. Content/Drift deadline 取最早值，另一方状态不被 Wake 改写；
9. `RUNTIME_STOPPING` 后 timer、poll、ACK、Turn task 和 snapshot lease 全部归零；
10. candidate Root 的 timer、poll、external call、Turn 和正式写入均为零。

如果这些 fixture 通过，不能新增 Content capability 到 Core。如果跨插件依赖、exact Root 或 lifecycle 确实无法表达，先记录具体失败调用链和最小违反路径，再单独评审来源无关的 Core 修补。

### S2 · Delivery 缺口 fixture

使用 recording channel 在“provider delivered 后、Content commit 前”注入崩溃。当前实现预期暴露重发歧义；测试报告把它标成 `known_gap`，不能伪装为通过，也不能通过 mock success、跳过 crash 点或放宽 oracle 获得全绿。

只有 S2 证明差距并单独批准 source-neutral settlement 合同后，才进入 Core delivery 实现。S2 之前允许实现 fixture-grade 普通 Content/source/Wake 插件来复现差距，但不得迁移正式来源、激活生产或声称 exactly-once delivery 已成立。

## 9. 验收矩阵

| Case | 观察点 | known-wrong mutant |
|---|---|---|
| source 正常轮询 | submit receipt、cursor、next_due、一个新 Timer | cursor 在 submit 前推进 |
| source 重启 | 重复 poll、零重复 inbox item、旧 task/handle 为零 | Content 不做 idempotency |
| submit/hint crash | item 与 wake_needed 已提交；删除 hint 后仍恢复 | 把 hint 当 durable queue |
| 并发 submit/decline | snapshot high-watermark、CAS、wake_needed 不丢更新 | decline 只 return 不 commit |
| Content/Drift 同时 due | 固定 Content→Drift 顺序、首个命中、未选 owner 不写 | 依赖 listener 注册碰巧排序 |
| quiet Wake | reasoner/tool/delivery 为零；无空消息；允许的诊断写清楚 | listener return 被误当 Turn abort |
| delivery crash | stable identity、receipt、Session projection、是否重发 | provider 成功后直接重跑 send |
| ACK 首次失败 | delivery 一次、ACK 重试、最终 settled | ACK 失败触发再次 delivery |
| provider ACK 后本地 crash | 只重试 Content.ack，不再远端 ACK | 忽略 provider_acked receipt |
| non-idempotent ACK unknown | durable uncertain、无自动重试 | unknown 当 failure 盲重试 |
| Root reload | old drain 后 new 恢复；无双 timer/双 Turn | module global runtime 串 Root |

调试回执至少串起：

```text
source poll id
  → submit receipt / snapshot_seq
  → Wake timer receipt
  → scoped turn id / selected item refs
  → logical delivery id / provider receipt
  → Content settlement_ref
  → source ACK receipt / final settled state
```

## 10. ADHD 与 Concept Integrity 结论

ADHD 发散候选最终收敛为三类：source-owned Timer、durable fact + lossy hint、source-owned ACK state。被淘汰的主要陷阱是：

- Content 枚举并轮询所有来源：邮箱变成 scheduler 与外部协议 owner；
- 每次 poll 都创建 programmatic Turn：采集和“是否值得思考”被绑定；
- 用 `BACKGROUND_JOBS` 包装所有简单 poll：更厚的 ledger/LLM/Turn 语义被无条件引入；
- hint 成为权威队列：恢复事实分裂为数据库和事件总线两份；
- delivery receipt 返回后直接清 Content：崩溃窗口会丢投影、重复发送或漏 ACK。

Terra xhigh 的首轮 Concept Integrity Gate 找出了 duty gate 决策冲突、delivery owner 表述、lifecycle 来源和 ACK view 权限问题。本文已经按真实调用链修订；最终批准以当前 HEAD 的复审结果为准。

## 11. 停止条件

出现任一情况即停止第一阶段，不自行扩展实现：

- 需要 Core 识别 Content、Wake、Drift、Fitbit 或插件 ID；
- candidate Root 启动 timer、poll、Turn、delivery 或正式写入；
- Content 开始拥有外部 cursor、poll recurrence、Wake Timer 或 provider ACK 调用；
- source 在 Content submit 前推进 cursor；
- Gate decline 不产生领域 transition，导致重启 hot-loop；
- delivery crash 被测试包装成 success；
- provider ACK unknown 被自动重试；
- 在栈顶删除 Gate 通过前改动旧 proactive island，或在没有独立 activation gate 时切换正式 runtime、改写正式 DB。
