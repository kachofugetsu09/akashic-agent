# Akashic v4：消息日志与可组合的 Agent 链路

- 状态：设计已批准（2026-09-05 用户确认）；按 stacked PR 实施；本文件包含线上插件功能复核，生产实现尚未切换。
- 修订日期：2026-09-05。
- 源码与原提案基线：`51f1467456881e7302abf76a931e9dfe698fef6c`。
- DSH 参考基线：`49a606bc5b5934603f22a26957a07dc799ab0291`。
- 实施基线：`6a15444009c807994d33691e0b756167880fad5d`，worktree `message-plugins-stack`。实施按第 16 节分层；当前已实现边界见第 18 节，业务全量切换尚未完成。
- 已批准执行原则：回复业务由 100% 非特权插件组合；非灰度、非 shadow；可删除经核实不必要或冗余的功能，但逐项记录依据、影响、承接职责、验证与恢复点。开发可分批，正式运行采用完整新链路。
- 前版概念复核：独立 reviewer `/root/design_concept_review`，调用请求配置 `gpt-5.6-terra / xhigh`，2026-09-05 设计层 PASS；首轮九项 must-fix 已闭合。被审正文 SHA256 为 `64c30bf568ff66fc50a79c73e73cac9af156bacf73dfb5c2da9fa612ac3b03a4`。本次进一步明确 Core/普通插件边界与切换方式，前版结论不自动覆盖本次修订。
- 上次概念复核（不覆盖本次功能合同修订）：同一独立 reviewer 于 2026-09-05 对正文 SHA256 `2f06db73a347dd8f08b6292d6ae07caea8cc01cca5036cb828010140428967ff`（不含本条记录）给出设计层 PASS，无新增 P0/P1；可以开始实施准备，第 14.3 节合同与正式迁移/恢复验收仍未完成。

## 1. 目标与判断标准

完整移除 Core 中固定的 passive 回复业务链。默认聊天由普通插件组成；其中模型、工具、上下文和发送可以分别使用，替换 Agent 算法不需要替换这些能力。Message 独立保存，Turn 由消息投影得到，执行尝试不再成为对话领域对象。

正交性用变化的传播范围衡量。一个模块应拥有一项独立的设计决定、状态或真实边界；它内部的步骤共同完成这项职责，对外只暴露必要操作。函数少、目录多、只有一个 `react()` 入口都不能单独证明正交。

本设计按下列问题划分模块，不按 Before/After 执行阶段划分：

| 独立问题 | owner | 不应随它变化的部分 |
|---|---|---|
| 已经接纳了哪些事实？ | Session 消息存储 | Agent 算法、模型、记忆分组 |
| 新输入怎样触发、暂停和继续响应？ | conversation 等来源插件 | 模型协议、工具实现、Turn 投影 |
| 一次模型请求应该看到什么？ | context 插件 | 存储、渠道、循环算法 |
| 怎样完成一次推理请求？ | model/provider 插件 | Session、Turn、发送、记忆 |
| 怎样安全执行一次具体工具调用？ | 普通 Tool execution 插件与工具插件 | 模型、Prompt、循环算法 |
| 怎样把输出协议变成消息内容？ | content 插件及其解码器 | 工具调度、消息提交、发送 |
| 下一步继续推理、执行工具还是返回？ | 默认 ReAct 插件 | 核心存储、来源业务 |
| 怎样把一条消息送到某个目的地？ | 普通 Delivery 与渠道插件 | Agent 算法、Akasha |
| 哪些消息组成一个学习单元？ | Turn 投影与 Akasha | 写入授权、执行调度 |

不把序号分配、幂等检查、INSERT 各拆一个插件：它们共同保证一次消息提交。不把工具的权限检查、实际参数、外部执行和恢复拆给多个竞争 owner：它们共同保证一次调用的真实性。也不为普通 `while`、一次 `return` 或纯字段转换注册公共 Service。

## 2. 当前链路与问题

### 2.1 已核对的 Core 实现

```text
┌ Channel / Mobile / Web ingress ┐
└──────────────┬─────────────────┘
               ▼
      PassiveMessageWorker
      附件租约、入站 custody、按 session 排队、等待结果与 ACK
               ▼
      ConversationRuntime
      attempt 准入、输入暂存、工具 checkpoint、中断、实时状态
               ▼
      AgentLoop._react
               ▼
      PassiveTurnPipeline
      command → BeforeTurn → BeforeReasoning
               ▼
      DefaultReasoner
      Prompt、上下文 Gate、模型重试、工具发现与执行、停止策略
               ▼
      AfterReasoning
      引用/媒体处理、用户与最终回复批量持久化、构建 outbound
               ▼
      AfterTurn
      TurnCommitted、观察者、发送
               ▼
      worker 结算渠道 envelope 与资源
```

核心源码地图如下。行号是本次调查定位，符号与路径是维护入口。

| 现有职责 | 源码证据 |
|---|---|
| 入站 lane、附件、durable handoff、渠道结果任务 | [`PassiveMessageWorker`](../../bootstrap/passive_worker.py)，约 96、257、440 行 |
| 恢复 attempt 前驱、用户正文和工具轨迹 | [`ConversationRuntime._open_interaction_attempts / _attempt_user_inputs / _attempt_tool_chain`](../../agent/control/runtime.py)，651、715、871 行 |
| 精确中断、资源回收和实时 replay | 同文件 `interrupt_turn / _run / subscribe` |
| 固定阶段、命令短路、模型绑定 | [`PassiveTurnPipeline.run_command / run`](../../agent/core/passive_turn.py)，440、524 行 |
| 上下文投影、旧 attempt replay、工具预加载 | 同文件 `DefaultReasoner._run_turn_with_projection`，1157 行 |
| 预算收尾、空回复修复、结构化终态、工具名特判 | 同文件 `DefaultReasoner.run`，1506 行；`tool_search` 和 `message_push` 分支约 2064 行 |
| provider overflow 后强制 compaction、continuation 与 usage | 同文件 `_call_provider`，2698 行 |
| Prompt 合并、记忆/Skill frame、渠道 envelope | [`PromptAssembler.assemble`](../../agent/prompting/assembler.py)、[`ContextBuilder`](../../agent/context.py)、[`prompt_render`](../../agent/lifecycle/phases/prompt_render.py) |
| 单次工具 prepare、authorize、execute、observe | [`ToolExecutor`](../../agent/tools/executor.py) |
| reasoning 后暂存 user，最后批量追加 user/assistant | [`_PersistUserMessageModule / _AppendMessagesModule`](../../agent/lifecycle/phases/after_reasoning.py)，257、413 行 |
| 观察者先于出站的固定顺序 | [`_FanoutTurnCommittedModule / _DispatchOutboundModule`](../../agent/lifecycle/phases/after_turn.py)，243、346 行 |
| 连续 control ID 与 proactive 特殊分组 | [`logical_history_unit_ranges`](../../session/manager.py)，190 行 |
| 按全部 user IDs 与 final assistant ID 建立 Akasha 样本 | [`AkashaEngine._commit_source_event`](../../plugins/akasha/engine.py)，1211 行 |
| 插件发布借用父 Turn terminal 触发 | [`TurnPluginRollout.turn_terminal`](../../agent/plugins/turn_rollout.py)，205 行 |
| 来源无关的一次性等待、Scheduler 业务状态 | [`PluginTimers`](../../agent/plugin_composition/timers.py)、[`Scheduler plugin`](../../plugins/scheduler/plugin.py) |

Citation 与 Meme 的本地源码也显示了另一类耦合：Citation 改共享 `ctx.reply` 并提取引用，Meme 改同一正文和 media，Meme Prompt 还依赖 `citation.prompt`。检查路径是 `/mnt/data/coding/akashic-plugin/citation/plugin.py` 与 `/mnt/data/coding/akashic-plugin/meme/plugin.py`。这些 checkout 声明的是 V2 接口，只证明该份源码的依赖，不证明正式运行中的安装版本；迁移前必须核对正式 generation 对应的源码、cache 和实际消费者。

### 2.2 相对原提案的修改

| 原提案选择 | 本版选择与理由 |
|---|---|
| `responds_to` 同时表示关系与一次性结算 | 引用只表示引用。是否处理过归具体消费者；消息没有全局 consumed 状态 |
| 必须先从 TURN_VIEW 取得 exact causes 才能运行和提交 | Agent 直接使用日志；Turn 只供读者分组，不进入写入授权和调度 |
| `CauseOpen`、`NoNewHumanInputAfter` 等提交条件 | 默认输出使用来源内 head CAS；存储无需理解 Human、passive 或 Turn |
| 一条 Reaction 最多追加一条 Agent Message，工具结果再触发 Reaction | 默认循环可执行多步；每条消息及时提交，循环中断后从日志和 effect 事实恢复 |
| 每条输出只允许一个工具调用以省去 call 身份 | 支持多个调用；调用地址是消息中的内容块位置，不为省字段改变模型能力 |
| 将整个默认算法放入 AGENT_PROGRAM | context、model、tool、content 各自有可独立使用的接口，ReAct 只组合它们 |
| Turn grouping 统一控制全部上下文与学习消费者 | 共享日志；确实需要同一学习定义的消费者共享一个投影版本 |
| 为压低概念数而禁止运行事实出现在独立 owner | 保留无法由对话还原的外部 effect、来源业务和发布凭据；不复制正文 |

## 3. Message 与 Session

### 3.1 唯一对话记录

```text
Session = session_id + 按 seq 排序的 Message

Message {
  message_id
  session_id
  seq
  recorded_at
  author
  source
  body               # 有明确 schema 的消息内容
}
```

- `message_id` 是提交前已知的稳定身份，用于去重、幂等和引用。
- `seq` 由 Session 存储在提交时分配，单调且不复用。过滤来源后的 seq 允许有间隔。
- `recorded_at` 表示持久接纳时间；它不能冒充外部事件原始发生时间。外部发生时间若有意义，由对应消息类型保留。
- `author` 是实际作者/执行者的引用，可以指人、Agent、Tool 或应用服务；不是 provider role。
- `source` 是同一 Session 中稳定的来源流标识，见下节。
- `body` 是封闭校验的类型，不增加任意 `meta/mode/variant` 配置袋。新增类型由声明它的能力拥有 schema。

Session 可拥有名称等独立元数据，但不拥有“当前 Turn”“当前模型请求”或一份工具链正文副本。消息表是领域上的追加日志；底层仍可使用 SQLite，不需要另建 AOF 文件或双写 SessionEvent 表。

默认内容类型示意：

```text
Input       { parts }
Output      { parts, finish: continue | complete | quiet }
ToolResult  { call_ref, outcome: success | denied | error | unknown, parts }
Control     { action: pause | resume | abandon | failure, through_seq, reason? }

parts = text | artifact_ref | tool_call | citation | 其他已声明内容类型
tool_call = { binding_id, arguments }
call_ref  = { message_id, part_index }
```

Input/Output 表示消息用途，不替代 author：应用可以提交待处理输入，人类之外的来源也可以产生 Input。工具正文是否成功由 ToolResult 表达，不从自然语言猜测。`quiet` 是生产者明确选择无可见回复；空 provider 输出不是 quiet。`failure` 只记录已观察到的失败，不谎称 Agent 生成了一段错误正文。

`binding_id` 指向能力 owner 保存的不可变工具绑定，包含工具、generation、参数 schema 与授权依据；它不等于可重新解析到最新版的工具名。调用提交前确定该绑定。各内容 owner 的纯 schema 检查返回 `ContentReferences(binding_ids, artifact_ids)`，无引用则两项为空 tuple。binding 按集合合并；artifact 严格保留本块正文中的出现次序和重复项，不额外附带无关资源。`artifact_ref` 正文只保存 ID；不可变元数据唯一归附件表拥有，reader 按消息返回完整引用。日志不识别具体内容 kind，在同一事务检查附件已 ready，并将有序引用及 ToolCall.binding_id 一并保存，避免 call 已落盘但 receipt 尚未 prepare 时热更丢失旧实现。执行前仍检查当前权限；旧授权记录不能绕过后来撤权。

Control 的目标由消息自身的 Session/source 与 `through_seq` 共同确定：仅影响该来源截至此序号的未完成输入及其已提交调用，不是“执行时碰巧存在的任务”。来源 owner 校验目标属于当前可控制的前缀；旧 scope handle 只作接纳时的并发条件，不落成持久运行身份。完整控制规则见 7.4。

`model.facts` 是普通 ContentPart，由 Model owner 定义 schema，用于保存实际模型归属、usage 和 provider 要求的 opaque replay 数据。组合只向模型 Output writer 授予此内容类型，Input、ToolResult 和普通命令回复没有该 grant；模型 replay 与预算读者只消费经此合同接纳的事实。Core 不为它新增字段或专属授权。它不是第二份 history。provider role、请求 envelope、缓存 key 由模型 adapter 从有效内容投影；外部 Input 不能仅凭作者或来源取得 system/developer 权限。

### 3.2 来源与作者独立

`source` 表示输入与响应共同所属的消息流，不是“哪一个函数调用了 append”。例如 Human、Agent、Tool 都可以属于 `conversation`；Wake 工作属于 Wake 选择的流。它由来源插件的配置/既有身份建立，恢复时保持稳定，Core 不枚举其名称。

默认会话来源插件命名为 `conversation`，替代 passive。Scheduler 可使用既有 job identity 区分独立来源；同一来源的响应按序处理，多个来源可以并行。不能每遇到并发就临时生成一个 source，再宣称没有 RunId：只有独立、实际存在的来源才值得有这项身份。

`source` 不决定访问权限，也不限制 context 只能看该来源。conversation 可以看到已进入共同 Session 的 Wake 消息；看见或引用 P 不意味着 P 要被并入该次学习样本。

同一来源中任意多个任务乱序完成，无法仅由来源与顺序无损反推归属。本版默认来源采用有序响应；确有乱序任务的插件使用其已有任务引用声明归属并提供对应投影，不让 Core 猜，也不先增加通用 Run 树。

### 3.3 写入与追赶

消息写入只有一个 owner。公开能力按权限分别授予 read、append 或特定 call result 的写入权；普通算法不获得完整 SessionStore、任意 SQL、管理删除或任意渠道发送权。

```text
reader.read(after_seq, through_seq?, source?) → immutable Message[]
reader.follow(after_seq)                       → committed Message feed
writer.append(message_id, body, expected_source_head?) → receipt
```

writer 已绑定 Session、source、author 及允许的消息类型，调用时不反复传这些身份。身份由可信准入/组合边界签发，插件不能靠 body 自授权限。ToolResult 写入额外绑定 exact call；不把六类业务 writer 工厂变成核心领域。

一次事务依次完成：身份与 schema 校验 → 相同 ID 幂等核对 → 可选来源 head 检查 → 序号分配 → INSERT → commit。同 ID、同不可变内容返回原 receipt；同 ID 不同内容失败。先处理同 ID 重放，再判断 head，避免 ACK 丢失后的重试误报冲突。

- Input 在接纳时持久化，不等模型或最终回答。
- Output 在内容完整、协议解码及附件持久化成功后逐条追加。
- 工具请求在执行前追加，结果完成后单独追加；不再嵌进最终 assistant 的 `tool_chain`。
- token delta 是临时预览；完整提交前不作为可恢复 Message。预览丢失不等于已接纳输入丢失。
- 接纳 ACK、回复完成、渠道送达是三个不同事实，不能用一个 terminal flag 代替。

`follow` 是持久日志的追赶接口，callback 只负责降低唤醒延迟。commit 后崩溃、通知重复或断连都按 seq 重读。启动先建立通知订阅再读取一致 head，重读到该 head 后继续追赶；缓冲溢出明确要求重扫，不能依赖没有丢过通知。消费者保存自己的 cursor/规则版本，cursor 不推进到尚未成功应用的位置。副作用消费者按自身幂等/receipt 协议提交，不能由 cursor 假装保证 exactly-once。

## 4. Turn 是读者的分段

### 4.1 默认定义

令 `L_s` 是按 source 过滤后的有序消息。以 `Output.finish=complete|quiet` 的 seq 为回答结束点；`Control.abandon` 在其 `through_seq` 处关闭被放弃的前缀，该段不是成功学习样本。相邻结束位置为 `a[k-1]` 与 `a[k]`：

```text
候选段(s,k) = 该 source 中 seq ∈ (a[k-1], a[k]] 的消息
Turn 主体    = 候选段中的 Input / Output
工具观察     = call_ref 指向该主体内调用，且在该段结束前提交的 ToolResult
```

Control 不充当用户正文或 Akasha 学习正文。没有结束点的尾段保持 open；暂停和一次 provider 失败不把它拆成多个 Turn。正常完成前需结算全部待等调用，因此普通对话仍等价于按相邻最终回答分段。

被放弃调用的晚到结果继续保存并显示在时间线上，但不进入新 Turn、默认学习样本或当前 context 尾段；诊断按已有 call_ref 找到其原调用。例：`U1 → call1 → abandon(through=call1) → U2 → result1 → A2`，新段主体是 `{U2,A2}`，不能把 result1 当作 A2 的工具观察。这项关联是 call/result 已有的必要事实，不新增 TurnId。仅靠到达顺序无法同时处理这种交错；本设计不声称纯区间足以覆盖任意取消。

```text
全局日志                          默认投影
1 U1  conversation ─────────┐
2 U2  conversation ─────────┼── {U1,U2,A1}
3 P   wake complete         │
4 A1  conversation complete ┘

wake 的投影：{P}；全局时间线中的 P 仍只保存一次。
```

工具请求是 continue 输出；所有本次需等待的工具结果处理完后才产生 complete/quiet。直接推送可以是一条独立来源的 complete Output。最终回答的 message_id 可作为分段锚点，不创建新的 Turn row 或随机 TurnId。

### 4.2 投影与执行无依赖

Turn 投影只读日志，不产生 pending cause、不授权输出、不决定取消或工具执行。conversation 和 ReAct 可以读取同一条明确的 finish 事实，但不调用 TURN_VIEW 来获得执行权。

UI 按 seq 显示实际时间线，group 只是标签。Akasha 使用固定版本的学习投影，引用 exact member IDs 和 final message ID。在线与离线使用同一规则、文本连接、向量聚合和 provenance。多个用户输入仍能形成一个样本；中止、quiet、failed、abandoned 是否进入哪类学习由 Akasha 明确配置，默认只有完整可学习对话进入普通问答样本。

Turn Projection 是独立普通插件，只提供无状态消费能力，不建表、不保存消息正文、cursor 或学习状态，也不拥有持久投影缓存。它对调用者给定的不可变日志前缀执行分段，返回消息引用与 open/complete/quiet/abandoned 状态。消费进度由各消费者保存。持久学习样本可用 `(policy_version, ending_message_id)` 定位；这是 Akasha 的样本身份，不进入 Message。更换投影版本不重写消息，也不自动清空已学习权重、embedding 或反馈。迁移这些派生但有历史意义的状态需要单独的对照与恢复协议。

Context 可选用同一 Turn 分组来保持完整历史单元，也可以使用适合模型协议的切点；它必须保留工具请求/结果配对，不能因为换了学习分段而生成非法模型请求。

## 5. 完整目标拓扑

```text
┌ 渠道 adapter ┐  ┌ Wake / Scheduler / 其他来源 ┐
└──────┬───────┘  └────────────┬───────────────┘
       └───────────┬──────────┘
                   ▼
        ┌ Session 消息存储 ┐◀─────────────────────────────┐
        └─────────┬────────┘                              │
                  │ committed read/feed                   │
       ┌──────────┼───────────────────────┐               │
       ▼          ▼                       ▼               │
 conversation   Turn / Akasha          Delivery            │
 来源响应策略    学习与历史投影          route + receipt    │
       │                                  │               │
       ▼                                  ▼               │
 默认 ReAct 插件                         渠道发送            │
       │                                                  │
       ├── context.build ── Prompt / recall / compaction    │
       ├── model.complete ── provider                      │
       ├── content.decode ── typed Output ── append ───────┤
       └── tools.execute ── tool plugin ── result append ──┘

横向底座：组合与权限、取消/任务 scope、generation lease、Timer、Artifact
```

箭头是明确调用或已提交事实的读取关系，不是一串全局可变 hook。依赖注入只解析依赖，不自动决定业务顺序；ReAct 的控制流由普通程序直接写出。

### 5.1 分层与可独立调用的接口

以下是职责级 API 草图，优先复用现有类型与 Service；不是要求为每一行创建新 class 或 package。

Core 需要补齐或改造的能力限定为四组。插件提供业务能力，Core 提供来源无关的存储、权限、组合和资源机制。

| Core 能力 | 公开合同 | 与现有能力的关系 |
|---|---|---|
| Message 日志 | 窄 read/append/follow，幂等、顺序和条件提交 | 改造已有 Session 存储；不再等待最终 transcript 批次 |
| 通用 Task scope | start/cancel/join、排他 key、资源释放与 writer 失效 | 复用现有 Fiber/任务机制；移除 Turn、Prompt 和 terminal tool 业务字段 |
| 受限持久化事务 | 获授权的消息写入与 owner 状态可以一致提交 | 补齐 cursor、工具回执、发送回执所需合同；不开放任意 SQL |
| 能力绑定与耐久资源引用 | 固定实际实现，跨重启保留未结算调用所需 generation | 扩展已有 binding/generation 机制；Core 不解释工具或发送状态机 |

服务注册/依赖解析、事件、Timer、Artifact、进程与 Workload 能力继续复用。事务只接纳各自有权验证的操作：消息 writer、owner 状态写入与资源引用各有明确授权，统一提交不授予任意跨插件写权。插件定义自己的 receipt schema 与状态转换，Core 只保证存储与提交；不能把一张业务表移到 Core 再称其 owner 已插件化。

状态访问权由组合配置授予，不能写死为某个内置插件 ID。替换有持久状态的实现时，先核对其公开状态合同；兼容者获得同一状态访问能力，不兼容者使用显式迁移。未完成调用仍引用原绑定，不能因更换插件把旧回执当成空状态重跑。此处事务参与、状态授权与 durable binding 的具体合同列为实施前置项，见第 14 节。

| 能力 | 最小调用 | owner 与真实边界 | 独立使用的例子 |
|---|---|---|---|
| Session | `read / follow / append` | Core 保存不可变消息、原子提交、稳定身份 | 只接收与同步消息，无 Agent |
| Context | `build(snapshot, model, tool_view)` | 普通 context 插件；Prompt、检索结果、历史和容量组成一份请求 | 一次性总结、多个 Agent 算法复用 |
| Model | `complete(request, scope)` | 普通 models/provider；协议、认证、流、usage、可重试传输错误 | 分类、embedding 以外的单次推理，无 Session |
| Tool | `execute(request, scope)` | 普通 Tool execution 插件拥有准入、prepare、调用、回执与恢复；具体工具提供领域动作 | 无 Session 的命令或固定流程直接调用 |
| Content | `decode(output, references)` | 普通 content 插件；输出协议、引用与媒体解析 | 模型输出和导入输出使用同一内容格式 |
| Delivery | `send(message_id, sink)` | 普通 Delivery 插件拥有 route、发送状态与恢复，普通渠道 adapter 实现外部协议 | 已保存消息的通知、多目的地发送 |
| Task scope | `start / cancel / join` | Core；有界任务、资源、取消与代际保护 | 后台监控、定时等待、工具任务 |
| Timer | `wait(deadline)` | Core；一次等待 | Scheduler、退避、Wake |

模型接口已经存在于 [`BoundChatModel.complete`](../../agent/plugin_composition/models.py)，本版复用它，不再包一层同义 LLMPort。旧 ToolExecutor 的领域动作迁到普通 Tools 注册和执行能力；旧 Turn 上下文、可变事件与编排入口在消费者切换后删除。

Tool 提供 `execute(key, binding, args)` 与 `execute_call(reply)` 两个入口，共用同一执行 owner。前者接收程序自有的持久 key；后者从已提交 Output 的 call_ref 读取绑定和原始参数，调用者只提供结果 Message 身份与已获授权的读写口。两种 key 使用分开的空间。对话调用以 ToolResult Message 为唯一结果正文，receipt 只存结果指针；独立调用由自身 receipt 保存结果，调用程序按需要另行产生面向用户的消息。记录位置在开始前固定，不能在恢复时变换位置或生成新 key 再执行。工具不要求程序伪造 assistant 输出。

Core 持有权限授予、资源访问和原子存储的硬边界；Tool execution、Delivery、provider、具体工具、渠道均由普通插件实现。内置插件与仓库外插件使用相同的权限、安装链和生命周期合同，不存在 builtin 专享 writer、私有业务入口或执行顺序。可信进程内插件的窄接口用于减少误用，不伪装成恶意代码沙箱；不可信执行仍须经过进程/OS 隔离边界。

普通插件之间的依赖有明确公共合同，但不要求每项合同各装一个包；同一职责内的函数保持内聚。Core 不直接导入默认业务实现作为 fallback。缺少必需服务时组合明确失败；没有安装自动回复程序时，来源插件仍可使用 Core 的接纳与同步能力。没有来源接纳 provider 时明确拒绝渠道输入，Core 不代替它选择来源或签发业务 writer。

### 5.2 conversation：只拥有输入响应策略

conversation 订阅输入和控制事实，决定何时启用默认程序、哪些补充输入一起处理、是否暂停、怎样恢复。它取得一个来源内唯一的活动 scope，再把只读日志、该来源 writer 和选定程序所需能力交给程序。

它不拼 Prompt、不循环 provider、不执行工具、不学习记忆、不直接发送回复，也不等待这些观察者完成后才 ACK 输入。command 匹配属于其输入解释；命令处理器通过领域能力完成工作，返回明确结果，conversation 将必要的回复追加为 Output。

`/stop` 是控制操作：在同一来源准入边界内验证活动 handle、固定 through_seq 并追加 pause，再取消和等待 scope。Input 持久接纳、scope 替换、控制接纳及 effect start 必须参与这个边界的排序，不能“先检查 handle，再无条件写 pause”。`resume` 不复制输入。默认新 Input 唤醒尚未完成的来源；明确 abandon 才放弃指定前缀。暂停意图持久记录，重启不能仅凭 open tail 自动复活。

无已知指令的正文进入默认 Agent。未知命令、拒绝、失败保留可区分结果；不能使用 `handled=true` 或空返回偷偷消费输入。无外部副作用的只读命令不必创建工具调用；有副作用的命令须使用该领域持久 receipt，同一命令重放不能重复产生效果。

命令结果也有稳定提交身份，绑定原输入与 handler；无可见回复则追加 quiet。effect 已成功而回复未提交时，从领域 receipt 恢复同一结果，再按同一 message_id 提交，不能重跑副作用或追加第二份答复。

入站直接调用普通来源能力，不再经过内存 inbound queue 或 `PassiveMessageWorker`。ChannelHost 从已取得的 exact snapshot 调用 `CHANNEL_INPUT`，来源插件提交 Input 后，传输 owner 完成 custody 与 ACK。Mobile 在提交前已有的耐久 handoff 继续保留：崩溃恢复使用相同身份重新接纳，Input 已存在时直接结算，不能重启模型。普通输入不再占用“直到最终回复才释放”的 passive lane 计数。

来源接纳与自动回复分开安装：`conversation` 提供输入、控制和同来源 Task 准入；`reply` 追赶日志并选择默认程序。`Conversation.start(program)` 不把程序绑定在接纳对象上，关闭自动回复仍可保存输入和历史同步。这里修正原验收中“关掉 conversation”的包名说法，保留关闭自动回复而不丢输入的行为。启动的程序接收 `(task, reader, source)`；它取得 Content 与 Model 资源后自行签发 Output writer，紧接着同步登记 Task 撤权。Source 无需在同步准入中打开异步 Content 资源。

### 5.3 ReAct：只拥有循环决策

```text
读取当前来源的可继续状态
        │
        ├── 有已提交但未结算的调用 → Tool owner 恢复/等待
        │
        ▼
context.build → model.complete → content.decode
                                      │
                           追加完整 Output
                             ┌────────┴────────┐
                             ▼                 ▼
                          tool calls       complete/quiet
                             │                 │
                        tools.execute         return
                             │
                         追加结果
                             └──── 回到读取
```

循环内部只保留下一步选择、迭代/预算限制、是否接受当前输出、需要模型纠正还是结束。`model.complete` 与 `tools.execute` 都能绕过 ReAct 单独使用。一次性 Agent 只做一次 context/model/content；固定工具程序只做工具调用；它们不模拟一次 ReAct 或伪造 Turn。

空回复纠正、必须以某工具提交决定、超预算后的收尾，属于具体 Agent 算法的策略。默认实现可以在自己的普通函数里处理；不为每个分支建立 RETRY_POLICY、TERMINAL_POLICY、STEP_EXECUTOR 等只有一个消费者的接口。真正有独立替换需求时再暴露对应算法边界。

transport retry 由 Model owner 实施；context overflow 是明确的可恢复错误，由 ReAct 请 context 用同一输入快照准备更小请求，最多按已声明预算重试；Tool effect 恢复由 Tool owner 实施。这三类重试不进入一个万能 RetryManager。

预算必须注明适用范围。单次请求内的纠正/传输重试计数可以在内存；默认来源的已完成循环步数从最后 complete/quiet 或 abandon 边界后的 Output 重建，不因 scope 替换归零。若配置跨重启的模型调用/费用硬上限，额度 owner 在发请求前耐久占额，Model owner 保存实际调用、binding、usage 或 unknown；未知费用不得算作零。调用方用来源与边界消息引用绑定额度，Model 只接收额度凭据，无需理解 Session/Turn。没有这套 provider/query 证据时，只能承诺本地调用次数限制，不能声称精确费用硬上限。

Tool Search 自己提供候选 schema 和选择状态。ReAct 只取得 tool view，不识别字符串 `tool_search`，更不写 `message_push._commit_role=passive`。工具发现不等于授权；Tool 执行边界始终验证真实权限。可见工具集、LRU/preload 属于发现插件，catalog 注册与 exact binding 属于能力 owner。

### 5.4 Context：组成请求，不管理执行

Context 接收固定的日志快照、模型容量与可见工具描述，产出完整请求。它组合 Prompt、当前输入、历史、检索材料和 compaction checkpoint；不修改权威消息，不运行业务工具，不发送消息。

Prompt 的角色权限、人格、任务和候选材料由组合配置明确。来源文本与检索材料保持低信任；实际 system/developer 内容来自有权限的 Prompt provider。独占内容有单 owner；需要顺序时显式声明内容依赖，而不是沿用 `citation.prompt` 一类无关插件名称依赖。

纯投影是“给定日志、配置和已取得材料，得到相同请求”。记忆检索或摘要生成不是纯读取：recall 的反馈/查询 ticket 归记忆插件；compaction 生成新摘要并持久发布后，再构造请求。不能声称调用一次非确定 LLM 即可从原文逐字重建旧摘要。

Compaction 与 Markdown memory 保持已确认的独立 owner。摘要 source IDs、范围、模型与摘要本身持久保留；新的规则只改变后续请求视图，不删除历史。缺少受保护的模型 replay 数据或必要 call/result 时明确失败，不降级成空历史。Prompt warmup 归 context 缓存，预热失败不得被当成一份已准备请求。

### 5.5 Content：完整消息只有一个最终组装者

Citation、Meme、媒体与文本清理不能全部回流到 ReAct，也不应成为提交后的 mutator。

content 插件把完整模型输出转换成 typed Output。具体解码器只处理自己声明的协议，例如引用标记或 meme 标记，返回不可变的内容片段与引用；统一组装者拥有片段范围、重叠与最终顺序。Meme 解码器不改 Citation 的结果，Citation 也不清理任意其他协议。解码器存在实际顺序依赖时，优先收拢共享语法的解析责任，不用数字 priority 掩盖冲突。

图片/文件由 Artifact owner 导入并确认 durable，Output 才能引用；随机选图的实际选择落入 Output，重建不重新抽签。引用只能指向允许引用的真实材料，不能仅凭模型给出的 ID 赋权。现有从 recall 结果推断 cited IDs 的行为须列入迁移样例，明确保留或替换，不能漏掉。

provider 输出到中立模型响应的转换归 provider；引用/meme 等产品协议到 Message 的转换归 content。直接产生 typed Output 的程序可跳过文本解码，避免所有插件都被迫输出特殊标记。

### 5.6 Delivery：消费消息，独立结算

发送键是 `(message_id, sink)`。选择哪些消息、哪些目的地由普通 delivery policy 决定；默认发送完整可见回复与明确通知，不发送 Input、工具请求、ToolResult、quiet 或控制正文。中间进度是否发送是显式产品选择，不由 `assistant` 身份推断。

正文与附件只按 message_id 从 Session/Artifact 读取，不在 delivery 表复制。实际目的地、adapter generation、provider receipt 与发送状态属于发送 owner，无法从消息存在这一事实推导。

发送成功不能被 Akasha 失败回滚；Akasha 慢也不阻塞发送。若某来源确实要求送达后推进 ACK，例如 Wake，它直接等待 Delivery receipt，再推进自己的业务状态。发送失败可重放同一消息的同一 effect；不重跑模型生成另一条回复。

Delivery 独占以下提交顺序：按生效策略选出 sink → durable prepared，固定 message_id、实际地址、adapter generation 与幂等 key → durable started → 外部发送 → delivered/rejected/unknown。prepared 与来源消费进度在本地事务或明确的 durable handoff 中提交；多 sink 的选择集合也一并固定。策略热更不能在旧消息重扫时生成另一组发送；追加新目的地必须是明确的新发送动作。

started 后缺回执时，只能查询远端或在其支持幂等时用原 key 重试，否则记 unknown，不能把超时当 rejected。cancel 先于 start 则不发出，start 之后只能如实结算。未解决发送耐久持有原 adapter generation 和地址；凭据撤销时暂停并报错，不换一条路线重发。来源 ACK 只消费已确认 delivered 或该来源明确接受的失败处理结果；unknown 不算送达。

### 5.7 取消、流与 generation：围绕真实资源

Task scope 是通用、短命的运行对象，可有一个跨重启失效的 handle 来精确取消；它不是持久 Attempt，不拥有对话正文或逻辑分组。旧 UI handle 不能取消新的 scope；传输请求的幂等 identity 也不成为对话 RunId。

预览流绑定该 scope 与预分配 message_id。重试时撤掉旧草稿；完整 Message 提交后，以 message_id 接替预览。reconnect 从 durable seq 重读，再获取当前活动 scope 快照，不恢复任意旧 token stream。

每次模型调用与工具调用固定实际使用的 generation。默认程序可在一个活动 scope 内固定模型选择；新输入导致新 scope 时按模型插件规则重新绑定。Core 只保护 lease，不把“同一个 Akasha Turn 必须用同一 generation”写成规则。

插件安装、重启等操作保留已有 domain journal。发布的验证、切换和恢复由 operation owner 统一处理；等待真实使用旧 generation 的资源排空，不依赖学习 Turn 结束。模型/工具生命周期与热重载交叉部分在迁移中单独验收，不能把取消内存指针当作已回滚外部服务。

## 6. 去掉 Attempt 后，职责放在哪里

现在的 Attempt 并非完全无用：它暂存尚未进入 messages 的输入，保存工具执行 checkpoint，承担精确取消与 UI replay，还被发布流程引用。真正冗余的是把这些不同职责收进一个对话实体，再与 logical interaction 互相转换。

| 旧 Attempt 职责 | 新 owner 与恢复依据 |
|---|---|
| 暂存用户正文、ordinal | 接纳时追加 Message；顺序由 seq 给出 |
| 累积 assistant/tool transcript | 每条 Output/ToolResult 及时追加；历史按日志读取 |
| provider 第几次尝试、计费和失败 | Model 调用诊断/usage 记录，不参与 Turn 归属 |
| 活动任务、取消、等待和资源清理 | 短命 Task scope；需要跨重启保留的暂停/失败意图写 Control |
| 工具可能已经产生外部效果 | Tool 的调用 receipt 与不可变请求/结果 |
| 实时 token/item replay | 可丢预览加 durable Message 同步 |
| 插件更新的 owner_turn_id | 既有发布 operation 自己的 journal，引用发起调用与实际资源排空事实 |
| 学习单元与所有输入的归属 | 版本化 Turn 投影和 exact source Message IDs |

不新增 Run、Reaction、Interaction 等同义持久对象。存储可以有来源 head 索引、工具 receipt 和消费者 cursor：前者是派生索引，后两者各有不同 owner，均不复制消息正文，也不能彼此替代。

## 7. 并发、提交和恢复

### 7.1 用来源 head 拒绝过期输出

定义 `head(session, source)` 为该来源最后一条已提交消息的 seq，可由日志重建并由索引加速。默认程序读取一致快照时取得 `h`，追加输出时要求当前来源 head 仍等于 h。比较与 INSERT 在同一事务完成，不是先读后写。

```text
U1(conversation) → 读取 h=1，开始模型请求
P(wake)         → conversation head 仍是 1
A(conversation, expected_source_head=1) → 可提交

U1(conversation) → 读取 h=1，开始模型请求
U2(conversation) → conversation head 变成 2
旧 A(expected_source_head=1) → conflict，未进入日志，也不执行其工具
新请求读取 U1/U2 → 新 A 可提交
```

这是通用来源版本检查，Core 不知道 Human、pending cause 或“回答了谁”。默认部署只有一个持有 workspace 排他锁的运行 authority；第二个独立 runtime 必须启动失败，外部进程通过该 authority 的接口接入。锁归 OS/进程生命周期，崩溃后可重新取得；本版不声称支持多个节点直接共写一个 workspace，也不增加分布式 lease 系统。

来源插件用 `(session, source)` 作为通用 Task admission 的排他 key。authority 串行完成启动、scope 替换与控制接纳；旧 scope 退出/完成结算前，不启动同来源的新程序。回复 writer 同时绑定该活动 scope，提交事务先验证 writer 仍有效，再做来源 head CAS；scope 失去所有权后即使 head 没变也不能提交。ToolResult 使用 exact call 的结算 writer，不能因原 scope 结束而丢弃真实效果。Core 只理解排他 key、writer 生命周期与版本条件，不解释来源名称。

进程死掉不证明远端模型请求已停止。恢复先查询 Model 调用记录；没有可查询结果时明确记录 unknown，按实际调用预算决定是否准许新请求，不能承诺网络故障下绝无重复费用。cursor 只推进已完成接纳或已持久移交的进度，不能用“已读消息”代替工作完成。

所有会影响该来源回答的 Input/Control 都进入该来源，不能靠 callback 先后改变事实顺序。跨来源 P 若只增加可读背景，不使当前回答作废；确实依赖全局最新快照的程序可使用全局 head 条件，额外隔离由调用者主动选择。

### 7.2 工具等待与新输入

工具请求一旦提交就是事实。新输入可以取消过期的模型草稿，却不能抹掉已经开始的工具效果。默认程序对该来源的已提交调用先等待/恢复，之后读取全部新输入，再构造下一次请求；final 提交前该来源没有需要等待的 unresolved calls，且来源 head CAS 通过。

同一 Output 支持多个 tool_call。结果按 `(message_id, part_index)` 关联，实际完成顺序由 seq 保存；模型 adapter 可投影为其协议允许的配对顺序，不改日志。工具是否并发由工具执行策略决定，不为了省一个随机 ID 禁止批量调用。

真正脱离等待的后台 job 由对应工具返回已创建的 job 引用，本次 call 已结束；job 将来的通知是一条新 Input，不伪装成同一个 call 的第二个 terminal result。

### 7.3 工具执行事实

普通 Tool execution 插件对稳定调用 key 保持一个 receipt。对话 key 是 call_ref，独立程序的 key 来自其领域请求；两者固定实际参数、授权/binding 与 effect 是否已跨过外部边界。模型提出的参数保留在 call 消息中；经过合法 prepare 得到的执行参数保存在 receipt，这是不同事实，不是正文副本。

```text
call 已提交
    ▼
prepared（最终参数、exact binding）
    ▼ durable start intent
started ── 外部调用 ──▶ result / unknown
```

- 当前没有 receipt：在验证权限、控制状态和已耐久固定的 binding 后可以 prepare；缺失旧绑定必须失败，不重绑最新版。
- 已有 terminal ToolResult：返回原结果，不执行。
- 有 started 而无结果：优先 query；provider 支持相同幂等 key 才可重试；否则产生真实 unknown 并停止自动继续。
- 工具异常只能由能解释它的边界转成 denied/error；内部不变量损坏 fail-loud。
- 对话 result 追加与本地 receipt 的结果指针在同一存储事务完成；独立调用在其 receipt 提交结果。不同存储时必须有已验收的 outbox/handoff，不能默认跨库原子。
- 取消与 effect start 由执行 owner 排序：取消先被接纳则不新发起；已开始则结算为真实结果或 unknown，不能假称没执行。

unknown 是自动执行路径的终止结果，不证明远端失败。人工核对后产生新的明确管理事实/新获授权调用；不改写原 unknown，也不以后台重试偷偷重复效果。未处理的调用与 receipt 持久引用其 exact generation，进程重启后按该引用打开所需目标。内存 lease 排空后释放资源，耐久归档没有自动 GC，也不另存 active claim 或 refcount。

Tool 与 Delivery 都需要外部效果记录，但各自拥有不同状态和查询协议。暂不创建统一 EffectManager；相同 SQL/锁 helper 只有在实现重复且合同相同时才共享。

### 7.4 取消与重启不自动复活工作

控制的 owner 是来源插件；同一来源的接纳排序由存储/任务能力支持，不依赖 Turn 投影。Control 的 through_seq 必须引用已有同来源消息，不能覆盖未来输入。四种动作的规则如下：

| 动作 | 指定前缀内的未完成输入 | 已提交工具调用 | through_seq 之后的新 Input |
|---|---|---|---|
| pause | 暂停，正文保留 | 未开始的不发起；已开始的继续真实结算 | 可按来源策略唤醒，旧 pause 不覆盖新输入 |
| failure | 记录失败并暂停 | 不把未知外部效果当失败重跑 | 与 pause 相同；具体错误仍归实际 owner |
| resume | 恢复尚未 abandon 的输入 | 先处理原调用状态，不复制调用 | 不改写它们原有状态 |
| abandon | 明确放弃，在 through_seq 关闭该前缀 | 未开始的结算 denied/cancelled-before-start；已开始的保留真实结果或 unknown | 保持独立，不能被并入放弃范围 |

默认 conversation 的唤醒策略在本层明确为：`resume` 或新的 Input 都恢复该来源尚未关闭的工作，先按当前权限恢复原 prepared 调用，再读取全部输入继续推理。暂停期间只到达 ToolResult 不会唤醒；unknown 不因新 Input 获得重试授权。这保持来源有序，不建立“已经 final，却另有旧调用等 resume”的并行工作模型。后文“保留待 resume”也包括该默认来源由新 Input 明确唤醒后的恢复。

pause/failure 后到达的 ToolResult 不解除暂停；abandon 后的 late result 不进入新段，按 4.1 的 call 归属规则处理。暂停尚未开始的调用可以保留待 resume；abandon 则必须明确拒绝再启动它。新输入不会自动授权重试 unknown 效果，解除该阻塞需要工具领域的明确核对结果或新授权。

UI 的 scope handle 和来源 head 前置条件在控制提交时一起核对；过期则返回 conflict，不落控制事实。旧 scope 在失去所有权后报告的失败只进入该调用诊断，不能再写当前来源的 failure。发生重启后，仅按已接纳 Control 的持久边界恢复。若产品以后需要“忽略所有后续输入，直到解除”的整个来源开关，它属于来源插件配置；不能把一次 /stop 偷换成这种模式。

活动 scope 与并发 admission 本身不持久化为 Attempt。进程重启时，来源插件从未完成输入和控制事实决定是否恢复；Tool/Delivery 先处理已开始的 effects，随后才允许运行新的决策。对无法确定的效果保持 unknown/需处理状态，不能用重新请求模型掩盖它。

### 7.5 已存在的独立发布与重启协议

发布 journal、child 验证证据、parent operation、endpoint 切换、服务停止与恢复各有真实外部事实，不能随 Attempt 表删除。复用发布领域已有 operation 身份，创建时就在其 journal 固定发起 call/message、目标 generation、变更内容与恢复点。工具返回的是“已接纳 operation”，不是“发布成功”；它因此可以结束自身资源占用，不会等待自己正在持有的 generation 排空。

发布 owner 持久记录 `prepared → validated → switching → committed` 或明确失败。验证必须来自指定 candidate 的真实证据；资源排空是独立前置条件。唯一发布提交点由该 owner 在核对实际 endpoint 与 generation 切换后记录，不能由 Task 结束、模型最终回答或 child 自称成功替代。跨服务切换不是数据库原子操作：journal 在 I/O 前记录切换意图，重启后查询真实 endpoint/进程，完成切换或按已有恢复协议记录回退；状态未确定时保持 switching/unknown。

提交点前取消由该 owner 停止后续动作并执行可证明的恢复；已跨切换边界但状态未知时先核实，不能直接标 cancelled。提交后取消只能成为明确的 revert 操作，不能改写已发生的发布。调用方用 operation receipt 查询 pending/terminal；完成通知作为发布来源的新消息，经正常 Delivery 发送。child 验证属于 operation，不需要伪造用户 Turn。

自动重启同样在 supervisor journal 中先保存请求与恢复点，再由 supervisor 执行停启并核对新进程。重启前确认应保存的消息/effect receipt 已持久；重启后报告实际结果。新一版文档批准前仍执行既有协议；本设计不授权提前替换安全闸门。

## 8. 被动链每项职责的去向

这张表是迁移账单。删除旧类必须以对应行的行为验收完成为前提，不能以“已经有一个新 react()”结案。

| 当前行为 | 目标 owner / 具体处理 | 主要验收 |
|---|---|---|
| inbound 身份、route、认证 | 渠道 adapter + 窄身份/Session 准入能力 | 同输入重放只一条 Message，越权拒绝 |
| 附件上传、导入与租约 | Artifact；Input/Output 只引用 durable artifact | commit 前后 crash、hash 与引用完整 |
| 队列、容量与接纳 ACK | 传输准入保障容量；接纳后以日志为待处理事实 | 不接纳则明确拒绝；已 ACK 可重启找回 |
| 每 Session/来源响应 owner | conversation + 通用 Task scope | 并发输入、重复唤醒、来源间并行 |
| command 与 slash short-circuit | 来源插件匹配，命令 handler 使用领域能力 | 无模型仍可执行；副作用重放不重复 |
| 中断、失败、继续、放弃 | 来源控制事实 + scope | 旧 handle 不影响新执行，停止不删输入 |
| Session/history 装载 | Session reader + context | 消息即时可见，不从 Attempt 补正文 |
| 人格、规则、任务、Skills、host hints | Prompt/context 的明确内容贡献 | 权限与顺序可解释，删无关插件不改变其他内容 |
| Prompt warmup/cache | context 私有缓存 | 丢缓存只影响性能 |
| Akasha recall 与反馈 ticket | Akasha 普通能力，context 取得结果 | 同输入重试不重复强化；ticket 不依赖 Attempt |
| history compaction、provider overflow | context + compaction；ReAct 有限重试 | 原文零减少、摘要先 durable 后使用 |
| MEMORY/SELF 更新、旧 PENDING receipt | Markdown memory owner | 沿既有 before-image/receipt 恢复，无任意正文写权 |
| 模型选择、generation 与 credentials | 已有 models/provider | 单调用 binding 不漂移，切换不改 Session |
| streaming、usage、provider continuation | model 协议；scope 预览；日志保存必要 replay facts | 断流不产生完整消息，usage 不伪造 |
| 空回复修复 | ReAct 普通策略函数 | empty 与 quiet 可区分，重试有界 |
| 最大迭代、terminal tool、最终收尾 | 使用该算法的插件 | 一次完整调用流程，不新建 Core source 分支 |
| Tool schema、preload、LRU、Tool Search | catalog + discovery 插件 | 更换发现算法不改执行器或循环 |
| 参数 prepare、授权、调用、真实结果 | Tool owner + 普通 Tool | 未授权零调用；真实参数、effect、result 一致 |
| 多 tool call 与并发/取消 | Tool 执行策略 + call_ref | 部分成功、晚到结果、未知效果不重复 |
| 长 tool result 与 artifact | Tool/Artifact 保存全文，context 选择展示范围 | provider 视图裁切不损失原始结果 |
| message_push、主动输出 | 工具/来源插件追加目标消息，Delivery 发送 | 不注入 passive commit role；原 call 与目标发送分别结算 |
| subagent 与 background job | 普通插件 + Task/process + 可选子 Session | 子程序可组合模型/工具，父子取消和结果真实 |
| Citation、协议标记与 fallback IDs | content 的引用解码器 | 引用合法，现有来源提取行为有明确去向 |
| Meme、媒体、文本清理 | content 解码器 + Artifact | 标记不泄漏、随机选择固定、不依赖 Citation 顺序 |
| 输出持久化与 UI 完成 | writer；按 message_id 连接预览和完整消息 | ACK 丢失幂等，完成早于慢观察者 |
| TurnCommitted、Akasha learning、索引 | committed feed + 版本化投影 | 在线/离线一致；观察者失败不回滚消息或发送 |
| 出站 route、发送、provider ACK | Delivery + adapter | 多 sink、部分失败、unknown、重试同一 effect |
| Wake ACK、cooldown、schedule 状态 | 各来源自己消费 delivery receipt | 送达与业务结算不互相改写 |
| plugin install/revert 与 self restart | 各 domain journal + 资源 lease | 不再靠逻辑 Turn 判定安全；真实服务恢复证据 |
| shell/PTY、MCP、workload cleanup | 实际资源 owner，scope 协调 join | cleanup 失败保留 owner，不报成功回收 |
| 诊断、日志、timeline、预算统计 | 明确调用事件与消息/effect 投影 | 按实际模型/工具调用定位；无 UI 反写 |
| 撤销与删除 | 显式数据管理边界 | preview、备份、消息及学习来源完整性 |

本次对 Core 主链做了源码检查；外部已安装插件全集、正式数据 schema lineage、设备协议和 crash/provider 行为尚未验收。上表是待迁移职责完整账单，不是现有插件全部安全可删的证明。

## 9. 持久状态的增、改、减与恢复

| 对象与 owner | 正常增加 | 原位更新 / 逻辑失效 | 物理减少与恢复 |
|---|---|---|---|
| Session Message | 每次接纳 Input/Control、完整 Output、ToolResult 追加 | 正常路径不改正文；既有撤销/删除仍执行 SES-003，未定义新逻辑失效标记 | 仅显式数据管理协议可物理减少；先整库/附件备份并验证引用、seq、数量与 hash |
| Session metadata / source head 索引 | 创建 Session、首次来源 | 名称按用户动作更新；head 与 INSERT 同事务推进，可从日志重建 | Session 明确删除；索引重建不删消息 |
| Artifact | 内容导入并耐久发布 | 已发布字节不改；引用随消息/管理事实变化 | 无自动 GC 新授权；孤儿识别、备份和独立删除协议 |
| Tool receipt | call 准备与执行产生 | 状态单向推进、真实执行参数与 binding 固定 | 未解决效果不得自动减少；依据 call/result、provider query 和备份恢复 |
| Model 调用/额度记录 | 实际请求前占额、调用与计费事实 | usage/失败/unknown 由实际回执推进；绑定固定 | 对账与保留由 Model/额度 owner 负责；重启不清零硬限额 |
| Delivery receipt | 为具体 message/sink 创建 | prepared/started/terminal，保存实际 provider 回执 | 未解决效果不得自动减少；不由来源 ACK 清除 |
| UI/搜索投影缓存（不含 Turn 插件） | 读取新的日志前缀计算 | 用版本/来源范围失效并重建 | 在不承担唯一历史事实时可重建；不影响 Message |
| Akasha vectors/学习/反馈 | 已批准学习与反馈产生 | 沿既有学习 owner 的更新与来源失效协议 | 不把学习状态统称缓存；删除/重建先核对固定输入与保留义务 |
| compaction/Markdown receipts | 新摘要、before-image、applied receipt | 摘要按来源失效，Markdown 沿现有 receipt 更新 | 无新自动减少授权；从已提交摘要、before-image 和备份恢复 |
| consumer cursor | 消费器首次订阅 | 只在成功应用后推进；有副作用时与 receipt 协调 | 卸载不能顺带删除业务事实；重扫必须幂等 |
| 来源 schedule/cursor/ACK/quota/journal | 各业务动作 | 各 owner 的既有状态机 | preserve/replace/retire 逐项批准，不能因 Core 删除 proactive 分支清库 |
| 发布/重启 journal | 明确的外部管理操作 | 按实际验证、进程/服务切换推进 | 保留现有恢复证据；迁移不自动减少 |
| scope、token 预览、内存队列 | 活动进程创建 | 活动期更新，结束后清理 | 可丢；任何已接纳输入或未解决 effect 都有其他 durable owner |
| 旧 turns/attempt 表 | 切换后停止新写 | 迁移前只读保留，未落 messages 的已接纳输入需要显式转换 | 验收和用户数据管理授权前不得 DROP/DELETE；代码删除不是删除历史授权 |

本提案不改变既有 SES-003 撤销/删除路径，也不承诺一种尚无耐久管理事实的“读视图自动失效”。独立 Message 的新删除合同必须另行设计：明确管理 owner、固定批准集合、备份、停止写入的顺序、晚到结果处理及重建行为，不能用无限引用闭包扩大范围。删除回复不能重新激活旧输入，隐藏不能冒充外部撤回。第 10 节 B 阶段切换前须批准该合同；未批准前不能把现行 interaction 删除机械套到新日志。

## 10. 实施顺序与退出条件

设计方向与重构已获用户批准，按下列顺序准备与实施。开发可以分批提交，但不引入灰度、shadow、双执行或长期双写。正式切换前旧 release 继续服务；新链路在隔离环境和受控 fixture 上验收，正式环境最终只有新 writer 与新链路。旧 release 与数据备份用于显式整套恢复，不成为请求级 fallback。

正式切换要求停止新工作接纳与调度、结算或明确冻结未完成外部效果、完成一致备份及迁移，再启动新链路。恢复必须成套匹配代码、插件与数据；已经发生的远端效果按 receipt 核对，不能靠恢复旧文件假称已撤回。本次只读核查没有执行这些动作，也没有证明正式备份可恢复。

### A. 固定行为与删除边界

逐行补齐第 8 节 baseline fixture，包括正式 generation 的插件源码、当前 Session/attempt 状态和客户端身份协议。为语义变化列 preserve/change/retire。新旧 Prompt、正文、引用、附件、tool calls、ACK、effects 和恢复必须可独立比较。

退出：每项现有行为有 owner、样例和验收；已确认故意变化单独列出。没有运行证据的动态消费者仍列未知，不删除其入口。

### B. 先让 Message 独立成立

先以独立合同和隔离 fixture 验证接纳时保存 Input、工具请求/结果与完整输出分别追加、作者/来源/内容类型与稳定身份；此阶段不接管正式入口、不改变运行库 schema。新算法直接读取新日志，不添加旧 hook 兼容壳或双 writer。保留原始工具全文与所需 provider replay facts。

退出：`U1/stop/U2/P/A1`、ACK loss、call/result、重启后输入保全成立。旧 attempt 字段不再是任何新正文的唯一来源。schema、历史转入及 yoyo 随引入实际持久变化的 PR 提交并隔离验收；整个栈全部 review 后统一合并和发布，最终 writer、全部消费者和 Delivery 一起接管。

### C. 拆开模型、工具、上下文和内容能力

从 DefaultReasoner 中移走 provider 调用协议、Tool执行与恢复、context 准备、content 解码；依次用单次推理、无模型工具程序、不同 context 策略、Citation/Meme 样例证明可独立使用。工具 discovery 同期去掉字符串特判。

退出：ReAct 只保留循环决策和自身有限策略；这些能力无需 Turn、passive worker 或 ReAct 即可调用。没有把一整段 DefaultReasoner 迁成一个同样复杂的 service。

### D. 消息投影接管学习与历史

使用固定输入与明确预期分组验收默认 Turn 投影；历史数据副本用于迁移和重建演练，不接入实时 shadow。Akasha 与重建使用同一版本和 exact source IDs；UI 按 seq 同步。compaction 独立证明模型协议与保留切点。

退出：更换学习投影不影响输入准入、模型调用次数、工具执行和输出可提交性。移除新路径的 control_turn_id 依赖。

### E. conversation 与 Delivery 接管外层

普通 conversation 插件消费日志并启动默认程序；渠道在接纳后 ACK，发送由 Delivery 独立消费。接管停止/继续、命令、流完成、容量、来源 head CAS 和资源收束。

退出：外部插件能替换默认 conversation/Agent；关掉默认 reply/ReAct 后入站保存和历史同步仍可工作；关掉 Akasha 不影响回复送达。发送重试不调用模型。

### F. 收回旧运行身份与特权链

发布/重启/子任务 owner 脱离 Attempt，工具和发送恢复完成后，移除 ConversationRuntime 中 attempt transcript 状态及旧 Worker/Pipeline 的固定业务编排。Task、Timer、generation、Artifact、工具/渠道安全边界保留各自实际职责。

退出：bootstrap 只组装能力与默认插件；Core 没有 passive/proactive、具体工具名或插件 ID 分支。跨仓库安装、热更、关闭、恢复和真实 provider 验收完成后才删除旧接入点。旧运行数据保留至独立迁移/管理批准。

## 11. 验收：证明变化互不牵连

### 11.1 正交性验收

| 改变一件事 | 应变化 | 必须保持 |
|---|---|---|
| 换 Turn 分段规则 | 分组与学习样本 | 消息字节、seq、调用轨迹、写入权限 |
| 换 provider | 请求/响应 adapter 与 binding | 来源响应、工具调用合同、Delivery |
| ReAct 换成单次推理 | 默认算法插件 | Session、context、model、content、Delivery 接口 |
| 换成固定工具程序 | 组合与程序 | 不需要伪造模型请求、Turn 或 Attempt |
| 更换 Tool Search | schema 选择 | 工具授权与执行恢复 |
| 启用/关闭 Meme 或 Citation | 对应 Prompt/内容能力 | 无关解码器、循环、消息存储 |
| 启用/关闭 Akasha | recall 和学习 | 输入接纳、输出提交、发送 |
| 增加一个渠道或发送目的地 | adapter/route | 一条 Message 及其原始身份 |
| Scheduler 调整 cron/misfire | Scheduler 状态 | Timer、模型、工具、Turn |

### 11.2 行为与故障样例

1. 两次输入被中断，中间插入独立来源输出，最后一个回答：完整消息仍在，默认分组得到 `{U1,U2,A1}` 与 `{P}`。
2. U2 与旧 A 同时提交：事务先后唯一决定结果；U2 先提交则旧 A conflict，不能漏读 U2 却将其计为已处理。
3. P 在模型执行时提交：默认来源 CAS 不冲突，时间线不重排。
4. call 已提交、工具尚未开始时崩溃；工具已开始、结果尚未保存时崩溃；结果已保存、通知前崩溃：分别证明可开始、query/idempotency/unknown、复用结果。
5. 一个输出含多个 call，结果乱序与取消交错：调用地址不串，已执行效果不重跑，未执行的明确停止。
6. pause 后重启、late ToolResult 到达：保持暂停；旧 handle 在新 Input 后提交控制须 conflict；resume 不复制输入。`U1/call1/abandon/U2/result1/A2` 中 result1 不进入新段。
7. 输入/输出 commit 成功但 ACK 丢失：同 ID 重试取回原 seq，内容不同则拒绝。
8. context overflow 与摘要提交窗口：原始消息无减少；已使用摘要可恢复；不能默默降到空 Prompt。
9. empty provider 输出、quiet、工具拒绝、provider failure、scope cancel 均可区分；不同 owner 不互相伪造成功。
10. Citation/Meme 共存、单独卸载、 malformed 标记、媒体导入失败：真实内容和影响范围明确。
11. Akasha observer 延迟或失败：Message 和 Delivery 不倒退；重放样本不重复学习。
12. 两个 sink 一成功一失败：分别记录，重试只操作失败 effect；Wake ACK 独立恢复。
13. 旧 generation 有已提交 call 或未完成 delivery 时热更/停用：引用保留、权限不扩大、不切到新代码偷偷重试。
14. plugin publication、self restart、shell/PTY cleanup 失败：报告真实资源状态，不由“Turn complete”伪装全部完成。
15. 只读历史、分组、compaction、记忆查询拿不到任意 SQL、删除或发送权限。
16. 同一套公开能力由仓库外插件经 install → candidate → stable → reload → uninstall 使用，Core 没有 builtin 特权补丁。
17. 无 Session 的固定程序调用 Tool，effect 成功但返回前崩溃：复用其领域 key 和结果；不得创建伪造 assistant 消息。
18. 反复在模型请求发出后、Output 提交前崩溃：有硬限额配置时仍计入 durable 占额；没有精确计费证据则不声称已知实际费用。
19. 删除默认 Conversation/ReAct 后 Core 可启动并提供消息接纳/同步；由仓库外插件经同一公开安装与权限合同恢复完整响应行为。
20. 替换有未完成回执的 Tool execution/Delivery：不得取得任意他方状态权限、重绑旧调用或重复发送；状态合同不兼容则明确停止并要求已定义的迁移。

采用受控并发和显式 crash point，不用 sleep 猜测竞态。上述是实现验收要求，本次文档检查不证明它们已经通过。

## 12. 已批准方向与待落实的规格变化

本提案优先实现用户本次要求的独立消息与完整插件化；它不能被当作普通兼容 refactor。

| 当前合同 | 提案变化 |
|---|---|
| SES-001、SES-008 最终 transcript batch | 每条接纳消息独立 durable；事务原子性仍保证单条或确需原子的批次 |
| SES-007 与 decision 0025 的 attempt continuation / fresh-after-failure | 用来源 Input/Control 表达继续、暂停和放弃；默认失败后补充输入继续 open tail，放弃需明确动作 |
| SES-003 整个 interaction 删除 | 当前路径保持；新 Message 管理协议独立批准后才能切换，不能从 Turn 投影推断删除权 |
| decision 0034 持久 logical Turn / Attempt 双身份 | Turn 成为版本化投影，活动任务使用短命 scope |
| MEM-010/011 统一 interaction 身份与分组切点 | Akasha 固定学习投影；其他读者按各自合同读取同一日志 |
| decision 0039 Core 提供完整 scoped react | Core 提供独立存储/资源/安全能力，默认 ReAct 为可替换普通组合 |
| decision 0026 发布由父 Turn terminal 触发 | 发布 operation 等待真实资源排空与验证事实 |
| decision 0050 模型绑定时机 | 复用模型插件 owner，将绑定生命周期与学习 Turn 脱钩 |
| decision 0052 compaction / Markdown 插件 | 保留两者独立性及持久化保护，仅替换来源与读取接口 |
| 既有 Mobile/control identity 与 replay | 使用 Message 同步和活动 scope handle，旧协议切换需单独版本与客户端验收 |

用户已批准本设计方向、剩余重构和有记录的冗余功能删除，不再重复请求架构批准。实现前把对应长期条款与 decisions 同步为一致合同；尚未定义的 Message 删除、迁移和恢复协议须补成可审阅的具体方案。代码/功能删除许可不等于自动减少正式会话、学习状态、附件或旧插件数据。

## 13. 参考与取舍

- [Parnas：模块分解标准](https://www.cs.lafayette.edu/~gexia/cs301/resources/parnas.html)：按需要隐藏的设计决定划分模块，而不是直接沿流程图切段。本版用第 1、11 节检验变化的传播。
- [Ousterhout：模块设计](https://web.stanford.edu/~ouster/CS349W/lectures/abstraction.html)：小接口应承载完整职责。单次工具执行保留准入与恢复的一致 owner，不拆成一串浅包装。
- [ReAct 原论文](https://arxiv.org/abs/2210.03629)：推理、行动、观察交替。论文不规定持久 Turn、Attempt 或插件 API；本版的分层是工程设计推导。
- [Maka 的日志设计](https://github.com/apache/maka/blob/main/docs/blogs/log-is-the-runtime.md)：采用事实与读取投影分离；外部恢复仍需实际 execution facts，不宣称聊天消息能还原全部系统状态。
- [Pi 新 harness 的 Session/Branch/Lane 分离](https://github.com/earendil-works/pi/blob/main/packages/agent/docs/work-packages/06-session-branch-lane-separation.md)：采用数据与执行分开、短事务串行的思路；不移植其整套 Branch、Lane、Operation 模型。
- 本地 DSH `packages/core/session/src`、`packages/core/agent-loop/src/agent.ts`、`packages/core/agent-loop/src/tool-calls.ts`、`packages/session/session-turn-outline/src/projection.ts`：采用普通能力组合与日志投影；其 loop 仍写 `turn/start/end`，本版不继承该分组前提。

设计的最小结构是：Session 保存事实，来源插件启动工作，Context 准备请求，Model 推理一次，Tool 执行一次，Content 组装消息，ReAct 选择下一步，Delivery 发送，Turn/Akasha 读取。每项职责能独立解释、独立验证；不为省一个必要事实改变产品能力，也不为拆分而增加只转发的层。

## 14. 开工核查与删除记录

### 14.1 2026-09-05 正式运行事实

本次按 hua-home-server 技能只读检查；未切换服务、触发模型、调用业务工具或改写正式数据。以下是 2026-09-05 06:46～06:53 Asia/Shanghai 的观测，不是长期固定值。

- 正式 release：`403e6924ff9e57066b8fb78cfe66ae74f1d7fe25`；activation、runtime.env、Core 容器源码挂载一致。release doctor 返回 healthy；Core/Workload 容器 healthy、restart count 0、无 OOM，主机 failed units 为 0。
- 主机 health-check 返回 0，但完整 Borg archive/check/extract/hash 恢复证据本次未取得；不能把服务健康当作已有迁移恢复点。
- 正式插件根：`/srv/data/services/akashic/state/plugin-home`；workspace：`/srv/data/services/akashic/state/workspace`。开发机镜像不作为正式事实。
- 活跃能力来自只读 `/api/chat/runtime/capabilities`，snapshot `38288192a5b14574`：17 个内置插件、16 个外部插件、22 个 Skill、5 个 MCP；33 个插件 composition ready，无 missing service 或当前 unhealthy health 项。Steam 累积 10 个网络刷新 incident，不能把 ready 解读为从未出错。
- `sessions.db` 一致读事务：296 个 Session、14,608 条 Message、2,238 条旧执行记录（completed 1,647、failed 471、interrupted 113、cancelled 7）、8 个 Artifact、4 个 compaction、0 个 compaction prepare。失败/中断/取消记录仍包含 input 或工具 item，需核对唯一事实再迁移，不能整表当缓存删除。
- 正式 Session DB 的 `user_version=0`；真实迁移另由 `migrations.sqlite3` 的 yoyo ledger 记录，本次读到 28 条。迁移不能仅按 user_version 判断谱系。
- 最新远端 main 已是 `6a15444009c807994d33691e0b756167880fad5d`（PR #538 移除固定测试数量 Gate）。旧设计 worktree `add-akashic-v4-design` 的 HEAD 为 `51f1467456881e7302abf76a931e9dfe698fef6c`；当前实现 worktree 已从 `6a1544…` 建立独立基线并携带本设计，不能沿用旧 Gate 数量合同。

### 14.2 17 个内置插件的拆分落点

| 已加载插件 | 真实耦合或现有能力 | 施工处理 |
|---|---|---|
| akasha | `AFTER_TURN_COMMITTED`、Prompt/AfterReasoning 事件、`core.interaction_undo` | recall 保留独立能力；学习改日志投影；反馈与撤销按明确 Message 来源合同迁移 |
| compaction、markdown_memory | compaction storage、Prompt 与投影提交事件；Markdown 调用 models.chat | 保留独立持久 owner，迁移输入/输出合同，不把摘要和已写记忆当缓存删掉 |
| scheduler、subagent、wake | 都依赖 `core.scoped_turns`；另有 continuations、deliveries、semantic_interest | 改为普通程序、Task、Message 与 Delivery 组合；消除旧 scoped react 入口 |
| models、codex、openai-compatible、opencode-go | model service 与 driver 注册 | 复用单次请求接口，脱离 Turn 绑定时机；保持凭据与实际 generation owner |
| eventmail、drift | 领域消息池、来源服务与各自 SQLite 状态 | 保留真实来源业务与状态，把 Agent 执行交给普通能力组合 |
| computer | `core.mcp_servers`、`core.workloads` | 复用公开资源能力，保持真实进程、profile 与取消/重启合同 |
| shell-ui、conversation-ui、runtime-ui、workbench-ui | Web module 的静态贡献 | 跟随消息/控制协议迁移；不能因 Python apply 为空就删除插件 |

已额外核对直接消费旧链路的正式 stable artifacts：Citation/Meme 仍监听 Prompt 与 AfterReasoning 事件，Meme 依赖 `citation.protocol`；Observe/Proactive Feedback 消费 `AFTER_TURN_COMMITTED`；Shell Safety/Restore 消费 Tool authorize/prepare；Plugin Undo 消费 `core.interaction_undo`。这些是线上 V3 的真实入口，不能用本地旧 V2 checkout 判定无人消费。本次先完成仓库内替代合同及普通插件验收样例；这些外部插件随后在各自源码仓库迁移并经正式安装链验收，不能编辑 cache。新 release 正式切换以所需外部插件完成迁移为前提，不保留旧 hook 兼容壳，也不宣称旧 artifact 可直接运行。

### 14.3 开始拆解，补齐四项具体合同

方向研究已经足够。当前可以做功能分解、建立最新 main 基线、准备 fixture 与证据充分的局部改造；下列合同是接入新权威写入路径前的退出条件，不是继续泛化研究的理由。

| 前置工作 | 明确产物与出口 |
|---|---|
| 真实消费者与源码版本 | 17 个内置及受影响外部插件的 source→artifact→generation→合同清单；未归位动态入口不得删除 |
| 受限事务与插件替换 | Message+owner state+binding reference 的原子提交、授权及 crash 合同；证明外部实现无私有权限也能接管 |
| Message 与旧数据迁移 | 旧输入/工具轨迹去重映射、Akasha/compaction来源、撤销协议；原生一致备份、完整性核对与隔离恢复演练 |
| 客户端与一次性切换 | Web/Mobile/SDK 的 message_id/seq/活动 handle 合同；新链路单独验收，正式切换与成套恢复步骤明确 |

### 14.4 删除规则与账本

用户已允许删除核实为不必要或冗余的功能。每项删除在实施当批写入[既有重构账本](../refactor/clean-code-ledger.md)，记录：对象与位置、独立事实判断、静态/动态/正式消费者证据、为何不需要、是否有承接 owner、可观察行为变化、状态处理、验证、commit 与恢复点。没有独立事实也没有消费者的包装直接删除；确有产品行为变化则明确记录，不伪称语义不变。

已实施的删除与恢复点记录在清理账本；旧业务链路尚未退休。旧 Attempt/Turn 身份与重复 transcript、固定 Phase pipeline、工具名特判和只转发包装属于待核实候选；历史 plugin-data、消息、学习状态、附件和回执不因对应代码退休而自动删除。已批准的删除许可不要求逐个重复确认普通冗余代码；涉及未定义的数据减少或无法判定是否仍需的产品行为时，先给出具体影响再处理。

## 15. 按线上插件功能修订扩展合同

本节为 2026-09-05 第二轮只读核查。再次查询 hua-home capabilities，snapshot 仍为 `38288192a5b14574`，33 个插件 ready。以下依据该主机 stable artifact 的生产代码，不以开发机旧 checkout 或安装测试代表当前功能；本轮没有发送消息或触发工具，功能端到端结果仍须后续验收。

### 15.1 内容协议属于内容，不属于回复收尾

线上 Citation `plugin.py:58–102` 先改正文、写引用 metadata，再清除所有尾随协议标签；Meme `plugin.py:18–75` 注册另一份正文修改器，并依赖 `citation.protocol` 保证顺序。后者提供的对象只有版本号，依赖没有独立业务事实。Meme `runtime.py:80–92,144–151` 从已启用类别随机选图片。以上顺序依赖、共享可变 reply 和宽泛标签清除不迁入新接口。

Content 普通插件拥有一次内容组装；协议插件拥有自己的语法、提示和解析。每个协议贡献一个内聚定义：协议标识、提示片段、对不可变原始输出的解析函数。解析返回精确文本区间和已声明的内容块，不得改其他解析器的输入，也不能吞掉不属于自己的标签。Content 检查区间重叠并一次性生成结果；冲突明确失败，不靠安装顺序裁决。字面文本和代码区间不参与模型协议解析。

```text
┌ Context ─────────────────────────────────┐
│ 人格、记忆、协议定义的提示 → 模型请求       │
└──────────────────┬───────────────────────┘
                   ▼
            原始模型输出（不可变）
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  Citation 自有解析      Meme 自有解析
  引用及证据类型        类别 → 固定图片
         └─────────┬─────────┘
                   ▼
       Content 一次组装 text / citation / artifact
                   ▼
       导入附件完成 → 追加 Message → Delivery
```

注册、读取协议定义仍使用已有普通 Service/Effect。一次请求固定协议集合及其 generation，Context 提示和 Content 解析使用同一集合，热更不在二者之间替换语法。固定业务顺序只存在于 Content 内完成一次组装所必需的阶段，不重新命名一组 Before/After 全局 hook。

一次请求从构建 Context 到内容组装、附件导入并 append 完成持有同一协议集合的 exact generation lease。进程崩溃或 decode/import/append 失败后尚未提交的模型输出属于未接纳的临时结果，允许丢弃，不承诺从诊断原文恢复；重启后从已提交 Message 建立新请求和新协议集合，不把旧文本交给新解码器。Model owner 在 I/O 前耐久记录调用与配置要求的费用占额，在 I/O 后记录实际 usage 或计费 unknown；新请求独立计费并受剩余预算约束，不以丢弃输出清零额度。工具只有在 call Output 已提交后才执行，因此这一窗口不会隐藏已执行工具。原始模型输出若留作诊断仍由 Model owner 管理；已经提交的 provider replay facts 随 Message 保留，不能依靠可清理诊断恢复会话。

两种直接调用足够：模型文本在 `async with content.bind()` 中调用 `view.decode(output, references)`；已经结构化的程序直接构造 text/artifact/citation parts，并走同一附件导入与 Message 校验。普通回复、Wake、定时发送和 `message_push` 都能使用 Content；ReAct 没有专属入口。用户输入、工具结果和引用的代码不会因为经过存储或 Delivery 被再次解析。

Citation 保留隐藏内部标记、清理其自有 inline 引用、显式引用列表与召回兜底。新内容区分证据：`declared` 表示模型明确声明，`retrieved` 表示实际召回候选。无显式引用时，候选可作为旧产品行为的兜底，但不能被持久记录成“模型确认用过”。召回 owner 返回结构化 reference，Citation 不扫描名字恰好叫 `recall_memory` 的任意 JSON。未知引用记录无法解析的状态；不得造出来源或静默映射成别的记忆。Akasha 决定不同证据怎样参与强化；既有已学习权重不因新标记自动重算。

引用块的最小内容为 `{ref, declared, retrieval_ref?, resolved_ref?}`：`ref` 保存模型声明或候选的原始引用；`declared` 明确它是否来自模型；`retrieval_ref` 指向本次实际召回记录；`resolved_ref` 指向已确认的领域对象及其不可变 revision。没有 `resolved_ref` 就是未解析，不另设一份可能矛盾的 resolved 状态。`declared=false` 必须有真实 `retrieval_ref`；声明与召回命中同一对象时一块同时保留两项事实，不复制两份 citation。Content 只在本请求固定的 reference 快照上关联；Akasha 不从字段缺省或出现顺序猜测证据等级。

Meme 保留启用类别、动态提示、每条回复最多一张图、文本清理、分类管理面板、Skill 和附件显示。解析只消费协议允许的位置，代码示例保持原文。类别合法但没有可用图片时返回明确的缺失结果，由调用程序选择文本回复并记录诊断；导入失败不谎称附件 ready。图片选定后必须先导入 Artifact，再提交消息；Delivery 重试读取同一 artifact，不能重新随机选图。Meme 不依赖 Citation，两者单独启用、卸载、交换注册顺序都成立。分类文件和图片仍由 Meme 领域 owner 管理，不迁成 Core 配置。

### 15.2 保留能力，而非旧事件形状

| 活跃插件与已核对位置 | 要保留的用户功能 | 新组合与持久 owner | 验收边界 |
|---|---|---|---|
| Citation `plugin.py:58–99` | 回复不泄露内部引用协议，记忆引用可追踪 | Context 协议贡献 + Content 引用块；召回 owner 提供来源 | 显式/兜底可区分，未知 ID 不伪造，Meme 缺席仍可用 |
| Meme `plugin.py:28–75`、`runtime.py:80–151` | 按类别表达表情、图片和分类面板 | Content 解码 + Artifact；目录与 UI 归 Meme | 普通/主动内容、代码示例、空分类、固定附件重发 |
| Shell Restore `plugin.py:80–101` | 简单 rm 改为可恢复移动 | Tool owner 的参数准备贡献；restore 目录归插件 | 原始与最终参数可查；复合语法不误称已保护 |
| Shell Safety `plugin.py:61–95` | 阻止会卡住的交互编辑、sudo、包管理命令 | Tool owner 在全部转换后授权最终参数 | 直调和 ReAct 一致；拒绝不调用工具、不记成功 |
| Proactive Feedback `plugin.py:133–180,231–270,457–535` | 判断主动消息是否被接续、引用命中与反馈历史 | 拉取 Message/Turn；评分、幂等记录和 cursor 归反馈插件 | 断开通知可重扫；保留原候选时间窗，不丢漏多输入 |
| Emotion `plugin.py:55–113,145–208` | 显式引用反馈、反馈影响语气与主动偏好 | 消费消息引用与反馈 history；Context 返回语气贡献，Timer/Drift 复用 | 去掉对拼接 Prompt 标记的依赖；累计状态不清空 |
| Observe `plugin.py:51–119,193–231` | 缓存命中、token、工具、记忆与运行健康可查 | 各 owner 发布其事实；按 Message 引用聚合，trace DB 归 Observe | 一次输出前的多次模型调用不丢失；诊断失败不阻塞送达 |
| Plugin Undo `plugin.py:17–49` | 撤销最近对话、展示备份和派生状态处理结果 | Turn 只给候选 ID；独立 Message 管理能力固定集合并执行 | 不把投影当删除权；失败显示真实已删除/待处理状态 |
| Status Commands `plugin.py:37–85` | 记忆整理状态、命令别名、移动面板 | Command registry + Message 只读 + compaction 自有状态读口 | 不因读取历史授予 compaction 写权或 SessionStore |
| Setup Helper `plugin.py:28–54` | 查询当前渠道/chat identity | Command invocation 的已验证渠道上下文 | 无模型可运行；命令结果按同一 Message 合同保存 |
| Calendar、Feed、Fitbit、GitHub Watch、Steam、Huayue Skills | 已声明工具、事件来源、Skill/MCP 和面板 | 保留领域 API、Timer、Tool catalog、UI/Workload；不复制 Agent 执行模型 | 本轮 fleet 身份已核对；每个实际入口的安装与业务验收仍须所属层完成 |

工具的准备和授权有先后不等于不正交：它们共同保证“实际调用的参数经过授权”，必须由一个 Tool owner 收口。内容解码、反馈学习和诊断不具有这个不变量，不再被塞进工具或回复完成事务。

Observe 不依靠最终 Turn 事件夹带整份 context/tool/model 状态。Model 记录每次请求的实际 usage，Tool 记录调用结果，Context 记录请求视图统计，Message 记录已提交输出；相关 owner 发布已有身份的诊断事实。诊断 request ID 可以存在，但不充当准入、恢复、学习或持久 Turn 身份。不能为了删除 Attempt 同时删除真实失败请求的可观测性。

### 15.3 无状态 Turn 的消费合同

`turn_projection.project(messages, source)` 对一个明确、完整的日志前缀返回分段引用。返回值只有来源、边界、状态、Input/Output 成员 message IDs 和实际工具观察的 `(call_ref, result_message_id)` 引用；正文从 Message 读取。工具是否应当结算后才写 finish 由生产者保证，投影不复查执行授权或结算规则。插件没有数据库、后台 worker、订阅 cursor、学习队列或新的 Turn 身份。分段版本是算法合同，不是用户消息字段；它必须固定算法 artifact digest 和输入 schema，不能由可复用的显示版本号或 latest 指针代替。

Akasha 声明依赖该服务；在线学习与离线重建调用同一函数。它自己保存 `(learning_binding, ending_message_id)` 的已应用事实、消费进度、向量和反馈，并以普通 learning binding 保留学习服务及其显式依赖的 projection artifact；离线重建按此身份解析。旧 artifact 缺失时明确停止对应重建，不能自动换成新版；更换规则要先比较样本并显式迁移学习状态。处理成功及其进度在 Akasha 自有事务内一致提交。其他需要逻辑 Turn 的消费者可复用同一服务，各自拥有进度；不要求只需要 seq 的读者使用 Turn。

分页必须包含待处理的 open 前缀或由消费者保存足够的消息引用后重新读取；不能把每一页开头误当新 Turn。投影本身不保存这些引用。若删除管理使已读成员失效，由明确的管理结果通知消费者，不能让投影通过读到空内容猜测删除已成功。

### 15.4 外部插件与正式切换边界

当前栈完成仓库内实现，并用只依赖公开 API 的测试插件验证上述扩展形态。测试插件可以证明 API 支持这些功能，不能替代真实外部插件发布验收。此阶段不改外部源码、安装 cache 或正式 workspace。旧 artifact 使用已删除 hook 时应在候选验证明确报缺失合同，禁止半加载后悄悄丢功能。

正式切换前必须补齐实际启用且受影响的外部插件源码迁移、各自 yoyo（若改变其状态）、候选 fleet 与功能验收。这个发布前提不要求仓库内新实现保留旧 pipeline。外部插件的持久配置、图片、反馈、恢复文件和 trace 不因新接口而删除。

## 16. Stacked PR 执行与迁移归属

任务类型为 `refactor + migration`，`semantic_delta=breaking`，owner 为 `mixed`。共享 Message 的持久接纳、资源与绑定必须由 Core 提供，客户端实现会复制权威记录与恢复逻辑，不能替代。消费者范围是仓库内插件、Core/Bootstrap、Web/SDK 及外部插件的公开合同；外部源码和正式部署不在本栈写入范围。唯一 writer 为本任务主 Agent，worktree `message-plugins-stack`；从 main `6a15444009c807994d33691e0b756167880fad5d` 开始，每层以前一层为 base。

| 层 | 可独立评审的改动与出口 | 同层 yoyo / 数据证据 |
|---|---|---|
| 01 合同 | 本设计、真实插件功能落点、验收和删除边界 | 无持久变化，不创建空迁移 |
| 02 Message 类型与 Turn 能力 | 不可变 Message 合同、无状态投影普通插件；用独立调用证明分组无需执行对象 | 不改 schema，不接管旧 writer；Turn 不建表，不创建空迁移 |
| 03 Message 日志 | 窄读取/追加/CAS、schema 与历史转换在隔离 fixture 验收 | yoyo 与新 schema 同层提交；转换已有 messages，核对 ID/seq/正文、重复执行和 crash 恢复；旧 turns 原样保留，由第 08 层接管 |
| 04 内容与 Context | 提供普通 Content/Context 能力与临时内容索引；Citation/Meme 形态样例；第 08 层接入后取代共享可变编排 | 仅改变新消息内容无需重写旧消息；有配置归属迁移时随本 PR |
| 05 任务、事务与调用记录 | 通用 Task、受限 owner 事务；现有 Model 单次入口在 I/O 前后记录调用，冻结请求 | `owner_records` 与 `model_calls` 各附 yoyo；原子回滚、取消、费用 unknown 与重复迁移 |
| 06 耐久绑定 | 归档完整代码与配置闭包；按引用打开 exact 短命 scope，不维护业务执行状态 | 所需绑定 schema/归档脚本同 PR；卸载当前 cache、重启与归档损坏验收 |
| 07 单次工具与模型投影 | Tool 授权、receipt 与恢复；Model 的 Message 投影；固定 Python 环境、按目标打开 MCP 与借用 Workload | 复用 05 的 owner_records，无新 SQL；独立调用、crash point、环境丢失与资源排空 |
| 08 回复与消费接入 | ReAct/conversation、Akasha/compaction/记忆读者及实时协议使用新日志；接管命令、暂停/恢复 | 本层引入的旧 turns 尾部、Akasha provenance/消费 ledger 与控制状态迁移同 PR；学习状态保全 |
| 09 Delivery 与来源 | 独立发送/重试；Scheduler/Subagent/Wake 组合公开能力；发布操作脱离父 Turn | Delivery/来源状态脚本随本 PR；已发送不重发、ACK 丢失与重启验收 |
| 10 删除与累计验收 | 删除旧 Worker/Pipeline/Attempt 执行权与重复入口；同步长期规格、能力手册和删除账本 | 不 DROP 历史表或自动删除数据；剩余实际 schema 变化仍附本层脚本，不能补漏前层迁移 |

用户补充的发布背景：整个 stacked PR 栈全部 review 后一起合并、统一上生产。分层按职责与相邻 diff 的可审阅性拆分，不要求每个中间 PR 单独可部署，也不为此建立兼容壳或把所有实现塞进一张大 PR。对应 yoyo 放在引入该持久变化的层；正式发布时按依赖执行整套迁移。新能力在隔离环境分别验收，最终栈顶必须证明 writer、所有消费者、Delivery、来源和所需外部插件同时可用；正式环境一次性接管。每层描述相邻验证、尚未接入部分和累计未完成项，不以中间代码尚未完成冒充功能已删除。

每个涉及状态的 PR 都列已知 schema lineage、影响表/字段、增加/原位更新/逻辑失效/物理减少条件及 owner。`sessions.db/messages` 既有正文不得因迁移、摘要或投影减少；旧 turns 中尚未落消息的输入和工具事实先核对映射，不能按 status 直接丢弃。迁移前使用原生一致备份；核对行数、稳定身份、正文摘要、引用、附件、外键及 `integrity_check`，未知形状 fail-loud。

Yoyo 脚本只追加到 `migrations/yoyo/`，显式声明 `__depends__`，不修改已合入历史脚本。仅初始化新空库的 schema 不足以通过升级验收。脚本在 schema owner 所在层评审并覆盖正常升级、重复执行、执行中断和恢复；涉及多文件发布时有耐久备份及 receipt。回滚若无法无损表达新写入，应明确要求停写、保存新事实并成套恢复，不能写一个 DROP 新表的伪逆迁移。

每层执行相关行为测试与 change-impact Gate，独立 Terra xhigh 概念审查相邻 diff；栈顶再做累计协议、持久化和公开场景 Gate。只创建 draft PR，等待用户逐个 review；不合并、不发布。当前恢复点为独立 worktree 基线和 `_backups/message-plugins-contract-20260904-232201`，原设计 worktree 的未提交修改保持不动。

## 17. 旧日志迁移的具体保全合同

第 03 层引入 Message schema 与同层 yoyo，仅转换已有 messages；旧 turns 和其引用原样保留。第 08 层在回复、控制和消费者接管时，附自己的 yoyo 处理下述旧 turns 尾部及消费边界。整栈一起 review、合并和发布，不要求中间 release 独立运行。以下转换只在隔离副本验收，本次不操作正式 workspace。

旧 messages 的 `id/session_key/seq/ts/content/tool_chain/extra/role` 都是迁移输入。原 message ID、seq、接纳时间和正文文本保持；原始 `tool_chain`、`extra` JSON 文本作为已声明的历史内容保全，不通过 parse/re-encode 冒充字节相同。`role` 只是旧协议角色，不足以证明真实作者；作者标为 `legacy-attribution-unknown`，原 role 留在 provenance。没有来源证据的旧消息进入稳定 `legacy-unattributed` source；不从 proactive=true 猜 Wake、Scheduler 或具体 producer。该来源没有默认响应插件，不自动接纳新工作。

原 assistant 的工具轨迹保留在其原消息的 `history.transcript` part：raw JSON、SHA-256、旧 schema 标识及 `complete/truncated/unknown` 完整性。当前数据不能证明完整时使用 unknown，不恢复已被截断的字节，也不把旧轨迹伪造成独立 ToolCall/ToolResult。这样同时保持旧 seq 与真实记录顺序；新的运行路径只追加原生 call/result Message，不再生成历史轨迹块。

`history.provenance` 保存原 role、原始 extra 及 digest。已登记的附件仍使用原 artifact ID 和原 message binding；媒体、delivery、client ID、旧 interaction 等真实关联必须在新历史投影中可读取，不能只在备份中存在。Context 仍能读取旧聊天正文、附件及不可执行的历史工具轨迹；“不调度旧 source”不等于把用户过去的对话从上下文抹掉。

旧可续接输入不能一律归档。第 08 层按最新未完成记录及明确的 `continuedFromTurnId/interactionId` 恢复旧链，再独立核对旧 channel/chat/inbound 身份；执行链本身不证明来源。缺少来源证据的记录只归档并列入迁移报告，不升级成可执行 conversation。已经有确切最终 Message 引用的尾部先核对提交事实，不能仅按 failed 状态判定尚待回复。

对已证明属于 conversation 的 open 链，按整条链的 `USER_MESSAGE.ordinal` 枚举全部用户输入。每项通过明确的 item/message/client 引用映射已有 Message，或按旧 record/item 身份迁入 Input；不按正文或时间去重。旧时代缺少 ordinal 的记录保留原证据，不猜恢复顺序。每个已证明的 open conversation 链，无论是否新增 Input，都在同一 SQLite 事务提交 migration-author 的 `pause(through_seq)`、需要迁入的 Input 及全部旧 item→existing/new Message 映射 receipt。through_seq 固定为该链迁后 conversation 前缀的最高已接纳 seq，覆盖全已落消息与部分未落两种情况；commit 后才通知消费者。启动不为 `seq <= through_seq` 的输入创建任务；以后真实接纳的 Input 超过该边界便可唤醒，并在 Context 中带上这些旧输入。pause 是前缀截止，不是永久禁用来源。

旧已结算工具仅供不可执行的历史 Context；可续接尾部存在 `in_progress/interrupted/cancelled` 工具项或缺少 terminal effect receipt 时，迁移评估及正式切换必须停止并报告实际记录，不能静默略过或盲重跑。无精确信息时不假称旧外部效果已回滚。上述输入映射与下述归档共享旧 record/item 引用，不把同一输入当作两条新聊天事实。

旧 turns 中不能通过确切 record/item/message 引用证明已转入 Message 的事实，追加为迁移者产生的 `Output(quiet, history.record)`，不新增 Core body。该 part 包含原行所有 nullable 字段的原始标量/JSON 文本、表与 record 引用、来源快照及 digest；不是可执行 Input，也不包含可被 Tool 发现的原生 call。quiet 仅表示迁移者不生成新的用户回复，不重述原执行成功或失败。原时间和状态保存在记录内，Message 的 recorded_at 是实际迁移追加时间。导入顺序固定 `(created_at, id)`，身份由旧 table/record 确定，映射与提交结果保证中断后重跑不会再分配一组 seq。不能按相同正文猜测两条输入是同一事实。

`history.record` 使用独立、无默认 producer 的 `history` source，author 为迁移者。默认 conversation Context 和 Akasha 不把这类迁移记录当新聊天或学习输入；历史检查通过明确标注的只读视图展开。它与前段仍可用于上下文的旧聊天 Message 不同。历史内容 schema 必须在受限 writer append 前校验；普通输出 writer 没有 `history.*` 内容授权，迁移 grant 仅允许指定 source、body 和历史 schema。

Akasha 的旧样本、权重、向量、原 message IDs 与旧 provenance 原样保留。新投影消费 cursor 从切换边界开始，不能因迁入 quiet 记录或启用新分组而重学旧样本。旧样本的重建仍要求固定旧规则与来源；不得自动用新 Turn 插件替换。旧 turns 表作为只读恢复证据保留，不参与新工作准入、调度或工具恢复，也不自动 DROP。

迁移先验证已知表结构和引用，再做原生 SQLite 一致备份，固定来源 logical digest 与行数；未知同版本异构 schema、缺失关联或 digest 不符均停止且不修改权威数据。转换在单库事务中完成，保留已有 FK/附件绑定与 Session 的不复用序号；提交后复核全部旧消息正文、身份、seq、raw provenance、附件绑定以及 integrity/foreign_key_check。Yoyo ledger 在实际转换完成后落账；若转换已提交但 ledger 尚未写入，重跑先核对提交 receipt，不再次导入。数据库之外的文件发布另有耐久 receipt，不假称能靠 SQLite rollback 撤回。

第 03 层删除旧 messages.content FTS 派生表及其三个触发器，原消息 rowid 保持。第 04 层由历史读取/检索能力建立新内容索引；整栈顶验收必须证明搜索可用，中间层不宣称已经接管旧检索。迁移前识别其实际 schema，不因对象名字相同就视作可删除缓存。


## 18. 历史素材与端到端验收

用户要求本栈完成更多真实数据与组合行为验收。历史记录是测试素材，旧实现不是唯一正确答案；不为通过旧新逐字比较恢复 Attempt 或旧 hook。改变可观察行为时，报告旧结果、新结果、设计理由、涉及的原始 Message IDs 和学习状态影响。

```text
hua-home 原生 SQLite 一致快照（记录实际 release 与哈希）
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
旧实现隔离重放         副本运行 yoyo → 新实现隔离重放
       │                    │
       └─────────┬──────────┘
                 ▼
Turn 成员、消费次数、引用、特征、权重与恢复差异报告
```

- 每个数据库使用 native backup，传输后校验哈希、SQLite 完整性与外键；跨数据库独立快照不冒充全局同一时刻。原始副本只读保存，迁移、旧/新重放分别使用独立输出目录。真实正文与数据库不提交 Git 或 PR。
- 固定相同历史、embedding、配置、时钟和算法版本。先比输入成员与来源，再比 Akasha 逻辑状态；文件字节相同不是唯一标准。旧实现也在隔离目录运行，不把重放写入正式 Akasha，重放不执行历史工具或发送消息。
- 旧实现跨主机对照已验证 OpenBLAS 线程数会改变浮点归约及后续特征选择。固定同一旧代码、embedding 和 16 线程后，两端 5571 个 Turn、5403 个 hub、24735 条关系及完整逻辑状态哈希一致（`9f0e647b9a249e5176d3a7ed2f27cdf04f7c7b073177003e56d2fe2dc110c197`）。新旧比较沿用固定线程数；18 线程旧重放的差异不计为新投影语义变化。此项仅证明旧基线可复现，新消费者仍须独立验收。
- `u1 → interrupt → u2 → interrupt → u3 → a` 必须由同一来源投影成包含三个输入与一个最终回答的完整 Turn；暂停/失败不产生提前学习。Akasha 实际消费只提交一次，重复通知、分页、重启和重复重放不能重复强化。
- 在上述序列每个切点插入其他来源的 proactive、定时输入/输出及其工具结果；这些来源可以成为共同 Context，但不能切断 conversation 的学习单元或混入它的成员。单独验证各来源的完整 Turn。
- 补充 abandon 与晚到工具结果、同一 Output 多工具乱序返回、新输入使旧模型输出 CAS 失败、终态前崩溃、学习成功但消费 ACK 丢失、插件 generation 切换、未知费用/工具/发送结果和精确暂停控制。
- 迁移覆盖生产副本、干净空库、已知历史 schema、重复执行与中途进程退出。检查正文、ID、seq、附件、embedding、旧学习状态与迁移 receipt；未知 schema 或未决副作用明确阻止切换，不忽略或重新执行。
- 端到端覆盖接纳 ACK、预览与持久消息同步、命令、ReAct、内容、附件、工具授权/恢复、Delivery、Wake ACK、Scheduler/Subagent、Akasha/compaction 与插件卸载/热更。组合测试使用真实存储和普通插件；受控 Model/Tool/渠道用于确定性故障注入，实际 provider 验收单独标明。
- 每张 PR 提交前由 `gpt-5.6-terra`、`xhigh` 独立审查相邻 diff；最终再审累计行为。测试通过、独立审查、历史重放和生产验证分别报告，不互相替代。


### 18.1 第 04 层可观察边界

Content 的 `ContentSchema` 只声明内容校验；需要文本语法时用 `TextProtocol` 一起声明提示与 decoder。所有实际 view 都来自 `Content.bind()`，共享请求 generation lease；结构化程序无需伪造文本协议。

Context 接收 `Materials` 中已取得的系统提示、低信任检索内容和已发布摘要。摘要只声明实际覆盖的连续消息区间；Context 把完整快照、覆盖末尾 `after_seq` 和摘要引用交给 Model 的只读投影，不能先裁掉 provider facts。明确提供摘要引用时，从摘要开始新请求；只接续同一摘要之后成功 Output 的 opaque state。只给 `after_seq` 不授权清空 replay，无法安全接续时明确拒绝。系统提示只在请求 messages 中出现一次，避免不同 provider 重复加入 instructions。

历史搜索使用 Context 内存 FTS 索引，正常只幂等追加已提交消息；重启按消息日志重建，不保存另一份磁盘正文或持久 cursor。控制记录不作为正文搜索，但仍参与消息身份冲突检查。源消息的显式管理删除由接入 owner 重建索引，不把缓存删除升级成源消息删除权。第 04 层不引入新的持久 schema，因此没有空 yoyo；第 03 层旧 FTS 的替代索引在此提供，实际消费者在第 08 层一次切换。

### 18.2 第 05 层的执行事实与状态边界

`Tasks.admit(key, callback)` 的同步回调不跨 await；同一事件循环中启动、控制和 effect start 按实际调用顺序接纳，不另存锁表。每个 Task 只拥有短命 handle、同步资源撤销和实际排空。取消先撤销 writer，再通知运行任务；重复取消不打断已经开始的结算。失败的准入撤销本次新任务，完成的任务按 exact identity 移出 registry；调用者持有的 handle 仍可 join。Task 不持久化、不提供逻辑 Turn 或业务终态。

`OwnerStore` 只授予一个状态空间。同步事务内可以用已获授权的 Message writer 追加，并以版本 CAS 更新本 owner 的 JSON 状态；不能执行任意 SQL、跨库写入或在 await 期间持锁。部分写入失败即使被调用方捕获，整个事务仍回滚。`owner_records` 正常增加记录，允许 owner 按其领域规则原位推进版本；没有自动减少路径，也不复制 Message 正文。公开服务和实际消费者在第 08 层一起接入。

Model 的网络调用仍只有 `_BoundChat.complete` 入口。`ModelRequest` 在构造边界冻结 messages/tools；Context 不重复拥有冻结规则。Model 在调用 driver 前向自有 `model-registry.sqlite3/model_calls` 追加 `started`，返回后原位记录实际 usage；异常或取消记为 `unknown` 并继续抛出原错误。成功响应带 `call_record_id`，没有 usage 就保留未知值，不能补零。进程崩溃留下的 started 只证明请求曾被接纳，不能证明没有费用或触发自动重放。配置 revision 不因模型调用变化，不通过配置修改的整库备份路径记账，也不持久保存请求正文、原始输出或 credential。调用记录没有自动删除路径。Model 的事实引用可从此记录读取 binding 与 usage，避免在 Message 再存一份计费账。

本层两个 yoyo 分别增加 `owner_records` 和 `model_calls`。已有库须先迁移，初始化只为新空库创建新表。迁移使用独立 SQLite 备份，重复执行保留既有记录；正常模型调用不因此重写旧会话、模型配置或学习状态。生产数据副本验收中，两项迁移均成功且重复运行返回 current；14613 条 Message、2244 条旧 Turn 及所有模型配置表的行数与内容摘要未变，两个数据库 integrity/FK 检查通过。原始数据、恢复副本及完整摘要只留在受限的本地 fixture，不进入 Git。

### 18.3 耐久绑定只保存引用，不重述业务执行状态

执行层 Gate 还复现了旧 Shell 在 deadline 附近丢失最后输出的竞态。DSH 的 subprocess 同样把进程 exit 与输出 close 分开；本栈沿此边界修复：收集截止时发现进程已退出，先完成有界输出排空，再生成终态响应和清理记录；继承 pipe 的残留子进程不能无限拖延结束。这不改变 Message 或逻辑 Turn 的终态。

第 06 层按内容 hash 保存独立于 installed cache 的不可变归档。binding 引用完整代码、运行要求、manifest 与可复建配置闭包；凭据只保存受保护引用，plugin-data 仍由原 owner 管理。需要历史业务状态的能力必须自行保存其不可变输入，Core 不快照整个运行 workspace。打开 binding 时只从该归档构建短命 exact scope；缺失、损坏或不兼容必须明确失败，不切到 stable/latest。

本轮不增加 active claim、持久 refcount 或 terminal 表：Tool、Delivery、Akasha 已各自拥有调用、发送或消费事实，是否需要打开 binding 从这些事实计算。内存 lease 关闭后释放运行资源；归档文件不阻挡当前插件 drain 或卸载。归档是耐久恢复材料，没有自动 GC；当前 cache 的清理不拥有它。归档写成但 Message/receipt 提交失败时允许留下未引用文件和不可变 binding descriptor row，不自动减少恢复材料。未来若要回收归档，须单独制定显式减少协议，不能倒推当前需要另一套业务状态机。


第 06 层的实现边界：

- 代码在导入前按完整文件树归档，正常 generation、候选 clone、延迟 import、静态命令和资源读取均使用归档路径。`plugin_dir` 保留安装来源和发布指针含义；`code_dir` 从实际模块入口计算，不再保存第二份路径字段。Skills 的展示软链也指向该 generation 的归档资源；原安装目录变化不能改变旧 lease 的正文。
- 代码归档保留 manifest 和 requirements；`.venv`、`node_modules` 属于运行环境，不作为代码归档。第 06 层只分开代码/cwd 与安装环境，不打开历史外部 runtime。第 07 层已由安装 owner 在最终路径创建并固定 Python 环境；历史调用按所选目标校验环境引用，缺失或不匹配明确失败。当前 installed cache 不参与历史环境恢复，具体边界见下一节。
- binding 从所需 Service 的实际 provider 出发，只向上收集插件与子 Fiber 的声明依赖。Content、Tool 等注册表由自己选择目标的注册 Context，再将真实 Context 交给 binding；Core 校验其属于当前所选 Root 的存活 Fiber，随后将其 owner 纳入同一闭包。调用者不拼 plugin ID，也不恢复整个 fleet。目标选择和 definition 身份属于 registry 的不可变 metadata；第 07 层完成这些具体注册表消费者接入。
- 配置正文与 revision 来自同一次读取。归档保存可复建投影和已捕获的静态启用选择，不重算当前环境下的 `is_active`。日期和 CredentialRef 使用明确的值编码；凭据解析仍通过其 owner 的 revision fence，不在归档中存 secret 原文。
- 打开历史 binding 只装配其闭包和 Root 本地注册表，不执行 `runtime.started`。第 08 层允许归档目标显式声明所需的通用 Message 读写、owner state、Bindings、Tasks、Timer 和 Artifact 读口；未声明的能力不可用。旧 Session、Turn、Delivery 和 Undo 业务端口仍不进入归档。纯注册不打开目标，实际资源由目标的 open 和 scope 释放；不能用空数据或 candidate 兼容壳冒充恢复成功。
- scope 退出先停止接纳新 lease，等待保留的 lease 排空，再在取消保护中释放 Root 和模块。它管理真实运行生命周期，不是任意 Python 对象的可撤销沙箱；服务必须在 `async with` 内使用。具体 Tool/Model/Delivery/MCP facade 在执行入口检查 exact scope 和目标，这是各资源 owner 的职责，不增加透明代理。
- 本层复用第 03 层的 `bindings`/`message_bindings` 表，不引入 SQL schema 或配置迁移，因此没有空 yoyo。binding descriptor 与 archive 都是不可变恢复材料；Message 对它的引用仍与正文同事务提交，不持久保存另一份 active claim。


### 18.4 第 07 层：工具调用、模型投影与所需资源

普通 Tools 插件拥有单次调用。准备阶段只固定最终参数；真正执行前才授权这些参数并记录 started。重复请求复用同一个 Task 与 receipt；已知结果直接读取，started 在恢复时先查询原目标。不能查询且不保证幂等的副作用停在 unknown。对话 ToolResult 与结果指针在同一事务提交，独立调用由 receipt 保存结果正文；两者不复制同一份正文。

Model 的只读投影把 Output 与对应 ToolResult 排成 provider 要求的完整调用组，Message 的 seq 和正文保持原样。成功调用记录提供 binding 与 usage，Output 只引用该记录及必要的 tool ID、thinking 和 continuation；跨来源上下文可见，但 provider continuation 只属于选定来源。图像从 Artifact 的只读内容产生；Chat Completions 的 assistant/tool 图像放在完整工具组后的 user 行。纯图像原行留下明确提示，不发送不合法的空 content。

Content 的协议类型作为稳定 API，实际解码器与注册 Context 仍属于归档 generation。Context/Content/Tools 从同一个所选 Root 捕获依赖，Overlay 只接受自身拥有的真实 Context。

```text
┌─────────────────────────────┐
│ 固定 binding / 注册 owner    │
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│ 打开所选 Tool / MCP          │
│ 校验所需代码与 Python 环境    │
└──────┬──────────────┬───────┘
       ▼              ▼
┌──────────────┐ ┌────────────────────┐
│ 私有 process │ │ 借用匹配的 Workload │
│ / MCP 连接   │ │ 正式 owner 保持控制 │
└──────┬───────┘ └──────────┬─────────┘
       └─────────┬─────────┘
                 ▼
          调用 → 断开并排空
                 ▼
             归还借用
```

Python 环境在最终目录创建，之后不移动虚拟环境；console script 的绝对解释器路径因此保持有效。component descriptor 第 2 版按每个 Python runtime 保存环境引用。打开纯 Content 不检查该插件未使用的 MCP 环境；调用具体 MCP/process 时才校验它自己的代码、requirements、环境树和宿主基础 Python 身份。该协议只覆盖同一 POSIX 主机和基础解释器，不宣称冻结整个操作系统或动态库。Node 等未实现的运行环境不能套用 Python 的恢复承诺。

MCP 的公开 open 要求调用 Context 是该声明的实际 owner；其他插件须使用 owner 明确提供的能力。历史 MCP 使用独立连接和私有 process 端口，不执行正式启动事件。需要 Desktop 等 Workload 时，只借用当前 ready 且完整 descriptor 相等的已有资源；不替换或停止它。正式 owner 在停止前禁止新借用，等待已有借用排空。历史 MCP 断开、process 停止成功后才归还借用，清理失败保留实际 owner 和重试证据。调用 scope 的失败从资源 host 的 tombstone 查询，通过 Manager 的 `resource_failures / retry_resource_cleanup` 按精确 scope 重试；不写正式插件 reload journal，不重建 stable Root。Manager 关闭时同步停止接纳新调用，再清理已接纳的调用资源，最后停止正式 Workload。同一个调用 owner 的锁覆盖启动、停止与重试，关闭不取消整个调用者 Task；并发重试只归还一次借用。监督运行中进程继承 boot 身份，Gateway 退出后的进程组清理由既有 guardian 拥有。

本层没有新的 SQL 或配置 schema，Tool receipt 复用第 05 层 owner_records，因此不增加空 yoyo。环境新增、保留及恢复材料见[状态地图](persistence-state-map.md)的第 07 层。没有上线过的第 06 层 component v1 明确拒绝，不增加临时格式兼容壳；整栈一起上线。旧安装 cache 缺少环境引用时，显式重装发布新的代码 artifact 与引用，旧 cache 不被原位改写；普通读取不会偷偷安装依赖。

本层验证了实际归档、Python 环境、MCP 子进程、工具 receipt 和 provider 请求投影。ReAct、conversation、Akasha 及现有 Computer 工具注册仍待第 08 层接入，Delivery 与各来源待第 09 层切换；本节不代表完整被动链路已经运行在新组合上。

### 17. 第 08 层的消费与恢复约束（实施中）

Akasha 不再另建 Turn 内容表或 SQL 消费表。现有学习图本来按完整 SQLite 快照发布，`metadata.consumer_state_json` 随图保存版本、旧学习前缀身份、切换时各 Session head，以及已应用的 learning binding、结束消息、成员/观察引用和来源 digest。learning binding 同时固定学习代码、实际投影依赖、文本规则和 embedding 空间；不再并存一个可与它矛盾的投影版本字段或无可执行物的 policy 名称。数组位置就是旧前缀之后的图节点位置，不再复制 node ID 或一份跨 Session 序号。消费状态不保存消息正文；它属于 Akasha，Turn Projection 仍然只有无状态读取函数。

旧前缀由固定旧索引及完整学习材料 digest 验证，恢复直接装载已学习图，不重复 `MemoryCycle.commit`。新后缀恢复必须打开原 learning binding，从结束消息对应的完整日志前缀重投影，逐项校验成员、观察和 digest，再使用固定消息 embedding 还原临时学习材料。缺失 artifact、来源、向量或版本不一致都停止恢复；不存在自动换新投影、自动重嵌入或自动重学。默认问答学习规则接纳有非空输入和最终回答的 complete 样本；quiet、没有输入的通知与 open/abandoned 仍可读取，但不成为普通问答节点。来源选择由 Akasha 配置，不在 Core 固定 proactive/passive 名称。

图与消费进度在同一候选文件中完整校验，文件 fsync 后 replace，再 fsync 目录。发布发生错误时当前消费者停止服务；它不能仅回退内存来声称文件未提交。重新读取耐久快照后决定是否已经消费，重复结束消息不重复强化。旧 writer 不得覆盖已有消费状态的图。普通程序只使用消费者读取/提交接口，不获得任意数据库改写能力。

本层 yoyo `20260905_04_akasha_consumption` 在安装锁内验证真实 schema 谱系、旧学习图和旧索引，创建可恢复 SQLite 备份，在副本仅增加切换元数据，验证学习图逻辑 hash 与旧索引均不变后发布。切换记录本身允许“文件发布成功、yoyo 未落账”后的幂等重试。`20260905_05_message_embeddings` 接管已有消息向量表；旧库缺表时先备份再建表，不改已有向量。新库由 MessageLog owner 初始化同一 schema，现有库不能借普通启动绕开 yoyo。

放弃工作保留聊天正文，但请求投影剥离该来源未结束区间里的工具协议、model facts 和旧 opaque continuation。此前已完成的调用/结果仍然配对；其他来源按各自的放弃边界处理。晚到结果继续留在日志，不混入新段。未启动调用可由 Tool 的普通拒绝能力提交 denied，且无需打开旧可执行物；已有执行先排空并保留真实结果，只有 start 而无可确认结果时不能伪称 denied。

实际程序组合已验证 Content 检查器覆盖工具结果与最终 Output 的提交；ReAct 建立子任务时显式转交 exact RuntimeScope，不允许靠继承 ContextVar 冒用权限。Output 写权随来源任务撤销，已开始工具的 ToolResult 写权保留到真实结算。普通工具执行仍沿原有取消合同；新 Input 或 Control 在取消通知到达前，也会阻止尚未开始的旧效果。

这些条目还在接线和验收中，不代表第 08 层可以单独运行于生产。整栈仍须完成真实 ingress、读者、Delivery 与其他来源切换以及最终旧链路删除。

工具需要调用出处时，`prepare(arguments, CallSource | None)` 只取得已提交 ToolCall 的不可变 Session 前缀和 CallRef。来源和上界由消息本身推导，不重复保存字段。独立调用没有 CallSource；目标不拿 MessageReader、writer 或 Task。所有影响实际调用的隐式参数必须在 prepare 解析为最终参数并随 prepared receipt 固定；invoke/query 不读取当前 Session，重启也不重新 prepare。


反馈工具的最终参数保存真实 Message ID；旧 `current_user_message` 仅是工具输入别名，按已提交调用的前缀解析为最后一个同源 Input。反馈目标从已发布 Consumption 与固定旧索引读取，所有多输入成员映射到同一图节点。未知目标、遗忘尚未学习的当前输入和同节点相反动作在准备阶段返回可修正的工具 error；该结果与 done receipt 一次提交，没有 start 或外部调用。一个 Output 中的反馈请求联合检查，因此不必读取后来接纳的 Input 或等待同批 ToolResult。若同批反馈带参数转换贡献，原始请求不足以证明最终联合语义，明确要求分开调用；普通单次调用仍执行已固定转换。

新图的 embedding 空间不另存第三份字段：旧图的模型身份来自固定旧索引，新后缀来自原 learning binding，维度来自真实向量。恢复在每个 binding 的材料还原前核对；在线在任何 embedding 调用或写入前核对。不同模型即使维度相同也不能混进一张图；不匹配必须走单独的显式重建合同。

实际查询由 Akasha 的 `recall:*` OwnerStore 记录拥有，与 `cycle.recalls` 的学习读出证据分开。后者允许因图推进重算，不能冒称曾出现在 Prompt 中。新记录保存原学习绑定、图版本、context 前缀或实际 CallRef、筛选参数、按结果顺序排列的完整学习成员引用与分数；`presented_message_ids` 另保存预算内实际呈现的消息，Citation 只能从这份材料取得本地引用证据；没有 Message 来源的独立程序查询保存自己的 key 与 query 文本。查询完成并发布记录后才能返回 Materials.references；零命中也有查询事实。Citation 的 `retrieval_ref` 引用这份记录，continue 与 complete 都不依赖稍后的学习提交。记录表示实际查询，Citation 表示本地材料关联；目前 Model 仅记录 request digest，不能据此独立证明具体材料已发送给 provider。新查询记录使用已有 owner_records，因此没有空 yoyo；旧学习表与索引继续由本层 04 迁移保留。实际召回、Context 材料和归档 recall 工具已在候选入口接通；正式入口切换仍须完成本层其余功能与累计验收。


消息运行对象已将 Context 查询与学习接在同一个串行 owner 上。查询使用调用者固定的完整 Message 前缀，多输入共享学习阶段的文本与向量聚合；已有消息向量只读，缺口先补齐。图读取不推进学习；先按预算构造 Materials，再将命中成员与实际呈现成员一次保存，发布成功后才返回引用。零命中仍保存真实查询记录。关闭自身被取消时，也先排空查询线程并归还 writer，之后才传播取消。

归档召回所需的读取路径先用 SQLite 原生备份固定临时副本，再复用原 learning binding 的完整恢复校验。它不抢正式 writer、不打开模型或重学历史，调用者只拿副本图与消费出处。正式 writer 在读取期间继续发布不会改变该副本；缺图、缺出处或缺归档不能降级为空图。临时副本在读取作用域退出后清理，不减少正式学习状态。归档 recall 工具的 prepare 固定学习 binding、模型选择和呈现预算。插件同名配置由 Manager 一次读取并随 binding 归档；重启、当前配置变化或源码移除后，invoke 仍读取原图和原规则，query 只恢复原查询材料。候选入口仅在正式启动事件取得学习 writer，打开归档闭包不会启动第二个学习运行。


候选 Inspector 导航读取真实查询记录及其原 Message，命中成员和预算内实际呈现分别标记。移动 DTO `akasha.queries.v1` 使用有明确截断标记的文本预览，保留每项命中、全部成员和顺序；完整编码仍超过 192 KiB 时明确失败，不用丢掉尾部来伪造完整结果。UI 回调只持有启动时固定的读取函数和 MessageCatalog，不在线程池申请运行权限、调用 embedding 或重算学习图。当前列表仍读取该 owner 的全部查询记录后排序，第一页成本随查询总量增长；本层没有为显示新增一份并行持久索引。旧学习诊断、本轮召回卡片及桌面接线还须按各自事实来源迁移，不能把导航 Inspector 通过当作这些功能完成。

### 08：摘要缩减与回复请求的接入进度

候选回复链已把缩减回调归入 `ContextMaterials` 的唯一 `summary_source` 授权。一次请求只取得一遍 Prompt、检索正文和引用；摘要 owner 只返回新的已发布 `Summary`，不能重写其他材料。Context 仍是纯组装器，本地容量错误携带已构造的完整请求。ReAct 先询问软水位或本地超限缩减，真实 provider 返回容量错误后至多强制缩减并重试一次；没有缩减进展时保留原错误。

`None` 表示保留原摘要；同一引用不能换正文或来源；只换引用而正文、来源都不变不算缩减。耗时缩减结束后，普通回复程序再次核对新输入与任务状态，避免通知尚未送达时继续旧 provider 请求。当前模型和材料的原 binding 作用域覆盖整个过程。相关 40 项测试通过，独立 Terra/xhigh 子范围评审通过；正式 compaction 入口尚未切换。

新 `SummaryRecords` 候选存储使用 compaction 自己的 `OwnerStore`，同事务创建不可变摘要与推进 Session head。它保留完整来源 ID、parent/generation、正文、实际模型调用 ID 和生成条件；同一出处只能幂等重放相同内容，旧 parent 无权覆盖新 head。`compaction.summaries.v1` 只按 binding 中固定的 `record_ref/session_id` 解析原记录并检查父链，没有发布或模型调用权。候选归档测试证明：head 推进、进程重启、当前源码移除后，仍读取原摘要而非当前 head。

候选生成器保留完整近期分组；跨来源交错不会切开 call/result。Context 和生成器共用已结算或同来源明确放弃的前缀判断，abandon 无须伪造 ToolResult。摘要请求按模型容量分批，provider 拒绝时只减少本批完整分组；单组过大明确失败。可恢复的生成错误才使用当前 execution 已固定的 DEFAULT，配置或请求合同错误直接传播。正式 shell adapter 接线前仍须补齐活跃 execution 的原文保留，不能只凭摘要提示替代旧合同。

当前 open Turn 的 Inputs 由现有 TurnProjection 选定，摘要覆盖它们后仍在本次请求中按原 seq 呈现，并读取原 Artifact；旧输入、其他来源输入及工具协议不重复。摘要覆盖所选近期窗口中的连续区间，不回填首次窗口之前的旧历史，也不另存一份 tail 正文。新的摘要引用明确开启 fresh 请求，原 opaque 留在日志；成功 Output 同时保存 `model.facts` 和实际 `context.summary` 后，后续同摘要请求可以接续自己的 continuation。直接只传 cutoff 的旧调用保留拒绝不明 opaque 的边界。

成功模型 Output 的 `continue` 与 `complete` 都随正文原子追加实际使用的摘要 binding；普通 Input 没有该内容 grant。失败 provider 或未提交 Output 不产生使用记录。候选 Markdown 消费者在启动后跟随这些消息，按摘要出处复用既有 before-image/draft/applied receipt。若某个 parent 摘要从未被使用，后续使用 child 时仍从最近已写入祖先之后取完整原文，不遗漏 parent 覆盖的事实。恢复测试证明 Output 先提交后停机可以补写；MEMORY 已写而 SELF 失败，或 SQLite 准备只写了一部分时，重启从完整 model draft 补齐 order 与两份 document draft，不重新调用模型。进度从既有双文件 applied receipt 派生，并核对同一父链；迟到的旧 generation 不会在已应用 child 之后重写档案。文件锁使用可取消的非阻塞等待，等待中与持锁时取消都关闭句柄。

当前真实候选默认回复已验证“生成并发布摘要 → 业务模型 → 工具结算 → 最终回答”，两次成功响应保留同一摘要引用，所有旧 Message 原文不变。请求只发送摘要正文和 binding，完整来源 IDs 留在 owner 记录中，避免长历史身份列表再次撑满窗口。本子范围 90 项相关回归通过，目标源码类型检查无错误和警告；独立 Terra/xhigh 评审的 P1 已清零。旧 Markdown 入口原有 18 个类型 warning 不在此零告警范围内。本文的候选进度不等于第 08 层完成：旧 ledger/prepare/receipt 迁移、新 programmatic 来源的投影资格、活跃 execution 保留，以及正式入口切换仍待整体验收。


摘要模型材料已补齐迟到结果过滤：按完整 Message 快照中的实际 ToolCall、同来源 abandon 与 `through_seq` 判定；控制事实在上一代摘要范围内时同样有效。只有放弃之后才到达的对应 ToolResult 正文被排除，其他来源的观察保留。原始日志、摘要来源 ID 范围和近期分组不变。相关摘要、默认回复与归档记录 17 项回归通过，两个目标模块类型检查无错误和警告；独立 Terra/xhigh 对本子边界复核 PASS。Markdown 随后的原文消费也使用完整快照的同一迟到结果过滤；实际双文件投影测试证明迟到正文不进入模型或档案，而原消息保留。

**首次范围已确认（2026-09-06）：** 用户要求取最近的 `<= 阈值` 窗口。Context 估算完整业务请求，包含系统材料、检索、工具与当前输入，并同时检查输出预留后的硬边界；从最近完整单元逐步增加，下一单元使其越界就停止。已完成 Turn 及跨来源交错的重叠区间不能拆开；open 工作的已结算工具批次仍可按既有合同压缩，当前 Input 原文重呈。`Summary.source_message_ids` 定位实际连续区间，不增加重复的 start_seq；子代必须同起点且严格扩展。窗口之外的更早消息既不进入摘要模型，也不进入业务请求，但保留在原始日志。

旧 `session_compactions` 的首代可能只覆盖近期窗口，后续行保存增量；旧 v4 receipt 也没有新 `model_call_ids` 或成功 Output 的 `context.summary` 使用证据。因此不能自动将旧行发布为新摘要 head，或虚构模型调用与使用记录。旧 ledger、prepare 与 receipt 保持恢复证据，转换暂留代码 TODO。用户同时授权：后续未决语义保守保持原部分，写 TODO 与记录后继续独立工作；本次先完成仓库内重构及全部 draft PR 的审阅准备，外部插件源码迁移另行处理。


历史记忆排除已核对到现行事实：0041 及 `20260826_01_migrate_turn_effects` 把旧排除转换为每条原消息的 `effects.post_commit=suppress`；Message 迁移将原始 extra 保存在 `history.provenance`。已退役的 Session 标记与 scheduler 前缀规则不得重新进入 runtime。当前默认来源选择与 Akasha 切换 head 防止常规历史重学，但 operator 允许其他来源、或旧 open 输入跨切换边界闭合时，仍须由学习规则核对整个样本；Markdown 也不能只删 suppress 行后学习同一旧工作单元的其余正文。Content 已提供严格的历史原文 decoder，验证已知 schema、原始 extra digest 与重复 JSON 字段，再复用现行 effects 语义；不恢复旧字段。Akasha 和 Markdown 均按完整成员排除，Akasha 在 embedding 之前拒绝；恢复已学习样本也核对同一资格。Markdown 空资料不调用模型或制造 applied receipt，后续有真实资料的子代仍能消费。新的 programmatic 来源准入是另一项待验收合同，历史 decoder 不能代替它。


近期范围收尾验证：8 组相关测试在历史 decoder 接入前为 70 项通过；独立复核后相同集合含新历史用例为 75 项通过，近期范围概念 Gate PASS。历史排除与 Markdown 迟到结果的专项集合 30 项通过，3 个目标模块类型检查无错误和警告。以上均是第 08 层的分项证据，整层命令接入、旧 turns 保全以及最终累计 Gate 仍须完成。


压缩触发后的失败由摘要 owner 明确抛出：没有合法近期窗口、不能满足 raw tail 保留量、没有摘要资料、没有降低请求容量，或重建后仍超过软水位/硬边界，均不发布新的摘要 head、不调用业务模型。摘要模型可能已有真实调用账，它作为未被使用的审计事实保留。正常回复链覆盖有合法切点、硬容量无切点、仅软水位无切点和摘要后仍超水位四种情况；不在通用 Context 或 ReAct 中写入 compaction 特判。


### 08：命令与旧执行的迁移进度

conversation 的 `/stop` 在同一 Task 准入回调中选择当前 handle、提交 pause 并撤权；返回前等待原工作排空。重复停止不把后来接纳的新 Input 再次暂停。命令匹配发生在默认回复打开模型、材料与工具之前。不可变 command intent 只保存原 Input 与 handler binding，结果正文只在稳定身份的 Output 中存在；`CommandInvocation.message_id` 供领域 receipt 固定调用身份。只读 handler 可声明 `read_only`；带副作用的恢复只能使用原归档 handler 的 `recover` 查询领域回执，缺回执时明确失败，不能重跑。暂停和失败保留恢复意图，abandon 阻止后续调用；原 intent 与领域证据不删除。多条待恢复命令按 Input.seq 结算，旧结果不能关闭较新的输入。空成功回复以 quiet 提交。候选 Akasha 保留 `/akasha_reindex` 名称并明确报告暂不可用；固定旧规则的重建转换仍为 TODO，不自动重学。

`20260906_01_turn_messages` 只在隔离副本验收。迁移保存旧 turns 每行全部 nullable 标量及原始 JSON 文本为 `history.record`，由独立 history source 发布 quiet；默认模型内容投影不读这种审计记录。旧 messages、turns、向量与附件对象不改写或删除，原附件关系只允许增加新 Input 的引用。新增 `history.turn_input` 保留已核对的旧 item 身份与 metadata，历史 effects 排除仍由 Content owner 解读。

恢复来源的证据比旧 Turn metadata 更严格：旧 generic Control 客户端可以填写 `channelMessageId/channelSnapshotId/channelGenerationId/channelBindingToken`，所以这些字段齐全也不能证明渠道接纳。当前仅接管每个 USER_MESSAGE 都有同 Session/client 身份的独立 `inbound_handoffs`、正文及 channel/chat/sender、metadata 和 media 一致、ordinal 连续且精确 Message 引用无冲突的 open 链。缺少独立证据或旧媒体没有明确映射时完整归档并记录原因。任一可续接链仍含旧 toolCall 时，因现行 status/resultPreview 不是领域 terminal receipt，迁移停止并报告原 record/item；这项恢复合同继续保留 TODO。

Input 映射、必要的新 Input、归档、pause 与 manifest 在单个 SQLite 事务内提交。既有已映射 Message 保持原 legacy source；只有迁移 writer 可以把 conversation pause 截止固定在对应的旧全局 seq，普通 Conversation.control 仍要求同来源。迁移前创建并校验 SQLite 原生备份，提交前核对旧正文、身份、原始 JSON、向量、附件和外键；提交成功而 yoyo 尚未落账时，按 manifest 核对同一批 Message，不能重复分配 seq。本轮已验证事务提交前故障全回滚、提交后中断幂等恢复、未知 schema 和原记录损坏拒绝；这仍是仓库内候选证据，正式 workspace 未运行迁移。


Channel 接纳以 Input 原子提交为边界，Core 启动在全部渠道完成 start 后重放 durable handoff，不依赖旧回复 worker。恢复按 `(created_at, handoff_id)` 扫描有限页，跳过仍在处理的 exact owner；单行失败释放本页尚未执行的恢复 claim，保留原始交接记录。已提交 Input 重放复用原身份与 seq，只完成传输清理，不重跑回复。提交前取消保留原始附件引用，提交后取消等待删除交接记录与 lease 释放；清理暂时失败后关闭进程仍保留 durable row。旧 BUS/LANE/LOOP 测试替换关系见测试与 Gate 清理账本，公开 Mobile receipt 场景同时覆盖新 Channel 输入与独立回复消费者。正式 Core 的唯一 MessageLog 生命周期和全部客户端读取切换仍属第 10 层，当前 fixture 证据不冒充正式启动验收。

第 08 层交付前核对远端 main `31136520`：摘要请求同步 `e17cb95f` 的 provider 默认输出长度合同，不再添加旧 8192/model-cap 输出上限。当前批准的单个 Summary 与近期窗口模型不恢复旧引擎的 persistent/temporary 双摘要。其余 Host Bridge、Computer 与界面更新在最终栈顶整合并重新验收，已发布分支不重写历史。
