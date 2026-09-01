# 普通插件 Agent 骨架与被动链迁移合同

- 状态：accepted
- 日期：2026-09-01
- 决策：[0054 · Agent 内骨架由七个普通插件组成](../decisions/0054-agent-spine-is-ordinary-plugins.md)
- 取代范围：[React Core 与 Scheduler/Subagent 设计](react-core-scheduler-subagent.md) 中
  “React 实现属于 Core”的结构结论；既有 Turn、Session、Scheduler、Subagent 行为合同不变
- 实施分支：`codex/react-plugin-spine`
- 基线：`f1f4560892ae92e96779ff89f848223afdcc9919`
- Git worktree：`/mnt/data/coding/akasic-agent-worktrees/react-plugin-spine`
- 恢复引用：`backup/pre-react-plugin-spine-20260901-f1f45608`

## 1. 结果、范围与停止条件

### 1.1 目标

把完整被动回复从“bootstrap 构造一条懂所有功能的 Core 私有链”迁成“普通插件通过公开 Service
拼出一条链”。迁移后，Core 不再构造或识别 AgentLoop、PassiveTurnPipeline、Tool Search、
Command、Shell、Compaction、Markdown memory 或任何业务插件；它只发布和租用完整 snapshot。

### 1.2 完成标准

- 七个 Agent 内骨架插件全部通过同一 v3 loader、generation Root、Fiber、Effect 和 lease 运行。
- 被动 Channel、Control、Scheduler、Wake 和 Subagent 都只通过公开 `agents` Service 发起工作。
- 一次工作从 source 到 provider/tool、Session commit、delivery/ACK 始终绑定 exact snapshot。
- 当前硬编码特殊功能都有普通 owner，或被证明是 ReAct 直接算法的一部分；Core 无名称特判。
- 旧 `AgentLoop`、`PassiveTurnPipeline`、`ConversationRuntime` bootstrap wiring、总 phase bundle 和
  专用桥零 consumer 后物理删除；最终无 alias、adapter、flag、fallback、双写或兼容壳。
- 受保护行为、持久 write set、事件、外部调用、错误和取消语义尽可能等价；只有单独批准的
  差异才能进入验收清单。
- Core 完成后停止；不修改独立外部插件源码仓库，不直接编辑安装 cache，不写正式 workspace。

### 1.3 Change intent

```yaml
change_type: migration
semantic_delta: none
capability_owner: mixed
consumer_scope:
  - passive channel
  - control
  - scheduler
  - wake
  - subagent
runtime_patch: required
runtime_patch_reason: 当前 Core 私有调用链阻止普通插件拥有完整 Agent 组合
authoritative_state_owner: sessions ordinary plugin; each external effect keeps its domain owner
client_only_alternative: 客户端无法拥有服务端 Turn、Session、snapshot 或 delivery 语义
concept_gate: required
concept_gate_reason: 改变 Core、bootstrap、Service owner、lifecycle 和公共扩展边界
invariants:
  - SES-001～SES-008
  - RUN-001～RUN-012
  - OUT-001～OUT-005
  - PLG-001～PLG-018
  - TST-001～TST-006
protected_state:
  - sessions.db 全部既有内容与 schema
  - messages 只追加和 seq 高水位
  - compaction ledger、附件、memory 文件与 plugin-data
  - generation、lease、candidate isolation 和 publication journal
  - 渠道发送、ACK、取消、中断、stream 和错误分类
allowed_paths:
  - agent/plugin_composition/**
  - agent/plugins/**
  - agent/control/**
  - agent/core/**
  - agent/looping/**
  - agent/lifecycle/**
  - agent/tools/**
  - bootstrap/**
  - bus/**
  - session/**
  - plugins/**
  - tests/**
  - docs/**
forbidden_paths:
  - 正式 Akashic workspace
  - ~/.akashic-plugin/cache
  - 独立外部插件源码仓库
allowed_effects:
  - 独立 Git worktree 内源码、测试和文档
  - pytest 临时 workspace 与隔离 candidate 数据
  - 一个持续工作的 Draft PR
forbidden_effects:
  - 生产流量灰度或 shadow
  - 旧新双执行、双写、双 sender
  - 正式数据库、消息、远程 API、服务或插件安装变更
validation:
  - 每批关键行为和 write-set oracle
  - 每批两个独立 Terra xhigh review
  - 最终 zero-consumer、全量 test 和 project Gate
rollback: 上一完整 commit、不可变 generation、执行前备份；不伪造外部效果回滚
worktree_writer: /root
external_revisions: []
schema_lineages: [sessions.db current schema unchanged]
```

## 2. 六岁小孩版

现在像一辆玩具火车：车头里同时焊死了电池、方向盘、喇叭、货箱、售票员和清洁刷。换一只
喇叭，也要拆车头。

目标是七块普通积木：

```text
┌──────────┐  保存故事      ┌──────────┐  选择大脑
│ sessions │               │ models   │
└──────────┘               └──────────┘

┌──────────┐  使用工具      ┌──────────────┐  拼系统提示
│ tools    │               │ system-prompt│
└──────────┘               └──────────────┘

┌────────────────────┐  把故事投影成这次要看的页面
│ session-projections│
└────────────────────┘

┌──────────┐  管“这次工作是谁、能否取消、何时结束”
│ agents   │
└────┬─────┘
     │ runner slot
     ▼
┌──────────┐  重复“大脑想一下 → 调工具 → 再想一下”
│agent-loop│
└──────────┘
```

还有一个很小的门卫，但门卫不是第八块特权积木。门卫不认识故事、大脑或工具；它只给每项工作
一张不透明号码牌，锁住这项工作使用的同一代积木，并记住“有人按停止时通知哪项工作”。演完后
号码牌和锁一起归还。这样一次 ReAct Turn 全程走同一 snapshot，旧工作仍能被停止，盒内每块积木
仍然平等。定时器醒来只能拿回自己的旧盒；旧盒已经退休就明确失败，不能偷偷换新盒。

Tool Search、Compaction、Markdown memory、Shell、Scheduler 等是拿这些积木拼出的玩具，不是
Core 为每个玩具新发明一块特殊原子。

## 3. 已核对现实

### 3.1 当前调用链与 owner

```text
PassiveMessageWorker
  └─ ConversationRuntime
       ├─ admission / active attempt / cancel / terminal
       └─ bootstrap executor
            └─ AgentLoop._react / _process
                 ├─ command short-circuit
                 ├─ plugin rollout prompt fact
                 ├─ model selection
                 ├─ shell cleanup
                 └─ PassiveTurnPipeline
                      ├─ before/after phase bundle
                      ├─ prompt + provider + tool loop
                      ├─ tool_search / message_push special cases
                      ├─ Session transaction
                      └─ outbound projection
  └─ durable terminal / handoff delete / Channel ACK
```

代码事实：

- `bootstrap/tools.py:687-763` 在插件加载前构造 Session、Tool registry、PluginManager 和
  AgentLoop；`bootstrap/app.py:254-287` 再把固定 executor 塞入 ConversationRuntime。
- `agent/plugins/manager.py:408-421` 在 Root 构建前接收 ConversationRuntime；
  `agent/plugins/manager.py:5323-5340` 又从它制造 Core-owned `SCOPED_TURNS`。这构成当前环。
- `agent/plugins/snapshot.py:76-114` 已把完整 Composition Root 放入 RuntimeSnapshot；
  `agent/plugins/snapshot.py:876-907` 已保证 exact lease 排空。
- `agent/plugin_composition/context.py:236-311` 已提供 mount/inject/provide/require/effect，
  `agent/plugin_composition/context.py:350-397` 已提供 typed dispatch。
- `agent/plugins/manager.py:5451-5458` 已有 `snapshot.sealing`；不需要新建第二套 readiness 图。
- `session/manager.py:642-802` 已有原子 message commit、append 和 durable delivery 事务 owner；
  迁移 owner 不能改变这些 write set。

### 3.2 DSH 的七个零件

DeepSeek Harness 的 AgentLoop 注入 `agents`、`sessions`、`llm`、`tools`、`systemPrompt` 和
`sessionProjections`，自己是第七个零件。它的 loop 只负责模型/工具重复，持久化等由其他插件
补充。Akashic 复用这个依赖形状，不复制 DSH 的内存 SessionStore，也不放弃现有 SQLite、
generation 或 recovery owner。

### 3.3 事实、推断与未知

| 类型 | 内容 |
|---|---|
| 已核对事实 | Context/Service/Fiber/Effect、Root sealing、snapshot lease、普通 models/compaction/markdown 插件已经存在 |
| 已核对事实 | `SCOPED_TURNS` 由 Core 从 ConversationRuntime 制造，是 bootstrap 环的关键，不是必须保留的业务 owner |
| 已核对事实 | 当前 passive 文件按工具名识别 `tool_search`、`message_push`，AgentLoop 按名称/类型识别 Shell |
| 设计推断 | 一个泛型 snapshot Service 调用边界足以让 snapshot 外入口进入普通 `agents` Service |
| 设计推断 | runner slot 让 `agents → sessions` 与 `agent-loop → agents + 其余服务` 保持无环 |
| 实施中核对 | 每个旧 phase 的动态外部 consumer；未完成零 consumer 证明前不删除对应接入点 |
| 实施中核对 | Mobile attention、Meme/Citation 和 attachment 的精确外部 payload；只迁 owner，不改协议字段 |

### 3.4 当前外部 consumer 风险

2026-09-01 对 `/mnt/data/coding/akashic-plugin` 与已安装 cache 的只读扫描未发现外部插件注入
`SCOPED_TURNS` 或 import `AgentLoop`/`PassiveTurnPipeline`；当前直接 consumer 都在本仓库的
Scheduler、Wake 与 Subagent，因此该 Core bridge 可以在 M6/M7 内完整替换。

外部 Citation、Meme、Observe、Emotion、Proactive Feedback 和 GitHub Watch 使用
`AFTER_REASONING_*` 或 `AFTER_TURN_COMMITTED` 等 stable typed event。它们是已有普通 v3 接入点，
不是 `SCOPED_TURNS` 特权桥，Core 迁移期间必须保持可观察 payload 和顺序。外部 Observe 测试仍
import 旧 phase frame，GitHub Watch 的跨仓 Gate 仍构造 fake ConversationRuntime；这些是下一阶段
需要迁移的源码 consumer，不授权本 PR 保留旧 Core 实现、re-export 或测试兼容壳。M8 必须输出
exact consumer/commit 清单后停下，待外部源码 PR 更新后再做跨仓最终组合 Gate。

### 3.5 直接复用与必须退役的资产

| 资产 | 处理 |
|---|---|
| Context / ServiceKey / Inject / Fiber / Effect / typed dispatch | 原样作为唯一 composition kernel，不另建容器或 hook bus |
| RuntimeSnapshot、Root sealing、stable/latest、lease、candidate closure | 原样作为 publication 真源，只补中性 port/task/operation 能力 |
| `TOOL_CATALOG`、`PluginTools`、工具 snapshot freeze | 演进为 `tools` 插件的唯一 registry，不创建平行 ToolRegistry |
| 现有 `plugins/models` Services | 直接作为 `models` 基础插件，不复制 provider/model catalog |
| 现有 compaction/markdown-memory 普通插件 | 保留 owner，只改为注入新 prompt/projection/session public port |
| SessionManager/SessionStore 的事务和恢复算法 | 行为与测试资产保留，真实实现迁入 `sessions` owner；不包旧 singleton |
| `PluginScopedTurns` 的 exact root、accepted handle、retired error 语义 | 迁入 AGENTS + RootTaskScope；旧 `SCOPED_TURNS` key/bridge 最终删除 |
| existing ActivityHost/admission-drain 模式 | 用作 operation supervisor 的实现证据，不复制 Agent 专用 publication plane |
| AFTER_REASONING/AFTER_TURN 等 stable typed events | 有真实外部 consumer 的公共事实继续保留 payload/order，不保留旧 mutable phase wrapper |
| bootstrap AgentLoop/SessionManager/ToolRegistry construction 与 manager Core-service manufacturing | deprecated 后退役；它们是待删除 owner，不是可长期复用 adapter |

## 4. 最终能力与唯一 owner

### 4.1 Core publication kernel

Core 只保留：

- 插件 artifact、generation 和完整 Root 的构建、验证、发布、丢弃与恢复；
- stable/latest 指针、exact lease、retire/drain 和 Effect cleanup；
- 绑定单一 `ServiceKey[T]` 的 `RuntimeServicePort[T].call(operation) -> R`；
- 每个 Fiber 平等取得的 `RootTaskScope`，以及按 Service namespace 隔离的
  `OperationSupervisor`/窄 cancel port；
- composition diagnostics、最小 workspace file grant 和外部 host 的通用资源开关。

kernel 在 bootstrap composition 时为外部 host 创建绑定一个 exact `ServiceKey` 和固定 lease source 的
`RuntimeServicePort`；host 不取得任意 service lookup，插件也不取得 port factory。普通 host 的 lease
source 永远取得 stable；公开 `call(operation)` 不接受 selector、snapshot ID、plugin ID 或 lease。
attached validation child 只使用 Core 根据父 Turn、candidate generation/source identity 铸造的一次性
exact lease，不能由 host 或插件选择 latest。port 绑定当前 task，从 exact Root
`require(bound_key)`，完整等待 operation，再解除绑定并释放。Service 缺失、Root/identity 不一致、
继承到错误 task 或 lease 已退休全部 fail-loud。它不解析 request，不创建 background task，也不
捕获领域错误。

### 4.2 七个普通插件

| 插件 | 独占事实或变化轴 | 公开能力 | 明确不拥有 |
|---|---|---|---|
| `sessions` | Session/Message/Turn/attachment 的 SQLite 事实与事务 | `SESSIONS`: read snapshot、admit/terminal、atomic commit、窄 compaction/attachment/delivery ports | Prompt、模型、工具、Channel 发送、任意删除 |
| `models` | provider、model revision、role 与 Turn-frozen binding | 现有 `MODEL_DRIVERS`、`CHAT_MODELS`、`EMBEDDINGS`、catalog/settings | Session、Prompt、loop |
| `tools` | 工具定义、当前 Turn 可见集合、调用结算 | `TOOLS`: register、open turn view、present、authorize、execute；结构化 `ToolOutcome` | Prompt 文案、Session SQL、特定工具策略 |
| `system-prompt` | 有序 Prompt section registry | `SYSTEM_PROMPT.build(input)` 与 section contribution | persistent history、provider 调用、记忆文件 |
| `session-projections` | Session 快照的可重建 provider/展示/提交后投影 | `SESSION_PROJECTIONS.prepare/committed` 与窄 contribution | 权威 history、cursor 删除、外部发送 |
| `agents` | agent registry、Turn admission/取消/terminal 的领域规则 | `AGENTS.start/cancel/read`；runner register slot；typed Turn facts | task/lease 的跨代机械路由、ReAct、模型、工具、Channel 规则 |
| `agent-loop` | 一次直接 ReAct 的控制流 | 向 `AGENTS` 注册唯一默认 runner；内部 provider/tool loop | 持久 owner、发送、来源枚举、业务插件名 |

`sessions` 可以声明 `workspace_files=("sessions.db",)`，但只有它获得正式 writable grant。
candidate closure 中的 `sessions` 使用插件自己创建的全新临时 schema 和 programmatic Session，
不复制、读取或写正式 `sessions.db`；它是验证数据，不是第二名正式 writer。需要历史语义的回归由
测试把固定 fixture 恢复进一次性 workspace 后串行运行，不从 live DB 取样。

### 4.3 三个中性执行原子

| kernel atom | 只拥有 | 不拥有 |
|---|---|---|
| `RuntimeServicePort[T]` | 构造时固定的 ServiceKey 与 lease source；一次完整 call | selector、request 解析、background task、领域 fallback |
| `RootTaskScope` | owning Root identity、task/Effect cleanup、root-bound lease acquire | stable/latest 选择、领域 retry、跨 Root 重投 |
| `OperationSupervisor` | opaque scope/operation claim、exact lease、task、cancel callback、terminal release | Message/Turn/Session、runner、持久状态、错误解释、delivery |

`OperationSupervisor.claim(scope_key, operation_key, exact_lease, task, cancel)` 对整个进程原子，
同一 opaque scope 跨 generation 只能有一个 active operation。`agents` 负责把自己的 session/attempt
领域身份映射成稳定 opaque key，并负责何时允许 start/cancel/terminal；accepted receipt 与 durable
active-attempt fact 保存同一个 operation key。supervisor 只执行 claim、按该 key 通知原 owner 的
cancel callback 和最后释放。Control host 只获得 `cancel(opaque_operation_key)` 窄 port，不能枚举
operation、读取结果、创建工作或取得 snapshot。新 ingress 要 interrupt 旧 attempt 时，先从
`SESSIONS` 窄 read port 取得 durable active operation key，不能按内存对象或 current stable 猜测。

这使新 stable 的 `agents` 能请求取消旧 Root 的仍活 task，但旧 `agents` 和旧 runner 继续唯一负责
terminal/Session settle，并在最后释放旧 lease。这里没有内存状态搬家、两代共同写或特权 Agent
service。`ActivityHost`/generation lease 的现有 admission/drain 语义是实现资产；不得再创建一份
Agent 专用 publication 平面。

`RootTaskScope` 由每个 Fiber 平等取得。`agents` 实例在 apply 时绑定自己的 root scope；
`AGENTS.start` 只复用同 Root 的 current lease，或向该 scope 取得 owning Root lease，遇到其他 Root
binding 直接失败。Scheduler/Wake 的 timer callback 因而可以直接调用同 Root 注入的 `AGENTS`；Root
已退休时原样得到 `TurnAdmissionRetiredError`，由 Scheduler/Wake 自己 settle/rearm，绝不 fallback
到 current stable。candidate Root 的普通 background scope 关闭，只有 Core 铸造的 attached
validation capability 能启动一次 candidate operation。

### 4.4 无环注册

```text
foundation providers
  sessions ──► SESSIONS
  models ────► CHAT_MODELS ...
  tools ─────► TOOLS
  prompt ────► SYSTEM_PROMPT
  projection ► SESSION_PROJECTIONS

agents injects: SESSIONS
agents provides: AGENTS + empty runner slot

agent-loop injects: AGENTS, SESSIONS, CHAT_MODELS, TOOLS,
                    SYSTEM_PROMPT, SESSION_PROJECTIONS
agent-loop effect: register(default runner) ── cleanup unregisters

snapshot.sealing: exactly one default runner, every registry frozen
```

`agents` 不 inject `agent-loop`，因此没有 Service cycle。Root 未 seal 前不能取得正式 lease；
seal 后 runner slot 不再改变。热重载发布的是另一棵完整 Root，不原位替换 live runner。

### 4.5 代码与公共合同边界

七个插件的实现最终位于各自 `plugins/<name>/` 包，或其独立安装 artifact 中。普通插件只能 import
版本化 public Plugin API、结构 DTO/Protocol/ServiceKey 与自身包；不得为了复用旧实现继续 import
`bootstrap.*`、`agent.looping.core`、`agent.core.passive_turn`、`PluginManager`、SessionManager 私有
store 或兄弟插件源码。

`Message`、`Turn`、`Session` 的稳定结构合同和 Service protocol 可以留在中立 public API 包；
它们不包含实现、全局 singleton、workspace root、任意 SQL 或 publication 控制。迁移旧算法时移动
真实 owner 的代码，而不是在新插件里包一层旧 Core class。Core/Bootstrap 只为边界 host import
公开 `ServiceKey` 来绑定窄 `RuntimeServicePort`，不 import provider implementation。

## 5. 完整链怎样走

### 5.1 被动消息

```text
ordinary Channel plugin
  └─ ordinary conversation plugin
       ├─ explicit command? ──► COMMANDS ──► delivery settle
       └─ normal Message ─────► AGENTS.start
                                │ fork exact current lease for owned task
                                ▼
                         agents admission/Turn owner
                                │ registered runner
                                ▼
                         ordinary agent-loop
                ┌───────────────┼────────────────┐
                ▼               ▼                ▼
          session snapshot   prompt/project   model + tools
                └───────────────┬────────────────┘
                                ▼
                     sessions atomic commit
                                │ committed fact
                                ▼
                    conversation delivery + ACK
```

一次 Turn 的 exact lease 从 `AGENTS.start` 原子 claim 到 `OperationSupervisor`，直到 terminal 后释放。
Channel 回调提前返回时 lease 仍由 supervisor 持有；取消只通知当前 attempt，旧 Root 的 agents/runner
继续完成 cleanup 和 terminal。新 generation 不能 claim 同一 session scope。

### 5.2 Control、Scheduler、Wake 与 Subagent

- Control host 只持有 bootstrap 为 `AGENTS` 绑定的 `RuntimeServicePort`；它不直接 import AgentLoop，
  也不能借该 port 查询其他 Service。
- 正常 `/stop` 可以由当前 `AGENTS` 读取 durable active operation key；publication 暂停、没有 stable
  service lease 时，Control 从 accepted receipt/Control store 取得同一 key，只用 kernel 给它的窄
  operation cancel port 通知已接受的旧工作。
- Scheduler/Wake/Subagent 已在 Root 内时直接 inject `AGENTS`；`AGENTS` 实例自己的 RootTaskScope
  保证 timer/后台 callback 只能取得 owning Root。Root 已退休就收到
  `TurnAdmissionRetiredError` 并由来源 settle/rearm，不得改投 current stable。各自 gate、spawn、
  持久状态和 delivery 仍由原插件拥有。
- 来源只构造普通 Message/Turn request，不复制模型、工具、Prompt、Session commit 或 cancel loop。
- 不适用的 feature plugin 没有 contribution；不存在“先运行 passive hook 再 early return”。

### 5.3 snapshot 本身包住完整 ReAct Turn

这是需要保留的安全性质，不是需要特权插件的理由：

```text
outside snapshot             inside one exact snapshot
─────────────────┬────────────────────────────────────────────
AGENTS-bound port │ require(bound key)
acquire lease ────┼─► agents ─► runner ─► model/tools/session
                  │                    └─ supervisor owns opaque op + lease
wait result ◄─────┼────────────────────────────────────────────
release lease ────┘
```

“谁保管 lease/task/cancel callback”与“谁解释 Turn 并实现 ReAct”是两条正交轴。前者属于领域中性
operation supervisor，后者属于普通 `agents`/`agent-loop`。把二者写进一个 privileged plugin
反而重新制造 bootstrap cycle。

## 6. 当前特殊功能清单与目标 owner

| 当前特殊点 | 当前位置 | 目标组合 | Core 新增专用原子？ |
|---|---|---|---|
| command 在模型前短路 | `AgentLoop._process`、`PassiveTurnPipeline.run_command` | conversation source 注入普通 `COMMANDS`，识别后不创建 Agent Turn | 否 |
| plugin rollout fact 塞入下一轮 Prompt | `AgentLoop._process` metadata | rollout 插件向 `SYSTEM_PROMPT` 提供一次性 section；事实文件由其声明 | 否 |
| session 模型选择 | `AgentLoop._resolve_model_selection` | models 插件通过 `SESSIONS` 窄 metadata port 读取/提交，返回 frozen binding | 否 |
| Shell 按工具名和类 cleanup | `AgentLoop._cleanup_shell_owner` | Shell 插件监听 `agents` 的 Turn terminal，并清理自己拥有的 execution | 否 |
| Tool Search enable、schema cap、LRU、名称解锁 | `DefaultReasoner` 多处分支、ToolRegistry meta set | Tool Search 普通插件注册普通 tool；用 `TOOLS` 的 catalog search 与 turn-local schema grant | 否 |
| 未解锁工具的提示文字 | `DefaultReasoner` | `TOOLS.authorize` 返回结构化 denial；Tool Search 插件提供模型可见说明 | 否 |
| `message_push` 媒体抽取 | tool loop 按名称收集 | 普通工具返回 `ToolOutcome` 的 typed durable items/delivery facts；投影 owner消费 | 否 |
| `mobile_attention` | Reasoner/Turn result 固定字段 | Mobile output projection 插件消费 typed tool/turn fact并保持现有协议字段 | 否 |
| Meme/Citation response decoration | after-reasoning/after-turn consumers | `SYSTEM_PROMPT` section + `SESSION_PROJECTIONS`/outbound contribution | 否 |
| Skills、memory、hints | before-reasoning phase | 普通 prompt/tool contribution；required Service 显式 inject | 否 |
| Compaction request gate | provider call seam | 已有普通 compaction 插件向 provider request projection 注册 | 否 |
| Markdown MEMORY/SELF 写入 | committed checkpoint 后 | 已有普通 markdown-memory 插件 | 否 |
| streaming、thinking、tool progress | AgentLoop sink + EventBus | agent-loop 发算法事实；agents observer/source projection 消费 | 否 |
| Session commit 与 outbound 混在 after-turn | PassiveTurnPipeline | `sessions` 先原子 commit；conversation/Channel 后 delivery/ACK | 否 |
| 六组可任意改写总状态的 phase | `agent/lifecycle/phases/**` | 收敛到 owner 明确的 Prompt/Tool/Turn/Projection 接入点 | 否 |
| provider retry、max iteration、tool batch、continuation | `DefaultReasoner` | `agent-loop` 内部直接算法，不拆成 feature plugins | 不适用 |
| attempt admission、interrupt、cancel、terminal | `ConversationRuntime` | `agents` 插件唯一 owner | 否 |
| durable inbound handoff 与 ACK 顺序 | `PassiveMessageWorker` | ordinary conversation plugin，持久写只请求 `SESSIONS` 窄 port | 否 |

禁止用 `TURN_EFFECTS`、万能 middleware、任意 mutable context 或一个“passive hooks”总 Service 把这些
重新装进一只袋子。每个 public seam 必须指向表中已有 owner 与一种明确变化轴。

## 7. 持久状态、外部效果与恢复

| 对象 | 正常增加 | 可原位更新/逻辑终态 | 物理减少 | 唯一 owner 与恢复证据 |
|---|---|---|---|---|
| `sessions.db/messages` | completed transcript 原子 INSERT | 不更新正文 | 仅 SES-003 显式用户撤销/删除 | sessions；DB backup、row/seq/write-set diff |
| `sessions` metadata / `turns` | admission、attempt、terminal 写入 | 仅既有状态机和白名单 metadata | 仅既有管理协议 | sessions；turn identity、terminal、restart recovery |
| attachments/compaction/delivery rows | 既有事务增加 | 按各自 prepare/commit/settle 状态机 | 只按现行独立合同 | sessions 窄 port；digest、receipt、prepare fence |
| MEMORY/SELF 与 receipt | committed checkpoint 触发 | backup + atomic replace / idempotent receipt | 只按 MEM 条款 | markdown-memory；backup、source_ref、receipt |
| plugin rollout fact | rollout terminal 增加一次临时事实 | consume 逻辑终态 | 成功消费或已批准恢复 | rollout plugin；fact/journal |
| Shell/process | 工具显式启动 | active → terminal/cleanup_degraded | owner 确认退出后 | Shell/Workload plugin；process registry/report |
| Channel send / remote API | prepared 后调用 | committed/partial/failed/outcome_unknown | 外部效果不可由 Git 删除 | Channel/Delivery/tool owner；provider receipt |
| snapshot/candidate | publication transaction 增加 | state、lease count、stable/latest 指针 | drain 后清理不可达代 | Core kernel；journal、identity、zero lease |

本迁移不改 schema，不迁正式 workspace，不 UPDATE/DELETE 既有消息，不复制正式数据库做第二个 writer。
`sessions` owner 切换必须暂停新 admission、排空旧 snapshot lease、关闭旧 SQLite owner、打开新 owner、
核对同一路径 integrity 和高水位后才恢复；失败反向关闭新 owner并用旧完整 artifact重开。该窗口没有
两名正式 writer。

## 8. 失败、取消、并发与 reload

- **缺依赖：** required Service、runner 或 exclusive writer 缺失时 Root sealing 失败，stable 不变。
- **普通错误：** provider、tool、Prompt contribution、Session commit 和 delivery 保持现有错误分类；
  只有拥有恢复动作的边界转换错误。
- **取消：** 当前 attempt 收到取消；agent-loop 完成工具/外部效果既有 settle，agents 只提交一次
  terminal，operation supervisor 最后移除 opaque record 并释放 lease。reload 后 cancel 仍调用旧
  record 保存的原 owner callback；重复取消幂等，不吞 cleanup failure。
- **并发：** Turn 继续按 session 串行而非全局串行；同一 runner registry seal 后不可变。
- **热重载：** 新 Root 完整 seal 后才可发布；旧 Turn 用完旧 Root。`sessions` 等独占 writer 的
  publication 走 pause → drain → close → open → publish，不跨代共写。普通插件 publication 可以让
  旧 opaque operation 持有旧 lease 到 terminal，但 supervisor 拒绝新代 claim 同一 scope；这不是
  两条实现处理同一请求，也不是双写。
- **候选验证：** 只在隔离 workspace/recording adapter 下运行，不接生产流量、不发真实 Channel、
  不读取或写正式 Session。candidate sessions 使用全新临时 schema/programmatic Session；candidate 与
  stable 不同时处理同一正式请求。
- **进程崩溃：** 恢复只依据 Session/receipt/publication journal 等持久 owner；内存 snapshot 指针
  不能证明消息、进程或远程调用已回滚。

## 9. 无灰度的迁移顺序

每批只迁一个 owner。批次内允许短命 deprecated 标记，但正式调用者始终只有一条路径。

### M0 · 正式设计

- 本合同、0054、PLG-018、INDEX/NOW 对账。
- 一个 Terra xhigh reviewer 只审查：正交、原子、非特权、整链可走通；P0/P1 为零才接受。
- 仅文档 commit 并打开 Draft PR；不修改 runtime。

### M1 · 中性 snapshot 执行原子

- 增加 `RuntimeServicePort`、`RootTaskScope`、`OperationSupervisor` 和 kernel-private Root lookup；
  三者接口不增加 Agent/Turn/Session/Scheduler 等领域字段。
- fixture 证明 single-key/stable-only port、owning Root background acquire、跨代 opaque cancel、
  same-scope claim exclusion、terminal release、错误 task 继承和退休 Root fail-loud。
- 本批没有被替换的旧 owner，不提前标 deprecated；caller 先作为后续唯一切换的中性前置能力。

### M2 · Prompt 与 Session projection owner

- 建立普通 `system-prompt`、`session-projections` 插件，先迁已有普通 contribution。
- 迁 rollout fact、skills/hints 与仓库内 output metadata consumer；现有外部 pure-v3 typed event
  consumer 若合同已正交则原样保留，确需换公共合同的只记录后续迁移，不改源码。
- 唯一新 registry 生效后删除对应旧 Core default phase、mutable wrapper 与 metadata bridge；
  不把仍有外部 consumer 的稳定 typed event 误当兼容壳删除。

### M3 · Tools owner 与特殊工具退役

- 普通 `tools` 插件取得 registry、turn-local view、authorize/execute 和 typed outcome owner。
- Tool Search 只用 catalog/search/grant；message_push、media、mobile attention 只用 typed facts。
- 删除 `_META_TOOLS`、`requires_turn_search`、工具名分支、提示拼接和 Shell 名称 cleanup。

### M4 · Models owner 收口

- 复用现有普通 models plugin，把 session selection/frozen execution binding 的唯一入口迁入该 owner。
- 删除 AgentLoop 的 model metadata 读写和 bootstrap model branch；保留现有 provider/usage语义。

### M5 · Sessions 独占 writer

- 普通 `sessions` 插件创建 SessionManager 和全部窄 port；所有其他插件只注入端口。
- 用维护窗口式测试执行 pause/drain/close/open，证明正式路径任一时刻只有一个 SQLite writer。
- bootstrap、PluginManager 和工具不再持有 `_store`、任意 repository 或 SessionManager 私有引用。

### M6 · Agents owner 与所有 ingress

- 普通 `agents` 插件取得 ConversationRuntime/admission/cancel/terminal/active owner。
- 把 ConversationRuntime 中 process-wide task/lease/cancel 的中性机械部分迁到 M1 supervisor；
  `agents` 只保留领域状态机，并让 accepted/durable fact 使用同一 opaque operation key。
- passive、control、scheduler、wake、subagent 的仓库内入口一次切到 `AGENTS`；没有 runtime fallback。
- agent-loop 尚未迁移时，只允许一个明确 deprecated runner 注册旧算法，零其他 consumer。

### M7 · Agent-loop 与 conversation source

- 把直接 ReAct 算法作为普通 `agent-loop` 插件注册；把 durable handoff、command route、delivery/ACK
  组合放入普通 conversation plugin。
- 迁 streaming、interrupt、tool batch、provider retry、commit 观察点；保留算法而删除总 phase。
- 物理删除 deprecated runner、`AgentLoop`、`PassiveTurnPipeline`、旧 ConversationRuntime wiring、
  `SCOPED_TURNS` Core bridge 和 PassiveMessageWorker 私有业务链。

### M8 · 最终收口并停止

- Core/Bootstrap 搜索证明零 Agent/Tool/Session/feature 插件 ID 特判和零旧 consumer。
- 运行关键场景、全量测试、静态检查、项目 Gate；对最终 topology 和 write set 生成证据。
- 输出外部 typed-event/test consumer 的 exact repo/commit/符号清单；不为它们新增 Core 兼容层。
- Draft PR 保持等待维护者；不开始修改独立外部插件仓库。

顺序只能在证明依赖和风险更低时调整；任何调整都要先更新本合同并重新过 Concept Gate。

## 10. 每批 deprecated、Review 与删除协议

1. 在旧 owner 入口写静态注释：`DEPRECATED(Mx): no new consumers; remove in this batch after review`。
   不发运行时 warning，不新增 alias，不创建 `legacy_mode`。
2. 新 owner 成为唯一正式路径；旧代码只留给该批 reviewer 看差异，不接流量、不双写。
3. 运行该 owner 的最小关键测试和 deterministic recording 场景。
4. 同时启动两个互相独立的 Terra xhigh reviewer。两者都读取完整 batch diff、合同和相关 source，
   检查 owner、权限、失败路径、行为损失和兼容壳；任一 P0/P1 都阻断删除。
5. 修复 finding；涉及 owner/接口变化时让同一 reviewer 复审到 P0/P1 为零。
6. 物理删除 deprecated 文件、分支、配置、导出、测试替身和文档入口，运行 zero-consumer 查询。
7. 重新运行关键测试。删除 delta 若改变调用链或公共面，由两位 reviewer 快速复核最终 diff。
8. 形成一个语义连贯、可独立回滚的 commit，再进入下一 owner。

不把“以后再删”留到 PR 外。唯一例外是本合同明确排除的外部插件源码 consumer；Core public contract
会先保留到后续外部插件 PR，但不得保留旧 Core 私有实现。若外部 consumer 阻止删除公共面，M8
必须停止并报告，而不是加兼容壳。

## 11. 合格测试

只补能保护现实行为或非平凡边界的测试：

- snapshot service port 的窄 key、stable-only public policy、candidate capability identity、exact lease、
  task ownership、cancel 和 cleanup；
- reload-mid-Turn 后用原 operation key `/stop`，必须到达旧 owner，产生一次 terminal 并释放旧 lease；
- Scheduler/Wake/Subagent 在 reload-before-fire 与 fire-during-drain 下只取得 owning Root；retired
  admission 单次 settle/rearm，不重复 provider、Session commit 或 delivery；
- Root sealing 对缺 Service、重复 runner、循环依赖、重复 writer 的 fail-loud；
- sessions 单 writer、原子 commit、messages 只追加、seq、restart recovery；
- 同一固定场景在旧基线 artifact 与新代码上**依次**运行，比较 provider payload、tool trace、
  Session rows/write set、typed events、stream、delivery/ACK、attachment、error/cancel/interrupt；
- Tool Search grant 通过结构化 outcome 工作，改名 tool 后仍工作；普通工具不能越权 grant；
- feature plugin disabled/removed 后只失去自身 contribution，不触发 Core fallback；
- fault injection 覆盖 provider/tool/commit/delivery/cleanup 的真实失败边界；
- zero-consumer 和 forbidden-token Gate 证明没有旧入口、名称特判、双写或 compatibility flag。

不为常量映射、显然控制流、已删除功能的内部形状或覆盖率数字补测试。并发测试使用 barrier/event，
不用 sleep。比较测试不接正式 workspace 或真实不可逆 sender，也不称为 shadow。

## 12. Concept Gate

第一阶段 reviewer 只回答以下四项，不能扩成一般代码风格 review：

| 问题 | PASS 标准 |
|---|---|
| 足够正交？ | 每个事实只有一个 owner，变化轴之间没有强制联动或万能 context |
| 足够原子？ | Core atom 只有 composition/publication/lease/泛型 call；业务能力可直接组合且没有 feature-shaped Core API |
| 是非特权插件？ | 七项与其他插件使用同 loader、权限、lifecycle 和 failure；Core 无 ID/名字/fallback |
| 整条链走得通？ | passive/control/recursive source、完整 snapshot、commit、delivery、cancel、reload 和单 writer 都有闭合路径 |

P0/P1 任一非零即 `BLOCK`。2026-09-01 独立 Terra xhigh reviewer 完整复核当前代码与本合同；
在收窄 stable-only single-key port，并补齐 RootTaskScope、跨代 opaque cancel 和后台 exact Root
路径后，四项均 `PASS`，P0/P1 为零。该结论只批准设计，不能代替 M1～M8 的实现 review 与行为 Gate。

## 13. 交接边界

本 PR 最终只交付 Core 仓库内的通用内核、七个普通基础插件、仓库内置 conversation/feature
组合和旧私有链删除。独立安装的 QQ/Feishu/Citation/Meme 等外部源码若需要改用新公共合同，记录
exact repo、consumer、版本和阻塞点，等本 PR 停下后另开迁移。禁止直接修改 cache 伪造完成。
