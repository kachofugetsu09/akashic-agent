# 普通插件 Agent 骨架与被动链迁移合同

- 状态：accepted
- 日期：2026-09-01
- 决策：[0054 · Agent 由普通插件组成](../decisions/0054-agent-spine-is-ordinary-plugins.md)
- 取代范围：[React Core 与 Scheduler/Subagent 设计](react-core-scheduler-subagent.md) 中
  “React 实现属于 Core”的结构结论；既有 Turn、Session、Scheduler、Subagent 行为合同不变
- 实施分支：`codex/react-plugin-spine`
- 初始迁移基线：`f1f4560892ae92e96779ff89f848223afdcc9919`
- 本次 Concept Gate 基线：`b8f38583a51dee4cde9a689f1a5f49560d654bd2`
- Git worktree：`/mnt/data/coding/akasic-agent-worktrees/react-plugin-spine`
- 当前实现 head：`b8f38583a51dee4cde9a689f1a5f49560d654bd2`；M1 中性原子及 owner 修正已落地，M2 尚未开始
- 恢复引用：`backup/pre-react-plugin-spine-20260901-f1f45608`、`backup/pre-dsh-spec-rewrite-20260901`、
  `backup/pre-m1-retired-error-fix-20260901`

## 1. 结果、范围与停止条件

### 1.1 目标

把完整被动回复从“bootstrap 构造一条懂所有功能的 Core 私有链”迁成“普通插件通过公开 Service
拼出一条链”。迁移后，Core 不再构造或识别 AgentLoop、PassiveTurnPipeline、Tool Search、
Command、Shell、Compaction、Markdown memory 或任何业务插件；它只发布和租用完整 snapshot。

### 1.2 完成标准

- 最小 Agent 能力图全部通过同一 v3 loader、generation Root、Fiber、Effect 和 lease 运行；
  插件数量不作为设计约束。
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
  - 每个实现批次的关键行为和 write-set oracle
  - M0 一个 Terra xhigh concept review 和一个独立 name review
  - M1～M9 每批两个独立 Terra xhigh implementation review 和一个独立 name review
  - 最终 zero-consumer、全量 test 和 project Gate
rollback: 上一完整 commit、不可变 generation、执行前备份；不伪造外部效果回滚
worktree_writer: /root
external_revisions: [hua-home:f1f4560892ae92e96779ff89f848223afdcc9919]
schema_lineages: [sessions.db current schema unchanged]
```

### 1.4 外部插件真源

外部 consumer 只以 `hua-home` 的实际启用状态为准，不以开发机源码目录或 cache 猜测。
2026-09-01 的只读检查确认：live release 指向本合同基线
`f1f4560892ae92e96779ff89f848223afdcc9919`；boot log 记录 33 个 active generation；其中有 stable
GitHub artifact 的 16 个外部插件是：

`calendar`、`citation`、`emotion`、`feed`、`fitbit`、`github-watch`、`huayue-skills`、`meme`、
`observe`、`plugin_undo`、`proactive_feedback`、`setup_helper`、`shell_restore`、`shell_safety`、
`status_commands`、`steam`。

对这 16 个 stable artifact 的运行源码做零 consumer 查询，本批所有已删公开名均为零；
`github-watch/scripts/core_v3_gate.py` 的 `FakeConversationRuntime` 只属于离线 Gate，不是 runtime
consumer。后续每个 public API 删除都必须对当时 `hua-home` artifact 重新查询；本段不是永久豁免。

consumer 只决定迁移顺序，不决定设计好坏。每个外部接入点都要先按最终模型独立判断：

- `keep`：接入点本身同时满足 owner 明确、deeply immutable、权限窄和单一变化轴；consumer 可原样继续。
- `move`：能力要保留，但当前入口暴露可变总状态、metadata bag、phase 顺序或另一 owner 的实现细节；
  先建立新入口，最后迁 consumer 并删除旧入口。
- `remove`：能力和入口都没有独立 owner 或现实用途；consumer 一起删除。

“线上正在使用”不能把 `move` 或 `remove` 升级成 `keep`。反过来，“零 consumer”也不能证明一个
新接口值得存在。最终判断只看 owner、变化轴、权限和失败语义。

### 1.5 外部接入点迁移账本

以下判断基于 2026-09-01 `hua-home` stable artifact；每个后续阶段都要重新核对 live generation：

当前 exact artifact commit 是：Citation `a886c74c55c4ef400ecd81451eb84b0970b60869`、Meme
`c185ea7a3847d67a3c61ede9819d6a94636d69c1`、Emotion
`d828fd7ec97e027bc1ee4a39e5501a2cf25296a2`、GitHub Watch
`b9266ab3ca9932c074a6d91cf48ab69691bcf1ce`、Observe
`09214c23f287f659eee6280706208b9ba7d2ed13`、Proactive Feedback
`d9d90fd4d3027d444091fd6a38453c33f372b7ed`、Status Commands
`8d119e8cfa53bd91e4dd1e2d4dcf67edfe047cb4`。这些 commit 只固定本次审计输入；M9 前仍以当时
`hua-home` stable pointer 重查，不能把本清单当成永久事实。

| 当前接入点 | live consumer | 判断 | 最终入口 |
|---|---|---|---|
| `PROMPT_RENDER_EVENT`、可变 `PromptRenderCtx`、`PromptSectionRender` | Citation、Meme | `move` | `system-prompt` 接受普通 section contribution |
| `CONTEXT_PREPARED_EVENT`、`BeforeTurnCtx.extra_hints` | Emotion | `move` | system section 或普通 context Message；依是否需要进历史决定 |
| `AFTER_REASONING_PREPROCESS_EVENT`、`AFTER_REASONING_CLEANUP_EVENT`、可变 `AfterReasoningCtx` | Citation、Meme | `move` | 模型指引归 `system-prompt`，展示归 outbound view，插件事实归私有 ledger；无通用 reply 改写链 |
| `AFTER_TURN_COMMITTED`、`TurnCommitted` 总 payload | Emotion、GitHub Watch、Observe、Proactive Feedback | `move` | sessions 只发布小而 immutable 的 `TurnSaved`；诊断由原 owner 单独发布 |
| `is_context_frame` 与 provider dict 编码 | Status Commands | `move` | sessions 拥有 typed `MessageKind`，`SessionRead` 返回 typed `MessageView` |
| `persist_assistant_metadata["cited_memory_ids"]` | Citation | `move` | Citation 自己的 `MemoryIds` ledger；Core 不新增通用 data bag |
| Session metadata `skip_memory_retrieval` | GitHub Watch | `move` | 每次 Turn 显式传 `PromptUse`，不保存成 Session policy |
| Session metadata `source/repo/item` | GitHub Watch | `move` | github-watch 自己的 job ledger，不进入 Session metadata |
| `PromptRenderInput`、旧 phase frame/slot、`ConversationRuntime` fake | 仅仓库或外部离线 Gate | `remove` | 新 Service fixture；不保留 runtime alias |

`move` 项允许在 Core 阶段保留一个清楚标记的 live migration block，只为账本中的 exact external
consumer 服务；它不是目标设计，也不能新增 consumer。Core 内部必须先全部切到新入口。外部源码
迁移完成的同一收尾批次删除 block、旧公开类型、事件、导出和测试。全部 `move/remove` 清零前，
整个迁移不能宣称最终完成，Core Draft PR 也不能作为“干净终态”合并。

`TurnSaved` 的最终合同现在锁定为
`TurnSaved(session_key, turn_id, message_ids, reply_id, saved_at)`。所有字段及 `message_ids` tuple 都是
deeply immutable；事件只在 sessions 原子提交成功后发布。它不增加 channel、正文、reply、tool、
model、prompt、统计、展示或 `extra`。GitHub Watch 可直接使用 identity；Emotion 和 Proactive
Feedback 以 identity 调窄 `SessionRead` 读取已保存的 typed `TurnView`。

Observe 不再等待一只总事件，也不新增 `*Log` 总袋子。算法进行时：`model-input` 每次 build 只发
`InputSize(turn_id, call_id, try_number, tokens, quality, changed)`；每次 provider call 由 `models` 发一条 immutable `ModelUse`；每次
tool call 由 `tools` 发一条 immutable `ToolUse`；每次 ReAct step 由 `agent-loop` 发一条 immutable
`LoopStep`。每种 fact 只含本 owner 的标量、tuple 或 immutable value。最后 sessions commit 并发
`TurnSaved`，作为本轮 facts 已齐的 fence。Observe 以 `turn_id` 在自身 turn-local state 中 join，
并从 `SessionRead` 读取已保存正文；无 `TurnSaved` 的失败或取消 Turn 不伪装成已保存。

这些小事实的字段也在实现前锁定：
`InputSize(turn_id, call_id, try_number, tokens, quality, changed)`；
`ModelUse(turn_id, call_id, usage)`，其中 `usage` 复用 models 的 immutable `ModelUsage`；
`ToolUse(turn_id, call_id, name, args_json, result, status)`；
`LoopStep(turn_id, step, text, call_ids, final)`。`args_json` 是已验证参数的 immutable 编码，
`call_ids` 是 tuple。没有 dict、`extra`、跨 owner payload 或“以后可能用”的字段。

旧 metadata 三组键分别这样结束：

- Citation 的 `cited_memory_ids` 变成 Citation 私有的 immutable `MemoryIds`，不进入 Core public
  API。Citation 若引导模型生成引用，用 system section 或普通 Tool；若只改变 Channel 展示，用
  outbound view。不允许在模型返回后用通用 `ReplyEdit` 改写待持久正文。Citation 在 `TurnSaved`
  后以 `reply_id` 写自己的 plugin-data ledger，boot 时用窄 `SessionRead` 修复漏处理。M9 从旧
  assistant metadata 一次性导入历史 Citation rows；之后只读 Citation ledger，旧 key 零 consumer，不留
  dual read。Core 不认识 citation，也不新增通用 data bag。若精确行为对比证明“模型后改写且
  作为持久正文”不可替代，M9 必须停止并单独决定 Message 合同，不得因此复活 phase。
- `skip_memory_retrieval` 不再是 durable Session metadata。GitHub Watch 每次 start Turn 时传
  immutable `PromptUse`，只关闭 memory 拥有的 section；`system-prompt` 只按该 request 过滤本轮 section。
  其他 Turn 不继承这项 policy。
- `source/repo/item` 属于 github-watch job identity。GitHub Watch 在 Agent create/resume 前写自己的 durable
  job ledger，start 失败也由该 ledger 记录；运行时只传普通 session/turn identity。M9 从旧 Session
  metadata 一次性导入仍有效 job，之后插件只读自己的 ledger，sessions 不再认识 GitHub 字段。

上述 M9 导入遵守 messages 只追加边界：旧 `extra.cited_memory_ids` 和 GitHub Watch metadata 作为
不再解释的历史字节保留，不 UPDATE/DELETE 既有 message。这里的“旧 key 零 consumer”只指代码面
零读取，不授权清洗或改写历史行。

`MessageKind` 只表示普通 message 或 context message 这一条轴，不包含 role、内容编码或 delivery
状态。sessions 是唯一 decode owner；`SessionRead` 返回 typed `MessageView`。旧 DB marker 只在
sessions 的存储 adapter 内解码，外部插件和其他 Core 模块都不能 import `is_context_frame`。

## 2. 六岁小孩版

现在像一辆玩具火车：车头里同时焊死了电池、方向盘、喇叭、货箱、售票员和清洁刷。换一只
喇叭，也要拆车头。

目标不是必须凑成七块，而是每块只做一件事：

```text
┌──────────┐  保存故事      ┌──────────┐  选择大脑
│ sessions │               │ models   │
└──────────┘               └──────────┘

┌──────────┐  使用工具      ┌───────────────┐  只拼系统提示
│ tools    │               │ system-prompt  │
└──────────┘               └───────────────┘

┌─────────────┐  把完整故事装进大脑这次装得下的小包
│ model-input │
└─────────────┘

┌──────────┐  管“有哪些 Agent，请哪一种来工作”
│ agents   │
└────┬─────┘
     │ factory
     ▼
┌──────────┐  管“一次工作怎样开始、停止，以及怎样想”
│agent-loop│
└──────────┘

┌──────────────┐  可选：把已保存的故事折成可读小纸条
│ session-view │  它不给大脑拼历史，也不改故事
└──────────────┘
```

还有一个很小的门卫，但门卫不是特权 Agent 积木。门卫不认识故事、大脑或工具；它只给每项工作
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
- `agent/plugins/service_call.py:21-105`、`agent/plugin_composition/context.py:60-145` 和
  `agent/plugin_composition/tasks.py:18-220` 已分别落地中性 `ServiceCall`、`RootScope` 和 `TaskControl`。
- `agent/plugin_composition/context.py:236-311` 已提供 mount/inject/provide/require/effect，
  `agent/plugin_composition/context.py:350-397` 已提供 typed dispatch。
- `agent/plugins/manager.py:5451-5458` 已有 `snapshot.sealing`；不需要新建第二套 readiness 图。
- `session/manager.py:642-802` 已有原子 message commit、append 和 durable delivery 事务 owner；
  迁移 owner 不能改变这些 write set。

### 3.2 DSH 参考快照

本合同对照 DSH commit `dd6322d604e00eec1ba5e0c8541159906a21094a`。每一块都以当前源码为老师，
不把名字相似当成行为等价：

| 问题 | DSH 证据 | Akashic 结论 |
|---|---|---|
| 插件是否同权 | 产品部件全是插件，无特权 Core：`docs/architecture.md:9-13`；bundle row 顺序不表示加载顺序：`packages/bundle/base/cordis.patch.yml:12-13` | 内置只表示默认发行；同 loader、权限、failure 和 lifecycle |
| 默认装配有多少块 | `AgentLoop` 注入六个 Service：`packages/core/agent-loop/src/index.ts:351-354`；官方测试挂载这六个加 loop：`packages/core/agent-loop/tests/interception.spec.ts:31-40` | “七块”只是 DSH 当前默认装配事实，不是 Akashic 验收数字 |
| Session 拥有什么 | append/commit：`packages/core/session/src/index.ts:567-653`；model history：同文件 `:699-745` | `sessions` 独占权威历史与 model history 派生 |
| Session projection 是什么 | 纯 `init + apply` fold：`packages/session/session-projection/src/index.ts:34-85`；只消费 committed event：同文件 `:169-211` | `session-view` 若存在只是可选通用 fold，不是 model history owner |
| Compaction 怎样进入 | DSH 在窄 `agent/pre-step` 决策中检查压力并追加 surface replacement：`packages/compaction/compaction-basic/src/index.ts:127-225` | Akashic 不复制 before-step；以独立 `model-input` provider 保留现有 append-only message + compaction ledger 边界 |
| 谁驱动每次模型尝试 | DSH 的具体 Agent loop 创建请求、调用模型、处理错误并继续工具循环：`packages/core/agent-loop/src/agent.ts:341-438`；prepared call 钉住同一 adapter generation：`packages/llm/llm/src/index.ts:882-935` | `agent-loop` 仍驱动 attempt；`model-input` 只对每次 attempt 做 build/settle，不成为第二个 loop |
| Prompt 如何扩展 | section/context/tools/variable 独立注册：`packages/core/system-prompt/src/index.ts:424-524`；assemble：同文件 `:526-611` | 只保留有 owner 的 system section 注册；不复制可改全体 assembly 的逃生 hook |
| Tool Search 如何涌现 | 具体工具通过普通 registry 注册：`packages/core/tools/src/index.ts:1022-1053`；progressive disclosure 替换 scoped restriction：`docs/cookbook/extension-cookbook.md:100-114` | Tool Search 只是 `tools` scoped view 的普通 consumer；Core 无元工具特判 |
| Agent 和 loop 怎样分 | registry/factory：`packages/core/agent/src/index.ts:235-247,352-422`；具体 Agent 与完整 Turn/Step：`packages/core/agent-loop/src/index.ts:612-640`、`packages/core/agent-loop/src/agent.ts:254-438` | `agents` 只管合同和 factory；`agent-loop` 管具体 Agent 的整个生命周期 |
| 是否需要 before/after phase | 只有窄 `agent/pre-step`：`packages/core/agent/src/runtime-types.ts:55-63,226-238`；无 `agent/after-step` | 不创建替代 before/after 套件；当前 `before_step` 无生产 consumer，直接删除 |
| 模型后是否任意改 reply | assistant 结果直接追加为事实：`packages/core/agent-loop/src/agent.ts:410-427` | 不创建 `ReplyEdit`；按 system prompt、plugin ledger 或 outbound view 拆开 |
| 事实怎样观察 | DSH 区分 durable event、waterfall 和 notification：`docs/architecture.md:64-70`；Session observer 只在 commit 后收到事实：`packages/core/session/src/index.ts:63-74,567-653` | 能力 owner 只发自己的 immutable live fact；`sessions` 以 `TurnSaved` 做原子提交 fence；无总 Turn bag |

DSH 不是需要逐字复制的模板。Akashic 保留四项有证据的差异：SQLite 原子事务、完整 Root 的
exact snapshot lease、Channel delivery/ACK，以及不 UPDATE/DELETE 旧 message 的 compaction ledger + `model-input`。每项差异都只保留
现有 owner 与安全不变量，不引入特权 Agent 接口。

`model-input` 的 Akashic 证据是：`session/manager.py:409-462` 已从只追加 Session row 构造完整
history units；`agent/core/passive_turn.py:2698-2860` 证明每次 provider call 都必须 prepare/settle，
overflow 还会在同一 call 强制重建；`agent/core/passive_turn.py:2258-2266` 证明闭合 tool batch 是下一次
输入的必要状态；`agent/plugin_composition/request_projection.py:69-141` 与
`plugins/compaction/plugin.py:87-286` 证明“权威 history + turn-local progress → 有限 provider input →
usage/fact settle”是真实边界；`plugins/compaction/runtime.py:238-353` 已证明 checkpoint 使用独立 durable
ledger 与 provenance。现有 mutable binding、Core pass-through fallback 和特制 ServiceKey 不是被背书的
终态，而是 M2 要收窄并删除的迁移资产。

### 3.3 事实、推断与未知

| 类型 | 内容 |
|---|---|
| 已核对事实 | Context/Service/Fiber/Effect、Root sealing、snapshot lease、普通 models/compaction/markdown 插件已经存在 |
| 已核对事实 | `SCOPED_TURNS` 由 Core 从 ConversationRuntime 制造，是 bootstrap 环的关键，不是必须保留的业务 owner |
| 已核对事实 | 当前 passive 文件按工具名识别 `tool_search`、`message_push`，AgentLoop 按名称/类型识别 Shell |
| 设计推断 | 一个泛型 snapshot Service 调用边界足以让 snapshot 外入口进入普通 `agents` Service |
| 设计推断 | factory slot 让 `agents` 不依赖具体 `agent-loop`；loop 反向注册 factory，因而无环 |
| 实施中核对 | 每个旧 phase 的动态外部 consumer；先判 `keep/move/remove`，consumer 只决定删除顺序 |
| 实施中核对 | Mobile attention、Meme/Citation 和 attachment 的精确外部 payload；只迁 owner，不改协议字段 |

### 3.4 当前外部 consumer 风险

2026-09-01 对 `/mnt/data/coding/akashic-plugin` 与已安装 cache 的只读扫描未发现外部插件注入
`SCOPED_TURNS` 或 import `AgentLoop`/`PassiveTurnPipeline`；当前直接 consumer 都在本仓库的
Scheduler、Wake 与 Subagent，因此该 Core bridge 可以在 M6/M7 内完整替换。

外部 Citation、Meme、Observe、Emotion、Proactive Feedback 和 GitHub Watch 使用
`AFTER_REASONING_*` 或 `AFTER_TURN_COMMITTED` 等入口。“typed”与“已有 consumer”都不自动代表
正交：可变 `AfterReasoningCtx` 判为 `move`；当前 `TurnCommitted` 虽是 frozen dataclass，却混入
可变 list/dict、`extra` bag、工具 trace、模型用量和展示字段，也判为 `move`。Core 阶段必须
保持 live consumer 的可观察行为，但 Core 内部只能走新入口；旧入口以 1.5 的 migration block
存在。外部 Observe 测试仍 import 旧 phase frame，GitHub Watch 的跨仓 Gate 仍构造 fake
ConversationRuntime；它们都进入 exact 账本。M8 输出 repo/commit/符号清单并停下，M9 迁外部
源码后物理删除所有 `move/remove` public surface，再做跨仓最终组合 Gate。

### 3.5 直接复用与必须退役的资产

| 资产 | 处理 |
|---|---|
| Context / ServiceKey / Inject / Fiber / Effect / typed dispatch | 原样作为唯一 composition kernel，不另建容器或 hook bus |
| RuntimeSnapshot、Root sealing、stable/latest、lease、candidate closure | 原样作为 publication 真源，只补 `ServiceCall`、`RootScope`、`TaskControl` |
| `TOOL_CATALOG`、`PluginTools`、工具 snapshot freeze | 演进为 `tools` 插件的唯一 registry，不创建平行 ToolRegistry |
| 现有 `plugins/models` Services | 直接作为 `models` 基础插件，不复制 provider/model catalog |
| 现有 compaction/markdown-memory 普通插件 | 保留持久 owner；`PROVIDER_REQUEST_PROJECTION` 另行判为 `move`，不把特制 request gate 当终态 Service |
| SessionManager/SessionStore 的事务和恢复算法 | 行为与测试资产保留，真实实现迁入 `sessions` owner；不包旧 singleton |
| `PluginScopedTurns` 的 exact root、accepted handle、retired error 语义 | 领域中性部分迁入 `RootScope`/`TaskControl`；旧 `SCOPED_TURNS` key/bridge 最终删除 |
| existing ActivityHost/admission-drain 模式 | 用作 `TaskControl` 的实现证据，不复制 Agent 专用 publication plane |
| committed fact 的一次发布语义 | sessions 只发小 `TurnSaved`；system-prompt/model-input/tools/models/agent-loop 各发自己的窄 immutable fact |
| mutable phase ctx、metadata bag、编码 helper | 标成 `move`，Core 内部先停用；外部 consumer 迁完后删除 |
| bootstrap AgentLoop/SessionManager/ToolRegistry construction 与 manager Core-service manufacturing | deprecated 后退役；它们是待删除 owner，不是可长期复用 adapter |

## 4. 最终能力与唯一 owner

### 4.1 Core publication kernel

Core 只保留：

- 插件 artifact、generation 和完整 Root 的构建、验证、发布、丢弃与恢复；
- stable/latest 指针、exact lease、retire/drain 和 Effect cleanup；
- 绑定单一 `ServiceKey[T]` 的 `ServiceCall[T].call(action) -> R`；
- 每个 Fiber 平等取得的 `RootScope`，以及按 service key 隔离的
  `TaskControl` 与窄 `TaskStart`/`TaskCancel`/`TaskWait`；
- composition diagnostics、最小 workspace file grant 和外部 host 的通用资源开关。

kernel 在 bootstrap composition 时为外部 host 创建绑定一个 exact `ServiceKey` 和固定 lease source 的
`ServiceCall`；host 不取得任意 service lookup，插件也不能创建 `ServiceCall`。普通 host 的 lease
source 永远取得 stable；公开 `call(action)` 不接受 selector、snapshot ID、plugin ID 或 lease。
attached validation child 只使用 Core 根据父 Turn、candidate generation/source identity 铸造的一次性
exact lease，不能由 host 或插件选择 latest。`ServiceCall` 绑定当前 task，从 exact Root
`require(bound_key)`，完整等待 action，再解除绑定并释放。Service 缺失、Root/identity 不一致、
继承到错误 task 或 lease 已退休全部 fail-loud。它不解析 request，不创建 background task，也不
捕获领域错误。

### 4.2 最小普通插件图

| 插件 | 独占事实或变化轴 | 公开能力 | 明确不拥有 |
|---|---|---|---|
| `sessions` | Session/Message/Turn 的 SQLite 事实、事务与 model history 派生 | `SESSIONS`: read、history、append、save；附件/delivery 只留有已证明跨表事务的窄端口 | system prompt、模型、工具、Channel 发送、任意删除 |
| `models` | provider/model registry、冻结执行绑定与流式调用 | 复用 `MODEL_DRIVERS`、`CHAT_MODELS`、`EMBEDDINGS`、catalog/settings | Session metadata、模型选择 policy、system prompt、loop |
| `tools` | 工具定义、scoped view、调用与结算 | `TOOLS`: register、view、run；结构化 `ToolOutcome` | system prompt 文案、Session SQL、Tool Search 特制 grant/unlock |
| `system-prompt` | 有序 system section registry | `SYSTEM_PROMPT`: register、build | model history、provider 调用、记忆文件、任意 reply 改写 |
| `model-input` | 每次 provider attempt 的有限不可变输入与结算 | `MODEL_INPUT`: open、build、settle；一个 Root 恰有一个 basic 或 compaction provider | 权威 history、system section/tool registry、provider 调用、通用 middleware |
| `agents` | 公开 Agent 合同、live registry、source 归属和 factory slot | `AGENTS`: create、resume、get；register factory | 具体 inbox/Turn/Step、cancel/terminal 实现、ReAct、模型、工具 |
| `agent-loop` | 默认具体 Agent 的完整生命周期 | 向 `AGENTS` 注册默认 factory；拥有 inbox、Turn/Step、cancel/terminal 和 provider/tool loop | 持久 writer、发送、来源枚举、业务插件名 |
| `session-view`（可选） | 已提交 Session fact 的纯同步 fold | `SESSION_VIEW`: register、state、snapshot | model history、I/O、Session 回写、命令、发送 |

`model-input` 的 definition、provider 和 consumer 是一条完整 capability seam。中立 public API 只有：

```text
MODEL_INPUT.open(TurnInput)          ──► InputState
InputState.build(InputCall)          ──► ProviderInput
InputState.settle(CallResult)        ──► InputRetry
```

`TurnInput` 每个 Agent Turn 只创建一次，冻结 `session_key`、`turn_id`、Session 创建时间、`sessions`
派生的 immutable history units 和一只窄 ledger read grant。`InputCall` 每个 provider attempt 创建一次，冻结：

- `call_id`、1-based `call_number`、1-based `try_number`；
- `cause=normal|too_long`；
- 当前完整 Turn transcript、当前 system text 和当前 tool schemas；
- `ModelChoice`、context limit、max output 和 continuation；
- 之前已经 settle 的 immutable usage tuple。

`ProviderInput` 是 provider-ready content payload：最终 messages、tool schemas、max output、可继续使用的
continuation、`InputSize`、opaque `InputReceipt`，以及 build 中额外模型调用产生的 immutable usage。
stream callback、transport retry 和 auth 不进入 `model-input`，仍属于 `models`。每只 receipt 必须恰好一次以
`CallResult(receipt, status, usage)` settle；status 只能是 `done`、`too_long`、`failed` 或 `cancelled`。
`InputRetry` 只回答同一逻辑 call 是否可用 `cause=too_long, try_number=2` 再 build；禁止第三次尝试、换 provider
或 Core fallback。缺 provider、双 provider、序号倒退、receipt 错配或重复 settle 均 fail-loud。

basic provider 原样组合且永不要求 overflow retry。compaction provider 依自己的 durable ledger 投影，
可以在 `InputState` 内私有保存 ledger head、token meter、已闭合 tool batch 与待发布 fact。它从每次冻结的
完整 transcript 与上一只 receipt 的边界识别新增闭合 batch，而不是要求 loop 调用 compaction 专用方法；
成功 settle 记录 response usage，并在单次运行只发布一次待发布 fact。`too_long` settle 可允许第二次 build 强制压缩；
失败或取消不伪造已提交 checkpoint 回滚，下一次 `open` 从 ledger/receipt 重放并补发。若 build 改变
messages，返回的 continuation 必须为空。`agent-loop` 只 require `MODEL_INPUT` 并传 typed input/result，
不识别 compaction、不读写 provider 私有 state、不接受 mutable request binding 或 listener 列表。

`build` 与 `settle` 不是可分别注册的 before/after hook。一个 Root 只有一个 provider，同一只
`InputState` 同时实现两者，receipt 把一次 build 和一次 settle 配对；没有 listener order、任意 ctx、
跨 capability 改写或第二条控制流。普通返回路径必须 settle；进程崩溃来不及 settle 时，compaction
provider 按 `source_ref` 幂等补发 committed fact，不能宣称外部效果被回滚。

`sessions` 可以声明 `workspace_files=("sessions.db",)`，但只有它获得正式 writable grant。
candidate closure 中的 `sessions` 使用插件自己创建的全新临时 schema 和 programmatic Session，
不复制、读取或写正式 `sessions.db`；它是验证数据，不是第二名正式 writer。需要历史语义的回归由
测试把固定 fixture 恢复进一次性 workspace 后串行运行，不从 live DB 取样。

### 4.3 三个中性执行原子

| kernel atom | 只拥有 | 不拥有 |
|---|---|---|
| `ServiceCall[T]` | 构造时固定的 ServiceKey 与 lease source；一次完整 call | selector、request 解析、background task、领域 fallback |
| `RootScope` | owning Root identity、task/Effect cleanup、root-bound lease acquire | stable/latest 选择、领域 retry、跨 Root 重投 |
| `TaskControl` | opaque scope/task claim、exact lease、task、cancel callback、terminal release | Message/Turn/Session、Agent/factory、持久状态、错误解释、delivery |

`TaskStart.claim(scope_key, task_key, lease, run, cancel) -> TaskWait` 对整个进程原子；`lease` 是
`TaskLease`。
同一 opaque scope 跨 generation 只能有一个 active task。具体 Agent 负责把自己的 session/attempt
领域身份映射成稳定 opaque key，并负责何时允许 start/cancel/terminal；accepted receipt 与 durable
active-attempt fact 保存同一个 task key。`TaskControl` 只执行 claim、按该 key 通知原 owner 的
cancel callback 和最后释放。Control host 只获得 `TaskCancel`，不能枚举
task、读取结果、创建工作或取得 snapshot。新 ingress 要 interrupt 旧 attempt 时，先从
`SESSIONS` 窄 read Service 取得 durable active task key，不能按内存对象或 current stable 猜测。

这使新 stable 的 Control 能依已知 task key 请求取消旧 Root 的仍活 task，但旧具体 Agent 继续唯一负责
terminal/Session settle，并在最后释放旧 lease。这里没有内存状态搬家、两代共同写或特权 Agent
service。`ActivityHost`/generation lease 的现有 admission/drain 语义是实现资产；不得再创建一份
Agent 专用 publication 平面。

`RootScope` 由每个 Fiber 平等取得。`agent-loop` 创建的具体 Agent 绑定自己的 root scope；
它只复用同 Root 的 current lease，或向该 scope 取得 owning Root lease，遇到其他 Root
binding 直接失败。Scheduler/Wake 的 timer callback 因而可以直接调用同 Root 注入的 `AGENTS`；Root
已退休时原样得到 `RootRetired`，由 Scheduler/Wake 自己 settle/rearm，绝不 fallback
到 current stable。candidate Root 的普通 background scope 关闭，只有 Core 铸造的 attached
validation capability 能启动一次 candidate task。

### 4.4 无环注册

```text
foundation providers
  sessions ──► SESSIONS (includes model history)
  models ────► CHAT_MODELS ...
  tools ─────► TOOLS
  system prompt ► SYSTEM_PROMPT
  model input ──► MODEL_INPUT (exactly one provider)
  session view? ► SESSION_VIEW (only with proved fold consumers)

agents provides: AGENTS + empty factory slot

agent-loop injects: AGENTS, SESSIONS, CHAT_MODELS, TOOLS, SYSTEM_PROMPT,
                    MODEL_INPUT
agent-loop effect: register(default factory) ── cleanup unregisters

snapshot.sealing: exactly one default factory, every registry frozen
```

`agents` 不 inject `agent-loop`，因此没有 Service cycle。Root 未 seal 前不能取得正式 lease；
seal 后 factory slot 不再改变。热重载发布的是另一棵完整 Root，不原位替换 live factory。

### 4.5 代码与公共合同边界

骨架插件的实现最终位于各自 `plugins/<name>/` 包，或其独立安装 artifact 中。普通插件只能 import
版本化 public Plugin API、结构 DTO/Protocol/ServiceKey 与自身包；不得为了复用旧实现继续 import
`bootstrap.*`、`agent.looping.core`、`agent.core.passive_turn`、`PluginManager`、SessionManager 私有
store 或兄弟插件源码。

`Message`、`Turn`、`Session` 的稳定结构合同和 Service protocol 可以留在中立 public API 包；
它们不包含实现、全局 singleton、workspace root、任意 SQL 或 publication 控制。迁移旧算法时移动
真实 owner 的代码，而不是在新插件里包一层旧 Core class。Core/Bootstrap 只为边界 host import
公开 `ServiceKey` 来绑定窄 `ServiceCall`，不 import provider implementation。

## 5. 完整链怎样走

### 5.1 被动消息

```text
ordinary Channel plugin
  └─ ordinary conversation plugin
       ├─ explicit command? ──► COMMANDS ──► delivery settle
       └─ normal Message ─────► AGENTS create/resume
                                │ registry chooses one factory
                                ▼
                         ordinary agent-loop Agent
                                │ inbox/admission/Turn/cancel/terminal
                ┌───────────────┼───────────────┐
                ▼               ▼               ▼
          session history   system prompt   tools view + model limit
                └───────────────┬───────────────┘
                                ▼
                         model-input open
                                │
                      ┌─────────▼──────────┐
                      │ build each attempt │◄── too_long + InputRetry
                      └─────────┬──────────┘
                                │ ProviderInput + receipt
                                ▼
                         models call
                                │ done / too_long / failed / cancelled
                                ▼
                         model-input settle
                                │
                                ├── tool calls ──► tools run ──► next transcript
                                └── final assistant Message
                                ▼
                     sessions atomic commit
                                │ committed fact
                                ▼
                    conversation delivery + ACK
```

一次 Turn 的 exact lease 在具体 Agent 接受工作时原子 claim 到 `TaskControl`，直到 terminal 后释放。
Channel 回调提前返回时 lease 仍由 `TaskControl` 持有；取消只通知当前 attempt，旧 Root 的具体 Agent
继续完成 cleanup 和 terminal。新 generation 不能 claim 同一 session scope。

### 5.2 Control、Scheduler、Wake 与 Subagent

- Control host 只持有 bootstrap 为 `AGENTS` 绑定的 `ServiceCall`；它不直接 import AgentLoop，
  也不能借该 `ServiceCall` 查询其他 Service。
- 正常 `/stop` 可以由当前 Agent 读取 durable active task key；publication 暂停、没有 stable
  service lease 时，Control 从 accepted receipt/Control store 取得同一 key，只用 kernel 给它的窄
  `TaskCancel` 通知已接受的旧工作。
- Scheduler/Wake/Subagent 已在 Root 内时直接 inject `AGENTS`；它们通过 registry 创建或恢复同 Root
  的具体 Agent。该 Agent 的 `RootScope` 保证 timer/后台 callback 只能取得 owning Root。Root 已退休就收到
  `RootRetired` 并由来源 settle/rearm，不得改投 current stable。各自 gate、spawn、
  持久状态和 delivery 仍由原插件拥有。
- 来源只构造普通 Message/Agent request，不复制模型、工具、system prompt、Session commit 或 cancel loop。
- 不适用的 feature plugin 没有 contribution；不存在“先运行 passive hook 再 early return”。

### 5.3 snapshot 本身包住完整 ReAct Turn

这是需要保留的安全性质，不是需要特权插件的理由：

```text
outside snapshot             inside one exact snapshot
─────────────────┬────────────────────────────────────────────
AGENTS ServiceCall │ require(bound key)
acquire lease ────┼─► agents ─► factory ─► concrete Agent
                   │                          ├─ model/tools/session
                   │                          └─ TaskControl owns opaque task + lease
wait result ◄─────┼────────────────────────────────────────────
release lease ────┘
```

“谁保管 lease/task/cancel callback”与“谁解释 Turn 并实现 ReAct”是两条正交轴。前者属于领域中性
`TaskControl`，后者属于普通 `agents`/`agent-loop`。把二者写进一个 privileged plugin
反而重新制造 bootstrap cycle。

DSH 也把 live Agent 停止/清理与 factory registry 放在普通 effect 中
（`packages/core/agent/src/index.ts:149-204`，`packages/core/agent-loop/src/index.ts:560-583`）。Akashic 的更强
保证是当前 `RuntimeSnapshot` 已冻结整棵 Root 并用 exact lease 等待全部工作退出
（`agent/plugins/snapshot.py:76-114,876-907`）。这是 publication/lifetime 差异，不是 Agent 特权。

## 6. 当前特殊功能清单与目标 owner

| 当前特殊点 | 当前位置 | 目标组合 | Core 新增专用原子？ |
|---|---|---|---|
| command 在模型前短路 | `AgentLoop._process`、`PassiveTurnPipeline.run_command` | conversation source 注入普通 `COMMANDS`，识别后不创建 Agent Turn | 否 |
| plugin rollout fact 塞入下一轮 Prompt | `AgentLoop._process` metadata | rollout 插件向 `SYSTEM_PROMPT` 提供一次性 section；事实文件由其声明 | 否 |
| session 模型选择 | `AgentLoop._resolve_model_selection` | 具体 Agent 通过 `SESSIONS` 读取已保存选择，将 `ModelChoice` 显式传给 `models`；`models` 只校验、解析、冻结与调用 | 否 |
| Shell 按工具名和类 cleanup | `AgentLoop._cleanup_shell_owner` | Shell 插件消费具体 Agent 的 terminal fact，并清理自己拥有的 execution | 否 |
| Tool Search enable、schema cap、LRU、名称解锁 | `DefaultReasoner` 多处分支、ToolRegistry meta set | Tool Search 普通插件注册普通 tool；只替换 `TOOLS` 的 scoped view/restriction | 否 |
| 未解锁工具的提示文字 | `DefaultReasoner` | Tool Search 自己的 tool outcome 或 system section 说明可用工具；`tools` 不知道“解锁” | 否 |
| `message_push` 媒体抽取 | tool loop 按名称收集 | 普通工具返回 `ToolOutcome` 的 typed durable items/delivery facts；delivery owner 消费 | 否 |
| `mobile_attention` | Reasoner/Turn result 固定字段 | Mobile output projection 插件消费 typed tool/turn fact并保持现有协议字段 | 否 |
| Meme/Citation response decoration | after-reasoning/after-turn consumers | 模型指引归 `SYSTEM_PROMPT`，显示归 outbound view，事实归私有 ledger；无 reply 改写 hook | 否 |
| Skills、memory、hints | before-reasoning phase | 普通 system section、context Message 或 tool contribution；required Service 显式 inject | 否 |
| Compaction request gate | `PROVIDER_REQUEST_PROJECTION` | 能力与 ledger 保留，当前 mutable request binding 判为 `move`；M2 必须在不引入 before-step 的前提下收窄为独立 model-input 边界 | 否 |
| Markdown MEMORY/SELF 写入 | committed checkpoint 后 | 已有普通 markdown-memory 插件 | 否 |
| streaming、thinking、tool progress | AgentLoop sink + EventBus | 具体 Agent 发算法事实；Observe/Channel 的可选 session-view 消费 | 否 |
| Session commit 与 outbound 混在 after-turn | PassiveTurnPipeline | `sessions` 先原子 commit；conversation/Channel 后 delivery/ACK | 否 |
| 六组可任意改写总状态的 phase | `agent/lifecycle/phases/**` | 删除；收敛到 owner 明确的 section/model request/tool/fact/view | 否 |
| provider retry、max iteration、tool batch、continuation | `DefaultReasoner` | `agent-loop` 内部直接算法，不拆成 feature plugins | 不适用 |
| attempt admission、interrupt、cancel、terminal | `ConversationRuntime` | 默认具体 Agent（`agent-loop`）唯一 owner；`agents` 只管公开合同与 factory | 否 |
| durable inbound handoff 与 ACK 顺序 | `PassiveMessageWorker` | ordinary conversation plugin，持久写只请求 `SESSIONS` 窄 Service | 否 |

禁止用 `TURN_EFFECTS`、万能 middleware、任意 mutable context 或一个“passive hooks”总 Service 把这些
重新装进一只袋子。每个 public seam 必须指向表中已有 owner 与一种明确变化轴。

## 7. 持久状态、外部效果与恢复

| 对象 | 正常增加 | 可原位更新/逻辑终态 | 物理减少 | 唯一 owner 与恢复证据 |
|---|---|---|---|---|
| `sessions.db/messages` | completed transcript 原子 INSERT | 不更新正文 | 仅 SES-003 显式用户撤销/删除 | sessions；DB backup、row/seq/write-set diff |
| `sessions` metadata / `turns` | admission、attempt、terminal 写入 | 仅既有状态机和白名单 metadata | 仅既有管理协议 | sessions；turn identity、terminal、restart recovery |
| attachments/compaction/delivery rows | 既有事务增加 | 按各自 prepare/commit/settle 状态机 | 只按现行独立合同 | sessions 窄 Service；digest、receipt、prepare fence |
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

- **缺依赖：** required Service、default factory、`MODEL_INPUT` provider 或 exclusive writer 缺失时
  Root sealing 失败，stable 不变。
- **普通错误：** provider、tool、Prompt contribution、Session commit 和 delivery 保持现有错误分类；
  只有拥有恢复动作的边界转换错误。
- **取消：** 当前 attempt 收到取消；具体 Agent 完成工具/外部效果既有 settle，只提交一次
  terminal，`TaskControl` 最后移除 opaque record 并释放 lease。reload 后 cancel 仍调用旧
  record 保存的原 owner callback；重复取消幂等，不吞 cleanup failure。
- **并发：** Turn 继续按 session 串行而非全局串行；factory slot 和 provider registry seal 后不可变。
- **热重载：** 新 Root 完整 seal 后才可发布；旧 Turn 用完旧 Root。`sessions` 等独占 writer 的
  publication 走 pause → drain → close → open → publish，不跨代共写。普通插件 publication 可以让
  旧 opaque task 持有旧 lease 到 terminal，但 `TaskControl` 拒绝新代 claim 同一 scope；这不是
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
- 一个 Terra xhigh reviewer 只审查：DSH 忠实度、正交、原子、非特权、整链可走通；P0/P1 为零才接受。
- 另一个 Terra xhigh name reviewer 只审查公开名称是否最多两个简单英语单词，并给出 `NAME PASS`。
- 仅文档 commit 并打开 Draft PR；不修改 runtime。

### M1 · 中性 snapshot 执行原子

- 增加 `ServiceCall`、`RootScope`、`TaskControl` 和 private lease source；
  三者接口不增加 Agent/Turn/Session/Scheduler 等领域字段。
- fixture 证明 single-key/stable-only `ServiceCall`、owning Root background acquire、跨代 opaque cancel、
  same-scope claim exclusion、terminal release、错误 task 继承和退休 Root fail-loud。
- 本批没有被替换的旧 owner，不提前标 deprecated；caller 先作为后续唯一切换的中性前置能力。

### M2 · System prompt 与 Model input

- 建立普通 `system-prompt` plugin，先迁已有普通 section contribution；动态且需要进入历史的内容改成
  context Message，不塞进可变 prompt ctx。
- 建立 `MODEL_INPUT` 中立合同和两个二选一普通 provider：每 Turn 一次 `open`，每 provider attempt
  一次 `build` 与一次 `settle`。basic 原样组合；compaction 复用当前 ledger/provenance/recovery、
  tool batch、usage 和 overflow retry 算法。Root 必须恰有一个 provider；`agent-loop` 只按
  `InputRetry` 决定同一 call 的第二次 attempt，不识别 compaction。
- 将 mutable `ProviderRequestBinding`、pass-through Core fallback 和 `PROVIDER_REQUEST_PROJECTION` 标成 deprecated；
  唯一新 provider 生效后同批删除。不新增 before-step、reply edit 或通用 middleware。
- 迁 rollout fact、skills/hints 与仓库内 output metadata consumer；按 1.5 判断外部入口。当前总 payload、
  mutable ctx 与 metadata bag 均标成 `move`。可选 `session-view` 不在本批预建；只有后续 exact fold
  consumer 证明需要时才增加。

### M3 · Tools owner 与特殊工具退役

- 普通 `tools` 插件取得 registry、turn-local view、authorize/execute 和 typed outcome owner。
- Tool Search 只用 catalog/search/grant；message_push、media、mobile attention 只用 typed facts。
- 删除 `_META_TOOLS`、`requires_turn_search`、工具名分支、提示拼接和 Shell 名称 cleanup。

### M4 · Models owner 收口

- 复用现有普通 models plugin，保留 provider/model registry、冻结 binding 和 call/stream 的唯一 owner。
- 具体 Agent 从 `SESSIONS` 读已保存 selection，以 `ModelChoice` 显式传入；`models` 不读写 Session metadata。
- 删除 AgentLoop 的 model metadata 分支和 bootstrap model branch；保留现有 provider/usage 语义。

### M5 · Sessions 独占 writer

- 普通 `sessions` 插件创建 SessionManager 和全部窄 Service；所有其他插件只注入 Service。
- 用维护窗口式测试执行 pause/drain/close/open，证明正式路径任一时刻只有一个 SQLite writer。
- bootstrap、PluginManager 和工具不再持有 `_store`、任意 repository 或 SessionManager 私有引用。

### M6 · Agents registry 与所有 ingress

- 普通 `agents` 插件只取得 public Agent contract、live registry、source 归属和 factory slot owner。
- 把 ConversationRuntime 中 process-wide task/lease/cancel 的中性机械部分迁到 M1 `TaskControl`；具体
  admission/cancel/terminal 暂时仍属于待迁的旧 Agent 实现，不塞进 registry。
- passive、control、scheduler、wake、subagent 的仓库内入口一次切到 `AGENTS` 的 create/resume/get；
  没有 runtime fallback。
- agent-loop 尚未迁移时，只允许一个明确 deprecated factory 注册旧具体 Agent，零其他 consumer。

### M7 · Agent-loop 与 conversation source

- 把 inbox、admission、Turn/Step、interrupt/cancel/terminal 和直接 ReAct 一起迁入普通 `agent-loop`，作为完整具体
  Agent factory 注册；不在 `agents` 与 `agent-loop` 各留半个 driver。把 durable handoff、command route、delivery/ACK
  组合放入普通 conversation plugin。
- 迁 streaming、interrupt、tool batch、provider retry、commit 观察点；保留算法而删除总 phase。
- 物理删除 deprecated factory、`AgentLoop`、`PassiveTurnPipeline`、旧 ConversationRuntime wiring、
  `SCOPED_TURNS` Core bridge 和 PassiveMessageWorker 私有业务链。

### M8 · Core 收口并停止

- Core/Bootstrap 搜索证明零 Agent/Tool/Session/feature 插件 ID 特判和零旧 consumer。
- 运行关键场景、全量测试、静态检查、项目 Gate；对最终 topology 和 write set 生成证据。
- 输出所有外部 `move/remove` consumer 的 exact repo/commit/符号清单；列出仍在工作的 migration
  block，不把它们误报成干净设计。
- Draft PR 保持等待维护者；不开始修改独立外部插件仓库。

### M9 · 外部插件收尾

- 为账本中每个 `move` consumer 修改真实插件源码并走正式安装链；不编辑 cache。
- 在同一收尾批次删除 Core migration block、旧 public type/event/export 和外部离线 fake。
- 对当时 `hua-home` active generation 做 zero-consumer 查询，重装插件并跑跨仓组合 Gate。
- 只有 1.5 中全部 `move/remove` 清零，且没有 alias、adapter、fallback、双路或旧名字，才宣称
  整个被动链路迁移完成。

顺序只能在证明依赖和风险更低时调整；任何调整都要先更新本合同并重新过 Concept Gate。

## 10. 每个实现批次的 deprecated、Review 与删除协议

1. 在旧 owner 入口写静态注释：`DEPRECATED(Mx): no new consumers; remove in this batch after review`。
   不发运行时 warning，不新增 alias，不创建 `legacy_mode`。
2. 新 owner 成为唯一正式路径；旧代码只留给该批 reviewer 看差异，不接流量、不双写。
3. 运行该 owner 的最小关键测试和 deterministic recording 场景。
4. 同时启动两个互相独立的 Terra xhigh reviewer。两者都读取完整 batch diff、合同和相关 source，
   检查 owner、权限、失败路径、行为损失和兼容壳；任一 P0/P1 都阻断删除。另开一个独立 Terra
   xhigh name reviewer，只检查新增和改名的公开词：每个名字最多两个简单英语单词，六年级学生能懂。
5. 修复 finding；涉及 owner/接口变化时让同一 reviewer 复审到 P0/P1 为零，name reviewer 复审到
   `NAME PASS`。
6. 物理删除 deprecated 文件、分支、配置、导出、测试替身和文档入口，运行 zero-consumer 查询。
7. 重新运行关键测试。删除 delta 若改变调用链或公共面，由两位 reviewer 和 name reviewer 快速复核最终 diff。
8. 形成一个语义连贯、可独立回滚的 commit，再进入下一 owner。

不把未知的“以后再删”留到 PR 外。唯一暂存项是 1.5 已列出 exact consumer、目标入口和删除阶段的
external migration block；它必须标成 `DEPRECATED(EXTERNAL): no new consumers; remove in M9`，
Core 内部不得再调用。若出现未入账 external consumer，M8 必须停止并更新账本，而不是把旧入口
解释成长期 public contract。M9 完成后不允许留下任何 migration block 或兼容壳。

## 11. 合格测试

只补能保护现实行为或非平凡边界的测试：

- `ServiceCall` 的窄 key、stable-only public policy、candidate capability identity、exact lease、
  task ownership、cancel 和 cleanup；
- reload-mid-Turn 后用原 task key `/stop`，必须到达旧 owner，产生一次 terminal 并释放旧 lease；
- Scheduler/Wake/Subagent 在 reload-before-fire 与 fire-during-drain 下只取得 owning Root；retired
  admission 单次 settle/rearm，不重复 provider、Session commit 或 delivery；
- Root sealing 对缺 Service、重复 factory、重复 `MODEL_INPUT` provider、循环依赖、重复 writer 的 fail-loud；
- sessions 单 writer、原子 commit、messages 只追加、seq、restart recovery；
- 同一固定场景在旧基线 artifact 与新代码上**依次**运行，比较 provider payload、tool trace、
  Session rows/write set、typed events、stream、delivery/ACK、attachment、error/cancel/interrupt；
- Tool Search 只通过 scoped tool view 工作，改名 tool 后仍工作；Core 和 `tools` 不认识 search/grant/unlock；
- basic/compaction `MODEL_INPUT` provider 用同一固定 history 得到已批准的不同 projection；无 provider 和双 provider 都
  fail-loud，没有 Core pass-through fallback；
- fixture 覆盖首 call、tool batch 后 call、空回复/结构化终态的后续 call、context overflow 的同 call
  第二次 attempt、done/failed/cancelled settle、usage 计量、checkpoint fact 单次发布和 crash 后 receipt
  补发；每只 `InputReceipt` 只能 settle 一次，禁止第三次 attempt；
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
| 是非特权插件？ | 骨架能力与其他插件使用同 loader、权限、lifecycle 和 failure；Core 无 ID/名字/fallback |
| 整条链走得通？ | passive/control/recursive source、完整 snapshot、commit、delivery、cancel、reload 和单 writer 都有闭合路径 |

P0/P1 任一非零即 `BLOCK`。旧版合同因把 `session-view` 当成 model history owner、把 `agents` 与
`agent-loop` 分成两个半 driver，以及新增 `ReplyEdit`，已被 DSH 对照复核判为 `BLOCK`。当前版本已在
`b8f38583a51dee4cde9a689f1a5f49560d654bd2` 基线上取得 `CONCEPT PASS`（P0/P1/P2 全零）与
`NAME PASS`，可以开始 M2。该结论只批准设计，不能代替 M2～M8 的实现 review 与行为 Gate。

## 13. 交接边界

本 Core PR 交付通用内核、最小普通 Agent 能力图、仓库内置 conversation/feature 组合和旧私有链删除，
并保留 1.5 中可枚举、不可新增 consumer 的 migration block。它是 M9 前的停靠点，不是最终架构。
上面列出的 `hua-home` 外部插件记录 exact repo、consumer、版本和阻塞点；Core 阶段停下后另开源码
迁移。M9 删除 migration block 后才完成整体交付。禁止直接修改 cache 伪造完成。
