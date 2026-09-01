# 0054 · Agent 由普通插件组成

- 状态：accepted / implementing
- 日期：2026-09-01
- 关联条款：STA-001～STA-003、CAP-001～CAP-002、RUN-001～RUN-012、OUT-001～OUT-005、PLG-001～PLG-018、TST-001～TST-006
- supersedes：0039 中“React 原子能力留在 Core”的 owner 结论；0036 中“真实第 4 名 consumer 出现前不增加通用 publication participant 协议”的暂缓结论
- superseded by：无

## 背景

当前被动回复由 bootstrap 先构造 `SessionManager`、`ToolRegistry` 和 `AgentLoop`，再构造
`ConversationRuntime`，最后加载普通插件。`AgentLoop._process()` 和 `PassiveTurnPipeline` 因此同时
拥有 ReAct、Session 提交、模型选择、Prompt phase、`tool_search`、`message_push`、命令短路、rollout 提示
和 Shell cleanup 等不同变化轴。普通插件虽已拥有 Context、Service、Fiber、Effect 和 exact lease，
完整被动链仍要先经过一条 Core 私有控制流。

DeepSeek Harness（DSH）以 commit `dd6322d604e00eec1ba5e0c8541159906a21094a` 为本决策的参考快照。
它的原则是“所有产品部件都是插件，没有特权 Core”（`docs/architecture.md:9-13`）。默认
`AgentLoop` 注入 `agents`、`sessions`、`llm`、`tools`、`systemPrompt` 和 `sessionProjections`，加上自己
恰好是当前默认装配的七个 Service（`packages/core/agent-loop/src/index.ts:351-354`）。但数量不是
架构律：每块是否存在，只能由它是否拥有独立事实、不变量、控制流或真实边界决定。

DSH 的 `systemPrompt` 内部已经把 `PromptSection` 与 `PromptContext` 定义成不同输入、存进不同 registry，
并在 assemble 后保持不同输出（`packages/core/system-prompt/src/index.ts:52-84,354-385,425-476,536-610`）。
DSH 随后把 context 投影成 durable user Message（`packages/core/agent-loop/src/agent.ts:234-251,288-296`），
因为它要求所有 model-visible 内容可从 Session log 重建（`docs/architecture.md:103-107`）。Akashic 的
临时检索结果不得反向变成权威 Session 事实，因此不能直接复制这段持久语义，也不能把两个 registry
为了凑数重新合并成一个 prompt 总袋子。

DSH 还给出两个必须保留的边界：`Session` 自己从 append-only log 派生模型历史
（`packages/core/session/src/index.ts:567-653,699-745`）；`session-projection` 只对已提交事件做可回放 fold
（`packages/session/session-projection/src/index.ts:34-85,169-211`）。两者不能变成重复的 prompt history owner。

DSH 的 skills 也不在 AgentLoop 必需注入中：`dsh-skill` 只拥有 registry，`dsh-skill-filesystem` 是普通
source provider，`dsh-tool-skill` 才组合 catalog 与 tool；不挂最后一只产品插件时 Agent 仍能工作
（`packages/skill/skill/src/index.ts:285-298,347-392,464-500`；
`packages/skill/skill-filesystem/src/index.ts:129-146`；
`packages/skill/tool-skill/src/index.ts:127-161,163-251`）。Akashic 也不能为了默认启用 skills 把它塞成
agent-loop 的 required spine dependency。

## 决定

Akashic 的 Agent 骨架是一张由同一 v3 loader 装载的最小无环能力图。最小 spine 包含：

1. `sessions`：唯一拥有 Session、Message、Turn 权威事实、事务与模型历史派生；
2. `models`：唯一拥有 provider/model registry、执行绑定与流式调用；选择作为输入传入，不读写
   Session metadata；
3. `tools`：唯一拥有工具注册、作用域可见集合、运行与结构化结果；
4. `system-prompt`：唯一拥有有序 system section 注册与组装，不拥有 prompt history、Session 或模型调用；
5. `context-input`：唯一拥有 Root-local context part 集合，并在每个 Turn 把冻结事实变成不可变
   附加 Message；不改写 prompt history、current Message、system text、tools 或其他 part；
6. `provider-input`：为一次 Agent Turn 打开一只私有输入状态，并在**每次**模型尝试前把 prompt history、
   PromptText、ContextMessage、当前 Turn transcript、tool schemas 和 model limit 变成不可变 provider input；
   basic provider 直接组合，compaction provider 按已提交 ledger 投影，Root 只允许一个 provider；
7. `reply-output`：唯一把最终 provider raw reply 解码成要持久和发送的同一份 FinalReply；所有 ReplyPart
   只在同一原文声明不重叠 span，没有 listener order 或任意 reply chain；
8. `agents`：唯一拥有公开 Agent 合同、live registry、`TurnSource` 校验/冻结和 factory slot；
9. `agent-loop`：提供默认具体 Agent，拥有 inbox、Turn/Step/ReAct、取消与 terminal，并向 `agents` 注册
   factory。

`agent-loop` 显式 inject SESSIONS、models、TOOLS、SYSTEM_PROMPT、CONTEXT_INPUT、PROVIDER_INPUT、
REPLY_OUTPUT 与 SAVE_NOTICE；正常 save fence 与 boot ready 都不得从 sessions implementation 偷取
SAVE_NOTICE，也不得用 Service lookup 补漏。

这九块是最小 Agent spine，不是完整默认产品清单。默认产品另挂 `host-check`、`skills`、`skill-files`
与无状态 `skill-use`：host-check 只给 bin/env name 的 available/missing partition；skills 只拥有 Root-local
provider set 与 SkillView；skill-files 只拥有受保护 agent/drift roots、现有 link projection/journal 和 typed
catalog check；skill-use 把 SKILLS 分别贡献给 system-prompt、context-input 与 tools。agent-loop、PromptCall
与 ContextCall 都不携带 skill 状态。当前 drift roots 没有 runtime list/get consumer，只保留 check/projection，
不预建第二只 registry。

Root seal 只冻结 SkillProvider set，SkillView 仍在原有 lookup 边界读取 workspace、frontmatter 与
host availability；installed plugin body 只读 generation-private copy。SkillCall 只有 turn identity 与
cancellation，不含 RootRef、path 或 mutable bag。load-skill 的 plugin/catalog provenance 来自 SkillBody，
runtime snapshot identity 由拥有 exact scope 的 tool boundary stamp，skill provider 不 ambient 读取
RuntimeSnapshot。现有 normal/drift 声明在外部插件迁移前只能经过 exact `DEPRECATED(EXTERNAL)` bridge
注册 SKILL_FILES；最终外部源码直接注册，旧字段和 Core parser 清零。

`SYSTEM_PROMPT.add(ctx, PromptSection)` 注册一项可逆 Effect；
`await SYSTEM_PROMPT.build(PromptCall, skip_sections)` 每个 Turn 恰好调用一次并返回 immutable
`PromptSet(items, size)`；items 是 PromptText tuple，size 是同一次 build 的 PromptSize。agent-loop 直接保留
两者，system-prompt 再 observe 同一只 size，observer failure 不改变返回；
`SYSTEM_PROMPT.render(tuple[PromptText, ...]) -> str` 只按既定顺序纯 join。
`PromptSection` 只公开唯一 `name`、整数展示 `order`、`DropLevel drop` 和
`async build(PromptCall) -> str | None`。`PromptCall` 只冻结
session/turn identity、channel 和 chat identity，不携带 skip sections、workspace grant、prompt history、Message、tools、
provider request、mutable bag 或 rewrite callback。每个 section 的窄 grant/ledger 只能由注册它的普通插件
自己注入并闭包持有；任一 section 失败或取消都让整次 build 失败，不交付 partial PromptText tuple。
`skip_sections` 只有 service 看见；service 校验名单语法，在完整 registry 验名后、调用 section build 前过滤
matched name，unknown name 是“已经没有该 section”的幂等 no-op；被跳过的 section 不得读文件或 ledger。
PromptSection 看不见这份名单。
完整 tuple 随 Turn 冻结；provider retry 和同 Turn 的后续 tool call 不重新 build。`PromptText` 保留 name、
order、drop 和 text。`provider-input` 只选择保留哪些 PromptText，再调用同一纯 render；不自己复制 join 规则。

`CONTEXT_INPUT.add(ctx, ContextPart)` 注册一项可逆 Effect；
`await CONTEXT_INPUT.build(ContextCall, context_texts, skip_parts)` 每个 Turn 恰好调用一次，返回按
`(order, name)` 排序的 `tuple[ContextMessage, ...]`。`ContextCall` 只冻结 session/turn identity、current Message、
immutable runtime history view、channel、chat identity 与 message time；没有 `context_texts`、skip parts、
skills、mutable list/dict、slot、metadata bag、`next()`、
waterfall 或任意 request rewrite。每个 `ContextPart` 只公开唯一 `name` 与
整数展示 `order`、`DropLevel drop` 和 `async build(ContextCall) -> str | None`；每个 `ContextText` 只含
name、order、text 与 drop。
`context-input` 合并 `context_texts` 与 registry part，以各自 name 作为 provider-visible source，固定标记
`trust=derived`、`kind=context`、`role=user`，再构造与 current user 分开的
immutable `ContextMessage`。ContextText/ContextPart 之间互不可见。`None` 只表示该 part 本轮不适用；依赖缺失、I/O 失败、非法 text
或取消必须在模型调用前 fail-loud；整次 build 失败且不能交付部分输入。registry 在 Root sealing 冻结；
重复 registry name 使 candidate 失败，stable 不变；重复 ContextText name 或 ContextText 与 registry name 冲突
在模型调用前失败。skip parts 校验语法，并在完整 ContextText/registry 验名后、调用 ContextPart build 前
过滤 matched name；unknown name 是幂等 no-op，被跳过的 part 不得运行 query。过滤不能掩盖冲突。

这不是 `before-step` 换名。`context-input` 不决定 Step 是否进入，不接触 Session writer，也不包围一次
调用。Akasha retrieval 与 active skills 各自注册 part；source hints 直接使用 ContextText；Tool Search 的 schema/
restriction 仍归 `tools`，稳定工具说明才是 system section，tool 调用结果仍归 `ToolOutcome`。若某来源需要
持久事实，由该来源自己的 ledger/event 拥有，不能借 `context-input` 写 Session。

`ToolOutcome` 也不是新的 fact bag：它只含 call identity、done/failed/cancelled 和模型可见的 immutable
text/MediaItem content。`TOOLS.run` 返回 closed `ToolUse(turn_id, name, args_json, outcome)`；outcome 是同一
ToolOutcome。agent-loop 直接读取并累积这个返回值给 ReplyCall，tools 的 typed observer 收到同一对象但不
拥有这条业务通路。Tool 自己通过窄 Service 完成发送或 mobile output 等外部效果；agent-loop 不按
tool name 或 fact type 解释结果。`message_push` 复用现有 committed Channel dispatcher，在 execute 内完成
独立 outbound Turn 并返回普通 receipt content；mobile confirmation 由产生它的 Tool 直接请求 Mobile
output projection。通用 ToolInput 不带 send mode 或 commit role；这类 policy 由具体 Tool 注册时 inject 的
窄 sender 闭包固定。

Active-skills part 以 `ContextCall` 的普通 Message 事实和 skill-use 注入的 SKILLS catalog/rules 完成选择，骨架
不接收或搬运 `selected_skills`。Wake、Subagent、Emotion 等 source-only hint 在调用 Agent 前构造自己的
immutable `ContextText`；`TurnRequest` 经 `AGENTS` 原样传给具体 Agent，`agent-loop` 构造
`PromptCall`/`ContextCall`，把 `skip_sections` 作为独立 SYSTEM_PROMPT.build 参数，并把 `context_texts`
与 `skip_parts` 作为独立 CONTEXT_INPUT.build 参数。Wake 分别传 long-term-memory section 与 memory part；GitHub Watch 的
`skip_parts` 走同一路径。禁止 ambient `ContextVar`、Message metadata、
Session metadata 或 durable Inbox transport。

`DropLevel` 只有 `extra`、`repeat`、`keep`：装饰内容先丢，可再次生成的 skills/memory/context 其次，必须
保留的规则或 hint 不丢。展示 `order`、source `name` 与预算 `drop` 是三个独立轴；context 也只按
`(order, name)` 排序，name 只是相同 order 的稳定 tie-breaker。`provider-input` 只能依次
移除 extra、repeat，再按完整 prompt history 边界缩窗；Core 与 provider-input 都不能按插件名判断。

领域 `TurnRequest(start, message, ...)` 不是新的总 context。它只并列携带各 owner 的 immutable typed
call fact，且没有 metadata。现有 `agent.control.models.TurnRequest(thread_id, input, metadata)` 在 M2 改名
`ControlTurn`，只属于 control transport/store；control 与 channel source adapter 各自在唯一边界把外部
输入先接受成 TurnStart、再解析成领域 TurnRequest。M2 同批把所有 ingress 改送 AGENTS；进入 Root-bound concrete Agent 后不再
携带 raw metadata。

`TurnSource(name, ref)` 是开放的来源身份；两项必须非空、无首尾空白，ref 复用 source 已有的 durable
identity。插件可以提供新 name，所以不用封闭 enum；两项只用于幂等 admission 和诊断，不能代替行为选择。
所有 source 先 `Agent.accept`，再 `Agent.run`；run 前 skip/fail/cancel 只用 `Agent.finish`。agent-loop 先让
sessions reserve、再由 TaskControl claim、最后 mark claimed，三步全成才返回 TurnStart；boot 在 AGENTS ready
前按 durable reserve/claim receipt 收敛。Wake 的 select/claim/quiet 仍是自己的 source gate，不能伪装成
ContextPart 或 before hook。`TurnWait.result() -> AgentResult` 只返回 immutable terminal
fact。`AgentOutput` 只含 text/thinking/validated media/attachment 与 optional persisted message identity；
user-only/no-save completed 时 message identity 必须为 None，sink 只用 TurnSource ref 幂等。AgentResult 另含
typed items/usage/status/error/ended_at，并保留现有 items_json/usage_json write set，没有 Channel、chat、reply
target、sender、delivery flag、callback 或 metadata bag。agent-loop 必须先以 SESSIONS.finish 把它 commit 到现有
turns terminal row，TurnWait 才能返回；相同
TurnSource ref 的 terminal run 只以 SESSIONS.result 返回同一结果，provider/tool/save/observer=0。M2c 后
agent-loop 零发送，每个 source 只调用自己的窄 sink。TurnWait 另有与输出流正交的 one-shot
`input_seal() -> InputSeal | None`；Agent 在旧 lock 点发无 payload seal 并等回执，source 锁自己的 active input
后 done，失败则 fail 并让 Agent 终结。source scope 的 finally 调幂等 TurnWait.close，它切 discard 并让
未回执 seal fail，但不伪造 caller cancel。normal 保持 seal→lock→done→OutputDone，三条 fallback 保持
OutputDone→executor return→seal→lock→done。seal 不含 source/callback/registry/reason/metadata。需要 live
progress 的 caller 以 run 的普通 watch 参数取得
TurnWait 的单 consumer、有界 `TextUpdate | ItemStart | ItemDone | OutputDone` stream。TextUpdate 保留 text 与
thinking，四种 update 共用严格递增 sequence，start/done 同 id/kind 配对。OutputDone 一比一保持
现有 normal/安全拦截/上下文过长/模型超时四条 output stream 封口时机，不冒充 terminal，也不拥有
input source lock。projection 错误、断开或提前 close 只把
feed 切成 discard，Agent 仍能 terminal；caller cancel 才 cancel 同一 Turn。它没有 listener registry、callback
field 或改写能力，terminal 重入只返回 durable items/usage/final output。M2b 删除旧 Channel bridge、五种
lifecycle event/producer/registration 和全部 product subscriber；外部 artifact 有 subscriber 就阻断 deploy，不双发。
内部 TurnItem 同样收窄成 `UserItem | AssistantItem | ThinkingItem | ToolItem | ErrorItem`。工具
args 只用 immutable JsonValue/JsonList/JsonMap，provenance 只用空或四项 `SkillRef`，TurnError 只用
type/message/retryable；旧 error data 或其他 provenance 有任一真实值就阻断 M2。sessions/Control adapter
分别唯一拥有 DB/wire 编码，replyTo、client id 与 transport metadata 留在 source ledger，不允许
kind+dict、assistant_data 或 OutboundMessage bag。

- `past_read: PastRead(full|empty)` 只交给 `sessions`；sessions 从同一次 persistent history 选择返回
  `HistoryViews(runtime_history, prompt_history)` 两个不同 typed projection，分别进入 ContextCall 与
  TurnInput；empty 让两者都空但不修改 Session；
- `context_texts`、`skip_sections`、`skip_parts` 分别只交给 context-input/system-prompt/context-input；
- `tool_grant: ToolGrant`、`tool_picks: tuple[ToolPick, ...]` 与 `turn_tools: tuple[Tool, ...]` 只交给 `tools`；
- `save: SaveChoice(user, assistant)` 与 `effects: EffectMode(run|skip)` 只交给 `sessions`；
- `step_limit` 只交给 `agent-loop`；`source: TurnSource` 只由 `agents` 在 create/resume 边界校验并冻结，
  agent-loop 原样带给 Tools 和诊断，任何 consumer 都不得按 source 值改变行为。

`AGENTS` 除自己拥有的 TurnSource 外只转发 TurnRequest，不能解释其他值。`agent-loop` 只把每个字段交给唯一 owner；没有通用 metadata
迭代、ambient `TurnExecutionScope` 读取或 feature name 分支。当前 `skip_session_history`、`omit_user_turn`、
`omit_assistant_turn`、混合 `disabled_prompt_sections` 和 scope 内同义字段按 owner 批次标 deprecated，唯一
typed path 生效后同批删除。persisted `effects.post_commit` 仍是 sessions 的现行数据语义，不等于保留旧
runtime metadata decoder。

`provider-input` 是 Akashic 相对 DSH 的有意差异。DSH compaction 通过 `agent/pre-step` 改变进入 Step 的 Message
（`packages/compaction/compaction-basic/src/index.ts:148-166`），并向 Session log 追加带
`surfaceOp: replace` 的 user/message（`packages/compaction/compaction-basic/src/region.ts:436-475`）。Akashic 的
`sessions.db/messages` 不得因上下文裁切而 UPDATE/DELETE，且已有独立 compaction ledger 与 request projection。
因此这条“prompt history 与本 Turn 进展 → 每次有限模型输入”是真实边界，但它不是 before-step hook，
不拥有 prompt history，也不允许任意中间件。定义、provider 和 consumer 必须齐全：公共结构合同定义
`open`、每次 `build` 和每次 `settle`；basic/compaction 普通插件二选一提供，`agent-loop` 唯一消费。

`TurnInput` 冻结本 Turn 的 prompt history、PromptText tuple 与 ContextMessage tuple；
`PROVIDER_INPUT.open(TurnInput) -> InputState` 每个 Turn 恰好一次。随后每次 provider attempt 都必须调用
`InputState.build(InputCall) -> ProviderInput`。`InputCall` 冻结 `call_id`、call/try 序号、`normal` 或
`too_long` 原因、当前完整 Turn transcript、tool schemas、model/input/output limit 和此前已结算 usage；
`ProviderInput` 返回完整 provider payload、`InputSize` 与一只 opaque `InputReceipt`；agent-loop 直接保留 size，
provider-input 再 observe 同一只 size，observer failure 不改变返回。该 attempt 无论成功、
上下文过长、失败或取消，都必须恰好一次调用 `settle(CallResult)`；只有返回的 `InputRetry` 允许同一
逻辑 call 再 build 一次。basic provider 永不要求 overflow retry；compaction provider 可以在第二次 build
收缩输入。两者只能按 CTX-002 先移除 `drop=extra`、再移除 `drop=repeat` 的 system/context 输入，然后
减少完整 prompt history 边界；`drop=keep` 不得移除。它们必须保留 concrete loop 给出的
system → prompt history → context → turn transcript lane 顺序、每条 Message 的 role 与既有
`normal|context` kind，以及 transcript 内执行顺序，不能重排或合并 lane。首个 call 的 transcript 只有
独立 current Message；后续 call 在它后面保留已发生的 assistant/tool 进展。
具体 loop 只传这些 typed value，不识别 compaction，不读写 provider 的 turn-local state。
models 每次 call 直接返回 immutable `ModelReply(text, tool_calls, thinking, finish, continuation, use)`；use 是
同一次 ModelUse。agent-loop 直接消费 reply，models 再 observe 同一只 use，observer failure 不改变 reply。

compaction provider 可在 `InputState` 内私有保存 ledger head、已闭合 tool batch、token meter 和待发布
fact。它从下一次 `InputCall.transcript` 识别新闭合 batch；成功 settle 记录 usage，并在单次运行只发布
一次已提交 fact；崩溃恢复按 `source_ref` 幂等补发。
失败或取消不能把已经提交的 checkpoint 伪装成回滚，下一次 `open` 必须从 ledger/receipt 重放并补发。
这保留了当前每次调用 gate、overflow retry、tool batch、usage 与 recovery 行为，却删除 mutable
`ProviderRequestBinding` 和 Core 私有 gate。

`build`/`settle` 不是一对公共 phase hook：两者不能分开注册，没有 listener 顺序，不接受任意 Agent
状态，也不能改写别的 capability。它们只属于同一个 exclusive provider，并由同一只 receipt 强制配对。

`reply-output` 是 Akashic 相对 DSH 的第二项有意差异。DSH 把 provider stream 组装成 typed content blocks
后直接 append 同一 assistant message（`packages/core/agent-loop/src/agent.ts:341-427`）；Akashic 的
Citation/Meme 仍用模型内 hidden marker，必须在 Session commit 前解码，但不需要复活 after phase。
`REPLY_OUTPUT.add(ctx, ReplyPart)` 注册普通 part；所有 part 只看同一 immutable ReplyCall，并返回引用 raw
text 的 `ReplyMark(start, end, text, media)`。service 拒绝越界或重叠，一次合并为 FinalReply；name 不是
order，part 看不到对方结果。reply-output 自己仅保留现行 trailing `<name:value>` hidden-marker grammar，
对未被 part claim 的 trailing marker 做固定空替换。它没有任意 callback、priority 或总 mutable ctx。
span 必须非空；已有 media 保持原顺序，mark media 按 source span 追加，mark 内顺序与重复项不变。
media 只接受由 tool boundary 或 part 的窄 file grant 铸造的 immutable MediaItem；reply-output 没有文件权限。

`REPLY_OUTPUT.open(ReplyCall) -> ReplyState` 在附件与 Session 写入前 prepare 每只 part 的私有 receipt；
caller 只把 ReplyState.output 的正文和 media 同时用于持久与发送。Session save 的 saved/skipped 或 Turn 的
failed/cancelled/interrupted 确定后必须恰好一次 `settle(ReplyState, ReplySave)`；saved 带 deeply immutable TurnSaved，part 只能提交自己的 ledger/
fact。prepare crash 以 SessionRead 的 turn identity 恢复。没有 feature part 时，basic decoder 仍执行
既有 parser 和 fixed hidden-marker decoder；没有 marker 的正文与已有 media 保持不变。open 中任一 part
失败或取消时，service 先收敛已经 prepare 的 receipt，再抛出原错误，
不能交付半只 ReplyState。
每只 ReplyPart 显式实现 ready/open/settle；receipt 只回给同一 part。产生 durable receipt 的 part 必须由
同一 artifact/generation 注册同名 SwitchPart，pure part 的 receipt 恒为 None 且不得假注册。旧 generation
lease 未归零或自己的 receipt 未 settle/abort 时不能 remove/replace。
M6b 到 M9 只允许账本中的 private `OldReply` 把 Citation 旧 metadata 交给 sessions 同一事务；它不
进入 FinalReply 或 public Service，Citation ledger 迁完后同批删除。
重启时 publication gate、ingress 和 sender 先关闭；exact stable Root 中普通 agent-loop 的 start Effect 是唯一
顺序 owner，Core 只等待 Root ready，不识别 Service 名。sessions 完成 schema/integrity 与 writer ready 后，
agent-loop 必须先调 `SESSIONS.recover()`，把 crash 遗留 queued/running Turn 收敛成 cancelled/interrupted，
且 provider reboot call=0。
随后才依次运行 REPLY_OUTPUT receipt recovery、SavePart ready/saved notice replay 和 source delivery replay；
recover failure 保持 degraded 且不发送。hot reload 仍由完整 Turn lease 与 SwitchPart 保护。

sessions 总在 `turn_saves` 写 immutable saved/skipped outcome 与已验证 channel；saved 还在 Message
commit 同一事务写 pending `saved_notices`、冻结当时 SavePart 的 name/artifact/generation recipient tuple，
并返回 SaveResult.saved(TurnSaved)，skipped 不写 Message/outbox，也不伪造 TurnSaved。saved/skipped 都调用
`SAVE_NOTICE.send(SaveResult)`。`SESSIONS.size(SaveResult)` 直接返回同一 save 后的 immutable HistorySize，
sessions 再 observe 同一个值。唯一 caller 固定 reply settle → old-commit → SAVE_NOTICE → live SaveResult
observe → SESSIONS.finish → source sink。M2c 已物理删除 dispatch/return 两个 module、Agent sender 与 flag。
M5b 在 PromptSet、ProviderInput、ModelReply 和 ToolUse 都成为 closed return 后一次删除
剩余 after-turn phase 与八个 builtin；extra/AfterTurnCtx/telemetry 零 consumer 就删除，绝不恢复 Agent sender。
只保留 private `DEPRECATED(EXTERNAL)` old-commit，以显式 immutable value 生成完整
TurnCommitted 和 exact budget log；它不得读取 observer、raw Session、ContextBuilder、metadata 或旧 ctx。
M7a 只删除其余总 phase suite，不再次拆 after-turn。
SavePart 用普通 `SAVE_NOTICE.add` 注册，只有 ready/accept；part 之间并发、全部等待并聚合错误，name 只作
身份。saved 时 sessions 按 recipient 持久化 done，全部 done 才终结 notice；skipped 是 live checkpoint，
part 以自己的 prepare + SessionRead.status 恢复。每只 part 以 Turn identity 幂等并注册同名 SwitchPart；
自己的 frozen pending recipient 未清空时拒绝 remove/replace。sessions writer 有任一 pending saved notice 时
也拒绝切代。boot 固定 session ready → agent-loop crash terminal → reply receipt → SavePart/saved notice →
source delivery，不能让新 generation 解释旧 notice。
`SessionRead.turn(TurnSaved)` 只返回 source-neutral TurnView：session/turn/channel/chat/time identity、按 seq 排列
且与 receipt message_ids 完全一致的 typed MessageView 与 EffectMode；没有 dict、extra 或 Observe
统计。`SessionRead.status` 还返回 immutable TurnInfo：session/turn/optional channel、terminal result、save outcome、
ended_at 与 saved 时的同一 TurnSaved；running 或 pre-save failed/cancelled/interrupted/skipped 的 save=none
合法，completed terminal 或依赖 save 的 durable row 却缺 turn_saves 才是损坏；未知返回 None，非法组合
fail-loud。
agent-loop 在 provider/tool 前用 PLG-014 `parallel` 一次发布完整 InputBatch，Emotion durable prepare 后才
继续。Observe 的 InputSize/PromptSize/HistorySize、ModelUse、ToolUse、LoopStep/RawReply、ReplyText、MemeUse
都使用 PLG-014 `observe`；它只在 turn-local memory join，并在 SAVE_NOTICE 成功后的 live SaveResult 恰好
enqueue 当前非阻塞 writer 一次。HistorySize 在 SaveResult 后按当前 provider-facing rendered history 计算；
saved 含本轮 Message，skipped 使用未改变历史。Observe 不是 SavePart、不写 per-fact staging，queue drop、
observer error 和 final fence 前 crash 都可像现行不可重放 TurnCommitted 一样丢 trace，但不能改写主链。
agent-loop 另在 sessions durable terminal 后、TaskControl release 前以 `observe` live 发布窄 TurnEnded；
Emotion 用它 abort failed/cancelled/interrupted/save=none 的 InputBatch prepare，Akasha/Shell 只用
identity + SessionRead 收敛自己的 pending。listener failure 只记 cleanup_degraded，不能回滚 terminal 或阻止
最终 task release，也不新增总事件 replay。

`session-view` 只有在证明需要“对已提交 Session 事实做通用可回放 fold”时才是额外普通
插件。它不参与 prompt history 构造，不允许 I/O 或回写 Session。若真实 consumer 可以直接读已提交
事实，就不安装它。数量由能力决定，不由 DSH 当前“七块”反向决定能力。

Channel、Command、Scheduler、Wake、Subagent、Compaction、Markdown memory、Tool Search、Shell 和其他产品能力
仍是普通插件。它们注入上述 Service，不能要求 Core 增加插件 ID、工具名或来源特判。

```text
sessions ───────► SESSIONS ────────────────────────────► agent-loop
models ─────────► CHAT_MODELS ─────────────────────────► agent-loop
tools ──────────► TOOLS ───────────────────────────────► agent-loop
context-input ──► CONTEXT_INPUT ───────────────────────► agent-loop
system-prompt ──► SYSTEM_PROMPT ── build ──────────────► agent-loop
                          └────── render ──► provider-input ──► PROVIDER_INPUT ──► agent-loop
reply-output ───► REPLY_OUTPUT ─────────────────────────► agent-loop ──► sessions
agents ─────────► AGENTS ◄──── register default factory ───────────────── agent-loop

session-view? ── committed Session facts ──► observers (只有真实 fold consumer 时)

Channel / Command / Scheduler / Wake / Subagent
                    │ public Service
                    └──────────────► agents
```

安装、权限和生命周期上同权不等于角色对称。Registry、provider 和 driver 本来就有不同依赖位置。
`agent-loop` 位于图顶端不是特权；它仍以同样的 manifest 装载，可禁用、卸载或被另一个 Agent
factory 替换。Root 在 `snapshot.sealing` 时要求正式拓扑恰有一个默认 factory；缺失或冲突在发布前
fail-loud。DSH 也通过 `agents.setFactory()` 把 registry 与具体 driver 分开
（`packages/core/agent/src/index.ts:352-422`）。

## Core 边界

Core publication plane 继续唯一拥有 artifact、generation、candidate isolation、Root readiness、stable/latest、
lease、原子 publication、drain 和恢复日志。它保留五个领域中性执行原子：

1. `ServiceCall`：绑定一个 `ServiceKey`、取得构造时已固定的 exact lease、绑定当前 task、等待调用完成并
   逆序释放。公开 `call(action)` 没有 selector、snapshot ID 或 plugin ID；
2. `ServiceHold`：同样绑定一个 ServiceKey，并由 sealed caller capability identity 铸造不可伪造
   HoldKey；两个 holder 调同一 Service 也不能相互 pending/call/drop。在 live exact lease 中 reserve
   全局 HoldId，Core journal 只冻结 HoldKey、sealed ServiceKey 与 exact Root/snapshot/artifact/generation 并 pin；
   owner row 另行冻结 HoldId、
   source generation、Channel generation/config、target 和 stable delivery key 后 activate。done/abort 必须先
   durable 写入，再 drop，最后删 row；unknown 保留 hold/degraded。reboot 只新建 ephemeral binding，
   不复活旧 binding；artifact 缺失 fail-loud，不 fallback current stable；
3. `RootScope`：只取得自身 exact Root lease；Root 退休后返回 `RootRetired`，不改投新 stable；
4. `TaskControl`：只按 opaque scope key/task key 原子 claim，保存 exact lease、task 和 cancel callback，terminal 后
   release；
5. `RootSwitch`：只协调跨 Root 不能共存的共享 owner。普通插件以
   `ROOT_SWITCH.add(ctx, SwitchPart)` 注册唯一 name 和只操作自己窄 grant 的 stop/leave/enter/start/recover。

五者不识别 Message、Turn、Session、Agent、ReAct、工具名或来源，不为缺失 Service fallback。snapshot lease
包住完整 ReAct Turn 不授予某个插件特权；它只保证同一项工作始终使用同一盒插件。
attached validation 只用 Core 铸造的 exact candidate `ServiceCall<AGENTS>`；同一 call action 内 accept、run 或
finish，并等待已经 terminal 的 AgentResult；TaskControl 此时可以 release，但 action 的 exact Root lease 仍由
ServiceCall 持有。有 output 时 source adapter 调 recording Channel 并等 ACK，随后 action 返回，ServiceCall
才释放 lease。它只使用临时 sessions，不触碰正式 workspace 或 sender。M2c 后 passive、Control、Scheduler、Wake、Subagent 与 attached
分别使用 conversation Channel、control response、job delivery、Wake durable delivery、parent Tool/continuation
和 recording Channel；没有明确 sink 的 source 不发送。M2c 切换这些 caller 后同批删除 concrete Agent direct
dispatch，不保留 flag 或 fallback；M7b 只移动 passive source/sink 的代码位置。
durable source 在 AgentResult commit 后、自己的 delivery row prepare 前 crash 时，以同一 TurnSource ref 和
原 typed request 重新 accept/run，只读 exact durable AgentResult，再幂等 prepare/send；不得从 Message、
TurnSaved、notice 或正文猜 output，ACK 前不得删除 handoff。source 在 admission 时按 reserve→
source row→activate 建立 hold；row 冻结 HoldId、source generation、Channel generation/config、target 和
stable delivery key，不冻结 live binding/token/socket。reboot 用 exact old Root/config 新建 binding；done/abort
先写入、再 drop、最后删 row，unknown 保留 hold/degraded。缺 Root/artifact/config 时 degraded、
sender=0，不 fallback current stable。M2c 同批物理删除旧 dispatch/return
module、direct dispatch 和 flag；这是把一名旧 sender owner 原子交给已有 source sink，不是双路径。

RootSwitch 只把 owner artifact/generation identity 改变的 part 放进 closed plan。publication gate 先关新
lease，并等待每只 changed old owner generation 的 lease_count=0 且 hold_count=0；snapshot 与 ServiceHold 对其中每个 generation 计数，
所以覆盖旧 Turn 尚未调用该 part 的未来路径，而不拖住无 shared owner 的普通 generation。Core journal
随后 pin old/new immutable artifact，并记录两边 participant identity、step 和 terminal；gate 内依次
stop/leave old、enter/start new。start 后仍无新 lease，只有 stable
identity 与 terminal record 原子提交后才开放。失败逆序恢复 old。进程崩溃时 terminal 前收敛 old，terminal
后收敛 new；install/remove/replace 即使某一边 stable Root 已没有该 part，也能从 journal pin 重建
recovery-only closure。恢复失败保持 degraded 且不开放 lease。0036 的延后门槛已经由 skill link、sessions、
Activity、Channel 与 command 这些现有 shared owner 满足；durable ReplyPart/SavePart 也按同一条件注册。
Core 不按 participant 的数量或名称分支。

## 非特权判定

- 所有骨架插件使用与外部插件相同的 manifest、loader、Context、Fiber、Effect、PluginRuntime、generation 和
  disabled builtin 规则；“内置”只表示默认发行。
- 实现只能 import 版本化 public Plugin API、结构合同和自身包；不得 import 旧 Core class、Session 私有 store
  或兄弟插件源码。
- Core 与 bootstrap 不按插件 ID、`tool_search`、`message_push`、Shell、模型、记忆文件或 Channel 来源分支。
- `sessions` 是 `sessions.db` 的唯一正式 writer；其他插件只获得窄 Service，不获得任意 SQL 或全功能 repository。
- 缺少 required Service、默认 factory 或唯一 writer 时 Root 不发布；运行时没有 `try plugin else legacy`、
  旧新双写、双 sender 或静默默认值。
- 一个内置插件移到独立仓库后，只要声明和版本化公共合同相同，就不需要 Core patch。

## 行为归属

ReAct 的 provider/tool 迭代、工具 batch 结算、max-iteration、stream、inbox、取消与 terminal 是一个具体
Agent 算法，留在 `agent-loop` 内，不为每个步骤各造插件。模型选择从 Agent request 传入 `models`；
Tool Search 只使用 `tools` 的 scoped view；system sections 由 `system-prompt` 拥有；model-facing 权威
prompt history 由 `sessions` 派生，临时附加 Message 由 `context-input` 构造，有限 provider input 由
`provider-input` 构造；最终 raw reply 由 `reply-output` 一次解码成同一份持久/发送 FinalReply；
`agent-loop` 唯一固定 system → prompt history → context → turn transcript 的 envelope
顺序；transcript 以 current Message 开始，再接本 Turn 已发生的 assistant/tool 进展；
提交由 `sessions` 拥有；发送由
Channel/Delivery 拥有；Shell cleanup 由 Shell 插件消费 terminal fact。

现有 `before_turn`、`before_reasoning`、`before_step`、`after_step`、`after_reasoning` 和 `after_turn` 总 phase 全部退役。
DSH 虽有只返回 `reject | enter(messages)` 的窄 `agent/pre-step`
（`packages/core/agent/src/runtime-types.ts:55-63,226-238`），但没有对称的 after-step 套件；Akashic 当前也没有
非测试 `before_step` consumer。因此本迁移不预建 `before-step`、`step-decision` 或对称 phase API。真实 consumer
未来必须用独立需求证明新 seam。

迁移后只保留 owner 明确的领域接入点，例如 system section、context Message、provider input、reply mark、
tool view/run、`SaveResult` 和 outbound view。不新增 `ReplyEdit`；它只会成为缩小的 after-reasoning 有序
改写链。模型指引归 `system-prompt`；模型 hidden protocol 归 reply-output 的同源不重叠 mark；已提交的
插件事实归插件自己的 ledger；仅 Channel 显示变化归 outbound view。

## 迁移与回滚

迁移按 owner 串行切换，不进行灰度、流量分组、运行时 shadow、双执行或双写。isolated candidate 只做拓扑、权限和
无正式副作用的验证。每个实现批次先标记待退役 owner，再把正式调用者一次切到唯一新 owner。关键测试、两个独立
Terra xhigh review 和一个独立 name review 通过后，同批物理删除旧实现。外部 consumer 只能以明确
`DEPRECATED(EXTERNAL)` migration block 阻塞公共面删除；跨仓收尾后不保留 alias、adapter、flag、fallback 或 block。
M8 Core stop 只允许 1.5 账本列出的九类 exact block：prompt、context、turn metadata、reply、agent skill
roots、drift skill roots、Shell、committed event 和 message frame。它们只能服务 2026-09-01 hua-home exact
stable consumer，不得新增 consumer；M9 按 seam 串行迁 consumer，每类最后一名离开时只删除对应 block，
最终 Gate 不再承担集中清壳。
Plugin Undo、Setup Helper 与 Status Commands 的 stable artifact 已是普通 COMMANDS consumer，属于 keep，
不需要 command bridge。

回滚只选择上一个完整 Git commit、不可变 generation 和执行前备份。已提交的 Session 行、已发送消息、远程调用
或文件写入不因代码回滚而伪装撤销。

## 影响

- `sessions.db/messages` 继续正常只追加；本决策只新增 forward-only `turn_saves` save outcome 与
  `saved_notices` outbox schema，不改
  Turn/Attempt/Interaction 身份、compaction ledger、附件或删除权限。
- 新 user row 停止写 `llm_user_content`/`llm_context_frame`，history reader 同批删除；旧字节保留且零 reader，
  避免本轮 transient context 在下一 Turn 重放。新 row 也停止写 `akasha_reinforce`/`akasha_forget`；历史 marker
  经备份、count/hash 校验一次导入 Akasha 私有 ledger 后零 reader，不双写、不 dual read。
- Proactive Feedback 只接受同一 TurnView 的 exact persisted user/assistant；缺 assistant identity 时不再按
  正文从旧历史猜一条同文回复，user-only/no-save 明确 no-op。
- passive、control、scheduler、wake 与 subagent 最终使用同一 `agents` Service，但各来源仍拥有自己的准入前
  规则、领域状态和 delivery settle。
- reload 后可以通过 opaque `TaskCancel` 取消旧 Root 内尚未终结的工作；旧 owner 持有自己的 lease 到
  terminal，不迁移内存业务对象。
- 禁用任一 required plugin 使依赖拓扑不可发布，不触发 Core 私有实现。

## 验收

- [ ] snapshot 外入口只能通过泛型 Service 调用边界进入；该边界源码与测试无 Agent 领域词。
- [ ] root-bound task 在 reload 前后只运行自己的 exact Root；opaque task 可跨代取消，同一 scope 不由两代
  同时 claim。
- [ ] RootSwitch 在 publication gate 内完成 stop/leave/enter/start；journal pin 两代 participant，逐 step
  crash 对 install/remove/replace 都只收敛到 terminal 指定一边；sessions 与其他四名共享 owner 不由 Core
  名称分支切换。
- [ ] 最小骨架全部由正式 v3 loader 挂载，依赖图无环，Root sealing 恰有一个默认 factory。
- [ ] `sessions` 唯一派生 prompt history；`provider-input` 每 Turn 只 open 一次、每个 provider attempt
  恰好 build/settle 一次，overflow retry、tool batch、usage 与 recovery 等价；`session-view` 若存在只
  fold 已提交事实，且有 exact consumer 证据。
- [ ] `context-input` 每 Turn 只 await build 一次；每项 part 只返回自己的 text，service 统一构造带
  source、`trust=derived` 和 context kind 的独立 immutable Message；provider retry 不重复查询，且
  Session write set 不因临时 context 增加；任一 part 失败或取消都不向模型交付 partial tuple；Akasha
  retrieval 的 content、prompt history、channel、chat identity 与 message time 和旧基线一致。
- [ ] PastRead.full/empty 让 HistoryViews 的 runtime_history/prompt_history 同时为完整/空选择且类型不
  混用，并且不改 Session；SaveChoice、
  EffectMode、ToolGrant/ToolPick/turn_tools、step_limit 与 source 分别只由指定 owner 解释，旧
  `skip_session_history`、`omit_*`、`disabled_prompt_sections`、PostCommitEffect runtime decoder 与
  TurnExecutionScope behavior consumer 最终全为零。
- [ ] `agents` 只拥有合同/registry/factory；默认具体 Agent 的 inbox、Turn/Step、cancel/terminal 只属于 `agent-loop`。
- [ ] `reply-output` 的 part 只声明同一 raw reply 上的不重叠 ReplyMark；FinalReply 同时用于 Session 与
  Channel；Citation/Meme 的正文、ids、media、附件和 crash settle 与旧行为等价。
- [ ] 没有 `before-step`、`after-step`、`ReplyEdit`、总 mutable ctx 或对称 phase 套件。
- [ ] Core 零插件 ID/工具名/来源特判，零旧 `AgentLoop`/`PassiveTurnPipeline` consumer，零兼容壳。
- [ ] 固定场景依次比较迁移前后 Session write set、事件、provider/tool trace、stream、delivery、error/cancel/
  interrupt 和附件结果；未超窗 provider payload 只允许三项逐字段入账的 prompt-shape 差异：Akasha memory 从
  system 回到 context、旧 after-current hints 移到 current 前、每条 context 显示 source 与
  `trust=derived`。超窗时另允许 CTX-002 的 extra → repeat → prompt history projection，并逐项记录
  lane、name、DropLevel、size 与剩余 lane；其他字段相同，能力结果等价。
- [ ] M0 由一个 Terra xhigh concept review 和一个独立 name review 批准；M1～M9 每个实现批次有
  两个独立 Terra xhigh implementation review 和一个独立 name review；最终全量 Gate 通过。
- [ ] M8 只剩账本锁定的九类 `DEPRECATED(EXTERNAL)` block；M9 每个 seam 的最后一名 live consumer
  重装后立即只删除对应 block、旧 event/type/export、旧源码与离线 fake，最终不留兼容壳。
