# 普通插件 Agent 骨架与被动链迁移合同

- 状态：accepted / implementing
- 日期：2026-09-01
- 决策：[0054 · Agent 由普通插件组成](../decisions/0054-agent-spine-is-ordinary-plugins.md)
- 取代范围：[React Core 与 Scheduler/Subagent 设计](react-core-scheduler-subagent.md) 中
  “React 实现属于 Core”的结构结论；既有 Turn、Session、Scheduler、Subagent 行为合同不变
- 实施分支：`codex/react-plugin-spine`
- 初始迁移基线：`f1f4560892ae92e96779ff89f848223afdcc9919`
- 最终复审实现 head：`df06efa1809dc181cf3465825f1aeaed4e89cec7`
- Git worktree：`/mnt/data/coding/akasic-agent-worktrees/react-plugin-spine`
- 当前实现 head：`df06efa1809dc181cf3465825f1aeaed4e89cec7`；M1 中性原子及 owner 修正已落地，M2 尚未开始
- 恢复引用：`backup/pre-react-plugin-spine-20260901-f1f45608`、`backup/pre-dsh-spec-rewrite-20260901`、
  `backup/pre-m1-retired-error-fix-20260901`、`backup/pre-m2-system-prompt-20260901`、
  `backup/pre-concept-review-20260901`、`backup/pre-concept-p1-fixes-20260901`、
  `backup/pre-concept-p1-round2-20260901`、`backup/pre-concept-p1-round3-20260901`、
  `backup/pre-concept-p1-round4-20260902`、`backup/pre-concept-p1-round5-20260902`、
  `backup/pre-m2-delivery-gap-fix-20260902`

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
semantic_delta: breaking
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
schema_lineages: [sessions.db adds turn_saves and saved_notices with a forward-only migration]
```

未超出模型预算时，批准的 prompt-shape 差异只有三项，全部在 M3e 发生：

1. Akasha 当前注册名是 `memory`，旧 assembler 却只把 `active_skills|retrieved_memory` 放进 context，导致
   memory 实际误入 system；新路径把它移到 source-owned context Message。
2. 旧 `extra_hints` 在 current user 之后追加；新路径把 source-owned hints 放在 current 前的 context lane，
   让 context 与独立 current 的相对顺序固定；后续 assistant/tool 进展仍按执行顺序接在 current 后面。
3. 旧 context frame 只有一层总 wrapper；新路径让每条 context Message 都携带并向模型显示自己的 source
   与固定 `trust=derived`。

这三项是修复 owner、顺序与信任标记所必需的形状变化。仅在输入超出预算时，另批准 CTX-002 已要求、
当前实现尚未正确提供的 budget projection：先移除 drop=extra，再移除 drop=repeat，最后才缩 prompt
history。每次 projection 必须逐项记录 lane、name、DropLevel、移除前后 size 与剩余 lane；同一
fixture 未超窗时不得触发。

另批准两项把临时或插件事实移出 Session Message 的 persistence-shape 修正，不伪装成字节等价：

4. 新 user row 不再写 `llm_user_content`/`llm_context_frame`。这些字段把本轮 retrieval/hint 的模型投影
   反向持久化，并在下一 Turn 被 `session/manager.py:247-265,409-462` 当 prompt history 重放；新 sessions
   只从权威 Message 正文派生 history。既有 row 字节不 UPDATE/DELETE，历史字段从切换点起零 reader，
   所以旧 transient context 不再在下一 Turn 重现。
5. 新 user row 不再写 `akasha_reinforce`/`akasha_forget`。维护窗口先把所有历史 marker 按原 message/turn
   identity 一次性导入 Akasha 私有 ledger 并核对 count/hash，再切换 feedback Tool；既有 row 字节保留但
   零 reader，不留 dual read。
6. Proactive Feedback 只接受同一 TurnView 中 exact persisted user/assistant identity。当前 stable 在
   `assistant_message_id is None` 时按正文从全历史挑最后一条 assistant
   （`proactive_feedback/plugin.py:596-612`），可能误绑旧的同文回复；最终 user-only/no-save 明确 no-op，
   不保留这条 identity fallback。M9h4 以重复 assistant 文本 fixture 证明不会跨 Turn 误绑。
7. Observe 与其他纯 telemetry observer 的普通错误只记 Incident，不再让已完成的 provider/tool/save 或
   Channel delivery 失败。当前 Observe writer 本来就是 bounded nonblocking queue，queue full 会丢弃
   （`observe/writer.py:35-65`）；新路径把同一“观测可丢、主事实不可改”的边界前移到 PLG-014 observe。

除上述七项外不批准内容丢失、Session write set 变化、tool trace 变化、重复 retrieval、重复 provider call 或
新的用户可见能力差异。baseline oracle 必须分别验证“未超窗三项 prompt shape 差异”、“超窗 budget
projection”和“两项 persistence-shape 修正”；其余 provider payload 字段相同，除上述四个旧 extra key
不再写入、`turn_saves`/`saved_notices` rows，以及 1.5 明确批准的 committed consumer 相对顺序变化外，Session
rows/write set、stream、delivery/ACK、attachment、error/cancel/interrupt 和能力结果保持等价；第 6 项只
删除跨 Turn 的错误 identity fallback，不改变 exact persisted pair 的评分算法。事件合同按
各 owner 批次从旧总事件一次切到窄事实，不把“事件类型相同”误列成目标。

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
`d9d90fd4d3027d444091fd6a38453c33f372b7ed`、Plugin Undo
`86941208ea9313086c1c5d8f33b38cf4432e599d`、Setup Helper
`3d9671bfee523e78bf421535ed77cbc94a8a4744`、Status Commands
`8d119e8cfa53bd91e4dd1e2d4dcf67edfe047cb4`、Shell Restore
`d9b9e17c7e783463a4981f60c05170361fd29827`、Shell Safety
`5230f8ac8aec521636e28e9c313ba60a49552fde`、Feed
`fd74018c2a397fcc1e6bfc2c6f5726cc0ba8e098`、Steam
`a0fda0602185a0a49aefc9fc0a381451c58d26e5`、Huayue Skills
`65273781113a23058fa1ce79acf8ce176ec9026a`。这些 commit 只固定本次审计输入；M9 前仍以当时
`hua-home` stable pointer 重查，不能把本清单当成永久事实。

| 当前接入点 | live consumer | 判断 | 最终入口 |
|---|---|---|---|
| `COMMANDS`、`CommandDefinition` | Plugin Undo、Setup Helper、Status Commands | `keep` | 三只 stable artifact 已是普通 v3 command consumer；只把 registry provider 从 PluginManager 迁入普通 commands plugin，public contract 和短路语义不变 |
| `PROMPT_RENDER_EVENT`、可变 `PromptRenderCtx`、`PromptSectionRender` | Citation、Meme | `move` | `system-prompt` 接受普通 section |
| `CONTEXT_PREPARED_EVENT`、`BeforeTurnCtx.extra_hints` | Emotion | `move` | 稳定模型规则归 `system-prompt`；本轮临时材料归 `context-input` |
| `AFTER_REASONING_PREPROCESS_EVENT`、`AFTER_REASONING_CLEANUP_EVENT`、可变 `AfterReasoningCtx` | Citation、Meme | `move` | 普通 `reply-output` 在持久化前合并不重叠 `ReplyMark`；Citation/Meme 各自拥有 mark、media 与私有 ledger，不保留有序 reply hook |
| `AFTER_TURN_COMMITTED`、`TurnCommitted` 总 payload | Emotion、GitHub Watch、Observe、Proactive Feedback | `move` | sessions 发布 immutable SaveResult；业务 SavePart 与各 owner telemetry fact 分开 |
| `TOOL_INPUT_PREPARE`、`TOOL_EXECUTION_AUTHORIZE` | Shell Restore、Shell Safety | `move` | M4a 由 exact external block 保留 rewrite→authorize；M9 把两段算法收进普通 Shell tool owner，删除通用 transform/authorize event |
| `is_context_frame` 与 provider dict 编码 | Status Commands | `move` | sessions 拥有 typed `MessageKind`，`SessionRead` 返回 typed `MessageView` |
| `persist_assistant_metadata["cited_memory_ids"]` | Citation | `move` | Citation 自己的 `MemoryIds` ledger；Core 不新增通用 data bag |
| Session metadata `skip_memory_retrieval` | GitHub Watch | `move` | 每次 Turn 在 `TurnRequest.skip_parts` 显式传入，再作为 `CONTEXT_INPUT.build` 独立参数；M9 前由只读 `DEPRECATED(EXTERNAL)` adapter 映射旧 metadata |
| Session metadata `source/repo/item` | GitHub Watch | `move` | github-watch 自己的 job ledger，不进入 Session metadata |
| `skill_roots` | Feed、Meme、Steam、Huayue Skills | `move` | 外部源码直接向普通 SKILL_FILES 注册 agent SkillRoot；M9 前只走 exact `DEPRECATED(EXTERNAL)` bridge |
| `drift_skill_roots` | Emotion | `move` | 外部源码向 SKILL_FILES 注册 drift SkillRoot，只做现有 check/projection；不能进入默认 SKILLS |
| `PromptRenderInput`、旧 phase frame/slot、`ConversationRuntime` fake | 仅仓库或外部离线 Gate | `remove` | 新 Service fixture；不保留 runtime alias |

Plugin Undo、Setup Helper 与 Status Commands 的 hua-home stable artifact 已直接注册 `COMMANDS`，没有
`before_turn_modules`。本机三个源码仓的 `main` checkout 仍落后并保留旧 before-turn 实现，只是 M9 选择
真实发布分支时要拒绝误用的风险线索，不是 live consumer，也不能据此发明 command bridge。M2 保持现有
`COMMANDS` public contract、命中顺序、CommandResult、Session write set 与 delivery settle；M8 只删除
PluginManager 的 registry manufacturing。三只 artifact 无需为了 provider 换 owner 而重写。

Shell Restore 与 Shell Safety 的 live 顺序是先把简单 `rm` 改成插件还原目录中的 `mv`，再对最终参数拒绝
交互式 editor、会等待密码的 sudo 和缺少 `--noconfirm` 的包管理写操作。这个顺序和拒绝原因是安全能力，
不能在 M4 丢失；但通用 transform/authorize event 仍让任意插件串行改写别的 Tool，因此判为 `move`。
M4a 只保留一只 exact `DEPRECATED(EXTERNAL)` block，逐字复用两只 stable artifact 的顺序和错误传播；
M9 将两段算法移入真正注册 `shell` 的普通 Tool owner，并删除两只旧插件、event 和 adapter。oracle 比较
最终参数、deny text、ToolOutcome status、是否实际执行、restore path/文件和失败恢复；其他工具不得经过
Shell 分支。

同一次 hua-home 只读库存还确认：Linker journal 拥有 13 个 symlink projection；不在 journal 的真实目录是
normal `novel-reader`、`video-to-subtitle-summary` 与 drift `explore-curiosity`。这些真实目录现在仍被
SkillsLoader 当第二来源读取，所以不是可直接删除的垃圾。M9 先做名称清楚的 workspace backup，把三项
完整内容和 hash 迁入 Huayue Skills 的 ordinary normal/drift provider，安装并验证新 generation 后，再把
原目录替换成 Linker-owned projection；之后 manual-directory loader 零 consumer 并物理删除。README.md
不是 skill，不进入 catalog，也不需要伪装成 provider。

`move` 项允许在 Core 阶段保留一个清楚标记的 live migration block，只为账本中的 exact external
consumer 服务；它不是目标设计，也不能新增 consumer。Core 内部必须先全部切到新入口。每个 seam 的
最后一名 external consumer 迁完并重装的子批删除对应 block、旧公开类型、事件、导出和测试。全部
`move/remove` 清零前，
整个迁移不能宣称最终完成，Core Draft PR 也不能作为“干净终态”合并。

M8 停下时允许存在的 `DEPRECATED(EXTERNAL)` block 只有下列九类；“普通 COMMANDS 保持不变”不在表内，
也不是第十只 bridge：

| migration block | exact live consumer | M9 删除条件 |
|---|---|---|
| prompt section | Citation、Meme | 两者直接注册 PromptSection，旧 prompt event/import 零 consumer |
| context text | Emotion | 直接注册自己的 system/context part，旧 context event 零 consumer |
| turn metadata | GitHub Watch | `skip_parts` 与 job ledger 生效，旧 Session metadata read/write 零 consumer |
| reply protocol | Citation、Meme | 两者直接注册 ReplyPart；Citation ledger 完成导入；OldReply 与旧 reply event 零 consumer |
| agent skill roots | Feed、Meme、Steam、Huayue Skills | 四者直接注册 agent SkillRoot，旧 `skill_roots` 字段零 consumer |
| drift skill roots | Emotion | 直接注册 drift SkillRoot，旧 `drift_skill_roots` 字段零 consumer |
| shell safety | Shell Restore、Shell Safety | 算法进入普通 Shell Tool owner，旧 rewrite/authorize event 零 consumer |
| committed event | Emotion、GitHub Watch、Observe、Proactive Feedback | 四者改读 SaveResult/SessionRead/owner fact，旧总事件零 consumer |
| message frame | Status Commands | 改读 SessionRead 的 typed MessageKind，旧 `is_context_frame`/provider dict 零 consumer |

每一类只有账本锁定的原 public import 和唯一 adapter call site；没有通用兼容 Service、动态 consumer 名单
或 alias。M8 的 zero-consumer Gate 允许这些 exact external import，拒绝 Core 内部、新插件和第十类 consumer。

`TurnSaved` 的最终合同锁定为
`TurnSaved(session_key, turn_id, message_ids, saved_at)`。所有字段及 `message_ids` tuple 都是
deeply immutable。`SESSIONS.save` 总在 sessions 的 `turn_saves` 写一条 immutable save outcome；至少一条
Message 被保存时，它还在 messages 同一事务中 append 一条 pending `saved_notices` outbox row，并把当时
sealed SavePart 的 `(name, owner artifact, generation)` tuple 冻结进同一 row，再返回
`SaveResult.saved(TurnSaved)`。两项 SaveChoice 都为 false 时只写
`SaveResult.skipped(session_key, turn_id, decided_at)`，不写 Message 或 notice，也不伪造 TurnSaved。两种 row
都冻结 SESSIONS.save 已验证的 channel，供 no-save recovery/read 使用。
这两条结果都只是**保存收据**，事务内不向 observer 发布。caller 先用同一结果完成
`REPLY_OUTPUT.settle`，再运行 M5b 建立的 exact private old-commit；saved 和 skipped 都调用 sessions 拥有的窄
`SAVE_NOTICE.send(SaveResult)`。saved 读取 frozen outbox，skipped 只对当前 sealed generation 做一次 live
dispatch。`SAVE_NOTICE.add(ctx, SavePart) -> Effect` 是普通 contribution 入口；
每只 part 有唯一 name，只实现 `ready(SessionRead)` 与 `accept(SaveResult)`，看不到 registry、其他 part、
reply state 或 Session writer。saved send 只把同一 immutable value 发给 outbox 中仍 pending 的 frozen
recipients；每名 consumer 成功后 sessions 把它自己的 recipient state 置为 done，全部 done 后才终结 notice。
失败项保留 pending 并阻止 Channel delivery。skipped send 对本次 Root 中 sealed parts 做同样的 structured
parallel call；失败也阻止 delivery，但没有 outbox/replay 声明。各 part 对不适用的 outcome 显式 no-op。
它不增加 channel、正文、reply、tool、model、prompt、
统计、展示或 `extra`。GitHub Watch 可直接使用 identity；Emotion、Akasha 与 Proactive Feedback 以 identity
调窄 `SessionRead` 读取已保存的 typed `TurnView`。

`SESSIONS.size(SaveResult) -> HistorySize` 是 save 后的同步窄读取：saved 从同一事务已经提交的
provider-facing history 计算，skipped 从未改变的 committed history 计算；它先验证 result 与
`turn_saves` identity，再返回 HistorySize 给 caller。sessions 随后 typed observe 发布**同一个**值，
observer failure 不影响返回值。它不返回 history rows、Session 对象或 repository。

`SessionRead` 不再返回 `Mapping[str, object]` 或 Message `extra`。public read contract 只有
`message(message_id) -> MessageView | None`、`turn(TurnSaved) -> TurnView` 与
`history(TurnSaved) -> HistorySize(turn_id, messages, chars, tokens)`，以及
`status(session_key, turn_id) -> TurnInfo | None`：

- `MessageView(message_id, client_id, seq, role, kind, content, time)`；client_id 是 source 提供且已验证的
  optional identity；role 只有已持久化的 `user|assistant`，kind 只有 typed `normal|context`，content 是
  数据库权威正文，全部字段 immutable；
- `TurnView(session_key, turn_id, channel, chat_id, started_at, messages, effects)`；messages 是按 seq
  排好的 MessageView tuple，恰好包含 TurnSaved.message_ids，所以 ordered user ids、assistant id/content、
  是否保存 user 和 MessageKind 都可直接派生；effects 复用 immutable `EffectMode`，不带 metadata bag。
- `TurnInfo(session_key, turn_id, channel, result, save, ended_at)`；channel 在 save=none 时可以是 None，
  save=saved/skipped 时必须是 SESSIONS.save 已验证并 durable 冻结的 source fact；result 只有
  `running|completed|failed|cancelled|interrupted|skipped`，save 只有 `none|saved|skipped`。save=saved 时还带同一
  immutable TurnSaved，save=skipped 时以 `turn_saves` 证明“已明确不保存”，
  save=none 只表示尚未走到 save，不能猜成 skipped 或 failed。running 的 ended_at 必须为 None，其他
  result 必须有 ended_at；字段组合不合法即损坏。

`turn` 以 saved-notice row 的 identity、recipient-independent payload 与 exact message ids 构造 view；普通
`message(id)` 对未知 id 可以返回 None，但拿已提交 TurnSaved 查询 turn 时，缺 row/message、重复或倒退 seq、session/turn
identity 不符、非法 role/kind/effects 都是持久化损坏并 fail-loud，不能返回空 view。status 只读 sessions
durable turn/save outcome，不创建 Turn，也不把未知伪装成 failed。已知 running Turn，或 terminal
failed/cancelled/interrupted/skipped 在 `SESSIONS.save` 前结束，都合法返回 save=none；completed terminal、
saved_notices row 或带本 turn identity 的已提交 Message 都证明 save 已发生，若此时缺对应 `turn_saves` row
就是损坏。saved 缺 TurnSaved、skipped 带 message ids、terminal 倒退或 save outcome 冲突同样 fail-loud；未知
session/turn 才返回 None。channel/chat/started_at/
effects 是 commit 时已经验证并冻结的 source/session facts，不从插件 metadata 猜。TurnView 不含 Observe 的
tool/model/retry stats、raw reply、诊断或任何 feature 字段；那些事实仍由各自 owner 发布。DSH 同样让下游从
已提交 append-only Session fact 派生 view，而不是共享 mutable turn bag
（`packages/core/session/README.md:40-47,70-72`）。

saved notice 是 at-least-once 的 durable fact，不声称跨插件数据库 exactly-once；skipped notice 是 live
checkpoint，插件靠自己的 prepare + SessionRead.status 做 crash recovery。consumer 必须以
`(session_key, turn_id)` 拥有自己的幂等 receipt；`accept` 只有在自己的结果已提交，或已 durable
接受恢复工作后才能返回。SavePart 彼此没有执行顺序；send
以 structured concurrency 启动全部 frozen recipient，等待全部终结并聚合错误，不因一个 part 先失败就
取消兄弟，name 只作身份、不作 order。saved crash 或 consumer failure 因而可能重放已经成功的 part，但不能
重复其可观察结果；已经明确标成 done 的 recipient 不再调用。

进程启动只有一条 recovery barrier。顺序 owner 是 exact stable Root 里普通 `agent-loop` 的 start Effect：它
本来就 inject sessions、reply-output 与 save-notice，因而只组合自己运行前必须 ready 的依赖；Core 只等待整棵
Root ready，不识别这些 Service 名，也没有 Agent boot table。顺序不能由各 part 自行交换：

1. publication gate、所有 ingress 和所有 delivery 都保持关闭；先恢复并 pin exact stable artifact 与 Core
   ServiceHold journal 中全部 reserved/active Root，ServiceKey/Root/artifact identity 不符就 degraded；
2. sessions owner 完成 schema migration、integrity check 和单 writer ready，但此时不运行 ReplyPart、SavePart
   或 provider；
3. agent-loop 调 `SESSIONS.recover()`，按现行 `SessionStore` 合同在一个事务中把 crash 留下的 queued Turn 终结为 cancelled、
   running/in-progress Turn 终结为 interrupted；这项 recovery 由第 1 步 pin 的 exact stable Root 中 agent-loop
   调用 sessions 的窄 recovery Service，收敛全部遗留 durable Turn，进程重启**不重放 provider**
   （`session/store.py:2692-2766`）；
4. 确认不存在 stale `running + save=none` 后才调用 `REPLY_OUTPUT.ready()`；ReplyPart 以已收敛的
   saved/skipped/terminal 结果 settle 或 abort 自己的 receipt；
5. 再调用 `SAVE_NOTICE.ready()`；SavePart 先收敛自己的 prepare，再重放 saved notice；
6. agent-loop 的 start Effect 完成，整棵 Root 才可取得；最后由各 source 以自己的 HoldKey + ServiceKey namespace
   对账 Core hold 与 delivery ledger。reserve 后/row 前 crash 的 HoldId 由该 HoldKey owner 确认无 row 后 drop；
   row 后/activate 前幂等 activate；active row 用冻结的 source generation、Channel generation/config 创建新的
   ephemeral binding/token，再恢复 prepare/send/ACK。先 durable 写 `done|abort`，再 drop hold，最后删 row；
   `unknown` 保留 hold 且 runtime degraded。全部 ready 后才开放 ingress 与 sender。

因此 REPLY_OUTPUT/SavePart 不会等待一个永远不会继续的 running Turn，也不能在第 3 步前报告 ready。
hot reload 不走这条 boot 恢复：完整 Turn lease、pending ServiceHold 与每只 owner 的 SwitchPart 保护 work。任一步 identity
未知、冲突或 recovery failure 都让 runtime degraded，observer call=0、sender call=0；notice consumer
failure 时 outbox 保持 pending、sender call=0。确定性 crash fixture 必须覆盖 InputBatch prepare 后、
ReplyPart prepare 后、Session save 前退出：重启严格记录上述调用顺序，同一 Turn 只终结一次 interrupted，
receipt 只 abort 一次，provider reboot call=0，所有 ready 完成前 sender call=0，且未保存回复之后也不重放发送。

这里直接学习 DSH 的 awaited parallel durability checkpoint：每个 listener 都运行、全部 settle 后聚合失败，
而不是 waterfall veto（`packages/core/session/src/index.ts:76-82`）。Akashic 额外保留 recipient outbox，原因是
reply delivery 必须等外部插件自己的 durable accept，不能照搬 DSH fire-and-forget `session/event`。

每只 SavePart 都必须由同一插件以同一 name 注册 SwitchPart；Root sealing 绑定 owner artifact/generation。
`SAVE_NOTICE.pending(part)` 只回答冻结给该 registered instance 的 pending row，是它的 SwitchPart stop 可用的
窄 check。remove/replace 前 stop 必须得到 false，否则拒绝 switch，旧 stable artifact 继续 pinned 并负责
replay；sessions 自己的 SwitchPart 也必须在任一 saved notice pending 时拒绝 writer switch。skipped live
call 由完整 Turn lease 保护，stop 还必须确认自己的 skipped prepare 已由 SessionRead.status 收敛。新装 part
不接收安装前已冻结的 saved notice。这样 pending notice 不会漏掉 old consumer，也不会交给 new generation 猜测。

当前 `_DispatchOutboundModule` 位于 committed fanout 之后（`agent/lifecycle/phases/after_turn.py:346-373`），
唯一 caller 是 `agent/core/passive_turn.py:787-793`。M2c 在迁 source sender owner 时已经物理删除它和
`_ReturnOutboundMessageModule`，让 Agent direct return；M5b 禁止复活这两个 module 或把 body 搬回 Agent。
其余八个 builtin 不能活到 M6 再临时改依赖：M3f 已删除 ContextBuilder owner，M5a 也已切 models。因此新增
独立 **M5b Old tail** 批次，在所有上游 closed return 可用后一次删除剩余 after-turn phase。M5b 只保留一只账本已批准的 private
`DEPRECATED(EXTERNAL)` old-commit function；它不是 Service、registry、phase 或 public type，只以显式
keyword 接受下表的 immutable value，不能拿 TurnSnapshot、AfterReasoningCtx、Message metadata、raw Session、
ContextBuilder、Service lookup 或 observer queue。函数保持完整旧 TurnCommitted build/fanout 与两条 budget
log；M9 最后一名 consumer 迁走时物理删除。delivery 从 M2c 起只属于各 source-owned sink。

| 当前 after-turn module | M5b 唯一处理 |
|---|---|
| `_BuildTurnWorkModule` | 删除；old-commit 只从 PromptSize、HistorySize、InputSize tuple、ModelUse tuple 计算同一 budget/react stats |
| `_CollectAfterTurnExtraSlotsModule` | Core/loader/hua-home stable artifact 零产品 producer，删除且不替代 |
| `_BuildTurnCommittedModule` / `_FanoutTurnCommittedModule` | 行为移入 private old-commit；不传旧 ctx 或 metadata |
| `_LogBudgetModule` | exact log 移入 old-commit；M9 由 owner-local fact/log 接管后随 function 删除 |
| `_BuildAfterTurnCtxModule` / `_CollectAfterTurnTelemetrySlotsModule` / `_FanoutAfterTurnCtxModule` | 零产品 consumer，删除且不替代 |
| `_DispatchOutboundModule` | M2c 已物理删除；M5b 不得重建 module 或 Agent sender |
| `_ReturnOutboundMessageModule` | M2c 已物理删除并改为 Agent direct return；M5b 不再处理 |

旧读取也逐项在 owner 批次切到可靠返回值；typed observe 只收到同一对象，绝不作为 old-commit 输入：

| 被删除的旧读取 | closed typed return | 切换批次 |
|---|---|---|
| `ContextBuilder.last_debug_breakdown` | `PromptSet.size: PromptSize` | M3f |
| `context_retry.react_stats` | 每次 `ProviderInput.size: InputSize` 与 `ModelReply.use: ModelUse` 的 direct tuple | M3g、M5a |
| raw `state.session.history_units()` | 当前 session owner 在 M5b 边界返回 HistorySize；M6a 改为 `SESSIONS.size(SaveResult)` | M5b、M6a |
| `_control_turn_input_source` 与 input metadata | `TurnRequest` 的 frozen Message tuple、SaveChoice、EffectMode | M2b；M6a 删除最后 metadata decoder |
| `model_binding` metadata | `ModelReply.use.choice` | M5a |
| outbound metadata 中的 persisted ids | 旧 writer 在 M5b 返回 private immutable `SavedIds(user_ids, assistant_id)`；M6a 换 TurnSaved/TurnView 并删除 SavedIds | M5b、M6a |
| mutable reply/thinking/raw/tool-chain fields | OutboundMessage 的正文/媒体、ModelReply tuple、ToolUse tuple；旧 Meme bridge 直接返回 narrow MemeUse | M4a、M5a、M5b |
| channel/chat/time/current message | TurnStart 与 immutable Message | M2b |

M5b 后唯一顺序是 `old save/reply → old-commit → old terminal AgentResult → source sink → direct return`。M6a 改为
`SESSIONS.save → SESSIONS.size → old-commit → SAVE_NOTICE.send → live SaveResult observe → SESSIONS.finish → source sink`；
M6b 再加入 `REPLY_OUTPUT.open/save/settle`。saved 和 skipped 都走同一顺序，不注册新 phase 或 listener。
这份清单来自基线十个 builtin 的真实顺序（`agent/lifecycle/phases/after_turn.py:89-402`）；M5b 实际删除其中
尚存的八个。两个生产调用者都
没有传 `plugin_modules`（`agent/core/passive_turn.py:424-435`、`bootstrap/tools.py:504-515`）；hua-home 当前
stable GitHub Watch artifact 还用测试明确禁止 `after_turn_modules`，其产品代码只监听 TurnCommitted
（`/srv/data/services/akashic/state/plugin-home/cache/github/github-watch/.artifacts/3.0.0-b9266ab3ca9932c0/tests/test_plugin_runtime.py:331-349`、
`plugin.py:278-301`）。M5b 执行前仍须重新扫描全部 enabled stable artifact；证据有新增 consumer 就先迁 owner，
不得照表强删。

Observe 不再等待一只总事件，也不新增 `*Log` 总袋子。算法进行时：agent-loop 在 provider/tool 前把本轮
全部已接受 input 一次发布为 `InputBatch(turn_id, messages)`；这不是 telemetry，而是 Emotion 的 business
checkpoint，使用 PLG-014 `parallel`，Emotion 必须在返回前按 turn identity durable prepare 完整 tuple，失败
会在任何 provider/tool 外部效果前安全停止。`provider-input` 每次 build 发 `InputSize`；system-prompt 每 Turn
build 发 `PromptSize`；每次 provider call 由 `models` 发 `ModelUse`；每次 tool call 由 `tools`
发 `ToolUse`；每次 ReAct step 与最终未清洗回复由 `agent-loop` 分别发 `LoopStep`、`RawReply`；Meme ReplyPart
只发自己的 `MemeUse`；reply-output 合并后只发 `ReplyText`；sessions 在 SaveResult 确定后，从当时已经
提交的 provider-facing history projection 发 `HistorySize`；saved 因而包含本轮新 Message，skipped 使用未
改变的已提交历史。messages/chars/tokens 继续按当前 rendered message JSON 算法计算，不从 MessageView row
重新发明口径。每个 Agent terminal 由 `agent-loop` 发 `TurnEnded`。每种 fact 只含本 owner 的标量、tuple 或
immutable value。除 InputBatch 外的诊断事实全部复用 PLG-014 的 typed `observe`：调用全部 observer 并等待 async settle，但
普通失败只变成 Incident，不改写 ReAct、Session、delivery 或 terminal；没有 Core listener 名单、name order
或总 Turn payload。DSH 也明确把 post-commit notification 的隔离失败
（`packages/core/session/src/index.ts:63-74`）与必须等待成功的 parallel durability checkpoint
（同文件 `:76-82`）分成两条合同。

Observe 只在 turn-local memory 收集这些 typed fact；它不写 per-fact staging，不是 SavePart，也不进入
saved-notice outbox。SAVE_NOTICE 全部成功后，sessions 以普通 typed `observe` 再发布一次同一 immutable
SaveResult，随后才 delivery；这只 live observation 不由 outbox 重放。Observe 对 saved 用 TurnSaved +
SessionRead 和已收集 facts 封口，对 skipped 直接用收集事实封口，assistant id 与 persisted user 保持 None。
无论哪条路径，Observe 仍只调用当前非阻塞 `writer.emit` 一次；writer queue 满、
observer error，或进程落在 old-commit/新 live fence 前都会丢这条 trace，与现行不可重放
TurnCommitted + queue-drop 边界相同，不伪造 boot trace，也不阻止发送。这里保留的是观测能力和失败边界，
不是把 telemetry 提升成新的提交条件。

这些小事实的字段也在实现前锁定：
`InputSize(turn_id, call_id, try_number, tokens, quality, changed)`；
`InputBatch(turn_id, messages)`，其中 messages 是 AGENTS 已验证并冻结的非空 Message tuple；
`PromptSize(turn_id, tokens)`；`HistorySize(turn_id, messages, chars, tokens)`；
`ModelUse(turn_id, call_id, choice, usage)`，其中 choice/usage 复用 models 的 immutable `ModelChoice`/
`ModelUsage`；
`ToolUse(turn_id, name, args_json, outcome)`；其中 outcome 是本次同一 immutable `ToolOutcome`，不复制
call_id、status 或 model-facing content；
`LoopStep(turn_id, step, text, call_ids, final)`；
`RawReply(turn_id, text)`；`ReplyText(turn_id, text)`；`MemeUse(turn_id, tag, media_count)`；
`TurnEnded(session_key, turn_id, status, ended_at)`，其中 status 只有
`completed|failed|cancelled|interrupted|skipped`。`args_json` 是已验证参数的 immutable 编码，
`call_ids` 是 tuple。没有 dict、`extra`、跨 owner payload 或“以后可能用”的字段。
PromptSize.tokens 必须保持当前 `sum(last_debug_breakdown.est_tokens)` 口径；HistorySize.chars 必须保持当前
provider-facing rendered history 的 `json.dumps(..., ensure_ascii=False)` 字符数，tokens 保持
`0 if empty else max(1, chars // 3)`。

Observe 的字段迁移必须逐项对账，不能用“small facts”一句带过。hua-home exact stable
`observe/.artifacts/1.4.1-09214c23f287f659/plugin.py:193-231` 确认 live consumer 正在读取 raw reply、Meme、
完整 post-reply budget、ReAct stats、model usage 与 tool chain：

| 当前 TurnTrace 字段 | 最终来源 |
|---|---|
| source | saved 从 TurnView.channel、skipped 从 SessionRead.status.channel 复用现行映射：wake→proactive、drift→drift、其他→agent |
| session_key、turn_id、assistant_message_id、user_msg | saved 用 TurnSaved + SessionRead.turn：assistant id 取唯一 persisted assistant，没有则 None；user_msg 按 seq 取 persisted user content 并以现行 `\n\n` join，没有则 None；skipped 用 SaveResult identity，后两项为 None |
| llm_output | ReplyText；saved 且存在 assistant Message 时与其 content equality check，不保存 assistant 或 skipped 时仍保留同一 clean text |
| raw_llm_output | RawReply |
| meme_tag、meme_media_count | optional MemeUse；未安装 Meme 时保持 None |
| tool_calls、tool_chain_json | 按 LoopStep.call_ids join ToolUse，沿用当前截断与 JSON 规则 |
| history_messages、history_chars、history_tokens | sessions 的 HistorySize；saved 时与 `SessionRead.history(TurnSaved)` equality check，skipped 直接使用同一 live fact |
| prompt_tokens | PromptSize |
| next_turn_baseline_tokens | Observe 只做 `HistorySize.tokens + PromptSize.tokens` 的同一算式 |
| react_iteration_count、react_input_sum/peak/final_tokens | 按 call/try identity 聚合 InputSize，并用 LoopStep 验证 step 边界 |
| model_output_tokens、react_cache_prompt_tokens、react_cache_hit_tokens | 按 call_id 聚合 ModelUse.usage |
| history_window | 当前 producer `build_post_reply_context_budget()` 从未写此 key，旧 trace 恒为 None；新 writer 继续写 None，不伪造值 |

ModelUse.choice 还替代旧 TurnCommitted.model_binding 的 typed 诊断来源，但 Observe 当前 TurnTrace 不写该字段，
不得为了未来需要扩 TurnView。M9h3 让 Observe 用 typed observe 在 turn-local memory 聚合，并在 live
SaveResult 恰好调用现有 writer 一次；旧 artifact 与新实现对同一完整场景逐字段比较
TurnTrace/DB row，包含多 tool step、overflow retry、cache usage、Meme on/off、SaveChoice.neither 与 queue full。
任何没有上表来源的旧非空字段都是 Gate failure；不批准静默删除。进程在 final fence 前 crash 时两边都没有
trace，boot 不补造；这项 oracle 也必须固定。

`TurnEnded` 只在 sessions 已 durable 写入同一 terminal 后、TaskControl release 前由 agent-loop 以
PLG-014 typed `observe` live 发布
一次；它没有 reply、tool、model、error bag 或 cleanup action。Akasha 只用 identity + SessionRead 撤销未
saved feedback prepare；Shell 只用 identity 查自己的 execution registry 并清理。consumer 必须先 settle
自己的幂等 receipt，或 durable 接受 recovery work，再返回。listener error/cancel 不改写 completed/failed/
cancelled/interrupted/skipped，不回滚 Session、外部效果或已合法 delivery；agent-loop 记录 `cleanup_degraded` diagnosis
后仍完成 terminal release。进程在发布前或 fanout 中崩溃时，各 owner 只扫描自己的 pending receipt/registry，
再以 SessionRead 的 terminal identity 收敛；没有总事件 replay 或 feature listener 名单。

M5b 到 M9 之间还有一只明确不是终态的 private `DEPRECATED(EXTERNAL)` old-commit。它只服务上表四个
exact stable consumer，不开放新 Service、registry 或订阅入口。M5b 删除整个 after-turn phase 时，把当前
`_BuildTurnCommittedModule` 与 `_FanoutTurnCommittedModule` 的字段构造和 fanout 语义原样收进这一只
private function；concrete Agent 在 sessions 保存完成且 REPLY_OUTPUT settle 后、save notice 前恰好调用
一次。function 只得到同一 Turn 的 frozen typed owner return，不允许回查 ambient ctx 或改变它们；它仍生成完整旧
`TurnCommitted`，先执行原 EventBus fanout、再执行 `AFTER_TURN_COMMITTED`，listener 失败或取消仍阻止
后续 delivery，和当前顺序一致。固定 oracle 比较全部字段、deep copy、事件次数、listener 顺序、异常、
取消、Session write set 与 delivery 是否发生。它不双发 `TurnSaved`，也不成为第二名状态 owner。

这只 function 是阶段性外部迁移债务，不是插件或公共能力，因此 M8 只能称为 Core migration stop，不能称为
最终架构。M9 依次让
Emotion、GitHub Watch、Observe 和 Proactive Feedback 改读 `SaveResult`、`SessionRead` 与各 owner 的小 fact：
Emotion、GitHub Watch、Proactive Feedback 注册 ordinary SavePart；Akasha 已在 M6c 注册第四只仓库内
SavePart；Observe 只注册 typed observer，绝不进入 delivery gate。每只 SavePart 对不适用的 skipped/saved
结果显式 no-op，不要求 Core 按插件名筛选；
最后一名 live consumer 重装并通过后，在同一批删除 adapter、`TurnCommitted`、
`AFTER_TURN_COMMITTED`、EventBus event 和旧测试。任何新 consumer 都是 Gate failure。

| consumer | saved | skipped | failed/cancelled/interrupted |
|---|---|---|---|
| Akasha | durable enqueue，SessionRead 投影 | abort prepared feedback | TurnEnded abort |
| Emotion | InputBatch + TurnView 更新情绪；有 persisted user 才更新 presence | 用 InputBatch 处理 explicit quote，不更新 presence | TurnEnded abort InputBatch |
| GitHub Watch | 清理 job/checkout ledger | 同样清理 | 现有 TTL/boot cleanup |
| Proactive Feedback | 仅同时有 persisted user/assistant 时 durable enqueue，否则 no-op | no-op | no-op |
| Observe | live SaveResult 写一次可丢 trace，不是 SavePart | 同左，persisted ids/content 为 None | 不伪造 completed trace |

这四只 live consumer 的相对顺序变化是明确批准的 migration delta，不伪装成旧事件字节等价。对 1.4
锁定的 stable artifact 做只读源码审计得到：Emotion 只写自己的情绪 DB，并以自己的 source key 幂等；
GitHub Watch 只写自己的 job/checkout ledger；Observe 只写自己的 observe DB；Proactive Feedback 在 handler
内先 durable enqueue，boot 再发现未完成输入。四者没有互相 import、读取对方表或共享事务，当前注册先后
只是 EventBus 的偶然顺序。M6c 后固定为“全部未迁 legacy consumer 先完成，再并发运行全部已迁
SavePart，最后 delivery”；legacy consumer 仍保留原注册顺序和 fail-fast，SavePart 组则全部终结并聚合
错误。saved 时一个新 part 失败，兄弟可能已 durable 接受工作，但 notice 仍 pending、delivery 仍为零；
skipped 时同样阻止 delivery，但不声称进程 crash 后 replay。这项
error-cut 变化也是已批准 delta，因为这些 owner 没有共享状态且都必须幂等。因此迁移期间结果不依赖旧/新
组间顺序，但每个 SavePart 都必须先 durable 接受工作或提交自己的幂等 receipt 再返回。Observe 不是
SavePart，继续保留现有非阻塞 writer 与 queue-drop 行为。每个 M9h 子批的 live source 复查若发现
跨插件状态或顺序依赖，整个 committed seam 必须一次切换，不能继续逐个激活。

Emotion 的 live handler 还有两项不能由“读已保存正文”含糊替代：它以全部本轮 input content 识别
`【你当前新消息】`，并要求 Wake selection 在 Turn consumer 运行时尚可读取
（stable `emotion/plugin.py:145-190`、`runtime.py:146-180`）。最终 Emotion 在 provider/tool 前 durable prepare
完整 InputBatch；SavePart.accept(saved) 用 InputBatch + TurnView 保持相同 aggregate content、client/user/
assistant identity、channel 与 time，accept(skipped) 用 InputBatch 继续处理 explicit quote，但不更新 presence。
它还以 `observe` 消费 TurnEnded：failed/cancelled/interrupted 或任何 save=none terminal 都 abort InputBatch prepare；live
observer failure 由 durable receipt 在下次 ready 修复。它的 ready 扫描自己的 pending InputBatch receipt，
并用 SessionRead.status 的 saved/skipped/terminal 收敛；因此没有 partial aggregate，也不需要把 input
content 塞进 sessions 或 TurnSaved。
Wake 继续在 `TurnWait.result()` 返回后才 settle 自己的 selection，而 SAVE_NOTICE.send 在
run 返回前完成，所以 selection 时序不变。M9h1 oracle 必须覆盖 multi-input、SaveChoice.user=false、普通
conversation 与 Wake selected/unselected；不能把 input content 或 Wake selection 塞进 TurnSaved。

旧 metadata 三组键分别这样结束：

- Citation 的 `cited_memory_ids` 变成 Citation 私有的 immutable `MemoryIds`，不进入 Core public
  API。Citation 用 system section 引导模型，并向普通 `reply-output` 注册只认自己协议的 ReplyPart；
  它与 Meme 都看同一份 frozen raw reply，只返回不重叠 ReplyMark，不能看到或改写对方结果。Citation
  在 open 时把 MemoryIds prepare 到自己的 crash journal，`reply-output` 在 Session save 后以 SaveResult
  settle；只有 settle 完成后 sessions 才发布 save notice。boot 时用窄
  `SessionRead` 收敛 pending。M9 从旧
  assistant metadata 一次性导入历史 Citation rows；之后只读 Citation ledger，旧 key 零 consumer，不留
  dual read。Core 不认识 citation，也不新增通用 data bag。
- `skip_memory_retrieval` 不再是 durable Session metadata。GitHub Watch 每次 start Turn 时传
  immutable `TurnRequest.skip_parts`，由 AGENTS/concrete Agent 原样传成 `CONTEXT_INPUT.build` 独立参数，
  只关闭 memory 拥有的 context part；`context-input`
  只按该 call 过滤本轮 part。M3e 到 M9 之间，只允许一只 `DEPRECATED(EXTERNAL)` adapter 只读旧 metadata
  并映射到本 Turn 的 `skip_parts`；新代码不得再写该 metadata，也不得让后续 Turn 继承。
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

最小 Agent 服务骨架是九块；不安装技能时，它仍能完成一次 Turn。Akashic 默认产品再用普通
`skills`、`skill-files` 与 `skill-use` 拼上技能能力。数字不是目标，每块只做一件事才是目标：

```text
┌──────────┐  保存故事      ┌──────────┐  选择大脑
│ sessions │               │ models   │
└──────────┘               └──────────┘

┌──────────┐  使用工具      ┌───────────────┐  只拼系统提示
│ tools    │               │ system-prompt  │
└──────────┘               └───────────────┘

┌───────────────┐  只放本轮临时参考纸；不改故事，也不写回故事书
│ context-input │
└───────────────┘

┌─────────────┐  把完整故事装进大脑这次装得下的小包
│ provider-input │
└─────────────┘

┌──────────────┐  把大脑写出的隐藏小纸条取下，再得到要保存和发送的同一份答案
│ reply-output │
└──────────────┘

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

技能像另装的一盒卡片，不是火车发动机：

```text
host-check ──► skill-files ──► skills ──► skill-use
  看主机有啥       找卡片          管目录       把目录说明、选中的卡片和 load-skill
                                        分别接到 system/context/tools
```

Drift 的卡片不混进 Agent 的普通技能目录；今天没有代码会在运行时读取它们，所以 `skill-files` 只保留
它们原有的检查和软链接投影，不凭空造第二只 registry。

还有一个很小的门卫，但门卫不是特权 Agent 积木。门卫不认识故事、大脑或工具；它只给每项工作
一张不透明号码牌，锁住这项工作使用的同一代积木，并记住“有人按停止时通知哪项工作”。演完后
号码牌和锁一起归还。这样一次 ReAct Turn 全程走同一 snapshot，旧工作仍能被停止，盒内每块积木
仍然平等。定时器醒来只能拿回自己的旧盒；旧盒已经退休就明确失败，不能偷偷换新盒。

`agent-loop` 是唯一把系统说明、故事、临时参考纸和当前消息按固定次序放到桌上的孩子。大脑答完后，
`reply-output` 让每个协议插件只圈出自己认识的字；所有圈都基于同一张原稿，圈重叠就报错，不能一个
接一个改。其他积木不能伸手改别人的纸，也不能要求一个 `before-step` 插槽。

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
| Session 拥有什么 | append/commit：`packages/core/session/src/index.ts:567-653`；model-facing projection：同文件 `:699-745` | `sessions` 独占 persistent history 与 prompt history 派生 |
| Session projection 是什么 | 纯 `init + apply` fold：`packages/session/session-projection/src/index.ts:34-85`；只消费 committed event：同文件 `:169-211` | `session-view` 若存在只是可选通用 fold，不是 prompt history owner |
| Compaction 怎样进入 | DSH 在窄 `agent/pre-step` 检查压力：`packages/compaction/compaction-basic/src/index.ts:148-166`；另在 region 追加带 `surfaceOp: replace` 的 user/message：`packages/compaction/compaction-basic/src/region.ts:436-475` | Akashic 不复制 before-step；以独立 `provider-input` provider 保留现有 append-only message + compaction ledger 边界 |
| 谁驱动每次模型尝试 | DSH 的具体 Agent loop 创建请求、调用模型、处理错误并继续工具循环：`packages/core/agent-loop/src/agent.ts:341-438`；prepared call 钉住同一 adapter generation：`packages/llm/llm/src/index.ts:882-935` | `agent-loop` 仍驱动 attempt；`provider-input` 只对每次 attempt 做 build/settle，不成为第二个 loop |
| Prompt 如何扩展 | section/context 是不同类型、不同 registry 和不同输出：`packages/core/system-prompt/src/index.ts:52-84,354-385,424-476,536-610`；最终组合由 loop 完成：`packages/core/agent-loop/src/agent.ts:234-251,341-359,444-543` | `system-prompt` 与 `context-input` 分别拥有一条 lane；固定 envelope 顺序属于 `agent-loop`；不复制 tools/variables 或可改全体 assembly 的逃生 hook |
| Skills 怎样共享 | 独立 `skills` Service Definition 合并 provider 并提供 list/get：`packages/skill/skill/src/index.ts:1-13,285-298,347-392,464-500`；filesystem 是注入它的普通 provider：`packages/skill/skill-filesystem/src/index.ts:1-9,129-146`；tool-skill 再组合 agents/tools/skills：`packages/skill/tool-skill/README.zh.md:28` | `skills` 独占一份 Root-local catalog；`skill-files` 只提供来源；system catalog、active skills 与 load-skill 消费同一 catalog，不从 Core snapshot ambient 读取 |
| Tool Search 如何涌现 | 具体工具通过普通 registry 注册：`packages/core/tools/src/index.ts:1022-1053`；progressive disclosure 替换 scoped restriction：`docs/cookbook/extension-cookbook.md:100-114` | Tool Search 只是 `tools` scoped view 的普通 consumer；Core 无元工具特判 |
| Agent 和 loop 怎样分 | registry/factory：`packages/core/agent/src/index.ts:235-247,352-422`；具体 Agent 与完整 Turn/Step：`packages/core/agent-loop/src/index.ts:612-640`、`packages/core/agent-loop/src/agent.ts:254-438` | `agents` 只管合同和 factory；`agent-loop` 管具体 Agent 的整个生命周期 |
| 是否需要 before/after phase | 只有窄 `agent/pre-step`：`packages/core/agent/src/runtime-types.ts:55-63,226-238`；无 `agent/after-step` | 不创建替代 before/after 套件；当前 `before_step` 无生产 consumer，直接删除 |
| 模型后是否任意改 reply | assistant 结果直接追加为事实：`packages/core/agent-loop/src/agent.ts:410-427` | 不创建有序 `ReplyEdit`；隐藏协议只经普通 reply-output 的同源不重叠 ReplyMark 解码，最终正文/媒体同时持久与发送 |
| 事实怎样观察 | DSH 的 Session observer 隔离失败 `packages/core/session/src/index.ts:63-74`，parallel checkpoint 等全部 listener `:76-82` | InputBatch/SavePart 是必须等完的 business checkpoint；Observe telemetry 用 observe 隔离失败；sessions 返回 SaveResult，saved 才有 durable outbox；无总 Turn bag |

DSH 不是需要逐字复制的模板。Akashic 保留五项有证据的差异。前四项是 SQLite 原子事务、完整 Root 的
exact snapshot lease/跨进程 ServiceHold、Channel delivery/ACK，以及不 UPDATE/DELETE 旧 message 的 compaction ledger + `provider-input`。每项差异都只保留
现有 owner 与安全不变量，不引入特权 Agent 接口。

还有一项由持久边界决定的窄差异：DSH 明确要求 model-visible 内容进入 Session log
（`docs/architecture.md:103-107`），所以 dynamic context 经 `preStep` 后追加为 durable `user/message`
（`packages/core/agent-loop/src/agent.ts:234-251,288-296`）。Akashic 的临时检索和 runtime view 无权
反写 `sessions.db/messages`，因此 `context-input` 只生成本 Turn 冻结的临时 Message，不持久化、不经
Inbox；若输入本来就是需要恢复的用户 Message，才由具体 Agent 的 inbox 与 sessions 正常提交。

`provider-input` 的 Akashic 证据是：`session/manager.py:409-462` 已从只追加 Session row 构造完整
prompt history units；`agent/core/passive_turn.py:2698-2860` 证明每次 provider call 都必须 prepare/settle，
overflow 还会在同一 call 强制重建；`agent/core/passive_turn.py:2258-2266` 证明闭合 tool batch 是下一次
输入的必要状态；`agent/plugin_composition/request_projection.py:69-141` 与
`plugins/compaction/plugin.py:87-286` 证明“prompt history + turn-local progress → 有限 provider input →
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
| RuntimeSnapshot、Root sealing、stable/latest、lease、candidate closure | 原样作为 publication 真源；补 `ServiceCall`、`ServiceHold`、`RootScope`、`TaskControl`，并把已证明的五类跨代共享 owner 收敛到 `RootSwitch` |
| `TOOL_CATALOG`、`PluginTools`、工具 snapshot freeze | 演进为 `tools` 插件的唯一 registry，不创建平行 ToolRegistry |
| 现有 `plugins/models` Services | 直接作为 `models` 基础插件，不复制 provider/model catalog |
| 现有 compaction/markdown-memory 普通插件 | 保留持久 owner；`PROVIDER_REQUEST_PROJECTION` 另行判为 `move`，不把特制 request gate 当终态 Service |
| SessionManager/SessionStore 的事务和恢复算法 | 行为与测试资产保留，真实实现迁入 `sessions` owner；不包旧 singleton |
| `PluginScopedTurns` 的 exact root、accepted handle、retired error 语义 | 领域中性部分迁入 `RootScope`/`TaskControl`；旧 `SCOPED_TURNS` key/bridge 最终删除 |
| existing ActivityHost/admission-drain 模式 | 用作 `TaskControl` 与 `RootSwitch` 的实现证据，不复制 Agent 专用 publication plane |
| `PluginSkillHost`、`PluginSkillLinker`、RuntimeSnapshot skill 字段与 `SkillsLoader` 资产 | 扫描、frontmatter、可用性、冻结副本、precedence、link journal 和 Gate 行为迁入普通 `skills` + `skill-files`；删除 Core catalog owner、skill publication branch 与 ambient snapshot lookup |
| committed fact 的发布语义 | sessions 总写 turn_saves；saved 同事务写 pending notice；reply settle 后 SAVE_NOTICE 运行 SavePart，再 live observe 同一 SaveResult；telemetry failure 不阻止 delivery |
| mutable phase ctx、metadata bag、编码 helper | 标成 `move`，Core 内部先停用；外部 consumer 迁完后删除 |
| bootstrap AgentLoop/SessionManager/ToolRegistry construction 与 manager Core-service manufacturing | deprecated 后退役；它们是待删除 owner，不是可长期复用 adapter |

## 4. 最终能力与唯一 owner

### 4.1 Core publication kernel

Core 只保留：

- 插件 artifact、generation 和完整 Root 的构建、验证、发布、丢弃与恢复；
- stable/latest 指针、exact lease、retire/drain 和 Effect cleanup；
- 绑定单一 `ServiceKey[T]` 的 `ServiceCall[T].call(action) -> R`；
- 同样绑定单一 ServiceKey、把 exact Root 跨进程 pin 到业务 receipt 结束的 `ServiceHold[T]`；
- 每个 Fiber 平等取得的 `RootScope`，以及按 service key 隔离的
  `TaskControl` 与窄 `TaskStart`/`TaskCancel`/`TaskWait`；
- 只协调不能跨代并存的共享 owner 的 `RootSwitch`；
- composition diagnostics、最小 workspace file grant 和外部 host 的通用资源开关。

kernel 在 bootstrap composition 时为外部 host 创建绑定一个 exact `ServiceKey` 和固定 lease source 的
`ServiceCall`；host 不取得任意 service lookup，插件也不能创建 `ServiceCall`。普通 host 的 lease
source 永远取得 stable；公开 `call(action)` 不接受 selector、snapshot ID、plugin ID 或 lease。
attached validation child 只使用 Core 根据父 Turn、candidate generation/source identity 铸造的一次性
exact lease，不能由 host 或插件选择 latest。`ServiceCall` 绑定当前 task，从 exact Root
`require(bound_key)`，完整等待 action，再解除绑定并释放。Service 缺失、Root/identity 不一致、
继承到错误 task 或 lease 已退休全部 fail-loud。它不解析 request，不创建 background task，也不
捕获领域错误。

`RootSwitch` 的 public contract 只有 `ROOT_SWITCH.add(ctx, SwitchPart) -> Effect`。SwitchPart 声明唯一
`name`，并只用注册它的普通插件闭包持有的窄 grant 实现 `stop`、`leave`、`enter`、`start` 与
`recover`。每个动作只改变这一个 part 自己的共享 owner：stop 停止 admission 并 drain，leave 撤掉本代
claim/projection，enter 安装本代 claim/projection，start 让本代 owner ready。它不接收 request、Message、
plugin config bag、任意 Service lookup 或对方 Root 的 grant。Root sealing 冻结 registry，重复 name
拒绝 candidate。只有同时满足“跨 Root 共享一名 live owner、old/new 不能安全共存、lease 不能恢复共享
状态”的 owner 才能注册；普通 cache、catalog reader、MCP process 或 background task 不得借此进入
publication plane。
同一已安装插件必须在自己的所有 generation 持续注册同名 part；不能在插件仍存在时临时新增或移除。
一名 plugin owner 只能注册一个 part；需要第二个 part 就必须安装第二名独立 owner。
新增 part 通过安装独立 owner，移除 part 通过移除整个 owner。owner 转移时 old/new 两边提供同名 part，且
同批移除整个 old owner、安装整个 new owner；两名 owner 都继续安装时拒绝转移。
这样 absent tombstone 始终表示 owner 整体不存在，不会把仍提供其他能力的普通插件误删。

Core 从 old/new 两棵已 seal registry 构造 closed plan；part 不产生 callback bag。candidate validation
不调用任何动作。closed plan 只包含 old/new part owner identity 不同的 name；identity 同时含 owner artifact
与 generation，复用同一 owner 的 part 不切换。publication gate 先阻止新 lease，再等待 plan 中每只 old part
owner generation 的 lease_count 与 hold_count 都归零；RuntimeSnapshotLease 与 ServiceHold 分别为 live
snapshot 和 durable receipt 引用的每个 generation 计数，所以这会
覆盖“旧 Turn 尚未走到该 part”的未来调用，而不必把普通无共享状态插件一起 drain。相关 owner generation
zero lease/hold 前不得调用它的 stop/leave，也不得 commit 新 stable。全部 changed owner 都 quiescent 后，Core 才
写 publication journal，再按 name 执行所有 old stop、old leave、new enter、new start；start 完成时 service
仍被 gate 关住，不能接到请求。全部成功后，Core 才在同一份 crash-safe
publication record 中原子写入新 stable identity 与 `use_new` choice，然后开放 lease。任何一步失败都在
gate 仍关闭时按完成动作的相反顺序停止/离开 new、重新 enter/start old；取消不能截断恢复，forward 与
reverse failure 全部聚合并保留 part/resource label。旧 stable 在选择新边前始终是唯一 committed
pointer，因此 start 失败不会留下“pointer 已新、owner 仍旧”的半状态。

Core publication journal 是跨代恢复的唯一 transition owner。它只在全部 changed old part owner generation
zero lease/hold 后、第一项动作前写入 old/new snapshot identity、两边每个 part 是否存在、part name、owner
artifact/generation identity、当前 step 与 `use_new`，并 pin 两边 immutable code、私有 exact config file 和普通
Service dependency closure，直到 journal 清理完成。config value 与 secret 不进入 publication journal；Core 只在
`0700` 私有目录保存 exact file，文件为 `0600`，journal 只保存 path、hash 与 revision。恢复仍经正常 config parser
读取所选 file；path、hash 或权限漂移都保持 degraded。`switch_choices` 每个 part name
只保留一条长期 choice：
selected ref 或 absence tombstone；清掉 transition record 后它仍是下次 boot 的 stable selector，不能重新让
可变磁盘目录裁决当前版本。只有新 publication 覆盖该 name 后，旧 pin 才可减少。进程崩溃后、开放任何 lease
前，Core 按 journal 重建两边所引用 part 的 fresh recovery-only Root：选择新边前一律收敛到 old
active/new inactive，选择新边后收敛到 new
active/old inactive。install 后 pointer 前崩溃仍能找到 new part；remove 后崩溃仍能找到 old part；replace
能同时找到两边。selected snapshot lineage、完整 ref、generation、path、source type、config revision 与
每个 owner 全部 Fiber 的依赖 closure 必须一致。Activity 在 M1d 迁成 SwitchPart 前，它的 publication 与 RootSwitch change 不能出现在
同一批；RootSwitch restore failure 的同进程 retry 保持 admission 关闭并要求 restart，不能绕过 durable choice。
part 可用自己的资源 journal实现幂等 recover，但 Core 不解释其内容。artifact pin
缺失、identity 不符或任一 recover 失败时 runtime 保持 degraded 且不开放 lease，不能只信当前 stable
Root 里恰好还存在的 part。

这不是通用 transaction hook：没有 arbitrary phase、priority、waterfall、request rewrite 或可注册
undo callback；五个动作只能收敛该 part 声明的一名跨代共享 owner。0036 要求“第 4 名真实 consumer
出现后再提窄协议”；PluginSkillLinker 的共享 symlink + journal 已满足该门槛。进一步审查证明
`sessions` 的单 SQLite writer 也是第 5 名 consumer。M1d 先 cold start Activity，M3c 再迁 skill link，M6 迁
sessions，M8a～M8b 再把 Channel 和 command 私有分支逐名迁入同一 registry，M8c 删除硬编码 participant table。

### 4.2 最小普通插件图

| 插件 | 独占事实或变化轴 | 公开能力 | 明确不拥有 |
|---|---|---|---|
| `sessions` | Session/Message/Turn 的 SQLite 事实、事务，以及 runtime history view、prompt history 与 save notice outbox | `SESSIONS`: read、history_views、append、save、size、finish、result、recover；finish/result 只提交/读取 source-neutral AgentResult，recover 只把 crash 遗留 queued/in-progress Turn 收敛为 cancelled/interrupted；`SAVE_NOTICE`: send、ready；向 RootSwitch 注册唯一 SQLite writer part；附件/delivery 只留有已证明跨表事务的窄端口 | system prompt、模型、工具、Channel 发送、任意删除 |
| `models` | provider/model registry、冻结执行绑定与流式调用 | 复用 `MODEL_DRIVERS`、`CHAT_MODELS`、`EMBEDDINGS`、catalog/settings | Session metadata、模型选择 policy、system prompt、loop |
| `tools` | 工具定义、scoped view、调用与结算 | `TOOLS`: register、view、run；`run` 返回 closed `ToolUse`，其中含中立 `ToolOutcome` | system prompt 文案、Session SQL、Tool Search 特制 grant/unlock、任意 fact/delivery bag |
| `system-prompt` | 有序 system section registry | `SYSTEM_PROMPT`: add、build、render | prompt history、provider 调用、记忆文件、任意 reply 改写 |
| `context-input` | Root-local transient context Message registry | `CONTEXT_INPUT`: add、build | system text、Session 写入、prompt history 改写、工具、provider 调用 |
| `provider-input` | 每次 provider attempt 的有限不可变输入与结算 | `PROVIDER_INPUT`: open、build、settle；一个 Root 恰有一个 basic 或 compaction provider | prompt history、envelope 顺序、current Message 位置、system section/tool registry、provider 调用、通用 middleware |
| `reply-output` | provider 最终原文到持久/发送同一回复的 typed 解码与结算 | `REPLY_OUTPUT`: add、open、settle；同一 raw reply 上的不重叠 ReplyMark | Session commit、Channel delivery、按顺序改写、任意 metadata bag、feature ledger |
| `agents` | 公开 Agent 合同、live registry、`TurnSource` 校验/冻结和 factory slot | `AGENTS`: create、resume、get；register factory；Agent accept/run/finish contract | 具体 inbox/Turn/Step、cancel/terminal 实现、ReAct、模型、工具 |
| `agent-loop` | 默认具体 Agent 的完整生命周期 | 向 `AGENTS` 注册默认 factory；拥有 inbox、Turn/Step、cancel/terminal、TurnEnded 和 provider/tool loop | 持久 writer、发送、来源枚举、业务插件名 |
| `session-view`（可选） | 已提交 Session fact 的纯同步 fold | `SESSION_VIEW`: register、state、snapshot | prompt history、I/O、Session 回写、命令、发送 |

`ToolOutcome` 在实现前锁定为 `ToolOutcome(call_id, status, content)`。`status` 只有
`done|failed|cancelled`；`content` 只是给模型看的 immutable content tuple，只允许普通文本与已验证的
typed `MediaItem`。它没有 `facts`、`metadata`、delivery、attention、callback、任意插件数据或打开文件的
能力。`TOOLS.run` 返回 closed `ToolUse(turn_id, name, args_json, outcome)`；validated args 只按固定 immutable
JSON 编码一次，outcome 就是同一次调用的原对象。agent-loop 对所有工具只读取 `use.outcome.status/content`，
并把收到的 exact ToolUse tuple 累积进最终 ReplyCall；因此 Citation fallback 的业务输入不依赖可丢 observer。
tools 还以 typed `observe` 发布**同一个** ToolUse 对象，observer failure 不能删掉或改写已经返回的对象。
tools 可以在自己的私有 execution trace 中保存同一参数、content 与结算状态，但不能把 trace receipt 变成
agent-loop 需要解释的公开袋子。其中的 `MediaItem` 只走统一 media 收集，不按 tool name 或 content type
之外的 feature fact 分支。这个
合同有意比 DSH 更窄：DSH 投给模型的 projection 同样只取 content/isError，但自己的 durable result 还可带
presentation meta，execution 内部也有其他字段；Akashic 不复制这些字段进公开 ToolOutcome。工具自己的
外部效果由工具 owner 调窄 Service 完成（`packages/core/agent-loop/src/tool-calls.ts:268-289`；
`packages/core/tools/src/index.ts:282-295,548-573`）。

models 的每次 terminal call 返回 deeply immutable
`ModelReply(text, tool_calls, thinking, finish, continuation, use)`；`use` 是同一次
`ModelUse(turn_id, call_id, choice, usage)`，没有第二份 choice/usage。agent-loop 直接用 ModelReply 继续
ReAct 并保留 exact `use` tuple；models 以 typed `observe` 发布同一个 use，observer failure 不能删掉或改写
已经返回的 reply。stream delta 仍由 models/Channel 的现有窄 sink 发送，不塞进 ModelReply。

通用 `ToolInput` 也只含每只工具都成立的 validated args、call/turn/source identity 与 cancellation；没有
send mode、commit role、delivery policy 或 feature option。某只 Tool 需要的发送 policy 必须在该 Tool 注册时
由 dependency-bound narrow sender 闭包固定，不能由 agent-loop 按工具名填写。

`skills` 不是 Agent spine 的必需依赖，而是默认产品装配中的普通 registry。它的 public contract 只有：

```text
SKILLS.add(ctx, SkillProvider) ──► Effect
await SKILLS.open(SkillCall)    ──► SkillView
SkillView.list()                ──► tuple[SkillInfo, ...]
await SkillView.get(name)       ──► SkillBody | None
```

容器和本机对 `available/missing` 的判断走另一只普通、窄而 source-neutral 的 `host-check` service：

```text
await HOST_CHECK.check(HostNeed) ──► HostState
```

`HostNeed` 只含 bin names 与 env names；`HostState` 只把这些名字严格分成 available/missing，不返回 env
value、PATH、任意 shell 或文件权限。local provider 用 login PATH/env，bridge provider 复用现有带 boot、token、
release 与 toolchain identity 的 `SkillRequirements` RPC；一个 Root 恰有一个 provider，RPC、身份或 partition
失败全部 fail-loud。`skill-files` 只 inject HOST_CHECK，不 import host factory/client。这是 Akashic 容器边界
比 DSH 本机 filesystem provider 多出的普通 adapter，不是 Agent spine 或 skill 特权。

`SkillCall` 只含 turn identity 与 cancellation；workspace/root grant 由 provider 在注册时闭包持有，不能经
call 传任意路径或 publication identity。`SkillProvider` 只公开唯一 provider `name`、
`async list(SkillCall) -> tuple[SkillItem, ...]` 与
`async get(SkillItem, SkillCall) -> SkillBody | None`。`SkillItem` 声明整数 `rank`；低 rank 胜出，相同 rank
的同名 skill fail-loud，不能靠加载顺序或静默覆盖。`SkillView` 冻结一次 list 的 winning items 和当前
Root 的 `catalog_id`；list 是纯读取，get 只把 view 内保存的 opaque winning item 交回原 provider，不重新
选 owner。

Root sealing 只冻结 provider set，不冻结 workspace 文件或实时依赖。每次现有 consumer 原来会重扫时，
迁移后仍新开一只 `SkillView`；所以下一 Turn 以及同 Turn 后续原有 lookup 仍能看见 workspace skill、PATH
或 env 的变化。这里不偷偷增加“整 Turn 技能缓存”。安装插件的 provider 则只读取 generation-private
copy，旧 Root 全程保持旧 artifact。这个边界与 DSH 相同：registry 冻结 provider registration，filesystem
provider 在 list/get 时观察 cwd/source，并可 invalidate，而不是在 Root seal 时永久冻结 catalog
（`packages/skill/skill/src/index.ts:271-298,347-392,464-490`；
`packages/skill/skill-filesystem/src/index.ts:55-73,129-146`）。

`SkillInfo` 保留 name、display name、description、when-to-use、always、available/missing、provider、source
与 `source_id`；`SkillBody` 再保留 text、只读 resource root、同一 provider/source identity 和
`catalog_id`。load-skill 的可观察 provenance 继续输出 `pluginId`、`skillCatalogGenerationId` 与
`runtimeSnapshotId`：前两项来自 SkillBody，最后一项由拥有 exact execution scope 的当前 tool boundary
统一 stamp，SkillProvider/tool 都看不到 RootRef，也不能 ambient 读取 RuntimeSnapshot。candidate Gate 也只
验证这只 typed provenance 对应 exact candidate Root/provider；
`PluginManager` 不再扫描 skill roots 或解释任意 provenance dict。迁移 oracle 固定比较上述字段。

`skill-files` 是普通 provider 插件，不是 Core helper：workspace root、产品 bundled root 与插件
generation-private root 分别成为显式 provider contribution，并保持现有 workspace → plugin → builtin
precedence。现有 `workspace/skills`、`workspace/drift/skills` symlink 与
`runtime/plugin-skill-links.json` 不暗删；它们迁给 `skill-files`，继续只是当前 generation 的可重建
projection/recovery journal。`skill-files` 以窄 workspace grant 参加 generic publication transaction，拥有
prepare/commit/recover、冲突拒绝和完整性证据；Core publication 只看普通 check/result，不识别 skill path。
catalog descriptions、body hashes 与 candidate skill use 同样由 skills/skill-files 产 typed check，
PluginManager status/Gate 不再扫描 roots。

文件来源通过 `SKILL_FILES.add(ctx, SkillRoot) -> Effect` 注册。SkillRoot 只有受保护 root grant 和
`group: agent|drift`；source/plugin identity 由 ctx 提供，caller 不能伪造。agent group 进入 skill-files
注册给 SKILLS 的 normal provider；drift group 只保留现有 Drift catalog check 与 link projection，因为当前
代码和 hua-home artifact 都没有 runtime drift list/get consumer。没有现实 consumer 就不预建
`SkillSet`、`DRIFT_SKILLS` 或 arbitrary scope factory。

普通 `skill-use` 是无状态产品插件：它只 inject `SKILLS`，再用三个 dependency-bound child
effect 分别在 `CONTEXT_INPUT`、`SYSTEM_PROMPT` 与 `TOOLS` 存在时注册 active-skills part、catalog section
和 load-skill tool；没有 `get()` fallback，也不让 Agent spine 携带 skill 状态。M3b 只建立 `skills`，
M3c 建 `skill-files` 并让三个旧 consumer 显式读取 SKILLS；M3e 首次挂 `skill-use` 的 context child，M3f
挂 prompt child，M4 挂 tool child，每批同时删除对应旧 consumer。

现有外部 `skill_roots` 与 `drift_skill_roots` 到 M9 前分别只允许一只 exact-generation
`DEPRECATED(EXTERNAL)` bridge 向 SKILL_FILES 注册 agent/drift SkillRoot；新内部插件不得使用旧声明。
M9 把当时 live 外部源码改成直接注册后，删除两只 bridge、manifest 字段和 Core/PluginManager 解析。未来
只有真实 runtime drift consumer 出现并另行证明时，才从普通 `drift` owner 增加读取 seam。

`system-prompt` 的 public contract 只有：

```text
SYSTEM_PROMPT.add(ctx, PromptSection)  ──► Effect
await SYSTEM_PROMPT.build(
    PromptCall,
    skip_sections=frozenset[str],
) ──► PromptSet
SYSTEM_PROMPT.render(tuple[PromptText, ...]) ──► str
```

`PromptSection` 只公开唯一 `name`、整数展示 `order`、`DropLevel drop` 与
`async build(PromptCall) -> str | None`。
`PromptCall` 只冻结 `session_key`、`turn_id`、channel 和 chat identity；没有 skip sections、workspace grant、prompt history、current
Message、context Message、tools、provider request、mutable bag 或任意 rewrite callback。service 按
`(order, name)` 构建非空 immutable `PromptText(name, order, drop, text)`；重复 name 在 Root sealing 失败。
closed `PromptSet(items, size)` 保存这次 immutable PromptText tuple 与同一次 section breakdown 计算出的
`PromptSize`；agent-loop 直接保留 `size`，system-prompt 以 typed observe 发布**同一个**值，observer failure
不影响返回值。它没有 history、context、tools 或任意 fact bag。
任一 section 失败或取消时整次 build 失败，不向模型交付 partial tuple。`skip_sections` 只有 service
看见；它校验 skip name 语法，在完整 registry 验名后、调用 section build 前过滤 matched name；unknown
name 是“该 section 已不存在”的幂等 no-op，被跳过的 section 不得读文件或 ledger。PromptSection 看不见名单。
每个 section 所需的 workspace grant、ledger 或其他资源只能由
注册它的普通插件自己注入并闭包持有，不能从 `PromptCall` 借用调用者能力。registry 是 Root-local Effect，
在 `snapshot.sealing` 冻结。每个 Turn 恰好 await build 一次，PromptSet 随 Turn 冻结；provider retry 与
同 Turn 的后续 tool call 都不重新 build。`render` 是无 I/O、无状态的纯 join，只按 tuple 现有顺序输出；
它不筛选、不重排、不读取 registry。

`context-input` 的 public contract 只有：

```text
CONTEXT_INPUT.add(ctx, ContextPart)     ──► Effect
await CONTEXT_INPUT.build(
    ContextCall,
    context_texts=tuple[ContextText, ...],
    skip_parts=frozenset[str],
) ──► tuple[ContextMessage, ...]
```

`ContextPart` 冻结唯一 `name`、整数展示 `order`、`DropLevel drop` 与
`async build(ContextCall) -> str | None`。`ContextText` 只冻结 `name`、order、非空 text 与 drop；来源在
`TurnRequest.context_texts` 里传入，不注册临时 part。
`ContextCall` 只冻结 `session_key`、`turn_id`、current Message、immutable runtime history view、channel、
chat identity 与 message time；没有 `context_texts`、skip parts、skills、mutable list/dict、slot、metadata bag、callback、waterfall、system sections、tools
或 provider request。每项 part 只能返回自己的 text 或 `None`，不能看见、改写、删除或重排
别的 part 或 ContextText。service 合并 `context_texts` 与 registry part，校验全体 name 唯一，以 name 作为
provider-visible source，固定标记 `trust=derived`、`kind=context`、`role=user`，构造与 current user 分开的
immutable `ContextMessage(name, order, text, trust, role, kind, drop)`，再按 `(order, name)` 顺序构建完整 tuple；
provider adapter 必须把 source/trust 放进模型可见的独立 context block，
不能丢弃或混入 current user。duplicate name、build error、非法 text 或取消都在模型调用前
fail-loud；registry duplicate 在 sealing 失败，ContextText duplicate 或 ContextText/registry 冲突在 build 失败；
skip parts 校验语法，并在完整验名后、调用 ContextPart build 前过滤 matched name；unknown name 是幂等
no-op，被跳过的 part 不得 query，过滤不能掩盖冲突。整次 build 失败且不返回部分结果。`None`
只表示该 part 按自身正常合同本轮不适用，不能掩盖依赖、I/O 或数据损坏。

registry 是 Root-local Effect，在 `snapshot.sealing` 冻结。每 Turn 只 build 一次，得到的 tuple 随该 Turn
冻结，provider retry 不重复 retrieval。`context-input` 不写 Session；需要 durable 的来源事实继续由
来源自己的 ledger/event 保存。`agent-loop` 唯一拥有固定 envelope 顺序：system → prompt history → context
Messages → turn transcript。transcript 以 current Message 开始，再接已发生的 assistant/tool 进展。它只组合
四条 immutable lane，不暴露可插入步骤，因而不是第二条 phase。

Active-skills part 直接以 `ContextCall` 的普通 Message 事实和 skills 插件自己的 catalog/rules 运行现有选择
算法；`ContextCall` 和 concrete loop 都没有 `selected_skills` 字段或 Skills 分支。Wake、Subagent 等
source-only hint 在 `Agent.accept` 后、`Agent.run` 前构造自己的 immutable `ContextText`。领域 `TurnRequest` 携带 typed
`context_texts: tuple[ContextText, ...]`、`skip_sections: frozenset[str]` 与
`skip_parts: frozenset[str]`；conversation/Wake/Subagent 等
source 经 `AGENTS` 把 request 原样交给具体 Agent；`agent-loop` 构造 ContextCall，并把 request 的两字段作为
独立 build 参数传给 `context-input`，同时把 skip_sections 独立传给 `system-prompt`，不解释 name 或内容。ContextPart 只收到 ContextCall，看不见 ContextText
或 skip parts。不得以 ambient `ContextVar`、Message metadata、Session metadata 或 durable Inbox 代替这条
transport。M3e 的 Emotion `DEPRECATED(EXTERNAL)` bridge 只生成 ContextText；GitHub Watch 旧 metadata
adapter 只生成 skip parts。M9 外部源码改为直接填写 TurnRequest 后，两只 adapter 一起删除。

`DropLevel` 只有 `extra`、`repeat`、`keep`。装饰内容使用 extra；可再次生成的 skills、memory、catalog 和
retrieval 使用 repeat；调用所需的规则或 source hint 才能使用 keep。`PromptSection.order` 只控制展示，
name 只控制身份和相同 order 的稳定 tie-break，drop 只控制预算，三轴互不代替。`provider-input` 只能先移除所有 extra，再移除
所有 repeat，最后按完整 prompt history 边界缩窗；keep 不得移除。它看不到 feature 类型，也不能按 name
分支。未超窗时把全部 PromptText 交给 `SYSTEM_PROMPT.render`；扣除 1.3 已批准移到 context lane 的 Akasha
memory section 后，其余每个 system section 的文本、相对顺序与 section 间分隔符必须逐字节保持；
超窗时 `provider-input` 只筛选 tuple，再调用同一个 render。

`TurnRequest(start, message, ...)` 只是一只 immutable Agent call record，不是 transport DTO 或 mutable
lifecycle context；`start` 是 Agent 铸造的 `TurnStart`，`message` 是独立 typed Message，channel/chat/time
都从它读取。所有 source 使用同一只两步握手：

```text
await Agent.accept(session_key, TurnSource) ──► TurnStart
await Agent.run(TurnRequest(start, ...), watch) ──► TurnWait
await TurnWait.input_seal()                 ──► InputSeal | None
async for update in TurnWait.updates()       ──► TurnUpdate  # watch=true only
await TurnWait.result()                      ──► AgentResult
await TurnWait.close()                       ──► None        # source scope finally
await Agent.finish(TurnStart, status)        ──► None   # 仅 run 前 skip/fail/cancel
```

`TurnSource(name, ref)` 的 name 是开放、严格的来源名；ref 是该 source 已有的 durable message/job/attempt
identity。两者必须非空、无首尾空白，只用于幂等 admission 与诊断，任何 consumer 都不得按值改变行为。
具体 `agent-loop` 是 accept/run/finish 的唯一实现 owner：accept 先让 sessions 以 `(name, ref)` 幂等写入
reserved Turn，再以同一 turn/task identity claim TaskControl，最后让 sessions 把 reserve 置为 claimed；
只有三步都成功才返回 deeply immutable `TurnStart(session_key, turn_id, source, accepted_at)`。同 session 已有
task 时沿用现行 interrupt/wait policy，不能绕过 claim。cancel 或非重试错误发生在 claimed 前，accept 自己
先把 reserve terminal failed/cancelled 再抛错；claim 后 mark 失败则 cancel 原 task、terminal reserve，并
聚合 cleanup error。同一 ref 的 reserved/claimed/terminal 重试都返回同一 start；source/session/request identity
不符才 fail-loud。run 只接受由同一 Agent/Root 铸造的 start：claimed 时只能首次执行，terminal 时只能读取
同一 durable AgentResult，其他状态冲突 fail-loud。finish 只把尚未 run 的 start 置为
skipped/failed/cancelled/interrupted，重复同终态幂等，冲突终态 fail-loud。
`AgentOutput(text, thinking, media, attachments, message_id: MessageId | None)` 是 source-neutral final output；
只有 assistant Message 确实保存时，message_id 才等于那一条 persisted Message identity；user-only/no-save
completed Turn 必须为 None，任何 sink 都只用 TurnSource ref/自己的 delivery key 幂等，不能拿 message_id 去重。
media/attachments
只含统一边界已经验证的 immutable item/ref，不能带 channel、chat、reply target、sender、delivery flag、
callback 或 metadata bag。`AgentResult(status, output, items, usage, error, ended_at)` 是 deeply immutable terminal
fact；output 是 `AgentOutput | None`，items 是现有 control/turn wire contract 收窄后的 immutable
`tuple[TurnItem, ...]`，usage 是 typed `TurnUsage | None`，error 是 typed `TurnError | None`，字段组合由 sessions
校验。agent-loop 只能以 `SESSIONS.finish(start, status, output, items, usage, error) -> AgentResult` 取得**已经 durable commit** 的
结果，再交给 TurnWait；sessions 还提供窄 `SESSIONS.result(start) -> AgentResult | None`，只从权威 terminal
Turn row 重建同一事实，不返回 row/repository。现有 `turns.status/items_json/usage_json/final_response/error_json`
继续保存相同 write set；这里把它收窄成 typed read，不新增第二份结果表或 owner。

实时进度不用 callback bag 或总事件。`Agent.run(..., watch: bool)` 先建立并返回 TurnWait，再允许 provider/tool
产生进度；watch=false 时不创建进度队列。watch=true 时 source adapter 必须恰好一次 drain
`TurnWait.updates()`；它是单 consumer、有界、按生产顺序 backpressure 的 live stream，只包含 closed immutable
union `TextUpdate | ItemStart | ItemDone | OutputDone`：

- `TextUpdate(sequence, text, thinking)` 保留现有 `content_delta` 和 `thinking_delta`，两者至少一项非空；
- `ItemStart(sequence, item)` 只带一只 closed TurnItem，active 事实由 ItemStart 本身表达；
- `ItemDone(sequence, item)` 只带同 id、同 kind 的 terminal TurnItem；
- `OutputDone(sequence)` 只表示不会再有可见输出，不表示 Turn 成功、已保存或已 terminal。

sequence 从 1 开始严格递增，横跨四种 update；重复、倒退、start/done 不配对、OutputDone 后再出
可见 update 都是内部合同错误。source 在 drain 前自己投影 turn started，按 update sequence 原样投影，
收到 OutputDone 就在当下投影 output completed，最后以 AgentResult 投影 terminal。OutputDone 的 producer 点
必须一比一保持现有 `TurnOutputCompleted`：正常最后可见输出、安全拦截回复、已知上下文过长回复
与模型超时回复各产生一次，AfterStep/reply/save/AgentResult 可以尚未完成。它不拥有或宣称 input
source lock；该 lock 仍由各 source 自己拥有，三条现有 fallback 不藉 M2 改时序。其他在原事件点前终结的
failure/cancel 不产生 OutputDone。
所有已产生 update 被取走或明确丢弃、且 AgentResult 已 durable 后，stream 才正常关闭。Agent 失败不从
stream 抛第二种错误；唯一 terminal 结果在 `result()` 里。

input lock 不进这条输出 stream。`TurnWait.input_seal()` 是与 updates 正交的 single-consumer、one-shot
rendezvous：concrete Agent 到达现有 input lock 点时产生一只无 payload 的 `InputSeal`，并停在该点。source
adapter 把自己的 active input 状态锁定后调 `seal.done()`，Agent 才继续；锁定失败就调
`seal.fail()`，Agent 以明确 input-lock error 终结，不继续 reply/save/sink。Turn 若在原锁点前已 terminal，
`input_seal()` 返回 None；terminal 重入也返回 None。source 必须在 run 返回 TurnWait 后立即启动这只 wait，
并与 updates/result 并行 drain；watch=false 也不例外。没有可追加 active input 的 source 立即 done。caller 失联或
离开 source scope 时必须在 finally 调幂等 `TurnWait.close()`；close 把 updates 切成 discard，并让当前或未来的
未回执 seal fail。Agent 因而得到同一 input-lock error，不无限等待。close 不伪造 caller cancel；若要取消
Turn，source 仍须先请求 Agent cancel，等 result，再 close。terminal 后 close 是 no-op。

InputSeal 不含 source name、lock callback、registry、listener、字符串 reason 或 metadata；它也不能改写输入。
正常回复保持 `InputSeal → source lock → done → OutputDone`；安全拦截、已知上下文过长和模型超时
保持 `OutputDone → executor return → InputSeal → source lock → done`。这只是把现有 `InputLock` callback
换成 typed one-shot handshake，不修复或改变三条 fallback 的时序。

这不是插件注册点，也不能改写 ReAct。Control 与需要流式 Channel 的 source adapter 直接投影同一
stream；其他 source 传 false。source projection 抛错、下游断开或提前 `aclose()` 时，adapter 原子把 feed
切成 discard，释放已阻塞 producer，记录自己的 preview failure，并继续等 AgentResult 与自己的 final sink；它
不改 AgentResult 或 TaskControl terminal。只有 caller 明确取消 Turn 时，adapter 才先关 feed，再请求
Agent cancel 并等同一 terminal。既有 Control 慢 consumer 错误也只关该 feed，Agent 仍能终结。进程崩溃不重放
live update，terminal 重入得到空 update stream 和 durable AgentResult，Control 仍从 result 取得 exact
items/usage，并用 final output 完成 terminal frame。
这对应 DSH 对 durable Session fact 与 live `agent/*` event 的区分（`docs/architecture.md:64-70`），但 Akashic
把唯一 caller 的流收窄在 TurnWait 上，不开放新的全局 hook。

`TurnItem` 也不是 `kind + dict` 的内部逃生口。agents contract 把现有五种 wire kind 收窄成 closed immutable
union：

| item | 固定字段 |
|---|---|
| `UserItem` | id、`MessageId | None`、ordinal、text、validated media、time |
| `AssistantItem` | id、`AgentOutput` |
| `ThinkingItem` | id、text、`in_progress|completed|failed|cancelled|interrupted` |
| `ToolItem` | id、call id、name、args、`in_progress|success|denied|error|failed|cancelled|interrupted`、`result preview | None`、iteration、`SkillRef | None` |
| `ErrorItem` | id、`TurnError`、`failed|cancelled|interrupted` |

ItemStart/ItemDone 只在当前行为真实有 live lifecycle 时产生；它们不为每个 durable item 伪造一对事件。
Control adapter 继续在自己的协议边界投影 user/assistant item；Channel 不因为 Control wire 需要这两种 frame。

`TurnError(type, message, retryable)` 只有这三项；现有代码的任意 `data` 不进新合同。M2 Gate 同时
扫源码、已安装 artifact 和正式 DB：只有证明 `TurnError.data` 零 producer、零 reader、零持久值才能删；
任一非空值都先阻断 M2，不能静默丢弃或留兼容 bag。`SkillRef(skill_name, plugin_id,
catalog_generation, root_snapshot)` 是唯一工具来源类型，精确承载现有 Skill Loader 生产的四项；空来源是
None，其他 live key 阻断 M2，不恢复 generic provenance map。

工具 args 只使用 closed `JsonValue = null | bool | int | finite float | str | JsonList | JsonMap`。
`JsonList` 只持有 tuple；`JsonMap` 只持有按信任边界已验证顺序冻结的唯一 string-key/value tuple，禁止
重复 key、NaN、infinity、dict/list 引用或用户对象。同一 canonical encoder 递归产出无空白 UTF-8 JSON；保留
边界字段顺序是为了保持现有 `items_json` 字节，不代表可变 map。sessions storage adapter 是 items_json
的唯一 encoder/decoder，Control adapter 是现有 TurnItem wire object 的唯一 serializer；两者保持现有 wire key 和字段顺序。
replyTo、client message identity 与 transport metadata 由 source ledger 自己投影，不进入内部 TurnItem；M2 Gate
必须扫描所有真实 outbound metadata producer，把每个 live key 归到已有 typed source/plugin owner。出现未入账 key
就阻断 M2，不能用 Mapping、assistant_data bag 或旧 OutboundMessage 偷渡。这样现有 Control wire 保持字节
等价，内部合同仍是 source-neutral closed value。

同一 TurnSource ref 的重入也只有一条路：accept 对 terminal Turn 返回原 TurnStart；run 先验证 start/request
identity，发现 terminal 就以 SESSIONS.result 返回已经 resolved 的 TurnWait；updates 为空，provider/tool/save/observer call
全部为 0。只有 claimed 且未 terminal 的第一次 run 才执行 ReAct。这样 durable source 在 AgentResult commit
后、自己的 delivery prepare 前 crash，可以用同一 ref 和原 typed request 取回 exact AgentResult；它不能从
Message、TurnSaved、notice 或正文猜一份 output。

Agent Service 开放前，agent-loop 必须扫描 durable reserve/claimed 与 active-task receipt：reserved 且没有
claim receipt 的记录确定性 terminal `failed_before_claim`；claimed idle 则以同一 task key 重建 claim，
in-progress 交给既有 Turn recovery。未知、重复 active task 或 identity 分裂使 Root degraded，不能猜测
成功。barrier fixture 固定覆盖 reserve 后/claim 前 cancel、old task occupied、claim 后/mark 前 crash、
mark 后/return 前 crash 与重复 ref，Session reserve/terminal 和 TaskControl claim/release 都恰好一次。

这条握手让普通 source 能在模型前安全使用已接受 turn identity，而不是复活 before hook。crash-before-accept
没有 Turn fact，source 原 attempt 可重试；accept 返回后崩溃时，source 以 ref 重新取得同一 TurnStart；Wake
若已写 selection，则按同一 turn_id 恢复后 run，否则重新执行自己的 prepare；run 前决定 quiet 时调用
finish(skipped)，模型 call=0。accept-before-model 的 cancel/reload 也只 terminal 同一 start；一旦 run，正常
cancel/terminal 仍由 concrete Agent 唯一拥有。现有
`agent.control.models.TurnRequest(thread_id, input, metadata)` 在 M2 同批改名为 `ControlTurn`，只留在
control transport/store 边界。control adapter 把 ControlTurn、channel adapter 把 InboundMessage 恰好一次
解析为 TurnSource/Message，先 accept 再构造领域 TurnRequest；M2 起同一领域 request 只经 AGENTS 进入 Root-bound concrete Agent，不增加第二只
request 类型，也不再携带或读取 raw metadata。

TurnRequest 没有可遍历的行为 metadata；每个字段都有唯一 owner：

| typed field | 唯一解释者 | 可观察语义 | 迁移批次与旧入口 |
|---|---|---|---|
| `start: TurnStart` | `agent-loop` | 同一 source ref 的 accepted turn identity；run/finish 只能消费一次 | M2；替代 source 各自伪造/晚取 turn_id |
| `past_read: PastRead(full|empty)` | `sessions` | 从一次 persistent history 选择派生 `HistoryViews(runtime_history, prompt_history)`；分别进入 ContextCall/TurnInput，不混类型、不改 Session | M2 建 carrier；M3e 删 `skip_session_history` metadata 和 `session_history_read` ambient 解码 |
| `context_texts` | `context-input` | source-owned ContextText | M2 建 carrier；M3e 删 `prompt_hints`/`extra_hints` 汇聚 |
| `skip_sections` | `system-prompt` | 只过滤 PromptSection | M2 建 carrier；M3e 由待退役 system owner 消费，M3f 切到 system-prompt 并删旧 owner |
| `skip_parts` | `context-input` | 只过滤 ContextPart/ContextText | M2 建 carrier；M3e 删 `disabled_prompt_sections` 的 context 解码 |
| `tool_grant: ToolGrant` | `tools` | 限制可见和可调用工具 | M4；删 scope/metadata grant 解码 |
| `tool_picks: tuple[ToolPick, ...]` | `tools` | 为本 Turn 选择普通 registry tool，并声明 preload/terminal | M4；删 `preloaded_tools`/`terminal_tools` 名单特判 |
| `turn_tools: tuple[Tool, ...]` | `tools` | 本 Turn 的普通临时 Tool 定义 | M4；删 `tool_overrides` map |
| `save: SaveChoice(user, assistant)` | `sessions` | 分别决定 user/assistant Message 是否进入原子 Turn commit | M6；删 `omit_user_turn`/`omit_assistant_turn` metadata |
| `effects: EffectMode(run|skip)` | `sessions` | 写入现行 typed `effects.post_commit`，供 saved SavePart 判定 | M6；删 runtime `PostCommitEffect` metadata decoder |
| `step_limit: int | None` | `agent-loop` | 限制本 Turn 的 ReAct step 数 | M7；删 scope `max_iterations` |
| `source: TurnSource`（在 start 内） | `agents` | 在 accept 边界校验并冻结来源 identity；agent-loop 只原样带给 Tools 和诊断，不按值分支 | M2；删 scope `tool_source` |

`ToolPick` 只有 name、preload 与 terminal；`tools` 校验 name 存在、grant 允许和 tuple 无重复，再返回冻结
view，agent-loop 只消费 view 的 terminal fact。TurnRequest 不携带 registry、任意 lookup 或 callback。
`SaveChoice(user=False, assistant=False)` 必须搭配 `EffectMode.skip`；`user=False, assistant=True` 保留现有
stateless continuation；其他组合按 sessions 的显式合同校验。`PastRead.empty` 只改变这次读取：sessions
仍保留原始 persistent history；`HistoryViews.runtime_history` 与 `.prompt_history` 是不同 typed projection，
empty 时两者都为空。programmatic validation 的 `session_history_read=false|true` 分别迁为 empty|full。

`AGENTS` 只校验并冻结自己拥有的 `TurnSource`，其余 TurnRequest 字段对它 opaque；它把完整 request 交给
factory/Agent。`agent-loop` 不循环解释字段，也不把它转成 dict；它只在固定算法位置把字段传给上表 owner，
并把冻结的 source 原样传给 ToolInput 和诊断。任何 consumer 都不得按 source 值改变行为。外部 Message metadata、ambient ContextVar、Session
metadata 和 durable Inbox 都不能代替这些 typed facts。

`provider-input` 的 definition、provider 和 consumer 是一条完整 capability seam。中立 public API 只有：

```text
PROVIDER_INPUT.open(TurnInput)          ──► InputState
InputState.build(InputCall)          ──► ProviderInput
InputState.settle(CallResult)        ──► InputRetry
```

`TurnInput` 每个 Agent Turn 只创建一次，冻结 `session_key`、`turn_id`、Session 创建时间、`sessions`
派生的 immutable prompt history units、本 Turn 的 PromptText tuple、ContextMessage tuple 和一只窄 ledger
read grant。`InputCall` 每个 provider attempt 创建一次，冻结：

- `call_id`、1-based `call_number`、1-based `try_number`；
- `cause=normal|too_long`；
- 当前完整 Turn transcript 和当前 tool schemas；
- `ModelChoice`、context limit、max output 和 continuation；
- 之前已经 settle 的 immutable usage tuple。

`ProviderInput` 是 provider-ready content payload：最终 messages、tool schemas、max output、可继续使用的
continuation、`InputSize`、opaque `InputReceipt`，以及 build 中额外模型调用产生的 immutable usage。
其中 `InputSize` 是这次 `build` 的 closed typed return 字段；agent-loop 直接保留它，provider-input 以 typed
observe 发布同一个值，observer failure 不影响返回的 ProviderInput。
`InputCall.transcript` 以独立 current Message 开始，后续 call 再按执行顺序带上本 Turn 的 assistant/tool
进展；每条 Message 保留 role 与既有 `normal|context` kind。prompt history、context 与 turn transcript
只是 lane，不是新的 Message kind。`provider-input` 可按 CTX-002 先移除 drop=extra，再移除 drop=repeat，随后按完整语义边界
缩小 prompt history，也可在这些 lane 内做已批准的 compaction projection；drop=keep 不得移除。
它不得重排 lane、把 context 与 turn transcript 合并、改变 transcript 内执行顺序，或把 system text 降成
普通 Message。因此 `ProviderInput` 仍保持 agent-loop 给出的
system → prompt history → context → turn transcript 顺序；首个 call 的 transcript 只有 current Message，
后续 call 保留 current 开头和已发生的 assistant/tool 进展。
stream callback、transport retry 和 auth 不进入 `provider-input`，仍属于 `models`。每只 receipt 必须恰好一次以
`CallResult(receipt, status, usage)` settle；status 只能是 `done`、`too_long`、`failed` 或 `cancelled`。
`InputRetry` 只回答同一逻辑 call 是否可用 `cause=too_long, try_number=2` 再 build；禁止第三次尝试、换 provider
或 Core fallback。缺 provider、双 provider、序号倒退、receipt 错配或重复 settle 均 fail-loud。

basic provider 原样组合且永不要求 overflow retry。compaction provider 依自己的 durable ledger 投影，
可以在 `InputState` 内私有保存 ledger head、token meter、已闭合 tool batch 与待发布 fact。它从每次冻结的
完整 transcript 与上一只 receipt 的边界识别新增闭合 batch，而不是要求 loop 调用 compaction 专用方法；
成功 settle 记录 response usage，并在单次运行只发布一次待发布 fact。`too_long` settle 可允许第二次 build 强制压缩；
失败或取消不伪造已提交 checkpoint 回滚，下一次 `open` 从 ledger/receipt 重放并补发。若 build 改变
messages，返回的 continuation 必须为空。`agent-loop` 只 require `PROVIDER_INPUT` 并传 typed input/result，
不识别 compaction、不读写 provider 私有 state、不接受 mutable request binding 或 listener 列表。

`build` 与 `settle` 不是可分别注册的 before/after hook。一个 Root 只有一个 provider，同一只
`InputState` 同时实现两者，receipt 把一次 build 和一次 settle 配对；没有 listener order、任意 ctx、
跨 capability 改写或第二条控制流。普通返回路径必须 settle；进程崩溃来不及 settle 时，compaction
provider 按 `source_ref` 幂等补发 committed fact，不能宣称外部效果被回滚。

`reply-output` 是模型最后一次返回与 Session commit 之间唯一的 typed 输出边界。DSH 的具体 loop 先把
stream 组装成 typed content blocks，再把同一 assistant message 直接 append 到 Session
（`packages/core/agent-loop/src/agent.ts:341-427`）；它没有一条可按顺序改写 assistant 的 after phase。
Akashic 还要兼容 Citation/Meme 的模型内隐藏协议，因此增加一只普通 decoder，而不是复制 after hook：

```text
REPLY_OUTPUT.add(ctx, ReplyPart) ──► Effect
await REPLY_OUTPUT.ready()          ──► None
await REPLY_OUTPUT.open(ReplyCall) ──► ReplyState
ReplyState.output                  ──► FinalReply
await REPLY_OUTPUT.settle(ReplyState, ReplySave)
```

`ReplyCall` 只冻结 session/turn/channel/chat identity、provider 的 raw text、已有 typed `MediaItem`、
typed ToolUse tuple 与
cancellation；没有 Session writer、Channel、Prompt、Service lookup、mutable ctx 或 metadata bag。
每个 ReplyPart 只有唯一 name，并基于**同一份** ReplyCall 返回零个或多个
`ReplyMark(start, end, text, media)`。start/end 只引用 raw text 的 code-point span；service 校验边界、
非空 span、同一 part 内和跨 part 都不重叠，再按 source span 一次合并成 immutable
`FinalReply(text, media)`。已有 media 保持原顺序；每个 mark 的 media 按 mark 的 source span 顺序追加，
每只 mark 内部顺序和重复项保持不变。所有
part 并行独立读取原稿，name 只作身份，不是 order；任一 overlap、越界、非法 media、失败或取消都在
附件导入和 Session 写入前 fail-loud，不交付 partial reply。一个 part 看不到别的 part、别的 mark 或合并后
文本，也不能重排/删除它没有明确圈出的字符。这是“解码声明过的模型协议”，不是任意 ReplyEdit。
`ReplyMark.media` 只能含由 tool boundary 或该插件自己的窄 file grant 铸造的 immutable `MediaItem`；它不
接受 raw path、URL、open file 或任意读 capability。attachment owner 在 Session commit 前验证 provenance
并导入；reply-output 自己没有文件系统权限。非法 item 或导入失败走同一 open/failed settle，不能写 Session。
`reply-output` 自己只保留当前已经公开给模型的 trailing `<name:value>` hidden-marker grammar：part claims
先按上述规则验明，service 再对**未被 claim** 的 trailing marker span做固定空替换，防止内部协议泄漏；
这条 base rule 没有插件名、priority 或 callback，不能清理其他文本。它保留 Citation 当前 cleanup 的
可观察安全行为，同时让 Meme 对同一 span 的 media claim 成为唯一解释。

ReplyPart 的 extension contract 不是一只隐含 callback。每只 part 明确实现
`ready(SessionRead) -> None`、`open(ReplyCall) -> PartState(marks, receipt)` 与
`settle(PartState, ReplySave) -> None`。`marks` 是 immutable tuple；`receipt` 是只回给同一 part 的 opaque
immutable value，`None` 只明确表示本次没有 durable work。open 必须在返回前完成自己的 prepare；settle
只能消费自己铸造的 PartState，identity mismatch、重复冲突或未知 receipt fail-loud。ready 只扫描该 part
窄 plugin-data grant 中自己的 journal，并用 SessionRead 收敛，不能枚举别的 part、Session SQL 或 registry。
reply-output 只保存 PartState、校验/合并 marks，并逐 part 调 settle；不解释 receipt。

任何 open 会返回非空 durable receipt 的 ReplyPart，必须由同一普通插件以同一 name 注册一只 SwitchPart；
Root sealing 绑定两者的 owner artifact/generation identity，缺失或错配拒绝 candidate。该 generation 的
snapshot lease 覆盖完整 Turn，因此 part remove/replace 前 Core 必须先等所有含 old generation 的 lease
归零；stop 再断言本代 open receipt 全部 settle/abort，未清空就拒绝 switch。pure ReplyPart 不写 journal、
receipt 恒为 None，不能为了进入 publication plane 假注册 SwitchPart。

若任一 part 在 open 中失败或取消，service 必须先以 failed/cancelled 收敛已经 prepare 的 receipt，再向 caller
抛出原错误；不能交付半只 ReplyState，也不能等待下一次 boot 才做正常错误清理。
最终合同中，Session/attachment owner 只使用 FinalReply，保证持久正文、附件与 Channel 发送是同一份结果；无论
commit saved、明确 skip、失败或取消，caller 都必须恰好一次 settle。`ReplySave` 只表示
saved/skipped/failed/cancelled/interrupted，saved 携带 deeply immutable TurnSaved。`SESSIONS.save` 返回 closed
`SaveResult`：只要至少一条 Message 进入事务就返回 saved(TurnSaved)；SaveChoice 两项均 false 时不写
messages/outbox，返回 skipped，不能伪造 TurnSaved。每 Turn 只有一个 terminal reply，所有 receipt 直接使用
已有 `(session_key, turn_id)`；assistant message id 仍只属于 Message，user-only/no-save 也不需要第二套 identity。part 在 saved 后提交
自己的 ledger/fact，在 skipped/failed/cancelled/interrupted 撤掉 prepare；崩溃时
按 prepare journal + SessionRead 的 turn identity 幂等收敛。settle failure 和当前
TurnCommitted listener failure 一样发生在 Session save 已确定、delivery 尚未开始的 fence，并且 fail-loud。
settle 成功后两条路径都运行 M5b 建立的唯一 private old-commit，再调用
`SAVE_NOTICE.send(SaveResult)`；saved 从 outbox dispatch，skipped 对 sealed generation live dispatch。
M6b～M9 的 old-commit 固定夹在 settle 与 notice/delivery 之间。进程重启时必须先由 agent-loop 把 crash
遗留的 queued/running Turn 收敛到 cancelled/interrupted，且 provider reboot call=0；只有随后
REPLY_OUTPUT 才能用 SessionRead.status 的 save=saved/skipped 或 terminal failed/cancelled/interrupted 收敛全部
pending receipt并报告 ready。再由 `SAVE_NOTICE.ready()` 让 parts 收敛 prepare 并重放 pending saved notice，
最后各 source 才恢复 delivery。未知、冲突或 recover failure 让 Root degraded，且零 observer/零 sender。
这样 crash 落在 ReplyPart prepare 后、Session save 前，或 save 后、settle 前、notice 前时，都先取得唯一
durable terminal/save 事实，绝不会等一只 stale running Turn，也不会先送 reply 再补插件 ledger。
没有 feature part 时，basic reply-output 仍执行既有 parser 与上述 fixed hidden-marker decoder；没有 marker
的正文和已有 media 保持不变，Agent 仍能工作。它不是 raw identity decoder。

M6b 到 M9 的唯一例外是 1.5 入账的 private `OldReply`：它只把 Citation 当前已写入 assistant row 的
exact metadata 交给 sessions 同一事务，不改变 FinalReply，也不进入 public contract。它是必须在 M9 删除的
external migration debt，不是 FinalReply 的隐藏字段。

M6b 先把当前 Citation→Meme→Citation cleanup 的 exact serial 行为包成**一只**
`DEPRECATED(EXTERNAL)` ReplyPart，旧 listener 仍只服务现有 artifact；内部调用者立刻只认
REPLY_OUTPUT。M9 把 Citation 与 Meme 改成两只独立 part：Citation 只圈自己的 `§cited`/inline-memory
span，Meme 只圈自己的 `<meme:...>` span并加入 media。两者使用同一 raw reply，不再互相 require 或按
先后改写。固定 fixture 必须覆盖 citation trailing meme tag、inline tag、code tag、fallback tool ids、
最终正文、附件 path/count、Session rows、outbound 和失败恢复。当前 Citation 对任意未知 trailing tag 的
兜底删除由 reply-output 的固定 hidden-marker base rule 等价承接，不变成插件有序 cleanup。

`sessions` 可以声明 `workspace_files=("sessions.db",)`，但只有它获得正式 writable grant。
candidate closure 中的 `sessions` 使用插件自己创建的全新临时 schema 和 programmatic Session，
不复制、读取或写正式 `sessions.db`；它是验证数据，不是第二名正式 writer。需要历史语义的回归由
测试把固定 fixture 恢复进一次性 workspace 后串行运行，不从 live DB 取样。

### 4.3 五个中性 Core 原子

| kernel atom | 只拥有 | 不拥有 |
|---|---|---|
| `ServiceCall[T]` | 构造时固定的 ServiceKey 与 lease source；一次完整 call | selector、request 解析、background task、领域 fallback |
| `ServiceHold[T]` | 构造时固定的 ServiceKey 与不可伪造 HoldKey；Core 铸造全局 HoldId、exact Root/artifact pin、跨进程 call 与 drop | source 名称、payload、Channel live binding、delivery、ACK 解释或任意 Service lookup |
| `RootScope` | owning Root identity、task/Effect cleanup、root-bound lease acquire | stable/latest 选择、领域 retry、跨 Root 重投 |
| `TaskControl` | opaque scope/task claim、exact lease、task、cancel callback、terminal release | Message/Turn/Session、Agent/factory、持久状态、错误解释、delivery |
| `RootSwitch` | 跨代共享 owner 的 closed plan、两代 artifact pin、publication record 与逆序恢复 | 业务数据、request hook、普通 Root-local resource、feature 名称或任意 callback phase |

`ServiceHold` 不是 source 插件或第二个 snapshot manager。kernel 像 ServiceCall 一样只为一个固定
ServiceKey 铸造它，并从 sealed caller capability identity 铸造稳定、不可伪造的 `HoldKey`；两个调同一
Service 的 host 也有不同 HoldKey。host/插件不能创建、改 key 或选择 generation。仍持有 live exact lease
的 durable owner 先调用 `reserve()`；Core 铸造全局唯一 `HoldId`，在一份 journal row 持久写入
HoldKey、sealed ServiceKey identity、
exact Root/snapshot/artifact/generation 和 `reserved`，并立即 pin artifact。owner 再在自己的 ledger 写
HoldId、source generation、Channel generation/config identity、target 与 stable delivery key，最后调用
`activate(HoldId)` 把 Core row 置为 active；active 前 Agent call=0。Core 不读取 source row，也不保存 payload、
target、Channel config 或业务 completion。

`pending()` 只列出这只固定 HoldKey + ServiceKey namespace 的 reserved/active HoldId；`call(id, action)` 只从
journal 中的 exact Root require 同一 sealed ServiceKey，wrong HoldKey/ServiceKey/Root/artifact fail-loud；`drop(id)` terminalize
并释放 pin。boot 在 publication gate 关闭时先从 Core journal 重建所有 reserved/active Root，再让每名 exact
source 对账自己的 row：reserve 后/row 前 crash 没有 source row，该 HoldKey 的 owner 确认后直接 drop；
row 后/activate 前幂等 activate；active 后按 row 恢复。source row 的 outcome 初始为空，只能终结为 closed
`done|abort|unknown`：delivery ACK、
Control/result receipt、明确 no-sink 均为 done；accept failure、确定未产生外部效果的 run/cancel 为 abort；
外部效果是否发生无法证明才是 unknown。owner 必须先 durable 写 done/abort，再 drop，最后删 row；completion
后/drop 前与 drop 后/delete 前 crash 都按同序幂等收敛。unknown 保留 hold、证据与 degraded 状态，不猜成功，
也不允许换代。row 先删再 drop 被协议禁止。

reserved/active hold 与 live lease 一起计入 RootSwitch drain，所以 old source/Channel 有 pending work 时不能
换代。Core hold 只 pin 可重建的 exact Root/artifact；source row 只冻结 source generation、Channel
generation/config 和 stable delivery key。live binding/token/socket 是 Channel owner 每次进程启动建立的
ephemeral 资源，绝不持久化。reboot 在 held exact Root 中由同 generation 用冻结 config 创建**新的**
live binding/token，再以 stable delivery key 幂等发送；dispatch envelope 使用这只新 token。这样
不会拿 current stable 解释旧 output，也不假装复活已死亡连接。

`TaskStart.claim(scope_key, task_key, lease, run, cancel) -> TaskWait` 对整个进程原子；`lease` 是
`TaskLease`。
同一 opaque scope 跨 generation 只能有一个 active task。具体 Agent 负责把自己的 session/attempt
领域身份映射成稳定 opaque key，并负责何时允许 start/cancel/terminal；accepted receipt 与 durable
active-attempt fact 保存同一个 task key。`TaskControl` 只执行 claim、按该 key 通知原 owner 的
cancel callback 和最后释放。Control host 只获得 `TaskCancel`，不能枚举
task、读取结果、创建工作或取得 snapshot。新 ingress 要 interrupt 旧 attempt 时，先从
`SESSIONS` 窄 read Service 取得 durable active task key，不能按内存对象或 current stable 猜测。

publication drain 时，bootstrap Control 可以依已知 task key 请求取消 old Root 的仍活 task；old concrete
Agent 继续唯一负责 terminal/Session settle，并在最后释放旧 lease。只有包含 changed shared owner 的旧
snapshot 会阻止该 owner switch；完全不含该 owner、或复用同一 generation 的旧 task 可继续。因而不存在
旧 Turn 在自己依赖的 sessions/reply part leave 后继续调用，也没有内存状态搬家、两代共同写或特权 Agent
service。`ActivityHost`/generation lease 的现有 admission/drain 语义是实现资产；不得再创建一份 Agent
专用 publication 平面。

`RootScope` 由每个 Fiber 平等取得。`agent-loop` 创建的具体 Agent 绑定自己的 root scope；
它只复用同 Root 的 current lease，或向该 scope 取得 owning Root lease，遇到其他 Root
binding 直接失败。Scheduler/Wake 的 timer callback 因而可以直接调用同 Root 注入的 `AGENTS`；Root
已退休时原样得到 `RootRetired`，由 Scheduler/Wake 自己 settle/rearm，绝不 fallback
到 current stable。candidate Root 的普通 background scope 关闭，只有 Core 铸造的 attached
validation capability 能启动一次 candidate task。

### 4.4 无环注册

```text
foundation providers
  sessions ──► SESSIONS (includes prompt history)
  models ────► CHAT_MODELS ...
  tools ─────► TOOLS
  system prompt ► SYSTEM_PROMPT
  context input ► CONTEXT_INPUT
  provider input injects SYSTEM_PROMPT.render
  provider input ──► PROVIDER_INPUT (exactly one provider)
  reply output ───► REPLY_OUTPUT
  sessions ───────► ROOT_SWITCH.add(session writer part)
  session view? ► SESSION_VIEW (only with proved fold consumers)

agents provides: AGENTS + empty factory slot

agent-loop injects: AGENTS, SESSIONS, CHAT_MODELS, TOOLS, SYSTEM_PROMPT.build,
                    CONTEXT_INPUT, PROVIDER_INPUT, REPLY_OUTPUT, SAVE_NOTICE
agent-loop effect: register(default factory) ── cleanup unregisters

default skill feature (not required by agent-loop)
  host-check ──► HOST_CHECK
  skills ──────► SKILLS
  skill-files injects HOST_CHECK + SKILLS
  skill-files provides SKILL_FILES and registers one filesystem provider in SKILLS
  skill-files ──► ROOT_SWITCH.add(skill link part)
  internal roots / external migration bridge inject SKILL_FILES ──► add(SkillRoot)
  skill-use/context injects SKILLS + CONTEXT_INPUT ──► ContextPart
  skill-use/prompt  injects SKILLS + SYSTEM_PROMPT ──► PromptSection
  skill-use/tool    injects SKILLS + TOOL_CATALOG  ──► load-skill Tool
  M4 evolves the same TOOL_CATALOG into TOOLS; it does not register a second tool

drift files (no runtime reader today)
  external root ──► SKILL_FILES(group=drift) ──► check + link projection only
  default SKILLS never sees this group

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
        ┌───────────────┬───────┴────────┬───────────────┐
        ▼               ▼                ▼               ▼
  prompt history    system prompt   context input   tools view + model limit
        └───────────────┴───────┬────────┴───────────────┘
                                │ agent-loop fixes envelope order once
                                ▼
                         provider-input open
                                │
                      ┌─────────▼──────────┐
                      │ build each attempt │◄── too_long + InputRetry
                      └─────────┬──────────┘
                                │ ProviderInput + receipt
                                ▼
                         models call
                                │ done / too_long / failed / cancelled
                                ▼
                         provider-input settle
                                │
                                ├── tool calls ──► tools run ──► next transcript
                                └── final provider reply
                                ▼
                         reply-output open
                                │ FinalReply + receipt
                                ▼
                     sessions atomic save
                                │ SaveResult
                  ┌─────────────┴─────────────┐
                  │ saved: Message + outbox   │ skipped: turn_saves only
                  └─────────────┬─────────────┘
                         reply-output settle
                                │ M5b-M9 private old-commit
                          save notice send
                                │ live SaveResult observe
                                ▼
                         sessions finish
                                │ durable AgentResult
                                ▼
                   source sink + receipt / ACK
                    (passive = conversation)
```

具体 Agent 的 task lease 在 accept 时由 `TaskControl` 原子 claim，直到 Agent terminal/AgentResult 后释放。
source 同时持有更外层的同一 exact Root lease：snapshot 外 host 的 `ServiceCall` 一直包住整个 action，Root 内
source 的 owning Fiber/RootScope 一直包住自己的调用。因此 M2c 后 source 可以在 AgentResult 已终结后调用自己
的 sink 并等待 receipt/ACK；只有 sink 完成或确定无需发送后，外层 action/scope 才返回并释放 Root lease。
取消只通知当前 attempt，不能让外层 lease 在 action 返回前消失。这样 snapshot 覆盖完整 ReAct 与 source
delivery，却不要求 agent-loop 持有 Channel，也不把 AgentResult 伪装成 pre-terminal result。新 generation
不能 claim 同一 session scope。durable source 在 admission 时还用 ServiceHold pin 同一 exact Root；若进程在
AgentResult 与 source prepare 之间崩溃，hold 代替已经消失的 live lease 继续阻止换代，boot 只在该 exact Root
恢复 sink。Core journal 里的全局 HoldId 把 fixed ServiceKey 与 exact Root/artifact 绑在一起；source row
另外冻结 Channel generation/config，不冻结进程内 binding。source 先持久 `done|abort`，再 drop hold，最后删
row；`unknown` 保持 pin 和 degraded。这不是一只特权插件，而是所有 durable caller 都可按固定
ServiceKey 获得的领域中性寿命原子。

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
- 不适用的 feature plugin 没有 part；不存在“先运行 passive hook 再 early return”。

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

reload 也不能把这张图依赖的 shared owner 从中间剪开：publication gate 先关新 lease；某只 sessions、
durable ReplyPart 或 Channel part 的 owner identity 改变时，Core 等所有包含该 generation 的 old snapshot
lease 回到零，才允许 RootSwitch 触碰它。旧 Turn 因而始终在自己的 owner generation 完成 reply open、
Session save、reply settle、save notice 和 delivery；无关普通插件若没有 shared part 可以独立换代。新
stable 永远看不到一只由已移除 old part 留下、却要求 new part 解释的业务 receipt。

DSH 也把 live Agent 停止/清理与 factory registry 放在普通 effect 中
（`packages/core/agent/src/index.ts:149-204`，`packages/core/agent-loop/src/index.ts:560-583`）。Akashic 的更强
保证是当前 `RuntimeSnapshot` 已冻结整棵 Root 并用 exact lease 等待全部工作退出
（`agent/plugins/snapshot.py:76-114,876-907`）。这是 publication/lifetime 差异，不是 Agent 特权。

## 6. 当前特殊功能清单与目标 owner

| 当前特殊点 | 当前位置 | 目标组合 | Core 新增专用原子？ |
|---|---|---|---|
| command 在模型前短路 | `AgentLoop._process`、`PassiveTurnPipeline.run_command`；三只 live v3 artifact 已注入 `COMMANDS` | 普通 commands plugin 提供现有 registry；conversation source 识别后不创建 Agent Turn | 否 |
| plugin rollout fact 塞入下一轮 Prompt | `AgentLoop._process` metadata | rollout 插件向 `SYSTEM_PROMPT` 提供一次性 section；事实文件由其声明 | 否 |
| session 模型选择 | `AgentLoop._resolve_model_selection` | 具体 Agent 通过 `SESSIONS` 读取已保存选择，将 `ModelChoice` 显式传给 `models`；`models` 只校验、解析、冻结与调用 | 否 |
| Shell 按工具名和类 cleanup | `AgentLoop._cleanup_shell_owner` | Shell 插件消费具体 Agent 的 terminal fact，并清理自己拥有的 execution | 否 |
| Tool Search enable、schema cap、LRU、名称解锁 | `DefaultReasoner` 多处分支、ToolRegistry meta set | Tool Search 普通插件注册普通 tool；只替换 `TOOLS` 的 scoped view/restriction | 否 |
| 未解锁工具的提示文字 | `DefaultReasoner` | Tool Search 自己的 tool outcome 或 system section 说明可用工具；`tools` 不知道“解锁” | 否 |
| `message_push` 媒体抽取 | tool loop 按名称收集 | Message Push Tool 注入 committed Channel 的窄发送 Service，在 execute 内完成独立 outbound Turn 与 receipt；只返回普通 model-facing receipt/media content | 否 |
| `mobile_attention` | Reasoner/Turn result 固定字段 | 产生确认请求的 Tool 注入 Mobile output projection 的窄 Service；projection 按 turn identity 记录并随现有协议发送，agent-loop 不解释 attention | 否 |
| Meme/Citation response decoration | after-reasoning/after-turn consumers | 模型指引归 `SYSTEM_PROMPT`；普通 `REPLY_OUTPUT` 在同一 raw reply 上合并不重叠 ReplyMark，结果同时持久与发送；事实归私有 ledger | 否 |
| Shell 参数改写与拒绝 | `TOOL_INPUT_PREPARE`、`TOOL_EXECUTION_AUTHORIZE` | 最终收进普通 Shell Tool owner 的固定 rewrite→authorize 算法；M4a～M9 用 exact external block 保安全 | 否 |
| Skills catalog、active skills、load-skill | `SkillsLoader`、prompt blocks、tool 与 RuntimeSnapshot ambient lookup | `skills` 管 provider/view；`skill-files` 管来源与 projection；`skill-use` 向 system/context/tools 各注册普通 contribution | 否 |
| Skill host availability | SkillsLoader import host factory，local/bridge 两套判断 | 普通 `host-check` 二选一 provider；skill-files 只调用 name-only check | 否 |
| plugin skill freeze、link journal、Gate/status/provenance | `PluginSkillHost`、`PluginSkillLinker`、PluginManager 和 snapshot 特制字段 | skill-files/skills 拥有 projection、typed check 与 SkillBody provenance；Core 只做 generic publication | 否 |
| normal/drift 两套 skill roots | `PluginContributions.skill_roots/drift_skill_roots` 与一只总 host | SKILL_FILES 的 agent/drift 两组受保护来源；只有 agent 组进入 SKILLS，drift 组保持 check/projection；M9 旧字段清零 | 否 |
| Memory、hints | before-reasoning/prompt-render phase | 稳定规则注册普通 system section；Akasha retrieval 注册 `CONTEXT_INPUT` part，本轮 source hints 随 TurnRequest 传 ContextText | 否 |
| Wake select/claim/quiet | `CONTEXT_PREPARED_EVENT` + mutable BeforeTurnCtx | Wake 在 Agent.accept 后、Agent.run 前执行自己的 source gate，得到 WakeRun 或 WakeSkip；ContextText 只运送已选 hint | 否 |
| Akasha feedback marker | `AFTER_REASONING_PREPROCESS` 写 user metadata | feedback Tool 按 turn_id prepare Akasha 私有 ledger；TurnSaved commit，failed/cancel/boot 用 SessionRead settle | 否 |
| Akasha committed ingestion | `AFTER_TURN_COMMITTED` + TurnCommitted bag | Akasha 以 TurnSaved identity durable enqueue，再用 SessionRead 投影已提交 Turn | 否 |
| Compaction request gate | `PROVIDER_REQUEST_PROJECTION` | 能力与 ledger 保留，当前 mutable request binding 判为 `move`；M3g 必须在不引入 before-step 的前提下收窄为独立 provider-input 边界 | 否 |
| Markdown MEMORY/SELF 写入 | committed checkpoint 后 | 已有普通 markdown-memory 插件 | 否 |
| streaming、thinking、tool progress | AgentLoop sink + EventBus | 具体 Agent 发算法事实；Observe/Channel 的可选 session-view 消费 | 否 |
| Session commit 与 outbound 混在 after-turn | PassiveTurnPipeline | `sessions` 先原子 commit；conversation/Channel 后 delivery/ACK | 否 |
| 六组可任意改写总状态的 phase | `agent/lifecycle/phases/**` | 删除；收敛到 owner 明确的 section/model request/tool/fact/view | 否 |
| provider retry、max iteration、tool batch、continuation | `DefaultReasoner` | `agent-loop` 内部直接算法，不拆成 feature plugins | 不适用 |
| attempt admission、interrupt、cancel、terminal | `ConversationRuntime` | 默认具体 Agent（`agent-loop`）唯一 owner；`agents` 只管公开合同与 factory | 否 |
| durable inbound handoff 与 ACK 顺序 | `PassiveMessageWorker` | ordinary conversation plugin，持久写只请求 `SESSIONS` 窄 Service | 否 |
| control `TurnRequest(..., metadata)` 与领域 TurnRequest 同名 | `agent/control/models.py`、`bootstrap/control_execution.py` | transport DTO 在 M2 改名 `ControlTurn`；唯一 source adapter 产出无 metadata 的领域 TurnRequest | 否 |

禁止用 `TURN_EFFECTS`、万能 middleware、任意 mutable context 或一个“passive hooks”总 Service 把这些
重新装进一只袋子。每个 public seam 必须指向表中已有 owner 与一种明确变化轴。

## 7. 持久状态、外部效果与恢复

| 对象 | 正常增加 | 可原位更新/逻辑终态 | 物理减少 | 唯一 owner 与恢复证据 |
|---|---|---|---|---|
| `sessions.db/messages` | completed transcript 原子 INSERT | 不更新正文 | 仅 SES-003 显式用户撤销/删除 | sessions；DB backup、row/seq/write-set diff |
| `sessions` metadata / `turns` | admission、attempt、terminal 写入；terminal 同次冻结 source-neutral AgentResult projection，包括既有 items/usage | 仅既有状态机和白名单 metadata；AgentResult commit 后 immutable | 仅既有管理协议 | sessions；TurnSource ref、AgentResult、restart recovery |
| `sessions.db/turn_saves` | 每次 SESSIONS.save 增加 saved/skipped outcome、channel 与时间；saved 同事务冻结 TurnSaved | immutable，不原位改 outcome/identity | 本迁移不自动减少，未来 retention 需独立批准 | sessions；SaveResult、SessionRead.status、reply recovery |
| `sessions.db/saved_notices` | saved Message commit 同事务 append pending，并冻结 TurnView source facts 与 SavePart recipient name/artifact/generation tuple | 每名 recipient pending → done；identity/payload/recipient 不改 | 本迁移不自动减少，未来 retention 需独立批准 | sessions；TurnSaved、TurnView、outbox state、boot replay receipt |
| attachments/compaction/delivery rows | 既有事务增加；durable source admission 在 Core reserve 后冻结 HoldId、source generation、Channel generation/config、target 与 stable delivery key | payload/identity/key 不改；outcome 从空值只能终结为 done/abort/unknown | done/abort 先 drop hold 后才可删 row；unknown 不减少 | 各 owner 窄 Service；digest、receipt、prepare fence、ServiceHold |
| Akasha feedback ledger | feedback Tool 成功前按 turn_id prepare；历史 marker 一次导入 | saved notice commit 或 failed/cancelled/interrupted settle | 只按 Akasha 已批准管理协议 | Akasha；plugin-data backup、message/turn identity、import count/hash、settle receipt |
| MEMORY/SELF 与 receipt | committed checkpoint 触发 | backup + atomic replace / idempotent receipt | 只按 MEM 条款 | markdown-memory；backup、source_ref、receipt |
| plugin skill symlink 与 link journal | generation publication prepare 增加 pending；commit 原子换 link/owner map | 当前 generation replacement、rollback/recover | 仅 owner-confirmed stale link；手工目录必须先迁入插件并有 backup | skill-files；journal、link target、source hash、pre/post inventory |
| plugin rollout fact | rollout terminal 增加一次临时事实 | consume 逻辑终态 | 成功消费或已批准恢复 | rollout plugin；fact/journal |
| Shell/process | 工具显式启动 | active → terminal/cleanup_degraded | owner 确认退出后 | Shell/Workload plugin；process registry/report |
| Channel send / remote API | prepared 后调用 | committed/partial/failed/outcome_unknown | 外部效果不可由 Git 删除 | Channel/Delivery/tool owner；provider receipt |
| snapshot/candidate | publication transaction 增加 | state、lease count、stable/latest 指针 | drain 后清理不可达代 | Core kernel；journal、identity、zero lease |
| Core service holds | 持有 live lease 时 reserve 全局 HoldId、不可伪造 HoldKey、sealed ServiceKey、exact Root/snapshot/artifact/generation 并立即 pin | reserved 只能 activate 为 active；drop 为幂等 terminal；业务 payload/config/binding/delivery/outcome 不进入 | 仅 source 已 durable 证明 done/abort 后 drop；unknown 不 drop | Core kernel；ServiceHold journal、artifact pin、HoldKey + ServiceKey namespace、RootSwitch hold count |

AgentResult 复用现有 turns terminal row 的 status/items/usage/final_response/error 权威字节，不新增第二张结果表、
不从 messages 猜结果，也不改变既有 row 减少协议。本迁移只为 sessions 的 save outcome 与 durable saved notice 增加一只 forward-only
`turn_saves`/`saved_notices` schema migration，为通用 ServiceHold 增加 Core journal，并为
Akasha 私有 feedback ledger 增加 owner 自己的表；执行前分别备份 formal workspace、跑 integrity/count/hash，
失败恢复旧 artifact 与备份。除此之外不迁正式 workspace，不 UPDATE/DELETE 既有消息，不复制正式数据库做
第二个 writer。
`sessions` owner 切换由它自己的 SwitchPart 在 publication gate 内 stop/drain old、leave old claim、
enter/start new，并在选择新 pointer 后才开放 lease；切换中核对同一路径 integrity 和高水位。
失败或选择新边前崩溃按 Core journal pin 的旧完整 artifact 逆向恢复。该窗口没有两名正式 writer。

## 8. 失败、取消、并发与 reload

- **缺依赖：** required Service、default factory、启用 skill-files 时的 `HOST_CHECK`/`SKILLS`、
  `CONTEXT_INPUT`、`PROVIDER_INPUT` provider、`REPLY_OUTPUT` 或 exclusive writer 缺失时 Root sealing 失败，stable 不变。
- **普通错误：** provider、host check、skill view、tool、system section、context part、reply part、Session commit 和 delivery 保持现有错误分类；
  只有拥有恢复动作的边界转换错误。
- **取消：** 当前 attempt 收到取消；具体 Agent 完成工具/外部效果既有 settle，只提交一次
  terminal，`TaskControl` 最后移除 opaque record 并释放 lease。reload 后 cancel 仍调用旧
  record 保存的原 owner callback；重复取消幂等，不吞 cleanup failure。
- **并发：** Turn 继续按 session 串行而非全局串行；factory slot 和 provider registry seal 后不可变。
- **热重载：** 新 Root 完整 seal 后才可发布。publication gate 先拒绝新 lease；对每只 identity 改变的
  SwitchPart，Core 等 old owner generation lease_count=0 且 hold_count=0，`TaskCancel` 只能通知原 owner，不能强杀或跳过
  settle。随后才走 old stop/leave → new enter/start → `use_new` pointer choice 并开放新 lease。同 identity
  part 直接复用；没有 shared owner 的普通 generation 不加入 plan。任何 old task 都不能跨过自己依赖的
  owner switch，等待相关长 Turn 是有意的安全代价。SKILLS 的 provider set 和 installed artifact root 跟随 Root；
  workspace skill 与 host availability 仍在原有 lookup 边界重查。skill-files link journal 失败时
  publication 不提交，按 journal 恢复旧 projection。
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

### M1 · 中性 Core 原子

- **M1a · Exact work（已完成）：** 增加 `ServiceCall`、`RootScope`、`TaskControl` 和 private lease source；
  三者接口不增加 Agent/Turn/Session/Scheduler 等领域字段。fixture 证明 single-key/stable-only call、
  owning Root background acquire、跨代 opaque cancel、same-scope claim exclusion、terminal release、错误
  task 继承和退休 Root fail-loud。
- **M1b · Root switch（已完成）：** 增加 `ROOT_SWITCH.add(SwitchPart)`、closed transaction 和 crash recovery；先只用
  隔离 fake part 证明 install/replace/remove、重复 name、candidate 不触发、按 name 的 stop/leave/enter/start、
  start failure 的逆序恢复、恢复 failure 资源留痕、取消保护，以及 crash 落在每个 step 时都依据两代 pin
  收敛到 journal `use_new` 指定的一边；same-path builtin、absent tombstone、selected side、exact config、
  owner 全部 Fiber 的 ordinary dependency closure 和 exact snapshot/generation 都必须连续两次 boot 不漂移。Activity 在 M1d
  迁成 SwitchPart 前与 RootSwitch change 明确互斥；RootSwitch restore failure 的同进程 retry 必须拒绝并
  保持 admission 关闭，installed pointer update 与 recovery 共用一把 pin lock。
  本批不迁现有 production participant，也不写正式共享状态；M1d 必须先删除 Activity 私有 participant，
  M3c 的 skill link 才是第一名业务 consumer，M6 迁 sessions writer，M8a～M8b 再迁 Channel/command 并
  删除 PluginManager 私有 participant table。

- **M1c · Service hold：** 增加绑定单一 ServiceKey 与不可伪造 HoldKey 的 `ServiceHold` 与通用 hold journal，只用 fake Service
  和一次性 Core/source 数据库验证 reserve→source row→activate→call→source outcome→drop→row delete。
  crash 点覆盖 reserve 前后、row 前后、activate 前后、done/abort 前后、drop 前后与 delete 前后；另外固定覆盖
  accept failure、run failure、cancel、no-sink、stream/projection failure、outcome unknown、artifact 缺失、wrong
  HoldKey/ServiceKey、两个 holder 调同一 Service 也不能相互 pending/call/drop、reboot exact call，以及 pending
  hold 与 live lease 一起阻止 RootSwitch。Core 只看到全局 HoldId、HoldKey、sealed ServiceKey 和 exact
  Root/artifact identity，不看到 source、payload、Channel config/binding、delivery 或 outcome。
  fake source 必须证明 done/abort 不泄漏 hold，unknown 必须保留 hold 和 degraded。本批不接 production source；
  M2c 是首批 consumer。

- **M1d · Activity switch：** 在第一名业务 RootSwitch consumer 前完成一次 supervised cold start。existing
  Activity owner 注册自己的 SwitchPart，同批物理删除 PluginManager 的 Activity participant、双 owner 互斥
  guard 与对应 fake；进程尚未开放 lease 时只用该 part 的幂等 `recover(True)` 建立现行 catalog，随后保存
  selected choice。失败保持启动关闭并由下次 supervised boot 重试，不经过旧 Activity finalize/open，也不产生
  双 commit owner。fake + real fixture 覆盖 cold start、install/remove/replace 与逐 step crash。这个批次完成前
  禁止接入 skill link、sessions 或其他 production RootSwitch consumer。

M1a/M1b/M1c 是中性前置能力；M1d 是第一项旧 owner 迁移。M1d 前 Activity participant 明确
`DEPRECATED(CORE)`，同批删除后不留 guard、alias 或兼容壳。

### M2 · Root-bound Agent 入口

- **M2a · Agents：** 只建立普通 `agents` registry、immutable `TurnStart`/`TurnRequest`/`TurnSource` 合同与唯一 factory
  slot。isolated Root 可暂时没有 factory；seal 只有在要发布正式 Agent Root 时才要求恰有一个。M2a 不搬
  concrete Agent、不接流量、不创建 wrapper 或 fallback，只用 fake factory fixture 验证 register/seal/
  create/resume/get、Agent accept/run/finish、InputSeal done/fail/terminal-before-seal/TurnWait close、重复 factory 和 cleanup。
- **M2b · Agent loop：** 把当前 concrete Agent 实现直接移动到普通 `agent-loop` plugin，由它注册默认
  factory。实现只做一次源码移动，不复制第二份算法、不保留旧 import alias，也不创建跨阶段的 deprecated
  factory。新插件此时仍直接依赖尚未迁走的 prompt、tools、models、sessions 等真实旧 owner；这些是后续
  owner 批次的输入，不再包一层兼容对象。
  passive、control、scheduler、wake、subagent 的仓库内入口在这一批一次切到 AGENTS create/resume/get；
  外部 host 只持有绑定 AGENTS 的 ServiceCall。TaskControl 接管 process-wide opaque claim/lease/cancel
  mechanical state；移动后的 concrete Agent 继续唯一拥有 admission、interrupt、terminal 和领域错误。
  每个 source adapter 在边界恰好一次把现有 transport/metadata 解析成 M2a typed fields，以 durable
  TurnSource ref 调 accept，随后 run 或 run 前 finish；control DTO 改名 ControlTurn。concrete Agent 只把
  每项 typed value 交给当前唯一 owner，不重建 raw metadata bag；M3～M6
  逐项替换 owner 调用。Control 与需要流式 Channel 的 source 用 `run(..., watch=true)` drain TurnWait 的 closed
  TurnUpdate stream；其他 source 用 false。所有 source 都必须并行处理 InputSeal；Control 在 done 前锁自己的
  active input，无该状态的 source 立即 done。M2b 同批删除 `_controlItemEvent`、`_controlTurnInputSource`、
  旧 `InputLock` callback、EventBus collector 与 `suppress_stream_events` metadata carrier，不能用 callback bag 过桥。
  source adapter 成为产品 live frame 的唯一投影者：
  物理删除 `ChannelTurnPresentationBridge`及 bootstrap 注册，删除 Web、Telegram、QQ、Mobile 与 Control 对
  `TurnStarted`、`StreamDeltaReady`、`ToolCallStarted`、`ToolCallCompleted`、`TurnOutputCompleted`
  的全部订阅，再删除五种事件类、producer 和注册代码。旧 EventBus 与新 TurnWait 不同时向产品
  consumer 发同一 frame。M2 执行前还要重扫 `$hua-home-server` 正式 artifact；发现任一外部订阅者就阻断
  deploy 并先更新迁移顺序，不为它保留第二条展示链。source 对每个 turn 只发一次 started，逐个
  TurnUpdate 发一次，只在 OutputDone 到达时发一次 output completed，最后按 AgentResult 发一次 terminal。
  M2b 同时把现有 `turns.status/items_json/usage_json/final_response/error_json` terminal row 收窄为 durable AgentResult 合同；
  同一 ref 的 terminal 重入只读取这只 result，不重跑 provider/tool/save。M2b 仍原样保留当前 direct dispatch
  的调用位置和失败语义，不在移动 Agent 的同一批改变 delivery fence；紧接着由独立 M2c 把 terminal commit
  移到 source sink 前。M6a 只把同一 finish/result 能力移到普通 sessions plugin，不改变字节或恢复语义。
  attached validation 的 recursive child 也只用 Core 铸造、绑定 exact candidate Root 的一次性
  `ServiceCall<AGENTS>`。同一 `call(action)` 内，唯一 source adapter 以 candidate artifact/source identity
  构造 TurnSource 和 typed TurnRequest，依次 accept、run 或 run 前 finish，并等待 terminal AgentResult。
  M2b 暂由移动后的 concrete Agent 在 run 内调用 recording Channel、等待 ACK 后才返回 AgentResult；
  source adapter 不重复发送。M2c 删除这处 direct call；此后若 terminal AgentResult 有 output，source adapter
  在同一 action 的外层 exact Root lease 内调用 recording Channel 并等待 ACK，然后 action 返回、ServiceCall
  释放 lease。任一时刻只有一名 caller，M2c 后 agent-loop 零发送。它只使用 candidate 的临时
  sessions 与 recording Channel，PastRead/SaveChoice 仍由 fixture 显式给出，绝不读写正式 workspace 或真实
  sender。这里的 recursive source 是 publication validation，不是第六种产品 ingress；普通 Subagent 仍走
  上一句的正式路径。固定 oracle 必须断言 accept/run/Agent terminal/result/source send/ACK/action return 全在
  同一 exact ServiceCall lease 内，run 前 finish 时 sender call=0。
  Control oracle 还必须逐项比较 live TextUpdate 的 text+thinking、全局 sequence、ItemStart/ItemDone、
  OutputDone 与现有 output completed 相对 AfterStep/save/terminal 的时序，以及 normal、安全拦截、
  上下文过长、模型超时四条路径的达到次数；input source lock 独立对比，不得由 OutputDone 猜测、
  normal 必须是 InputSeal→lock→done→OutputDone，三条 fallback 必须是 OutputDone→executor return→
  InputSeal→lock→done。seal fail/TurnWait close 必须让 Agent 以 input-lock error 终结而不继续 sink，
  terminal items/usage/final output、错误与 cancel；terminal ref 重入的 updates=0，items/usage 与 durable row
  byte-equivalent。固定慢 consumer、projection error、断开、提前 close 和 caller cancel：前四者只切 discard 且
  Agent 终结，最后一项只产生同一 cancel terminal，均不死锁。现有 `items_json` 与 `usage_json` write set
  不得漏列或改格式。live oracle 还必须证明旧 bridge/五种 event 与所有 product subscriber 零 consumer，
  每帧 exactly once。
  同批物理删除旧 AgentLoop 模块、bootstrap fixed executor、所有直接 AgentLoop ingress 和 process-wide
  SCOPED_TURNS owner；不让旧路径跨到 M3。固定 recording
  oracle 必须证明 prompt/tool/session/event/delivery/cancel 与旧基线等价，且同一 Turn 的 exact Root lease
  从 AGENTS accept 持有到 terminal。

- **M2c · Source delivery：** 把 M2b 已切到 AGENTS 的每个 source adapter 同批切到第 7 节的唯一 sink；
  passive、Control、Scheduler、Wake、Subagent、attached validation 和 plugin job 各自只使用自己的既有
  delivery/return/receipt，不增加通用 Delivery Service，也不把 source 名称、sender、callback 或 reply target
  放进 TurnRequest、AgentOutput 或 AgentResult。agent-loop 先调用 old session owner 的 `finish`，取得已经
  durable 的 AgentResult；只有随后 source sink 才能发送或返回。DSH 也先 append 权威 assistant message 再让
  loop 返回（`packages/core/agent-loop/src/agent.ts:410-431`）；这里额外用 source 自己的 receipt 处理进程崩溃。
  这是把**一名旧 agent-loop sender owner** 原子交给已有 source owners 的不可拆切口：若逐 caller 部署就必须
  保留 sender flag，反而形成两名 owner，所以本批可以同时修改多个 caller，但只迁这一项事实轴。它不是并行
  双跑；同一正式 Turn 在 commit 前只选择新 source path。M2c 同批物理删除 `_DispatchOutboundModule`、
  `_ReturnOutboundMessageModule`、concrete Agent direct dispatch、`dispatch_outbound`、默认 sender 与 fallback，
  并让 Agent 普通 direct return，不能保留到 M7，也不能由 M5b 复活。
  每个 durable source 在 admission、Agent accept 之前，还持有 live exact lease 时先用 M1c ServiceHold
  `reserve()` 取得全局 HoldId；再在自己的 ledger/handoff 冻结 HoldId、source generation、Channel
  generation/config、target 和 stable delivery key；最后 `activate()`，active 前 Agent call=0。live call 的外层
  exact lease 和 crash 后的 hold 指向同一 Root。boot 先按 HoldKey + ServiceKey namespace 对账 hold 和 source row：
  reserve 无 row 就由该 HoldKey owner 确认后 drop，row 未 active 就 activate，active 才以同一 TurnSource ref 和原 typed request
  重新 accept/run，从 old session owner 的 `result` 只读 exact AgentResult，再幂等 prepare/send。reboot 使用冻结
  Channel generation/config 创建新的 ephemeral binding/token，不恢复旧 binding。delivery ACK、Control/result
  receipt 或明确 no-sink 先 durable 写 done；accept failure 或能证明零外部效果的 run/cancel 写 abort；
  无法证明是否发生外部效果写 unknown。done/abort 再 drop hold，最后删 row；unknown 保留 hold 与
  degraded。缺 exact Root/artifact、source generation 或 Channel generation/config 时 degraded、sender=0，不得用
  current stable。固定 crash fixture 落在 AgentResult commit 后、source prepare 前，并插入一次 generation switch；
  switch 必须被 hold 阻止，reboot 由 exact old source 和新建 binding 完成，provider/tool/save/observer=0、source
  prepare/send/ACK=1，done→drop→delete 后才允许 switch。没有明确 sink 的 source 只得到返回值，
  durable no-sink 仍写 done 并释放 hold，sender=0。
  第 10 节完整 source matrix、watch updates、normal/error/cancel、ACK/receipt 与 no-save fixture 全部是本批
  deploy Gate；不能延到 M7b。no-save completed 的 AgentOutput.message_id=None，terminal result/recovery/sink
  仍各一次且 Message row=0。
  本批单独 review、name Gate、zero-consumer 和 commit；M7b 只把已经成立的 passive source/sink 搬进普通
  conversation plugin，不再改变 delivery owner 或 fence。

先迁这层不是先重写 ReAct；它把同一 concrete Agent 原样放进普通 Root，给后续 owner 提供真实 inject
carrier。M2 不标记后续 owner 的代码 deprecated；每项 owner 只在自己的批次开始时标记，并在该批验证后
同批删除。这样没有一只从 M2 活到 M7 的内部兼容壳。

### M3 · Host check、Skills、Skill files、Context input、System prompt 与 Provider input

M3 分成七个可独立回滚的实现批次；每批都单独执行第 10 节的两个 implementation review、一个 name
review、deprecated 删除和 zero-consumer Gate，不能把七个 owner 堆成一次大切换。

- **M3a · Host check：** 建立普通 `host-check` service 和二选一 local/bridge provider。只迁现有
  SkillRequirements 的 name partition 行为，不顺带搬 shell/file host bridge。本批先添加并经 ServiceCall/
  isolated fixture 验证，不让 Root 外旧 SkillsLoader 取得临时 bridge；M3c 由 skill-files 成为首名生产
  consumer，并在同批删除 `build_skill_capability_checker()` 和 SkillsLoader 的 host factory import。证明
  同一 bin/env fixture 在 local 与认证 bridge 上得到同一 HostState，且 bridge failure、identity mismatch
  和不完整 partition fail-loud。
- **M3b · Skills：** 只建立普通 `skills` registry、Root sealing 和 `SKILLS.open(SkillCall)`。本批不创建
  filesystem source、link projection、load-skill Tool 或 prompt/context child。固定 provider fixture 证明
  rank/name 胜出、重复名失败、Root-local set 冻结和每次 open 的 lookup 时机。现有 SkillsLoader 仍是唯一
  production owner，到 M3c 才切换；新增 registry 只在 isolated candidate 中验证，不从 Root 外取得 host。
- **M3c · Skill files：** 建立普通 `skill-files` source provider，inject M3a HOST_CHECK 与 M3b SKILLS，迁移
  normal SkillRoot、link projection/journal、candidate check 与 catalog diagnostics。仓库内
  `plugins/computer` 等 normal source 直接注册；live external `skill_roots` 只经过 1.5 exact
  `DEPRECATED(EXTERNAL)` bridge。保持原有重扫时机和 workspace → plugin → builtin 胜出顺序；Root sealing
  只冻结 provider set，每次 SkillView open 才读取 frontmatter、available/missing、always、资源根与 body。
  现有 system catalog、active skills 和 load-skill Tool 直接 inject 同一 SKILLS，不再读 ambient snapshot；
  它们的最终 `skill-use/prompt|context|tool` 注册分别在 M3e、M3f、M4 完成，本批不预建总 skill-use owner。
  验证后删除 SkillsLoader 的 normal catalog、bootstrap 全局 loader、RuntimeSnapshot 的
  `plugin_skill_index`/`skill_catalog_generation_id`，以及 PluginManager 对 normal root/provenance dict 的解释。
  旧 PluginSkillHost 此后只剩明确标记 `DEPRECATED(M3d)` 的 drift lane，不再产 normal catalog。
- **M3d · Drift files：** 不新增 runtime registry。把 drift projection/journal、description/body hash 与
  candidate check 切到 skill-files 的 drift group typed fact；live Emotion `drift_skill_roots` 只经过 1.5
  exact `DEPRECATED(EXTERNAL)` bridge 注册成 drift SkillRoot，绝不进入默认 SKILLS。验证后删除 PluginSkillHost、
  `PluginGeneration.skill_catalog`、PluginManager drift catalog 分支和旧内部类型。旧声明字段本身仅作为两只
  external bridge 的输入留到 M9；Core 内部和新插件零 consumer。
- **M3e · Context input：** 建立普通 `context-input` plugin。Akasha、active skills、Wake/Subagent 的本轮
  动态内容分别注册 `ContextPart` 或随 `TurnRequest` 传 `ContextText`。Active skills 只读 M3b 的
  `SKILLS`；本批首次挂普通 `skill-use` 的 context child，验证后删除旧 ActiveSkillsPromptBlock。Subagent
  直接传自己的 ContextText。Wake 先以自己的 durable attempt ref 调 `Agent.accept`，再在**自己的 source
  边界**运行现有 Content/Drift/Alert prepare/admission；它只返回 `WakeRun(ContextText, skip_sections,
  skip_parts)` 或 `WakeSkip`。WakeRun 才构造 TurnRequest 并 `Agent.run`；WakeSkip 调
  `Agent.finish(start, skipped)`，模型 call=0。select/claim/quiet、Content screening→investigation、Alert、
  Drift、admission transition、异常与取消仍由 Wake 私有 state/Service 拥有，不能进入 ContextPart、Core 或
  agent-loop。Tool Search 不进入 context：它的稳定说明暂留待迁 system owner 到
  M3f，catalog/restriction 到 M4 归 `tools`；host runtime rule 同样暂留待迁 system owner 到 M3f。
  M2 已建立的 source adapter → domain TurnRequest → AGENTS → agent-loop concrete Agent →
  CONTEXT_INPUT.build args 是本批唯一 transport，不使用 ambient state，不重新编码字段。
  Akasha part 从 ContextCall 取得与旧路径相同的 channel、chat identity 和 message time；迁移 oracle 必须
  比较 retrieval query 的 content、prompt history、channel、chat 与时间字段。
  Wake alert/investigate 必须继续以 `skip_sections={long_term_memory}` 和 `skip_parts={memory}` 分别关闭两条
  lane；其他 caller 按真实旧值拆分，禁止混合名单或 metadata bag。`skip_sections` 在本批先由待退役的
  system-only owner 消费，M3f 再原样交给 system-prompt；`disabled_prompt_sections` 的混合 decoder 在 M3e
  删除。
  同批把 `PastRead` 接到 sessions 的一次派生：full/empty 得到 HistoryViews 的 runtime/prompt 两个
  typed projection，分别进入 ContextCall 与当前 provider path；`skip_session_history` metadata 和
  `session_history_read` ambient 解码标 deprecated，新 typed path 验证后删除。M3g 只把已选 tuple 放入
  TurnInput，不重新解释 mode。
  每 Turn 只 await build 一次，当前 concrete loop 按 system → prompt history → context → turn transcript
  的固定次序构造 transcript，provider retry 不重查。Akasha 在此处直接从旧 system 位置切到 ContextPart，
  所以 prompt shape 只改变一次；不增加临时 PromptSection，也不把 current Message/runtime history 塞进
  PromptCall。Core 内 prompt pipeline 的 context fields/slots、`extra_hints` 总汇聚、ContextBuilder 的
  context ownership 和 `PromptRenderCtx` 的内部 context consumer 在同批删除；剩余 system-only pipeline
  标成 `DEPRECATED(M3f)`，只能产生“旧 system 去掉 Akasha section 后”其余 section 逐字节一致的 system
  text，不能再产生 context Message。
  Emotion 只允许留在 1.5 已入账且只产生 `ContextText` 的 `DEPRECATED(EXTERNAL)` context bridge；GitHub
  Watch 只允许一只只读旧 `skip_memory_retrieval` metadata 并映射到本 Turn
  `TurnRequest.skip_parts` 的 `DEPRECATED(EXTERNAL)` adapter。新路径不再写该 metadata。
  Wake fixture 必须逐窗覆盖 crash-before-accept、accept-before-select、select-before-run、quiet、cancel 与
  reload：同一 source ref 总是恢复同一 TurnStart，selection/admission 不重复，quiet 不创建模型 call，run
  只发生一次。通过后删除 Wake 的 `CONTEXT_PREPARED_EVENT` listener 和 BeforeTurnCtx abort/extra_hints
  consumer；Wake source 本身保留，绝不改注册成 ContextPart。
  同批停止把 `llm_user_content`/`llm_context_frame` 写入新 user row，并删除 sessions history reader；旧 row
  字节保留但不再解释。two-turn + reload fixture 必须证明第一 Turn 的 retrieval/hint 只在本轮 context lane，
  第二 Turn 的 prompt history 只含权威保存正文，不重放前一轮 transient context，同时核对新 row 不含两键。
- **M3f · System prompt：** 建立普通 `system-prompt` plugin，把 Veda、identity、behavior、skills catalog、
  session、Markdown SELF/MEMORY、rollout 和仓库内稳定模型指引迁为 named section。Skills catalog 由
  `skill-use` 的 prompt child 只读 M3b SKILLS；Akasha 已在 M3e 属于 context-input，禁止为它增加过渡
  system section。
  M2 已迁入 `agent-loop` 的 concrete loop 成为唯一
  `await SYSTEM_PROMPT.build(PromptCall, skip_sections)` caller；每个 section 声明独立 DropLevel；每 Turn
  恰好 build 一次并冻结 PromptText tuple，provider retry 与同 Turn 后续 tool call 不重建。具体 loop 用
  `SYSTEM_PROMPT.render` 生成与上一批唯一 system-only 基线逐字节相同的 provider system text；M3g 只把
  筛选职责移入 provider-input，仍复用同一个 render，不增加第二只 join owner。
  同批由 system-prompt 从这次 build 的现行 section breakdown 以 typed observe 发布一次 PromptSize；
  producer 不读取 Observe，也不持有 TurnTrace。
  Core 内 `DEPRECATED(M3f)` system-only pipeline、旧 block/builder、system fields/consumer 在新路径验证后
  同批删除。Citation/Meme 只允许保留 1.5 已入账的 `DEPRECATED(EXTERNAL)` section bridge；它们只能向
  SYSTEM_PROMPT 注册 section，不能保留旧 system owner。M9 迁完 Citation/Meme、Emotion 和 GitHub Watch
  外部源码后，物理删除对应 prompt/context/turn-metadata block 与剩余旧类型。
- **M3g · Provider input：** 建立 `PROVIDER_INPUT` 中立合同和两个二选一普通 provider：每 Turn 一次 `open`，
  每 provider attempt 一次 `build` 与一次 `settle`。basic 原样组合；compaction 复用当前
  ledger/provenance/recovery、tool batch、usage 和 overflow retry 算法。Root 必须恰有一个 provider；
  当前 concrete loop 只按 `InputRetry` 决定同一 call 的第二次 attempt，不识别 compaction；M7 只删除
  剩余总 phase 和多余 wrapper，不改变合同。provider-input 按 DropLevel 筛选 PromptText/ContextMessage，并调用
  `SYSTEM_PROMPT.render` 输出 system text。将 mutable
  `ProviderRequestBinding`、pass-through Core fallback 和 `PROVIDER_REQUEST_PROJECTION` 标成 deprecated；
  唯一新 provider 生效后同批删除。
  每次 build 从同一 ProviderInput 以 typed observe 发布 InputSize，call/try identity 与当前 overflow retry
  一致；observer failure 不改变 build/settle。
七批都不新增 before-step、reply edit、通用 middleware 或第二条 prompt/input loop。可选
`session-view` 不在 M2 预建；只有后续 exact fold consumer 证明需要时才增加。

### M4 · Tools owner 与特殊工具退役

- **M4a · Tools：** 普通 `tools` 插件取得 registry、turn-local view、authorize/execute 和 typed outcome
  owner。`TurnRequest.tool_grant/tool_picks/turn_tools` 成为唯一 turn-local 输入；tools 校验 grant、name、
  preload、terminal 与临时 Tool 定义，返回冻结 view。旧 TurnExecutionScope 的 `tool_grant`、
  `tool_overrides`、`preloaded_tools`、`terminal_tools` 和 `tool_source` 同批标 deprecated、切换、删除。
  为保持 live Shell 安全，唯一 executor call site 同时接上 1.5 的 exact external shell block；它仍严格先
  rewrite 后 authorize，其他 tool 不进入。这只是已入账债务，不扩成 tools public phase。
  每次工具结算后由 tools 构造并返回 closed ToolUse；agent-loop 直接用其中同一 ToolOutcome 投给模型、收集
  media，并累积 exact ToolUse tuple 给 ReplyCall。tools 同时以 typed observe 发布同一对象；它不把 telemetry
  塞回 ToolOutcome，也不因 observer error 改写或丢失返回值。
- **M4b · Skill tool：** `skill-use/tool` 只 inject M3b SKILLS 与 M4a TOOLS，注册 load-skill Tool；tool
  boundary 统一 stamp exact Root provenance。验证同一 catalog/body/provenance 后删除 bootstrap 全局
  LoadSkillTool 和旧注册，不改 system/context child。
- **M4c · Tool Search：** 普通 Tool Search 插件只用 TOOLS catalog/search/grant 和自己的 LRU；它注册普通
  tool，返回 typed restriction/view。验证 schema cap、unlock、preload、terminal 与 prompt 结果后删除
  `_META_TOOLS`、`requires_turn_search`、工具名解锁分支和提示拼接；tools 不认识 `tool_search`。
- **M4d · Push tool：** Message Push Tool 直接 inject committed Channel 的窄发送 Service，在 execute 内完成
  独立 outbound Turn、真实发送与 receipt，再返回普通 `ToolOutcome` content；调用参数和同一 receipt content
  仍由 tools 私有 trace 保存。agent-loop 不按 tool name 或 fact type 收集，delivery owner 也不回读
  ToolOutcome。现有 `MessagePushTool` 已拥有 committed dispatcher，`PushToolOutboundPort` 也已能在注册闭包
  固定 passive commit role；本批让 Message Push Tool inject 这只 dependency-bound narrow sender，通用
  ToolInput 只带 call/turn/source/cancel identity。固定 fixture 验证 committed dispatcher、独立 identity、
  media-only push、失败和取消后，只删除 `_commit_role` 注入与 message_push 名称判断。
- **M4e · Media output：** media-producing tools 只返回通用 `ToolOutcome` 中的 typed `MediaItem` content，
  attachment/delivery owner 消费；固定 fixture 比较 path、digest、count、导入失败和发送结果后，只删除
  Reasoner 的 media 汇总字段/分支。
- **M4f · Attention output：** 产生 confirmation 的 Tool 直接 inject Mobile output projection 的窄 Service；
  projection 用 turn identity 拥有 turn-local 状态、失败结算和既有协议发送。`ToolOutcome` 不含 attention，
  agent-loop 不读取该 Service，也不按 fact type 分支。固定协议 fixture 通过后，只删除 Reasoner/Turn 的
  mobile-attention 总字段与名称分支。
- **M4g · Shell：** 把 Shell tool、manager、write/stop tools 和 terminal cleanup 移入普通 Shell product
  plugin；agent-loop 只发送 `TurnEnded`，Shell 以 turn identity 查自己的 execution registry、settle cleanup
  receipt，不能要求 agent-loop 按 Shell 名称 cleanup。listener error 只形成 cleanup_degraded，boot 由 Shell
  registry + SessionRead 恢复。M9 前 M4a 的 exact external
  block 继续保存 Shell Restore/Safety 的 rewrite→authorize 与错误语义；M9 把两段算法并入真正注册
  `shell` 的同一普通 Tool owner后，同批删除两只旧插件、event 和 block。M4a～M4g 每批独立 review、
  name Gate、旧 owner 删除和 zero-consumer 查询。

### M5 · Models 与 old tail

- **M5a · Models：** 复用现有普通 models plugin，保留 provider/model registry、冻结 binding 和 call/stream
  的唯一 owner。具体 Agent 从 `SESSIONS` 读已保存 selection，以 `ModelChoice` 显式传入；models 不读写 Session
  metadata。每次 provider call 直接返回 immutable
  `ModelReply(text, tool_calls, thinking, finish, continuation, use)`；`use` 是同一次
  `ModelUse(turn_id, call_id, choice, usage)`，没有第二份 choice/usage。agent-loop 直接消费 ModelReply，models
  再以 typed observe 发布**同一只** use；observer error 不改变 reply、provider retry 或 continuation。删除
  AgentLoop 的 model metadata 分支和 bootstrap model branch，保留现有 provider/usage 语义。
- **M5b · Old tail：** 只有 M3f 的 PromptSet/PromptSize、M3g 的 ProviderInput/InputSize、M4 的 ToolUse 和
  M5a 的 ModelReply/ModelUse 都已成为 closed return 后，才执行 1.5 的逐项迁移表。一次删除 after-turn phase
  与 M2c 后剩余的八个 builtin module；不等到 M6 再改它们的依赖。零 consumer 的 extra/telemetry/AfterTurnCtx
  直接删除。dispatch/return 两个 module 已在 M2c 物理删除，本批不得恢复 Agent sender。仅保留 private
  `DEPRECATED(EXTERNAL)` old-commit：它只以显式 keyword 接收 typed immutable return，重建完整旧
  TurnCommitted 并保持 exact budget log；没有 Service、phase、registry、observer input、raw Session、
  ContextBuilder 或 metadata bag。旧 session owner 在这一个边界直接返回 immutable HistorySize 和 SavedIds；
  M6a 以 SESSIONS.size/TurnSaved/TurnView 替代后二者立即删除。M5b 后固定顺序是
  `old save/reply → old-commit → old terminal AgentResult → source sink → direct return`。本批与 M5a
  分别执行 review、name Gate、
  zero-consumer 和 commit。

### M6 · Sessions、Reply output 与 Akasha fact

- **M6a · Sessions：** 普通 `sessions` 插件创建 SessionManager 和全部窄 Service；所有其他插件只注入 Service。
  `TurnRequest.save: SaveChoice` 与 `effects: EffectMode` 成为唯一 commit policy；sessions 决定 user/assistant
  write set，并把 EffectMode 映射到现行 typed `effects.post_commit`。`omit_user_turn`、
  `omit_assistant_turn` 和 runtime `PostCommitEffect` metadata decoder 同批标 deprecated、切换、删除。
  本批以 forward-only schema migration 新增 `turn_saves` 与 `saved_notices`。`SESSIONS.save` 总写 immutable
  saved/skipped outcome；saved 在 Message commit 同一事务写 pending row、冻结当时 SavePart recipient identity
  tuple 并返回 TurnSaved，skipped 不写 Message/outbox。M6a 仍由唯一旧 reply owner 在 commit 前完成全部现行
  preprocess/cleanup；M5b 已删除 after-turn phase，所以本批不再搬 module 或建立 phase adapter。
  caller 对 saved/skipped 固定执行
  `SESSIONS.save → SESSIONS.size → old-commit → SAVE_NOTICE.send → live SaveResult observe → SESSIONS.finish → source sink`。
  `SESSIONS.size(SaveResult)` 从当时 provider-facing committed history 用现行 rendered JSON 算法直接返回
  immutable HistorySize；sessions 以 typed observe 发布**同一个**值。SAVE_NOTICE 成功后再 typed observe
  同一 SaveResult。两者的 observer failure 都不改变 save/delivery，
  boot notice replay 不重放 telemetry。
  normal path 只有 `SESSIONS.finish(completed, AgentOutput, items, usage)` commit 后才有 completed AgentResult；此前任一
  business error 由 agent-loop 以同一 TurnStart 恰好一次 finish failed/cancelled/interrupted。finish 自己失败
  就没有可交付 AgentResult，sender=0；finish 成功后的 crash 由 source 按同一 ref 读取 durable result恢复。
  M6b 切 reply-output 后，send 前再插入正式 settle。saved part failure 保留 pending，skipped failure
  保留 part 自己的 prepare；两者都阻止当前 delivery。
  M6a boot 也必须服从全局 recovery barrier：agent-loop 先终结 crash 遗留 Turn，再运行 SavePart ready/
  saved-notice ready；M6b 起在两者之间加入 reply-output ready。conversation delivery 始终最后，不能并行。
  sessions 向 ROOT_SWITCH 注册自己的 writer SwitchPart；stop 必须阻止新 save/notice、等本代 publish call
  到 terminal，并在仍有 pending outbox 时拒绝 switch。每只 SavePart 也注册同名 SwitchPart，以
  `SAVE_NOTICE.pending(part)` 在 remove/replace 前阻止自己的 frozen recipient 跨代；旧 stable 保持 active，
  原 owner 在 reply-output ready 后恢复，不能由新 owner 解释旧 notice。维护窗口 fixture 执行 stop/leave/enter/start，
  并在每个 crash step 重启恢复，证明正式路径任一时刻只有一个 SQLite writer，pointer 未 terminal 时
  只能回到旧 writer。不得由 Core/Bootstrap 保留 sessions 名称分支。
  bootstrap、PluginManager 和工具不再持有 `_store`、任意 repository 或 SessionManager 私有引用。
- **M6b · Reply output：** 只有 M6a 已让唯一 sessions owner 原生返回 SaveResult/TurnSaved outbox 后，才建立
  普通 `reply-output` plugin；当前 concrete loop 在最终 provider reply 后只调用一次 REPLY_OUTPUT.open；
  正文、media 与 attachment 只接 FinalReply，并在 commit 结果确定后恰好 settle 一次。本批在唯一 concrete
  caller 直接固定 SESSIONS.save → SESSIONS.size → REPLY_OUTPUT.settle → old-commit → SAVE_NOTICE.send → live
  SaveResult observe → SESSIONS.finish → source sink；old-commit 已在 M5b 成为唯一旧边界，M7a 不再拆它，也不允许
  临时 TurnSaved builder、新 phase 或 M3→M6 bridge。
  reply-output 合并 FinalReply 后从同一 clean text 以 typed observe 发布 ReplyText；它不读取 Session 或
  Observe，observer failure 不改变 FinalReply。
  先把现有 parser 与 Citation→Meme→Citation cleanup 完整顺序搬进一只 exact
  `DEPRECATED(EXTERNAL)` ReplyPart；它是两只 live artifact 的临时 bridge，不开放新 listener，也不让
  新内部插件使用旧 event。为保持 Citation 当前 Session row write set，bridge 还私有保存旧
  `persist_assistant_metadata`，只让这一处 migration call site 把它作为 `OldReply` 交给 sessions 的
  同一事务；它不进入 FinalReply、REPLY_OUTPUT public protocol、Service 或插件 registry，禁止新增 consumer。
  固定 fixture 逐项比较 raw/clean text、cited ids、assistant metadata、meme tag、media、attachment、
  Session rows、outbound、notice timing 和失败/取消；另覆盖 reload-after-open/before-commit、commit-before-
  settle、part remove/replace 与 process crash。验证后删除 Core 内可变 AfterReasoningCtx 的 reply/media writer
  与 preprocess/cleanup dispatch。两只 live artifact 仍从原 public import path 取得旧 event/type；该 path
  直接由 bridge owner 提供同一符号，不建 alias，也不允许新 consumer。M9 两个外部源码改注册独立
  ReplyPart、Citation ledger 完成历史导入后，同批物理删除 OldReply call site、bridge module、旧 import path
  与 event/type；此后 sessions 只接 FinalReply。
- **M6c · Akasha fact：** Akasha feedback Tool 在返回 `staged` 前，把 immutable remember/forget marker 以
  turn_id prepare 到自己的 plugin-data ledger；不再放进 AfterReasoningCtx 或 user Message extra。维护窗口
  先停新 Turn admission，分别备份 sessions.db 与 Akasha data，把历史 `akasha_reinforce`/`akasha_forget`
  按 message/turn identity 一次性导入，核对 source count/hash 与 ledger receipt 后才切新 artifact；切换后
  旧 row 字节保留但零 reader，临时 importer 与旧 metadata writer 同批删除，不双写、不 dual read。
  Akasha 改为普通 SavePart：handler 以 `(session_key, turn_id)` 幂等提交 prepared
  feedback，并在自己的 DB durable enqueue 一只 projection job 后返回；worker 只用 SessionRead 取得已提交
  TurnView，再执行现有 embed/stage/publish 算法。failed/cancelled/interrupted Agent terminal 用同一 turn identity 触发
  `TurnEnded`，再由 SessionRead 收敛 prepare；boot 扫描 pending marker/job，已保存则 commit/enqueue，
  durable terminal 且未保存则 failed，未知或
  conflict fail-loud。验证 projection、feedback target/boost/reason、post-commit suppress、queue/backpressure、
  source invalidation、error/cancel/reload 后，删除 Akasha 的 AFTER_REASONING_PREPROCESS 与
  AFTER_TURN_COMMITTED listener、TurnCommitted reader、直接 sessions.db read 和新 user metadata keys。
  M6a/M6b/M6c 各自独立 review、name Gate、zero-consumer 和 commit。

### M7 · Loop cleanup 与 conversation source

- **M7a · Loop cleanup：** M2 的普通 `agent-loop` 已完整拥有 inbox、admission、Turn/Step、interrupt/
  cancel/terminal。M2c 已删除 dispatch/return 两个 module，M5b 已删除剩余 after-turn phase 与八个 builtin；本批不再次搬运、包装或改名
  after-turn。它只把 `PassiveTurnPipeline` 中仍存在的 ReAct、stream、tool batch、provider retry 与 commit
  调用点内联回 concrete Agent，并删除 before-turn/before-reasoning/before-step/after-step/after-reasoning 等
  剩余总 phase suite。`agents` 仍只拥有合同/registry/factory slot。
  M5b 的 private old-commit 原样留在 sessions save、REPLY_OUTPUT settle 后和 SAVE_NOTICE.send 前；它的
  immutable input、完整旧 TurnCommitted、exact budget log、listener error/cancel 语义都不变，且进程 crash
  后不重放旧 event。M9 最后一名 committed consumer 迁走时才同批删除 old-commit 与 exact legacy budget
  log；HistorySize、InputSize、PromptSize 和 ModelUse 的 owner-local fact/log 已分别承接仍需保留的指标，
  不留下 after-turn 壳。
  同批由 agent-loop 在 provider/tool 前 parallel 发布一次完整 InputBatch，并从既有 step/raw/terminal
  观察点以 typed observe 发布 LoopStep、RawReply、TurnEnded；producer 对任何 feature name 零分支，
  telemetry observer failure 不改变 loop，InputBatch failure 则在外部效果前停止。
  `TurnRequest.step_limit` 成为 ReAct 上限的唯一输入；最后的 TurnExecutionScope/max_iterations ambient
  consumer 同批删除。此时 behavior metadata 与 ambient scope 全部零 consumer。
  验证后物理删除 `PassiveTurnPipeline` 和全部剩余总 phase 类型；本批不改 source/sink。
- **M7b · Conversation：** 把 durable handoff、command route 和 M2c 已建立的 delivery/ACK 组合放入普通 conversation
  plugin；它只作为 passive source/sink 调用 AGENTS 与 Channel Service，并服从全局 recovery barrier；不拥有
  ReAct、reply receipt、Session writer 或其他来源的发送。`AgentResult` 只是返回值，最终 sink 逐来源锁定：

  | source | 唯一 sink | 成功 fence | 当前等价证据 |
  |---|---|---|---|
  | passive Channel | conversation 调同一 inbound binding 的 Channel | delivery ACK | `bootstrap/passive_worker.py:461-498` |
  | Control | control adapter 返回/stream `ControlExecutionResult`，不调 Channel | control response 完成 | `bootstrap/control_execution.py:133-169,219-231`；现行 direct path 已不 dispatch（`agent/looping/core.py:875-930`） |
  | Scheduler soft job | scheduler 用自己的 delivery service 把 AgentResult 发到 job target | delivery receipt 返回 | `plugins/scheduler/plugin.py:277-316` |
  | Wake | Wake 从自己 durable selection 建 delivery row，并 reconcile/project/settle | durable delivery settled | `plugins/wake/plugin.py:776-779,1250-1288,1332-1375` |
  | Subagent sync | subagent 把 child result 返回 parent Tool | ToolOutcome 返回 | `plugins/subagent/plugin.py:178-216` |
  | Subagent background | subagent 自己的 continuation sink 把 child result交回 origin | continuation submit 完成 | `plugins/subagent/plugin.py:331-365` |
  | attached validation | candidate source adapter 在同一 exact `ServiceCall<AGENTS>.call(action)` 内调 recording Channel | recording ACK | M2b candidate fixture |
  | plugin job | 无通用 sink；submit 只返回 admission receipt，之后只有该 product 明确声明的读或 effect | `ProgrammaticTurnReceipt` 返回 | `agent/plugins/generation_job_host.py:344-421` |

  上表 owner 与 fence 从 M2c 起不变；本批只移动 passive conversation 的 source/sink code 和逐行固定
  `sender call` oracle，不再移动其他 source 或删除一条迟到的 Agent dispatch。对没有上表 sink 的 source，
  AgentResult 只返回 caller，sender call=0。attached action 必须等待 Agent result，再调 recording Channel 并等 ACK，之后才允许
  action 返回并释放外层 ServiceCall lease；Agent terminal 与 TaskControl release 已发生在 AgentResult 返回时。
  run 前 finish 始终 sender call=0。agent-loop 从 M2c 起就完全不知道哪种 source 要发送。
  boot 时以上 sink 全部关闭，严格等 agent-loop recovery → REPLY_OUTPUT.ready → SAVE_NOTICE.ready；随后每个
  source 先恢复自己的 delivery row；若 durable inbound/handoff 还在但 row 尚未 prepare，就以同一
  TurnSource ref 和原 typed request 重新 accept/run。terminal run 只从 SESSIONS.result 返回 exact durable
  AgentResult，provider/tool/save/observer call=0；source 再以该 result 幂等创建自己的 delivery row并发送。
  source 不能从 saved Message、TurnSaved、pending notice 或正文反向猜 payload。delivery ACK 后才 settle/delete
  handoff；随后才开放新 ingress。这样 old-commit/listener 在 save 后失败时恢复的是 durable failed
  AgentResult，而不是误发已经保存但未通过旧 fence 的 reply。
  M2 已让这只 source 在创建 Agent 前调用现有 COMMANDS；本批删除旧 command route，三只 live v3 command
  artifact 的注册、命中结果、短路 write set 与 delivery settle 不变。固定 fixture 证明普通 reply 与
  command short-circuit 都只 commit/send 一次，并逐行证明 Agent call=1、source sink call=0/1、ACK/receipt
  fence 与现行一致。验证后物理删除旧 ConversationRuntime wiring 和
  PassiveMessageWorker 私有业务链；M2 已删除的 SCOPED_TURNS 不在此复活。M7a/M7b 各自单独经过 review、
  name Gate 与 zero-consumer 删除。

### M8 · Core 收口并停止

- **M8a · Channel switch：** Channel owner 注册自己的 SwitchPart；同批删除 Channel participant 分支，
  保持 admission/drain、endpoint publication、delivery fence 和恢复语义。
- **M8b · Command switch：** commands plugin 注册自己的 SwitchPart；同批删除 command participant 分支，
  public COMMANDS contract 和三只 live consumer 不变。
- **M8c · Core close：** 删除已经零 consumer 的 PluginManager participant table。与 M3c skill link、M6
  sessions 合跑现有五类 shared owner 与 durable ReplyPart/SavePart 的组合 install/replace/remove 和逐 step
  crash recovery，证明 RootSwitch 不认识 participant 数量或 feature 名称。Core/Bootstrap 搜索必须证明零 Agent/Tool/Session/feature 插件 ID 特判和零内部
  旧 consumer；运行关键场景、全量测试、静态检查和项目 Gate，并输出最终 topology/write set。
  同时输出所有外部 `move/remove` consumer 的 exact repo/commit/符号清单和仍工作的九类 migration block，
  不把它们误报成干净设计。M8a～M8c 各自独立 review、name Gate 和旧分支删除。
- Draft PR 保持等待维护者；不开始修改独立外部插件仓库。

### M9 · 外部插件收尾

M8 停下并由维护者继续后，M9 仍按 seam 串行，不做一个跨所有仓库的“大收尾”。每个子批只修改列出的
真实插件源码并走正式安装链，不编辑 cache；同一插件跨多个 seam 时也分 commit/release。一个 seam 的
最后一名 stable consumer 的源码改动必须先通过关键测试、两个 implementation review 与 name Gate，再经
正式安装链重装并证明 active generation 已离开旧 import。随后只准备对应 Core block 的删除 diff，按第 10
节完成删除前 review；通过后才物理删除旧 type/event/export 与 fake，再跑 zero-consumer 和必要的 quick
review。不能先删再审，也不能等 M9z 一起删。

- **M9a1 · Citation prompt / M9a2 · Meme prompt：** 两只插件各是一批独立实现、安装与 review；M9a2
  作为最后一名 consumer 离开后按上述删除协议移除 prompt block。
- **M9b1 · Emotion prompt：** 只迁 stable PromptSection，让 context block 只剩本轮 text；独立安装与 review。
- **M9b2 · Emotion context：** 下一批只迁 ContextPart；重装后按上述删除协议移除 context block，不能与
  M9b1 合并。
- **M9c1 · Job data：** 只建立 GitHub Watch job ledger，导入旧 source/repo/item 后停止这些 metadata write；
  历史 message bytes 不改写，独立安装与 review。
- **M9c2 · Skip input：** 下一批只改送 skip_parts；重装后按上述删除协议移除 turn metadata block，不能与
  M9c1 合并。
- **M9d1 · Citation reply / M9d2 · Meme reply：** 两只插件各是一批独立实现、安装与 review；Citation batch
  同时完成 ledger 历史导入。M9d2 作为最后一名 consumer 离开后按上述删除协议移除 OldReply call site、
  reply block 与旧 reply event/import；Meme ReplyPart 还从自己的解析结果以 typed observe 发布 MemeUse，
  该类型由 Meme 插件公开，Core 与 reply-output 都不认识 meme。
- **M9e1 · Feed root / M9e2 · Meme root / M9e3 · Steam root / M9e4 · Huayue root：** 每只插件各是一批
  独立实现、安装与 review。M9e4 按 1.5 的 backup/hash 协议把真实 manual normal directories 收进 Huayue
  source；最后一名 consumer 离开后按上述删除协议移除 agent-root block/旧字段。
- **M9f1 · Emotion drift / M9f2 · Huayue drift：** M9f1 只让 Emotion 注册 drift SkillRoot；M9f2 下一批只把
  manual `explore-curiosity` directory 按 backup/hash 协议迁入 Huayue Skills 的 drift provider。两只外部
  owner 分别安装与 review；最后一名 consumer 离开后才按上述删除协议移除 drift-root block/旧字段。
- **M9g · Shell owner：** 把 Shell Restore/Safety 算法移入真正注册 `shell` 的同一普通 Tool owner，重装后
  按上述删除协议移除 shell block、两只旧 event/import 和已经被取代的外部插件。
- **M9h1 · Emotion facts：** agent-loop 在 provider/tool 前 parallel 发布完整 InputBatch；Emotion durable
  prepare 后注册 SavePart，saved 用 TurnView、skipped 用完整 InputBatch，ready 用 SessionRead.status 收敛。
  它也观察 TurnEnded 来 abort failed/cancelled/interrupted/save=none prepare；multi-input、no-save、Wake selection、
  provider/tool/reply/save failure 与 crash oracle 通过后才离开旧 event。
- **M9h2 · GitHub facts：** GitHub Watch 注册 SavePart，只用 session/turn identity 清理自己的 job/checkout
  ledger；saved/skipped 都运行，保留现有 OSError→TTL recovery，不读取正文或 telemetry。
- **M9h3 · Observe facts：** Observe 注册各 owner 的 typed `observe`，只做 turn-local join；SAVE_NOTICE 成功后
  的 live SaveResult 是唯一 final fence。保持现有单次 nonblocking `writer.emit`、queue full/drop/error 和
  crash 不重放边界，以 old/new TurnTrace row oracle 逐字段证明 agent/wake/drift、saved/skipped、tool/retry/
  cache/Meme 行为，不增加 staging、唯一 receipt 或 delivery gate。
- **M9h4 · Feedback facts：** Proactive Feedback 注册 SavePart；saved 时先 durable enqueue 再返回，skipped
  显式 no-op；user-only 也 no-op，重复 assistant 文本 fixture 必须绑定同一 Turn exact persisted id，不能
  回退匹配旧 row。作为最后一名 consumer 离开后，按上述删除协议移除 old-commit、
  TurnCommitted/EventBus event/import。
- **M9i · Message kind：** Status Commands 保持 COMMANDS 不变，只把旧 `is_context_frame`/provider dict
  consumer 迁到 SessionRead 的 MessageKind；重装后按上述删除协议移除 message-frame block。Plugin Undo
  与 Setup Helper 无此 consumer，只需重装验证 COMMANDS。
- **M9z · Final Gate：** 对当时 hua-home active generation 重跑九类 zero-consumer、跨仓组合 Gate 和完整
  capability fixture；这里不再删除任何 block。只有 1.5 中全部 `move/remove` 清零，且没有 alias、adapter、
  fallback、双路或旧名字，才宣称整个被动链路迁移完成。

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
除合同锁定的唯一 adapter call site 外，Core 内部不得再调用。若出现未入账 external consumer，M8 必须停止并更新账本，而不是把旧入口
解释成长期 public contract。M9 完成后不允许留下任何 migration block 或兼容壳。

## 11. 合格测试

只补能保护现实行为或非平凡边界的测试：

- `ServiceCall` 的窄 key、stable-only public policy、candidate capability identity、exact lease、
  task ownership、cancel 和 cleanup；
- `ServiceHold` 的固定 HoldKey + ServiceKey、全局 HoldId、reserve/source row/activate、exact artifact pin、reboot
  call 与 outcome→drop→delete；crash 遍历每个间隙，accept/run/cancel/no-sink/projection failure 都不泄漏
  hold，unknown 必须保留 hold/degraded。两个 holder 调同一 Service 也不能相互 pending/call/drop；
  pending hold 与 live lease 一起阻止 RootSwitch，wrong HoldKey/ServiceKey/缺 artifact degraded 且不投
  current stable。Core journal 不含 source/payload/Channel config/binding/
  delivery/outcome 字段；source row 不含 live binding/token/socket，reboot 必须新建 ephemeral binding 并复用 stable
  delivery key；
- RootSwitch 以 fake part 覆盖 old/new present/absent 的 install、remove、replace；在 stop/leave/enter/start、
  pointer choice 前后逐点 crash，重启只能按 Core journal 的两代 identity/pin 收敛；缺 artifact 或 recover
  failure 保持 degraded 且零 lease。barrier 把旧 Turn 分别卡在 model、tool、reply open：changed owner
  generation lease_count 未归零前不得 stop/leave，归零后不得再有 old call；unchanged generation 不必 drain。
  skill link、sessions、Activity、Channel、command 与实际 SavePart/ReplyPart production part 各跑自己的资源
  oracle，Core 零 feature name；
- reload-mid-Turn 后用原 task key `/stop`，必须到达旧 owner，产生一次 terminal 并释放旧 lease；
- Scheduler/Wake/Subagent 在 reload-before-fire 与 fire-during-drain 下只取得 owning Root；retired
  admission 单次 settle/rearm，不重复 provider、Session commit 或 delivery；
- Root sealing 对缺 Service、重复 context name、重复 factory、重复 `PROVIDER_INPUT` provider、循环依赖、重复 writer 的 fail-loud；
- control `ControlTurn` 与 Channel 输入各只在自己的 source adapter 解析一次；domain TurnRequest 的字段集合
  不含 metadata，旧 concrete Agent 收到的也是同一只 typed request；旧
  `agent.control.models.TurnRequest` 名称和 context/history behavior metadata decoder 为零 consumer；
- TurnItem 的五种 closed schema 拒绝未知字段、可变 dict/list、重复 JSON key、NaN 和 infinity；
  JsonValue 往返后 immutable value、现有 wire key/顺序与 `items_json` 字节不变。`TurnError.data` 在源码、
  正式 artifact 和正式 DB 均为零后才删；Skill Loader 四项来源往返为 `SkillRef`，空值为 None，
  其他 provenance key 阻断 M2；
- `Agent.accept` 的 reserve→TaskControl claim→mark claimed 三道 barrier 覆盖 reserve 后 cancel、occupied old
  task、claim 后/mark 前 crash、mark 后/return 前 crash 与重复 source ref；run/finish 只能消费同一 TurnStart
  一次。boot 在 AGENTS ready 前只收敛同一 durable identity，冲突保持 degraded；Wake 覆盖 accept-before-select、
  select-before-run、quiet/skip、cancel/reload，selection 不重复且 quiet 的 model call=0；
- InputSeal 是无 payload one-shot rendezvous；normal 固定 seal→source lock→done→OutputDone，安全拦截/
  上下文过长/模型超时固定 OutputDone→executor return→seal→source lock→done。seal fail 或 TurnWait
  close 不死锁、不 reply/save/sink，terminal-before-seal 和 terminal 重入返回 None；watch=false 也必须处理 seal。
  `TurnRequest`、TurnUpdate 和 InputSeal 都不含 lock callback/source metadata，旧 `_controlTurnInputSource` 与 InputLock
  callback 零 consumer；
- `CONTEXT_INPUT` 每 Turn 只 await build 一次；provider retry 不重复 Akasha/skill/hint 查询；顺序固定为
  system → prompt history → context → turn transcript；transcript 以独立 current 开始并按执行顺序保留
  assistant/tool 进展；临时 context 不增加 Session write set；一个 part
  disabled/removed 后只少自己的 Message；每条 Message 保留 source、`trust=derived` 与 context kind，
  provider block 对模型可见且不与 current user 合并；任一 async part 错误或取消时不返回部分 tuple；
  Akasha retrieval 的 content、prompt history、channel、chat identity 与 message time 和旧基线一致；
- two-turn + reload fixture 证明第一 Turn 的 retrieval/hint 只在当轮 ContextMessage，新 user row 不含
  `llm_user_content`/`llm_context_frame`，第二 Turn 的 prompt history 只读权威正文且历史两键零 reader；
- `SYSTEM_PROMPT` 每 Turn 只 await build 一次；provider retry 与同 Turn tool call 不重复读取 section，
  rollout 一次性 fact 也只能消费一次；任一 async section 错误或取消时不交付 partial PromptText tuple；
- skip section/part 的 builder invocation count 必须为 0；unknown skip 是 no-op；feature disabled/removed 后
  只失去自身输入，不让无关 Turn 失败；
- PastRead.full/empty 分别让 HistoryViews.runtime_history/prompt_history 得到同源完整/空选择但保持
  不同类型，Session rows/write set 均不因读取模式改变；programmatic validation 的
  session_history_read false/true 行为等价；
- system/context 的 extra 必须先于 repeat 移除，repeat 必须先于 prompt history 缩窗，keep 不得移除；
  改名或改变展示 order 不得改变 drop，Core/provider-input 零 feature name 分支；未超窗时除已批准移到
  context 的 Akasha section 外，其余 system section 文本、顺序和分隔符逐字节不变；
- sessions 单 writer、原子 save、messages 只追加、seq、restart recovery；SessionRead.turn/history 返回 exact
  ordered MessageView/HistorySize；SessionRead.status 覆盖 save=none/saved/skipped 与 running/五种 terminal，
  saved/skipped channel、TurnSaved、ended_at 或 state 组合冲突 fail-loud，未知才返回 None；
- `REPLY_OUTPUT` 的每个 part 只看同一 ReplyCall；disjoint mark 与 trailing hidden marker 得到确定性
  FinalReply，overlap/越界/part failure 不导入附件也不写 Session。Citation+Meme exact fixture 比较正文、
  cited ids、tool fallback、meme tag、media、attachment、OldReply metadata、Session/outbound；raw path、
  错 Root/插件 provenance 的 MediaItem 在导入前拒绝；saved/failed/cancelled/interrupted 恰好 settle 一次，prepare crash
  通过 SessionRead 收敛；durable ReplyPart 缺少/错配同名 SwitchPart 时 Root sealing 失败，pending receipt
  阻止 remove/replace，pure part 不得假注册；
- crash 注入覆盖 InputBatch prepare 后、reply prepare 后/Session save 前、saved/skipped result 后/settle 前、
  settle 后/delivery 前；重启时必须先把 queued/running Turn 终结为 cancelled/interrupted，provider reboot
  call=0，再收敛 pending reply receipt；recover failure 保持 degraded 且 sender call=0；
- crash 注入精确落在 durable AgentResult commit 后、source delivery row 首次 prepare 前；重启后同一
  TurnSource ref 取得 byte-equivalent AgentResult，provider/tool/save/observer=0，source prepare/send/ACK=1，
  ACK 前 mobile handoff 保留、ACK 后才删除；result 缺字段、ref/turn 冲突或非 terminal 都 degraded，不从
  Message/TurnSaved/notice fallback。admission 必须按 reserve→source row→activate；row 冻结 HoldId、source
  generation、Channel generation/config、target 与 stable delivery key，不冻结 binding/token。gap 中 generation
  switch 被 hold 阻止，reboot 使用 exact old owner 并新建 binding，先写 done、再 drop、最后删 row 才能换代；
- saved commit 冻结 exact SavePart recipient identity；两个 part 并发 accept 时一个失败不取消兄弟，outbox
  pending 且 sender=0；partial success crash/replay 不重复可观察结果。pending recipient 阻止对应 part
  remove/replace，任一 pending 阻止 sessions writer switch，新安装 part 不接旧 notice。skipped live call
  failure 同样 sender=0，process crash 后各 part 以自己的 prepare + SessionRead.status 收敛；boot 顺序固定
  stable/session ready → agent-loop crash terminal → reply receipt → SavePart ready/saved notice → source delivery；
- SaveChoice 覆盖 user+assistant、assistant-only、neither 三个现有场景；turn_saves 对 saved/skipped 恰写一次，
  neither 的 messages/outbox=0 且不伪造 TurnSaved；EffectMode.run/skip 保持现行 behavior，旧
  omit/post-commit metadata 零 consumer。neither completed 的 AgentOutput.message_id=None，durable result、
  terminal reentry 与 source sink 仍各一次，delivery key 只来自 TurnSource/source ledger；
- 同一固定场景在旧基线 artifact 与新代码上**依次**运行；provider payload 只允许 1.3 的三项
  prompt-shape 差异且逐字段入账，其他字段相同；tool trace、Session rows/write set、批准后的窄 fact 映射、stream、
  delivery/ACK、attachment、error/cancel/interrupt 与能力结果等价；
- M2c source matrix 逐行证明 passive/Control/Scheduler/Wake/Subagent/attached/plugin job 的 Agent call、sink
  call 与 ACK/receipt 数量；Control 与流式 Channel 的 TurnWait updates 保持 TextUpdate 的 text+thinking、
  跨种类严格递增 sequence、ItemStart/ItemDone 配对、OutputDone 对 normal/安全拦截/上下文过长/
  模型超时一比一保持现有达到时机，不代表 input source lock；terminal items/usage/final output 与现有结果等价，
  terminal ref 重入 updates=0。slow/projection error/disconnect/early close 切 discard 后 Agent 仍 terminal，caller
  cancel 只结束同一 Turn；旧 Channel bridge、五种 lifecycle event/producer/registration 和全部 product subscriber 零
  consumer，started/每个 update/每个已到达的 output completed/terminal 各 exactly once；旧基线未发
  TurnOutputCompleted 的 failure/cancel 仍为 OutputDone=0。attached 的
  accept/run/Agent terminal/AgentResult/recording send/ACK/action return
  全在同一 exact ServiceCall lease，TaskControl 在 AgentResult 时释放、ServiceCall 在 ACK 后释放，run 前 finish
  零发送；删除 direct dispatch 后 agent-loop、TurnRequest 和 AgentResult 都没有 sender flag 或 source-name 分支。
  M7b 只重跑 passive handoff/command/Channel code-move 等价，不延迟验证其他 source；
- Tool Search 只通过 scoped tool view 工作，改名 tool 后仍工作；Core 和 `tools` 不认识 search/grant/unlock；
- `ToolOutcome` 只接受 call_id/status/model-facing content，构造时拒绝 fact、delivery、attention、metadata
  与 callback；TOOLS.run 返回的 ToolUse 只增加 turn/name/validated args identity，并嵌入同一 ToolOutcome。
  agent-loop 累积 exact ToolUse 给 ReplyCall，typed observe 收到同一对象且失败不影响该业务通路；agent-loop
  对任意工具零 tool-name/fact-type 分支。Message Push Tool 自己完成 committed send
  和独立 outbound Turn，tools trace 保留同一 receipt content；Mobile output projection 自己完成
  confirmation 的记录、失败恢复与协议发送；
- Shell exact fixture 保持先 restore rewrite、后 safety authorize：比较 final args、deny text、ToolOutcome、
  invoker call count、restore path/file 与错误恢复；M4 bridge 和 M9 Shell owner 依次对同一 fixture 得到同一
  结果，非 shell Tool 不经过该算法；
- ToolGrant/ToolPick/turn_tools 固定场景保持 visible/preload/terminal/override/source 与 step limit；旧
  TurnExecutionScope behavior fields 和 metadata decoder 最终零 consumer；
- basic/compaction `PROVIDER_INPUT` provider 用同一固定 prompt history 得到已批准的不同 projection；无 provider 和双 provider 都
  fail-loud，没有 Core pass-through fallback；
- basic/compaction 都保留 system → prompt history → context → turn transcript lane 顺序、transcript 内执行
  顺序，以及每条 Message 的 role 与既有 `normal|context` kind；
  compaction 只按 CTX-002 减少可再生 context 或完整 prompt history 边界，禁止重排/合并 lane；
- fixture 覆盖首 call、tool batch 后 call、空回复/结构化终态的后续 call、context overflow 的同 call
  第二次 attempt、done/failed/cancelled settle、usage 计量、checkpoint fact 单次发布和 crash 后 receipt
  补发；每只 `InputReceipt` 只能 settle 一次，禁止第三次 attempt；
- feature plugin disabled/removed 后只失去自身 part，不触发 Core fallback；
- Akasha marker 迁移在停 admission、sessions/Akasha backup 后核对 source count/hash；feedback prepare、
  TurnSaved commit/enqueue、worker SessionRead projection、TurnEnded failed/cancelled/interrupted、boot pending recovery 各有
  crash barrier，切换后旧 keys 保留字节但零 reader/zero writer；
- TurnEnded 对 completed/failed/cancelled/interrupted/skipped 各只 live 发布一次；consumer failure 只记 cleanup_degraded，
  terminal 不回滚且 TaskControl 仍 release；进程崩溃后 Akasha/Shell 只用自己的 pending + SessionRead 收敛；
- Observe 对 InputSize/PromptSize/HistorySize、ModelUse、ToolUse、LoopStep/RawReply、ReplyText、MemeUse 都用
  observe；任一 handler failure 不改变 tool outcome/save/delivery，缺 required fact 时丢整条 trace 并记 Incident。
  old/new TurnTrace row 逐字段比较 source、raw/clean reply、meme、tool chain、history/prompt、ReAct input、model
  output/cache，并覆盖 saved/skipped、queue full、observer error 与 final fence 前 crash 均不补造 trace；
- Emotion 的完整 InputBatch parallel staging + SavePart 覆盖 multi-input aggregate、persisted/non-persisted user、
  explicit quote、presence、skipped crash recovery，以及 Wake selection 在 save consumer 后才 settle；source
  content/selection 不进入 TurnSaved；provider/tool/reply/save failed/cancelled/interrupted 由 TurnEnded abort，observer
  failure 后 boot ready 仍清掉 receipt；
- M5b old-commit 在 Session save 与 reply settle 后、save notice/delivery 前恰好生成一次旧 TurnCommitted；全部字段、deep copy、
  EventBus→AFTER_TURN_COMMITTED 顺序、listener error/cancel 对 delivery 的影响与旧 artifact 相同。M9 四名
  consumer 改读窄事实后，old-commit、旧 event/type/import 做 zero-consumer 删除；
- fault injection 覆盖 provider/tool/commit/delivery/cleanup 的真实失败边界；
- zero-consumer 和 forbidden-token Gate 证明没有旧入口、名称特判、双写或 compatibility flag。

不为常量映射、显然控制流、已删除功能的内部形状或覆盖率数字补测试。并发测试使用 barrier/event，
不用 sleep。比较测试不接正式 workspace 或真实不可逆 sender，也不称为 shadow。

## 12. Concept Gate

第一阶段 reviewer 只回答以下四项，不能扩成一般代码风格 review：

| 问题 | PASS 标准 |
|---|---|
| 足够正交？ | 每个事实只有一个 owner，变化轴之间没有强制联动或万能 context |
| 足够原子？ | Core atom 只有 composition/publication/live lease/跨进程 hold/泛型 call；业务能力可直接组合且没有 feature-shaped Core API |
| 是非特权插件？ | 骨架能力与其他插件使用同 loader、权限、lifecycle 和 failure；Core 无 ID/名字/fallback |
| 整条链走得通？ | passive/control/Subagent 与 attached validation child、完整 snapshot、save、delivery、cancel、reload 和单 writer 都有闭合路径 |

P0/P1 任一非零即 `BLOCK`。旧版合同因把 `session-view` 当成 prompt history owner、把 `agents` 与
`agent-loop` 分成两个半 driver，以及新增 `ReplyEdit`，已被 DSH 对照复核判为 `BLOCK`。M2 审计随后发现
动态 context Message 缺少明确 owner；本版按 DSH 的独立 context registry 与 AgentLoop 固定组合边界补入
`context-input`，又按 live external consumer 补齐 `reply-output`、RootSwitch crash recovery 和九类 exact
migration block。

2026-09-02 最终 Gate 记录如下。正文哈希是 reviewer 实际读取的字节；状态和本表在 PASS 后落盘，因而不计入
该哈希。

| Gate | reviewer | 模型 | 实现 head | 送审正文哈希 | 结论 |
|---|---|---|---|---|---|
| Concept | `dsh_hooks` | `gpt-5.6-terra` xhigh | `df06efa1809dc181cf3465825f1aeaed4e89cec7` | design `efd00ebc8b8ad47b966b5fced5f2fe7274b18187a302f117f22dcf9068fd5800`；projectneed `b9b970e1094908deb0e17f9c96bec85e3c2273b459e2f924201705e5385db0b6`；ADR `d73808b1d7a8c95028610b0947d81331037e5d68a9f72ff7c0e88a4d5c195729` | `CONCEPT PASS`；P0=0，P1=0 |
| Name | `name_gate` | `gpt-5.6-terra` xhigh | 同上 | 同上 | `NAME PASS` |

Concept 复审先后关闭五类 P1：ServiceHold 的跨进程身份与泄漏、TurnWait 的完整 stream/关闭语义、开放
metadata bag、把 OutputDone 误当 input lock、以及 source input lock 缺少闭合握手。最终合同以 HoldKey/HoldId
和 durable outcome、封闭 TurnItem/JSON/TurnError、TextUpdate/ItemStart/ItemDone/OutputDone、独立 InputSeal
及 TurnWait.close 闭合这些路径。Name 复审把与现有类型冲突的 `SkillSource` 改为 `SkillRef`，并把可见思考
统一为 `thinking`、`TextUpdate` 和 `ThinkingItem`。两名 reviewer 最终均无未关闭 finding。

该结论批准 M0 设计并允许开始 M1b；它不能代替 M1b～M9 的实现 review 与行为 Gate。

## 13. 交接边界

本 Core PR 交付通用内核、最小普通 Agent 能力图、仓库内置 conversation/feature 组合和旧私有链删除，
并保留 1.5 中可枚举、不可新增 consumer 的 migration block。它是 M9 前的停靠点，不是最终架构。
上面列出的 `hua-home` 外部插件记录 exact repo、consumer、版本和阻塞点；Core 阶段停下后另开源码
迁移。M9 删除 migration block 后才完成整体交付。禁止直接修改 cache 伪造完成。
