# Codex 式同 Turn 输入设计与任务合同

- 状态：accepted / implemented
- 日期：2026-08-06
- 需求：[Codex 式同 Turn 输入需求合同](codex-style-same-turn-input-requirements.md)
- 决策：[0025](../decisions/0025-codex-style-same-turn-input.md)
- 参考版本：`codex@2b5bdcf67547860f2e5c5a605009a70026796b2b`

## 1. 设计结论

采用 Codex 的 durable history continuity，但把 user-visible turn 明确定义为 logical interaction：每条普通输入只在没有 active attempt 时创建 execution attempt；最新 interaction 尚无最终 A 时，新 attempt 自动续接。active 时只允许 interrupt，系统不提供 user steer。

当前 `ConversationRuntime` 继续拥有 session lane、attempt identity、interrupt 和 terminal CAS，并用 `interactionId`、`attemptOrdinal`、`continuedFromTurnId` 连接 attempt。`DefaultReasoner` 把前驱 attempt 的有序 U 和已闭合工具调用/结果投影进下一次 prompt。`PassiveMessageWorker` 只负责 durable handoff 和最终 outbound；中止 attempt 不产生 outbound A。

## 2. 状态与调用合同

`ConversationRuntime.start_turn(request)` 只有两种结果：没有 active attempt 时创建新 execution attempt；存在 active attempt 时返回 `ThreadBusyError`，不写入任何 user item。`turn/steer` 不属于协议。

Mobile active 时发送入口关闭，只能调用 `turn.stop`；中止终态完成后，下一条普通 U 调用 `turn/start` 创建新 attempt。其他 channel 的 `/stop` 使用相同路径。turn-local input source 只承担 final seal 和恢复既有有序输入，不再拥有运行中输入 ingress。

## 3. Reasoner 行为

新 attempt 启动时，runtime 沿 `continuedFromTurnId` 回溯同一 interaction，恢复全部 user item，并把此前每个已完成 tool item 投影为 assistant tool call + tool result。每个 attempt 的 partial assistant delta 和没有 result 的工具不进入 replay；当前 U 始终是最后一条 user message。

provider 返回 tool call 时，先完成整个 tool batch并持久发布工具 item。provider 返回自然语言时原子 seal；成功后当前自然语言成为唯一 `A_final`。active 期间不存在 pending U。

已经实时发送的中间 delta 属于 attempt 流展示，不拥有 interaction 完成语义；只有 terminal assistant item 和最终 transcript commit 才关闭 interaction。

## 4. 持久化

### `turns.items_json`

- 正常增加：每个 attempt 创建时写当前 U；工具和最终 item 按现有 owner 写入。
- 原位更新：仅 active turn 的 items append 和既有状态 CAS。
- 逻辑失效：terminal 后 input source seal，旧 generation 不再写入。
- 物理减少：只随用户显式删除 session cascade。
- 恢复证据：attempt ID、interaction ID、前驱 ID、item ID、logical ordinal、status 和 tool item。

### `messages`

- 正常增加：completed interaction 一次 INSERT `U1..Un,A_final`。
- 原位更新：本功能不允许。
- 逻辑失效：不因 interrupt、Akasha 或 context projection 改变。
- 物理减少：只按 SES-003 的显式撤销/删除。
- 恢复证据：seq、message ID、共同 `control_turn_id`、input ordinal 和 terminal 标志。

### runtime input source

- 只投影从 durable predecessor 恢复的全部 U 和当前 attempt 的初始 U。
- active attempt 不接收新 U；seal 后释放。它不是真源，不能覆盖 turns 或 messages。

### Akasha sidecar

- 在线事件只在最终 attempt 成功提交时携带全部有序 user message IDs 与 final assistant ID。
- builder 对新格式按 `control_turn_id` 分组，对 legacy 数据保留严格相邻 pair。
- 多 U 文本按 ordinal 连接；dense 使用所有非空 user message embedding 的确定性归一化均值。缺失任一必需 embedding 时保持现有 fail-loud audit。
- interrupted/cancelled/failed attempt 只存在于 `turns`，不成为 Akasha 样本；Akasha 仍是可重建派生状态，不反向修改 SessionDB。

### Session history replay

- `max_messages` 表示逻辑历史单元数量，不再表示物理 message 数。一个显式 `U1..Un+A_final` 是一个单元；一次 proactive assistant 是一个单元。
- consolidation 保留尾部、分页、积压阈值、recent turns 和 `start_index` 使用同一分组。窗口若落在显式 `control_turn_id` 中间，必须退回 U1；展开后的 provider message 数允许超过预算。
- legacy 消息继续使用既有 user/proactive assistant 边界规则，不用相邻角色反推新格式 turn identity。

### Attempt replay 和 session compaction Gate

前驱 attempt replay 在 prompt 中按顺序保留 `U、assistant tool-call、tool result、interrupt marker`。runtime 把每个已闭合 tool-call/result 识别为 0030 session Gate 的可压缩批次；达到冻结模型的 74% soft watermark 后，旧批次进入 session ledger summary，最近完整批次和当前未闭合后缀保持原文。

摘要的 current-query anchor 不是最后一条 U，而是本 interaction 的全部 `U1..Un`。摘要无效、工具批次未闭合、切点与 `prior_tool_chain` 数量不一致、压缩后仍越过硬水位时都 fail-loud。`turns` checkpoint、最终 `messages.tool_chain` 和既有消息正文不因 provider 投影压缩而 UPDATE 或 DELETE。

这一选择结合两类参考实践：`pi-mono@74caa2649f10ed71b4378ce69f5d9fbfd2466ca5` 保留 append-only session，并用 summary + recent suffix 构造 prompt，切点不会落在 tool result；`maka-agent@785b0c4f202c0263fb59150ae903195932233466` 以 source-bearing checkpoint、coverage/digest、head anchor 和可读取 archive 保住 ledger/replay 一致性。本实现采用共同原则“权威账本完整、模型视图有界、切点只在闭合因果边界”，但本 PR 不新增通用工具结果 archive。

### Interaction 完成后的下一轮

以 `U1 → stop → U2 → stop → U3 → A1` 为例，最终 SessionDB 有四条连续 message，共享一个 `control_turn_id`。下一轮 U4 读取历史时，I1 作为一个逻辑单元展开：模型看到 U1/U2/U3、I1 的 compact summary 与最近工具后缀、以及 A1，然后看到 U4。I1 超出热窗口后，Markdown consolidation 会把其用户 query 和确认事实写入派生记忆，Akasha 可按一个 completed node 召回；SessionDB 的四条权威消息仍保留。

### Proactive

已成功送达的 proactive assistant 没有 user turn，也不生成 Akasha 学习节点；它在 canonical
history、prompt history、Markdown exact source plan 和 consolidation 边界中作为一个
独立逻辑单元。用户随后回复 proactive 时，回复 metadata 继续保留引用，新的被动
interaction 仍按正常 U/A 规则学习。本设计不把 proactive assistant 伪造成 Akasha 的 U 或 A。

## 5. Channel 和 UI

- Mobile：active 时无论草稿是否为空都只显示中止，草稿保留但发送不可用；中止收束后恢复发送。中止继续调用 `turn.stop`。
- Telegram、QQ、Web Chat：`/stop` 或现有 stop command 结束 active attempt；终态后的下一条普通消息自动续接未完成 interaction。
- Programmatic control：`turn/start` 在 active 时返回 busy；只有 `turn/interrupt` 能改变 active attempt。

## 6. 失败语义

- turn item 写入失败：输入 admission 失败，不进入内存 queue。
- active thread 收到普通 `turn/start`：返回 busy，不写 checkpoint，不 fallback 到新 turn。
- source 已 seal：旧 attempt 不再接收输入；terminal 后下一次 `turn/start` 创建新 attempt。
- tool batch 未闭合：不进入 attempt replay，也不能成为 compaction 切点。
- completed transcript batch 写入失败：turn 不得声称正式提交，现有错误向 owner 暴露。
- Akasha 失败：completed transcript 保持，sidecar degraded 并从固定输入重建。
- hard interrupt：沿现有唯一 interrupt owner 结束 attempt，不提交 canonical assistant；下一条 U 由 runtime 根据 durable 前驱续接 interaction。
- 单个 logical interaction 过大：先压缩旧闭合工具组；如果全部 U anchor 或最近不可拆后缀本身超过 provider 硬边界，明确失败。这是已接受风险，不能通过拆 U、删工具证据或伪造摘要规避。

## 7. 验证

- runtime：`U1/stop/U2/stop/U3/A` 的 interaction identity、attempt 前驱、logical ordinal、stale attempt 和 hard interrupt。
- reasoner：前驱 U、十个工具调用/结果、abort marker 在当前 U 之前完整可见；未闭合工具不重放。
- persistence：关闭重开 SQLite，核对 turn items、completed message batch、只追加 write set 和 seq。
- replay：`max_messages` 或 consolidation index 落在 U2/U3/A 时仍从同一 turn 的 U1 开始投影。
- compaction：两次中断留下的闭合工具组可以被压缩；摘要输入同时包含 U1/U2/U3，最近组保持原文，最终 persisted cut 不超过聚合后的完整 tool chain。
- proactive：主动 assistant 独占一个历史单元，不进入 Akasha；相邻被动 interaction 保持独立。
- channel：中止 attempt 不发送 assistant outbound；后续消息的新 attempt 最终只发送一次 A。
- Mobile：验证 idle 显示 send，active 空草稿和 active 有草稿都显示 stop，stopping 显示 pending stop；active 时快捷键和 native send 同样被拒绝。
- Akasha：`U1,U2,U3,A` 在线与离线得到一个相同 turn；legacy pair 不回归。
- known-bad：邻接配对、seal 后仍接收、第二 inbound 重复发送 final A、工具批次中途注入。
- known-bad：active `turn/start` 被隐式 steer、按物理 message 数裁掉 U1、consolidation 游标落在 multi-U turn 内、attempt replay 永远不可压缩。

## 8. 实现任务合同

```yaml
change_type: feature
semantic_delta: breaking
capability_owner: mixed
consumer_scope:
  - conversation runtime
  - passive channels and interrupt-only programmatic control
  - shared Mobile WebUI
  - SessionDB prompt replay
  - Akasha online and offline builders
runtime_patch: required
runtime_patch_reason: "只有 core 拥有 active turn、provider/tool 安全边界、terminal seal 和 transcript commit；客户端实现会复制并猜测权威状态。"
authoritative_state_owner: "ConversationRuntime owns admission and seal; SessionStore owns turn checkpoints and transcript; Akasha owns derived projection."
client_only_alternative: "rejected"
invariants:
  - STI-001
  - STI-003
  - STI-004
  - STI-007
  - STI-009
  - SES-002
  - SES-005
protected_state:
  - existing message bodies and seq
  - hard interrupt semantics
  - external tool effects
  - formal workspace data
allowed_paths:
  - agent/control/**
  - agent/core/**
  - agent/lifecycle/**
  - bootstrap/passive_worker.py
  - bootstrap/control_execution.py
  - bus/**
  - session/**
  - infra/channels/**
  - infra/mobile_realtime/**
  - frontend/**/src/**
  - plugins/akasha/**
  - tests/**
  - tests_scenarios/contracts/**
  - docker/debug/**
  - docs/**
forbidden_paths:
  - generated frontend bundles
  - plugin cache
  - formal Akashic workspace
allowed_effects:
  - isolated SQLite fixture writes
  - isolated frontend build
  - isolated Akasha rebuild
forbidden_effects:
  - update or delete existing messages
  - publish, deploy or restart formal services
  - replay indeterminate external effects
validation:
  - focused unit and semantic tests
  - SQLite close/reopen write-set checks
  - Akasha online/offline equivalence
  - shared WebUI build
  - isolated change-impact Gate
rollback: "Restore /tmp/akasic-codex-turn-input-backup.aSq3Tx and revert code consumers; never delete newly appended messages."
worktree_writer: "/mnt/data/coding/akasic-agent-worktrees/pi-style-interruption-design"
handoff_head: ""
external_revisions:
  - "codex@2b5bdcf67547860f2e5c5a605009a70026796b2b"
  - "pi-mono@74caa2649f10ed71b4378ce69f5d9fbfd2466ca5"
  - "maka-agent@785b0c4f202c0263fb59150ae903195932233466"
schema_lineages:
  - "turns.items_json attempt user/tool checkpoints"
  - "messages.extra control_turn_id/turn_input_ordinal/turn_terminal"
  - "TurnCommitted persisted_user_message_ids"
```
