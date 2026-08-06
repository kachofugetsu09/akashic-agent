# Codex 式同 Turn 输入设计与任务合同

- 状态：accepted / implemented
- 日期：2026-08-06
- 需求：[Codex 式同 Turn 输入需求合同](codex-style-same-turn-input-requirements.md)
- 决策：[0025](../decisions/0025-codex-style-same-turn-input.md)
- 参考版本：`codex@2b5bdcf67547860f2e5c5a605009a70026796b2b`

## 1. 设计结论

采用 Codex 的 durable history continuity，但把 user-visible turn 明确定义为 logical interaction：每条普通输入在没有 active attempt 时创建 execution attempt；最新 interaction 尚无最终 A 时，新 attempt 自动续接。`steer` 只保留为严格程序化协议，不成为用户选项。

当前 `ConversationRuntime` 继续拥有 session lane、attempt identity、interrupt 和 terminal CAS，并用 `interactionId`、`attemptOrdinal`、`continuedFromTurnId` 连接 attempt。`DefaultReasoner` 把前驱 attempt 的有序 U 和已闭合工具调用/结果投影进下一次 prompt。`PassiveMessageWorker` 只负责 durable handoff 和最终 outbound；中止 attempt 不产生 outbound A。

## 2. 状态与调用合同

`ConversationRuntime.start_turn(request)` 的返回句柄保留 admission kind：

- `started`：创建新 execution attempt。若前一 attempt 没有最终 A，则沿用其 interaction ID。
- `steered`：只供带 expected attempt ID 的严格程序化客户端；Mobile 不使用该交互。

显式 `turn/steer` 要求 `expected_turn_id`。Mobile active 时发送入口关闭，只能调用 `turn.stop`；中止终态完成后，下一条普通 U 调用 `turn/start` 创建新 attempt。其他 channel 的普通消息由 lane 顺序处理，`/stop` 先结束 attempt。

turn-local input source 继续为程序化 safe-boundary steer 提供两个操作：

- `drain()`：在完整步骤边界取出当前 pending U，turn 继续接收后续输入。
- `seal_or_drain()`：最终回复候选形成后，在 runtime admission lock 下检查队列；有输入则返回并继续，没有输入则 seal，拒绝任何迟到输入。

## 3. Reasoner 行为

新 attempt 启动时，runtime 沿 `continuedFromTurnId` 回溯同一 interaction，恢复全部 user item，并把此前每个已完成 tool item 投影为 assistant tool call + tool result。每个 attempt 的 partial assistant delta 和没有 result 的工具不进入 replay；当前 U 始终是最后一条 user message。

provider 返回 tool call 时，先完成整个 tool batch并持久发布工具 item；下一轮开始前再 drain 程序化 pending input。provider 返回自然语言时调用 `seal_or_drain()`：

- 有程序化 pending U：把当前自然语言作为本 attempt 内部 assistant context，追加 U，继续采样；它不成为 SessionDB terminal assistant。
- 无 pending U：source seal，当前自然语言成为唯一 `A_final`。

已经实时发送的中间 delta 属于 attempt 流展示，不拥有 interaction 完成语义；只有 terminal assistant item 和最终 transcript commit 才关闭 interaction。

## 4. 持久化

### `turns.items_json`

- 正常增加：每个 attempt 创建时写当前 U；严格程序化 same-attempt admission 可继续追加 U；工具和最终 item 按现有 owner 写入。
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

### runtime pending queue

- 只保存已经先写入 turn item 的内存投影。
- drain 或 seal 后释放；它不是真源，不能覆盖 turns 或 messages。

### Akasha sidecar

- 在线事件只在最终 attempt 成功提交时携带全部有序 user message IDs 与 final assistant ID。
- builder 对新格式按 `control_turn_id` 分组，对 legacy 数据保留严格相邻 pair。
- 多 U 文本按 ordinal 连接；dense 使用所有非空 user message embedding 的确定性归一化均值。缺失任一必需 embedding 时保持现有 fail-loud audit。
- interrupted/cancelled/failed attempt 只存在于 `turns`，不成为 Akasha 样本；Akasha 仍是可重建派生状态，不反向修改 SessionDB。

### Session history replay

- `max_messages` 和 consolidation `start_index` 只是历史窗口预算，不是 turn 边界。
- 窗口若落在显式 `control_turn_id` 的中间，必须退回该 turn 的第一条消息；因此实际返回量允许超过预算。
- legacy 消息继续使用既有 user/proactive assistant 边界规则，不用相邻角色反推新格式 turn identity。

## 5. Channel 和 UI

- Mobile：active 时无论草稿是否为空都只显示中止，草稿保留但发送不可用；中止收束后恢复发送。中止继续调用 `turn.stop`。
- Telegram、QQ、Web Chat：`/stop` 或现有 stop command 结束 active attempt；终态后的下一条普通消息自动续接未完成 interaction。
- Programmatic control：`turn/start` 自动 admission，另提供带 `expectedTurnId` 的 `turn/steer` 供严格客户端使用。

## 6. 失败语义

- turn item 写入失败：输入 admission 失败，不进入内存 queue。
- active turn ID 不匹配：显式 steer 拒绝，不 fallback 到新 turn。
- source 已 seal：普通 channel 输入等待 terminal 后创建新 turn；显式 steer 返回 conflict。
- tool batch 未闭合：不 drain pending U。
- completed transcript batch 写入失败：turn 不得声称正式提交，现有错误向 owner 暴露。
- Akasha 失败：completed transcript 保持，sidecar degraded 并从固定输入重建。
- hard interrupt：沿现有唯一 interrupt owner 结束 attempt，不提交 canonical assistant；下一条 U 由 runtime 根据 durable 前驱续接 interaction。

## 7. 验证

- runtime：`U1/stop/U2/stop/U3/A` 的 interaction identity、attempt 前驱、logical ordinal、stale attempt 和 hard interrupt。
- reasoner：前驱 U、十个工具调用/结果、abort marker 在当前 U 之前完整可见；未闭合工具不重放。
- persistence：关闭重开 SQLite，核对 turn items、completed message batch、只追加 write set 和 seq。
- replay：`max_messages` 或 consolidation index 落在 U2/U3/A 时仍从同一 turn 的 U1 开始投影。
- channel：中止 attempt 不发送 assistant outbound；后续消息的新 attempt 最终只发送一次 A。
- Mobile：验证 idle 显示 send，active 空草稿和 active 有草稿都显示 stop，stopping 显示 pending stop；active 时快捷键和 native send 同样被拒绝。
- Akasha：`U1,U2,U3,A` 在线与离线得到一个相同 turn；legacy pair 不回归。
- known-bad：邻接配对、seal 后仍接收、第二 inbound 重复发送 final A、工具批次中途注入。

## 8. 实现任务合同

```yaml
change_type: feature
semantic_delta: breaking
capability_owner: mixed
consumer_scope:
  - conversation runtime
  - passive channels and programmatic control
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
schema_lineages:
  - "turns.items_json same-turn user items"
  - "messages.extra control_turn_id/turn_input_ordinal/turn_terminal"
  - "TurnCommitted persisted_user_message_ids"
```
