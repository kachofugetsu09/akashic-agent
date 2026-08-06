# Codex 式同 Turn 输入设计与任务合同

- 状态：accepted / implemented
- 日期：2026-08-06
- 需求：[Codex 式同 Turn 输入需求合同](codex-style-same-turn-input-requirements.md)
- 决策：[0025](../decisions/0025-codex-style-same-turn-input.md)
- 参考版本：`codex@2b5bdcf67547860f2e5c5a605009a70026796b2b`

## 1. 设计结论

采用 Codex 的 user-visible turn 语义：普通输入首先尝试注入当前 active regular turn，没有 active turn 才创建新 turn。`steer` 只作为 core 协议术语，不成为用户选项。

当前 `ConversationRuntime` 继续拥有 session lane、turn identity、interrupt 和 terminal CAS；新增 turn-local input source。`DefaultReasoner` 在 provider response 与完整 tool batch 边界 drain。`PassiveMessageWorker` 只负责把 channel 普通消息交给统一 admission，并只为真正新建 turn 的 owner 等待和发送最终 outbound。

## 2. 状态与调用合同

`ConversationRuntime.start_turn(request)` 的返回句柄增加 admission kind：

- `started`：创建新 turn，调用者拥有等待最终结果和发送 outbound 的职责。
- `steered`：输入已进入既有 turn，调用者保留 durable inbound receipt 到共享 turn 终态，但不得再次发送该 turn 的最终 A。

显式 `turn/steer` 要求 `expected_turn_id`。普通 `turn/start` 和 channel inbound 在 active regular turn 存在时自动得到 `steered` admission。active turn 已 seal 时返回 busy；channel worker 等待旧 turn terminal 后把该 U 作为新 turn 重试。

turn-local input source 提供两个操作：

- `drain()`：在完整步骤边界取出当前 pending U，turn 继续接收后续输入。
- `seal_or_drain()`：最终回复候选形成后，在 runtime admission lock 下检查队列；有输入则返回并继续，没有输入则 seal，拒绝任何迟到输入。

## 3. Reasoner 行为

每轮 provider 调用前 drain pending U，并用与普通 current message 相同的媒体与时间 envelope 构造 user message。

provider 返回 tool call 时，先完成整个 tool batch并持久发布工具 item；下一轮开始前再 drain。provider 返回自然语言时调用 `seal_or_drain()`：

- 有 pending U：把当前自然语言作为本 turn 内部 assistant context，追加 U，继续采样；它不成为 SessionDB terminal assistant。
- 无 pending U：source seal，当前自然语言成为唯一 `A_final`。

已经实时发送的中间 delta 属于 turn 流展示，不拥有完成语义；`turn/completed` 和 terminal assistant item 才是最终边界。

## 4. 持久化

### `turns.items_json`

- 正常增加：创建时写 U1；每次 same-turn admission 原子追加 U；工具和最终 item 按现有 owner 写入。
- 原位更新：仅 active turn 的 items append 和既有状态 CAS。
- 逻辑失效：terminal 后 input source seal，旧 generation 不再写入。
- 物理减少：只随用户显式删除 session cascade。
- 恢复证据：turn ID、item ID、ordinal、status 和 tool item。

### `messages`

- 正常增加：completed turn 一次 INSERT `U1..Un,A_final`。
- 原位更新：本功能不允许。
- 逻辑失效：不因 interrupt、Akasha 或 context projection 改变。
- 物理减少：只按 SES-003 的显式撤销/删除。
- 恢复证据：seq、message ID、共同 `control_turn_id`、input ordinal 和 terminal 标志。

### runtime pending queue

- 只保存已经先写入 turn item 的内存投影。
- drain 或 seal 后释放；它不是真源，不能覆盖 turns 或 messages。

### Akasha sidecar

- 在线事件携带有序 user message IDs 与 final assistant ID。
- builder 对新格式按 `control_turn_id` 分组，对 legacy 数据保留严格相邻 pair。
- 多 U 文本按 ordinal 连接；dense 使用所有非空 user message embedding 的确定性归一化均值。缺失任一必需 embedding 时保持现有 fail-loud audit。
- Akasha 仍是可重建派生状态，不反向修改 SessionDB。

### Session history replay

- `max_messages` 和 consolidation `start_index` 只是历史窗口预算，不是 turn 边界。
- 窗口若落在显式 `control_turn_id` 的中间，必须退回该 turn 的第一条消息；因此实际返回量允许超过预算。
- legacy 消息继续使用既有 user/proactive assistant 边界规则，不用相邻角色反推新格式 turn identity。

## 5. Channel 和 UI

- Mobile：active turn 时同时允许发送普通消息和点击中止。发送自动 same-turn admission；中止继续调用 `turn.stop`。
- Telegram、QQ、Web Chat：普通消息自动 same-turn admission；`/stop` 或现有 stop command 继续 hard interrupt。
- Programmatic control：`turn/start` 自动 admission，另提供带 `expectedTurnId` 的 `turn/steer` 供严格客户端使用。

## 6. 失败语义

- turn item 写入失败：输入 admission 失败，不进入内存 queue。
- active turn ID 不匹配：显式 steer 拒绝，不 fallback 到新 turn。
- source 已 seal：普通 channel 输入等待 terminal 后创建新 turn；显式 steer 返回 conflict。
- tool batch 未闭合：不 drain pending U。
- completed transcript batch 写入失败：turn 不得声称正式提交，现有错误向 owner 暴露。
- Akasha 失败：completed transcript 保持，sidecar degraded 并从固定输入重建。
- hard interrupt：沿现有唯一 interrupt owner 结束 turn，不消费尚未进入 provider 的额外输入为新任务。

## 7. 验证

- runtime：start、两次 same-turn admission、seal race、stale turn、hard interrupt。
- reasoner：provider 响应期间到达 U、工具批次期间到达 U、多次 U 后唯一 final A。
- persistence：关闭重开 SQLite，核对 turn items、completed message batch、只追加 write set 和 seq。
- replay：`max_messages` 或 consolidation index 落在 U2/U3/A 时仍从同一 turn 的 U1 开始投影。
- channel：同 session 第二条普通消息不等待第一 turn 完成即可 admission，但只有首 owner 发送 final outbound。
- Mobile：active turn 同时显示 send 与 stop，二者命令不同。
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
