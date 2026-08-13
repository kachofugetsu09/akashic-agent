# 移动端投影审计：自造语言与重复 owner

- 状态：current（审计结论，修复路线待维护者批准后进入实施）
- 日期：2026-08-13
- 审计对象：`kachofugetsu09/akashic-mobile` `main`（head `aa87e10`，含未提交 Theme 改动）
- 对照基准：本仓库 `main`（head `bb60a749`，刚合并 #391）、[MOB-001～MOB-008](../projectneed.md)、[0034](../decisions/0034-turn-is-the-logical-work-unit.md)、[0019](../decisions/0019-mobile-long-messages-use-bounded-events.md)、[0020](../decisions/0020-mobile-history-content-uses-authenticated-http-ranges.md)、[0023](../decisions/0023-akashic-tokens-own-material-3-semantics.md)
- 引用符号以审计时点为准；行号只用于当次审查。

## 1. 审计基准

服务端（本仓库）是唯一权威：SessionDB、协议 schema、turn 身份、canonical 消息身份、WebUI 源码与 theme catalog 都在本仓库。移动端只应持有两类状态：

1. 可重建服务端投影：会话、消息、turn block、事件、历史和附件元数据，可由固定协议重新投影，重建时允许按正向白名单清理。
2. 明确归属的本地工作：outbox、草稿、附件 draft/transfer、通知待办、pending stop 等，不得随投影重建删除。

凡是移动端自造平行概念、自建 ID 命名空间、自设权威状态机，或两条路径声称拥有同一事实，就是审计要报告的重复 owner。

## 2. 刚合并 PR #391 的评估

#391 按 0034 给所有移动实时事件补发 `control_turn_id`，方向正确。但消费链是断的：

- 移动端 Kotlin 全文没有 `control_turn_id` 消费者（`WireEnvelope` 只有 `turn_id`，payload 透传）。
- 移动端 WebUI（本仓库 `frontend/chat/src/mobile-native.tsx`）只读 `turnId`，不读 `control_turn_id`。
- 移动端 `runtime-contract.lock.json` 与 `protocol/source.json` 仍固定 core `0e37d681`（#391 之前），协议快照落后。

因此 #391 修复的"replay 保持逻辑 turn 身份"对当前移动端组合无效：移动端仍以 attempt `turn_id` 构造本地消息主键（`assistant:$turnId`，`LocalDeliveryStore.kt:954`），interrupt 链上身份仍然漂移，canonical 合并仍靠正文加时间窗的启发式。

## 3. 移动端自造的平行语言

每项都是服务端已提供权威语义、移动端在旁边另造一套：

| # | 移动端自造概念 | 代码位置 | 服务端已有权威 |
|---|---|---|---|
| L1 | `assistant:`/`user:`/`proactive:`/`ephemeral:` 本地 ID 前缀 | `LocalDeliveryStore.kt:954`、`RealtimeSession.kt:1245`、`RealtimeEvents.kt` | `message_id`、`client_message_id`、`delivery_id` |
| L2 | `canonicalMessageAliases` 进程内别名表，上限 256 | `LocalDeliveryStore.kt:80,1240` | 0034：用 `control_turn_id` 做 canonical 合并 |
| L3 | `transientLocalSourceId`：按正文 + ±1h 时间窗启发式匹配身份 | `LocalDeliveryStore.kt:832` | `delivery_id`/`client_message_id` 稳定身份 |
| L4 | `tool.v1:` 内容编码；`history:`/`tool:`/`thinking:` block ID 命名空间 | `LocalDeliveryStore.kt:874,996` | core 生成的 `block_id` |
| L5 | `toolCallId` 三候选键 `tool_call_id/call_id/tool_id` | `LocalDeliveryStore.kt:1313` | schema 固定字段名 |
| L6 | `FinalMessageAttention`（COMPLETE/CONFIRMATION） | `RealtimeEvents.kt:62` | `mobile_attention` metadata 的再解释 |
| L7 | deliveryState 字符串状态机（pending/sent/complete/failed/failed_retryable/outcome_unknown/streaming/interrupted） | `Entities.kt` | outbox + ACK 投影，但无 schema 或枚举约束 |
| L8 | 默认标题"新对话"、`mobile:${UUID}` session ID | `RealtimeSession.kt:2508` | core 明确禁用 `session.create`，session ID 由客户端生成是现行合同，标题另有 `session.updated` |
| L9 | `ThemeColors` token 语言 + 手写三套十六进制色板 | `ui/design/Theme.kt`（未提交） | core `theme-catalog.json` 是唯一颜色真源（0023） |
| L10 | Protocol.kt 硬编码 `knownTypes` 命令/事件集合 | `Protocol.kt:540` | core schema 是真源，Kotlin 反向定义协议（违反 MOB-XREPO-004） |

## 4. 重复 owner 清单

| # | 事实 | owner A | owner B | 冲突 |
|---|---|---|---|---|
| O1 | "同一会话同一时刻只有一个活动 turn" | `TurnStopCoordinator.activeTurns`（进程内 Map） | `LocalDeliveryStore.activeAssistantTurn`（Room SQL） | 同一不变量两处断言，错误文案相同（`TurnStopCoordinator.kt:32` / `LocalDeliveryStore.kt:960`） |
| O2 | 投影清理保护白名单 | `deleteServerProjection`（`sync.reset_required` 用，保护列表含 `sent`） | `deleteReloadableServerCache`（`reloadFromServer` 用，保护列表不含 `sent`） | 同一事实两个答案（`Daos.kt:395-416`） |
| O3 | canonical 消息身份合并 | core：`control_turn_id` + `delivery_id` + `client_message_id` | 移动端：别名表 + 时间窗启发式（L2/L3） | 移动端机制绕过 core 权威身份 |
| O4 | 协议 schema | core `schema/mobile-realtime-v1.json` | 移动端 `protocol/` 快照 + `Protocol.kt knownTypes` | 三份副本，快照固定在旧 commit |
| O5 | turn 终态恢复 | core `reconcile_active_turns`（resume 时按 SessionDB 补发终态） | 移动端 `TurnStopCoordinator` + `pending_turn_stops` | 客户端再对账一套，且不消费 `turn.snapshot` |
| O6 | 主题颜色真源 | core `theme-catalog.json`（0023） | 移动端 `Theme.kt` 手写色板 | 两个色值真源，`mobile.theme` 只同步 ID 不同步 token |
| O7 | 消息/block 模型 | 实体层 `MessageEntity`/`TurnBlockEntity` + deliveryState/kind/status 字符串 | UI 层 `MessageUi`/`AssistantTurn`/`ProcessBlockUi` + 三套独立枚举 | WebUI 侧还有第三套 stream/snapshot 模型；同一事实三层平行表示 |

## 5. Bug 级发现

- **B1（最严重）**：ACK 后 outbox 命令被删（`deleteAcknowledged`），消息进入 `deliveryState='sent'`。此时 `reloadFromServer` → `clearReloadableCache` → `deleteReloadableServerCache` 不保护 `sent`，已确认送达但尚未 canonical 化的用户消息被物理删除；而 `sync.reset_required` 路径的 `deleteServerProjection` 保护 `sent`。同一事实两个答案。若服务端历史此刻尚未落库该消息，消息在 UI 消失，只能等历史重投（serverSeq 已变，回复引用与阅读锚点漂移）。
- **B2**：`FRAME_ID` regex 在 `Protocol.kt` 与 `LocalDeliveryStore.kt` 各复制一份，改动会漂移。
- **B3**：`canonicalMessageAliases` 超过 256 条丢最老别名，后续历史合并找不到 source，落入时间窗启发式可能配错消息。
- **B4**：协议快照与 runtime-contract 固定旧 core `0e37d681`，而 core 已推进（#391、model.catalog 等），当前组合整体落后一代；修 bug 修的是旧协议组合，上游修复到不了设备。
- **B5**：`toolCallId` 用三个候选键猜协议字段，字段名变更会被静默消化成另一个键，问题难暴露。

## 6. 未提交改动观察（审计时点）

移动端工作区有未提交 Theme 改动：默认主题 `system→light`（行为变化）；theme ID 校验从固定集合改为任意 `[a-z0-9-]{0,63}` regex，但 `ThemeSchemes` 只有 3 个键，未知 ID 静默 fallback `light`；手写 WarmPaper 色板与 core catalog 平行。合入前应按 O6 与 0023 决定原生壳 token 边界。

## 7. 修复路线（待批准）

分阶段、每阶段独立 PR、每阶段跑跨仓库 Gate：

1. **身份收敛**：删除 L1 前缀 ID 空间，流式阶段用内存临时键，`message.final`/`message.proactive` 原子迁移到服务端 ID；用 `control_turn_id` 替换 L3 时间窗启发式；用 `delivery_id` 替换 `proactive:` 前缀。
2. **清理白名单收敛**：合并 O2 两条 SQL 为一个 owner，统一保护列表，修复 B1。
3. **协议同步**：更新移动端 snapshot、`source.json`、`runtime-contract.lock.json` 到当前 core，跑固定组合 Gate；删除或生成 `Protocol.kt knownTypes`。
4. **turn 活动性单一 owner**：决定 Room（`turn_blocks`/消息状态）或 `TurnStopCoordinator` 之一为唯一事实，另一个只做转发。
5. **模型分层收敛**：实体层与 UI 层枚举对齐或明确为纯展示投影，删除第三套语义。
6. **主题**：原生壳色板改为从 core theme-catalog 构建期生成，或明确原生壳与 WebUI 的 token 边界后固化（MOB-UI-004）。

## 8. 跨仓库同步义务

见 [MOB-008](../projectneed.md)。要点：协议 schema/语义变更先在 core 合并 PR，移动端同一周期用配套 PR 更新 snapshot、lock 与消费代码并重跑 Gate；两边 PR 互相引用；移动端不得长期停留在旧 source commit 上修客户端 bug。

## 9. 验收标准

- 移动端 Kotlin 不再出现 `assistant:`/`user:`/`proactive:`/`ephemeral:` 前缀 ID，canonical 合并只消费服务端身份字段。
- 两条投影清理路径合并为一条，保护白名单一致，`sent` 消息在 reload 后保留。
- 移动端 snapshot/source/lock 指向的 core commit 与已合并协议语义一致，Gate 用新组合通过。
- `turn.snapshot` 与 `control_turn_id` 在移动端协议层有明确消费者或被显式移除出 knownTypes。
- 主题颜色只有一份真源；未提交 Theme 改动合入前完成 O6 决策。
