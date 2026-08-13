# 移动端投影审计：剩余重复 owner 与兼容债务

- 状态：current（固定于下述审计基准；交付状态由 `NOW.md` 维护）
- 日期：2026-08-13（勘误：初版基于过期基线 aa87e10，本版基于 a51b22a 重做）
- 审计对象：`kachofugetsu09/akashic-mobile` `main` head `a51b22a`（协议 pin `b7f62dd8`）
- 对照基准：本仓库 `main` head `8b1a7cf7`（审计时点）、[MOB-001～MOB-008](../projectneed.md)、[0004](../decisions/0004-cross-repository-evidence-is-an-immutable-combination.md)、[0034](../decisions/0034-turn-is-the-logical-work-unit.md)、[0019](../decisions/0019-mobile-long-messages-use-bounded-events.md)、[0020](../decisions/0020-mobile-history-content-uses-authenticated-http-ranges.md)、[0023](../decisions/0023-akashic-tokens-own-material-3-semantics.md)

## 1. 审计基准

服务端（本仓库）是唯一权威：SessionDB、协议 schema、turn 身份、canonical 消息身份、WebUI 源码与 theme catalog 都在本仓库。移动端持有两类状态：

1. 可重建服务端投影：会话、消息、turn block、事件、历史和附件元数据，可按固定协议重新投影。
2. 明确归属的本地工作：outbox、草稿、附件 draft/transfer、通知待办、persisted turn stop 等，不得随投影重建删除。

判断"重复 owner"的标准是：**两个组件都能改变同一事实，且写路径没有明确归属**。协议快照、进程内缓存、展示枚举等派生表示不构成第二 owner，除非存在两条真实写路径竞争同一持久状态。

## 2. 上游已完成项（移动端 PR 57 至 65，不再列为问题）

移动端 `main` 在 `aa87e10..a51b22a` 之间已经合入的修复，本审计确认后从问题清单移除：

| 上游 commit | 内容 | 对应原问题 |
|---|---|---|
| [kachofugetsu09/akashic-mobile#57](https://github.com/kachofugetsu09/akashic-mobile/pull/57) | 注册 `model.catalog.get` 命令类型 | 原"knownTypes 落后" |
| [kachofugetsu09/akashic-mobile#62](https://github.com/kachofugetsu09/akashic-mobile/pull/62) | 消息行增加 `controlTurnId`/`turnClientMessageId`，history 合并以 `control_turn_id` 精确匹配优先 | 原 L3 主体 |
| [kachofugetsu09/akashic-mobile#65](https://github.com/kachofugetsu09/akashic-mobile/pull/65) | 只有显式 wire 身份才补写或校验，旧增量缺字段不产生第二套本地事实 | 原 L3 主体 |
| [kachofugetsu09/akashic-mobile#58](https://github.com/kachofugetsu09/akashic-mobile/pull/58) | 治愈 stale streaming turns | 运行修复 |
| [kachofugetsu09/akashic-mobile#63](https://github.com/kachofugetsu09/akashic-mobile/pull/63) | history recovery 可达性修复 | 运行修复 |
| 协议 pin | `source.json` 已前进到 `b7f62dd8` | 原 B4 主体 |

时间窗启发式（`transientLocalSourceId`）在 [kachofugetsu09/akashic-mobile#62](https://github.com/kachofugetsu09/akashic-mobile/pull/62) 之后已降级为权威链（`client_message_id → delivery_id → control_turn_id`）之后、仅服务旧协议数据的兜底，符合移动端 `MOB-XREPO-003` 的旧数据兼容路径。

## 3. 剩余问题清单（修正后）

### 3.1 仍存在的重复 owner

| # | 事实 | 两个写路径 | 处置 |
|---|---|---|---|
| O1 | "同一会话只有一个活动 turn" | `TurnStopCoordinator.activeTurns`（内存断言）与 `LocalDeliveryStore.activeAssistantTurn`（Room 事务断言） | [kachofugetsu09/akashic-mobile#71](https://github.com/kachofugetsu09/akashic-mobile/pull/71) 已移除内存断言，Room 为唯一 owner |
| O2 | 投影清理保护白名单 | `deleteServerProjection`（含 `sent`）与 `deleteReloadableServerCache`（不含 `sent`），两条 DELETE 竞争同一批行 | [kachofugetsu09/akashic-mobile#68](https://github.com/kachofugetsu09/akashic-mobile/pull/68) 已合并为单一查询 |

### 3.2 缺陷

| # | 缺陷 | 触发路径 | 处置 |
|---|---|---|---|
| B1 | `sent`（已 ACK 未 canonical 化）消息在 `reloadFromServer` 时被物理删除，而 `sync.reset_required` 路径保留它 | ACK 后 outbox 命令已删，历史重投前 UI 丢消息 | [kachofugetsu09/akashic-mobile#68](https://github.com/kachofugetsu09/akashic-mobile/pull/68) |
| B2 | `FRAME_ID` regex 在 `Protocol.kt` 与 `LocalDeliveryStore.kt` 各一份 | 两份 regex 漂移导致边界校验不一致 | [kachofugetsu09/akashic-mobile#72](https://github.com/kachofugetsu09/akashic-mobile/pull/72) 统一为 `ProtocolCodec.FRAME_ID` |
| B3 | `canonicalMessageAliases` 满 256 丢最老别名，后续合并找不到 source | 长会话大量 canonical 迁移 | 保留：TODO 标注，随 L1 生命周期收紧整体删除 |
| B5 | `toolCallId` 三候选键（`tool_call_id`/`call_id`/`tool_id`）猜协议字段 | core 只发布 `call_id`，其余候选是纯防御 | [kachofugetsu09/akashic-mobile#72](https://github.com/kachofugetsu09/akashic-mobile/pull/72) 收敛为 `call_id` |

### 3.3 兼容债务（保留 + TODO 标注，等旧数据窗口滚出）

| # | 对象 | 为什么保留 | 删除条件 |
|---|---|---|---|
| L1 | `assistant:`/`user:`/`proactive:`/`ephemeral:` 本地临时 ID 前缀 | MOB-005 允许客户端用稳定投递身份引用服务端消息；前缀做命名空间隔离，且 final/history 到达后原子迁移为服务端 ID | 身份权威链全覆盖后评估主键策略，需 Room 迁移审批 |
| L2 | `canonicalMessageAliases` 进程内别名表 | 只服务 UI 持有旧临时 ID 的短窗口 | 与 L1 一并收紧 |
| L3 | 时间窗启发式兜底 | MOB-XREPO-003 明确允许旧协议数据兼容路径 | 旧数据滚出保留窗口后删除 |

### 3.4 待决策项

| # | 问题 | 需要的决策 |
|---|---|---|
| D1 | 协议 pin 陈旧风险：`b7f62dd8` 落后于 core `8b1a7cf7`（缺少 `DeltaPayload.control_turn_id` 与 `turn.output.completed` 能力门槛） | [kachofugetsu09/akashic-agent#393](https://github.com/kachofugetsu09/akashic-agent/pull/393) 与 [kachofugetsu09/akashic-mobile#73](https://github.com/kachofugetsu09/akashic-mobile/pull/73) 已按 MOB-008 固定组合围栏完成交付 |
| D2 | 原生壳 `Theme.kt` 手写色板与 core `theme-catalog.json` 的关系 | 原生壳（Compose）与 WebUI（CSS token）是两个渲染层的各自表示，不是同一事实双 owner；但色值一致性需要构建期产物或明确 token 边界 |
| D3 | 实体层字符串状态（deliveryState/kind/status）散落字面量 | 展示枚举（`AssistantTurnStatus` 等）是必要的 UI 投影，不构成重复 owner；收敛方向是状态常量集中，收益低，建议搁置 |

### 3.5 运行回归（审计时点发现）

`a51b22a` 上 `LocalDeliveryStoreTest` 有两个真实回归（[kachofugetsu09/akashic-mobile#62](https://github.com/kachofugetsu09/akashic-mobile/pull/62) 与 [kachofugetsu09/akashic-mobile#65](https://github.com/kachofugetsu09/akashic-mobile/pull/65) 引入）：

1. history 行不携带 block 权威内容时，投影合并无条件 `deleteBlocks` 清空流式迁移的 blocks；
2. legacy 流式行（无 `controlTurnId`）在 `message.final` 时身份列不补写，canonical 合并丢失 turn 身份。

[kachofugetsu09/akashic-mobile#69](https://github.com/kachofugetsu09/akashic-mobile/pull/69) 已修复。

## 4. 修复路线与交付入口

| 阶段 | 内容 | 交付入口 |
|---|---|---|
| 1 | 身份权威链（control_turn_id 消费） | [kachofugetsu09/akashic-mobile#62](https://github.com/kachofugetsu09/akashic-mobile/pull/62) 与 [kachofugetsu09/akashic-mobile#65](https://github.com/kachofugetsu09/akashic-mobile/pull/65) |
| 2 | 清理白名单合并（B1/O2） | [kachofugetsu09/akashic-mobile#68](https://github.com/kachofugetsu09/akashic-mobile/pull/68) |
| 3 | 运行回归修复 | [kachofugetsu09/akashic-mobile#69](https://github.com/kachofugetsu09/akashic-mobile/pull/69) |
| 4 | 兼容层 TODO 标注（L1/L2/L3/B3） | [kachofugetsu09/akashic-mobile#70](https://github.com/kachofugetsu09/akashic-mobile/pull/70) |
| 5 | turn 活动性单一 owner（O1） | [kachofugetsu09/akashic-mobile#71](https://github.com/kachofugetsu09/akashic-mobile/pull/71) |
| 6 | ID 卫生（B2/B5） | [kachofugetsu09/akashic-mobile#72](https://github.com/kachofugetsu09/akashic-mobile/pull/72) |
| 7 | 协议 pin 前进 + turn.snapshot 移除（D1） | [kachofugetsu09/akashic-agent#393](https://github.com/kachofugetsu09/akashic-agent/pull/393) 与 [kachofugetsu09/akashic-mobile#73](https://github.com/kachofugetsu09/akashic-mobile/pull/73) |
| 8 | L1/L2 生命周期收紧（可选） | 待评估，需 Room 迁移审批 |
| 9 | 主题 token 边界（D2） | 待决策 |

## 5. 跨仓库同步义务

见 [MOB-008](../projectneed.md) 与 [0035](../decisions/0035-mobile-protocol-delivery-is-phased.md)。要点：按变更性质选择交付阶段——兼容新增、能力门控、废弃期与 breaking removal 有不同的合并顺序与围栏，不把所有协议变化压成同一种 core-first 流程。

## 6. 验收标准

- 两条投影清理路径合并为一条，保护白名单一致，`sent` 消息在 reload 后保留。
- `turn.snapshot` 从两端协议面移除，移动端 pin 指向的 core commit 与已合并协议语义一致。
- 兼容层（L1/L2/L3）均有 TODO 标注与删除条件；没有新的双写路径。
- 移动端 `LocalDeliveryStoreTest` 全量通过（含审计时点发现的 2 个回归）。
- 主题色值只按 D2 决策后的边界维护；边界确认前不把两套渲染表示提升为同一 owner。
