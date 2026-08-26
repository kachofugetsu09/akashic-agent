# Akashic Channel 与 Web/Mobile Adapter 规格

- 状态：confirmed design；尚未授权实现
- 日期：2026-08-26
- 决策：[0044](../decisions/0044-akashic-channel-uses-web-and-mobile-adapters.md)
- 关联条款：AKC-001～AKC-003、MOB-001～MOB-008、SES-001～SES-008、MIG-001～MIG-002、WSP-003
- 取代范围：[未来路线草案](akashic-future-roadmap-issue-drafts.md) 第 5 节

## 1. 目标与边界

Web 与 Mobile 现在是两个 Core Channel，各自生成 `web:*`、`mobile:*` Session。本变更只做
三件事：

1. Core 只注册一个内建 `akashic` Channel。
2. Web 与 Mobile 成为它的两个边界 adapter，读取同一个既有 Session 空间。
3. 旧 Web/Mobile Session 及其真实引用一次性迁移为 `akashic:*`。

```text
Web transport ───┐
                 ├─ AkashicChannel ── existing Session / Message / Turn
Mobile transport ─┘
```

本规格不重做 Session、Message、Turn、附件、模型、流式协议、Turn 控制、通知、客户端本地
状态机或外部 Channel 投递。Web 与 Mobile 的认证、wire、缓存、Room、outbox、附件传输和
平台效果继续由各自现有实现拥有。

## 2. 为什么只新增一个组合 owner

当前 `bootstrap.app.App` 把 `WebChatChannel` 与 `MobileRealtimeChannel` 分别加入
`ChannelHost`，再分别发布 `CoreChannelDefinition`。Committed Channel catalog 禁止两个
Core definition 使用同名 Channel，所以不能只把两者的 `name` 都改成 `akashic`。

目标是一个很薄的组合 owner：

```text
AkashicChannel
├─ WebAdapter      现有浏览器认证、HTTP/WebSocket、上传与呈现
└─ MobileAdapter   现有配对、durable WebSocket、Room/ACK、附件与通知
```

`AkashicChannel` 只拥有一次 Channel 注册、共同的 `channel/chat_id` 路由和两个 adapter 的
启动/停止。它复用现有 `Channel`、`ChannelContext`、Core `ChannelAdapter`、SessionStore 和
EventBus，不增加 Port、客户端框架、共同 wire schema 或共同 reducer。

## 3. 身份与可观察行为

唯一身份是：

```text
channel     = "akashic"
chat_id     = "<bare canonical id>"
session_key = "akashic:<chat_id>"
```

- `channel` 与 `chat_id` 继续使用通用 Channel 合同；`chat_id` 不包含 `akashic:` 前缀。
- 新对话复用 Web 已有 `session.create` 的“只分配 ID、不持久化”语义；Mobile breaking 协议
  只增加这个既有操作。持久 Session 仍由首次消息提交路径创建，不增加空 Session 生命周期。
- Web 与 Mobile 都能列出、打开和继续同一批 `akashic:*` Session。
- 当前选择、草稿、已读位置和 UI 设置仍是各端本地事实，不同步导航。
- 历史分页、实时流、停止、并发、模型和附件继续服从现有 owner；本变更只让两个 adapter
  使用同一 `session_key`，不重新规定这些能力。
- endpoint/device identity 只留在各 adapter 的认证、去重和诊断内部，不进入 Session 身份。

Web 与 Mobile Akashic 入站都固定使用 `provider_identity = bare chat_id`、
`recipient = bare chat_id`。Mobile 的 device identity 不能继续充当 Channel identity，因为
同一设备可以进入多个 Session；旧 device→last-session identity rows 在迁移中退役。

## 4. 必须消除的一处旧耦合

Mobile durable handoff 目前在 Bus、PassiveWorker、Channel generation host 和部分 lifecycle
代码中通过 `channel == "mobile"` 选择耐久路径。统一为 `akashic` 后，这个判断会把 Mobile
输入误走成普通 Web 输入。

耐久性已经由 handoff ID、`mobile_v3_handoff` metadata 和既有 ownership 表达。本次只把
这些 channel-name 特判改为读取已有 handoff marker/owner；不增加新的耐久协议，也不让
Web 获得 Mobile handoff。

## 5. Schedule、Proactive 与 Akasha

- Schedule 继续使用既有 `{channel, chat_id}` target；迁移只把命中的 Web/Mobile target
  改为 `{"channel": "akashic", "chat_id": "<new id>"}`，不增加 `target_session_id`。
- Wake/Proactive 和 durable delivery 只迁移真实保存的 `channel`、recipient 与 Session
  引用。投递顺序、delivery receipt、Turn effect 和通知语义保持现状。
- 一个面向 `akashic` 的既有逻辑投递由 `AkashicChannel` 交给两个 adapter 投影，不复制
  Session Message，也不为此改写通用外部 Channel delivery。
- Akasha 算法不变。因为 sidecar 保存 `session_key`，Session rekey 后调用现有
  `rebuild_akasha_sidecars()` 从 SessionDB 固定输入备份并重建。
- 退役 `memory2.db` 归档不导入、不改写、不删除；0041 的 Turn effect 合同不在本规格重述。

## 6. 一次性 breaking 迁移

### 6.1 前置条件

迁移在维护窗口停止全部 workspace writer，取得独占锁并创建可验证完整备份。
`active_turn_count`、session admission、inbound handoff、compaction prepare 和 Mobile 未决
receipt/import 必须先收束到迁移工具明确支持的状态；不能证明安全时 fail-loud，什么都不改。

旧 APK 没有兼容窗口、alias、双读或双写。新 Mobile 版本保留 pairing/Keystore 和非 Session
设置，旧 Session-bound 投影怎样减少由移动仓库按真实 schema 单独设计；Core 规格不枚举
Android 本地表。

### 6.2 映射合同

每个完整旧 key 生成一个不同的新 bare ID：

```text
web:<old id>    ──▶ akashic:<new id A>
mobile:<old id> ──▶ akashic:<new id B>
```

映射必须确定、一对一且可在迁移 plan 中审阅。不得按尾部 ID、正文或相似历史合并。精确
算法属于 Yoyo 实现，不形成新的长期 mapping registry。

### 6.3 已证明需要处理的引用

| owner | 迁移动作 |
|---|---|
| `sessions.db` | rekey `sessions.key`、`messages.session_key`、`turns.session_key`、`session_compactions.session_key`、`session_source_mutation_audits.session_key` |
| 活动 fence | `session_admissions`、`inbound_handoffs`、`session_compaction_prepares` 在 preflight 收束，不猜测改写活动 owner |
| Channel identity index | 退役旧 Web/Mobile rows；Mobile device→last session row 不迁成 Akashic identity |
| Mobile Gateway DB | 对真实 `session_id` 列和 inbox envelope 中的 Session 引用做 plan；未决 receipt/import 先收束或阻断 |
| Scheduler/Wake/delivery | 只 rekey 实际命中的 target、recipient、accepted/projection Session 引用 |
| Akasha sidecar | 不逐行改写，调用现有固定输入 rebuild |

`proactive.db`、`wake_proactive.db`、`drift.db` 是否含命中引用仍是实现前需要用真实数据/schema
证明的未知，不能因为名称相关就写进修改清单。

### 6.4 受保护事实

迁移改变路由用的 Session key，不改变消息正文、role、seq、Turn/Interaction、附件 artifact、
模型选择或任务内容。

当前 `messages.id` 由创建时的 `session_key + seq` 组成。迁移保留既有 message ID，即使其中
保留旧字面前缀；只改 `messages.session_key`。因此 message embedding、附件绑定、FTS 和历史
审计不需要级联换 ID。“旧前缀归零”只检查活动路由字段，不能扫描历史 ID、审计 JSON 或
compaction provenance 后强行改写。

## 7. 验收

1. Core catalog 只有一个内建 `akashic` 对话 Channel；Web/Mobile 不再单独注册 Channel。
2. 两端创建的 Session 都使用 `akashic:<id>`，并能被另一端列出、打开和继续；本地导航互不
   强制跳转。
3. Web/Mobile 现有历史、实时、停止、附件、模型与 Mobile durable recovery 回归通过，证明
   统一身份没有重做其语义。
4. Schedule/Wake/Proactive 的真实目标引用迁移后仍指向同一会话；一次既有逻辑投递不会因
   两个 adapter 产生两条 Session Message。
5. 迁移保持 old→new 一对一、已知活动路由无旧前缀、无悬空引用；message ID 与其下游引用
   保持不变，Akasha 固定输入 rebuild 通过。失败不留下半新半旧 workspace。
6. 旧 APK 在协议边界 fail-loud；新 APK 在保留配对材料后完成投影迁移、全量同步和真实收发。

规格批准不等于实现授权。若实现需要新增 Port、共同客户端协议、Session 生命周期、通用
delivery 语义或未列出的持久 owner，必须停止并重新批准设计。
