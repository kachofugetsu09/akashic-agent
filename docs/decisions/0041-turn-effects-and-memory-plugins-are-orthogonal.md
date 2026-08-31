# 0041 · Turn 副作用与 Memory 插件保持正交

- 状态：accepted
- 日期：2026-08-25
- 关联条款：SES-001、SES-007～SES-008、MEM-002、MEM-009～MEM-011、PLG-001～PLG-014、RUN-003、RUN-007～RUN-009
- supersedes：0006 的宿主 memory engine 选择与 `skip_post_memory` 写入合同；0040 的 memory 专用 scoped Turn 字段
- superseded by：0052（移除 Core Markdown 特权通道并退役 PENDING/optimizer）

## 背景

Core 曾同时拥有 embedded memory engine、检索块、turn runtime、Dashboard 管理 API 和
`skip_post_memory`。Scheduler、Subagent、Wake、programmatic call 与 replay 为表达
“这轮不要进入记忆”又分别写入不同字段。这样一来，增加或替换一个 Memory 插件会迫使
Core、Session、Prompt 和主动插件一起变化。

本轮只迁移 embedded memory provider（Akasha/Default）的三件事：在 Prompt 生命周期注入
内容，在 Turn 提交后更新自己的投影，以及提供 Tool。Core 现有 Markdown/PENDING/SELF
链路不属于 Akasha，本轮保持原 owner；源码标记后续独立插件化 TODO。

## 决定

Core 只提供两个来源无关、彼此独立的 Turn 维度：

```text
TurnExecutionScope
├─ storage
│  ├─ durable   ─▶ U…U+A 写入 Session
│  └─ in_memory ─▶ 不写 Session
└─ post_commit_effect
   ├─ allow     ─▶ durable projector 可以消费
   └─ suppress  ─▶ durable projector 不消费
```

`in_memory` 必须与 `suppress` 一起使用；这是 Core 拥有的结构不变量。`durable + suppress`
表示“这轮是客观 Session 事实，但不应产生长期投影”。显式 `remember`、`forget` 等 Tool
是否可用由同一 Turn 的 `ToolGrant` 决定，不由 post-commit effect 猜测。

Memory provider 使用已有插件积木组合能力：

```text
普通 lifecycle listener ─▶ PromptSectionRender("memory")
TurnCommitted listener    ─▶ 插件自己的计算内核与存储
TOOL_CATALOG              ─▶ recall / remember / forget
UI_SLOTS                  ─▶ 插件自己的观察面
provide(plugin.claim.embedding_memory)
```

`plugin.claim.embedding_memory` 只是向量化记忆 provider 的互斥角色声明，不提供可调用服务，
也不由 Core 选择实现。任意两个插件同时 provide 时，组合根因重复服务 fail-loud；operator
必须禁用或卸载其中一个。未来 Markdown 插件不使用这个 claim，可以与
embedded memory provider 正交共存。Akasha 需要 embedding 配置时，消费来源无关的
`core.text_embedding.settings`。

Prompt 不再认识 `retrieved_memory_block`。Akasha 与 Meme 一样，在普通 Prompt lifecycle
产生一个 section。Core 不再提供 embedded memory engine、memory turn runtime、特殊
retrieval pipeline、交互删除 reconciliation 或 embedded memory Dashboard 路由；Core
Markdown runtime、optimizer 和对应 Dashboard 操作暂时保留。

Yoyo `20260826_01_migrate_turn_effects` 在 runtime 启动前把历史 session、message 和 durable
Turn 排除语义一次性投影为 `effects.post_commit=suppress`，并删除旧 boolean 字段。Scheduler
历史 session 同样逐消息、逐 Turn 落到这个原语；回放和在线路径不再保留旧字段或 session
前缀解码器。配置 Yoyo `20260825_02_select_akasha_embedding_plugin` 只翻译旧的显式 Akasha
开关：开启者选择 Akasha，关闭者同时禁用 Akasha 和依赖其 semantic-interest 服务的 Wake；
写前保留可恢复配置备份。旧 Default 私有数据库不导入、不删除，作为可恢复归档保留。

## 理由

- storage 只回答“Session 是否记录客观事实”，effect 只回答“提交后投影能否消费”，两个轴不互相猜测。
- Embedded Memory、Meme 和未来上下文插件共用 lifecycle，不再按领域名称获得 Core 特权。
- provide 的重复声明天然完成竞争检测，不新增 memory registry 或 selector。
- 一次性 Yoyo 迁移历史事实；在线、replay、Akasha 与 Core Markdown 随后只消费一个新语义。
- Wake 对 semantic interest 的真实依赖显式 inject；缺少 provider 时拓扑保持 pending 或启动失败，不静默降级。

## 影响

- Akasha 是唯一内置的 embedding memory 插件；经典记忆插件和私有 reconciliation 通道删除。
- Scheduler、Subagent 和 Wake 筛选 Turn 使用 `in_memory + suppress`。
- Wake 已送达投影、后台 programmatic Turn 和 continuation 使用 `durable + suppress`。
- Session 删除只删除 Session 事实；某个 Memory 插件若需要撤销自己的投影，应通过自己的领域 Tool 或生命周期协议拥有该能力。

## 验收

- [x] Akasha 单独启用可启动、注入普通 Prompt section、消费 TurnCommitted 并提供 Tool/UI。
- [x] 两个普通插件同时 provide `plugin.claim.embedding_memory` 时拒绝启动。
- [x] 旧记忆开启者迁移到 Akasha；旧记忆关闭者不启动 Akasha/Wake，也不触发 replay。
- [x] durable allow、durable suppress、in-memory suppress 三种 Turn 都有持久化和事件证据。
- [x] interrupted `U + U + A` 只在闭合后形成一个可消费 Turn。
- [x] Yoyo 对线上 SessionDB 副本迁移后旧字段归零，replay 与新 effects 得出相同投影集合。
- [x] Core 源码不存在 embedded memory engine、retrieved block 或 interaction reconciliation 通道；Markdown 特权通道保留并带未来插件化 TODO。
- [x] Wake 的 semantic interest 依赖缺失时 fail-loud，其他插件依赖扫描有明确结果。

## 关联设计

- [React Core、Scheduler 与 Subagent](../design/react-core-scheduler-subagent.md)
- [Content / Wake 现有原子能力盘点与第一阶段设计](../design/content-wake-existing-atoms-first-stage.md)
