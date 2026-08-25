# 0040 · Wake duty gate 属于 Wake scoped react

- 状态：accepted
- 日期：2026-08-23
- 关联条款：RUN-003、RUN-007～RUN-009、OUT-001～OUT-003、PLG-014、PRO-001～PRO-002
- supersedes：0039 中“全部 Wake 价值判断都位于 Turn port 外”的局部约束
- superseded by：[0041](0041-turn-effects-and-memory-plugins-are-orthogonal.md) 取代 memory 专用 scope 字段；Wake duty 与 delivery 决定保持有效

## 背景

0039 先用 `Timer → private gate → optional react` 验算 React Core 原子能力，要求没工作时插件不调用 Turn port。后续 Content、Drift 与 Wake 的组合设计进一步确认：到期事实检查和“这次由 Content 还是 Drift 承担 duty”的判断不是同一个问题。

到期检查只读取 durable due，可以在 Turn port 外完成。duty gate 则需要参与 Wake 本次 Prompt、工具和后续 lifecycle，并在 Content 与 Drift 之间固定串行选择。把它留在另一个私有 proactive loop 会再次复制 `react` 的阶段语义。

维护者已经确认 Wake 应被看作一种完整 `react`，duty gate 作为 before-turn 插件能力运行；不命中时由插件使用现有 lifecycle 状态记录并结束，而不是给 Core 新增 `Skip` 类型。

## 决定

Wake 保留两层职责不同的判断：

```text
Timer fired
   │
   ▼
durable due admission check
   ├─ none ─▶ record / re-arm / return（不创建 Turn）
   └─ due
        │
        ▼
   SCOPED_TURNS.start
        │
        ▼
   turn.context_prepared
        │
        ├─ Content duty proposal ─▶ shared react ─▶ share_content / skip_content
        ├─ Content decline → Drift duty proposal ─▶ shared react
        └─ both decline ─▶ domain transition + existing abort
```

外层 admission check 只回答“有没有到期事实”，不读取内容价值、不选择 Content/Drift、不构造 Prompt。内层 duty gate 是 Wake scoped Turn 的 lifecycle：固定先 Content、后 Drift；Gate 本身只读冻结 snapshot，领域 owner 负责 CAS selection 或 decline transition。

Content proposal 进入 reasoner 后，普通 assistant 正文只属于 Turn 执行诊断，不拥有用户发送语义。Wake 用本轮 scope 精确预加载两个插件 Tool：`share_content(message, items)` 与 `skip_content(reason)`；只有 durable Turn items 中恰好一个成功调用才是 Content 的权威终态决策。为保持大重构前的用户体验，Content 一次冻结最多 100 个候选，`share_content.items` 必须引用其中 1～5 个，整批只产生一条 `share_content.message`、一次通用 delivery 和一次 Session 投影。未引用候选继续 pending；`skip_content` 释放整批并保持 pending，不 ACK、不投影。缺失、冲突或越界引用 defer 且零发送。Core 只提供来源无关的 scoped Tool 可见性和 durable Turn items，不识别 Wake 名称，也不解析模型自然语言。

重构前的兴趣语义同样属于兼容合同。Core 提供只读、来源无关的 `ConversationSemanticInterest` 窄服务：它用当前 embedding runtime 对候选标题/正文编码，并只从最近 256 个完整的非 proactive 用户—助手 Turn 构造 prototype；20 条仅主动投影不能成为 prototype。Wake 把合成后的兴趣同时用于 hazard admission 与同一冻结候选页的排序，不能“因语义醒来、却仍按未增强 preprocess 顺序选择”。没有 embedding runtime 或候选正文为空时语义分为零，保留 preprocess 路径。candidate Root 只能验证拓扑，禁止读取正式 Session。

配置了 delivery target 时，Wake scoped Turn 归入目标 conversation Session，并以 `stateless + session_history_read + memory_read + memory_write=false` 执行：目标会话历史和记忆只作为本轮读入，临时 Wake 输入、内部 reasoning 与普通 `final_response` 不追加为用户消息，也不进入 Akasha；只有 provider 已送达并完成 durable projection 的主动 assistant Message 追加到目标 Session。没有 delivery target 的隔离 fixture 保持 `wake:default` 且不读取会话历史或记忆。

普通 lifecycle listener `return` 仍只结束 listener。两者都 decline 时必须使用现有 before-turn abort 合同，并先由 fixture 证明 quiet terminal、Session Message、after hook、memory 和 outbound 的真实行为。若现有 abort 不能满足 Wake 语义，停止实现并另立 Turn 合同；不得添加 Core `Skip`、插件名字分支或特殊返回字符串。

Wake listener 使用 scoped Turn 已有的 `channel="wake"` 分流。`channel` 已投影到 `BeforeTurnCtx`，而 `TurnExecutionScope.tool_source` 只负责 Tool 调用归因；本阶段不扩大 `tool_source`，也不新增 Core origin。只有同一 channel 内出现必须区分 exact execution source 的真实消费者时，才另行评估来源无关的不可变 Turn origin。

## 理由

- Wake 与 passive、Scheduler 和 Subagent 继续共用一条 `react`，before/after 仍属于 Turn lifecycle。
- due admission 与 duty selection 分别拥有不同事实，不因都叫“gate”而合并。
- Content/Drift 顺序由 Wake 私有 listener 明确调用，不依赖全局 listener 注册碰巧排序。
- quiet path 使用已有 lifecycle 语义，不把插件私有 skip 升格成 Core 控制对象。
- Content 的发送判断由 Wake 私有 typed Tool 拥有；通用 delivery 不猜 `final_response` 的含义。
- Content 批次、候选成员、引用集合和 settlement 由 Content 自己的 SQLite ledger 唯一拥有；Wake 只组合 snapshot、typed decision 和通用 delivery。
- Wake scoped Turn 的 `ToolGrant` 只允许上述两个 decision Tool；`message_push`、`tool_search` 和其他全局 Tool 均不可见、不可执行。
- Content admission 用稳定 source/item/revision identity 记录已抽签条目；future `not_before` 条目在真正到期前不得被较大的 snapshot watermark 标成已见。

## 影响

- 0039 关于 Core 原子、来源非特权、Timer 外置、无 Core `Skip` 和无来源特判的其余决定保持有效。
- 第一阶段先实现真实 fixture，不迁移正式 Wake，不修改旧 proactive/Wake/Drift 数据。
- characterization 已确认 quiet abort 不写 Session messages、不发送 outbound，也不运行 after hooks；Control runtime 仍保留 completed Turn、输入 item 和空 assistant item。实现 fixture 必须同时锁定这两类事实。
- `tool_source` 不会因本决策自动改名或扩大职责。
- `TurnExecutionScope.preloaded_tools` 只把已经注册且已被 `ToolGrant` 授权的 Tool 加入本轮可见集合；它不改变全局 Tool 定义、其他 Turn 或插件生命周期。

## 验收

- [ ] 无 due 时不创建 scoped Turn。
- [ ] 有 due 时只创建一个 Wake scoped Turn，并冻结 exact Root。
- [ ] Content 命中时 Drift 不执行；Content decline 后才执行 Drift。
- [ ] 两者 decline 时 reasoner、Tool 和 delivery 为零，领域 transition 已提交。
- [ ] quiet terminal 不产生空 outbound，不把临时 Wake input 错写成用户 Message。
- [ ] passive、Scheduler 和 Subagent Turn 不运行 Wake duty 逻辑。
- [ ] Core 没有 Content、Wake、Drift 名称分支或通用 `Skip`。
- [ ] Content completed Turn 必须恰有一个成功的 `share_content` 或 `skip_content`；普通 `final_response` 永不直接投递。
- [ ] `skip_content`、缺失决策和冲突决策均产生零 channel delivery、零用户 Session projection；重启恢复仍读取相同 durable Turn items。
- [ ] 20 个候选只创建一个 Wake Turn；share 最多消费 5 个并只投影一条消息，skip 后 20 个仍 pending 且零 ACK。
- [ ] 配置目标会话时，Wake 读取该 Session 历史和记忆但不写临时 input/reasoning；20 条未回复 proactive 后的 `u → a` 仍只形成一个 Akasha interaction。

## 关联设计

- [Content / Wake 现有原子能力盘点与第一阶段设计](../design/content-wake-existing-atoms-first-stage.md)
- [React Core、Scheduler 与 Subagent](../design/react-core-scheduler-subagent.md)
