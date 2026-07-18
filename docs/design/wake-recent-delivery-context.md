# Wake 最近主动消息上下文

- 状态：implemented in candidate，pending merge
- 确认日期：2026-07-18
- 关联条款：PRM-008、STA-002、CTX-004、SES-005、PRO-001、PRO-003、TST-001

## 1. 问题和用户意图

Wake 当前只向模型提供目标 session 的最近普通对话，并明确排除历史主动消息。真实数据中，Telegram 已经发送的 GPT-5.6 文件删除消息，之后又由 Mobile session 以另一个 Feed event 再次发送；Kimi K3 的多个不同测评也在短时间内重复占用主动消息。

用户希望模型继续结合长期偏好自主判断。额度重置等反复发生但具有实际价值的消息仍可发送，Kimi 测评等边际信息较低的内容则由模型结合最近发送记录判断。本次不增加 URL、主题、冷却时间或重要性硬规则。

## 2. 当前调用链和状态 owner

```text
sessions.db/messages
        │ WakeRuntime 只读
        ▼
_read_recent_passive_conversation(session_key, now)
        │ 排除 extra.proactive=true
        ▼
build_messages(recent_session=...)
        │
        ▼
Wake 标题初筛与最终决策
```

`sessions.db/messages` 是已经持久化的对话事实，由 Session store 拥有；Wake 只获得窄的只读查询。`PromptContext` 是临时运行时视图，不拥有消息写入、更新或删除权限。

## 3. 目标结构

```text
当前目标 session 的被动对话 ──►【截至当前时间的最近被动对话】

workspace 内跨 session 的主动消息 ──►【截至当前时间已经发送的主动消息】
                                          │ 每条带发送时间和 session
                                          ▼
                        MEMORY + 两类历史 + 当前候选
                                          │
                                          ▼
                               Wake 自主 share / skip
```

主动消息区块陈述已经发送的事实，并明确它不是用户陈述或本轮候选。区块让模型理解自己最近主动和用户聊过什么；曾经聊过什么只是连续性的事实背景，不是内容价值的扣分表。模型每次重新理解当前事件此刻对用户意味着什么，话题、结论或事件相近不自动禁止再次分享，也不预先标记重复、不重要或应跳过。模型始终对用户本人和他在意的一切保持真诚好奇，这种好奇不会因话题已经聊过、结论相同或事件反复发生而耗尽，但也不因此强行提问或打扰。模型继续使用现有 `scratchpad`、调查结果和终态工具自主判断。

初筛后的入选候选如果共同依赖一种内容形态或打扰类型的偏好，而固定记忆和最近上下文没有直接证据，可以生成一个批次级 `preference_probe`。主题兴趣和内容形态偏好是两个独立参考维度，任何一方都不预设 share 或 skip 倾向。正文事实足以解决歧义或上下文已有直接证据时不查询。探针不复述新闻标题；运行时随后执行一次 `intent=interest`、`effect=read_only`、`relevance_floor=strong` 的查询。Akasha 的 strong 只返回超过 `dense_seed_threshold` 的 Dense 证据，不把 Ripple、Graph 或 BlackHole 当作态度依据；其他 memory engine 使用各自拥有的原生强相关阈值。

## 4. 查询和边界

- 被动对话保持当前目标 `session_key`、截至 `now`、最近 20 行和 3,000 字符预算；历史主动消息不混入该区块。
- 主动消息查询整个 workspace 中 `role=assistant` 且 `extra.proactive=true` 的消息，严格限制 `ts <= now`，按时间倒序取最近 30 条，再恢复为时间正序。
- 主动消息只保留最近 7 天，并受独立字符预算约束；每条包含 ISO 时间、`session_key` 和真实消息正文预览。
- 数据库不存在时返回空；SQL、JSON 或数据库损坏继续 fail-loud，不伪装成没有历史。
- 同一轮标题初筛和最终判断复用同一份主动消息快照，避免两个阶段观察到不同历史。
- 偏好查询发生在标题初筛之后、最终判断之前；每轮最多一次，返回数量由 engine 的 strong 阈值决定，`limit=12` 只作为异常上限。

## 5. 状态变化和副作用

本次不改变对话、记忆和主动投递的权威持久化协议：

- `sessions.db/messages`：正常发送仍只 INSERT；Wake 新查询不执行 INSERT、UPDATE 或 DELETE。
- `wake_proactive.db`：run、observation、reservoir、hazard 和消费状态维持现有协议。当前 wake 的 `scratchpad_json` 与 `investigations_json` 诊断快照增加批次级探针和偏好证据；同一 run 仍按现有 UPSERT 更新，旧 run 不迁移、不覆盖、不删除。
- `MEMORY.md` 与 `PROACTIVE_CONTEXT.md`：只读且不修改。
- 外部发送：生产提交路径不变；Docker 验证使用 `replay_debug` CaptureChannel，只写隔离 outbox。

正常运行会继续追加新 message、wake run 和 observation；本次没有新增逻辑失效或物理减少路径。只有原有的用户删除、ack、消费和进程生命周期 owner 能按状态地图改变相应状态。恢复证据是冻结的 SQLite 备份、基线提交和 CaptureChannel outbox；正式 workspace 不参与回放写入。

回滚点是 `backup/wake-recent-delivery-context-before-20260718` 和基线提交 `6db411ac`。

## 6. 验收

1. 单元测试证明两个 section 分开渲染，主动消息带时间和 session，且不会进入被动对话区块。
2. 单元测试证明主动消息跨 session 可见、未来消息不可见、七天窗口和数量上限生效。
3. 单元测试证明偏好探针每轮最多执行一次只读查询，Akasha 只返回超过原生阈值的 Dense 证据，其他引擎使用自身原生强相关阈值。
4. 同一冻结 `sessions.db`、`MEMORY.md`、Akasha 和 Wake reservoir 在 Docker 中回放；base 与 candidate 的已声明差异是新增主动消息区块与单次偏好探针。
5. GPT-5.6 重复消息、Kimi K3 连续测评和 Tibo 额度重置分别记录模型的 `scratchpad`、偏好证据、`share/skip`、最终消息与 capture outbox，不把某个主题硬编码成固定结果。
6. 正式 workspace、正式 Mobile gateway 和正式 server 在验证前后保持不变。
