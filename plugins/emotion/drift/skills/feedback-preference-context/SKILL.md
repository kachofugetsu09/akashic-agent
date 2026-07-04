---
name: emotion:feedback-preference-context
description: 从 proactive 正反馈中归纳待审核推送偏好候选，追加到 proactive_pending.md 队列。
---

# Feedback Preference Pending

## 目标

读取尚未处理过的 `topic_follow` 和 `explicit_quote` 反馈，最多一批 50 条，把会影响主动推送决策的候选规则追加到 workspace 根目录的 `proactive_pending.md`。

这个 skill 只写 pending 队列。它的职责类似普通记忆里的 `PENDING.md`：先把候选沉淀成可审核队列，后续再由独立流程合并到正式规则。

## 流程

```text
feedback events
├─ evidence bundle
│  ├─ proactive text
│  ├─ user text
│  ├─ feedback id
│  └─ message id
├─ infer topic attitude by prompt
├─ append proactive_pending.md
└─ advance latest cursor only after write
```

## 固定脚本

```bash
python3 skills/emotion:feedback-preference-context/scripts/sample_feedback_context.py sample --drift-dir . --chunk-index 0 --chunk-size 10
```

脚本只负责取数：

```text
sample
├─ 读取 drift.db 中 cursor.latest_processed_feedback_id
├─ 查询 ../proactive_feedback/proactive_feedback.db
│  └─ feedback_type IN ('topic_follow', 'explicit_quote')
├─ WHERE id > latest_processed_feedback_id
├─ ORDER BY id DESC LIMIT 50
├─ 用 ../sessions.db 回填 proactive/user 原文
├─ proactive 文本截断到 100 字，user 文本不截断
├─ 每次只返回 chunk-size 条 events
└─ 返回 cursor_tail_feedback_id
```

脚本不判断 topic、不判断 effect、不生成 pending 内容。

## 执行规则

1. 先调用 `select_skill`。
2. 调固定脚本读取第 0 个 chunk：

```json
{
  "command": "python3 skills/emotion:feedback-preference-context/scripts/sample_feedback_context.py sample --drift-dir . --chunk-index 0 --chunk-size 10",
  "cwd": ".",
  "description": "读取 proactive feedback 第 0 个 chunk",
  "timeout": 30
}
```

3. 如果 `found=false`，用 `finish_drift(status="waiting", message_result="silent")` 静默结束，不推进 cursor。
4. 有样本时先读取 `../proactive_pending.md`，确认现有文件内容和末尾位置；文件不存在就用空内容处理。
   - 必须使用路径 `../proactive_pending.md`。
   - 禁止使用 `proactive_pending.md` 或 `./proactive_pending.md`，那会写到 drift 目录而不是 workspace 根目录。
5. 读取 `../memory/MEMORY.md` 作为长期偏好参考；文件不存在就用空内容处理。
6. 不要调用 `fetch_messages`。
7. 本轮只允许使用三类输入：固定脚本返回的 chunk、`../proactive_pending.md`、`../memory/MEMORY.md`。其他 workspace 文档不是证据来源，不要读取。
8. `shell` 只用于运行固定脚本；不要用它执行 `cat`、`cp`、`ls`、`sed`、`grep` 或读取/复制文件。
9. 只处理当前 chunk 的 `events`，自己追加到 `../proactive_pending.md` 尾部。
10. 每个 chunk 写一次：写完当前 chunk 的 pending section 后，才能读取下一个 chunk。
11. 如果当前 chunk 没有合格候选，也写一个 `no_candidate` section，说明这个 chunk 已审核。
12. 如果 `has_more=true`，按 `next_chunk_index` 调同一个脚本读取下一段，再重复“判断 -> 追加 pending section”。
13. 如果当前 chunk 返回 `has_more=true`，禁止调用 `finish_drift`。
14. 只有处理并写入所有 chunk，直到某个 chunk 返回 `has_more=false` 后，才能调用 `finish_drift`。
15. 不要等 50 条全部看完再写，也不要为每条 event 单独写文件。
16. 只追加新候选，不修改、不删除已有队列项；写文件时必须保留原文完整前缀，只在末尾增加新 batch/chunk 内容。
17. 只看 proactive 消息和 user 回复这一对文本。assistant 后续回答不是证据来源。
18. MEMORY 只用于理解长期兴趣边界和查重；新增候选必须由当前 chunk 的 feedback 证据支撑。
19. `signal_hints` 只是弱提示；最终 topic 粒度、用户态度、effect 都必须由你结合当前 chunk 和 MEMORY 推断。
20. 不要求用户显式说喜欢或讨厌。追问、纠错、补充背景、切换关注对象、持续互动都可以作为态度证据，但必须解释它对推送决策有什么影响。
21. topic 粒度由证据决定：
   - 同一大类中用户态度不同，必须拆成不同 topic。
   - 不同大类中用户态度相同，也不要为了合并而合并。
   - 如果 topic 只能写成一段原文摘要，说明粒度还没想清楚，宁可不写。
   - 如果只有单条弱信号且无法推断稳定推送决策，不写。
22. 粒度判定遵守这些规则：
   - `topic` 不能用斜杠拼接两个不同对象；如果想写 `A/B`，通常说明应该拆成两条，或改成一个单一父 topic。
   - `topic` 和 `action` 必须同宽。topic 写某个对象时，action 不得扩到同类所有对象；action 想覆盖同类所有对象时，topic 也必须写成该同类，并有多条证据支持。
   - 单条证据可以进入 pending，但 `effect` 优先用 `verify` 或 `tone`；只有用户态度强、或 MEMORY 已支持该方向，才用 `boost/block/timing`。
   - 追问不等于喜欢。追问只能说明“这个 topic 的某个侧面值得继续观察”，除非同 chunk 或 MEMORY 中还有更强证据。
   - `block` 只能用于明确低兴趣、反感、纠错、无实际价值；弱负反馈优先写 `verify`，action 写“降低/收窄/先验证”。
   - `boost` 的 action 用“提高优先级/更容易入选/可继续观察”，不要写“持续推送/优先推送所有相关内容/彻底保留”。
   - `timing` 只处理时机、频率、静默、ack，不承载长期兴趣判断。
   - 不要在 action 中新增 evidence 没出现的对象；比如 topic 只证明某个队伍，就不要把 action 写到其它队伍。
23. 每条 pending 必须包含 `granularity`，说明为什么这个 topic 这样切，并明确“不扩大到哪里”或“不缩小到哪里”。
24. 每条 pending 必须包含 `inference`，说明你从哪些用户回应推断出该态度。
25. 只写会影响具体推送决策的候选。每条必须能回答：
    - 什么候选内容会触发？
    - 对推送决策有什么动作？
    - 证据是哪条 feedback 和哪条 user message？
26. `effect` 只能是：
    - `block`：阻止某类候选推送。
    - `boost`：提高某类候选优先级。
    - `verify`：推送前必须额外核验或限定匹配方式。
    - `timing`：改变推送时机、静默、ack 或打扰条件。
    - `tone`：改变同一候选内容的表达方式。
27. 不写普通生活事实、人设事实、一次性寒暄、测试消息。
28. evidence 必须包含完整 `feedback#id` 和完整 user message id。
29. 全部 chunk 都写入成功后，`cursor_update.latest_processed_feedback_id` 必须等于第 0 个 chunk 返回的 `cursor_tail_feedback_id`。
30. `finish_drift.briefing` 必须使用实际处理结果，写清总样本数、chunk 数、pending 候选条数，不要估算。
31. 如果只处理了部分 chunk，必须 `status="paused"`，不要推进 `latest_processed_feedback_id`。

## proactive_pending.md 格式

如果文件为空，先写标题：

```text
# Proactive Pending

```

每批追加一个 section：

```text
## Batch feedback#1-feedback#50

### Chunk feedback#50-feedback#41

- [ ] effect=boost confidence=medium topic="..." granularity="..." inference="..." action="..." evidence=feedback#12 user_message_id=...
- [ ] effect=verify confidence=low topic="..." granularity="..." inference="..." action="..." evidence=feedback#18 user_message_id=...
```

要求：

- 每行只表达一个 topic 级候选规则。
- 每个 chunk 使用一个 `### Chunk feedback#高-feedback#低` 小标题。
- `topic` 写可用于未来推送匹配的短名称，不写整段原文。
- `granularity` 写为什么不放大或缩小 topic。
- `inference` 写用户对该 topic 的态度如何被推断出来。
- `action` 写 proactive 决策动作，不写聊天总结。
- `confidence` 必须写：
  - `low`：单条弱信号、追问、短回应、或推断需要后续验证。
  - `medium`：同一 topic 有多条一致证据，或单条 evidence 很明确。
  - `high`：同一 topic 有多条一致证据，并且 MEMORY 也支持。
- 如果本批没有合格候选，仍追加 section，并写：

```text
- [ ] no_candidate evidence=feedback#1-feedback#50 reason="本批没有足够明确且会影响推送决策的候选。"
```

## 收尾

成功追加队列：

```json
{
  "skill_used": "emotion:feedback-preference-context",
  "status": "completed",
  "briefing": "根据 proactive 正反馈样本追加 proactive_pending.md 队列",
  "message_result": "silent",
  "cursor_update": {
    "latest_processed_feedback_id": 123,
    "active_cursor_tail_feedback_id": null,
    "active_feedback_ids": null
  },
  "journal_append": [
    {
      "entry_type": "proactive_pending_appended",
      "key": "1-123",
      "payload": {
        "feedback_ids": [1, 2, 3],
        "cursor_tail_feedback_id": 123
      }
    }
  ]
}
```

## 约束

- 一次最多处理 50 条反馈。
- `explicit_quote` 必须包含，且视为更强证据，但不是自动规则。
- 只有 `proactive_pending.md` 成功尾部追加后才能推进 `last_feedback_id`。
- 写入前必须检查旧内容；禁止用新生成内容覆盖整个 pending 文件。
- 不打扰用户，不调用 `message_push`。
- 不读取或写入 `state.json`、`history.json`。
- 不修改 proactive_feedback 数据库。
