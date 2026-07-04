---
name: default_memory:audit-dirty-memories
description: 随机抽检 default memory 的长期记忆，回溯 source_ref 原始上下文，识别疑似脏记忆并向用户汇报证据。
---

# Audit Dirty Memories

## 目标

随机抽检一条尚未审计、且带 `source_ref` 的 default memory 长期记忆。

如果原始上下文与记忆摘要一致，静默记录“已审计”并结束本轮。
如果原始上下文不能支持该摘要，或摘要明显把提问、计划、否定、条件写错成事实，
向用户说明这条记忆为什么可疑，并给出证据，等待用户在被动回复链路里纠正。

这个 skill 只做调查和汇报，不做 `forget_memory` 或 `memorize`。

## 状态来源

```text
default_memory:audit-dirty-memories
├─ scratchpad
│  └─ 自然语言前情：上一轮做到哪一步、下次怎么接
├─ cursor
│  └─ 结构化游标：active_memory_id / source_ref / summary / stage
└─ journal
   └─ append-only 记录：entry_type=memory_audited 的已审计 memory_id
```

- `local_context.cursor` 是未完成候选的结构化来源。
- `local_context.journal_recent` 只用于快速了解最近审计结果。
- 已审计去重由固定脚本读取 Drift `skill_journal` 完成。
- 不读取、不创建 `history.json`、`state.json`、`audited.md`。

## 固定脚本

```bash
python3 skills/default_memory:audit-dirty-memories/scripts/sample_memory_for_audit.py sample --drift-dir .
```

脚本只做抽样：读取 `skill_journal` 里 `entry_type=memory_audited` 的 key，
查询 `../memory/memory2.db`，排除已审计项后返回 1 条候选。

## 工作流程

```text
run
├─ 读 local_context.cursor
│  ├─ 有 active_memory_id -> 继续这条候选
│  └─ 没有 -> shell sample
├─ fetch_messages(source_ref, context=10)
├─ 判断 clean / suspicious_reported / unverifiable_old_source
├─ 必要时 message_push
└─ finish_drift
   ├─ scratchpad_update: 自然语言接续
   ├─ cursor_update: 清空或保留 active 候选
   └─ journal_append: memory_audited
```

1. 先看 `select_skill` 返回的 `local_context.cursor`。
   - 如果有 `active_memory_id`，继续审计这条候选。
   - 如果没有，运行固定抽样命令。
2. 抽样命令：

```json
{
  "command": "python3 skills/default_memory:audit-dirty-memories/scripts/sample_memory_for_audit.py sample --drift-dir .",
  "cwd": ".",
  "description": "抽样一条待审计记忆",
  "timeout": 60,
  "auto_promote": false
}
```

3. 如果脚本返回 `{"found": false}`：
   - `finish_drift(status="waiting", message_result="silent")`
   - `briefing` 写“没有可审计候选”
   - `scratchpad_update` 写“等待出现新的、带 source_ref 且尚未审计的 default memory 记忆”
   - `cursor_update` 写 `{"next_action": "sample"}`
4. 如果脚本返回候选，记录 `memory_id`、`memory_type`、`summary`、`source_ref`、`happened_at`。
5. 用 `fetch_messages` 读取原始上下文：
   - `source_ref` 直接传候选的完整值。
   - `context` 固定为 `10`。
   - 如果 `source_ref` 是 JSON 列表字符串，整段原样传给 `fetch_messages`，不要拆开。
   - 如果候选意外仍是 `@post_response` 旧 source，视为 `unverifiable_old_source`。
6. 只做高置信判断：
   - 原文只是提问或讨论，摘要却写成已知事实。
   - 原文只是计划或意图，摘要却写成已经发生。
   - 原文明显是否定，摘要却写成肯定。
   - 原文有明显条件或时间限制，摘要却丢失了限制。
   - 摘要的主体、时间、结果与原文明显冲突。
7. 不要把下面情况报成脏记忆：
   - 摘要只是更短。
   - 省略了不重要细节。
   - 表达方式不同但核心事实一致。
   - 只是感觉奇怪但拿不准。

## 收尾要求

clean 或 unverifiable_old_source：

```json
{
  "skill_used": "default_memory:audit-dirty-memories",
  "status": "completed",
  "briefing": "审计 memory_id=...，结果 clean",
  "message_result": "silent",
  "scratchpad_update": "刚审计 memory_id=...，结果 clean；下次继续随机抽样。",
  "cursor_update": {
    "active_memory_id": null,
    "source_ref": null,
    "summary": null,
    "stage": null,
    "next_action": "sample"
  },
  "journal_append": [
    {
      "entry_type": "memory_audited",
      "key": "...",
      "payload": {
        "result": "clean",
        "source_ref": "...",
        "summary": "..."
      }
    }
  ]
}
```

suspicious_reported：

- 先 `message_push` 发给用户。
- 消息必须包含 `memory_id`、当前记忆摘要、可疑原因、原始证据说明、纠正提示。
- `message_push` 成功后调用 `finish_drift(status="completed", message_result="sent")`。
- `journal_append` 同样追加 `entry_type=memory_audited`，`payload.result` 写 `suspicious_reported`。
- `cursor_update` 清空 active 候选，并把 `next_action` 写成 `sample`。

paused：

- 如果抽到候选但没完成判断，`finish_drift(status="paused", message_result="silent")`。
- `scratchpad_update` 写清已经做到哪一步。
- `cursor_update` 必须写入：

```json
{
  "active_memory_id": "...",
  "source_ref": "...",
  "summary": "...",
  "stage": "sampled"
}
```

## 发消息格式

```text
我抽检到一条疑似有问题的长期记忆：

- memory_id: ...
- 当前记忆: ...
- 可疑原因: ...
- 证据: source_ref 对应的原始上下文显示 ...

如果你认为这条记忆确实有问题，直接纠正我即可，后续会在被动回复链路里处理。
```

## 要求

- 一次只抽检 1 条记忆。
- 不使用 `list_dir`。
- 不读取或写入 `history.json`、`audited.md`、`state.json`。
- 不临时查询 `memory2.db`；只能运行固定脚本。
- 不扩大 `fetch_messages.context`，固定为 10。
- 只审计带 `source_ref` 的记忆。
- 不审计 `source_ref` 为 `@post_response` 的旧记忆。
- 只做调查和汇报，不做清理。
- 如果判断为 `clean`，不要打扰用户。
- 如果判断为 `unverifiable_old_source`，也不要打扰用户，只记录已审计并结束。
- `finish_drift.message_result` 必须和实际动作一致。
- `finish_drift.status` 必须和实际进度一致。
- 结束前必须调用 `finish_drift`。
