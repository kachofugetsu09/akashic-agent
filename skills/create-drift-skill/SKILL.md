---
name: create-drift-skill
description: 在工作区 drift/skills 下创建或更新一个 drift skill，用于把新的长期小任务沉淀成可复用技能。
---

# 创建 Drift Skill

## 目标

把适合反复执行的小任务沉淀到工作区 `drift/skills/<skill_name>/SKILL.md`。

## 何时使用

- 发现有新的长期任务适合放进 drift
- 现有 drift skill 太旧，需要补充流程或 working files

## 工作流

1. 先确认目标 skill 名是否明确，并检查 `drift/skills/<skill_name>/` 是否已存在。
2. 读取已有 `SKILL.md`，如果已存在就在原基础上更新；不存在再创建。
3. 只把可长期复用、可独立闭环的小任务沉淀成 drift skill；一次性进展写入 `finish_drift`，不要创建新 skill。
4. 先定义一次 drift run 的最小闭环，再决定是否需要脚本。
5. `SKILL.md` 顶部 frontmatter 至少包含：

```text
---
name: <skill_name>
description: <一句话描述>
---
```

6. 正文只写完成当前任务真正需要的最小流程，避免空泛模板。

## 状态模型

新 drift skill 必须使用 runtime 统一状态，不要自行维护并行状态文件：

```text
drift run
├─ scratchpad_update
│  └─ 自然语言说明下次从哪里继续
├─ cursor_update
│  └─ 结构化游标，供脚本或下轮流程直接决定下一步
└─ journal_append
   └─ append-only 记录已经完成、问过、审计过、生成过的事实
```

- `scratchpad_update`：只保存自然语言前情，例如“下次先检查哪个文件”。
- `cursor_update`：只保存下一轮需要稳定读取的结构化字段，例如 `next_mode`、`last_category`、`next_action`。
- `journal_append`：只追加已完成事实，例如已问过的问题、已生成的文件、已审计的 memory id。
- 不要新建或继续使用 `history.json`。
- 不要把连续性状态写到 skill 目录下的 `state.json`。
- 脚本需要自动决策时，可以读取 `drift.db` 中本 skill 的 `cursor_json` 和 `skill_journal`，但写入状态必须通过 `finish_drift` 完成。

## 约束

- skill 文件必须写到工作区 `drift/skills/` 下，不要写到仓库内建目录
- 不要为了一个一次性动作创建 skill
- 如果只是当前 skill 的进展变化，优先通过 `finish_drift` 的 `scratchpad_update`、`cursor_update` 或 `journal_append` 保存连续性，不要修改 skill 文件
- 如果需要确定性处理、抽样、生成文件或读取 cursor/journal，再放一个最小脚本到 `scripts/`
- 结束流程必须写清 `finish_drift.status`：完成写 `completed`，未完成写 `paused`，等待条件写 `waiting`
- 结束流程必须写清 `finish_drift.message_result`：已成功推送写 `sent`，静默结束写 `silent`
- `paused` / `waiting` 必须写 `scratchpad_update`，说明下次从哪里继续或等待什么
- 需要脚本连续执行时，必须写清脚本如何从 `cursor_update` 产生的 cursor 里读下一步
- 已完成事实必须通过 `journal_append` 记录，避免下轮重复处理同一对象

## 推荐正文结构

```text
# <Skill 标题>

## 目标

一句话说明这个 drift skill 每次空闲时维护什么。

## 单次闭环

1. 读取必要上下文。
2. 执行一个最小动作。
3. 判断是否需要打扰用户。
4. 调用 finish_drift 保存状态。

## 状态延续

- scratchpad：保存自然语言前情。
- cursor：保存脚本下次自动决策所需字段。
- journal：追加已完成事实，避免重复。

## 工具与脚本

- 如无脚本，说明只用 runtime 工具。
- 如有脚本，列出固定命令和输出 JSON 语义。

## 收尾

- 成功闭环：finish_drift(status="completed", ...)
- 未完成但可继续：finish_drift(status="paused", scratchpad_update=...)
- 等待用户或外部条件：finish_drift(status="waiting", scratchpad_update=...)
```
