---
name: create-drift-skill
description: 在 Drift 工作区创建或更新一个 skill，让空闲时发现的新兴趣沉淀成可再次选择的活动。
---

# 创建 Drift Skill

## 目标

把空闲时仍愿意再次做的小活动沉淀到 `skills/<skill_name>/SKILL.md`。

本轮处于设计期，只负责给未来的 Drift 写一份可选择的活动说明书，不执行新活动本身。

## 未来如何被选择

新 skill 不会因为文件创建成功就立刻运行，也没有独立的定时触发器。

```text
当前 create-drift-skill
└─ 写入新 SKILL.md
   └─ 从下一轮 Drift 起进入候选列表
      ├─ runtime 展示 name、description 和过去状态
      ├─ agent 结合空闲时的意图、环境和前情自由选择
      └─ 选中后才读取正文并执行活动
```

因此，frontmatter 的 `description` 不只要说明“能做什么”，还要说明“什么情况下可能想选它”。
正文则说明选中以后如何完成一次最小活动。

## 何时使用

- 发现一种以后还可能想做的活动
- 现有 drift skill 太旧，需要补充流程或 working files
- 不要因为“应该扩充能力”而强行创建；没有自然想法时可以做别的或休息

## 工作流

1. 先说明为什么以后还会想做这件事，以及什么空闲情境下可能会选择它。
2. 从这个活动形成一个不同于 `create-drift-skill` 的目标名称，再检查 `skills/<skill_name>/` 是否已存在。不要把元 skill 自己当成目标 skill。
3. 读取目标的 `SKILL.md`；如果已存在就在原基础上更新，不存在再创建。
4. 只把可长期复用、可独立闭环的小活动沉淀成 drift skill；一次性进展写入 `finish_drift`，不要创建新 skill。
5. 先定义“何时可能选择”和一次 Drift run 的最小闭环，再决定是否需要脚本。
6. `SKILL.md` 顶部 frontmatter 至少包含：

```text
---
name: <skill_name>
description: <做什么，以及什么空闲情境下适合选择>
---
```

7. 正文只写未来选中后真正需要的最小流程，避免空泛模板。

如果活动需要了解用户兴趣，可以在设计期查找足以确定方向的上下文。但不要提前执行新活动：未来运行所需的检索、检查和工具调用，应写进新 skill 的流程，留到它以后真正被选择时执行。

搜索只服务于会改变目标名称、选择情境或单次闭环的信息。现有理解已经足以写出可修改的初版时，直接落盘；证据不足可以成为新 skill 未来运行时需要验证的内容。

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

- skill 文件必须通过 runtime 文件工具写到 `skills/<skill_name>/`，不要使用宿主机绝对路径
- 不要为了一个一次性动作创建 skill
- 如果只是当前 skill 的进展变化，优先通过 `finish_drift` 的 `scratchpad_update`、`cursor_update` 或 `journal_append` 保存连续性，不要修改 skill 文件
- 如果需要确定性处理、抽样、生成文件或读取 cursor/journal，再放一个最小脚本到 `scripts/`
- 结束流程必须写清 `finish_drift.status`：本轮创建或更新已闭环写 `completed`，尚未写完写 `paused`
- `paused` 必须写 `scratchpad_update`，说明下次从哪里继续
- `self_update.next_tendency` 写下次是否想试用新 skill，或暂时把它放下
- 需要脚本连续执行时，必须写清脚本如何从 `cursor_update` 产生的 cursor 里读下一步
- 已完成事实必须通过 `journal_append` 记录，避免下轮重复处理同一对象

## 推荐正文结构

```text
# <Skill 标题>

## 目标

一句话说明这个 drift skill 每次空闲时维护什么。

## 何时适合选择

- 哪种空闲意图、环境或前情下可能想做。
- 哪些情况下暂时不做。

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

- 成功闭环：finish_drift(status="completed", self_update={"next_tendency": "..."}, ...)
- 未完成但可继续：finish_drift(status="paused", scratchpad_update="...", self_update={"next_tendency": "..."}, ...)
```

## 当前元 skill 的完成条件

- 目标 `skills/<skill_name>/SKILL.md` 已成功写入。
- 文件包含可供未来选择的 `description`、选择情境和单次闭环。
- 不要求本轮试运行新 skill。
- 收尾时 `self_update.next_tendency` 记录未来某次 Drift 是否可能想试用它，而不是在本轮继续执行它。
