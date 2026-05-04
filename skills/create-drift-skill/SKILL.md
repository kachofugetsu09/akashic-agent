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
3. `SKILL.md` 顶部 frontmatter 至少包含：

```text
---
name: <skill_name>
description: <一句话描述>
---
```

4. 正文只写完成当前任务真正需要的最小流程，避免空泛模板。

## 约束

- skill 文件必须写到工作区 `drift/skills/` 下，不要写到仓库内建目录
- 不要为了一个一次性动作创建 skill
- 如果只是当前 skill 的进展变化，优先更新它的 working files 或 state，而不是新建 skill
- 结束流程必须写清 `finish_drift.message_result`：已成功推送写 `sent`，静默结束写 `silent`
