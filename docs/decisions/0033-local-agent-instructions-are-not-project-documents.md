# 0033 · 本地 Agent 指令不属于版本化项目文档

- 状态：accepted
- 日期：2026-08-11
- supersedes：[0001](0001-project-workbook-is-shared-reality.md)
- 关联条款：WBK-001～WBK-006、COM-001～COM-004

## 背景

根目录 `AGENTS.md` 与 `CLAUDE.md` 由本地 coding agent 运行环境提供。把它们提交到仓库会把个人工具配置与项目事实混在一起，也会让不同运行环境争夺同一路径。

## 决定

`AGENTS.md` 与 `CLAUDE.md` 保持本地且被 Git 忽略。仓库内可共享、可评审的协作规则由 `docs/INDEX.md`、`docs/WORKFLOW.md`、`docs/projectneed.md`、`docs/NOW.md`、`docs/decisions/` 和 `docs/writing-rules.md` 保存。

## 理由

项目文档应描述项目事实，不应绑定某一种 coding agent 的本地注入格式。固定入口和执行顺序已经分别由 `INDEX.md` 与 `WORKFLOW.md` 拥有。

## 影响

- CI 不再要求根 `AGENTS.md` 存在或进入版本控制。
- `AGENTS.md` 与 `CLAUDE.md` 继续由本地运行环境管理。
- 项目协作规则发生变化时更新版本化工作手册，不提交本地 agent 指令。

## 验收

- 干净 checkout 不需要根 `AGENTS.md` 即可通过工作手册契约测试。
- `INDEX.md` 仍是每个新会话的固定入口。
- 修改任务仍按 `WORKFLOW.md` 执行。
