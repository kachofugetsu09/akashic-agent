# Builtin Skills Index

本文件导航仓库中的技能源码，不代表运行时已安装目录。运行时只发现普通插件通过资产 provider 注册的固定制品；仓库顶层 `skills/` 不会自动发布。

## 目录与格式

- 每个技能目录必须包含 `SKILL.md`。
- `SKILL.md` 建议包含 frontmatter：`name`、`description`、`metadata.akashic`。
- 主循环可按需读取具体技能文件；本索引只做发现与导航，不承载执行细节。

## 当前内置技能

- `develop-akashic-plugin`
  - 在 canonical source 中创建或修改 Akashic 插件及插件内 Skill/MCP，并按 stable/latest 合同做递归行为验证。
  - 文件：`skills/develop-akashic-plugin/SKILL.md`

- `feed-manage`
  - 管理和查询 RSS/信息来源订阅，支持列订阅、查最新、查概况、关键词搜索。
  - 文件：`skills/feed-manage/SKILL.md`

- `create-drift-skill`
  - 在工作区 drift/skills 下创建或更新 drift skill。
  - 文件：`skills/create-drift-skill/SKILL.md`

- `codex-delegate`
  - 把长代码库任务委托给本机 Codex CLI 后台执行，并等待完成后回灌结果。
  - 文件：`skills/codex-delegate/SKILL.md`

- `akashic-call`
  - 指导 Codex 或其他外部程序调用已运行的固定 Akashic runtime，并复用持久 thread。
  - 文件：`skills/akashic-call/SKILL.md`

- `skill-creater`
  - 创建或改写技能 `SKILL.md`，用于新增技能与结构迁移。
  - 文件：`skills/skill-creater/SKILL.md`

- `plugin-system`
  - 说明安装、更新时机、局部换代、卸载排空和结果核对；由内置 `plugin_update` 插件注册，与外部 Skill 使用同一资产链。
  - 文件：[`plugins/plugin_update/skills/plugin-system/SKILL.md`](../plugins/plugin_update/skills/plugin-system/SKILL.md)

- `summarize`
  - 总结 URL/文件/YouTube 内容，支持提取转写。
  - 文件：`skills/summarize/SKILL.md`

- `weather`
  - 通过 wttr.in / Open-Meteo 查询天气与预报。
  - 文件：`skills/weather/SKILL.md`

## 维护约定

- 发布内置技能：放入所属插件的制品，通过 `INSTALLED_ASSETS.register` 注册，并验证目录及 `load_skill` 正文；只放顶层目录不算交付。
- 删除内置技能：移除条目，避免索引悬空。
- 本索引仅用于源码导航；运行时实际可用性以 Skill provider 的目录和读取结果为准，不扫描 workspace 手工技能目录。
