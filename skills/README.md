# Builtin Skills Index

本目录是随 Core 发布的内置 Skill：发行版把 `skills/` 打进 Core 制品，`standard_tools` 的 Skill 目录把它作为 `builtin` 来源读取。优先级从低到高：插件随包资产 < 内置 `skills/` < `~/.akashic/skills` < `<workspace>/skills`，高层同名覆盖低层。

## 目录与格式

- 每个技能目录必须包含 `SKILL.md`。
- `SKILL.md` 建议包含 frontmatter：`name`、`description`、`metadata.akashic`。
- 主循环可按需读取具体技能文件；本索引只做发现与导航，不承载执行细节。

## 当前内置技能

- `develop-akashic-plugin`
  - 在 canonical source 中创建或修改 Akashic 插件及插件内 Skill/MCP，并按 stable/latest 合同做递归行为验证。
  - 文件：`skills/develop-akashic-plugin/SKILL.md`

- `create-drift-skill`
  - 在工作区 drift/skills 下创建或更新 drift skill。
  - 文件：`skills/create-drift-skill/SKILL.md`

- `codex-delegate`
  - 把长代码库任务委托给本机 Codex CLI 后台执行，并等待完成后回灌结果。
  - 文件：`skills/codex-delegate/SKILL.md`

- `skill-creater`
  - 创建或改写技能 `SKILL.md`，用于新增技能与结构迁移。
  - 文件：`skills/skill-creater/SKILL.md`

- `plugin-system`
  - 说明安装、更新时机、局部换代、卸载排空和结果核对；由内置 `plugin_update` 插件注册，与外部 Skill 使用同一资产链。
  - 文件：[`plugins/plugin_update/skills/plugin-system/SKILL.md`](../plugins/plugin_update/skills/plugin-system/SKILL.md)

- `weather`
  - 通过 wttr.in / Open-Meteo 查询天气与预报。
  - 文件：`skills/weather/SKILL.md`

## 维护约定

- 发布内置技能：放入所属插件的制品，通过 `INSTALLED_ASSETS.register` 注册，并验证目录及 `load_skill` 正文；只放顶层目录不算交付。
- 删除内置技能：移除条目，避免索引悬空。
- 本索引仅用于源码导航；运行时实际可用性以 Skill provider 的目录和读取结果为准。运行时另外按来源分层发现本地目录中的 Skill（见 [0091](../docs/decisions/0091-skill-sources-are-layered-providers.md)），本索引不列举这些目录。
