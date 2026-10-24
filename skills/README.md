# Builtin Skills Index

本目录是随 Core 发布的内置 Skill：发行版把 `skills/` 打进 Core 制品，`skills` 插件的技能目录把它作为 `builtin` 来源读取。优先级从低到高：插件随包资产 < 内置 `skills/` < `~/.akashic/skills` < `<workspace>/skills`，高层同名覆盖低层。

## 目录与格式

- 每个技能目录必须包含 `SKILL.md`。
- `SKILL.md` 建议包含 frontmatter：`name`、`description`、`metadata.akashic`。
- 主循环可按需读取具体技能文件；本索引只做发现与导航，不承载执行细节。

## 当前内置技能

- `develop-akashic-plugin`
  - 给自己扩展能力：编写工具、WebUI 页面、Skill 或后台服务插件；附扩展点速查、WebUI 限制和可直接安装的模板。
  - 文件：`skills/develop-akashic-plugin/SKILL.md`

- `verify-akashic-plugin`
  - 把写好的插件提交、安装到当前实例（`plugin_install` 工具或实例自带命令行），核对状态并实际用一次后汇报。
  - 文件：`skills/verify-akashic-plugin/SKILL.md`

- `develop-and-verify`
  - 改带界面或服务的项目（含 Akashic 自身）时，在宿主上隔离地起真实服务、假掉外部依赖、用 computer 浏览器（`http://localhost:<端口>`）真实操作截图，验完再提 PR；附 OpenAI 兼容假模型脚本与 Akashic 开发实例参考。
  - 文件：`skills/develop-and-verify/SKILL.md`

- `codex-delegate`
  - 把长代码库任务委托给本机 Codex CLI 后台执行，完成时由后台任务通知回到当前会话。
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

- 发布内置技能：与某个插件绑定的放入该插件制品并通过 `INSTALLED_ASSETS.register` 注册；通用技能放本目录。两者都要在真实实例里验证目录和 `load_skill` 正文。
- 改动插件开发相关技能时，同步核对模板能真实安装：`develop-akashic-plugin/templates/` 下每个模板都应能按 `verify-akashic-plugin` 装上并用起来。
- 删除内置技能：移除条目，避免索引悬空。
- 本索引仅用于源码导航；运行时实际可用性以 Skill provider 的目录和读取结果为准。运行时另外按来源分层发现本地目录中的 Skill（见 [0091](../docs/decisions/0091-skill-sources-are-layered-providers.md)），本索引不列举这些目录。
