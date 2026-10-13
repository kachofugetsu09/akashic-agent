---
name: develop-akashic-plugin
description: 给自己扩展新能力时使用：编写一个 Akashic 插件（新工具、WebUI 页面、随插件发布的 Skill、后台服务），安装到当前运行实例并验收。用户说做个插件、加个工具/功能/页面、扩展你的能力、改某个插件时使用。
---

# 开发 Akashic 插件

插件是给自己加能力的唯一正式方式：写源码 → git commit → 安装到当前实例 → 实际用一次验收。
不要改安装 cache、`plugin-home`、`runtime/` 下的指针或别的插件的 `plugin-data`，也不要手写 manifest。

## 流程

1. **定目标和形态**：要的是模型可调用的工具、WebUI 页面、Skill，还是后台服务？只做能交付的最小版本。
2. **建源码仓库**：在 `$AKASHIC_WORKSPACE/plugin-src/<name>/` 新建目录并 `git init`。一个插件一个仓库，插件根目录必须有 `plugin.py`。
3. **从模板起步**：按下表用文件读取工具读模板，写一份到源码目录，改名字和逻辑。`name` 在当前实例里必须唯一。
4. **只查需要的接口**：按下面的知识来源顺序读真实代码，不凭记忆写 API。
5. **提交并安装**：`git add -A && git commit`，然后加载 `verify-akashic-plugin`，按它安装并验收。
6. **修到能用为止**：失败就改同一个插件，重新 commit、重新安装新的 commit。能用后汇报插件 ID、commit、验收证据和没验证到的部分，不追加没被要求的功能。

## 知识来源，按顺序

1. 本 skill 的模板和 `references/extension-points.md`：先确定用哪个扩展点。
2. 真实合同：Core 的 `$AKASHIC_CORE_ROOT/agent/plugin_composition/context.py`（`Context` 能做什么）；
   提供方插件的 `$AKASHIC_PLUGIN_DISTRIBUTION/sources/<插件名>/contract.py`（ServiceKey、协议、数据类型）。
   插件源码不在 `$AKASHIC_CORE_ROOT` 下。
3. 同类的真实插件：找一个用同一 ServiceKey 的内置插件读它的 `plugin.py`。
4. 设计文档只作背景：`$AKASHIC_CORE_ROOT` 下没有 `docs/`；源码仓库的 `docs/design/plugin-v3-capabilities.md` 与代码冲突时以代码为准。

`load_skill` 返回的 `base_directory` 就是本 skill 目录，下表路径都相对它：

| 用途 | 文件 |
|---|---|
| 扩展点速查：工具、Skill、页面、提示词材料、后台任务、配置与数据 | `references/extension-points.md` |
| WebUI 模块的硬性限制、挂载点、主题 token、亮暗主题 | `references/web-ui.md` |
| 工具插件模板（一个文件） | `templates/tool/plugin.py` |
| 页面插件模板（三个文件） | `templates/page/plugin.py`、`templates/page/web_module.js`、`templates/page/web_module.css` |

## 省步数

一轮最多约 40 次模型输出，开发控制在 15 次以内，留足验收和汇报。只读模板和真正要用的那一两个合同文件，不通读插件源码；
WebUI 页面读 `references/web-ui.md` 和 `templates/page/` 就够了，不再去找 ui 合同、类型定义或别的页面插件。不要用单独的 python 去 import 插件做自测：
插件模块只有宿主加载器能导入，安装本身就是加载检查，失败原因看 `plugin-status` 的 `error`。

## 硬规则

- 唯一入口是 `async def apply(ctx)`；`api_version = 3`、`name`、`version` 写成顶层字面量。
- 依赖的服务写进 `inject`，用 `ctx.require(KEY)` 取。资源、监听、后台任务都交给 `ctx.effect` / `ctx.spawn` 管理，
  不在 import 时启动进程、开端口或发消息。
- 配置读 `ctx.config`（没有配置时是空对象），自己校验；插件自己的持久数据只写 `ctx.data_root`。
- 外部 Python 依赖写 `requirements.txt`，没有时不要假设第三方包已装。
- 不恢复已删除的旧协议：`class Plugin`、`apply(ctx, config)`、`is_active`、`skill_roots`、TOML MCP 声明、
  `plugin_latest`、stable/latest 候选晋升都已退役。
