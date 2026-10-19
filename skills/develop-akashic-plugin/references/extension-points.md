# 扩展点速查

先按"想做什么"选扩展点，再去读表里的合同文件确认签名。签名以合同文件为准，本表不复制。
表中插件路径相对已核对的插件源码目录（分发的 `sources/` 或宿主运行 checkout 的 `plugins/`），`agent/...` 相对 Core 目录。定位方式见 SKILL.md；不假设宿主有容器内的环境变量或路径。

| 想做什么 | 用什么 | 合同文件 | 可参照的真实插件 |
|---|---|---|---|
| 给模型加一个可调用的工具 | `TOOLS`：`declare_group` 后 `register` | `tools/contract.py`（`ProviderBoundTool`、`ResultLike`） | 模板 `templates/tool/plugin.py`；`message_push/plugin.py` |
| 随插件发布 Skill | `INSTALLED_ASSETS.register(ctx, "skills", "skills")`，插件根目录放 `skills/<名>/SKILL.md` | `assets/contract.py` | `plugin_update/plugin.py` |
| WebUI 加页面、底栏动作、设置分节 | `UI.register(ctx, web="web_module.js", requires=(挂载点,))` | `ui/contract.py`；浏览器侧见 `references/web-ui.md` | 模板 `templates/page/` |
| 每轮往提示词里加一段上下文 | `MATERIALS_V4.register(ctx, name=..., kind="context", prepare=...)`，`prepare(messages, source)` 返回 `{"reminders": ({"name", "priority", "replay", "text"},)}` | `context/contract.py` | `working_directory/plugin.py` 的 working-directory |
| 后台常驻任务 | `await ctx.spawn(coro, name=...)`，随插件卸载取消 | `agent/plugin_composition/context.py` | `jobs/plugin.py` |
| 需要关闭的资源（连接、文件、线程） | `await ctx.effect(lambda: close, label=...)`：setup 返回清理函数 | 同上 | `timer/plugin.py` |
| 给别的插件提供服务 | 在自己的 `contract.py` 定义 `ServiceKey[协议]("名字.v1")`，`await ctx.provide(KEY, impl)` | 同上 | `timer/contract.py` + `timer/plugin.py` |
| 定时或延时 | `TIMERS` | `timer/contract.py` | `scheduler/plugin.py` |
| 执行 shell 命令、读写文件 | `PROCESSES`、`FILES`，不要自己 `subprocess` | `host_execution/contract.py` | `shell/shell.py` |
| 主动给用户发消息 | 优先用已有 `message_push` 工具；实现发送者或送达流程时分别核对 `DELIVERY_SENDERS`、`DELIVERY_GUARDED_START` | `delivery/contract.py` | `message_push/plugin.py`、`message_push/boundary.py` |
| 插件自己的配置和数据 | `ctx.config`（固定输入，空时为 `{}`），`ctx.data_root`（持久目录） | `agent/plugin_composition/context.py` | `akasha/plugin.py` |

## 依赖与生命周期

- `inject` 里列出的 ServiceKey 是硬依赖：提供方缺席时本插件不会激活。可选分支用 `await ctx.inject((KEY,), setup)`；临时可选调用用 `with ctx.borrow(KEY) as service:` 并判空，不能在借用范围外保存服务。`ctx.get` 不允许绕过依赖声明。
- `apply` 里注册的一切（工具、页面、材料、任务、effect）都归本插件；卸载或换代时宿主按相反顺序释放，不需要手写清理入口。
- 新版本安装后，已在进行中的调用仍用旧代码；新调用才看到新代码。
