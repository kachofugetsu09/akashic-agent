# 0167 · 非核心插件由官方 extra 仓库提供，Core 用 pkg 管理

- 状态：accepted / implemented
- 日期：2026-11-22
- 取代：[0148](0148-plugins-stay-in-core.md)
- 过程记录：[插件外移与包管理迁移计划](../design/plugin-packages-migration.md)

## 背景

0148 撤回了 #1973 的拆仓。撤回的原因是安装与更新流程没有跟上：一个插件改动要经过插件 PR、升 lock、部署三步；镜像、制品库、覆盖三处都可能是来源，说不清“现在跑的是哪份”。拆仓这个方向本身没有被否定。

这一次先重做初始化、安装、更新流程，再迁插件。参照 Arch 的分法：Core 是 `core` 仓库，官方插件是 `extra` 仓库，`pkg` 是 pacman。

## 决定

**Core 只保留最小闭环。** 也就是裸核能独立跑完长程循环（账本、上下文、压缩、工具、文件与 shell、一个通用模型驱动），能提供 Web 宿主与投递，能装卸插件。其余插件由公开仓库 `kachofugetsu09/akashic-plugins` 提供。分层写在 `plugin_boundary.toml` 的 `[tiers]`，门禁 R14 保证 `core` 层不 import `extra` 层。

**一个插件只有一个来源，版本只看 extra 的 `release/<名>` 分支。** Core 镜像不带 extra 插件，也没有 lock。首次安装和 `pkg upgrade` 时联网取当前 release 提交。`index` 分支只承载包组、说明、包间依赖、外部前置条件和镜像声明，不承载版本。extra 的 CI 只推 `release/*` 与 `index`，不删分支。

**`akashic pkg` 是唯一的日常入口。**

- 命令：`list`、`groups`、`install <名>|@<组>`、`upgrade [<名>]`、`remove`、`sync`。
- 在线实例上，命令经 Gateway 安装并热重载；离线时持插件目录的发布锁直接安装。
- `install` 按 index 补上还没装的依赖，依赖在前；依赖没有就绪时，不装依赖它的包。
- 设置页“插件”里的“可安装”分组列出 index 里还没装的包，一键安装走同一条 `pkg install`（后台任务，跨换代可查结果）。
- `pkg` 只管官方仓库；私人插件和其他来源继续用 `plugin-install --marketplace`。两条路径共用同一套安装制品、选择和生命周期。

**重依赖由插件自己声明。**

- 进程内依赖写在插件根的 `requirements.inprocess.txt`，装进插件自己的环境，挂在主进程 `sys.path` 末尾（Core 的包优先）。版本受 Core 锁约束（`pip -c`）。环境按声明内容复用；Core 锁变化本身不会重建健康的环境。
- 大文件（例如 Chromium）写在 `resources.json`，发布到 `runtime/plugin-resources/<sha>/`，完成后写 `COMPLETE.json`。插件经 `PluginRuntime.resource(name)` 取用。
- 启动时的资源修复是 60 秒内的尽力尝试；`pkg upgrade <名>` 做单包修复。
- 这些依赖从 Core 镜像删掉了：python-telegram-bot、telegramify-markdown、numpy、numba、llvmlite、jieba、threadpoolctl、lxml、html2text、pyromark，以及镜像自带的 Chromium。

**外部前置条件不满足时，照装但保持关闭。**

- 前置条件只有两种形式：`docker`、`command:<命令>`。
- 原因记在 `runtime/disabled-reasons.json`（带制品 head，head 变了视为没有记录），设置页、`pkg list`、`plugin-status` 都能看到。
- 启用时重新检测。声明了镜像的插件，启用前先确认镜像在宿主上。
- 前置条件补齐后在线启用（设置页，或 Agent 按内置 `plugin-system` 技能执行 `plugin-set-enabled`）。前置条件在 Core 进程里检测：Agent 的 Shell 在 Host Bridge 上，在那里装的命令不算数。

**首次安装在终端里选。**

- 有 TTY 时问三档：全部（默认）、最小、自选组；没有 TTY 时全装。“全部”指 extra 的全部公有插件。
- 先列出要装的包，再逐包显示 `(i/N)`。
- 联网或安装失败就退出，不进系统，可以重跑。
- 未选的 preset 行以 `disabled=true` 写进用户 `bundle.patch.toml` 的自动生成块，块外用户写的行优先。
- 网页引导不变，只管配置。
- 已有实例的升级不走首装。

**被 Core 收回的包有明确记录。** `bundles/absorbed.toml` 登记“已并入 Core 的哪个插件”（目前只有 `session_reference` → `conversation`）。升级时把它移出选择，`pkg install` 拒绝再装。

## 验收方式

迁移的不变量是：全量安装的实例，迁移前后能力完全一致。

- 每一批先在本机演练：用生产形态的状态和真实镜像升级，用 `capability_snapshot.py diff --ignore-source --ignore-build` 比对，必须为空或只有预期的来源变化。
- 有数据的插件核对数据摘要。
- 生产部署前后再比一次快照，并真实发一条消息。

## 已知限制

- `pkg upgrade`、`sync` 不补依赖，只处理点名或已装的包；Agent 直接调 `plugin_install` 也不经过依赖补全。
- 依赖补全读 index 的本地缓存；index 过期时可能漏掉新增的依赖，安装本身仍以 release 提交为准。
- 已选中但当前加载失败或被关闭的依赖不会被 `install` 修复；`pkg list` 会显示它们的状态。
- 首装自动生成块的两个边界：用户把一模一样的块粘进多行字符串；用 tomlkit 删掉块里最后一行时丢了结束标记。两种情况都会被识别为“块已损坏”并拒绝重写，需要手动修。
- `command:` 前置条件在 Akashic 运行环境里检测。容器部署的根文件系统只读，这类命令通常要由镜像或宿主提供，Agent 无法自己补齐；让 Agent 能给 Core 装命令行工具的路径见 issue #2223。
- 只改插件不再需要发布 Core；只改 Core 仍要完整发布并重启一次。
