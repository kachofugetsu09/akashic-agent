# 0148 · 内置插件留在 Core 仓库

- 状态：accepted / implemented for review
- 日期：2026-11-11
- 依据：#1973 栈（#1974–#1989、akashic-plugins #1–#8）上线后的使用反馈；用户拍板撤回。

#1973 把内置插件和 Web 界面拆到 akashic-plugins，Core 用 `plugins.lock` 锁定，并加了镜像外制品库、
只改插件时的热更新和单插件覆盖。上线后流程反而更绕：一个插件改动要经过插件仓库 PR、升 lock 的 Core PR
和部署三步；为了不升 lock 先用上，又加了覆盖和自动退役。插件代码同时存在于镜像、制品库和覆盖三处来源，
判断"现在跑的是哪份"变难。

本决定撤回整个栈，回到单仓库与镜像内发行版（a6d6f8e3 的发布流程）。改一个插件仍是一个 Core PR，
部署是一次完整发布。正交性由仓库内的边界保证，与 DSH 一样是同一仓库内的多个包，而不是多个仓库：

- 前端源码归属插件：`frontend/plugins/<插件>/`；共享的只能是 `packages/` 下声明 `akashic.kind`
  （`library`，或写明 `provider` 的 `contract`）的包；R13 检查。Core 的发行版构建只运行 `npm run build`，
  外壳产物由 `scripts/place-shell-assets.mjs` 放进所属插件，Core 不点名插件。
- 用户 patch 按字段覆盖 preset（ADR-0109 修订），没改过 preset 的实例跟随新 preset 的启停。

保留的其间修复：设置路由与 akasha 指纹不依赖代码目录名、通道停止时通知流正常结束、
首装与发布不再静默少装 preset（#1975）。

**已知限制：** 只改插件也要完整发布并重启一次。激活时从 `runtime.env` 去掉制品库时期的
`AKASHIC_PLUGIN_DISTRIBUTION`、`AKASHIC_PLUGIN_STORE`（宿主桥接读 `runtime.env`，留着会加载旧发行版的插件），
容器用镜像内的 `/opt/akashic/distribution`。`/srv/data/services/akashic/plugin-store` 不再使用，确认回不去后可删除。
akashic-plugins 仓库不再接收改动，是否归档另定。
