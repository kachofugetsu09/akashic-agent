# 0148 · 内置插件迁到独立仓库，按内容寻址发布

- 状态：accepted / implemented for review
- 日期：2026-11-10
- 关联条款：PLG-001、PLG-013、RUN-004、MIG-001
- 依据：#1973；用户拍板"插件放独立仓库、变成外置"，阶段一的契约/UI SDK/主题/preset 行仍随 Core 发版。
- 细化：[0108](0108-bundles-are-composition-inputs.md)。bundle 仍是组合输入，只是文件搬到插件仓库。

## 背景

内置插件的身份原来是"整仓源码提交"：任何一行 Core 改动都让 78 个内置插件全部换新身份，部署时全部替换、全部重启。
最近 38 次部署里有 17 次只改了一个插件的前后端（可热更新），10 次需要重启，11 次是 Core 改动。

## 决定

1. **插件按内容寻址。** 构建器为每个插件生成固定作者、固定时间的内容提交，`content_commit` 只由插件目录内容决定；
   发行版 `distribution_source` 改用它。Core 改动不再让未改动的插件换身份，部署只替换内容真正变化的插件。
2. **preset 必须进入选择。** 发行版 base bundle 里启用的行，必须出现在发布和启动的选择里，缺了就停止发布；
   `plugin-status.preset {bundle, missing, retired}` 把结果暴露给 doctor 和激活验收。
3. **插件迁到 [akashic-plugins](https://github.com/kachofugetsu09/akashic-plugins)。** 路径保持 monorepo 时的布局，
   两仓文件互不重叠。Core 根目录的 `plugins.lock` 锁定插件仓库 `main` 上的一个提交。
4. **发布身份是合成提交。** `scripts/compose_source.py` 把 Core 提交和锁定的插件提交取并集，写成无父提交、固定身份的
   合成提交；同样两个输入永远得到同一个提交号。发布器、自部署、`./start`、独立镜像、发行版构建与 CI 都用它，
   所以"单提交"发布模型不变。
5. **检查只维护在 Core。** `ci.yml`、`plugin-boundary.yml` 可被插件仓库以 `workflow_call` 调用，用 Core main 加插件 PR 提交合成后跑同一套检查。
6. **开发共用一个工作目录。** `scripts/plugins_workspace.py attach` 把插件仓库检出到 Core 的工作目录，`scripts/plugins-git` 操作插件仓库。
7. **实例可以单独覆盖一个插件。** 参照 DSH 的"profile 里装的版本优先于安装目录自带的版本"：`plugins.lock` 是实例默认用的版本，
   实例用 `--plugin <name>@<插件仓库 main 上的提交>` 把一个插件钉到更新的版本。合成时只替换这个插件拥有的
   `plugins/<name>/` 与 `frontend/plugins/<name>/`，覆盖写进合成提交说明和发行版 `source_parts.overrides`。
   插件身份（`<name>@release`）、数据目录、配置都不变，所以照常走热更新判定与兼容门。
   覆盖只记录在运行中的发行版里：之后的部署、重启、`revert-hot` 都跟着发行版走，没有第二份状态。
   覆盖只接受插件仓库 `main` 上的提交，和锁定提交总在同一条历史上；部署时锁定提交已包含覆盖提交就自动退役，
   结果里的 `retiredOverrides` 列出退役项。远端全部合进去以后，实例就回到纯 lock 状态，和新装实例逐字节一致。

## 理由

- 合成提交让拆仓只改"源码从哪来"，不改发布、回执、回滚、Bridge、镜像的任何身份规则，风险集中在一个可演练的函数里。
- 路径不变，插件导入 ABI（`plugins.<name>`）、来源前缀检查、迁移目录扫描、边界门都无需改写。
- 验收：同一份源码分别用 monorepo 和"Core + 插件仓库"构建，78 个插件的 `content_commit` 和 bundle 字节完全一致；
  所以拆分上线时生产会是"0 个插件变化"。

## 已知限制

- 新装实例用的插件版本仍需要 Core 提交一次 `plugins.lock`；插件仓库不会自动开 Core PR（避免给仓库加跨仓写权限的 secret）。
  单个实例可以先用覆盖上线，lock 之后再跟上。
- 覆盖不接受分支或 PR 提交：分支提交被 squash 或删除后旧版本无法重建，也无法判断远端是否已经包含它。
- 覆盖的插件仍和锁定提交里的共享代码（`packages/`、`frontend/chat` 等）一起构建；覆盖版本依赖新的共享代码或新的 Core 合同时，
  构建或兼容门会拒绝，此时要先升 lock 或 Core。
- 锁定提交只追上了覆盖、没有带来其他插件变化时，热更新判定是"没有插件变化"，走一次完整发布（多重启一次）。
  发布器无法在不比较两次合成整棵树的情况下区分这种情况与只改发布工具的 Core 提交，后者必须完整发布。
- `docker/computer` 与它的镜像发布工作流留在 Core：镜像按 digest 引用，搬走需要在 GHCR 包设置里给新仓库授权。
- 插件仓库 PR 暂不跑 harness-performance 对比。
- 热更新（只改插件、Core 与依赖不变）由发布器自动选择，见 `docs/design/operator-deployment.md`。界面探针不在发布器内自动运行：
  它需要浏览器，且要等插件重载结束；部署后由操作者或 Agent 运行，失败时 `akashic-release revert-hot`。
- 发行版行没有运行时字段，兼容门的运行时契约检查目前不生效；热更新要求同一镜像，影响有限。
- 热更新在实例换代成功后才改写 `runtime.env`。两步之间宿主进程崩溃时，下次启动按旧发行版 `ensure_bundle`，
  选择整体回到旧输入；结果一致，只是这次热更新没有生效，重新发布即可。
- 热更新被操作期限撤销时（新输入已提交、新版本仍在启动），实例不自行退回，热更新报告 `failed`，
  受影响插件在发布器随即进行的完整发布重启前不可用；完整发布按目标版本重新提交全部选择。
- 以 root 运行发布器时，制品库交还 release 根目录的属主，`runtime.env`、`active.json` 保持原属主；
  不支持 root 与普通用户以外的第三方共同写入。
- 回滚到 #1973 之前的 Core 必须用目标版本的发布器（自部署本来如此）：旧镜像的发行版不进制品库，
  运行时用镜像内路径，内置输入随之整体替换一次；状态里没有 `preset` 时只做原有核对。
- 尚未生成设计中包含合同、运行时、UI、迁移事实的 preset lock；`plugin-status.preset` 只报告 bundle、missing、retired。

## 影响与回滚

- 旧 Core 回滚兼容：`distribution_source` 仍是 40 位提交号，旧发布器能读新回执。
- 回滚拆仓：还原 Core 里被删除的路径（合成树就是完整源码）并删除 `plugins.lock`；插件仓库可以保留只读。
