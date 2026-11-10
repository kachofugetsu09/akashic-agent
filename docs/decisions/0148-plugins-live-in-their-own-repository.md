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

## 理由

- 合成提交让拆仓只改"源码从哪来"，不改发布、回执、回滚、Bridge、镜像的任何身份规则，风险集中在一个可演练的函数里。
- 路径不变，插件导入 ABI（`plugins.<name>`）、来源前缀检查、迁移目录扫描、边界门都无需改写。
- 验收：同一份源码分别用 monorepo 和"Core + 插件仓库"构建，78 个插件的 `content_commit` 和 bundle 字节完全一致；
  所以拆分上线时生产会是"0 个插件变化"。

## 已知限制

- 插件改动上线仍需要 Core 提交一次 `plugins.lock`；插件仓库不会自动开 Core PR（避免给仓库加跨仓写权限的 secret）。
- `docker/computer` 与它的镜像发布工作流留在 Core：镜像按 digest 引用，搬走需要在 GHCR 包设置里给新仓库授权。
- 插件仓库 PR 暂不跑 harness-performance 对比。
- 只改插件、无需重启的部署目前仍走完整发布；按改动分级热更新是 #1973 的后续步骤。

## 影响与回滚

- 旧 Core 回滚兼容：`distribution_source` 仍是 40 位提交号，旧发布器能读新回执。
- 回滚拆仓：还原 Core 里被删除的路径（合成树就是完整源码）并删除 `plugins.lock`；插件仓库可以保留只读。
