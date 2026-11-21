# 最小包管理器（Phase 1）设计

- 日期：2026-11-22
- 状态：设计稿，待拍板；本文不含实现
- 关系：落实 [plugin-packages-migration.md](plugin-packages-migration.md) 的 Phase 1；修订 [ADR 0148](../decisions/0148-plugins-stay-in-core.md) 的撤回原因（插件 PR、升 lock、部署三步；镜像、制品库、覆盖三处来源）。
- 数据目录前置：PR #2165（`/mnt/data/coding/akashic-pkg-datadir`，提交 64059c714）。本文凡写"#2165"都指该提交，它尚未合入 main。

文中标"推断"的内容没有读代码或没有实际运行确认；其余结论都带文件和行号，行号以本分支（基于 main d21475fc5）为准。

## 0. 结论摘要

1. 用 git 当索引可行，但"一个 `release` 分支，`plugins/<name>/` 的 tree 哈希当版本"和现有安装器不匹配：`install_git_plugin` 把仓库根当作插件根（`agent/plugins/install.py:182-186` 对 clone 根调用 `load_static_plugin_manifest`，`static_manifest.py:68-77` 要求根下有 `plugin.py`），`sparse` 只是 clone 选项，不会把子目录提升为根。要走子目录方案，得给安装器加"子目录即插件根"和"按子树哈希记身份"两处新代码。
2. 推荐的替代：**每个插件一个分支 `release/<name>`，分支内容就是插件根，CI 只在 tree 变化时推进该分支，且提交由 tree 唯一决定（无父提交、固定作者和时间）**。于是：
   - 版本 = 该分支的提交 SHA（等价于 tree 哈希，因为提交由 tree 唯一决定）；
   - 查"有没有新版本"只需一次 `git ls-remote <repo> 'refs/heads/release/*'`，不用克隆，对比已装 artifact 的 HEAD；
   - 安装就是现有 `install_git_plugin(source, ref_name=<sha>)`，一行新逻辑都不必为"子目录"写；
   - 已装 artifact 本来就是 clone 出来的 detached checkout（`install.py:353` 的 `copytree(clone_root, ...)` 把 `.git` 一起复制），HEAD 就是身份（`distribution_sources.py:193-202` 已经这样读）。
3. **身份（`name@marketplace`）是持久契约，别改**。ledger 的 `owner_records.owner` 形如 `plugin:<name>@<marketplace>`（`plugins/ledger/services.py:237`），Message metadata 的命名空间 key 就是完整 plugin_id（`services.py:141,171`、`log.py:2108`）。#2165 之后数据目录可以脱离 marketplace，但这两处不能。把 `watch@release` 改名 `watch@extra` 就要迁 ledger 和历史消息 metadata，三十几个插件各写一份迁移，风险大、收益为零。**推荐：从 Core 迁出的插件继续使用 `@release` 这个身份命名空间，"来源"另外表达，不进 plugin_id**。这样"换来源"只改选择记录里的代码路径和来源类型，其余持久状态全部原样有效。
4. 内置到 extra 的衔接必须发生在**停机发布的同一次事务里**，不能靠 Core 起来之后再联网补装：发布容器和预检都是 `--network none`（`scripts/akashic_release/activate.py:230`），所以"联网拉 extra"必须在宿主侧、停机之前完成，产出现有的部署清单（`targets` + git bundle + sha256，`scripts/install_plugin_distribution.py:78-139`）。联网失败就是预检阶段失败，原服务继续运行，Core 不会起不来。
5. 镜像：controller 创建容器时从不拉取镜像（`plugins/host_execution/controller.py:662-677` 直接用 `request.image` 创建）。声明在运行时 `Workloads.register` 里（`plugins/computer/plugin.py:168-171,386`），静态读不到。推荐在 `plugin.py` 加一个字面量 `images`，沿用 `entrypoints` 的 AST 静态解析方式；拉取放两处：宿主侧部署预检（`docker pull`，失败则不停机），以及 controller 启动时缺镜像就拉取（兜底，覆盖所有路径）。
6. 一个发现需要提前处理：**迁移（Yoyo）只执行 `source_type == "builtin"` 的插件**（`agent/migrations/bundles.py:130`），已安装的外置插件的 `migration.catalog.toml` 不会被宿主执行。`akasha`、`scheduler` 在 extra 层且带迁移目录。这对 Phase 1 的几个样本（`privacy_mask`、`watch`）无影响，但批量迁移前必须先解决，列为独立 PR（P1-X）。

## 1. 现状摘要

### 1.1 现有安装链路

命令分发（`main.py`）：

| 命令 | 做什么 | 行号 |
|---|---|---|
| `plugin-install-trusted-batch` | 离线，持工作区锁和发布锁，读 operator 的 JSON 批次，对每项直接调 `install_git_plugin(..., refresh_existing_artifact=True)`；要求 `--confirm-trusted`，拒绝在 active turn 内调用；每项 `ref` 必须是 40 位 SHA | `main.py:113-190`；`trusted_install.py:14-19,66,84-135` |
| `plugin-install` / `plugin-status` / `plugin-uninstall` | 由 `gateway` 插件提供的入口，参数校验后走 Gateway JSON-RPC（`plugin/install`、`plugin/status`、`plugin/update`、`plugin/uninstall`），**需要运行中的实例**；连不上就提示用 `plugin-enable gateway@...` | `plugins/gateway/plugin.py:29-30`；`plugins/gateway/cli.py:204-258`；`main.py:736-756` |
| `plugin-enable` / `plugin-disable` | 离线，直接改工作区 `bundle.patch.toml`（`set_plugin_choice`）；Agent 的 active turn 内被拒绝 | `main.py:29-34,680-695`；`install.py:76-92`；`bundles.py:136-158` |
| `plugin-doctor` | 离线只读：列已安装制品、patch 选择、未结算的安装恢复点 | `main.py:697-711`；`agent/plugins/doctor.py:14-60` |

插件命令由插件静态声明的 `entrypoints` 提供，`main.py` 只分发：`invoke_plugin_command` 只认当前选择里唯一的 provider；`invoke_distribution_command` 固定用镜像里的来源，不受工作区旧选择影响（`agent/plugins/entrypoints.py:25-101`；`main.py:25,739` 的 `_DISTRIBUTION_COMMANDS` 目前是 `dashboard`、`workload-controller`）。

运行实例内的安装（`Manager._install_public`，`agent/plugins/manager.py:1424-1494`）：`install_git_plugin` 在线程里跑 → 取静态 manifest → 准备输入（`_prepare_one_with_source_diagnostics`）→ 注册公共合同，合同 digest 变化就提交选择并要求重启（`:1484`），否则 `_update_live_generation` 换代。卸载走 `_uninstall_public`（`:1380-1422`）：先写停用选择，CAS 移除，排空旧代，再删 cache；内置来源只停用不删（`:1389-1411`）。

sparse 安装的输入输出（`install.py:125-234`）：

- 输入：`source`（git URL 或路径）、`marketplace`、`ref_name`、`sparse_paths`、`refresh_existing_artifact`、`offline_wheels`、`reserved_ids`。
- 流程：`--filter=blob:none --no-checkout` 克隆 → `sparse-checkout set` → `checkout --detach <ref>`（`:237-268`）；无 sparse 时是完整 `git clone`（`:265`），会拉所有分支。克隆根必须直接是插件根。
- 输出：`PluginInstallResult(plugin_name, plugin_version, marketplace, installed_path, data_path, source_revision, update_id)`（`:41-49`）。其中 `source_revision` 是 git HEAD 的 40 位 SHA，artifact 目录名是 `<version>-<sha前16位>`（`:333`）。
- 副作用：写 `cache/<marketplace>/<name>/.artifacts/<id>`，用 stable/latest 指针对原子切换（`:402-411`），写重载日志恢复点（`journal.arm_update`），创建数据目录，最后 `set_plugin_choice(enabled=True)`（`:229`）。失败时按恢复点回退指针并删除本事务新建的 artifact 和空数据目录（`:61-73,412-421`）。
- 防接管：同 `name@marketplace` 属于内置发行版、历史归属凭证，或 marketplace 等于发行版回执里的 marketplace 时拒绝（`:191-205`）。这是"换来源"必须显式放行的第一处。

**已装记录里的身份字段**（`agent/plugins/selection.py:211-256`，选择文件 `runtime/plugin-stable.json`）：`version`(5/6)、`code`（绝对路径）、`plugin_id`、`source_revision`（64 位十六进制，是 Core 对插件目录的内容摘要，排除 `.git`、`.venv`、`node_modules` 和 `.akashic-source.json`，`input_preparation.py:35-38,190-234`，不是 git SHA）、`config_revision`、`python_environments`、`source_type`（`builtin`/`installed`）、`data_dir`、`runtime`（`python_tag`、`binding_api`）、`entrypoints`（v6），内置输入另有 `distribution_source`（发行版源 commit，40 位）。内置插件还有 artifact 内的 `.akashic-source.json`（`{commit, path}`，`scripts/build_plugin_distribution.py:327-333`）。

所以"代码 SHA"在已装 artifact 上有两个：git HEAD（`distribution_sources.py:193-202` 的 `_head_revision`）和内容摘要 `source_revision`。**git HEAD 才是和远端对比的东西**。

**marketplace 现在怎么定**：内置发行版固定是 `release`（`build_plugin_distribution.py:467`，`bundles/base.toml` 里全是 `x@release`）；外置插件由安装者在 `plugin-install --marketplace` 指定，CLI 默认 `local`（`plugins/gateway/cli.py:207`），生产上外置插件用 `github`（`docs/design/hua-home-plugin-runtime-source-of-truth.md` 3.2 节；该文是 2026-09 的历史快照，现状未核对，推断）。marketplace 在已安装记录里只用作 cache 目录名、plugin_id 后缀和（#2165 前）数据目录名。

### 1.2 外置插件带前端时怎么交付

- 外置插件（fleet 里的 fitbit、calendar 等，`/mnt/data/coding/akashic-fleet`，只读核对）：前端是**手写 JS 直接提交进插件仓库**，没有构建步骤。例如 fitbit 仓库里 `git ls-files` 含 `web_module.js`、`mobile_panel.js/css`，`plugin.py` 直接引用 `web="web_module.js"`；`package.json` 只有 `node --test`。fleet 记录的是 2026-09 的状态，最新 main 未逐个核对（推断）。
- Core 内置插件：前端源码在 `frontend/plugins/<name>/src`，由 `scripts/build-web-plugins.mjs` 用 vite/esbuild 产出到 `plugins/<name>/web_module.{js,css}`、`plugin_ui.js`、`message_ui.js`、`static/`，这些文件被 `.gitignore` 忽略（`.gitignore:117-127`），只在发行版构建时生成并复制进每个插件的制品（`build_plugin_distribution.py:53-60,317-322`）。构建脚本里点名了具体插件（`build-web-plugins.mjs` 的 `chatSlotModule(...)` 一串调用），与"Core 不点名插件"相违，迁出时每个插件需要自己的构建入口。
- 共享库 `react`、`react-dom/client`、`@akashic/web-ui-v1`、`@akashic/chat-ui-v1` 是 external，由宿主页面的 import map 提供（`build-web-plugins.mjs:82`、`:102-104` 的 `chatSlotModule`），所以 extra 的构建需要能引用同版本的 `packages/*` 声明（Phase 2 已列）。
- Core 容器：`Dockerfile.distribution:40-41` 装了 `git` 和 `nodejs`，**不是"没有 Node"**，但没有 npm、vite 等构建工具链（没在软件包清单里，`nodejs` 包不含 npm，推断），且发行版构建要求 `/opt/akashic/source/plugins` 不存在（`:95`）。结论不变：**前端产物必须在 CI 构建好，放进 release 分支**，容器里只做 git 克隆，不做构建。

### 1.3 局部换代：不重启 Core 能不能生效

- **装、升**：可以，换代发生在同一个 live Root 内（`manager.py:1763-1830`）。顺序是：新代先导入（旧代仍在服务，`:1803-1805`），导入失败不提交选择、不退役旧代；再 CAS 提交选择；再排空旧代。
- **卸**：可以，`_uninstall_public` 排空后删 cache。
- **必须重启的唯一情形**：插件的 `contract.py` 内容变化。公共合同类型进程内只加载一次（`agent/plugins/public_contracts.py:114-150`，digest 变化返回 `changed`），此时安装仍然提交选择，但把本次更新标成"合同需重启"（`manager.py:1477-1490` → `reload_journal.py:278-287`，状态 `restart_required`）。新装一个带 `contract.py` 的插件（`previous is None`）不算变化。
- `plugin_update` 插件是给 Agent 用的 `install_plugin` 工具（`plugins/plugin_update/tool.py:27-33` 的 `InstallInput`：source/marketplace/ref/sparse），不参与局部换代逻辑。
- 离线时（实例没运行）装、升只是写好 cache 指针和 patch 选择，启动时由启动装配把"已提交安装"接到选择里（`reload_journal.py:329` 的 `bind_boot_input`，推断这条路径与 trusted-batch 的使用方式一致，未实机验证）。

### 1.4 2026-11-21 事故：镜像和 digest

- 镜像 digest 现在声明在插件运行时代码里：`plugins/computer/plugin.py:168-171`（`_IMAGE = "ghcr.io/kachofugetsu09/akashic-computer@sha256:..."`），在 `apply` 里通过 `ctx.require(WORKLOADS).register(ctx, Workload(image=_IMAGE, ...))` 注册（`:386`）。校验只要求 `@sha256:<64hex>`（`plugins/workloads/definitions.py:14,47`；`controller.py:871-872`）。
- 谁拉镜像：没人。controller 通过 Docker Engine API 创建容器（`controller.py:662-677`），镜像不在本机就失败，失败后还可能留下 `cleanup_failed` 事务（`controller.py:462-466,536`）。部署只处理 Core 镜像（`activate.py`、`scripts/akashic_release/image.py`），不碰 workload 镜像。
- 静态读不到：`static_manifest.py` 只 AST 解析 `name/version/api_version`（`:315-344`）、`entrypoints`（`:136-170`）、`ingress_guard`（`:116`），不认 `Workload`。

### 1.5 换来源要动的持久状态

假设 `watch@release`（内置）换成 extra 来源。逐项：

| 状态 | 位置 | 现状 | 换来源时 |
|---|---|---|---|
| 选择记录 | `runtime/plugin-stable.json`，`_input` | 内置输入 `source_type=builtin`、`code` 指向镜像内目录、带 `distribution_source`（`selection.py:211-256`） | **必须改**：换成 `installed` 输入（代码指向新 artifact，去掉 `distribution_source`，`data_dir` 不变）。`PluginSelection.commit` 是 CAS 原子写 |
| 同名唯一 | `selection.commit`（#2165） | #2165 拒绝同逻辑名多来源、共用数据目录 | 同一 commit 里旧输入被新输入替换，天然满足 |
| 防接管守卫 | `install.py:191-205`、`install_plugin_distribution.py:1029`、`distribution_sources.py:296-303`（"内置与外置共用数据身份，停止迁移"） | 都在拒绝"外置占用内置身份" | **必须显式放行**，只对本次切换的 plugin_id 放行 |
| 历史归属 | `distribution_adoption`，`distribution_sources.py:123-189` | 若该 id 在 adoption 条目里，要求当前选择仍是发行版输入，否则抛 `历史归属已被其它选择占用`（`:177-186`） | **必须在同一次 CAS 里删掉该条目**；#2165 给 `commit` 加了 `replace_adoption`，但清空（`entries` 不能为空，`:93-95`）需要再加一个"清除"语义 |
| bundle preset | `bundles/*.toml`（`bundles.py:70-80`） | preset 行 `watch = {plugin = "watch@release"}` | 身份不变则不用改。preset 行**不删**（见 2.9），所以"默认组合"仍成立 |
| 用户 patch | `workspace/bundle.patch.toml`，`set_plugin_choice`（`bundles.py:136-158`） | 与 preset 相同的行会被删掉（"跟随 preset"）；非内置身份没有 preset 行时，安装会写一条显式启用行 | 身份不变：不用动。身份变：要靠 `agent/migrations/successors.py` 的 `rename_row`（`:76-93`），它先留恢复点再写，**不与选择文件同一个原子提交** |
| ledger owner | `owner_records.owner = "plugin:<plugin_id>[:scope]"`（`plugins/ledger/services.py:237-244`） | 已有改名迁移示例 `20261019_01_working_directory_owner.py:14-47`（前缀匹配、冲突即失败回滚） | 身份不变：不用动。身份变：每个插件一份迁移，必须在新身份首次启动前完成 |
| Message metadata 命名空间 | `message_metadata_keys=frozenset({owner})`，owner 是完整 plugin_id（`services.py:141,171`；`log.py:2108`） | 历史消息 metadata 里的 key 是旧 plugin_id | 身份变就要改写 messages 表里的 metadata JSON（没有现成迁移，推断），**这是否决改身份的主要理由** |
| 数据目录 | `plugin-data/<name>-<marketplace>` 或 `<name>`（#2165） | 选择记录的 `data_dir` 为准 | 换来源沿用同一个 `data_dir`，不搬目录 |
| 插件配置 | 数据目录内 `config.local.toml`（`load_config(data_dir)`，`manager.py` 装配处，推断） | 随数据目录 | 不动 |
| 技能投影 | `INSTALLED_ASSETS.register(...)` 在每个 generation 的 `apply` 里注册，以 `ctx.runtime.plugin_id` 为 owner（`plugins/assets/plugin.py:51`） | 运行时重建，推断不落盘 | 不动（未读完 `assets/plugin.py` 全文，推断） |
| 迁移账本 | `migrations.sqlite3`，Yoyo 按 migration ID 去重（`agent/migrations/runner.py:73`） | 只执行 builtin 来源的 catalog（`bundles.py:130`） | 带迁移目录的插件换成 installed 后**不会再执行新迁移**；P1-X 解决 |
| 重载日志 | `runtime/plugin-reloads.sqlite3` | 安装有恢复点和 update_id | 切换是一次安装操作，沿用同一套恢复点 |

**同一次提交内必须一起改**：选择记录、历史归属条目。**可以事后补的**：无（按推荐方案，其余状态都不需要变）。如果走改身份的方案，ledger owner 和 metadata 必须在新身份启动前完成，也就是进了同一个停机窗口，不是"事后"。

失败时各状态：新 artifact 的准备（克隆、依赖环境、镜像）全部在 CAS 之前，失败只留下一个未被选择引用的 artifact，现有机制已经容忍（`install_plugin_distribution.py:786-792` 的 `owns_installed_artifact` 分支）。CAS 本身是单文件 `os.replace`，成功即完成；`SelectionWriteError(outcome="uncertain")` 时重读 `root_ref` 判定（`selection.py:122-135,198-201`）。

### 1.6 内置插件从 Core 删除后，老实例升级会怎样

读 `scripts/install_plugin_distribution.py`：

- `_distribution_candidate`：选择里归属发行版、但新发行版里没有的插件，直接 `continue`，注释就写着"没有来源就是退役，永不删数据"（`:765-776`）。它**会静默丢出组合**。
- `_require_preset_rows`（`:813-824`）：preset 里启用的行必须进入候选组合，否则抛 `preset 启用的插件没有进入选择，停止发布`。所以只要 preset 行还在，缺失会在预检阶段拦住发布，旧服务继续运行；反过来，如果迁移 PR 同时把 preset 行删了，就会静默少插件。
- 预检和发布都在 `docker run --network none` 里（`activate.py:230`），所以 extra 的拉取不能发生在容器里。
- 已有的外置输入机制（部署清单 `targets` + bundle + `target_commit` + 离线 wheels，`:78-139,150-235`）要求目标"已选择且是已安装输入"（`:163-167`），拒绝接管内置身份（`:1029`）。**这套机制不能直接拿来做首次装和换来源**，要扩展，见 2.9。

结论：谁来把 `x@release` 换成 extra 来源——只能是**发布流程本身**（宿主侧联网产出清单，容器内离线应用），不能是 Core 起来后的补装。

### 1.7 部署链路

- `scripts/akashic_release/cli.py` 的 `install`（`:44-100`）：解析目标 commit → 固定产物 `prepare_generation`（构建 Core 镜像、Bridge 环境）→ `activate_release`。`--plan` 可传部署清单，`--inputs` 传清单引用的 bundle 目录。
- `activate_release`（`activate.py:377-506`）：在线预检（`_publish(preflight=True)`，容器内 `--network none`，不写状态）→ 失败则服务不动；写 attempt → 自部署时等当前 turn 结束并关闭准入 → 停机 → `_publish` 真发布（迁移、安装目标、选择 CAS）→ 启动并验收。失败时若 `can_roll_back`（有旧版本且预检显示无待执行迁移，`:426`）则 `_recover_previous`：停机、把 `plugin-stable.json` 恢复为停机时的快照原文、换回旧 Core 单元并启动（`:509-563`）。自部署有待执行迁移时直接拒绝（`:424-425`）。
- 生产的 `~/akashic-deploy.sh` 本机不存在（`ls` 确认），其内容没读到，只据记忆和项目文档推断它调用 `cli.py install`；Agent 自部署是 `submit`/`self-deploy-worker`（`cli.py:196-205`，`self_deploy.py`）。
- 失败回退时 extra 插件的状态：artifact 是不可变目录，旧选择里的 `code` 路径指向旧 artifact；只要恢复选择快照，旧代码仍在。cache 里的 stable/latest 指针已被新安装推进，但运行以选择文件为准（`entrypoints.py` 与 `_selected_generations` 都读选择里的 `code`，`manager.py:2234`）。推断这样回退是一致的，需要在隔离实例上实测（见 PR-C2）。

### 1.8 兼容性声明

- `plugin.py` 里必须有字面量 `api_version = 3`，静态解析只接受 3（`static_manifest.py:228-232,342-343`），其他值就是 `PluginSourceContentError`。
- 选择输入里还有 `runtime.binding_api`，常量 `PLUGIN_INPUT_API = 3`（`input_preparation.py:33,154`），启动装配发现不等就拒绝（`manager.py:2230`），发布预检也拒绝（`install_plugin_distribution.py:793`）。
- 公共 import 面是 `scripts/plugin_boundary.py:66` 的 `PLUGIN_ALLOWED_MODULES`，门禁 R14 管层级。

Core 升级后已装 extra 不兼容：静态 manifest 读取失败 → `scan_plugin_sources` 返回 failure → 预检抛 `installed source is unavailable`（`install_plugin_distribution.py:759-760`）；binding_api 不等 → `:793` 抛错。两者都在停机之前，旧服务继续运行。

### 1.9 CLI 放哪、在线还是离线

- 在线：`plugin-install/-status/-uninstall` 走 RPC，需要运行中实例。
- 离线：`plugin-install-trusted-batch`、`plugin-enable/-disable`、`plugin-doctor` 直接读写文件。
- `main.py` 的 `_run_lightweight_command`（`:105-190`）是为离线命令预留的"不加载 Agent runtime"入口。

## 2. 设计

### 2.1 总体

```
extra 仓库 CI ──构建──▶ refs/heads/release/<name>   （每个插件一个分支，内容 = 构建好的完整插件根）
                         │  提交由 tree 唯一决定，tree 不变则不推
                         ▼
Core: pkg 命令 ──git ls-remote──▶ 最新 SHA 表
         │         对比已装 artifact 的 git HEAD
         ▼
   install_git_plugin(source, marketplace, ref_name=<sha>)    ← 现有安装器
         │   在线：Gateway RPC plugin/install（局部换代）
         │   离线：持锁直接调用（trusted-batch 同款）
         ▼
   换来源：同一 plugin_id，选择记录原子替换（2.8）
```

不建 index.json、不建 tarball、不用 GitHub Release，不新增持久状态文件。

### 2.2 extra 仓库约定（Phase 2 的 CI 实现，这里只定契约）

- 分支名：`release/<插件名>`，插件名即 `plugin.py` 的 `name`。
- 分支内容：就是插件根（`plugin.py` 在根），含编译后的 `web_module.js/css`、`plugin_ui.js`、`message_ui.js`、`static/`，以及 `requirements.txt`（如有）。可复用 `scripts/build_plugin_distribution.py` 的 `_bundle_plugin`（`:296-362`）和 `_build_web_assets`（`:154-215`）：它已经把每个插件构建成"单提交 git 仓库"。
- **tree 不能含来源提交号**。`_bundle_plugin` 现在把 `.akashic-source.json`（含源 commit）写进插件根（`:327-333`），会让每个源提交都产生不同的 tree。release 分支要把来源 commit 写进提交说明，不写进 tree。
- 提交由 tree 唯一决定：`git commit-tree <tree> -m "release <name> <tree>"`，无父提交，固定作者和时间。CI 推送前先比较新 tree 与远端分支 tree，相同就跳过。这样提交 SHA 就是版本。
- 没有新字段。兼容性、依赖、摘要都在插件自己的 `plugin.py`（`api_version`、`desc`、`images`，见 2.6）。

> 与你的方向相比：你想用 `plugins/<name>/` 的子树哈希。子树哈希的好处是单分支，坏处是安装器要新增"子目录即插件根"，artifact 身份也要从 HEAD 改成子树哈希，列表还要先做一次 blobless fetch 才能读到子树（`ls-remote` 只给提交）。每插件一分支用的是已有的所有部件，`ls-remote` 一次拿全。分支数量（约 35 个）对 GitHub 没有压力。如果你坚持单分支，代价和改动点见开放问题 Q2。

### 2.3 数据模型

**源配置**：不建配置文件。

- 默认 extra 仓库是常量 `https://github.com/kachofugetsu09/akashic-plugins.git`；`AKASHIC_EXTRA_REPO` 环境变量或 `--repo` 参数覆盖（隔离实例验收用 `file://` 本地仓库，私人插件也用 `--repo`）。
- 任意 git 源安装沿用现有 `plugin-install --source ... --marketplace ...`，不属于 pkg 管理的 extra 集合。

**已装记录新增字段：Phase 1 不新增。** 理由：

- 选择文件的输入结构是严格的字段集合（`selection.py:213-221`），加字段要升版本；
- 来源信息已经在 artifact 里：它是 clone 出来的 detached checkout，`git config remote.origin.url` 是仓库地址，HEAD 是 SHA，分支名按约定是 `release/<plugin name>`；
- 多存一份会违反"一个插件只有一个来源"。

`pkg list` 的"来源"列由此推导：`core`（`source_type=builtin`）、`extra`（`origin.url` 等于配置的 extra 仓库）、`git:<url>`（其他）。唯一的风险是有人手改 `.git/config`，这等同于手改代码，不在防范范围。

**marketplace**：按 2.3 与 3.2 的推荐，迁出的插件继续用 `release`；仓库里本来就只在 extra 的新插件用 `extra`（Q1）。两者在 plugin_id 里只是身份命名空间，不代表来源。

### 2.4 命令

命令组 `akashic pkg ...`，作为新增核心层插件 `packages`（在 `plugin_boundary.toml` 的 `[tiers].core` 登记）提供的 `entrypoints = {"pkg": "cli.pkg_main"}`，并加进 `main.py:25` 的 `_DISTRIBUTION_COMMANDS`。原因：包管理器必须在选择为空、损坏或旧选择不含它的时候也能工作（新实例首次 `sync`），`invoke_distribution_command` 固定用镜像里的来源，不依赖工作区选择（`entrypoints.py:81-101`）。现有 `plugin-install` 等命令原样保留，Phase 6 再决定删除。

在线/离线：`pkg` 先探测 Gateway 可连：

- 可连：对每个包调用已有的 `plugin/install`（`ref` 填 SHA）或 `plugin/uninstall`，局部换代，不重启 Core。换来源用一个新的 RPC 参数（见 2.8）。
- 不可连：持工作区锁和发布锁，直接调用 `install_git_plugin`（即 trusted-batch 的路径，`main.py:160-176`），下次启动生效。

| 命令 | 步骤 |
|---|---|
| `pkg list [--remote]` | 读选择和 cache 列出已装包（名称、版本、来源、artifact HEAD）；`--remote` 再 `ls-remote` 标出 `current`/`upgradable`/`removed-upstream`（远端没有该分支）/`incompatible`。只读 |
| `pkg sync` | 求"应装集合"：`composition_rows` 里启用的行，减去镜像发行版自带的，减去已装的（`composition_rows`/`distribution_plugin_sources`，`bundles.py:119-127`、`distribution_sources.py:259-277`）。对缺的逐个 `install`。用户停用过的行 `disabled=true`，天然不会重装（`bundles.py:136-158` 的语义） |
| `pkg install <name>...` | 1. `ls-remote` 拿到 `release/<name>` 的 SHA；2. 克隆该 SHA 并读静态 manifest，查 `api_version`、`images`；3. 预拉镜像（2.7）；4. `install_git_plugin(ref_name=sha)`；5. 在线则换代。多个名字逐个做，任何一个失败不影响已完成的（每个包一个恢复点） |
| `pkg upgrade [<name>...]` | `ls-remote` 全表，已装 extra 包里 SHA 变了的才重装，其余不动。**每个包单独换代、单独回退**，变化的才拉镜像 |
| `pkg remove <name>` | 即现有卸载：停用选择 → CAS 移除 → 排空 → 删 cache，数据目录保留（`manager.py:1380-1422`）。用户 patch 里留下 `disabled=true`，`sync` 不会再装回来 |
| `pkg switch <name> --to extra` | 换来源，见 2.8；第一版只支持 `core → extra`（给 Phase 2/3 用），反向和 `git` 源互换不做 |

### 2.5 install_git_plugin 的两处小改

1. **按 SHA 浅取**：`ref_name` 是 40 位 SHA 时，改用 `git init` + `git fetch --depth 1 origin <sha>` + `checkout`，不再克隆全部分支。已在 GitHub 公开仓库上实测可行：`git fetch --depth 1 origin b659557b...` 成功（6.8 秒，对象库 18MB，因为那是整个源码仓库；插件分支会小得多）。非 GitHub 服务器要求 `uploadpack.allowReachableSHA1InWant`，Phase 1 只面向 GitHub 和 `file://`，不满足时回退到现有克隆路径（推断，`file://` 未实测）。
2. **显式接管**：加参数 `takeover: frozenset[str]`，只对列出的 plugin_id 跳过 `install.py:191-205` 的防接管检查。`pkg switch` 用，其他入口不传。

### 2.6 兼容声明（Q5）

最简做法：**沿用 `api_version`，不加字段**。`api_version = 3` 是公共 import 面的版本号，只在 `PLUGIN_ALLOWED_MODULES` 发生不兼容变化时才 +1（同时 `PLUGIN_INPUT_API` +1）。extra 的 CI 固定到某个 Core 版本跑同一套门禁（迁移计划已写）。`pkg install/upgrade` 在安装前读静态 manifest，`api_version` 不被当前 Core 支持就拒绝该包并提示升级 Core，已装的保持不变。

已知缺口：release 分支总是对应 Core main，旧 Core 在 `api_version` 没变的情况下可能拿到依赖新接口的插件（符号级不兼容，不会被 `api_version` 发现）。第一版接受这个风险；缓解靠"删除 `PLUGIN_ALLOWED_MODULES` 里的符号必须 +1 `api_version`"这条规则和 CI 门禁。如果将来需要更细粒度，再加 `requires_core` 字面量，那是后话。

Core 升级后发现已装 extra 不兼容：见 2.9，预检阶段拦住，旧服务继续运行。

### 2.7 镜像预拉

**静态声明**：`plugin.py` 增加字面量

```python
images = ("ghcr.io/kachofugetsu09/akashic-computer@sha256:...",)
```

`computer` 里的 `_IMAGE` 改为引用 `images[0]`。`static_manifest.py` 在 `_plugin_entrypoints`（`:136`）旁加一个同样用 AST 只读字面量的 `_plugin_images`，结果放进 `StaticPluginManifest.images`。**不放进 `identity_digest`**（`:80-103`），避免改变现有记录里的 `manifest_digest`（历史归属凭证依赖它，`distribution_sources.py:172`）。仍然保持 `Workloads.register` 里的 `image` 校验不变，运行时声明与静态声明不一致时 `apply` 报错（防止两处写不同的值）。

**拉取位置（两层）**：

1. **兜底：controller 启动时缺镜像就拉**。`controller._start` 创建前先 `GET /images/<ref>/json`，404 就 `POST /images/create?fromImage=<repo>@<digest>`，且只接受 digest 引用（已由 `_IMAGE` 校验保证）。覆盖所有路径：人工安装、Core 重启、镜像被 GC。拉取耗时算进启动超时，这是这一层的代价。
2. **预检：部署侧宿主 `docker pull`**。`akashic-release install` 在生成部署清单后、停机之前，对清单里每个包的 `images` 逐个 `docker pull`（宿主有 Docker，`activate.py` 本来就在宿主跑 `docker run`）。失败 = 预检失败 = 不停机，旧服务继续运行，这正是 11-21 事故需要的位置。

在线 `pkg install`（Core 容器里没有 docker）不单独预拉，依赖第一层。是否要在在线路径加一个 `pull` RPC 做显式预拉，见 Q4。

### 2.8 换来源的原子操作（`pkg switch`）

目标：同一个 `plugin_id`（如 `watch@release`），来源由 `builtin` 变 `installed`（extra），在一次选择 CAS 内完成，沿用同一数据目录。

1. **准备（可失败，不改任何选择）**：解析 `release/<name>` 的 SHA；浅取、读静态 manifest、校验 `name` 与目标 plugin_id 一致、`api_version`；预拉镜像；`install_git_plugin(..., takeover={plugin_id})` 建 artifact、依赖环境、数据目录（已存在则复用，`data_dir` 取当前选择记录，#2165 的 `recorded_data_dir`）。
2. **组装新输入**：`prepare_plugin_input(...)`（`install_plugin_distribution.py:1091` 同款调用）得到 `source_type=installed` 的输入，`data_dir` 与旧输入相同。
3. **一次 CAS**：用 `selection.commit(components', expected_ref, distribution_adoption=adoption', replace_adoption=True)`：`components'` 里新输入替换旧输入；`adoption'` 是去掉该 plugin_id 条目的凭证。条目清空时需要 `commit` 支持把 `distribution_adoption` 写成 `None`（#2165 的 `replace_adoption` 只支持非空替换，需要补一个清除语义，约 10 行）。
4. **在线时**：走 `_update_live_generation`：新代先导入再提交，导入失败不提交，旧内置代继续服务；提交后排空旧代。**离线时**：只做 1-3，下次启动按新选择装配。
5. **回退**：CAS 之前任何失败都无需回退，只留下无人引用的 artifact（现有恢复点负责回退指针）。CAS 之后的失败（如启动后插件起不来）由现有的 `degraded_external` 与重载日志处理；要回到内置来源，就是对同一 plugin_id 反向切换——第一版不提供命令（不做 `rollback`），手工回退是恢复选择快照。

不需要改的：ledger、Message metadata、bundle preset、用户 patch、数据目录、配置、迁移账本（全因身份不变，见 1.5 表）。

若采用改身份方案（`@release` → `@extra`），这一节要多出：ledger owner 迁移、metadata 命名空间迁移、patch 行重命名（`successors.rename_row`，不与选择同一次原子提交），以及三者的停机窗口顺序。这是不推荐它的原因。

### 2.9 内置 → extra 的 Core 升级衔接

规则（不加 schema，靠推导）：**"需要从 extra 装的插件" = preset 里启用的行 − 新发行版自带的 − 已装的**。因此一个插件迁出 Core 时，只从 `plugins/` 删除源码，`bundles/*.toml` 里的行保留——"默认组合不变"由 preset 行承载，缺失由现有的 `_require_preset_rows`（`:813-824`）在预检阶段拦住。

发布流程（扩展现有部署清单，`schema_version` 保持 1，`targets` 增加两种条目）：

1. **宿主侧联网（停机前）**：`akashic-release install` 增加 `--extra`（默认开启，可 `--no-extra`）。读目标 commit 里的 `bundles/` 和旧选择，求出 (a) 需要新装的 extra 包（旧选择里是内置、新发行版没有了；或新增的 preset 行），(b) 已装 extra 包里 SHA 有变化的。逐个 `git clone`/`fetch` 成 git bundle 放进 `--inputs` 目录，算 sha256，写入清单。对声明 `images` 的包执行 `docker pull`。**这一步失败即整个 `install` 失败，服务未动**。
2. **预检（容器内，离线）**：新条目类型 `kind = "install"`（首装）和 `kind = "switch"`（换来源），与现有"升级已装目标"条目并存；`_stage_deployment_targets` 的"目标必须已选择"检查（`:163-167`）和接管守卫（`:1029`）按 `kind` 放行。预检同时检查 `api_version`，不兼容就失败，服务未动。
3. **停机发布**：与现有流程相同，Core 迁移 → 安装目标 → 一次选择 CAS。`switch` 条目同时处理 adoption。
4. **失败回退**：沿用 `_recover_previous`，恢复选择快照，旧 artifact 仍在，extra 新装的包只是未被引用的 artifact。回退到旧 Core 版本时旧 Core 的 preset 里仍有这些插件的行，所以旧版本不需要认识 extra。

联网失败的行为：宿主侧第 1 步失败 → 部署失败，旧服务继续。首次离线环境（新机器无网）：允许 `--extra-inputs <dir>` 指向预先下载好的 bundle 目录，也就是现有 `--inputs` 的用法。

**自部署（Agent 部署自己）**：同一个 `install`，宿主侧联网步骤在 worker 里运行，发生在"等当前 turn 结束并关闭准入"之前（`activate.py:451` 的 `wait_for_stop` 之前），联网失败不会打断对话。自部署遇到待执行迁移就拒绝（`activate.py:424-425`）这条规则不变。

### 2.10 失败与回退汇总

| 阶段 | 失败 | 状态 |
|---|---|---|
| `ls-remote` / 浅取 / 镜像预拉 / 兼容检查 | 任何一个 | 什么都没变；部署则整个不开始 |
| 依赖环境准备 | pip 失败 | 同上，`install_git_plugin` 的 staging 回滚 |
| `install_git_plugin` 已提交指针，换代失败（在线） | 导入失败 | 选择未提交，旧代继续服务，artifact 留在 cache（现有语义，`install_plugin_distribution.py:786-792` 容忍） |
| 选择 CAS | 写失败 | `SelectionWriteError`，按 `root_ref` 判定 unchanged/uncertain（`selection.py:122-135,198-201`） |
| CAS 之后，激活失败 | 插件起不来 | 在线：现有 degraded 处理；部署：`_recover_previous` 恢复选择快照 |
| 合同变化 | `contract.py` 变了 | 选择已提交，更新标 `restart_required`，由操作者安排重启（现有语义） |

## 3. 设计决定的理由

3.1 **为什么每插件一分支而不是单分支**：见 2.2 的比较。

3.2 **为什么保留 `@release` 身份**：见 0 的第 3 点和 1.5 的表。marketplace 在运行时只是命名空间，改它的代价由 ledger 和 metadata 承担，收益只是名字好看。#2165 已经让数据目录脱离 marketplace，说明方向本来就是"身份不变、来源可变"。这会让 plugin_id 里出现 `release` 却不再来自 Core 发行版，是名字上的包袱，接受它（Q1）。

3.3 **为什么 `pkg` 是核心层插件而不是 `main.py` 里的新分支**：命令由插件静态声明，`main.py` 只分发，这是现有约定（`entrypoints.py` 的文档字符串）；`packages` 不依赖任何 extra 插件，满足 R14。

3.4 **为什么部署要走清单**：发布容器离线是现有安全设计（`activate.py:230`），Core 迁移、安装、选择 CAS 在同一个停机窗口里，用已有的恢复机制；另起一条"Core 起来后再补装"的路径就会产生 Core 已升级而 extra 缺失的窗口，违背"默认组合不变"。

## 4. 明确不做的事

- `try`、`rollback`、`export`、`apply`、`dev`、来源优先级覆盖、一个插件多来源（迁移计划已定）。
- 包组、摘要元数据文件、依赖自动解析、`meta` 分支。Phase 1 只需要名字和 SHA，`list` 的摘要来自已装插件自己的 `desc`；包组是 Phase 5，到时再决定元数据放哪。
- 不建 index.json、tarball、GitHub Release；不在 Core 里存 lock；不加选择记录字段。
- 不改 marketplace 命名、不迁移 ledger、不做数据目录归一化（#2165 负责）。
- 不让安装在 Core 容器里构建前端或安装 npm 依赖。
- 不在 Phase 1 解决 extra 插件的 Yoyo 迁移（独立 PR P1-X，Phase 3 之前必须完成）。
- 不实现 `core ← extra` 反向切换命令和 `git` 源互换。
- 不新增单元测试（TST-001）。既有测试如 `tests/test_plugin_install.py` 随行为变化改写；验收一律用真实 bundle 和隔离实例。

## 5. PR 拆分

约束：每个 PR 新增不超过 500 行，能单独合并和验证。验收用隔离实例：

```bash
S=~/.cache/akashic-pkg/state-<pr>; C=~/.cache/akashic-pkg/cache-<pr>; PORT=<空闲端口>
python scripts/start.py --state "$S" --cache "$C" --port "$PORT" --no-browser
# 工作区 $S/workspace，插件主目录 $S/plugin-home，配置 $S/config.toml（scripts/start.py:144-146,195-217）
export AKASHIC_PLUGIN_HOME="$S/plugin-home" AKASHIC_EXTRA_REPO="file://$HOME/.cache/akashic-pkg/extra.git"
python main.py pkg list --remote --config "$S/config.toml" --workspace "$S/workspace"
```

数据放 `~/.cache`，不放 `/tmp`。`start.py` 默认用源码树 `plugins/` 当内置来源、没有发行版回执，所以它能验"在线装/升/卸/换代"；涉及发行版归属、`distribution_adoption`、停机发布的 PR（C2、B4 的 adoption 部分）要用本地构建的发行版目录（`scripts/build_plugin_distribution.py` 输出 + `scripts/install_plugin_distribution.py`）或 `scripts/standalone_compose_smoke.py`（推断可用，未实测）。

### 并行与依赖

```
A1 ──▶ A2                      镜像线（A2 也可独立合并，先修事故类问题）
C1 ───────────────┐           夹具仓库（尽早，所有验收都用）
B1 ──┐            │
B2 ──┴─▶ B3 ──▶ B4 ──▶ C2b
              C1,B2 ──▶ C2a ──▶ C2b
#2165 合入 ──▶ B4
P1-X 独立（Phase 3 之前）
```

可并行起步：A1、B1、B2、C1、P1-X。

### PR 清单

**C1 · 夹具仓库构建器（约 200 行）**
- 改：新增 `scripts/pkg_fixture_repo.py`。
- 做：用 `build_plugin_distribution._bundle_plugin` 把指定的 `plugins/<name>`（含前端产物）构建成 `release/<name>` 分支写入一个本地裸仓库，tree 不含 `.akashic-source.json`，提交由 tree 唯一决定。这同时是 Phase 2 extra CI 的雏形。
- 验收：对 `privacy_mask`、`watch` 各生成两次，第二次分支 SHA 不变；改一个文件后只有该分支 SHA 变；`git ls-remote file://...` 列出 `release/*`；`git clone` 某分支后根下有 `plugin.py` 和 `web_module.js`。

**A1 · 静态 `images` 声明（约 60 行）**
- 改：`agent/plugins/static_manifest.py`（`_plugin_images` + `StaticPluginManifest.images`）、`plugins/computer/plugin.py`（`images` 字面量，`_IMAGE` 引用它，运行时与静态不一致时报错）。
- 验收：隔离实例启动，能力快照（`scripts/capability_snapshot.py`）前后对比 `computer` 无差异；对 `images` 写成非字面量的临时插件，静态解析报错且不执行插件代码。

**A2 · controller 缺镜像时拉取（约 150 行）**
- 改：`plugins/host_execution/controller.py`（`_start` 前 ensure image）。
- 验收：在隔离的宿主桥接环境删掉 workload 镜像（`docker rmi`，只删测试用的 digest，勿碰生产），启动 `computer`，观察自动拉取、健康通过；断网时失败，不留 `cleanup_failed` 事务（核对 `_start_left_nothing` 路径）；已有镜像时零额外请求。

**B1 · 安装器两处小改（约 120 行）**
- 改：`agent/plugins/install.py`（按 SHA 浅取、`takeover` 参数）。
- 验收：对 C1 的夹具仓库 `plugin-install --source file://... --ref <sha> --marketplace release`，artifact 对象库只含单提交；对非 SHA 的 ref 行为不变；`takeover` 未传时防接管守卫照常拒绝。

**B2 · `packages` 核心插件：只读命令（约 300 行）**
- 改：新增 `plugins/packages/`（`plugin.py` 声明 `entrypoints`、`remote.py` 解析 `ls-remote`、`state.py` 读已装 artifact 的 HEAD 与 origin）、`main.py`（加入 `_DISTRIBUTION_COMMANDS`）、`plugin_boundary.toml`（tiers.core）。
- 做：`pkg list [--remote]`、`pkg sync --dry-run`。
- 验收：在 start.py 实例上，对夹具仓库 `list --remote` 正确显示 `current`/`upgradable`/`removed-upstream`；用 `plugin-install` 手装一个夹具包后 `list` 显示来源 `extra`；选择为空的新实例上也能运行。

**B3 · 装、升、卸、同步（约 350 行）**
- 改：`plugins/packages/`（`install.py`、`upgrade.py`）。
- 做：在线走 RPC、离线走持锁直装；`api_version` 检查；每包独立恢复点；`upgrade` 只处理 SHA 变化的包。
- 验收：对夹具仓库 `pkg install privacy_mask` 后，不重启实例，能力快照出现它的工具和 UI 槽位（真实发消息验证）；改夹具里 `watch` 一个文件重新生成，`pkg upgrade` 只换 `watch` 一个（日志里其他包 generation 不变）；`pkg remove watch` 后数据目录仍在；人为构造 `api_version=4` 的包，`install` 拒绝且已装不变；实例停止时离线 `pkg install`，启动后生效。

**B4 · 换来源（约 350 行）**
- 前置：#2165 合入。
- 改：`agent/plugins/selection.py`（`commit` 支持清除 adoption）、`agent/plugins/manager.py` 与 `plugins/gateway`（`plugin/install` 增加 `takeover` 参数，在线换代）、`plugins/packages/switch.py`。
- 验收：start.py 实例上把 `watch`（内置）切到夹具来源：选择里该条变 `installed`，`data_dir` 不变，旧数据可读，面板和技能在，ledger owner 记录（`owner_records` 里 `plugin:watch@release`）计数前后不变，不重启；故意让新版本导入失败，选择不变、旧内置代继续服务；发行版实例上验证 adoption 条目被同一次提交移除，再启动不报 `历史归属已被其它选择占用`；#2165 的归一化命令之后再切一次仍然成功。

**C2a · 宿主侧联网产出清单（约 300 行）**
- 前置：B2、C1。
- 改：`scripts/akashic_release/cli.py`（`--extra`、`--extra-inputs`）、新增 `scripts/akashic_release/extra.py`（求需要装/升的包、取 git bundle、算 sha256、`docker pull` 声明的镜像）。
- 验收：对本地发行版目录跑 `install --no-activate --extra`，生成的清单与 bundle 通过 `load_deployment_plan`（`install_plugin_distribution.py:78`）；断开夹具仓库，命令失败且没有任何状态变化；`images` 里放一个不存在的 digest，失败在停机之前。

**C2b · 发布消费新条目 + 回退（约 400 行）**
- 前置：B4、C2a。
- 改：`scripts/install_plugin_distribution.py`（`load_deployment_plan` 增 `kind`；`_stage_deployment_targets`、`publish_distribution` 按 `kind` 放行；预检检查 `api_version`）、`agent/plugins/distribution_sources.py`（`distribution_migration_sources` 对 `switch` 目标放行 `:296-303`）。
- 验收：用本地发行版和旧实例状态做完整一次升级——旧 Core 内置 `privacy_mask` → 新 Core 把它从发行版删除、preset 行保留、extra 装上：升级后能力快照与升级前对比为空（来源字段除外）；中途让预检失败，原服务不停；让启动后验收失败，`_recover_previous` 回到旧版本且 `plugin-stable.json` 与停机快照逐字相同；自部署路径（`submit`）同样走一遍。

**P1-X · 已装插件的迁移（独立，Phase 3 之前，约 200 行，规模待评估）**
- 改：`agent/migrations/bundles.py:130`（不再只认 `builtin`）、`agent/migrations/runner.py`。
- 已实现：只对官方 marketplace（`distribution.json` 的 `marketplace`，即 `release`）的已装插件执行迁移，时机与内置相同（停机发布、启动）；部署清单的目标取目标代码；在线安装遇到未执行迁移拒绝。其他 marketplace 维持现状。

PR 数量：C1、A1、A2、B1、B2、B3、B4、C2a、C2b，共九个，外加 P1-X。

## 6. 开放问题（需要你拍板）

**Q1 · 迁出插件的身份命名空间。** 推荐：迁出的插件继续用 `@release`；只在 extra 里新建的插件用 `@extra`。来源靠 artifact 的 git origin 推导，不进 plugin_id。代价是名字上 `release` 不再等于"来自 Core 发行版"。备选：全部改 `@extra`，必须为每个迁出的插件写 ledger owner 和 Message metadata 两份迁移，且必须在停机窗口内完成。

**Q2 · 单分支还是每插件一分支。** 推荐每插件一分支（`release/<name>`）。如果你要单个 `release` 分支和子树哈希：安装器要新增"子目录即插件根"，artifact 身份和 `pkg upgrade` 的比较要改成子树哈希（需要先对 release 分支做一次 blobless fetch 才能读子树，`ls-remote` 给不出），工作量大约多一个 PR（约 300 行）。

**Q3 · 内置到 extra 的衔接是否走"宿主侧联网、清单进发布"。** 推荐是（2.9）。唯一的备选是"Core 先升级，起来后在线 `pkg sync`"，它有 Core 已升级但 extra 缺失的窗口，联网失败时默认组合少插件，我不推荐。

**Q4 · 在线 `pkg install` 是否要显式预拉镜像。** 推荐 Phase 1 不做：宿主侧部署预拉（C2a）加 controller 缺镜像兜底（A2）已经覆盖事故场景；在线路径要新增一个 `pull` RPC 和 Host Bridge 协议字段，改动面大。代价是首次在线安装带大镜像的插件时，拉取时间计入启动超时（computer 是 90 秒，`computer/plugin.py:395`）。

**Q5 · 兼容声明是否只用 `api_version`。** 推荐是（2.6），接受"符号级不兼容不会被发现"的风险，靠"删符号必须升 `api_version`"和 extra CI 门禁兜底。备选是现在就加 `requires_core` 字面量。

**Q6 · 自动补依赖。** extra 插件之间有 import 级依赖（例如 `watch` 导入 `plugins.userinformation.contract`，`plugins/watch/plugin.py:20`）。推荐 Phase 1 不做自动解析：缺依赖时现有"新代先导入再提交"会让安装失败并保持旧状态（`manager.py:1803-1805`），`pkg sync` 按 preset 顺序装，所以首装不会遇到。`pkg install watch` 单独装时的报错信息不友好，Phase 5 再做包组和依赖提示。

**Q7 · 离线首装怎么办。** 推荐沿用 `--inputs` 目录：用有网的机器跑 `install --no-activate --extra` 生成清单和 bundle 目录，拷过去再装。不新增"离线仓库"概念。

**Q8 · P1-X（extra 插件的迁移）的信任模型。** 这是 Phase 3 的阻塞项，但会影响 Phase 1 的包格式（迁移目录要不要随 release 分支走）。推荐 Phase 1 先不动，等 `scheduler`/`akasha` 之前再单独设计。需要你确认顺序。

## 7. 决定（2026-11-22）

- Q1：迁出的插件保持 `@release` 身份；只在 extra 里新建的插件用 `@extra`。marketplace 是身份命名空间，不代表代码来源。
- Q2：每个插件一个 `release/<name>` 分支。
- Q3：内置到 extra 的衔接走宿主侧联网、清单进发布。
- Q4：在线 `pkg install` 不显式预拉镜像。
- Q5：兼容声明只用 `api_version`。
- Q6：Phase 1 不做自动补依赖。
- Q7：离线首装沿用 `--inputs` 目录。
- Q8：extra 插件的迁移（P1-X）在迁 `scheduler`、`akasha` 之前单独设计。

身份不变带来的结论：迁出插件的数据目录、ledger owner、Message metadata、preset 行都不需要变。因此数据目录归一化（#2165）不是迁移的前置条件，已搁置；B4 需要的"清除 adoption 条目"语义由 B4 自己实现，换来源沿用旧记录里的 `data_dir`。

## 8. 评审修订（2026-11-22）

以下两点修正前文，实现以这里为准。

**回退必须连同 cache 指针一起恢复（修正 2.8、2.9、2.10）。** `install_git_plugin` 不是只建立不可见制品：它会把 `cache/<marketplace>/<name>` 的 stable/latest 指针发布到新 artifact，并写用户 patch、提交安装恢复点。而部署回退 `_roll_back` 只恢复选择文件、旧单元和 `runtime.env`。换来源后同身份（如 `watch@release`）的 cache 指针指向 extra artifact，旧 Core 又自带这个插件，启动时会因"内置与外置共用数据身份"拒绝。因此：

- 部署在停机时保存的恢复快照，除选择文件外，还要包含本次清单涉及的每个插件的 cache 指针文件和用户 patch；`_roll_back` 一并恢复。
- 在线 `pkg switch` 在选择 CAS 之前失败时（如新代导入失败），同样要把该插件的 cache 指针和 patch 恢复到切换前。
- C2b 的验收必须包含：换来源成功后让启动验收失败，回退到旧 Core 能正常启动，选择、cache 指针、patch 与停机前逐字相同。B4 的验收包含导入失败后指针已恢复、重启实例正常。

**artifact 的 origin 要指向真实上游（修正 2.3、2.9）。** 部署走离线 bundle 安装时，clone 的 origin 是临时 bundle 路径，按 origin 判断来源会把这些包识别成 `git:<临时路径>`，`pkg upgrade` 会漏掉它们。因此部署清单的每个条目携带真实上游地址，安装完成后把 artifact 的 `remote.origin.url` 设为该地址；`pkg list` 与 `pkg upgrade` 仍只以 artifact 的 origin 作为唯一的来源表示，不另存状态。C2b 验收包含：经部署安装的 extra 包在 `pkg list` 里来源为 `extra`，随后 `pkg upgrade` 能升级它。

**镜像（修正 2.7）。** controller 里的拉取只是小镜像的兜底，带硬的总期限，到期干净失败；大镜像依赖部署前在宿主预拉。
