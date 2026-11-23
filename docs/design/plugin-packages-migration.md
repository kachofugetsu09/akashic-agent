# 非核心插件外移与包管理：迁移计划

- 日期：2026-11-22
- 状态：Phase 0–6 已完成（2026-11-22）。结论见 [ADR 0167](../decisions/0167-plugins-ship-from-extra.md)，它取代 0148。本文保留为过程记录：下面的“已知约束”“已定决策”是动手前的状态，后续阶段的记录写明了怎么解决的。
- 关系：0148 撤回 #1973 的原因是安装与更新流程没有跟上（插件 PR、升 lock、部署三步；镜像、制品库、覆盖三处来源）。本计划先重做初始化、安装、更新流程，再外移插件。

## 目标

- Core 仓库只保留能独立跑完长程循环、能装卸插件的最小闭环。
- 其余内置插件迁到官方 extra 单仓库，按插件粒度发布和更新；私人插件走同一套命令。
- 改一个外置插件不需要重新发布 Core。

## 不变量

**全量安装的实例，迁移前后能力完全一致。**

每个阶段都按下面的规则验收：

1. 能力快照对比。从运行中的实例导出：已激活插件、工具及 schema、UI 插槽、设置分节、技能、命令、RPC 方法、登记的文档。迁移前后差异必须为空，或只有预期内的来源字段变化。
2. 真实验收。每个迁出的插件在隔离实例上真实走一遍主功能，再上生产。
3. 数据核对。有持久数据的插件，迁移前后核对数据目录的文件数和关键记录。
4. 可回退。先在 extra 装上并验证，下一个 PR 才从主仓删除。
5. 一次只上线一批。开发可以并发，上生产串行。

Phase 5 之前默认组合不变：新装实例仍然装齐今天 `bundles/base.toml` 的全部插件，只是来源从内置变成 extra。

## 已知约束

- 插件自带的 Python 环境只对子进程命令生效，加载器不会把它加进主进程的 import 路径。进程内 import 第三方包的插件（`akasha`：numba、numpy、jieba；`design`、`visuals`：playwright；`telegram_channel`：telegram）在依赖方案确定前不能外移。（已由 Phase 4 的 `requirements.inprocess.txt` 解决。）
- `visuals` 没有浏览器时工具直接返回错误，没有降级路径。
- playwright 和 Chromium 由 `design`、`visuals` 使用，与 `computer` 无关。（Chromium 已改由两个插件的 `resources.json` 提供，镜像不再自带，见 Phase 4 记录。）
- 容器镜像不在本地时启动会失败（2026-11-21 事故）。声明镜像的包必须在安装和升级时先拉取再激活。
- 一个插件只有一个来源。来源优先级覆盖、临时试用暂不做，并且不得让“现在跑的是哪份”变模糊。

## 已定决策（2026-11-22）

- extra 仓库：公开的 `kachofugetsu09/akashic-plugins`。
- 插件来源：Core 镜像不带 extra 插件，也没有 lock。安装或 `pkg upgrade` 时联网拉取 extra 仓库当前最新版。
- 重依赖：插件声明自己的依赖，装进插件自己的环境并挂进主进程；安装时检查与已装插件的包版本冲突。
- 进程内依赖必须**显式声明**：插件根放 `requirements.inprocess.txt`（存在且非空才算），它是主进程依赖的全部清单，装进独立的环境（与根 `requirements.txt` 的子进程 runtime 各自独立，两者可并存）。只有它参与挂载（`sys.path` 末尾，Core 的包优先）、版本约束（Core 锁 + 其他插件的进程内环境）和按 requirements 复用环境。根 `requirements.txt` 语义不变：只给子进程命令，不挂载、不加约束。理由：生产上已装的外置插件有根 `requirements.txt` 但从不在主进程用，隐式约定会让它们的包变得可 import，并让钉了不同版本（如 `numpy==2.4.4`）的插件被拒。不改 manifest schema 与内容摘要。
- 数据目录：统一成 `plugin-data/<name>`。归一化是显式命令，不在启动流程里；遗留的非在用目录原地不动。
- Phase 5 先做包组和 Web 首次引导，其余做完后再定。（后来改为终端选装，网页引导不动，见 Phase 5 记录。）

## 分层

每个插件在 `plugin_boundary.toml` 的 `[tiers]` 声明层级 `core` 或 `extra`。门禁 R14：`core` 层不得 import `extra` 层的任何模块。

哪些插件属于最小闭环是设计判断，不由门禁决定。R14 只证明这个划分下受支持的 Python import 方向成立：它覆盖模块级、函数内、相对 import 和字面参数的 `import_module`；不覆盖计算出来的模块名、前端模块之间的依赖、`bundles/*.toml` 里对插件的点名。这三类在每批迁移时人工核对。

迁出的插件从 `[tiers].extra` 移到 `[tiers].external`（目录可以不存在，仍受 R14 约束）。extra 仓库自己的新插件不进 Core 的表：合成树根目录的 `extra_tiers.toml`（只有 `extra = [...]`）追加声明，只能追加 extra 层，不能改写 Core 的声明。

core 的判定标准：裸核能独立跑完长程循环（账本、上下文、压缩、工具、文件与 shell、一个通用模型驱动）、提供 Web 宿主与投递、能装卸插件。`workloads`（`ui`、`mcp` 依赖）、`runtime_inspection`（`prompt` 依赖）、`content_view`（工具大输出回读）因此留在 core。

| 依赖 | 现状 | 处理 |
|---|---|---|
| `runtime_inspection` → `scheduler` | 可选借用 | 已反转：端口由 `runtime_inspection` 定义，`scheduler` 实现 |
| `ui` → `workloads`、`mcp` → `workloads` | 借用与合同字段 | `workloads` 留在 core，不需要反转 |
| `prompt` → `runtime_inspection` | 登记文档 | `runtime_inspection` 留在 core |

## 阶段

### Phase 0：地基（只动主仓，不搬插件）

1. 能力快照脚本，导出生产基线。
2. 层级声明与门禁。
3. 反转上表的依赖。
4. 公开接口：插件可用的 Core 模块以门禁里的 `PLUGIN_ALLOWED_MODULES` 为准，不另建 SDK 包；extra 仓库的 CI 运行同一套门禁。
5. 数据目录：
   - 已选择的插件一律以选择记录里的 `data_dir` 为准。
   - 记录只接受 `<name>-<marketplace>`（旧形态）和 `<name>`（归一化形态）。
   - 归一化是一条在实例停止时手动执行的命令，带回执和反向命令；启动流程不改任何目录。
   - 本阶段不支持换来源：同逻辑名的其他来源一律拒绝安装。换来源是 Phase 1 的一个原子操作。

回退合同：只升级不归一化时，新旧版本可以来回切换。归一化之后旧版读不了选择记录（启动报错，不会静默使用空目录），回退前先执行反向命令。

已知限制：带资源声明（`resources.json`）的插件上线后，Core 不能单独回退到不认识资源键（安装记录里的 `resource/<name>/`）的版本，旧 Core 会拒绝这样的安装记录；完整的部署回退会同时恢复旧选择与 cache 指针，不受影响。

验收：快照与基线一致，生产数据零丢失。

### Phase 1：最小包管理器（主仓）

1. 源配置与索引格式；按插件子树哈希判断是否变化。
2. `akashic pkg sync / install / upgrade / remove / list`。
3. 包元数据声明容器镜像，安装时先拉取。
4. 部署脚本与 Agent 自部署衔接 `pkg upgrade`，失败可回退。
5. 换来源：在一次提交内把同逻辑名插件从旧来源替换为新来源，沿用同一数据目录，并同步 `distribution_adoption` 里的历史归属。

验收：隔离实例上用本地索引跑通；改一个包只更新一个包。

### Phase 2：extra 仓库与两个样本

1. 建仓、增量构建与发布；CI 钉 Core 版本做契约检查。
2. 前端共享包做成可被外部仓库引用的版本化包。
3. 迁 `privacy_mask`（无数据、无第三方依赖）。
4. 迁 `watch`（有数据、面板、技能）。

### 单个插件迁出清单（以 `privacy_mask` 为模板）

Core 侧一个 PR，extra 侧先行或同时：

1. extra 仓库：把 `plugins/<name>/`、`frontend/plugins/<name>/` 原样放进去；测试插件自身行为的 `tests/` 文件搬到 `tests/<name>/`。本地 `extra_ci.py compose/build` 后 `release/<name>` 的提交 SHA 必须与迁出前在 Core 上 `pkg_fixture_repo.py <name>` 得到的相同。
2. Core：删除 `plugins/<name>/` 与 `frontend/plugins/<name>/`；`plugin_boundary.toml` 把名字从 `[tiers].extra` 移到 `[tiers].external`。
3. **保留** `bundles/*.toml` 里的行：默认组合不变靠 preset 行。
4. 全仓库搜该名字（含连字符、驼峰变体与中文名）：测试里测 Core 行为只是拿它当例子的改写；脚本、`docker/`、`package.json`、`tsconfig.json` 的显式路径；文档与技能里的说明。历史 ADR 不改。
5. 门禁：`plugin_boundary.py check --base origin/main`、pyright、ruff、`pytest tests`、`npm run typecheck`、三项样式/动效合同、完整 `npm run build`。
6. 其余插件的 release 提交 SHA 迁出前后逐个相同。
7. 发行版不再含该插件，但 preset 行还在：`install_plugin_distribution.py --ensure-bundle` 的 `_require_preset_rows` 会以"preset 启用的插件没有进入选择，停止发布"终止，除非同名外置插件已经选中。所以合并前，部署流程必须能在 `--ensure-bundle` 之前从 extra 仓库装上它（`pkg sync` 应装集合会把它报成 `available`）。

带数据、带依赖、带测试的插件（以 `watch` 为模板）要多做这些：

- **找全持久状态再动手。** 读插件的 `workspace_roots`、`ctx.data_root`、`Files(...)`、`MESSAGE_CATALOG`、迁移目录，并核对三处：`workspace/<root>/` 与 `workspace/memory/<root>/` 下的文件、`plugin-data/<name>-<marketplace>/`（含 `config.input.json`、`state.json`）、`sessions.db` 的 `owner_records`。设置保存在选择记录的 `config_revision`/`config.input.json` 里，换来源必须原样带过去。`watch` 没有迁移目录，也不写 `owner_records`。
- **数据核对要用真实实例。** 在含该插件的版本上经设置接口配好并跑起来产生真实数据，停机后记文件摘要，再换成"已迁出"版本（`start.py --extra-repo <本地裸仓库>`）启动，对比：`data_dir` 与 `config_revision` 不变、数据文件摘要不变；运行中的插件会重写自己的游标和日志，那类文件的差异只能是它自己追加的内容。
- **依赖方向。** 列出 `from plugins.X.contract import`。`core` 层不得依赖被迁出的插件（门禁 R14 保证），被迁出的插件依赖仍在 Core 的 core 层或 extra 层插件没有问题（`watch` 依赖 `userinformation`：实测外置的 `watch` 能导入内置 `userinformation` 的合同）。被依赖的插件之后也迁出时，extra 仓库合成树里必须仍能找到它的源码：先把它放进 extra 仓库（同一批或更早），再让 Core 删除它。部署必须把两者在同一次发布里装上，否则"新代先导入"会让依赖方安装失败。
- **反向依赖。** 搜 Core 里对被迁出插件的 `ServiceKey` 字符串（如 `watch.events.v1`）、`contract` import、`bundles`、技能与提示词的点名；有就先反转依赖再迁。
- **测试。** `tests/` 里逐个判断：测插件自身行为的搬到 extra 仓库 `tests/<name>/`；只是拿它的名字当例子的（如 `test_material_execution.py`）改成中性名字。没有测试就不建目录。
- **源码目录里的构建产物。** 本地构建会在 `plugins/<name>/` 生成未入库的 `plugin_ui.js` 等；复制到 extra 仓库前清掉，并用 `git ls-tree` 的文件清单和逐文件 `cmp` 核对与 Core 提交一致。
- **技能与文档。** 技能随插件目录走，不用单独处理；`docs/INDEX.md` 的代码路径指向 extra 仓库；历史 ADR 不改。
- **能力快照。** 迁出前后 `capability_snapshot.py diff --ignore-source --ignore-build` 应只剩 `[preset].retired` 多出该插件（来源变化的预期信号）。面板用 `/api/chat/plugin-ui/query` 的 `day` 读一次，确认能从迁出后的实例读到旧数据。

批量迁出（Phase 3）再多做这些：

- **不在 preset、只靠用户 patch 启用的插件不要新增 preset 行**（也不动 `bundles/*.toml`）。迁出后启动补装只处理 preset 缺口，这类插件会被静默丢出选择（实例照常启动，`pkg sync --dry-run` 报"只在用户 patch 里启用、上游存在"）；`pkg install` 对 `@release` 身份只回 `needs-switch`。所以这类迁出必须等部署侧"当前选择里在用的内置插件也换来源"合并后再合入。
- **测试按行为拆。** 整个文件测同一插件的搬到 `tests/<name>/`（加空 `__init__.py`，避免同名模块冲突）；一个文件里混着 Core 与插件行为的，把插件那几个用例原样搬走；Core 测试拿被迁出插件当示例（如 `test_plugin_hot_reload.py` 里 standard_web 的提供方登记）时，在 Core 里去掉该段，把被去掉的断言放进 extra 的 `tests/<name>/`，保持覆盖。
- **非测试的名字引用。** 搜 `set_plugin_choice(..., "<name>")` 这类对被迁出插件的开关（基准、脚本）：插件不在 Core 树里时没有意义，删掉；顺带的文档文字若属于另一个 extra 插件的目录（如 `plugins/computer/bench/README.md`），不要改，否则那个插件的 release 提交会变。
- **只为这些插件存在的第三方依赖只列出，不删。** 本批只有 `web_fetch_http` 进程内导入 `html2text`、`lxml`（`requirements.txt`、`docker/host-runtime/requirements.lock`）；Core 镜像继续带着，直到依赖方案确定。
- **验证夹具。** 带 `distribution.json` 摘要校验的发行版不能用软链复制；要临时改 preset 行验证"迁出后的终态"时，用硬链接复制发行版目录、改 `bundles/base.toml` 并重算 `distribution.json` 里的 bundle 摘要，用 `start.py --distribution` 启动，不要改共享缓存里的发行版。

工作台类、主动与记忆类（批 B、C）再补：

- **宿主前端与 core 插件的点名要先扫。** 搜 `frontend/plugins/akashic_clients`、`frontend/plugins/ui`、`packages/`、其余 core 插件对被迁出插件 id、`plugins.<name>` import、服务键字符串的引用；批 B 的 7 个、批 C 的 6 个都没有反向依赖（唯一的跨插件 import 是 `scheduler`→`timer`，同批迁出）。`scripts/check_narrow_ui.mjs` 之类按插件名放行查询的脚本只是名单，插件缺席时不会失败，不用动。
- **带数据库迁移的插件**（`migration.catalog.toml` + `<name>_migrations/`）：两者都在源码目录里，随 release 分支一起发布；验证看 `migrations.sqlite3` 的 `_yoyo_migration`、`_yoyo_log` 行数与迁出前相同（迁出后历史迁移不重跑）。
- **随插件搬的测试里的相对路径与共用夹具。** 测试从 `tests/` 搬到 `tests/<name>/` 后，`Path(__file__).parents[1] / "plugins"` 要改成 `parents[2]`；只被这个插件的测试用的夹具（如 `tests/support/goal_harness.py`）一起搬进 `tests/<name>/`，import 路径同步改；Core 共用的夹具（`tests.fixtures`、`tests.support.delivery_sources`）留在 Core。
- **core 配置引用 extra 插件。** `bundles/base.toml` 里 `context` 的 `prompt_sources` 点名 `markdown_memory@release`：该插件被禁用（缺席）时实例照常启动，`context` 仍 active，日志没有告警或错误。
- **只在基准里配置的插件名。** `benchmark/computer_runtime/runtime.py` 的 `prompt_sources` 仍点名 `markdown_memory`；该基准直接从 Core 的 `plugins/` 目录装配，迁出后需要给它提供 extra 仓库的检出目录（`AKASHIC_EXTRA_PLUGIN_DIRS`），这次没有处理。
- **能力快照差异要排除同一分支里先前批次的影响。** 叠栈分支上，前一批迁出的插件（本来就没启用）会在 diff 里显示"少了"，只看当前批次的插件即可。

模型驱动、容器、目录名与 manifest 名不一致的插件（批 D）再补：

- **目录名与 manifest 名不一致**（`opencode_go` 的 manifest 名是 `opencode-go`；Core 里的 `conversation_ui`、`openai_compatible`、`shell_ui`、`web_auth` 同理）：安装器要求 `release/<name>` 的分支名等于 manifest 名，插件身份、cache 路径、`data_dir`、bundles 的行都用 manifest 名（`opencode-go@release`、`plugin-data/opencode-go-release`）；而 `plugin_boundary.toml` 的 `[tiers]` 与源码目录用目录名。所以 `scripts/pkg_fixture_repo.py` 改成：参数是目录名，分支、提交说明和返回行取 manifest 名，不再因两者不同而报错。extra 的 `extra_ci.py` 不用改。
- **模型驱动插件。** 搜 models、onboarding、plugin_settings、前端模型页对驱动名的点名：本批只有 `plugins/models/modelsdev_capabilities.py` 里的服务商别名表（驱动名到 models.dev 的 id，静态数据，不依赖插件存在）。裸核仍有通用驱动 `openai-compatible`（core 层）。驱动缺席时，Core 里已有的连接显示 `driver_unavailable`，模型页与引导不报错，连接、凭据和模型记录留在 Core 的 `model-registry.sqlite3`，换来源前后表内容不变（只有 meta 行的计数随启动递增）。
- **声明容器镜像的插件**（`images`）：release 分支里 `plugin.py` 原样带着；用 Core 的 `load_static_plugin_manifest` 读 release 分支确认 `images` 完整。镜像由 Core 的 `docker/computer` 与 `.github/workflows/computer-image.yml` 构建发布，仍留在 Core；镜像摘要变了要在 extra 仓库改 `plugin.py`。
- **自带评测/工具脚本的插件。** `plugins/computer/bench/` 以 `parents[3]` 找仓库根并 import `benchmark.computer_runtime`；`benchmark/computer_runtime`、`benchmark/computer_driver` 直接从 Core 的 `plugins/` 装配或读 `plugins/computer/skills`。迁出后它们只能在合成目录里运行（`extra_ci.py compose` 的输出），不在任何 CI 里跑。
- **Core 构建脚本里按插件名的特例。** `scripts/build-web-plugins.mjs` 里 `plugin === "computer" ? "es2022"` 仍在 Core，构建时对合成目录里的同名插件生效；不要为迁出改它，否则前端产物会变。
- **进程内导入第三方包的插件**（`telegram_sender`：`aiohttp`、`telegramify_markdown`）：Core 镜像继续带着这些包就能迁；只列出，不删 `requirements.txt` 与 `docker/host-runtime/requirements.lock` 里的条目。
- **没有 docker 的验证环境。** 能验证插件加载、active 与配置保留；容器启动与镜像预拉验证不到，要如实写明。

### Phase 3：批量迁纯 Python 插件

每批一个 PR，顺序以门禁结果为准：

- A：`subagent` 族、`standard_web` 族、`stable_view`
- B：`projects`、`main_session`、`session_title`、`session_reference`、`workspace_changes`、`markdown_reader`、`terminal`
- C：`goal`、`scheduler`、`timer`、`userinformation`、`message_push`、`markdown_memory`
- D：`codex`、`gemini`、`opencode_go`、`telegram_sender`、`vision`、`computer`

### Phase 3 完成记录

已迁出（身份保持 `@release`，登记在 `plugin_boundary.toml` 的 `[tiers].external`；Core PR / extra PR）：

| 批 | 插件 | Core | extra |
|---|---|---|---|
| 样本 | `privacy_mask` | #2178 | 建仓时一并推出 |
| 样本 | `watch` | #2180 | #9 |
| A | `subagent`、`subagent_codex`、`subagent_claude_code`、`standard_web`、`web_fetch_http`、`web_search_exa`、`web_search_searxng`、`stable_view` | #2182 | #10 |
| B | `projects`、`main_session`、`session_title`、`session_reference`、`workspace_changes`、`markdown_reader`、`terminal` | #2184 | #11 |
| C | `goal`、`scheduler`、`timer`、`userinformation`、`message_push`、`markdown_memory` | #2185 | #12 |
| D | `codex`、`gemini`、`opencode_go`（身份 `opencode-go`）、`telegram_sender`、`vision`、`computer` | #2186 | #13 |

配套：#2181 与 #2183（换来源按"现在是否在用"判定、官方 extra 包按 `@release` 全新安装）、#2172（已安装的官方身份插件执行迁移）、#2176/#2177（部署侧产出清单与首装补装）。

仍在 Core 的 `[tiers].extra`：`akasha`、`design`、`visuals`、`telegram_channel`，原因都是 Phase 4：进程内导入第三方包（numba/numpy/jieba、playwright、python-telegram-bot），在"插件自带依赖挂进主进程"实现前不能迁。除这四个外，extra 层没有被遗漏的可迁插件。

迁移门禁：`scripts/check_yoyo_migrations.py` 已支持 `--root` 指向外部仓库根。extra 仓库的 CI 接入它（PR 上跑"已发布迁移只能追加"）并把固定的 Core 版本升到包含该参数的提交之后，这条保证才由 extra 承接；在那之前 Core 自己的检查对已迁出的插件放行，迁出插件的迁移不可变没有门禁。

已知缺口：

- 聊天输入框的"引用会话"（拖入会话、`MAX_REFERENCES`）写在宿主前端，后端读取由 `session_reference` 提供；Phase 5 做按场景选装时，`session_reference` 不能当可选包，否则界面还能引用而后端没有人读。
- 声明镜像的插件（`computer`）的镜像仍由 Core 的 `docker/computer` 与 `.github/workflows/computer-image.yml` 构建；摘要变化要在 extra 仓库改 `plugin.py`。
- `scripts/build-web-plugins.mjs` 里 `computer` 的构建目标特例、visuals/design 的专用构建段仍在 Core 的构建脚本里（Phase 4）。
- 进程内导入第三方包的已迁插件（`web_fetch_http`：html2text、lxml；`telegram_sender`：aiohttp、telegramify-markdown）依赖 Core 镜像继续带着这些包。
- `benchmark/computer_runtime` 与 `benchmark/computer_driver` 的 `--agent` 模式只能在 extra 仓库 `extra_ci.py compose` 的合成目录里运行；`computer_driver` 的非 Agent 模式不受影响。

### Phase 4：带重依赖的插件

先实现“插件自带依赖挂进主进程”和安装时的版本冲突检查，再依次迁 `telegram_channel`、`visuals`、`design`、`akasha`。之后才从 Core 镜像删除对应依赖，并实测镜像体积与内存。

#### Phase 4 进展：`telegram_channel`（第一个带进程内依赖的迁出）

- 迁出：`telegram_channel` 进 extra 仓库并保持 `@release` 身份，`[tiers].extra` → `external`，preset 行保留。Core 里没有对它的 import、服务键、前端或技能引用；Core 里只有"渠道名是 telegram"的字符串（提示词的渲染限制、压缩页的标签），不是插件依赖。持久数据只有 `plugin-data/telegram_channel-release/` 的配置与凭据（`config.input.json`），迁出前后逐字相同；没有 `owner_records`。
- 依赖声明：`telegram_channel` 与 `telegram_sender`（已迁出，仍靠 Core 镜像带着依赖）各自新增 `requirements.inprocess.txt`，只列直接 import 的第三方包：`aiohttp>=3.14.3,<4`、`python-telegram-bot>=22.7,<23`（仅 channel）、`telegramify-markdown>=1.2.0,<2`。下界取 Core 的 `requirements.txt`，上界取下一个主版本：此刻与 Core 锁（aiohttp 3.14.3、python-telegram-bot 22.8、telegramify-markdown 1.2.0）兼容，Core 在同一主版本内升级也不会让安装被拒，Core 删掉依赖后插件环境按范围解析出可用版本。不钉死 `==`：Core 每次升锁都会让钉死的插件安装被拒。`pydantic`、`httpx` 是 Core 永久依赖，不声明。`telegram_channel` 原来的根 `requirements.txt`（`httpx[socks]`，没有任何子进程命令使用）随迁出删除。
- 此刻主进程实际用的仍是 Core 的版本（Core 的包优先）；插件环境是备份。

**之后从 Core 删除依赖的清单（单独一步，不在迁出里做）**

- `requirements.txt`：`aiohttp>=3.14.3`（第 1 行）、`python-telegram-bot>=22.7`（第 23 行）、`telegramify-markdown`（第 27 行）。
- `docker/host-runtime/requirements.lock` 重新生成后应少掉：`aiohttp`、`python-telegram-bot`、`telegramify-markdown`，以及只由它们带来的 `pyromark`、`aiohappyeyeballs`、`aiosignal`、`frozenlist`、`multidict`、`propcache`、`yarl`（`attrs` 还被 `jsonschema`/`referencing` 需要，保留）。
- 不用动：`docker/host-runtime/requirements.production.in`（只引用 `requirements.txt`）、`benchmark/computer_driver/*.lock`（评测自己的环境）；Core 里没有任何 `aiohttp`/`telegram`/`telegramify_markdown`/`pyromark` 的 Python import（评测脚本除外）。
- 删之前必须满足：(1) `telegram_channel` 与 `telegram_sender` 的新版本都已在生产运行（`pkg list` 显示来源 extra、带进程内环境引用）；(2) 本机用"去掉这些包的 Core 解释器"启动同一实例，两个插件仍 active、原生扩展来自插件环境（已演练，见 #本 PR 说明）；(3) 在目标镜像里重新求 wheel 通过（部署的 `pip download --only-binary` 不失败）；(4) 确认没有其他已装插件依赖这些包（`grep` 其源码与 extra 仓库 CI 的 `check_inprocess.py --hints`）；(5) 删除作为独立 PR，随后用 `pkg upgrade` 之外的完整部署演练一次回退。

#### Phase 4 进展：`visuals`、`design`（playwright 进程内依赖；Chromium 后来改为插件资源）

- 迁出：两个插件进 extra 仓库，`[tiers].extra` → `external`，preset 行保留；Core 里没有对它们的 import、服务键或宿主前端点名（`frontend/plugins/ui/dashboard` 里的 `./design/runtime` 是仪表盘自己的模块，与插件无关）。没有涉及它们的测试。`visuals` 的 `frame.html`、`preview-theme.css`、`bootstrap.html` 是入库的构建产物，release 分支取的是入库版本（`stage_plugin_package` 只用构建产物覆盖 `web_module.*`、`plugin_ui.*`、`message_ui.*`、`static/`），所以随插件一起迁，改前端源码后要重新生成并提交。
- 依赖声明：两者的 `requirements.inprocess.txt` 都是 `playwright==1.59.0`（纯 ASCII）。与 Core 的 `requirements.txt`（`playwright==1.59.0`）和锁完全一致；不用范围，因为 playwright 的版本决定它期望的 Chromium 构建号（1.59.0 对应 1217），镜像里的浏览器是按 Core 的 playwright 装的。两份声明逐字节相同，共用一个环境（实测同一个环境引用，约 240 MB，主要是 playwright 自带的 driver）。`fastapi`、`starlette`、`httpx`、`pydantic` 是 Core 永久依赖，不声明。
- 构建特例留在 Core：`scripts/build-web-plugins.mjs` 里 visuals 与 design 的专用段本来就按"目录存在才处理"生效，合成目录里存在，所以 release 提交与迁出前在 Core 上构建的相同（只多一个 `requirements.inprocess.txt`）。`package.json` 的 `typecheck` 原来写死了设计编辑器的 tsconfig，改成 `scripts/typecheck-plugin-projects.mjs`：对 `frontend/plugins/*/*/tsconfig.json` 存在的子项目才检查；Core 自己不依赖，合成目录里同一条 `npm run typecheck` 仍覆盖设计编辑器。设计编辑器的 npm 依赖（`@open-pencil/*`、`canvaskit-wasm`、`vue`、`@valibot/to-json-schema`）仍在 Core 的 `package.json`，因为 extra 的构建用 Core 的锁。

**之后从 Core 删除 playwright 的清单（单独一步）**

- `requirements.txt`：`playwright==1.59.0`（第 17 行）；`docker/host-runtime/requirements.lock` 少掉 `playwright`、`greenlet`、`pyee`（后两者只由 playwright 带来）。
- `docker/host-runtime/Dockerfile.distribution`：`python -m playwright install --only-shell chromium`（第 73 行）依赖 Core 的 playwright，`PLAYWRIGHT_BROWSERS_PATH`（第 14、61 行）也要重新安排：删 playwright 之前必须先让 Chromium 由插件拥有（另一条调研线），否则镜像里不再有安装浏览器的手段。
- 前提：(1) 两个插件的新版本在生产运行且带进程内环境；(2) 用去掉 playwright 的 Core 解释器启动并真实渲染（已演练）；(3) 插件环境里的 playwright 版本与浏览器构建号一致——删之前由插件自己持有"playwright 版本 + 浏览器"的配对；(4) 目标镜像里求 wheel 通过；(5) Core 之后升级 playwright 必须与插件的钉死版本同步：先放宽插件声明（例如 `>=1.59.0,<1.61`）发布，再升 Core。
- 浏览器定位：两个插件默认用 `PLAYWRIGHT_BROWSERS_PATH`（镜像里是 `/opt/playwright`，本机默认缓存）；`visuals` 的设置里本来就有"预检浏览器路径"；extra 仓库这一步新增环境变量 `AKASHIC_BROWSER_EXECUTABLE`（两个插件都认，设置优先于环境变量），并把找不到浏览器的 Playwright 原始提示整理成一行原因加修复办法。接已有 CDP 端点（对标 DSH：核心不拥有浏览器）只给方案：`chromium.connect_over_cdp(endpoint)` 替代 `launch`，设计的无头页面依赖独立 context 与 `--site-per-process` 启动参数，连外部浏览器时这些参数不可用，需要单独验证后再做。

#### Phase 4 进展：`akasha`（记忆核心；进程内依赖 numpy、numba、jieba、threadpoolctl）

- 迁出：`akasha` 进 extra 仓库并保持 `akasha@release` 身份，`[tiers].extra` → `external`（extra 层至此清空），preset 行保留。Core 里没有对它的 import、服务键或宿主前端点名：`wake` 已不存在；bundles 里只有 preset 行，`context`、`prompt`、`compaction` 的配置没有点名 akasha；宿主聊天前端没有 akasha 的字面量（回想浮层与召回卡片走 `message.actions` 等通用插槽，由 akasha 自己的 `message_ui.js` 贡献），仅 `vite.harness.config.ts` 的本地开发开关引用 `plugins/akasha/message_ui.css`（只在 `AKASHIC_LOCAL_PLUGIN_UI=akasha` 时用，需要合成目录）。`scripts/`、`docker/debug` 里没有 akasha 场景脚本。
- 测试：`test_akasha_execution`、`test_akasha_graphs`、`test_akasha_learning`、`test_interest_execution` 随插件搬到 extra 的 `tests/akasha/`（路径深度改为 `parents[2]`，`test_interest_execution` 的 import 指向新位置）。`WorkerGate`、`loop_turn` 是 Core 其他测试（`test_storage_read_execution`、`test_conversation_source`）也在用的通用夹具，抽到 `tests/support/worker_gate.py`，两处 import 改过去；extra 的 `tests/subagent` 改为从 `tests.akasha.test_akasha_execution` 取。
- 依赖声明：`jieba==0.42.1`（精确：切分缓存键含版本）、`numba>=0.68.1,<1`、`numpy>=2.4.0,<3`、`threadpoolctl>=3.6.0,<4`；`pydantic`、`yoyo-migrations` 是 Core 永久依赖，不声明。与 Core 锁（jieba 0.42.1、numba 0.68.1、llvmlite 0.50.0、numpy 2.5.2、threadpoolctl 3.6.0）兼容，插件环境解析出的版本与锁逐一相同（约束把 numba 的 llvmlite 与 numpy 也钉在锁上），环境约 420 MB。
- jieba 没有 wheel：由插件根的 `wheels/jieba-0.42.1-py3-none-any.whl` 提供（#2196 的纯 Python wheel 目录）。该 wheel 直接提交进 extra 仓库，`wheels.lock.json` 记录 sdist 的 URL 与 sha256、固定的 `SOURCE_DATE_EPOCH` 与构建后端版本、wheel 的 sha256，`scripts/bundled_wheels.py check` 在 CI 里从 sdist 重建并要求逐字节一致（不一致时再比对 wheel 内容并如实报告）。
- 迁移：`migration.catalog.toml` 与 6 个迁移脚本随插件走；存量库不重跑（`_yoyo_migration`、`_yoyo_log` 行数迁出前后都是 24），全新库在安装时执行全部 akasha 迁移并建出 18 张表。

**之后从 Core 删除这些依赖的清单（单独一步）**

- `requirements.txt`：`jieba>=0.42.1`（第 9 行）、`numba>=0.68.1`（第 12 行）、`numpy>=2.4.0`（第 13 行）、`threadpoolctl>=3.6.0`（第 14 行）；`docker/host-runtime/requirements.lock` 重新生成后少掉 `jieba`、`numba`、`llvmlite`、`numpy`、`threadpoolctl`（Core 里没有任何这些包的 import）。
- 前提：(1) akasha 新版本在生产运行，`pkg list` 显示来源 extra 且带进程内环境；(2) 用去掉这些包的 Core 解释器启动并真实召回（已演练）；(3) 目标镜像里求 wheel 通过（numpy、numba、llvmlite、threadpoolctl 都要有 manylinux wheel，jieba 走 `wheels/`）；(4) 在线升级 akasha 默认走部署（环境 420 MB，依赖变化需重启）；(5) Python 小版本变化时环境必须在同一个停机窗口里重建（#2194 与后续）；(6) 记下 `numpy` 与 Core 其他可能的 numpy 使用（目前没有）。

#### Phase 4 完成记录

- Chromium 由插件拥有：`visuals`、`design` 在插件根声明 `resources.json`（chrome-headless-shell，与 playwright 1.59.0 的构建号配对），安装时取回并发布到 `runtime/plugin-resources/<sha>/content/`，写 `COMPLETE.json` 后才可用；两个插件经 `PluginRuntime.resource(name)` 定位浏览器，共用同一份资源。启动时缺失或损坏的资源在 60 秒内尽力修复，`pkg upgrade <名>` 做单包修复（300 秒）。
- 验收技能的截图脚本（`skills/verify-akashic-plugin/scripts/shoot_page.py`）不再假设镜像里有浏览器：依次试 Playwright 自带、插件资源（最近发布的在前）、系统浏览器。
- 上面三份“之后从 Core 删除”的清单已在 #2211 一次执行：`requirements.txt` 与 `docker/host-runtime/requirements.lock` 去掉 html2text、jieba、llvmlite、lxml、numba、numpy、pyromark、python-telegram-bot、telegramify-markdown、threadpoolctl；`Dockerfile.distribution` 去掉 `/opt/playwright` 与 Chromium 安装（playwright 包本身保留）。镜像 2.25 GB → 1.63 GB。生产部署前后能力快照一致（只有 `session_reference` 并回 Core 的预期变化）。

### Phase 5：新用户体验

实际做法按 2026-11-22 的定稿，取代下面原来的“先做 / 做完后再定”：

- 首次安装在终端里选（`scripts/first_install.py`，经 `start.py`、`akashic-release install`）：有 TTY 时问全部（默认）/ 最小 / 自选组，没有 TTY 时全装；`--profile`、`--groups`、`--yes`、`--reselect` 可显式指定。先列出要装的包，再逐包显示 `(i/N)`。联网或安装失败就退出，状态不变，可以重跑。未选的 preset 行以 `disabled=true` 写进用户 `bundle.patch.toml` 里一个带标记的自动生成块，块外用户写的行优先。已有实例的升级不走首装。
- extra 仓库加 `index` 分支（`groups.toml` → `scripts/extra_index.py`）：包组、说明、包间依赖、外部前置条件、镜像声明。Core 端 `agent/plugins/package_index.py` 读取并缓存。
- 外部前置条件（`docker`、`command:<命令>`）不满足的包照装但保持关闭，原因写进 `runtime/disabled-reasons.json`（带制品 head）；`pkg list`、`plugin-status`、设置页显示原因，启用时重新检测；声明了镜像的插件启用前先确认镜像在宿主上。
- `pkg install` 按 index 自动补上未装的依赖，依赖没有就绪时不装依赖它的包（#2214）。
- `session_reference` 并回 Core 的 `conversation`，登记在 `bundles/absorbed.toml`。
- Agent 自助：内置 `plugin-system` 技能写明查原因、补前置条件、在线启用（#2215）。
- 设置“插件”页加“可安装”分组。

原计划里的终端 `akashic init`、插件目录页、缺能力提示、`try`、`rollback`、`export/apply` 不做。

### Phase 6：收尾

- 写 [ADR 0167](../decisions/0167-plugins-ship-from-extra.md) 取代 0148。
- `plugin-install` 不删：它是私人插件和其他来源的入口（`pkg` 只管官方仓库），与 `pkg` 共用同一套安装制品与选择。
