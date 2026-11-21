# 非核心插件外移与包管理：迁移计划

- 日期：2026-11-22
- 状态：进行中（Phase 0）
- 关系：修订 ADR 0148。0148 撤回 #1973 的原因是安装与更新流程没有跟上（插件 PR、升 lock、部署三步；镜像、制品库、覆盖三处来源）。本计划先重做初始化、安装、更新流程，再外移插件；迁移完成后写新 ADR 取代 0148。

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

- 插件自带的 Python 环境只对子进程命令生效，加载器不会把它加进主进程的 import 路径。进程内 import 第三方包的插件（`akasha`：numba、numpy、jieba；`design`、`visuals`：playwright；`telegram_channel`：telegram）在依赖方案确定前不能外移。
- `visuals` 没有浏览器时工具直接返回错误，没有降级路径。
- playwright 和 Chromium 由 `design`、`visuals` 使用，与 `computer` 无关。
- 容器镜像不在本地时启动会失败（2026-11-21 事故）。声明镜像的包必须在安装和升级时先拉取再激活。
- 一个插件只有一个来源。来源优先级覆盖、临时试用放到 Phase 5，并且不得让“现在跑的是哪份”变模糊。

## 分层

每个插件声明层级 `core` 或 `extra`。门禁：`core` 层不得 import `extra` 层的 contract。切分是否正确由门禁判定，不靠手列清单。

当前已知需要反转的依赖：

| 依赖 | 现状 | 处理 |
|---|---|---|
| `ui` → `workloads` | 可选借用 | 由 `ui` 定义端口，`workloads` 实现 |
| `runtime_inspection` → `scheduler` | 可选借用 | 由 `runtime_inspection` 定义端口，`scheduler` 实现 |
| `prompt` → `runtime_inspection` | 登记文档 | 文档登记口留在 core |
| `mcp` → `workloads` | 合同字段 | `workloads` 留在 core |

## 阶段

### Phase 0：地基（只动主仓，不搬插件）

1. 能力快照脚本，导出生产基线。
2. 层级声明与门禁。
3. 反转上表的依赖。
4. 外置插件需要的内部模块收进公开 SDK。
5. 数据目录去来源化（`plugin-data/<name>/`），带自动迁移。

验收：快照与基线一致，生产数据零丢失。

### Phase 1：最小包管理器（主仓）

1. 源配置与索引格式；按插件子树哈希判断是否变化。
2. `akashic pkg sync / install / upgrade / remove / list`。
3. 包元数据声明容器镜像，安装时先拉取。
4. 部署脚本与 Agent 自部署衔接 `pkg upgrade`，失败可回退。

验收：隔离实例上用本地索引跑通；改一个包只更新一个包。

### Phase 2：extra 仓库与两个样本

1. 建仓、增量构建与发布；CI 钉 Core 版本做契约检查。
2. 前端共享包做成可被外部仓库引用的版本化包。
3. 迁 `privacy_mask`（无数据、无第三方依赖）。
4. 迁 `watch`（有数据、面板、技能）。

### Phase 3：批量迁纯 Python 插件

每批一个 PR，顺序以门禁结果为准：

- A：`subagent` 族、`standard_web` 族、`stable_view`、`content_view`
- B：`projects`、`main_session`、`session_title`、`session_reference`、`workspace_changes`、`markdown_reader`、`terminal`
- C：`goal`、`scheduler`、`timer`、`userinformation`、`message_push`、`markdown_memory`
- D：`codex`、`gemini`、`opencode_go`、`telegram_sender`、`vision`、`computer`

### Phase 4：带重依赖的插件

先定依赖方案（子进程、进程内挂载插件环境、或依赖暂留 Core 镜像），再依次迁 `telegram_channel`、`visuals`、`design`、`akasha`。之后才从 Core 镜像删除对应依赖，并实测镜像体积与内存。

### Phase 5：新用户体验

包组与默认组合；Web 首次引导与终端 `akashic init`；插件目录页、缺能力提示、Agent 查包工具；`try`、`rollback`、`export/apply`。

### Phase 6：收尾

更新插件开发与验收技能和文档；删除旧的 `plugin-install` 冗余路径；写新 ADR 取代 0148。
