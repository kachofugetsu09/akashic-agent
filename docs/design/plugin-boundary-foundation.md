# 插件边界地基：从「能加载插件」到「可判定的边界」

- 状态：proposed foundation / step 1 of 3
- 日期：2026-09-10
- 基线：`origin/main@4f9173c188a16289f0ac08d786f667e75df185d4`
- 关联设计：[React Core 与 Scheduler/Subagent](react-core-scheduler-subagent.md)、[插件 V3 能力手册](plugin-v3-capabilities.md)
- 关联决策：[0064 插件边界由机器强制](../decisions/0064-plugin-boundary-is-machine-enforced.md)
- 参考实现：`/mnt/data/source-code/deepseek-harness`（2026-09-08 checkout，只作为设计输入）
- 本文权限：只批准第 1 步（地基）。第 2、3 步需要各自的独立授权与 Gate。

## 1. 问题与用户意图

目标是让 Akashic 只提供「插件系统 + 原子能力」，所有产品功能是原子能力的正交组合。
正交的判据是一条可观察结果：**新增一个来源插件时，不需要修改任何已有插件。**

当前实现达不到这条判据，原因不是原子能力不够，而是**边界既没有被定义，也没有被强制**：

| 现象 | 当前证据 | 后果 |
|---|---|---|
| Core 直接 import 插件 | 28 条，例如 `bootstrap/app.py` → `plugins.delivery.senders`、`infra/channels/message_view.py` → `plugins.tools.api` | 移出仓库即崩；Core 无法独立启动 |
| 插件 import Core 内部深路径 | 244 处（去重），涉及 `session.*`、`agent.restart`、`agent.tools.*` 等 | 插件与 Core 同属一个 import 图，不是可替换实现 |
| 插件互相 import 实现模块 | 238 处（去重），最重的目标是 `plugins.tools`、`plugins.delivery`、`plugins.context` | 「组合」与「耦合」在语法上无法区分 |
| 文档承诺超过代码实现 | `SCOPED_TURNS`、`CONTINUATIONS`、`BACKGROUND_JOBS` 在多份文档与 skill 中被当作现有积木，Python 代码中零命中 | 后续决策建立在假前提上 |
| Turn 执行没有原子接口 | `plugins/conversation/program.py::run_reply` 有 25 个参数、11 个 `ctx.require`，被 4 个来源各自复制调用 | 每个来源复制一套执行模型，正交性不成立 |

单独看每一条都像「还没迁完」。合起来看是同一个事实：**没有可失败的检查，边界只能靠人记得，代码会持续向单体回归。**

## 2. 当前真实调用链和状态 owner

### 2.1 插件加载真实边界

```text
source/config
     │
     ▼
resolve_plugin_sources()          agent/plugins/source_resolver.py
     │  builtin: 仓库 plugins/ 目录；installed: plugin home cache
     ▼
PluginManager._import_plugin()    agent/plugins/manager.py
     │  FreshPluginImporter.register(module_path, plugin_root)
     ▼
FreshPluginImporter.find_spec()   agent/plugins/importer.py
     │  唯一强制：模块路径不得越出 plugin_root（_require_inside）
     ▼
apply(ctx, config)                插件模块
```

**已确认事实**：`FreshPluginImporter` 只做路径越界检查，没有 import 白名单、没有
meta path 拦截、没有命名空间隔离。因此「插件」目前只描述加载方式，不描述依赖边界。

**已确认事实**：插件内部写 `from plugins.tools.api import ...` 时，走的是常规 import
系统解析仓库根的 `plugins/` 命名空间包，**不经过** `FreshPluginImporter`。所以插件之间
的依赖不经过任何加载期校验。

**已确认事实**：`ServiceKey` 是 `@dataclass(frozen=True, slots=True)`，字段只有 `name`
（`agent/plugin_composition/model.py`）。相等与 hash 只按名字，因此双方各自声明同名 key
是成立且零运行时成本的；这也意味着 ServiceKey 的定义位置移动是纯文本操作。

### 2.2 正交性的两个结构缺口

**缺口一：Turn 执行没有原子接口。** 四个来源（`plugins/reply`、`plugins/scheduler`、
`plugins/subagent`、`plugins/plugin_update`）各自调用 `plugins/conversation/program.py::run_reply`，
并各自组装 `ChatModels`、内容检查器、工具菜单、材料绑定与取消检查。跨插件的
`REPLY_PROGRAM` key（`plugins/reply/api.py`）存在的唯一理由就是让别人能调到这个函数。

**缺口二：Core 侧能力不是原子。** 37 个 Core-owned `ServiceKey` 中，
`MESSAGE_CATALOG`/`MESSAGE_WRITERS`/`SESSION_ADMISSION` 等已提供窄服务，但消费者同时
继续深 import 同一事实的 DTO（`session.message` 被 82 个插件文件、172 处引用）。

## 3. 已确认事实、推断和未知边界

**已确认（本设计据此立项）**

1. `ServiceKey` 按名相等，所以 key 归位与词汇表搬移是零运行时风险的文本操作。
2. 词汇表本体 `session/message.py` 不 import 任何项目内模块，可独立成为公开合同。
3. Core 侧的 `provide` 站点集中在 `PluginManager._provide_composition_services`，按
   `inject` 按需提供——Core 确实不认识具体插件名，这一点已经正确。
4. 三个被文档承诺的名字在 Python 代码中不存在。

**推断（需要维护者确认后才可升级为需求）**

1. `Message` 的公开位置应由 Core 侧结构合同模块拥有，而不是 `session/`。依据是
   DeepSeek Harness 把 `Message` 定义在 `llm/llm`（能力包）而非 `core/session`。
   本设计按此推断实施**词汇表位置**，但不改变任何持久语义。
2. `core.timers`、`core.credentials`、`core.channels`、`models.*` 属于 `seam` 角色
   （可替换实现），其余多为 `core` 角色。角色判定的依据是「是否存在第二种实现或
   结构合同 + candidate stub」。

**未知边界**

1. 压缩投影是否需要引入 surface replacement（DeepSeek Harness 的
   `SurfaceOp.replace`）以在日志内表达「替换区间」。本设计不触碰该问题，
   也未核对现有 `plugins/compaction/` 实现的全部细节。
2. Core 自建的 Telegram/QQ Channel（`bootstrap/channels.py`）与 v3 `CHANNELS` 的
   收敛方案。本设计只把它登记为 seam 角色，不决定删除顺序。
3. 对外部插件仓库（`akashic-plugin/*`）的迁移顺序与 Gate 归属。本门只扫描本仓库
   跟踪文件，外部仓库的同类检查由各自仓库负责。

## 4. 目标结构和权限边界

### 4.1 分层

```text
┌──────────────────────────────────────────────────────────────┐
│ 组合内核   agent/plugin_composition/                         │
│   ServiceKey / Context / Fiber / Effect / typed event /      │
│   generation / lease / candidate。不含任何业务词。            │
├──────────────────────────────────────────────────────────────┤
│ 结构合同   agent/plugin_contracts/                           │
│   Message、内容块、调用引用、终态值域、seam Protocol。         │
│   只定义不可变值与 Protocol，不 import 实现。                 │
├──────────────────────────────────────────────────────────────┤
│ 内置原子能力                                                  │
│   Session 存储、Turn 执行、Prompt 组装、Tool 管线、           │
│   Model、Delivery、Timer。以 core / seam / bundle 三种角色    │
│   登记在 plugin_boundary.toml。                               │
├──────────────────────────────────────────────────────────────┤
│ 业务插件   plugins/**                                        │
│   Markdown Memory、Akasha、Scheduler、Subagent、Wake、Drift。 │
│   只依赖前两层。                                             │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 能力角色

| 角色 | 判据 | 当前例子 |
|---|---|---|
| `core` | 独占权威事实或注册表，不提供第二种实现 | `core.message_writers`、`core.tool_catalog`、`core.tasks` |
| `seam` | 结构合同 + 可替换实现；消费者只依赖合同 | `models.drivers.v1`、`core.channels`、`core.timers` |
| `bundle` | 只随产品发布一份实现，且扩展插件禁止依赖本包 | 待第 3 步为 Turn 原子登记 |
| `claim` | 单 owner 占用声明，不是可注入服务 | `plugin.claim.embedding_memory` |

`bundle` 是当前缺失的角色。它的用途是固定一条规则：**扩展插件依赖 typed event 和更窄的
Service，不依赖具体执行实现。** 没有这条规则，`run_reply` 被四处复制的问题会复发。

### 4.3 边界规则

| 规则 | 内容 | 基线 |
|---|---|---|
| R1 | Core **运行时**不得 import `plugins.*`（范围与迁移豁免见决策 0064 决定 6） | 8 条，只许减少 |
| R2 | `plugins/**` 只能 import `agent.plugin_composition`、`agent.plugin_contracts` | 244 处，只许减少 |
| R3 | 插件之间只能经公开结构合同连接，不得 import 对方实现模块 | 238 处，只许减少 |
| R4 | 每个 Core-owned `ServiceKey` 必须在策略表登记角色 | 无基线，立即全绿 |
| R5 | 已记录为「文档承诺、代码未实现」的名字必须保持不存在 | 无基线，立即全绿 |

R1～R3 是既有欠账，用债务账本 `plugin_boundary_baseline.toml` 精确登记，**只允许减少**；
R4、R5 没有基线，因此新增 Core 能力必须同时登记角色，实现幽灵名必须同时更新文档。

## 5. 失败、取消、并发、迁移和回滚

- **失败语义**：`python scripts/plugin_boundary.py check` 违规时输出每条
  `<rule>: <importer> imports <module>` 并以退出码 1 结束。新增违规、未登记角色的
  Core ServiceKey、幽灵名出现、账本条目已还清，四类都失败。
- **取消与并发**：本门是只读静态检查，不持有锁、不写仓库、不访问网络。
  唯一的写操作是 `baseline` 子命令，由人工显式调用。
- **迁移**：本设计不改变任何持久状态。受保护对象逐项不变——`sessions.db/messages`
  的只追加合同、`turns` 状态机、`memory` 目录、plugin-data、proactive/Wake/Drift 数据库、
  外部效果提交语义。本 PR 只新增文件与一处 `session/message.py` 改为再导出。
- **回滚**：按 commit 反向回滚。词汇表再导出保证了旧导入路径继续可用，因此回滚
  词汇表位置不需要同时回滚任何调用点。

## 6. 分阶段实施方案

整体三步，每步一个 PR，逐步叠加（stacked）。

### 第 1 步 · 地基（本设计批准的范围）

1. 公开结构合同模块 `agent/plugin_contracts/`，承载消息词汇表；
   `session/message.py` 保留为再导出入口，既有调用点零改动。
2. 能力角色单一真源 `plugin_boundary.toml`，登记 37 个 Core-owned ServiceKey
   与 3 个幽灵名。
3. 边界门 `scripts/plugin_boundary.py` 与债务账本 `plugin_boundary_baseline.toml`。
4. 边界门测试 `tests/test_plugin_boundary.py`，含端到端拒绝与账本陈旧检测。
5. 本设计与决策记录；更新 `INDEX.md`。

### 第 2 步 · 删死代码（需独立授权）

**2026-09-10 勘误。** 本文初版写「删除不可达的 `agent/tools/` 副本（`RuntimeSnapshot.tool_registry`
只声明不赋值，生产调用方从不传入）」，该论断**错误**，已作废。正确的可达性事实如下。

`snapshot.tool_registry` 在生产路径被赋值：`agent/plugins/manager.py` 的
`_compile_snapshot_tools` 编译它并挂到每个 `RuntimeSnapshot`（赋值点见
`_refresh_composition_runtime_tools`、`_compile_generation_snapshot` 与 reload 分支），
读取方包括 `infra/mobile_realtime/runtime_inspection.py` 与 `agent/tools/registry.py` 自身。
按模块路径 `agent.tools.<name>` / `agent/tools/<name>.py` 静态扫描，20 个被跟踪文件的真实分布是：

| 状态 | 模块 | 证据 |
|---|---|---|
| 活代码，必须保留 | `base.py`、`filesystem.py`、`registry.py`、`search_backend.py`、`shell_command.py`、`shell_security.py`、`unified_exec.py`、`events.py` | 有生产 importer；`host_bridge/`、`plugins/standard_tools/`、`agent/skills.py` 等 |
| 无代码消费者，删除候选 | `forget_memory.py`、`memorize.py`、`message_lookup.py`、`message_push.py`、`recall_memory.py`、`skill_loader.py`、`tool_search.py`、`vision.py`、`web_fetch.py`、`web_search.py`、`shell.py` | 零生产 importer；仅出现在 `impact.toml`、设计文档或本门账本中 |
| 仅测试消费者 | `executor.py`（`tests/test_tool_executor.py`）、随之受影响的 `events.py` | 删除需要同时处置测试与 `events.py` 的归属 |

其中 `shell.py`、`web_fetch.py`、`web_search.py` 同时是 R1 违规来源（它们 import 插件），
删除它们会让 R1 减少 3 条，是第 2 步与第 3 步的天然交界。

因此第 2 步不是「整片删除」，而是**逐项可达性审计 + 目录登记联动**：

1. 对每个候选确认「无生产 importer + 无动态/字符串入口 + 无测试依赖」；有任一消费者即保留。
2. 删除时必须同步修改 `tests_scenarios/contracts/impact.toml` 的路径列表，并更新
   `coverage-baseline.json` 的 `catalogDigest`（catalog 变更会使 digest 失效，这是设计如此）。
3. 清理设计文档中指向被删模块的引用；确实需要预留的，按 `plugin_boundary.toml` 的幽灵名规则写明理由。

**不在仓库范围内。** 起初列出的「只剩 `__pycache__` 的插件目录」与 `plugin_packages/` 经核实
是某个工作 checkout 中的**未跟踪本地残留**，不是仓库内容（仓库实际跟踪 37 个插件目录，
`git ls-files plugin_packages` 为 0）。它们不属于任何仓库 PR，也不得被当作已确认的可删除对象。

**2026-09-10 实施结果（stacked 第 2 步）。** 11 个「无代码消费者」模块已按上述流程删除：
`forget_memory.py`、`memorize.py`、`message_lookup.py`、`message_push.py`、`recall_memory.py`、
`skill_loader.py`、`tool_search.py`、`vision.py`、`web_fetch.py`、`web_search.py`、`shell.py`。
`impact.toml` 中 7 个组的对应路径已从 `paths` 移入 `deleted_paths`，`coverage-baseline.json`
的 `catalogDigest` 已重算；R1 由 28 降到 25，`plugin_boundary_baseline.toml` 同步删除 3 条已还清条目。

`executor.py` 与 `events.py` 本次**保留**：`events.py` 是 R10
[插件 Tool 组合事件任务合同](plugin-tool-composition-events-task-contract.md) 声明 owner 的
reviewed 公开 seam，`executor.py` 由 `tests/test_tool_executor.py` 的 11 项合同测试覆盖，符合本步
「有任一消费者即保留」规则。两者是否退场、以及 seam 是否必须迁入 `agent/plugin_contracts/`
（R2 目前不允许插件 import `agent.tools.events`），属于独立归属决定，留到第 3 步。

**2026-09-10 R1 范围裁决（第 3 步前置）。** 第 3 步开工前先把 R1 的文件范围写死，
否则「降到 0」无法验收。裁决与理由见
[决策 0064 决定 6](../decisions/0064-plugin-boundary-is-machine-enforced.md#6-r1-范围裁决与迁移豁免2026-09-10-补充)：

- `docker/`、`scripts/` 排除（复用 `measure_production_sloc.py` 的生产源码定义；`docker/`
  的多数命中是探针里生成插件源码的字符串字面量，不是 import）。
- `migrations/**` 与 `agent/migrations/**` 作为**迁移 payload 结构性豁免**，
  登记在 `plugin_boundary.toml` 的 `[R1_exemptions]`，每次 `check` 打印命中条数。
- `migrations/` 下四个非 yoyo 历史脚本目录按第 2 步流程删除。

豁免后 R1 由 25 降到 8。**关键后果**：终点验收第 1 条「清空 `plugins/` 后 Core 仍能启动」
**不可达**——yoyo 的 `to_apply()` 会 import 每个迁移模块解析 `__depends__`（已实测），
所以从零安装的 workspace 必须能 import 迁移引用的插件。诚实的验收表述改为：
「迁移账本已应用到当前版本后，移除 `plugins/` 不影响 runtime 启动；从零安装的 workspace
仍需要迁移所引用的插件在场。」此条必须在第 3 步收尾时同步更新，不得放宽门伪造绿色。

### 第 3 步 · 机械迁移（需独立授权）

**执行进度（2026-09-10，stacked PR 第 3 层）**

| 批次 | 内容 | R1 | R2 | R3 |
|---|---|---|---|---|
| 起点（第 2 步后） | — | 25 | 244 | 238 |
| 1/6 | R1 范围裁决 + 迁移豁免（决策 0064 决定 6） | 8 | 244 | 238 |
| 2/6 | 删除四个非 Yoyo 遗留迁移目录 | 8 | 244 | 238 |
| 3/6 | `session.message` 76 处改经 `agent.plugin_contracts` | 8 | 168 | 238 |
| 4/6 | `session.message_codec` 移入 `agent.plugin_contracts`（move & re-export） | 8 | 138 | 238 |
| 5/6 | 修掉 `types` 假违规（标准库被当成 core 深路径） | 8 | 133 | 238 |
| 5b/6 | 修复 `message_codec` 再导出漏掉私有 `_unique_fields` | 8 | 133 | 238 |
| 6/6 | 重启 seam 归位：`RESTART_GATE` → 组合内核，类型 → 结构合同 | 8 | 122 | 238 |
| 7/8 | 纯值级模块移入结构合同（`turn_effects` / `timekit` / `llm_json`） | 8 | 116 | 238 |
| 8/8 | `session.artifacts` 附件值词汇表移入结构合同 | 8 | 110 | 238 |

第 3/6 批的做法：第 1 步已经把消息词汇表移入 `agent.plugin_contracts`、
`session/message.py` 只做再导出，因此本批是纯文本改写
`from session.message import X` → `from agent.plugin_contracts import X`（76 文件 78 处），
导出对象身份由 `session.message.Message is agent.plugin_contracts.Message` 守护，
不改任何运行时语义。

第 4/6 批的做法：`session/message_codec.py` 只依赖消息词汇表与标准库（纯编解码，无存储，
无 bootstrap），因此按第 1 步 `message.py` 的同一形态 **move & re-export**：
实现移到 `agent/plugin_contracts/message_codec.py`，旧路径保留为再导出，导出对象身份不变。
不是所有 `session.*` 都能这样处理——`session.log` 的 50 处 import 全是
`MessageReader`/`MessageCatalog`/`OwnerStore` 等**存储实现类**的运行时导入，
把它们塞进结构合同会让合同层依赖存储，违反本层职责；正确做法是让插件经
`core.message_catalog` / `core.message_writers` / `core.owner_state` 消费，
属于独立批次。

第 5/6 批修正了门自身的一个假违规：`CORE_TOP_LEVELS` 曾把 `types` 当成 core 顶层，
但仓库的 `types/` 只有 `assets.d.ts`、没有任何 `.py`，插件的
`from types import MappingProxyType` 是**标准库**用法。这类假违规会让账本虚高、
让「只许减少」的账本失去意义，必须修掉而不是留在账本里。新增
`test_core_top_levels_do_not_shadow_stdlib` 守护：将来若有与标准库同名的顶层要登记为
core，必须先有真实 `__init__.py`。

**move & re-export 的强制检查（第 4/6 批的真实教训）。** 该批首次提交时漏掉了私有 helper
`_unique_fields` —— 不可变的 yoyo 迁移 `20260907_03_message_metadata.py` 直接 import 它，
`from x import _name` 不受 `__all__` 限制但名字必须存在。结果 change-impact Gate 的
`model_owner_contract` 场景 17 项失败。修复后 Gate 目标命令由 `17 failed / 58 passed`
变为 `75 passed`。

因此后续任何 move & re-export 都必须：**按全库实际被 import 的名字集合导出**（含私有名），
而不是按原模块的公开 API 或 `__all__`。这一步不能省，因为不可变迁移会依赖私有名。

第 6/6 批执行设计文档原定的第 4 项（`core.restart_gate.v1` 归位）：

- `RESTART_GATE = ServiceKey[RestartGate]("core.restart_gate.v1")` 从 `agent.restart`
  移到 `agent/plugin_composition/restart.py`，并由组合内核包再导出。`ServiceKey` 只按
  name 相等，归位是零运行时语义的文本操作。
- `RestartRejectedError` / `RestartPendingError` / `ExternalRootPermit` 移入
  `agent/plugin_contracts/restart.py`；`ExternalRootPermit` 是真实的 frozen dataclass
  （不是 Protocol），因为它带行为（`release()`/`child()`），插件会实际调用。
- `RestartGate` 有状态机与 commit channel，实现留在 `agent.restart`；合同层用
  Protocol 描述插件可见的方法子集（`accepting`/`check_open`/`acquire`/`prepare`/
  `commit`/`abort`/`wait_until_open` + `supervised`/`execution_enabled`）。
- `agent.restart` 按原路径再导出全部名字，既有 Core 调用点与类型身份不变。

**迁移判据（第 3 步执行中固化，后续批次必须照此分类）。** 每个 R2/R3 目标模块先过三问，
再选动作；不允许「搬得动就搬」：

| 判据 | 动作 |
|---|---|
| 零仓库内 import、无状态、无 I/O（纯值词汇 / 枚举 / 纯函数 / 值模型） | **move 到 `agent/plugin_contracts/`** + 旧路径再导出 |
| 有状态、持有注册表或权威事实、需要 I/O 或环境态 | **登记为 `ServiceKey`**（core/seam 角色），消费者经 `ctx.require` |
| 描述可替换实现的接口（有第二种实现或结构合同 + stub） | **seam**：合同层放 Protocol，实现留在原处 |

已确认**不可 move** 的例子（留待 ServiceKey 批次）：`session.log`（存储实现类）、
`session.embedding_store`（存储）、`core.common.diagnostic_log`（contextvars + 日志配置）、
`core.error_context`（contextvar 环境态）、`agent.control.context`（铸造 capability）、
`agent.plugins.snapshot`（运行时全局）。

**move 的强制前置动作**：先跑一次全库名字扫描，列出该模块被 import 的**全部名字（含私有名、
含 Core 内部再导出名）**，再按该清单写再导出。第 5b 批（`_unique_fields`）与第 8 批
（`AttachmentReadLease`/`AttachmentReadPort`）都是没做这一步而漏名，前者被 change-impact Gate
以 17 项失败挡下。drop-in 的判据是：`python -c "import <所有消费者模块>"` 全部成功。

**R2 剩余条目的逐条分类（2026-09-10 第 25 批后，R2=23）。** 每一项都已定性，接手者不需要重新调查：

| 目标 | 条数 | 性质与做法 |
|---|---|---|
| `agent.plugins.snapshot` | 7 | **设计原定第 5 项**：隐式全局改注入 Service。需要新增 ServiceKey + provider + 改写 7 处调用点，并验证 generation 切换语义，属独立批次 |
| `core.net.http` | 6 | **真 seam，需要设计**：模型 driver 与 web 工具直接 `HttpClient()`/`RetryPolicy()`/`RequestBudget()`。要解耦必须让 HTTP requester 可注入（离线/测试可控），而不是把 httpx 实现搬进合同层 |
| `agent.persona` | 2 | `read_veda_file` 读产品文本、`AKASHIC_BEHAVIOR_RULES` 常量。owner 是 Core 产品语义，应由 prompt 组装的服务面提供，不是词汇搬迁 |
| `agent.skills` / `agent.plugins.archive` | 2 | `SkillRecord`/`skill_body` 可进合同层；`PluginArchive` 是 Core 插件归档存储，应经 seam 提供 |
| `agent.tools.filesystem` | 1 | 文件操作值对象（`ReadFileOperation` 等），可进合同层，但需与 `_FileOperation` 基类一起切 |
| `agent.model_runtime.catalog.litellm_registry` | 1 | `resolve_catalog_capabilities` 是模型能力的解析入口，属 `models.*` seam 面 |
| `agent.control.context` | 1 | `mint_plugin_child_capability`/`running_turn_id` 是控制面能力铸造，属控制 seam |
| `infra.channels.telegram_utils` | 1 | `strip_chunk` 是 Telegram 文本分段，owner 在 Telegram 适配层，应先定 channel seam |
| `infra.channels.message_view` | 1 | `session_row` 是 Core 的 Session 展示投影，应由 Core 的展示服务提供 |
| `agent.tools.unified_exec` | 1 | **已登记并注明**：`shell_backend` 直接构造 local 后端，应改经 `core.processes` 消费；会改变进程记账与 owner key 生成路径，属行为变更，需单独差分 Gate |

**已判定「不搬」的通用理由（避免后来者重复调查）**：有状态、持注册表、需 I/O 或环境态的模块
一律不进合同层 —— `session.log`/`session.embedding_store`（权威存储）、`core.common.diagnostic_log`
（contextvars + logging 配置）、`core.error_context`（环境态 ContextVar）、`agent.control.context`
（能力铸造）、`agent.plugins.snapshot`（运行时全局）。它们改由**能力定义处**再导出（第 19/22/24 批）
或登记为 seam（第 9/10/11/17/18/23 批）。

把 R1～R3 的欠账降到 0：

1. `session.*` 深路径（172 处）改经结构合同。
2. 插件间实现 import（238 处）改经版本化 `ServiceKey` + 结构合同模块。
3. Core→插件反向链路（28 条）改为经 Service 消费，逐条决定 owner。
4. `RESTART_GATE`、`CONTROL_FRAMES` 两个错位 key 归位到组合内核。
5. `agent.plugins.snapshot` 从隐式全局改为注入 Service。

### 不在本设计范围

Turn 原子（`SCOPED_TURNS` 形态的 inbox + scope + 句柄）、压缩 surface replacement、
Channel 双栈收敛、外部插件仓库迁移。这些是不可机械化的重构，需要各自的差分 Gate。

## 7. 验收标准

**本 PR（第 1 步）**

1. `python scripts/plugin_boundary.py check` 退出码 0，并打印当前/基线数量。
2. 新增一条 Core→插件 import 时，门必须失败——已用真实变异验证。
3. 新增一个未登记角色的 Core `ServiceKey` 时，门必须失败——已用真实变异验证。
4. 让 `SCOPED_TURNS` 等幽灵名出现在代码中时，门必须失败。
5. 账本中已还清的条目必须使门失败，防止账本掩盖真实状态。
6. `session.message` 与 `agent.plugin_contracts` 导出同一对象（身份唯一）。
7. 既有测试全量通过，证明词汇表搬移无语义变化。

**终点（三步全部完成）**

1. 清空 `plugins/` 目录后 Core 仍能启动。**（2026-09-10 修订）** 由于不可变 yoyo 迁移
   在 `to_apply()` 阶段被 import，本条按「迁移账本已应用到当前版本后移除 `plugins/`
   不影响 runtime 启动」验收；从零安装的 workspace 仍需要迁移引用的插件在场。
   理由见决策 0064 决定 6。
2. 任一插件目录原样安装到外部 cache，不改 Core，admission 通过。
3. `plugins/**` 中不存在指向 Core 内部或兄弟插件实现的 import。
4. 新增一个来源插件不需要修改任何已有插件。
5. R1～R3 数量为 0，`plugin_boundary_baseline.toml` 清空。
6. 模型实际收到的输入能由 Message 日志重建，并有运行时 invariant 断言。

## 8. 停止条件

出现以下任一情况即停止当前步骤：

- 需要改变 `sessions.db` schema、消息保留规则或任何正式 workspace 数据。
- 需要按插件 ID 在 Core 中分支。
- 只能通过放宽门、删除账本条目或跳过测试获得绿色。
- 发现 `plugin_boundary.toml` 的角色判定与真实能力 owner 冲突，且无法用现有证据裁决。
