# 0064 · 插件边界由机器强制

- 状态：accepted
- 日期：2026-09-10
- 关联条款：PLG-001～PLG-017、GOV-001～GOV-005、TST-001～TST-008
- supersedes：无
- superseded by：无

## 背景

插件 v3 加载链已经能加载、隔离、晋升和回滚插件：candidate Root、snapshot lease、
generation drain 和 Effect 逆序清理都已实现，`PluginManager` 也按插件 `inject` 按需提供
Core 能力，因此 Core 本身不认识具体插件名。

但「插件」目前只描述加载方式，不描述依赖边界。真实证据：

- Core 直接 import 插件 28 条，例如 `bootstrap/app.py` 引用 `plugins.delivery.senders`、
  `infra/channels/message_view.py` 引用 `plugins.tools.api`；
- 插件 import Core 内部深路径 244 处（去重），主要是 `session.*`、`agent.restart`、
  `agent.tools.*`、`agent.plugins.snapshot`；
- 插件互相 import 实现模块 238 处（去重），最重的目标是 `plugins.tools`、`plugins.delivery`、
  `plugins.context`。

`FreshPluginImporter` 的唯一强制是模块路径不得越出 plugin_root，没有 import 白名单或
meta path 拦截；插件内部的 `plugins.<name>` import 甚至不经过该 importer，而由常规
import 系统解析仓库根的命名空间包。

这个状态无法自愈，并且已经产生可观察后果：Turn 执行没有原子接口，四个来源各自复制
`run_reply` 的 25 参数调用；多份设计文档与 skill 把 `SCOPED_TURNS`、`CONTINUATIONS`、
`BACKGROUND_JOBS` 当作现有积木，而这三个名字在所有 Python 文件中零命中。

参考实现 DeepSeek Harness 的对照结论是：它不靠 import lint 维持边界，靠的是
包依赖图受检、生成式能力表带完整性守卫、以及每个包可选的运行时 invariant 伴随。
其中「能力表 + 完整性守卫」与「文档由代码生成」两点直接针对本项目的问题。

## 决定

### 1. 边界必须可失败

新增 `scripts/plugin_boundary.py`，用 AST 静态解析（不导入目标模块），实现五条规则：

| 规则 | 内容 |
|---|---|
| R1 | Core 不得 import `plugins.*` |
| R2 | `plugins/**` 只能 import `agent.plugin_composition`、`agent.plugin_contracts` |
| R3 | 插件之间只能经公开结构合同连接，不得 import 对方实现模块 |
| R4 | 每个 Core-owned `ServiceKey` 必须在 `plugin_boundary.toml` 登记角色 |
| R5 | 已记录为「文档承诺、代码未实现」的名字必须保持不存在 |

R1 的作用范围与迁移豁免见下方决定 6。

### 2. 既有欠账用只减不增的账本

R1～R3 的既有违规精确登记在 `plugin_boundary_baseline.toml`，条目标识为
`<importer>|<imported module>`，不含行号。账本规则两条：新增违规使门失败；
条目已不再是违规时，门同样失败，强制删除。

R4、R5 不设基线。新增 Core 能力必须同时登记角色；实现幽灵名必须同时更新文档，
否则门失败。这条机制保证文档承诺不能超过代码事实。

### 3. 能力角色三分法

Core 拥有的能力在 `plugin_boundary.toml` 登记角色：

| 角色 | 判据 |
|---|---|
| `core` | 独占权威事实或注册表，不提供第二种实现 |
| `seam` | 结构合同 + 可替换实现；消费者只依赖合同 |
| `bundle` | 只随产品发布一份实现，且扩展插件禁止依赖本包 |
| `claim` | 单 owner 占用声明，不是可注入服务 |

`bundle` 是本决定新引入的角色。它固定一条规则：扩展插件依赖 typed event 和更窄的
Service，不依赖具体执行实现。这条规则用于阻止「一个执行实现被多个来源复制调用」
的问题复发。

### 4. 公开结构合同独立成层

新增 `agent/plugin_contracts/`，承载可被插件依赖的不可变值词汇表与 Protocol。
本层不 import 实现、存储或 bootstrap。消息词汇表从 `session/message.py` 迁入该层，
`session/message.py` 保留为再导出入口，既有调用点不因本次决定改变。

### 5. Core 与插件都不得依赖对方实现

Core 侧对具体插件的依赖改为经 Service 消费；插件侧对 Core 与兄弟插件的依赖改为经
结构合同或版本化 `ServiceKey`。`ServiceKey` 按 `name` 相等（`model.py` 中
`ServiceKey` 是只有 `name` 字段的 frozen dataclass），因此 key 归位与词汇表搬移是
零运行时语义的文本操作。

### 6. R1 范围裁决与迁移豁免（2026-09-10 补充）

**问题。** R1 原本只说「Core 不得 import `plugins.*`」，但没有定义「Core 侧」的文件范围。
`CORE_ROOTS` 只含 `agent/session/infra/core/bootstrap/bus/utils/mcp_servers` + `main.py`，
于是 `migrations/`、`docker/`、`scripts/` 是否纳入从未被写清。第 3 步的验收标准是
「R1 降到 0」，范围不定会让这个验收反复被质疑。

**裁决一：`docker/`、`scripts/` 排除。**
它们不是生产运行源码——`scripts/measure_production_sloc.py::is_production_source_path`
早已把两者判为 `False`，本决定复用同一份机器可读定义，不再新造第二份。
（`docker/` 另有 20 处 `plugins.*` 命中，其中一半是探针文件里**生成插件源码的字符串字面量**，
不是 import；纳入只会制造假违规。）

**裁决二：`migrations/**` 与 `agent/migrations/**` 从 R1 豁免，且是结构性豁免。**
R1 保护的是「具体插件可以整体移除，Core 运行时不受影响」。迁移不是运行时：它在 runtime
启动前一次性改写**某个插件拥有的**持久数据，必须使用该插件自己的 schema 与读写实现。
把 schema 复制进 Core 会让同一事实出现第二个 owner，违反本仓库的
「同一事实只有一个 owner」。三点事实支持无法通过迁移消除：

1. `migrations/yoyo/**` 已进入 main 的迁移不可修改、移动或删除
   （`scripts/check_yoyo_migrations.py` 强制），而 workspace 账本仍要求它们存在。
2. yoyo 的 `DatabaseBackend.to_apply()` 会 import 每个迁移模块来解析 `__depends__`
   （已实测：模块内 `import` 失败会让 `to_apply` 抛 `BadMigration`），
   所以**从零安装的 workspace 必须有对应插件在场**。
3. `agent/migrations/**` 是这些不可变迁移的 payload，只被迁移与迁移测试引用。

因此这是**豁免而非欠账**：不进「只许减少」的 `plugin_boundary_baseline.toml`，
而是登记在 `plugin_boundary.toml` 的 `[R1_exemptions]`，每次 `check` 都打印命中的
路径规则与条数，并有测试禁止「匹配不到任何文件」的僵尸豁免。

**必须一起知道的后果。** 因为第 2 点，终点验收第 1 条「清空 `plugins/` 目录后 Core 仍能启动」
**在保留不可变 yoyo 迁移的前提下不可达**。诚实的表述是：
「迁移账本已应用到当前版本后，移除 `plugins/` 不影响 runtime 启动；但从零安装的
workspace 仍需要迁移所引用的插件在场。」第 3 步收尾时必须按此更新验收标准，
不得用放宽门的方式伪造绿色。

**裁决三：`migrations/` 下的非 yoyo 历史脚本应删除。**
`migrations/README.md` 明说 `akasha_sparse_index_v8/`、`provider_runtimes_and_akasha/`、
`workspace_veda/`、`workspace_veda_uppercase/` 「不进入 Yoyo catalog，也不再承诺自动执行
或兼容当前 runtime」，只供调查旧实现。它们不属于豁免要保护的对象，按第 2 步的可达性
审计流程删除。

## 理由

**为什么用静态门而不是缩小 API。** 边界必须能在代码写入时判定，而不是在评审时靠人发现。
AST 静态检查不执行目标代码，没有运行副作用，可以在 CI 与本地以同一命令运行。

**为什么账本必须双向失败。** 单向「只查新增」会让账本漂移成永久豁免清单；要求删除
已还清条目，账本才始终等于真实欠账。

**为什么 R4、R5 不设基线。** 这两条没有历史包袱，可以立即全绿。把它们设为无基线，
就得到了「文档承诺 = 代码事实」的机器版本，直接消除了本项目当前最危险的一类漂移。

**为什么不照抄参考实现的包依赖图。** DeepSeek Harness 用 pnpm workspace 与
TypeScript project references 做物理边界，那是约 200 个包的发布矩阵所需。
本项目是单 Python 包、44 个插件目录、37 个 Core 能力，用能力表 + 静态门达到同等
效果，不需要物理拆包。

## 影响与回滚

- **不改变任何持久语义。** `sessions.db/messages` 的只追加合同、`turns` 状态机、
  `memory` 目录、plugin-data、proactive/Wake/Drift 数据库、外部效果提交语义逐项不变。
- **不改变任何运行时行为。** 静态门只读源码；词汇表搬迁保留再导出，导出对象身份不变。
- **新增负担**：新增 Core 能力必须登记角色；实现被文档承诺的名字必须同步更新文档。
  这是本决定的目的，不是副作用。
- **回滚**：按 commit 反向回滚。词汇表再导出保证回滚位置不需要同时回滚调用点。

## 验收

1. `python scripts/plugin_boundary.py check` 在真实仓库退出码 0 并打印当前/基线数量。
2. 注入一条 Core→插件 import 时门失败，并指出精确的 importer 与 module。
3. 新增未登记角色的 Core `ServiceKey` 时门失败。
4. 幽灵名出现在 Python 代码中时门失败。
5. 账本条目已还清时门失败。
6. `session.message` 与 `agent.plugin_contracts` 导出同一对象。
7. 既有测试全量通过，证明本次搬迁无语义变化。

终点验收（三步全部完成，需各自独立授权）见
[插件边界地基](../design/plugin-boundary-foundation.md#7-验收标准)。
