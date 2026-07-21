# Git 驱动的一次性兼容迁移

日期：2026-07-21  
状态：implemented  
固定基线：`012e37c8b51df045353972bb551d8e868ab52455`

## 1. 结论

Akashic Agent 使用固定 Git 提交作为迁移纪元，用一个原子写入的 Git cursor 表示当前配置实例已经成功处理到哪个源码提交。正常启动只读取当前 `HEAD` 和 cursor；两者相同就立即进入 runtime，不扫描迁移目录、不导入历史脚本，也不访问业务数据库。

兼容逻辑放在版本控制内的独立迁移 bundle 中。源码更新后，runner 只从 `cursor..HEAD` 的 Git 历史发现新增 bundle，按提交顺序执行，并在每个提交的全部迁移通过验证后推进 cursor。旧脚本保持不可变；需要修正时新增迁移，不把兼容分支持续堆进核心 runtime。

首次出现 migration state 时，选中的 `config.toml` 已存在就是旧安装的充分条件。没有配置但 workspace 已有权威或连续性状态也按旧安装处理。只有配置不存在且 workspace 没有持久状态时，才按当前代码直接初始化最新结构，并把 cursor 设为当前 `HEAD`，不回放历史迁移。

```text
┌──────────────────────┐
│ 读取 Git HEAD 与 cursor│
└──────────┬───────────┘
           │
     cursor == HEAD ────────────────┐
           │ 否                     │ 是
           ▼                        ▼
┌──────────────────────┐      ┌───────────┐
│ cursor 是否存在？      │      │ 启动 runtime│
└──────────┬───────────┘      └───────────┘
           │
       ┌───┴───────────────┐
       │                   │
      存在                不存在
       │                   │
       ▼                   ▼
 cursor..HEAD       config 或持久状态存在？
 顺序执行迁移          │
                 ┌─────┴─────┐
                 │           │
                存在        不存在
                 │           │
                 ▼           ▼
          cursor = baseline  最新结构初始化
          再执行迁移         cursor = HEAD
```

## 2. 用户意图与成功标准

### 2.1 用户意图

- 代码更新后，历史配置、派生数据库或其他兼容操作自动执行一次。
- 不维护递增的产品版本号，不要求用户手动修改 migration version。
- 稳态启动路径足够短，不随迁移脚本数量增长而变慢。
- 兼容代码位于迁移 bundle，不进入 provider、配置加载、AgentLoop 等核心 runtime。
- 迁移失败时 runtime 不启动，旧状态和恢复点仍然存在。
- 新 clone 的新用户直接得到最新结构，不执行仅为旧状态准备的脚本。

### 2.2 成功标准

1. `cursor == HEAD` 时只做一次 Git revision 读取、一次小文件读取和一次比较。
2. 从任意受支持的 baseline 之前版本更新时，首批 adoption migration 能把旧状态收敛到当前结构。
3. 从 baseline 之后跳过多个提交更新时，只按 Git 顺序执行尚未处理的迁移。
4. 每个迁移最多产生一次有效状态变化；崩溃后重试由 `assess` 和 `verify` 识别已完成效果。
5. 迁移 apply 或 verify 失败时 cursor 不越过失败提交，runtime 保持停止。
6. 配置、权威 workspace 数据、派生数据库和 secret 备份均遵守各自 owner 与恢复协议。

## 3. 当前事实与约束

- 当前初始化入口 `bootstrap/init_workspace.py::init_workspace` 会先复制 `config.example.toml` 为实际配置，再加载配置并创建 workspace 文件和数据库。因此 fresh/legacy 判定必须发生在调用该入口之前。
- 主配置可以由 `--config` 指向任意路径，不保证位于 workspace。workspace 由 CLI、环境变量或主配置选择，也不能用 Git checkout 代替。
- `sessions.db/messages`、长期记忆和 plugin-data 属于权威或连续性状态；普通代码更新无权删除或覆盖。
- `akasha.db` 是可重建派生状态，但重建仍须备份、完整性检查、staging 构建和原子发布。
- 配置可能包含 API key。配置备份及迁移临时文件必须使用限制权限，日志不得输出 secret 内容。
- `WSP-003` 要求迁移离线、持锁并原子发布；`BAK-001` 要求备份可验证、可恢复。

## 4. 方案比较与选择

### 4.1 选择：固定 Git baseline、单 cursor、Git 历史发现

固定当前 `main@012e37c8` 为迁移纪元。baseline 只建立一次，不随发布递增。每次源码变化后用 `git rev-list` 查询 `cursor..HEAD` 中触碰迁移根的提交，并从这些提交发现首次加入的 bundle。

优点是稳态成本恒定，执行顺序来自仓库历史，兼容代码可以追加而不进入 runtime。代价是 legacy migration 需要完整包含 baseline 到当前提交的 Git 历史；浅克隆或非 Git 发行物需要显式处理。

### 4.2 未选择：扫描全部迁移并维护已执行 ID 集合

这是 Alembic、Django 等常见迁移框架的形态，但每次启动都要加载 catalog，状态会随脚本数量增长。即使可以优化，它也不符合本项目“HEAD 未变化时只做常数次比较”的目标。

### 4.3 未选择：使用 `ORIG_HEAD`、reflog 或用户原始提交推断起点

reflog 会过期，新 clone 没有旧 checkout 的 reflog，用户后续 Git 操作也会改变 `ORIG_HEAD`。迁移框架出现之前又没有应用成功记录，因此原始 checkout 提交不能证明某个迁移已经执行。所有 pre-framework 安装统一收敛到固定 baseline 更安全。

### 4.4 未选择：把 cursor 写进 `config.toml`

主配置是用户管理的业务配置，也是首批迁移的修改目标。把执行状态写入其中会让配置复制、格式化、回滚和 secret 恢复同时改变 migration state。设计改用配置旁的独立 sidecar cursor；配置只参与首次安装分类。

## 5. 权威组件与文件边界

```text
Git repository
├── migrations/.root
│   └── 固定 baseline SHA 与迁移根身份
├── migrations/<bundle>/migration.py
│   └── assess / apply / verify / revert
└── 核心 migration runner
    └── revision 比较、Git catalog、锁、执行、cursor 提交

选中的 config 路径
├── config.toml
├── config.toml.migration-cursor
├── config.toml.migration-lock
└── config.toml.migration-backups/

Akashic workspace
└── 会话、记忆、派生数据库及其他运行状态
```

`migrations/.root` 是追加式迁移历史的根，只保存固定 baseline 和格式身份；它不是递增版本文件。首版明确记录：

```text
012e37c8b51df045353972bb551d8e868ab52455
```

`config.toml.migration-cursor` 只保存最后成功处理的完整 Git commit SHA 和换行。它不保存脚本列表、产品版本、时间线或运行日志。cursor 使用临时文件、fsync 和原子 replace 发布。

cursor 是配置实例的 companion control state：

- 正常增加：首次分类或全新初始化成功时创建。
- 允许原位更新：一个 Git 提交的全部迁移 verify 成功后，推进到该提交；范围内没有迁移时最终推进到 `HEAD`。
- 逻辑失效：配置与 sidecar 被移动或只恢复其中一方时，旧 cursor 不再代表新的组合。
- 物理减少：自动启动不得删除；只有显式重置 migration state 的管理操作可以删除，并须先备份。
- 恢复证据：cursor、配置、workspace 与迁移备份 manifest 共同证明迁移前后状态。

选择 sidecar 而不是 SQLite，是因为运行路径只有一个 SHA cursor。崩溃恢复依靠 bundle 的 `assess`/`verify`，不需要长期保存每个迁移的 receipt。

配置与 cursor 是同一个安装身份的 companion state，备份和恢复必须同时处理。只在原路径覆盖旧配置、却保留较新的 cursor，无法仅靠 Git SHA 自动识别；这种不完整恢复必须通过显式 `migration adopt` 把 cursor 退回固定 baseline，再按正常迁移和 verify 收敛。自动启动不得因任意配置解析错误就擅自回退 cursor，否则普通拼写错误会触发高权限迁移。

## 6. 首次安装分类

分类发生在 `init_workspace` 创建任何文件之前，只检查用户实际选中的配置路径，不把仓库内的 `config.example.toml` 算作安装状态。

### 6.1 legacy adoption

cursor 不存在，且满足下列任一条件时，先原子写入固定 baseline，再执行 `baseline..HEAD`：

1. 选中的 `config.toml` 已存在。
2. 选中的 workspace 已存在受保护或连续性状态，例如 `sessions.db`、`memory/` 中的长期状态、`proactive.db`、`wake_proactive.db`、`drift/drift.db`、`plugin-data/` 或非空调度状态。

第一条是旧安装的充分条件。第二条防止配置丢失、路径移动或部分初始化时把已有 workspace 误判成新用户。仅有空目录、lock、PID、socket 或 readiness 文件不构成旧业务状态。

如果配置存在但已经是当前格式，迁移的 `assess` 返回 `SATISFIED`，runner 验证后推进 cursor，不重复改写。如果配置或 workspace 处于无法安全识别的部分状态，`assess` 返回 `BLOCKED`，runtime 不启动。

### 6.2 fresh initialization

cursor、配置和 durable workspace state 都不存在时：

1. 使用当前代码直接初始化最新配置和 workspace 结构。
2. 完成配置可加载、SQLite integrity check 和必要文件检查。
3. 将 cursor 直接写为当前 `HEAD`。
4. 启动 runtime，不执行任何历史迁移。

当前初始化在中途崩溃时可能已经创建配置。下一次启动会保守地进入 legacy adoption；迁移必须通过 `assess`/`verify` 把已经满足的状态判为 no-op，而不是重复产生效果。

### 6.3 典型场景

| 场景 | 结果 |
|---|---|
| 新 clone、无配置、空 workspace | 直接初始化最新结构，cursor=`HEAD` |
| 新 clone、复用旧 `config.toml` | cursor=baseline，自动迁移 |
| 新 clone、通过 `--workspace` 连接旧数据 | cursor=baseline，自动迁移 |
| 老 checkout 从 baseline 前更新 | cursor=baseline，自动迁移 |
| 配置已手工改成新格式但无 cursor | verify 为 satisfied，只推进 cursor |
| 配置缺失但 workspace 有会话或记忆 | 不判 fresh；进入 adoption 或明确 blocked |
| 配置和 cursor 一起复制到同一 `HEAD` | 快速路径；后续配置加载仍负责业务校验 |

## 7. Git catalog 与顺序

### 7.1 稳态快速路径

```text
current = git rev-parse HEAD
cursor = read(config_path + ".migration-cursor")

if cursor == current:
    start runtime
```

这条路径不得遍历 `migrations/`、计算 bundle hash、导入脚本、打开 workspace 数据库或检查远端 Git。

### 7.2 HEAD 变化路径

runner 验证 baseline、cursor 和 `HEAD` 的祖先关系，然后查询：

```text
git rev-list --reverse --first-parent <cursor>..<HEAD> -- migrations/
```

对每个提交，以其第一父提交为基准，只执行该提交首次加入的 bundle；因此普通提交、squash commit 和 merge commit 都能发现新增脚本。普通更新不允许修改、移动或删除既有 bundle；CI 对 `migrations/` 使用 append-only 检查。需要修正旧迁移时，在新提交增加 correction bundle。

同一个主分支提交最多应加入一个 bundle。若确需加入多个，按 bundle 相对路径字节序执行；其中任一失败时该提交的 cursor 不推进。一个 bundle 内可以用编号阶段组织多个相互依赖的兼容动作，因此 GitHub squash merge 不会丢失 PR 内部操作顺序。

范围内的迁移全部完成后，即使后续提交没有迁移，cursor 也推进到当前 `HEAD`。这样下一次启动仍走常数时间快速路径。

### 7.3 分支、降级和历史不足

- cursor 是 `HEAD` 的祖先：正常向前迁移。
- cursor 与 `HEAD` 分叉：自动启动 fail-loud，不猜测两个分支的迁移等价性。
- `HEAD` 位于 cursor 之前：视为降级，自动启动不调用 revert。
- legacy adoption 的浅克隆缺少 baseline 或中间提交：阻止启动并给出补齐 Git 历史的明确操作；不得把当前目录扫描结果冒充完整历史。
- fresh initialization 不需要读取 baseline 历史，因为它直接写入 `cursor=HEAD`。

第一版支持正常 Git checkout。`git archive`、wheel 或不含 `.git` 的镜像需要以后增加构建时注入的 revision 与不可变 migration catalog，不在本设计中静默 fallback。

## 8. Migration bundle 合同

每个 bundle 是独立目录，至少包含一个可执行的 `migration.py`。runner 通过固定子进程协议调用，不动态导入脚本到核心进程：

```text
assess  → NEEDED | SATISFIED | BLOCKED
apply   → 在 staging 或 candidate 上产生变化
verify  → 独立读取发布结果，成功或非零失败
revert  → 显式人工回滚时恢复备份或执行逆操作
```

runner 向脚本提供只读 context：完整 config path、workspace path、baseline、migration commit、current `HEAD`、唯一 staging 目录和备份目录。脚本不得自行推断 HOME、正式 workspace 或其他配置。

bundle 必须满足：

1. `assess` 只读，能区分已满足、需要执行和无法安全识别。
2. `apply` 在持锁、备份完成后执行；权威文件先写 candidate，SQLite 使用 backup API 或独立 staging DB。
3. `verify` 不复用 apply 的内存成功值，从文件、数据库或应用加载边界重新观察结果。
4. 重试时 `assess` 能识别“效果已发布但 cursor 尚未推进”，随后 verify 并提交 cursor。
5. `revert` 只由显式管理命令调用，自动检测降级时不得执行。
6. 脚本不导入易变的 AgentLoop、provider 或 UI 内部实现；通用能力限于标准库和一个小型、稳定的 migration context/toolkit。
7. stdout/stderr 不输出 secret、消息正文或完整用户配置。

退出码和状态是唯一成功边界。依赖缺失、配置损坏、命令失败和完整性检查失败必须非零退出，不允许空返回或假成功。

## 9. 执行、锁与提交顺序

```text
1. 解析 config/workspace 身份
2. 获得 config migration lock 与 workspace 单实例锁
3. 分类 fresh / legacy，必要时建立 baseline cursor
4. 验证 Git ancestry 并生成有序执行计划
5. 对每个 migration commit：
   5.1 assess
   5.2 为将改变的对象创建备份与 manifest
   5.3 apply 到 staging/candidate
   5.4 原子发布
   5.5 verify 发布后的真实状态
   5.6 该提交全部 bundle 成功后推进 cursor
6. 无迁移的剩余提交处理完后推进 cursor 到 HEAD
7. 释放锁
8. 重新加载最终配置并启动 runtime
```

runner 在 runtime 初始化和 provider 构造之前运行。配置迁移完成后必须从磁盘重新加载，不能继续使用迁移前解析出的对象。

并发启动时只有一个进程可以成为 migration owner。其他进程不得同时运行脚本或启动 runtime；它们应明确报告已有迁移 owner，而不是绕过迁移继续启动。

## 10. 持久状态、备份和删除边界

| 对象 | 允许变化 | 禁止变化 | 恢复证据 |
|---|---|---|---|
| `config.toml` | 经 bundle 声明的字段映射，原子替换，保留未迁移字段 | 静默删除未知字段、泄露 API key、用模板覆盖旧配置 | 权限受限的原文件备份、前后结构摘要、最终 Config load |
| cursor sidecar | 首次创建；成功后原子推进 SHA | verify 前越过提交、失败时写成 `HEAD`、自动删除 | cursor 文件及 Git ancestry |
| `sessions.db`/消息 | 本 bundle 不读写 | 任何 UPDATE/DELETE 或派生重建 | 迁移前后文件摘要 |
| `akasha.db` | 本 bundle 不读写 | 备份、重建、marker、替换或删除 | 迁移前后文件摘要 |
| migration backups | 每次实际 apply 前增加唯一快照 | 自动按启动次数或年龄删除 | manifest、hash、SQLite integrity check、隔离恢复 smoke |

备份目录和包含 secret 的临时文件使用目录模式 `0700`、文件模式 `0600`。自动迁移不 prune 历史备份；清理由以后独立、显式的数据管理操作负责。

## 11. 首批 migration bundle

首个引入框架的提交相对于固定 baseline 提供一个 adoption bundle，按内部阶段处理本轮已经确认的兼容变化。

### 11.1 模型 runtime 配置迁移

- 识别当前 legacy `[llm] provider`、`[llm.main]`、`[llm.fast]`、`[llm.vl]` 结构。
- 映射到命名 runtime 与 main/fast/vl 角色引用，保留现有 provider、model、base URL、API key/auth、上下文窗口、多模态和 tool-call 能力字段。
- 已经是新结构时返回 `SATISFIED`。
- legacy 与新结构混杂且无法确定优先级时返回 `BLOCKED`，不猜测覆盖。
- 发布后使用当前配置边界重新加载，并证明角色都能解析到预期 runtime。

Akasha 派生库与 provider runtime 配置没有同一个不变量，不属于本 bundle。需要重建时必须由独立、显式的数据管理操作发起，不得阻塞常规 runtime 启动。

## 12. 性能模型

稳态启动成本与迁移数量无关：

```text
T_stable = git_rev_parse + read_small_cursor + compare
```

只有 `HEAD` 变化或 cursor 不存在时才支付 Git range 查询、bundle 启动、备份和数据迁移成本。没有迁移的代码更新完成一次范围查询后也会把 cursor 推进到 `HEAD`，后续启动恢复快速路径。

首批 adoption 可以较慢，因为它只发生一次；正确性优先于启动时间。运行时不得为了隐藏迁移耗时在后台启动旧 schema，也不得先对外 readiness 再迁移。

## 13. 失败与恢复语义

| 故障点 | cursor | 状态与下一次行为 |
|---|---|---|
| baseline cursor 创建前失败 | 不存在 | 重新分类，不改变用户数据 |
| baseline cursor 创建后、apply 前失败 | baseline | 下次重新生成同一范围计划 |
| apply staging 中失败 | 保持上一个成功提交 | 正式对象不变，清理唯一 staging 后重试 |
| 发布后、verify 前崩溃 | 保持上一个成功提交 | 下次 assess 识别效果，verify 后推进 |
| verify 失败 | 保持上一个成功提交 | runtime 阻塞，保留正式结果、旧备份和诊断 |
| cursor 原子写失败 | 保持旧 SHA | 下次通过 assess/verify 安全重放 |
| 分支分叉或降级 | 不改变 | fail-loud，等待显式 reconcile/revert |

自动 revert 不安全，因为配置、会话和插件在升级后可能已经产生新状态。人工 revert 必须先停止 runtime，展示将恢复的对象和会丢失的升级后变化，再使用对应 bundle 的 revert 或经过验证的备份。

## 14. 验收设计

### 14.1 单元与 Git 历史场景

- cursor 等于 `HEAD` 时断言不扫描目录、不导入 bundle、不打开 workspace DB。
- 在临时 Git 仓库构造 baseline、无迁移提交、一个迁移提交和多个迁移提交，验证范围和顺序。
- 构造 squash 后同一提交中的多个 bundle，验证路径序和整提交 cursor 原子性。
- 修改或删除已合入 bundle，验证 append-only policy 失败。
- 构造 cursor 分叉、降级和浅历史，验证 fail-loud。

### 14.2 新旧用户矩阵

- 新 clone + 空 config/workspace：初始化后 cursor=`HEAD`，迁移脚本调用数为零。
- config 存在 + cursor 缺失：cursor 从 baseline 开始，执行 adoption。
- config 缺失 + durable workspace：不得进入 fresh 路径。
- 当前格式 config + cursor 缺失：只 verify 和推进，不改配置字节。
- 初始化在创建 config 后故障：重启进入 adoption 并收敛到最新结构。

### 14.3 持久化与故障注入

- 在 apply、原子发布、verify 和 cursor 写入各阶段注入进程终止，重启后结果只出现一次。
- 配置迁移前后核对字段、权限、未知键、secret 不出现在日志。
- Akasha 重建核对 `sessions.db/messages` 与 embedding write set 为零，原库 backup 可恢复，新库 integrity/parity 通过。
- 两个并发启动只能有一个 migration owner，另一个不能启动 runtime。
- 从备份恢复到隔离 workspace，执行 Config load、SQLite integrity 和只读关键路径 smoke。

### 14.4 系统边界验收

使用隔离 Git checkout、config、workspace、HOME 和 plugin home，分别模拟新用户和真实旧配置。迁移完成后只通过正常启动入口发送一轮消息，证明 provider 配置、workspace 初始化和 runtime readiness 共同成立。验收不得读取或写入正式 workspace。

## 15. 分阶段实施边界

1. 建立 migration root、固定 baseline、sidecar cursor、锁和 Git catalog；先用无业务写入的测试 bundle 验证顺序与快速路径。
2. 接入启动顺序和 fresh/legacy 分类，保证迁移发生在 `init_workspace` 与 runtime 构造之前。
3. 加入首批模型 runtime 配置与 Akasha rebuild adoption bundle。
4. 加入 append-only CI policy、故障注入、隔离恢复与系统边界 Gate。
5. 更新 README，说明普通用户只需 pull 后启动、失败时如何查看迁移错误和恢复备份。

每个阶段仍须按 `docs/WORKFLOW.md` 建立独立实现合同、worktree、备份和 Gate；本文不授权直接修改正式 config、workspace、数据库或运行中实例。

## 16. 非目标

- 不建设通用数据库 schema DSL，也不替换各模块已有的业务 storage owner。
- 不扫描和自动执行仓库之外的任意脚本。
- 不用 LLM 判断迁移是否需要或是否成功。
- 不自动 fetch Git 历史，不在启动时访问网络。
- 不自动降级、跨分支 reconcile 或删除 migration backup。
- 不保证第一版支持无 `.git` 的 archive、wheel 或镜像发行物。
- 不把 migration cursor 当成应用产品版本、数据库 schema version 或用户可编辑配置。
