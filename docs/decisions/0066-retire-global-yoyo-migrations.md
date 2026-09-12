# 0066 · 退役全局 Yoyo 迁移，由插件 owner 持有 schema 生命周期

- 状态：accepted
- 日期：2026-09-13
- 取代：[0021 · Yoyo workspace 账本定义迁移原点](0021-yoyo-workspace-ledger-defines-migration-origin.md)
- 关联条款：MIG-001、MIG-002、PLG-019、WSP-003、BAK-001、STA-001

## 背景

全局 Yoyo 账本、`legacy_upgrade`、历史 migration bundle 和 Core 迁移 runner 把不同插件的业务
schema 演进收进一个中心。它要求 Core 理解并装配并不属于 Core 的状态，也让插件之间通过迁移实现
隐式耦合。用户已经明确当前用户数据、schema 和 config 均按当前版本处理；本次变更接受一个
breaking baseline，不为旧安装保留自动升级路径。

## 决定

1. 当前 runtime 不加载全局 `migrations/yoyo/`、`legacy_upgrade`、迁移 bundle、全局 append-only
   migration gate 或 `<workspace>/migrations.sqlite3` runner，也不加载已退役的
   `agent/migrations/runner.py`、`context.py` 和 bundle 装配。`agent/migrations/session_db_backup.py`
   与 `agent/migrations/proactive_island/` 这类离线、显式调用的状态管理/盘点工具可以保留；它们
   不在启动链、不自动执行，也不构成全局迁移 API。退役 runner/context/bundle 只在 Git 恢复点中
   作为历史源码保存，不属于当前安装入口。
2. 空 workspace 由各实际 owner 直接创建自己的当前 schema 和配置。Core 只提供已声明的路径、
   租约、生命周期和组合，不建立全业务 schema 中央目录。
3. 已有 workspace 由各 owner 在第一次打开时检查 owner 自己声明的 schema identity lineage 集合。
   该集合可以包含全新库的当前 DDL，以及 owner 已明确核对、全部升级完成且仍合法的多个已知
   lineage；“exact”表示精确命中这份允许集合和数据不变量，不要求与全新库 DDL 逐字相等。
   例如 sessions owner 可以明确接受含已知额外列和合法 `attributes` 表示的历史 lineage。缺失、
   未知、部分升级、损坏或不匹配必须在业务写入前 fail-loud；不猜测旧版本、不清库、不用默认值
   伪装兼容。
4. 插件未来只为自己拥有的状态负责 schema 演进、备份、恢复和版本合同。跨插件协作继续使用
   版本化 `ServiceKey`、结构合同或 typed event，不能用兄弟 import 或中央迁移器重新建立依赖。
5. 本次 breaking baseline 不迁移正式 workspace，不改写 `sessions.db/messages`、正式数据库或
   历史 ledger。已有状态必须由发布前置条件保证为当前状态；恢复依赖明确的外部备份或 Git
   恢复点，不由运行时猜测历史。

## 理由

schema 的事实和演进都属于产生它的插件。把初始化和检查放回 owner 后，Core 只组合原子能力，
替换一个插件不会要求 Core 认识业务表或历史脚本；插件也可以在不导入兄弟实现的情况下通过
版本化能力声明协作。对已有状态采取显式拒绝，能让数据损坏和版本缺口在写入前可见。

## 影响

- 本次代码栈移除历史 `legacy_upgrade` 后，预期本地插件清单为 41 个；默认 profile 仍为 22 个。
  这两个数字是实施核对目标，不能由本决策文档单独证明 runtime 已验收。
- 新 workspace 不再生成或登记全局 migration ledger；每个实际 owner 负责自己的空状态创建。
- 旧决策 [0021](0021-yoyo-workspace-ledger-defines-migration-origin.md) 保留为历史记录，不再是当前
  启动或升级说明。旧维护手册已改为历史页；历史代码可从 Git 恢复点 `5e97cf798f2594b5f46803ad1b3d7b5a3485ea11`
  恢复，交付备份名为 `before-yoyo-retirement.bundle`。

## 验收

以下是实现验收边界；本决策只接受语义基线，不宣称代码、整组安装或 Gate 已完成：

- 当前代码和安装链不加载全局 Yoyo、`legacy_upgrade`、迁移 bundle、runner 或 append-only gate。
- 空 workspace 的每个启用 owner 能创建并打开自己的当前 schema；每个 owner 的已知、完整且仍合法
  lineage 集合都能打开，未知或部分升级状态在业务写入前失败。
- Core 没有全业务 schema 目录，插件之间没有兄弟实现 import；默认 22 个 profile 插件和本地 41
  个插件清单由代码/安装检查另行证明。
- 迁移实现不改正式数据库、Message 或历史 ledger；验证只使用一次性 workspace、隔离安装和
  可恢复备份。
