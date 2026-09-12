# 历史：Yoyo 迁移维护手册（已退役）

> 本页只保留历史入口名称和审计定位，不是当前开发、启动、升级或 Gate 指令。
> 全局 Yoyo、`legacy_upgrade`、migration bundle、runner 和 append-only gate 已由
> [0066 · 退役全局 Yoyo 迁移](../decisions/0066-retire-global-yoyo-migrations.md) 取代。

旧版本曾把 `migrations/yoyo/` 作为全局执行目录，把 `<workspace>/migrations.sqlite3` 作为
成功回执账本，并由 Core 在业务 runtime 前持锁运行 migration。`agent/migrations/runner.py`、
`context.py`、bundle 装配、Git cursor 脚本和迁移检查属于同一历史链；
`session_db_backup.py` 与 `proactive_island/` 则是可保留但必须显式调用的离线管理工具。退役链
现在只在 Git 恢复点中保留源码证据，不注册、不自动执行，不能用来接管当前 workspace，也不能
作为插件安装的依赖。

当前规则是：空 workspace 由实际插件 owner 创建自己的当前 schema；已有状态由 owner 按已声明、
已全部升级且仍合法的 schema lineage 集合检查，允许集合不要求与全新库 DDL 逐字相等，未命中就在
业务写入前 fail-loud。插件未来只为自己拥有的状态负责 schema 演进、备份和恢复；Core 不维护
全业务 schema 目录，插件之间通过版本化能力边界协作。

旧正文和旧脚本可从 Git 恢复点
`5e97cf798f2594b5f46803ad1b3d7b5a3485ea11` 或交付备份 `before-yoyo-retirement.bundle`
恢复。恢复历史源码不等于恢复正式 workspace，也不授权执行历史迁移。
