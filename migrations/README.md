# Workspace migrations

Yoyo 与通用 runner 保留，用于 owner 声明的显式升级。ADR-0066 在 2026-09-13
退役了当时已完成的历史脚本、旧 `legacy_upgrade` 包和清理 Git cursor 的 origin 脚本；
历史源码只保存在 Git 恢复点。这次基线确认不证明后来新增的迁移已经执行。

`core/` 保存 Core 中立持久状态的迁移，业务迁移由插件自己的 bundle 提供。
`catalog.toml` 不保留已退役的历史业务 requirement。新 workspace 由实际 owner 建立当前
结构，Yoyo 继续保存独立账本。既有 ledger 中已执行的历史 ID 保留，不重新执行，
也不伪造执行回执。源码移除不授权减少正式 workspace 的任何数据。

判断脚本能否退役，需要核对目标 workspace 的成功回执和仍支持的升级路径，不能只看
源文件日期或一台机器的账本。运行时旧格式读取另需核对实际数据和当前消费者，见
[兼容截止盘点](../docs/design/compatibility-cutoff-inventory.md)。

未来业务迁移放在插件自己的 bundle 中，由 `migration.catalog.toml` 声明源文件、依赖、事务属性
和 package digest，通过正式安装链校验与发现。Core 只执行这些声明，不导入业务插件源码或
建立业务 schema 目录。新迁移不得依赖仅存在于旧 ledger、源码已经退役的 ID。

执行前持有 workspace 锁；成功才记录 Yoyo 回执，失败阻止启动并允许同一 artifact 重试。
未来已发布脚本仍受 append-only 检查保护。本次历史删除的精确审计例外与恢复方式见
[Decision-0066](../docs/decisions/0066-yoyo-current-baseline.md)。
