# Workspace migrations

Yoyo 与通用 runner 保留，用于未来兼容升级。当前版本以所有现有用户已处于当前状态为基线；
旧 `legacy_upgrade` 包和清理 Git cursor 的 origin 脚本已退役，历史源码只保存在 Git 恢复点。

`core/` 当前没有迁移脚本；`catalog.toml` 当前没有历史业务 requirement。新 workspace 由实际
owner 建立当前结构，Yoyo 继续保存独立账本。既有 ledger 中已执行的历史 ID 保留，不重新执行，
也不伪造执行回执。源码移除不授权减少正式 workspace 的任何数据。

未来业务迁移放在插件自己的 bundle 中，由 `migration.catalog.toml` 声明源文件、依赖、事务属性
和 package digest，通过正式安装链校验与发现。Core 只执行这些声明，不导入业务插件源码或
建立业务 schema 目录。新迁移不得依赖仅存在于旧 ledger、源码已经退役的 ID。

执行前持有 workspace 锁；成功才记录 Yoyo 回执，失败阻止启动并允许同一 artifact 重试。
未来已发布脚本仍受 append-only 检查保护。本次历史删除的精确审计例外与恢复方式见
[Decision-0066](../docs/decisions/0066-yoyo-current-baseline.md)。
