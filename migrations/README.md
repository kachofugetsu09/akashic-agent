# Workspace migrations

`core/` 只保存 Core 自己拥有的 migration origin。旧 workspace 的业务迁移由
`plugins/legacy_upgrade` 外部 bundle 提供；Core 只读取
`migrations/catalog.toml` 的 ID、依赖和 owner 索引，不扫描 checkout 下的插件目录。

外部 bundle 的 `migration.catalog.toml` 固定每个 migration 的源文件、依赖、事务属性
和完整 package digest。已经发布的 ID 只允许在同一个 artifact 内追加；替换历史实现
必须保留 ID、依赖和数据格式，并通过新的 bundle 版本和完整性校验发布。

新 workspace 没有旧权威数据时可以只启动 Core；检测到既有 SessionDB、plugin-data、
runtime 或其他历史 SQLite 数据而缺少对应 bundle 时，启动在写入迁移账本前返回
`migration_blocked`。
