# 0066 · 保留 Yoyo，以当前状态清除历史迁移负担

- 状态：accepted
- 日期：2026-09-13
- 补充：[0021](0021-yoyo-workspace-ledger-defines-migration-origin.md)
- 关联条款：MIG-001、MIG-002、PLG-001、WSP-003、BAK-001

## 决定

维护者明确所有现有用户的数据、schema 和配置均按当前状态处理。本次只退役已经完成使命的
历史兼容脚本、`legacy_upgrade` 及其专属 helper 和测试；Yoyo、通用 runner、账本、插件自有
migration bundle 和未来 append-only 检查继续保留。它不是永久取消兼容升级。

Core 的历史业务 requirement 清空，原 Git cursor 清理脚本退役。未来业务迁移由相应插件拥有，
Core 只发现、校验并执行声明，不维护业务 schema 中央目录。Core 自己的中立持久状态仍可以
拥有迁移脚本。各 owner 既有初始化和未来升级能力不因本次历史删除而被整体重写。

当前空 workspace 由实际 owner 初始化；既有当前状态直接复用。新迁移只依赖仍可加载的迁移，
不得依赖仅在旧 ledger 中保留的历史 ID。执行仍持 workspace 锁，失败不记成功回执且阻止启动，
同一 artifact 可重试。新发布的迁移保持不可改写、不可随意删除。

## 状态与恢复

本次只改变 Git 源码和测试，不操作正式 workspace。现有 Message、数据库、旧迁移账本及其
已执行 ID、备份和配置 companion 文件均保留；不因删源码重跑历史、伪造回执或清除旧状态。
未来迁移需要的业务写入、备份、失败恢复由该状态 owner 的具体迁移声明。

恢复源码的固定提交为 `5cb8e8bbff5c2f8a2f37456b3433038091cada09`；完整恢复包为
`before-yoyo-scope-correction.bundle`。历史删除只能由精确路径与内容身份的审计规则接纳，
不能扩大为未来迁移的任意删除豁免。`migrations/retired.toml` 固定上述栈顶与
`816dcdf33c0c3e064c46290a3bcd22c66a812f09` 主分支旧路径的逐文件 SHA256；
它只审计这次历史删除，不是未来豁免名单。未来 Core 中立迁移和插件迁移仍受追加约束。

## 验收

- 当前 Core 空迁移目录和空 requirement catalog 可启动、重启。
- 旧 ledger 的历史记录保留；历史 bundle 不再参与当前装配。
- 合成未来插件 bundle 能真实执行，失败不记成功，重试后记录成功，缺依赖明确阻断。
- 正式安装、bundle digest/namespace 隔离、workspace 锁和 append-only Gate 继续有效。
- 本地插件组合、持久 Message 和生命周期在最终栈顶重新验证。
