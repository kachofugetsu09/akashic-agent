# 0021 · Yoyo workspace 账本定义迁移原点

- 状态：accepted
- 日期：2026-08-02
- 取代：[0005 · Git cursor 驱动一次性兼容迁移](0005-git-cursor-drives-one-shot-migrations.md)
- 关联条款：MIG-001、MIG-002、WSP-003、BAK-001、TST-002、TST-005

## 背景

多个开发者从同一 HEAD 并行开发时，各自提交迁移后会形成合法的 sibling 分支。用单个 Git commit cursor 表示数据库状态，会把源码祖先关系误当成迁移依赖关系：先部署任一分支后，另一分支即使只新增独立脚本，也会因 cursor 不是新 HEAD 的祖先而拒绝启动。浅克隆、rebase、merge 和无 `.git` 发布包还会增加与业务 schema 无关的失败。

## 决定

1. 使用 `yoyo-migrations==9.0.0` 执行 Python 迁移，唯一可执行目录为 `migrations/yoyo/`。
2. 每个 workspace 使用 `<workspace>/migrations.sqlite3` 保存已成功执行的 migration ID；一个 workspace 只拥有一份迁移历史。
3. 当前代码和持久化结构是新系统原点。旧目录中的四组脚本仅保留为历史源码，不注册、不自动执行，也不承诺接管旧状态。
4. 原点 migration 删除所选配置旁退役的 `.migration-cursor`、`.migration-lock` 和 `.migration-backups/`，不修改配置或业务数据。这是一次明确的破坏性兼容切断，不提供自动恢复。
5. 启动 owner 在 runtime 之前持有 workspace 单实例锁，执行全部缺失迁移。只有 step 成功后 Yoyo 才写入回执；失败保持未应用并阻止启动。
6. 已注册 migration 文件不可修改、移动或删除。修正使用新的唯一 ID，并通过 `__depends__` 表达真实依赖；分支拓扑不参与迁移判断。

## 理由

迁移状态的事实是“哪些 migration ID 已成功”，不是“最后运行了哪个 Git commit”。按 ID 和显式依赖记录回执，可以让 sibling 分支先后合并并各执行一次，也能在 release tarball、容器和浅克隆中保持同一行为。workspace 是业务状态根，因此账本随 workspace 备份和恢复，不与任意 config 文件名绑定。

## 影响

- 旧 cursor 无法导入为新账本；升级到本决定的代码后从原点开始。
- 第一次启动会创建 workspace 账本并删除旧迁移 companion state。
- 历史脚本仍可供审计，但测试、CI 和 runtime 不再把它们当成可执行合同。
- 新增依赖 `yoyo-migrations`；不再要求 Git 可执行文件、完整历史或固定 baseline。

## 验收

- Bob 与 Alice 从同一 HEAD 各新增 sibling migration，任一先执行后，合并目录仍能只执行另一条缺失 migration。
- 同一 workspace 重启不重复执行；迁移失败不写成功回执，修复外部条件后可以重试。
- 原点只删除三类退役 companion，配置和业务文件字节不变。
- CI 允许新增 migration，拒绝修改、移动或删除基线中已注册的 migration。
- 正式启动路径不读取 Git HEAD、祖先关系、baseline 或旧 cursor。
