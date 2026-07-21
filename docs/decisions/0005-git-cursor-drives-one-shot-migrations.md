# 0005 · Git cursor 驱动一次性兼容迁移

- 状态：accepted
- 日期：2026-07-21
- 关联条款：MIG-001、MIG-002、WSP-003、BAK-001、TST-002、TST-005

## 背景

模型 runtime 配置和 Akasha 派生结构会随代码演进。把旧字段兼容、旧数据库判断和一次性重建长期保留在配置加载与核心 runtime 中，会持续扩大启动路径，也无法准确回答某个旧安装已经成功执行过哪些兼容动作。

项目不希望维护人工递增的产品 migration version。用户从 Git 更新源码后，需要自动、顺序且只执行一次的迁移；新 clone 的新安装则应直接使用当前结构，不回放只为旧状态准备的脚本。

## 决定

1. 固定 `main@012e37c8b51df045353972bb551d8e868ab52455` 为迁移 baseline，此后不随发布更新。
2. 每个配置实例旁保存一个原子更新的 Git commit cursor。cursor 与当前 `HEAD` 相同就是启动快速路径，不扫描迁移目录。
3. HEAD 变化后，从 `cursor..HEAD` 的 Git 主线历史发现新增 migration bundle；同一提交的 bundle 全部 verify 成功后才推进 cursor。
4. migration bundle 只追加不修改。修正旧迁移时新增 correction bundle，不把历史兼容分支移回核心 runtime。
5. cursor 缺失且配置或 workspace 持久状态存在时，从固定 baseline 接管；两者都不存在时直接初始化当前结构并写入 `cursor = HEAD`。
6. 迁移在 runtime 启动前持有配置锁和 workspace 单实例锁。配置与 SQLite 先备份、在 candidate/staging 验证，再原子发布；失败时 runtime 不启动。
7. 自动启动只向前迁移，不自动跨分支 reconcile 或降级。bundle 提供显式 revert 边界，实际回滚仍要求维护者选择备份并确认升级后的状态影响。

## 理由

固定 baseline 把 migration 框架出现前的所有安装收敛成一个 adoption 起点，不依赖会过期的 reflog、`ORIG_HEAD` 或用户原始 checkout。单 cursor 让稳定启动成本与历史脚本数量无关，同时保留 Git 提交顺序作为迁移 catalog。把数据变换放在 append-only bundle 中，核心 runtime 只保留一个稳定 runner 和脚本协议。

## 影响

- 正常 Git checkout 可以在 pull 后启动时自动迁移。
- legacy adoption 需要 baseline 到 HEAD 的完整 Git 历史；浅克隆缺少历史时 fail-loud。
- `git archive`、wheel 和无 `.git` 镜像暂不自动迁移，后续需要构建时注入 revision 与不可变 catalog。
- 配置 cursor、锁和备份是主配置的 companion state，不进入 Git；备份可能含 secret，权限固定收紧。
- 迁移脚本成为长期可执行资产，CI 阻止修改、移动、删除或向既有 bundle 追加文件。

## 验收

- `cursor == HEAD` 时没有 migration catalog 扫描或业务数据库访问。
- 新安装不执行历史 bundle；旧配置和旧 workspace 从 baseline 顺序迁移。
- apply、发布、verify 和 cursor 写入任一阶段崩溃后，可以安全重试且不越过失败提交。
- 配置字段和 secret 保留；Akasha 重建不修改 `sessions.db/messages`，旧库有可验证备份。
- 并发 runtime 无法绕过迁移锁，分支分叉、降级和浅历史明确失败。

完整组件、状态流和场景矩阵见 [`../spark/2026-07-21-git-backed-one-shot-migrations-design.md`](../spark/2026-07-21-git-backed-one-shot-migrations-design.md)。
