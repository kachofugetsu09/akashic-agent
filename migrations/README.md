# Workspace migrations

`yoyo/` 是唯一自动执行的迁移目录。每个顶层 Python 文件使用 Yoyo migration ID
和 `__depends__` 声明依赖；已经进入主分支的文件只追加、不修改
（由 `scripts/check_yoyo_migrations.py` 强制）。

目录里只剩 `yoyo/`：2026-09-10 插件边界第 3 步删除了此前的四个 Git cursor
遗留子目录（`akasha_sparse_index_v8/`、`provider_runtimes_and_akasha/`、
`workspace_veda/`、`workspace_veda_uppercase/`）。它们不在 Yoyo catalog 内，
也不会被 `MigrationRunner` 读取，删除不改变任何 workspace 的迁移结果。
已应用的迁移记录保留在各自 workspace 的 `migrations.sqlite3`。

**运行时机与代价**：`MigrationRunner` 在 runtime 启动前调用
`backend.to_apply(read_migrations("migrations/yoyo"))`，而 yoyo 会 **import 每个
迁移模块**来解析 `__depends__`。因此从零安装的 workspace 必须能 import 迁移所引用
的模块（包括 `plugins/**`），迁移步骤里出现的插件依赖是结构性依赖，不是可随手
删除的耦合。范围与理由见
[决策 0064](../docs/decisions/0064-plugin-boundary-is-machine-enforced.md#6-r1-范围裁决与迁移豁免2026-09-10-补充)。
