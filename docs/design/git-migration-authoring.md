# Yoyo 迁移维护手册

本手册只描述当前迁移合同。架构取舍见
[0021 · Yoyo workspace 账本定义迁移原点](../decisions/0021-yoyo-workspace-ledger-defines-migration-origin.md)。

## 1. 目录与所有权

```text
源码 migrations/yoyo/*.py
          │ read_migrations
          ▼
┌──────────────────────┐       成功回执       ┌─────────────────────────────┐
│ MigrationRunner      │ ───────────────────▶ │ <workspace>/migrations.sqlite3 │
│ 持有 workspace 锁    │                      └─────────────────────────────┘
└──────────┬───────────┘
           │ Python step
           ▼
   明确的持久状态变换
```

- `migrations/yoyo/` 是唯一会被 runtime 加载的迁移目录。
- 其他 migration 子目录是旧 Git cursor 系统的历史源码，不注册、不执行。
- `agent/migrations/runner.py` 只负责加锁、选择待执行项、调用 Yoyo 和报告失败。
- migration step 拥有自己的变换、校验和恢复边界；它可通过
  `agent.migrations.context.current_migration_context()` 取得 config 与 workspace 路径。

## 2. 新增迁移

在 `migrations/yoyo/` 新增一个文件，文件名使用日期、同日序号和短职责，例如：

```text
20260803_01_add_example_index.py
```

最小 Python 迁移：

```python
from yoyo import step

__depends__ = {"20260802_01_yoyo_origin"}


def apply_change(connection: object) -> None:
    """执行并验证一次明确的持久状态变换。"""
    ...


steps = [step(apply_change)]
```

`__depends__` 只表达数据上的真实先后关系。两个从同一原点并行开发、互不依赖的迁移可以
拥有相同依赖；Yoyo 会分别记录 migration ID，不要求它们的 Git commit 互为祖先。

## 3. 实现边界

1. 修改前列出目标、正常增加、允许更新、物理减少条件、owner 和恢复证据。
2. 删除、覆盖或批量改写权威状态前，先创建名称清楚的恢复点并验证可读。
3. step 必须可在失败后安全重试；不要捕获错误并伪造成功。
4. 在 step 内完成结果校验。函数成功返回后，Yoyo 才能记录成功回执。
5. 不把 Git HEAD、分支名、产品版本号或旧 cursor 用作迁移状态。

已合入目标分支的 migration 文件不可修改、移动或删除。修复旧迁移时新增 correction ID，
并依赖需要修正的 migration。

## 4. 最小验证

```bash
python -m pytest tests/test_migration_runner.py tests/test_yoyo_migration_append_only.py
python scripts/check_yoyo_migrations.py --base origin/main
```

测试至少覆盖：第一次执行、重复启动不重跑、失败不落账且可重试、workspace 锁，以及并行
sibling migration 合并后的缺失项执行。涉及业务数据时，再按持久化状态地图验证数据库、
文件、write set 和恢复点，不能只断言返回值。
