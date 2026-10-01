# Yoyo 迁移维护手册

本手册只描述当前迁移合同。架构取舍见
[0021 · Yoyo workspace 账本定义迁移原点](../decisions/0021-yoyo-workspace-ledger-defines-migration-origin.md)
和 [0066 · 当前基线](../decisions/0066-yoyo-current-baseline.md)。部署选择以 [0082 · 内置代码随部署](../decisions/0082-distribution-owned-plugin-composition.md) 与[操作手册](operator-deployment.md)为准。

## 1. 目录与所有权

```text
Core 自有脚本 / 当前发行版全部内置 migration bundle
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

- `migrations/core/` 仅放 Core 自有迁移；内置业务迁移从当前固定发行版的插件 bundle 读取，范围不受默认 profile 或启停选择裁剪。外置插件自行负责兼容和迁移，框架不发现、校验或执行其 Yoyo catalog。
- 当前历史迁移已退役；Core catalog 为空。旧 ledger 记录保留，新脚本不得依赖已退役 ID。
- 其他 migration 子目录是旧 Git cursor 系统的历史源码，不注册、不执行。
- `agent/migrations/runner.py` 负责核对来源与成功账本、复用写入锁、调用 Yoyo 和报告失败。选择发行版即接受必要的 Core/内置迁移，发布与启动先完成迁移再加载新版组合；部署清单不再接受 `migrations` ID 列表。
- 内置与外置共用同 ID 的可证 owner 冲突在迁移前拒绝。Core 先升级自身账本结构，再核对安装与配置恢复 owner 已结算；未结算时只能按操作手册恢复原实例，不能发布新输入覆盖未决事实。
- migration step 拥有自己的变换、校验和恢复边界；它可通过
  `agent.migrations.context.current_migration_context()` 取得 config 与 workspace 路径。

## 2. 新增迁移

在内置 owner 的 migration bundle 中新增文件，并更新其 catalog/digest；Core 自有中立状态使用 `migrations/core/`。外置插件的迁移入口与运行方式由插件自己定义。文件名使用日期、同日序号和短职责，例如：

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
2. 说明破坏性写入与局部恢复机制，供部署者决定是否备份及如何恢复。不要在新 step 中强加每次 release 全状态备份策略；不可修改已发布 step 的既有恢复合同。
3. step 必须可在失败后安全重试；不要捕获错误并伪造成功。
4. 在 step 内完成结果校验。函数成功返回后，Yoyo 才能记录成功回执。
5. 不把 Git HEAD、分支名、产品版本号或旧 cursor 用作迁移状态。

已合入目标分支的 migration 文件不可修改、移动或删除。修复旧迁移时新增 correction ID，
并依赖需要修正的 migration。

## 4. 最小验证

按 [WORKFLOW](../WORKFLOW.md) 使用临时真实 workspace 验证：来源和 owner 冲突在业务写入前拒绝、
发布与启动自动推进必要 Core/内置迁移、停用内置仍被发现、外置 catalog 不被执行、
成功 ID 不重跑、失败不落成功账且按 step 合同可重试，以及持有 workspace 锁。
涉及业务状态时按持久化状态地图核对真实写入范围与完整性，不只检查返回值。

```bash
python scripts/check_yoyo_migrations.py --base origin/main
```

新增测试仅按 WORKFLOW 的概念回归范围；普通迁移验证不自动新增单元测试。
