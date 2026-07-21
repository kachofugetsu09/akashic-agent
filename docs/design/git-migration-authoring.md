# Git 一次性迁移维护手册

状态：已实现，供后续维护者与 Agent 执行

本手册回答两个问题：兼容逻辑应放在哪里，以及新增一次性迁移时必须怎样实现和验证。架构理由与完整状态语义见[决策 0005](../decisions/0005-git-cursor-drives-one-shot-migrations.md)和[设计说明](../spark/2026-07-21-git-backed-one-shot-migrations-design.md)。

## 1. 先判断是不是迁移

不要因为看到两个格式就立刻修改核心 loader。按下面顺序确定 owner：

```text
发现旧字段、旧文件或旧数据库
              │
              ▼
┌──────────────────────────────┐
│ 它是已发布状态的一次性历史形状吗？ │
└──────────────┬───────────────┘
          否   │   是
       ┌───────┘   └──────────────┐
       ▼                          ▼
当前仍合法的外部差异？       能确定来源形状与目标形状？
       │                          │
  是：边界 adapter          否：fail-loud，先补 lineage
  否：输入校验失败          是：新增 migration bundle
                                  │
                                  ▼
                         核心只接受迁移后的规范形状
```

- Provider、协议或平台在当前版本中仍会同时存在的差异，属于边界 adapter 或 provider runtime，不是一次性迁移。
- 旧版本曾经写入、当前版本只需转换一次的配置或持久状态，属于 migration bundle。
- 从未合法发布过的坏数据不应被猜测修复；在拥有该 schema 的边界明确失败。
- 核心 `Config`、provider、AgentLoop 和 storage loader 只拥有当前规范形状。不要在这些路径新增 `legacy_*`、旧字段 fallback、按旧版本分支或静默默认值。

## 2. 启动时怎样决定是否执行

固定 baseline 记录在 `migrations/.root`，不得随发布更新。每个配置实例的 companion cursor 位于 `<config>.migration-cursor`。

```text
读取 Git HEAD + cursor
          │
          ├─ cursor == HEAD ──► 立即启动；不扫描 migration，不访问业务 DB
          │
          └─ 不相等/缺失
                    │
                    ▼
          config lock + workspace lock
                    │
          ┌─────────┴──────────┐
          │ cursor 缺失         │ cursor 存在
          ▼                    ▼
 config/workspace 都空？   校验 cursor 是 HEAD 祖先
    │             │                 │
   是             否                ▼
 fresh；初始化后  从 baseline      按 first-parent 顺序
 cursor=HEAD      adoption         执行新增 bundle
```

一个提交内的全部 bundle 通过 `verify` 后，runner 才把 cursor 推进到该提交。所有待处理提交完成后，cursor 推进到当前 `HEAD`；因此没有迁移的代码更新也只承担一次慢路径查询。

## 3. 新增 bundle 的固定步骤

1. 从 `docs/INDEX.md` 读取相关状态 owner、`projectneed` 条款和持久化状态地图，列出所有已经发布的来源形状。无法区分的异构状态必须阻塞，不能猜测。
2. 在一个新的提交中新增 `migrations/<语义名称>/migration.py`。一个 bundle 目录只能首次加入一次；合入后不得修改、移动、删除，也不得向目录补 helper。修复旧迁移时新增 correction bundle。
3. 使用 runner 已定义的子进程参数：`action`、`--config`、`--workspace`、`--migration-commit`，以及 apply/revert 时的 `--backup-dir`。脚本只操作这些显式路径，不推断 `HOME` 或正式 workspace。
4. 实现四个动作：
   - `assess`：只读识别，输出单个 JSON 对象；仅返回 `needed`、`satisfied` 或带原因的 `blocked`。
   - `apply`：先建立可恢复备份，再写唯一 candidate/staging，完整校验后原子发布。
   - `verify`：使用当前正式 loader、schema 与完整性检查验证最终状态；不得顺便修复。
   - `revert`：只提供显式人工回滚能力，不由自动启动调用。
5. 配置和 manifest 文件使用 `0600`，备份目录使用 `0700`。日志、异常、测试名和 manifest 不得包含 API key、token 或配置全文。
6. SQLite 通过 online backup 生成恢复点，在唯一 staging 数据库完成迁移并执行 `PRAGMA integrity_check`，最后原子替换。不得通过删除 sessions、清库或 destructive fallback 获得成功。
7. `apply` 必须可重试：发布后、cursor 前崩溃时，下次 `assess` 应识别已完成效果，`verify` 成功后推进，而不是重复产生业务变化。
8. 不在迁移脚本中调用网络、LLM、provider 或易变的 runtime/UI 内部实现。历史迁移必须能在未来 checkout 中离线执行。

如果已合入的 bundle 本身会在 correction 运行前产生不可接受的副作用，修正原脚本是唯一可恢复方案。此时必须新增 `migrations/repairs/<语义名称>.toml`，用 `path`、`base_sha256` 和 `head_sha256` 锁定唯一允许的字节变化，并记录 `reason`。repair 声明本身仍只追加；hash 不匹配或未实际修改目标都会使 Gate 失败。不得用 repair 变更已经成功发布的业务语义。

## 4. 代码评审检查表

评审每个迁移时逐项回答：

- 来源 shape 是否有已发布证据，schema owner 是谁？
- 目标 shape 是否只由当前 canonical loader 定义？
- `assess` 能否区分 needed、satisfied 和不可安全识别的 blocked？
- apply 前有哪些恢复点，备份能否在隔离 workspace 恢复？
- 正常增加、允许原位更新、逻辑失效和物理减少分别是什么？没有已批准协议时物理减少必须为“无”。
- apply、发布、verify、cursor 写入任一点失败，cursor 是否都不会越过失败提交？
- secret 是否只存在于权限受限的配置、临时文件和备份中？
- 核心 runtime 中是否仍残留本次旧 shape 的分支？若有，指出为什么它是当前外部差异而不是历史兼容。
- 是否只新增 bundle，且 `python scripts/check_migrations_append_only.py --base origin/main` 通过？

## 5. 必跑验证

先运行迁移相关的快速测试，再运行 Docker case matrix：

```bash
.venv/bin/pytest -q -W error \
  tests/test_migration_runner.py \
  tests/test_provider_runtime_akasha_migration.py \
  tests/test_migration_append_only.py \
  tests/test_main_lightweight_commands.py

python docker/debug/migration_probe.py
python docker/debug/gate.py run --base origin/main
```

Docker probe 复用 runtime control Gate 镜像，把候选源码只读挂载到 `/app`，把 config、workspace、HOME、临时 Git 仓库和 JUnit 写到独立 sandbox。结果保存在 `docker/debug/reports/migrations/<run-id>/gate.json`；只有每个 testcase 通过、候选仓库前后 digest 相同且 Compose 清理成功，Gate 才返回成功。

当前 case matrix 覆盖：

| Case | 期望 |
|---|---|
| 新 clone，config/workspace 都空 | 返回 fresh；初始化成功后 cursor=`HEAD`，不回放历史 |
| config 存在，cursor 缺失 | 从 baseline 执行 adoption |
| config 缺失，durable workspace 存在 | 不误判 fresh，从 baseline 接管 |
| 当前格式 config，cursor 缺失 | assess satisfied；配置字节不变，cursor 前进 |
| cursor=`HEAD` | 不扫描 migration history |
| 多个迁移提交 | 按 first-parent 提交顺序执行 |
| merge commit 首次引入 bundle | 以 merge commit 作为 cursor 原子单元发现 |
| 纯代码提交 | 不执行 bundle，只把 cursor 推进到新 HEAD |
| assess 无法识别来源 lineage | blocked，cursor 保持 baseline |
| apply 失败 | 不发布有效状态，cursor 保持 baseline |
| apply 后 verify 失败 | cursor 保持在上一个成功提交 |
| 有效状态发布后 cursor 写入失败 | 重试只 verify，不重复 apply |
| cursor 与 HEAD 分叉或降级 | fail-loud，不改 cursor |
| shallow history 缺失 baseline | fail-loud，不执行 bundle、不推进 cursor |
| 并发启动 | 只有 lock owner 可迁移，另一进程不能启动 runtime |
| nested legacy provider 配置 | role、字段、secret 和未知配置无损进入 named runtimes |
| root-level legacy 模型字段 | 一次迁移为 named runtimes |
| legacy 与 named runtime 混杂 | blocked，不猜测合并 |
| 现存 Akasha DB | provider 配置迁移不读写、不备份、不重建 Akasha 状态 |
| 人工 revert | 从对应 backup 精确恢复旧配置字节 |
| append-only policy | 允许新 bundle，拒绝修改旧 bundle 或追加 helper |

新增一种迁移故障或来源 lineage 时，必须先向这张矩阵补 case 和测试，再实现 bundle。不要只在 PR 描述里记录手工验证。

## 6. 恢复与已知边界

- 自动启动只向前推进，不自动降级或跨分支 reconcile。
- 恢复旧配置时必须把配置、cursor 和相关 workspace 状态视为同一安装身份；不能只覆盖配置而保留更靠后的 cursor。
- migration backup 不自动清理。删除需要独立、显式的数据管理操作和新的恢复点。
- 第一版要求正常 Git checkout 与 baseline 历史。缺少 `.git` 的 archive/wheel/镜像和缺少 baseline 的 shallow clone 会明确失败；不得在 runner 中增加静默 fallback。
