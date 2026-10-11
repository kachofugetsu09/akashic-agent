# 0113 字节完整性只在安装边界校验

状态：proposed / implemented for review
关联：#1179；PLG、MIG

## 决定

发行制品与 wheel tree 的 SHA-256 在安装时校验一次；之后运行时不再对同一份字节重复取摘要。

- 插件 `migration.catalog.toml` 升为 schema 2：只声明 ID、路径、依赖和事务属性，删除
  `files` 清单及逐文件 `sha256`；运行时不再计算 catalog/bundle 摘要。Yoyo 扫描面检查、
  导入边界检查（覆盖 bundle 内全部 `.py`）和依赖图检查保留。
- Core `migrations/catalog.toml` 升为 schema 2：只记录 ID 和 owner bundle。依赖与事务属性由
  owner bundle 唯一声明，不再保存第二份并比较一致性。
- 已发布迁移不可改写由 `scripts/check_yoyo_migrations.py` 用 Git 基线逐字节比较；登记范围是
  Core 迁移文件和基线中每个 bundle 的整个 `migration_root` 目录。删除只服务于一次性历史审计的
  `migrations/retired.toml`（仅在基线恰为两个旧恢复提交时生效）和 `migrations/ledger-moves.toml`
  （#1257 合入后其旧路径已不在基线中）；Git 历史仍保留两者及其审计结果。
- 公共合同的重启判断改为对去掉 docstring 的 AST 取摘要，注释、空白和 docstring 改动不再触发
  `restart_required`；导入时不再重读合同文件比对摘要。

## 理由

对照 DSH：代码完整性只由发行 manifest 中每个包的 SHA-256 在安装时保证；运行时只认格式版本号，
不对代码或迁移文件取摘要。我们的同一迁移文件原先在 catalog、bundle 总摘要、CI 解析器和搬迁清单中
最多被核对四次，而这些字节都来自已经校验过的同一制品。

## 可观察影响与恢复

schema 1 catalog 会被明确拒绝；内置 bundle 与 Core 同版本发布，不存在混用。没有数据迁移、
schema 改动或 Yoyo 账本变化；已执行的迁移 ID 与源码字节不变。恢复点为父提交。
