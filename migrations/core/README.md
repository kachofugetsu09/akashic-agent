# Core migrations

`20260921_01_plugin_update_input_ref.py` 是待执行的 Core 迁移源：它只为已有
`runtime/plugin-reloads.sqlite3` 的 `plugin_updates` 增加 nullable `input_ref`，
先建立命名 SQLite 恢复点并保留全部旧行；新 workspace 没有该数据库时走明确的
fresh-install no-op，由 `ReloadJournal` 创建当前 schema。迁移尚未在当前 workspace
执行，不能把源文件或迁移账本当作已升级证据。

迁移回调接收 Yoyo 提供的 connection，但目标数据库由 migration context 确定；升级前
创建唯一的 `plugin-reloads.sqlite3.before-input-ref.<uuid>.bak`，在单一事务中只执行
`ALTER TABLE ... ADD COLUMN input_ref TEXT`，随后核对已知 ALTER 后 schema 和
`integrity_check`；测试源另核对旧行、外键与索引。已经是新 schema 时不重复备份或写入；旧 schema、缺表和未知形状
均 fail-loud。没有 downgrade：恢复必须由维护者选择命名备份并另行核对，普通启动也
不会偷偷迁移或重写整库。

本批只提交迁移源和测试源码，实际 durable data 增、改、减均为 0；正式 workspace
仍需另行授权、备份和运行验收。

业务迁移仍由插件自己的 migration bundle 提供。历史脚本见 Decision-0066 的恢复点。
