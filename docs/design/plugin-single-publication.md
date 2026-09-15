# 完整 Root 的单一发布路径

基线：`df9179cf`。状态：Manager 与示例已收敛，Store 旧晋升 API 已删除；行为验收未运行。
依据：[ADR 0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)。
本层只有静态检查，未运行测试或实验。

## 删除证据与保留边界

`_build_and_publish_root` 是 `_commit_snapshot_with_publication_participants` 唯一生产
caller；后者又是 `_commit_snapshot_participants` 唯一 caller。固定实参为
`promote_latest=False`、`force_provisional=True`、`provisional_started=False`、
`reopen_previous_on_failure=False`，且总是提供实际 preclosed Channel state。
包装层总是取得实际 startup lease。删除上述四个参数和不可达路径，将 state 与 lease
设为必需参数，没有新增状态 flag。

```text
┌────────────────────────────────────────────┐
│ 新正式 snapshot → closed provisional       │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│ endpoint / Channel / runtime 初始化        │
└─────────────────────┬──────────────────────┘
                      ▼
┌────────────────────────────────────────────┐
│ 原提交许可检查 → finalize → 旧 snapshot drain │
└────────────────────────────────────────────┘
```

保留 exact previous/candidate owner 检查、old Channel stop 检查、真实 endpoint 切换
与失败恢复、Channel 关闭失败句柄、Root 生命周期、durable commit 回执和取消处理。
旧组合已经释放，失败不能在本函数中重新开放旧接纳；外层仍负责真实重建。
没有修改 Channel 类、credential、selection、operation 或业务验证，也未绕过此前
Channel 迁移写入拒绝。本层不产生新的持久写入或数据减少协议。

无调用者的 `_record_drained_root_failure` 删除。实际进程内关闭失败仍由
Store `_drain_failures` 与对应 snapshot/Root 持有，未改 journal 历史记录。
`runtime_generation_ids` 的唯一现有调用位于 `test_plugin_fresh_root.py`；
本轮未扩写该测试，暂留此查询方法。

## 示例和行为边界

hot-reload 底座测试改为独立 Root：stable/latest lease 分别绑定实际 snapshot，
候选关闭成功后才创建新 formal；旧 stable lease 保留到释放，分别观察各 Root 清理。
实验脚本只在编译前做初始依赖装配变化，compile 后不再单独 dispose provider。
候选退出后以相同 provider 输入构造新实例、runtime、数据目录及正式 snapshot，
不复用物理 Root 或要求相同 snapshot ID。脚本的持久输出仍只属于显式新建实验目录。

## Store 的正式发布边界

`df9179cf` 已将 hot-reload 测试与 `scripts/plugin_composition_experiment.py` 改为
先关闭候选，再创建全新正式 Root。删除前全仓静态搜索确认
`RuntimeSnapshotStore.promote_latest` 与 `promote_latest_provisional` 没有代码消费者；
本层按明确授权删除两个方法及仅供它们使用的 `require_validation` 参数和分支。
`retain_publication_target` 只接受 Store 当前持有的 exact pending/provisional 事务，
删除已关闭 latest 候选的旁路授权。既有 fresh Root 测试补充候选事务与已完成事务拒绝、
pending/provisional 真实发布 lease 可用的断言；测试尚未运行。

保留候选及 pending Root 的封存入口、验证身份记录、编译冻结、拓扑与生命周期检查，
以及 snapshot 和 generation 各自的真实 lease 计数。候选记录不复制到全新正式 Root，
正式 Root 仍在发布边界独立检查。编译器只删除无人读取的 `identity` 局部累加器，
实际生成 snapshot ID 的 `canonical_identity` 不变。`catalog_generation` 仍有 Manager
调用，本层保留；Manager、Snapshot Channel 字段及其校验由原 owner 继续维护。

本层只修改 Git worktree 中的源码、测试与本文，不增加、更新、失效或删除正式 workspace
中的消息、回执、stable 选择、归档和 plugin-data。代码可从 `df9179cf` 恢复；
修改前文件备份为 `/tmp/akasic-plugin-snapshot-v2-df9179cf-backup/source-before.tar`，
该备份不是运行数据恢复证据。

验证只执行 `git diff --check`。tests、Gate、CI、build、lint、AST、runtime 及实验
全部未执行。整体权威设计和 INDEX 由协调器对账，本文件只记录本切片。
