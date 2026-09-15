# 卸载沿完整 Root 关闭结算

状态：基于 `2ecf7c4d` 实现，仅静态复核，运行验收未执行。

## 真实路径与修复

`AppRuntime._uninstall_plugin` 先写禁用，再等待 Manager，最后才调用安装 owner
删除 cache 与安装清单项，业务数据保留。Manager 原先完成 `_deactivate_plugin`
整组换代后，继续索引 `_draining_generations[plugin_id]`。成功关闭已经通过
`_dispose_generation → _forget_drained_generation` 移除该项，因而可能 KeyError。

```text
禁用安装项 → 等待 snapshot 全部租约 → 关闭旧 Root → 发布不含目标的新 Root
                                                       │
                                无失败 owner ←────────┘
                                     │
                                     ▼
                             安装 owner 删除 cache
```

`_replace_formal_root` 在关闭旧 Root 前关闭接纳并等待 `wait_for_no_leases`，
随后 `_close_formal_root` 等待停止与 Root.dispose；关闭失败直接保留原 owner，
不能发布新组合或返回卸载成功。删除随后重复的逐 generation 等待和资源接管。
没有 active 但仍有 draining owner 时同样拒绝卸载，交原 recovery/terminate 处理。
不静默 pop 失败集合，不用空列表掩盖未完成关闭。

## 计数消费者

全仓静态搜索确认 `wait_for_generation_drained` 唯一生产调用者是上述旧循环。
`PluginGeneration.lease_count` 只由 SnapshotStore claim/fork/release 镜像加减，
唯一读取是该等待方法。删除字段、三处镜像循环和等待方法；保留 snapshot 的
实际租约计数、fork、归还通知、引用检查、排空与失败重试。

`tests/test_mobile_ui_scope.py` 两个 SimpleNamespace generation fixture 也删除
无用字段，原跨快照读取断言保留。`tests_scenarios/contracts/oracles.py` 的
`old_generation_lease_count` 是独立 oracle 参数，未读取 PluginGeneration 字段；
语义测试仍有输入消费者，保留。其他 lease_count 读者均属于 RuntimeSnapshot。

## 状态、验证与恢复

选择提交仍由原整体换代协议拥有；失败/未知提交不伪造成功。
卸载只在已授权入口减少目标安装 cache 和清单项，不删除 plugin-data、消息或归档。
本切片不操作正式 workspace/cache，不修改其他 writer 的组合、验证或 latest 代码。
基线仍有的 validation_candidate_plugin_ids 完全不动，集成不得恢复主线已删除字段。

新增 App 卸载测试覆盖 snapshot lease 与 fork 阻塞、整组实际关闭、成功删除 cache
并保留业务数据，以及真实 Effect 关闭失败后的 owner/cache 保留。
测试、Gate、CI、lint、build、AST、产品命令均未运行；仅静态阅读与 diff 检查。

源码恢复点为基线；备份为 `/tmp/akasic-uninstall-root-2ecf7c4d-backup/source-before.tar`，
同目录 `test_mobile_ui_scope.py` 保存 fixture 修改前内容。新增文件可随本提交撤销。
设计索引由主协调者接入。
