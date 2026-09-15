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
随后提交新 Root；`_activate_snapshot` 登记旧 generations，最终发布才调用
`schedule_retired_drain` 启动异步 Root.dispose。卸载因此必须继续调用
`wait_for_snapshot_drained(previous)`，等待这一张既有旧 snapshot 的实际回调完成。
该接口只 join 已登记 task，并直接传播 `_drain_failures`，不调用 `retry_drains`
自动重试失败资源。成功后 draining owner 已由 dispose 路径移除，才允许安装 owner
删除 cache。没有 active 的取消重试从该插件现存 draining generation 的
`runtime_snapshot` 找回同一旧 snapshot 并继续 join；不创建任务或恢复 generation 计数。
等待后仍有 draining owner 时拒绝卸载，交原 recovery/terminate 处理。
调用取消若发生在新 Root 已提交后的 join 窗口，既有 operation owner 会暂停新 stable
接纳。重试确认原 snapshot 已排空且目标没有失败 owner 后，由新的 operation 恢复这张
current committed Root 的接纳，再允许安装 owner 删除 cache；失败或取消当下不恢复。

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

新增 App 卸载测试覆盖 snapshot lease 与 fork 阻塞，并用事件确定性阻塞旧 snapshot
drain，证明 cache 在 Root 真正收尾前保留、完成后才删除。取消测试证明调用取消不删除
cache，重试只 join 同一正在运行的 drain；真实 Effect 关闭失败测试证明卸载不调用
`retry_drains`，cache、模块和失败 owner 保留。
测试、Gate、CI、lint、build、AST、产品命令均未运行；仅静态阅读与 diff 检查。

本次修复基线为 `e131101f`；修改前备份为
`/tmp/akasic-uninstall-join-e131101f-backup/source-before.tar`。
设计索引由主协调者接入。
