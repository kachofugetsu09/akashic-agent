# 插件的第二 provider 验收

关联 #1179 P6/E7。替换必须换用独立实现；原实现增加注释不计作第二 provider。
每个场景明确覆盖的公开端口、实际行为和未覆盖范围，不把一个端口的通过扩展成全部实现等价。

| 范围 | 第二实现 | 实际证据 |
|---|---|---|
| `ledger.artifact_import.v1`、`ledger.artifact_read.v1` | `examples/artifact_provider` 用独立 SQLite BLOB 存储，未导入 Ledger 实现 | 正式安装、文件导入、完整和分片读取、旧引用读取、删掉原来源后重启、卸载；消费者 apply 增量 0 |
| `timers.v1` | `timer_plugin_scenario.py` 中事件循环 callback 实现 | 原实现使用 Task，第二实现使用 callback；真实 Scheduler 先后提交通知，取消、重启、卸载和无关 Fiber 保持 |

附件场景只接受本地文件，远程 URL 明确拒绝。它证明附件端口可以替换，
不声称这个示例可以替代 Ledger 的消息、同库事务、入站交接等端口。
第二实现重新导入相同内容后按公共摘要读取旧引用；这不是正式存量数据迁移。
原 Ledger 数据保留，所有操作均限于临时 workspace。

Timer 的硬依赖消费者按既有组合规则重新 apply；无关旁支不变。
请求期借用的附件消费者始终只 apply 一次。这个差别来自现有声明的依赖方式，
不能把硬依赖重新激活伪称为零变化。

```text
消费者 ──借用公开端口──► 当前 provider
  │                         │
  │                    卸载并排空
  │                         │
  └──同一读取/写入───► 独立实现
```

复现：运行 `scripts/artifact_provider_scenario.py` 和 `scripts/timer_plugin_scenario.py`。
场景返回 JSON 并明确各自临时目录。完整端口覆盖仍需逐项补齐 UI、Gateway、Models 与 Compaction 的证据。
