# 插件的第二 provider 验收

关联 #1179 P6/E7。替换必须换用独立实现；原实现增加注释不计作第二 provider。
每个场景明确覆盖的公开端口、实际行为和未覆盖范围，不把一个端口的通过扩展成全部实现等价。

| 范围 | 第二实现 | 实际证据 |
|---|---|---|
| `ledger.artifact_import.v1`、`ledger.artifact_read.v1` | `examples/artifact_provider` 用独立 SQLite BLOB 存储，未导入 Ledger 实现 | 正式安装、文件导入、完整和分片读取、旧引用读取、删掉原来源后重启、卸载；消费者 apply 增量 0 |
| `ui.plugin.v1`（公开 UiSlots 登记保持） | `examples/ui_query_provider` 串行查询，共用文件工作线程；不导入默认查询实现 | 同一目录、真实 JS、文件查询、旧线程池排空、重启、卸载；借用消费者 apply 增量 0 |
| Gateway 的 `session/list`、`message/read` | `examples/readonly_gateway` 独立 Unix JSON-RPC 服务，只依赖公开 Ledger/UI 合同 | 同一 Python SDK、真实账本、热替换、原 Ledger Fiber 不变、重启和端点排空 |
| `models.drivers.v1` 的驱动贡献 | `examples/text_model_driver` 独立 HTTP 文本驱动，复用同一已配置 driver identity | Models 与 Reply apply 增量 0；默认驱动、替代驱动、重启后各完成一次真实 CLI 回复 |
| `compaction.summaries.v1` | `examples/summary_archive` 独立 JSON 首代 v2 归档 | 真实 HTTP 模型压缩生成摘要；同一借用消费者读回相同来源与正文，apply 增量 0；卸载重开；原消息和 owner 行完整不变 |
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
场景返回 JSON 并明确各自临时目录。以上覆盖每个列出的能力族的代表端口；不等同于每个端口的完整语义等价验证。

UI 示例独立实现目录、资产和查询三个方法；贡献方和 provider 许可覆盖实际工作。它不提供默认并行配额与超时策略，因此只作组合可替换性证据，不作为产品默认配置。`ui_query_plugin_scenario.py` 负责实际安装和重开验收。

Gateway 示例只读，不接纳发送、订阅或管理操作；不声称 frames 回执合同已由第二实现覆盖。消费者是进程外 SDK，不存在插件 apply；同一 SDK 代码与查询结果保持不变，另核对 Ledger Fiber 未重建。复现 `scripts/gateway_second_provider_scenario.py <distribution>`。

Models 场景替换注册到 `models.drivers.v1` 的实际驱动贡献，保留 Models 的注册与执行 owner；不替换注册表本身。文本示例不提供工具调用、流和 embedding，非默认配置。复现 `scripts/model_driver_second_provider_scenario.py <distribution>`。

摘要示例只读导出的首代 v2 记录，不支持追加、后续父链、旧摘要格式或 `compaction.reader.v1` 的算法，不作为正式数据迁移。复现 `scripts/compaction_second_provider_scenario.py`：真实回复链触发压缩，14 条原消息与完整 owner 行在切换后保持不变。
