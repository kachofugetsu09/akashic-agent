# 内置插件逐项禁用验收

关联 #1179 E6。脚本：`scripts/builtin_disable_matrix_scenario.py`。

使用 `1a5327e495153c2b42d501ba42b671fafb2267f4` 的完整发行制品及正式离线 wheel，
通过安装入口提交 base bundle。每项先启动完整组合，再禁用一项并排空，
验证无关插件的 Fiber 不变且真实 apply 调用数不增加，随后以禁用后的持久选择冷启动。

依赖关系由实际组合图取得。新增 PENDING 的缺失 key 必须来自受影响的依赖链；
FAILED 一律失败。恢复在停机后通过 bundle 发布入口进行，不用外置安装接管内置数据身份。
消息与插件数据只在临时 workspace 创建；不访问正式数据。

结果：49/49 项通过，49 次禁用后的冷启动通过；下表全部 apply 增量为 0。
最终无活跃 generation，端点文件为空。CLI 返回 0；详细 JSON 由脚本写入临时证据目录。

```text
固定制品 ──正式安装──► 完整组合
                         │
                         ├──禁用一项──► 检查依赖链与 apply 次数
                         │                  │
                         │                  └──排空、冷启动、退出
                         └──离线发布恢复──► 下一项
```

| 禁用插件 | 无关插件数 | 无关 apply 增量 | 冷启动 |
|---|---:|---:|---|
| akasha@release | 47 | 0 | 通过 |
| akashic_clients@release | 48 | 0 | 通过 |
| akashic_sender@release | 48 | 0 | 通过 |
| assets@release | 40 | 0 | 通过 |
| channels@release | 46 | 0 | 通过 |
| codex@release | 48 | 0 | 通过 |
| commands@release | 42 | 0 | 通过 |
| compaction@release | 47 | 0 | 通过 |
| content@release | 33 | 0 | 通过 |
| context@release | 38 | 0 | 通过 |
| conversation-ui@release | 48 | 0 | 通过 |
| conversation@release | 45 | 0 | 通过 |
| delivery@release | 42 | 0 | 通过 |
| delivery_policy@release | 48 | 0 | 通过 |
| drift@release | 47 | 0 | 通过 |
| eventmail@release | 47 | 0 | 通过 |
| gateway@release | 46 | 0 | 通过 |
| gemini@release | 48 | 0 | 通过 |
| host_execution@release | 39 | 0 | 通过 |
| ledger@release | 23 | 0 | 通过 |
| ledger_invariants@release | 48 | 0 | 通过 |
| managed_processes@release | 48 | 0 | 通过 |
| markdown_memory@release | 48 | 0 | 通过 |
| mcp@release | 48 | 0 | 通过 |
| message_push@release | 48 | 0 | 通过 |
| models@release | 33 | 0 | 通过 |
| onboarding@release | 29 | 0 | 通过 |
| openai-compatible@release | 48 | 0 | 通过 |
| opencode-go@release | 48 | 0 | 通过 |
| programmatic@release | 48 | 0 | 通过 |
| projects@release | 48 | 0 | 通过 |
| prompt@release | 48 | 0 | 通过 |
| react@release | 45 | 0 | 通过 |
| reply@release | 48 | 0 | 通过 |
| reply_program@release | 46 | 0 | 通过 |
| runtime_inspection@release | 46 | 0 | 通过 |
| session_title@release | 48 | 0 | 通过 |
| shell-ui@release | 48 | 0 | 通过 |
| sources@release | 39 | 0 | 通过 |
| standard_tools@release | 41 | 0 | 通过 |
| telegram_channel@release | 48 | 0 | 通过 |
| telegram_sender@release | 48 | 0 | 通过 |
| timer@release | 47 | 0 | 通过 |
| tool_search@release | 48 | 0 | 通过 |
| tools@release | 37 | 0 | 通过 |
| turn_projection@release | 40 | 0 | 通过 |
| ui@release | 20 | 0 | 通过 |
| wake@release | 48 | 0 | 通过 |
| workloads@release | 48 | 0 | 通过 |

复现前先完成 `build_plugin_distribution.py` 与 `distribution_runtime.prepare_wheels`，再将制品目录传给场景脚本。该验收不声称没有配置的外部模型、Telegram 或容器服务已连通。
