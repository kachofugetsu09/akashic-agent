# 0100 · 消息工具调用以日志为唯一事实，崩溃恢复退到 turn 粒度

- 状态：accepted
- 日期：2026-10-10
- 关联：[0059](0059-abandon-settles-tool-calls.md)、[0063](0063-execution-failures-have-terminal-results.md)、[0079](0079-parallel-tool-calls-commit-in-model-order.md)、[0092](0092-model-generation-recovers-until-output.md)、[0097](0097-request-deltas-belong-to-one-react-run.md)、[0099](0099-ledger-commits-use-wal-normal.md)
- 部分取代：0059/0063 中默认消息调用的 requested/prepared/started/done 回执；0097 的冻结请求增量

## 问题

每次模型响应到下一次请求之间，ReAct 为一次工具调用写入 3 次阶段回执，并为下一次请求再写入
1 次冻结请求记录。它们只在“进程在工具执行中或请求发出前崩溃”时有用，却让每一轮都付出
写事务和线程往返的代价。实测（100 轮 shell 剧本）每轮 7 次写事务，其中 4 次属于这两类记录。

收益也有限：

- Shell 的 `query()` 恒为 None，崩溃后只能记为错误；真正需要按 key 恢复的工具
  （message_push、subagent、scheduler、目录切换）都有自己的账本。
- 0099 已接受宿主故障可能丢失 start intent，保证本来只覆盖进程崩溃。
- 重启后用当前材料重新组装请求，得到的是最新的时间、记忆与工作目录，并非错误。

Output 中的工具调用就是意图，工具结果就是完成；
崩溃留下的未匹配调用由恢复方补一条“结果未知”，模型自行决定重试或先核实。

## 决定

1. 默认消息调用（结果身份为 `tool-result:{message_id}:{part_index}`）的正常执行不再写 Tools 阶段回执。
   已提交的 ToolCall 即请求事实；ToolResult 消息即终态，消息唯一索引保证每个调用只有一个结果。
2. 执行前在事务外核对来源提交权（`StartCheck` 接受 `None`）。核对之后到达的 abandon
   仍由 abandon 结算并取消现存 Task。无阶段记录不能证明未执行，记为 interrupted，
   提醒工具可能已部分执行、先检查当前状态；只有旧回执明确未启动时才记为 denied。
3. 恢复判据：ToolCall 所在 Output 的 `recorded_at` 早于本 gateway 进程启动时刻
   （`PROCESS_STARTED_AT`，位于 Core 契约模块，插件热更新不重置），且尚无 ToolResult，
   则不执行工具，直接提交 `error`“结果未知”。只读操作可由模型重试，有副作用的操作先核实。
4. ReAct 不再保存冻结请求与请求增量。每次组装模型请求都分配新 key；重启和用户显式重试
   使用当前材料发起新请求，不恢复旧请求。旧请求可能已被 provider 处理，因此接受重复付费。
   同一次模型调用内部的网络重试继续共享原 key 与 Models 的重试预算。
5. 独立程序调用（`program:` key）与自定义结果身份保持原回执协议，它们的结果只存在于回执中。
6. 并行调用按模型顺序提交（0079）只依赖进程内顺序门，不变。

## 影响

- 每轮少 4 次写事务；React 删除约 370 行冻结请求代码。
- 进程崩溃时正在执行的工具不会被重跑，结果记为未知；与此前 shell 的实际行为一致。
- 主机时钟大幅回拨可能把新调用误判为上个进程的调用，只会得到“结果未知”，不会重复执行。
- 已有的旧回执与冻结请求记录不迁移、不删除；旧版 owner_records 保留只读。
- 新请求身份不授权重跑已经有 ToolResult 的工具；未结工具仍按上述未知结果规则处理。

## 验收

- 100 轮 shell 剧本：owner_records 中不再出现 `plugin:tools` 与 `plugin:reply:generation` 写入，
  全部 100 个工具结果为 success。
- 单元测试全部通过。
- 进程在 shell 工具执行中（`sleep 30`）被 SIGKILL 整个进程组后重启：命令没有再次执行
  （副作用标记只出现一次），未结调用得到 `error`“结果未知”，模型据此继续并正常结束 turn。
