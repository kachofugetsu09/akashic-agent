# Clean Code 重构账本

本文档记录 `refactor/code-clean` 系列重构的决策依据、能力变化、性能数据和测试调整。每个被接受的提交都必须补充一条记录；没有测量或调用链证据的“优化”不得合并。

## 基线

- 基准提交：`3b456e7b`（PR #109 合并后）
- Python 测试：`1484 passed`，耗时 22.55 秒
- Pyright：`0 errors, 3119 warnings`
- 前端 TypeScript：`npm run typecheck` 通过
- 前端 ESLint：`0 errors, 3 warnings`，均为 `frontend/dashboard/src/main.tsx` 的既有 React Hook 依赖警告
- 工作区：除本地 `.codegraph/` 外无未提交代码
- 关键历史约束：PR #105 全能力热重载、PR #109 事件流唤醒、PR #90 主动发送串行、PR #89 shell 超时取消、PR #75 memory fail-stop

## 验收原则

1. 重构默认保持外部行为；能力变化必须明确列出并由测试覆盖。
2. 性能优化必须记录修改前后的同一 workload 数据，并证明 freshness、hot reload、错误传播和一致性未退化。
3. 删除或保留防御性检查时，必须说明不变量、拥有层、上游保证和真实可达违反路径。
4. 测试只保留能够保护真实契约、历史回归或性能边界的内容；删除测试必须记录其重复、错误耦合或已失效的原因。
5. God file 是否拆分以阅读成本为准，不以行数为准。若拆分增加跨文件跳转、隐藏弱类型数据流或割裂同一状态机，应保留同文件并在函数级整理。
6. 新增或改写的 docstring 与注释使用简洁中文；保留解释约束、所有权和 workaround 的有效注释。

## 变更记录模板

### `f82be7b6` `perf(runtime): 回收空闲聊天通道状态`

- 范围：`bus/queue.py` 的 `ChatLane` 与直接回归测试。
- 历史依据：PR #90 固化被动优先、主动 FIFO 和取消 ticket 语义；PR #97 固化中断恢复边界。本次没有触碰 lifecycle、interrupt 或 turn 内容。
- 原问题：`ChatLane._states` 永久保留历史见过的 chat，唯一 chat 数持续增长时形成无界内存占用。
- 为什么这样修改：为每次公开操作成对持有状态引用，只在没有活跃用户、被动计数、发送、未完成 ticket 和取消残留时回收；等待者持有引用，因此不会与新进入者分裂到两个状态锁。
- 不变量与拥有层：`active_users` 由 `_acquire_state` / `_release_state` 唯一维护；FIFO 和取消 ticket 仍由同一 `_ChatLaneState` 拥有。
- 能力变化：串行、FIFO、被动优先、取消恢复和异常传播不变；空闲 chat 不再保留无语义状态。
- 性能变化：20,000 个唯一 chat 顺序执行 pending/done 后，保留状态由 20,000 降至 0；当前 tracemalloc 由 32,702,434 B 降至 374 B，峰值由 32,703,122 B 降至 3,026 B。
- 测试新增：覆盖 FIFO 完成、取消 waiter、被动生命周期和发送异常后的回收。
- 测试删除及原因：无。
- 验证结果：相关子系统 `48 passed`；修改文件 pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：回收依赖 asyncio 单线程事件循环中 acquire/release 之间无 `await` 的原子执行语义；跨线程调用不在 `ChatLane` 契约内。

### `3b962903` `fix(memory): 暴露向量存储故障`

- 范围：`memory2/retriever.py` 的统一向量检索链及直接回归测试。
- 历史依据：PR #23/#61 统一召回与 memory engine 协议；PR #75/#80 确立 memory 失败只有在存在明确恢复动作时才能恢复；PR #106 保证唯一 Memory Engine。
- 原问题：批量向量存储失败会被宽泛捕获，随后逐向量重复同一存储调用并继续吞错，最终把存储或反序列化故障伪装成空召回。
- 为什么这样修改：`MemoryStore2.vector_search_batch` 已拥有 sqlite-vec/full-scan 选择、时间过滤、反序列化和批量结果形状；Retriever 没有第二种恢复手段，应让该层错误向上传播。
- 不变量与拥有层：非空 vectors 必须获得同长度 outer result，由 `MemoryStore2` 保证；embedding 是外部边界，单 lane embedding 失败仍可跳过并保留关键词检索。
- 能力变化：正常 vector + keyword + RRF、零向量命中后的关键词召回、scope、top-k 和时间过滤不变；存储损坏由静默空结果变为显式失败。
- 性能变化：正常路径调用次数不变；故障路径由 `1 + N` 次重复存储调用收敛为 1 次后立即失败；生产代码净减少 21 行。
- 测试新增：覆盖向量存储失败向上传播且不会继续执行关键词 lane。
- 测试删除及原因：无。
- 验证结果：独立复验 59 个相关测试通过；修改文件 pyright `0 errors`，总 warning 由 150 降至 128；`git diff --check` 通过。
- 残余风险：该变化会让过去被误判为“无记忆”的存储故障显式中止 recall，这是预期错误语义修复。

### `9bb4913d` `perf(plugins): 复用热重载发现快照`

- 范围：`PluginManager.reconcile_changed` / `_prepare_changed` 与多插件热重载测试。
- 历史依据：PR #51 的拓扑依赖、PR #95 的代际 Skill Catalog、PR #104 的程序化能力声明、PR #105 的 generation/snapshot/lease/rollback 事务。
- 原问题：一次 reconciliation 已发现完整拓扑，之后每个活跃插件和每个变化候选又重复完整 `discover()`，调用次数为 `1 + N + C`。
- 为什么这样修改：同一发布事务应使用同一个 discovery topology；watcher 的 revision 在事务外采样，中途变化会在下一轮形成新 revision，不需要在同一事务内部漂移拓扑。
- 不变量与拥有层：单轮 topology 由 reconciliation 拥有；源码 revision、candidate gate、snapshot 编译和下一轮 freshness 仍由原有层负责。
- 能力变化：同轮一致性增强；generation、gate、snapshot、lease、drain、abort、rollback 和下一轮 hot reload 不变。
- 性能变化：两个活跃插件同时变化时 `discover()` 从 5 次降至 1 次，减少 80%；一般情况从 `1 + N + C` 降至固定 1 次。
- 测试新增：在既有多插件换代测试中增加 discover 次数断言，同时保留最终 snapshot 包含两个新 generation 的能力断言。
- 测试删除及原因：无；复用已有昂贵 fixture，避免新增重复测试。
- 验证结果：`137 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：单轮扫描后的文件变化不会混入当前事务，而由 watcher 下一轮重新 reconcile；这是 PR #105 的代际一致性边界。

### `c845327b` `fix(runtime): 暴露主动发送异常`

- 范围：主动发送的 `PushToolOutboundPort`、`TurnOrchestrator` 与直接测试。
- 历史依据：PR #90 的 ChatLane/outbound 串行链路，PR #97 的中断恢复与可见历史可信边界，PR #27/#31 的 persist/dispatch 和 lifecycle 职责。
- 原问题：端口把所有意外异常静默转换成 `False`，无法区分正常业务失败与 channel/tool 故障；同时用字符串归一化掩盖内部 `OutboundDispatch` 契约错误。
- 为什么这样修改：端口传播意外异常，由拥有恢复动作的 orchestrator 记录完整堆栈、保持 `sent=False`、禁止未送达消息落库并执行失败副作用。
- 不变量与拥有层：channel/chat_id/content/media 的结构由 `OutboundDispatch` 构造链拥有；“目标和内容可发送”仍由端口判断；失败恢复由 orchestrator 拥有。
- 能力变化：正常文本、多媒体发送与业务失败字符串不变；意外异常从无诊断 `False` 变为有堆栈的原失败路径；ChatLane 串行和持久化顺序不变。
- 性能变化：非性能提交，发送次数和调用顺序不变。
- 测试新增：覆盖端口异常传播，以及 orchestrator 记录错误、不落库并运行 failure effect。
- 测试删除及原因：无。
- 验证结果：Runtime/turn 子系统 `125 passed`；pyright `0 errors`，4 个既有容器类型 warning；`git diff --check` 通过。
- 残余风险：多媒体分批发送中后续图片失败时，用户可能已收到前序内容但整次 dispatch 仍判失败；这是既有非事务性外部发送语义，本提交未扩大范围。

### `<commit>` `<title>`

- 范围：
- 历史依据：
- 原问题：
- 为什么这样修改：
- 不变量与拥有层：
- 能力变化：
- 性能变化：
- 测试新增：
- 测试删除及原因：
- 验证结果：
- 残余风险：
