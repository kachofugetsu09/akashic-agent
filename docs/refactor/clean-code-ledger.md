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

### `a661c5f9` `fix(memory): 保持向量索引降级一致性`

- 范围：`MemoryStore2` 的 sqlite-vec 初始化、写入和删除故障路径。
- 历史依据：PR #72 的 embedding 维度配置、PR #41/#61 的单一 Memory runtime/engine、PR #75/#80 的显式失败与可恢复边界。
- 原问题：`vec_items` 写入或删除失败后 `_vec_enabled` 仍为真，主表与加速索引分叉，后续可能漏召回或继续触发 `OperationalError`。
- 为什么这样修改：`memory_items` 是 canonical 数据，`vec_items` 只是可选索引；store 层有明确恢复动作，应禁用已不可信索引并复用现有 fullscan。
- 不变量与拥有层：主表/索引同步和降级由 `MemoryStore2` 拥有；只处理 `sqlite3.Error`，embedding blob 等内部程序错误继续传播。
- 能力变化：正常 sqlite-vec KNN、排序、scope、hotness、事务和 freshness 不变；索引故障由错误或漏召回变为较慢但正确的 fullscan。
- 性能变化：正常路径不变；故障路径牺牲索引速度换取 canonical 正确性，不宣称提速。
- 测试新增：故障注入覆盖 vec 写入与删除失败，验证主表写入/删除结果和 fullscan 一致。
- 测试删除及原因：无。
- 验证结果：Memory 子系统 `124 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：禁用持续到进程重启，不自动重建损坏索引；这是避免不一致索引重新上线的保守语义。

### `ece6c837` `fix(plugins): 暴露 active 状态检查故障`

- 范围：插件 `is_active()` 协议边界与真实临时插件测试。
- 历史依据：PR #104 的程序化能力声明；PR #106 的单 Memory Engine active 过滤。
- 原问题：插件 `is_active()` 抛错后 runtime 记录 warning 并返回 `True`，把无法判断状态的插件错误加入 active generation 和 Drift skill roots。
- 为什么这样修改：runtime 无法从任意插件异常推导正确启用状态，只能补充插件身份并链式重抛。
- 不变量与拥有层：插件实现合法 `is_active()`；runtime 负责调用协议和错误上下文；未声明该方法仍按既有规则默认启用。
- 能力变化：正常 true/false 与缺失方法语义不变；故障插件由错误启用改为明确失败，generation/snapshot/lease/drain/rollback 未触及。
- 性能变化：非性能提交，正常调用次数不变。
- 测试新增：真实临时插件覆盖 `PluginManager.active_plugins()` 与 `RuntimeSnapshot.active_generations()` 的 cause 链。
- 测试删除及原因：无。
- 验证结果：相关 plugin 子系统 `145 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：第三方插件的 `is_active()` 旧错误现在会阻止状态枚举，这是预期 fail-loud 行为。

### `dffb1f69` `refactor(runtime): 收紧工具解锁结果边界`

- 范围：`ToolDiscoveryState` 的 tool-search JSON 解析与直接测试。
- 历史依据：PR #27/#31 的 lifecycle/tool discovery 阶段边界，PR #48 的工具循环与无限迭代能力。
- 原问题：宽泛 `except Exception` 会把解析函数内部程序错误也伪装成“没有工具可解锁”；现有英文 docstring 还保留无助于当前理解的搬迁历史。
- 为什么这样修改：JSON 语法和结构是明确外部边界，只恢复 `JSONDecodeError`、非对象顶层和非列表 `matched`；领域层继续过滤空名称与重复名称。
- 不变量与拥有层：输入参数的 `str` 类型由内部调用契约拥有；JSON 结构由解析边界拥有；工具名非空与去重由 `ToolDiscoveryState` 拥有。
- 能力变化：非法 JSON、`[]`、`null`、`matched=null` 仍不解锁工具；合法 unlocked/matched 顺序和去重不变；内部非 JSON 错误不再静默。
- 性能变化：非性能提交，仍是一次 JSON decode 和一次线性遍历。
- 测试新增：参数化覆盖合法 JSON 中的三种错误顶层/字段结构。
- 测试删除及原因：无。
- 验证结果：相关子系统 `58 passed`；pyright `0 errors`，无新增 warning；`git diff --check` 通过。
- 残余风险：旧的 `dict` 裸容器类型仍存在于同模块其他协议，已拒绝在本提交中用 `Any` 顺手掩盖，留给独立类型设计。

### `70f79c60` `refactor(plugins): 删除旧描述符声明标记`

- 范围：`ActivePluginInfo` 和直接构造测试。
- 历史依据：PR #96 引入旧 `.aka-plugin/plugin.json` descriptor；PR #104 明确删除 descriptor 并迁移到 `plugin.py` 程序化声明。
- 原问题：`declares_aka_plugin` 已无生产读取者，却继续暗示 runtime 支持已删除协议，并让测试持续构造无意义参数。
- 为什么这样修改：删除无主不变量和测试样板，不增加兼容层。
- 不变量与拥有层：插件能力声明只由当前程序化 `plugin.py` 协议拥有。
- 能力变化：无运行行为变化；skill/MCP 装配和 generation/snapshot/lease/rollback 未触及。
- 性能变化：非性能提交。
- 测试新增：无。
- 测试删除及原因：未删除测试，只移除四处失效构造参数。
- 验证结果：相关 plugin 测试 `78 passed`；pyright `0 errors` 且无新增 warning；字段全库搜索零残留；`git diff --check` 通过。
- 残余风险：无已知残余；若未来重新支持 descriptor，应以新协议显式设计，而不是恢复布尔标记。

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
