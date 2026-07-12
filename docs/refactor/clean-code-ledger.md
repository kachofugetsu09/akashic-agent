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

### `ba83aab2` `refactor(runtime): 收紧出站总线契约`

- 范围：`BusOutboundPort`、真实 `MessageBus` 测试夹具和直接出站测试。
- 历史依据：PR #90 的 MessageBus/ChatLane 被动出站链，PR #27/#31 的 after-turn dispatch 边界。
- 原问题：端口用 `Any + inspect.isawaitable` 兼容不存在的同步 bus，并对 typed dataclass 容器重复提供空值 fallback；测试的 `MagicMock` 反向维持了假契约。
- 为什么这样修改：生产构造链保证 `MessageBus`，其 `publish_outbound` 明确为 async；直接 await 真实契约并让发布异常继续传播。
- 不变量与拥有层：bus 类型由 `AgentLoopDeps`/bootstrap wiring 拥有；metadata/media 非空容器由 `OutboundDispatch` dataclass 拥有。
- 能力变化：channel、chat_id、content、thinking、metadata、media、ChatLane 计数和异常传播不变；测试与生产异步契约一致。
- 性能变化：删除动态 awaitable 判断和无效 fallback，但未做稳定 benchmark，不声明性能收益。
- 测试新增：使用真实 MessageBus 验证完整 typed `OutboundMessage`。
- 测试删除及原因：无；将违反生产契约的 MagicMock 夹具改为真实 bus。
- 验证结果：相关 runtime/turn 测试 `36 passed`；修改文件 pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：直接测试读取 MessageBus 私有队列以避免启动长期 dispatch loop；生产 API 语义仍由 publish/dispatch 集成测试覆盖。

### `8d1c4589` `fix(session): 暴露 metadata 损坏`

- 范围：`sessions.metadata` 数据库反序列化边界、SessionManager 转发层和三条读取入口测试。
- 历史依据：`708d6f251` 将 JSONL session 迁移到中心 SQLite；PR #75/#80 确立无恢复动作时的 fail-stop。
- 原问题：损坏 JSON 被 manager 宽泛捕获并归一化为整个 channel 空列表；合法 JSON list/string 会穿透到下游 `.get()` 才无上下文失败。
- 为什么这样修改：store 在读取 SQLite 时统一解析并验证 JSON object，错误携带 session key；manager 信任边界后的 dict。
- 不变量与拥有层：metadata JSON schema 由 `SessionStore` 拥有；NULL 是 schema 允许的旧记录，继续明确解释为 `{}`；identity index 无修复损坏数据的能力。
- 能力变化：有效 metadata、NULL 兼容、排序、cache 和 identity 映射不变；损坏数据由空结果或延迟错误变为带 key 的即时 `ValueError`。
- 性能变化：仍为一次 SQL 查询和每行一次 JSON 解析，无新增 I/O。
- 测试新增：注入损坏 JSON 和非 object JSON，覆盖 channel metadata、单 session metadata、dashboard 列表三个入口。
- 测试删除及原因：无。
- 验证结果：Session 相关调用方 `82 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：数据库中已有损坏 metadata 会在首次读取时显式暴露，需要人工修复数据；这是预期行为。

### `6f50a391` `test(runtime): 使用真实异步消息总线`

- 范围：所有直接构造 `AgentLoopDeps` 的测试夹具。
- 历史依据：`ba83aab2` 收紧 `BusOutboundPort` 后的集成回归。
- 原问题：10 处测试用同步 `MagicMock` 伪造生产中明确为异步 `MessageBus` 的依赖，其中两条 spawn completion 流程在完整测试中触发 `TypeError`。
- 为什么这样修改：统一改用真实 `MessageBus`，让测试遵循生产构造契约，不恢复同步兼容层。
- 不变量与拥有层：bus 类型与 async publish 由 `AgentLoopDeps`/`MessageBus` 拥有。
- 能力变化：无生产行为变化；测试现在能覆盖真实出站类型。
- 性能变化：非性能提交。
- 测试新增：无。
- 测试删除及原因：无；替换错误夹具。
- 验证结果：相关测试 `49 passed`，完整测试 `1497 passed`；pyright `0 errors, 0 warnings`；全库同类 `bus=MagicMock()` 搜索零残留。
- 残余风险：这笔修复证明目标测试不足以验收公共契约变更；后续公共类型收紧必须运行完整测试。

### `6c7a4ba5` `fix(plugins): 校验 KV 根节点结构`

- 范围：`PluginKVStore._read()` 数据文件反序列化边界与真实磁盘测试。
- 历史依据：插件 KV 可被用户、旧版本和外部插件绕过正常 `_write()` 直接修改。
- 原问题：合法 JSON array/scalar 会穿透边界，在后续 `.get()` 或赋值处以无文件上下文的异常失败。
- 为什么这样修改：KV 根节点必须是 JSON object；在唯一读取边界校验并以包含文件路径的 `ValueError` 失败，非法 JSON 继续保留 `JSONDecodeError`。
- 不变量与拥有层：KV object schema 由 `PluginKVStore._read()` 拥有；正常 `_write()` 始终写入 dict。
- 能力变化：正常 get/set/increment 和跨 manager 持久化不变；错误更早且带路径；plugin generation/snapshot 状态机未触及。
- 性能变化：非性能提交，正常路径仅增加一次 `isinstance`。
- 测试新增：真实 `.kv.json` 数组根节点拒绝测试。
- 测试删除及原因：无。
- 验证结果：相关 plugin 测试 `142 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：已有非 object KV 文件会在首次读取时显式失败，需要插件作者修复数据。

### `9d449162` `fix(session): 校验缓存向量维度`

- 范围：`MessageEmbeddingStore` 的向量写入与 `sessions.db` 缓存反序列化边界。
- 历史依据：PR #109 引入共享 message embedding cache，要求 cache hit 表示可直接复用的完整向量。
- 原问题：写入允许空 embedding；读取忽略持久化 `dim`，空 BLOB 会被错误计为 cache hit，BLOB/dim 不一致会按实际字节静默解码。
- 为什么这样修改：upsert 拒绝空向量；读取统一校验 BLOB 类型、正整数 dim 和 `len(blob) == dim * 4`，错误携带 message/model/dim/bytes。
- 不变量与拥有层：非空向量由 upsert 写边界拥有；持久化 BLOB/dim 一致性由读取边界拥有；元素数值错误继续由 `struct.pack` fail-loud，不重复检查。
- 能力变化：合法 cache、content hash miss、时间 cutoff、replay 顺序和 legacy migration 不变；空向量和损坏缓存变为即时失败。
- 性能变化：SQL 次数不变，正常读取增加常数级类型与长度比较，不宣称提速。
- 测试新增：空 embedding 写入拒绝且无缓存残留；空 BLOB 和维度/字节不一致覆盖 get/list/list_until。
- 测试删除及原因：无。
- 验证结果：Akasha/replay 相关 `84 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：已有损坏 cache 会阻止 replay，需删除或重建对应缓存；这是避免错误 cache hit 的预期行为。

### `943820ee` `refactor(runtime): 收紧回合副作用契约`

- 范围：`TurnResult` 三类副作用集合、`TurnOrchestrator` 执行边界和直接测试替身。
- 历史依据：PR #27/#31 将副作用放在明确的 lifecycle/commit 阶段；PR #90/#97 要求保持发送顺序，并禁止未送达消息进入历史。
- 原问题：副作用以 `list[Any]` 表示，orchestrator 用 `inspect.isawaitable` 兼容没有生产调用者的同步假实现。
- 为什么这样修改：现有生产副作用全部实现异步 `TurnSideEffect` 协议；将三类集合收紧到该协议并直接 await，让协议错误即时暴露。
- 不变量与拥有层：副作用的异步调用契约由 `TurnSideEffect` 拥有；通用、成功和失败副作用的选择与次序由 orchestrator 拥有。
- 能力变化：通用副作用仍先于 dispatch；成功/失败副作用仍只进入对应分支；单项异常仍记录并继续；持久化和 ChatLane 语义不变。
- 性能变化：删除一次动态 awaitable 判断，但无独立 benchmark，不声明性能收益。
- 测试新增：无；唯一同步测试替身改为真实异步协议。
- 测试删除及原因：无。
- 验证结果：相关 Runtime/proactive 测试 `144 passed`；副手完整测试 `1501 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：无已知生产同步副作用；未来扩展必须显式实现协议。

### `e16f2dcc` `fix(plugins): 拒绝无效清理动作`

- 范围：`PluginScope.defer()` 动态插件边界、`PluginContext` cleanup/task 类型和直接测试。
- 历史依据：PR #105 的候选初始化、回滚和 generation 换代要求资源清理动作在候选发布前有效。
- 原问题：动态外部插件可绕过静态类型注册不可调用对象，错误延迟到卸载或换代时才暴露，候选甚至可能已经发布。
- 为什么这样修改：在 cleanup 唯一注册入口验证 callable，并携带 plugin/resource 身份抛出 `TypeError`；同时把 context 类型收紧为 `Cleanup` 和 `Task[T]`。
- 不变量与拥有层：进入 scope 栈的 cleanup 必须可调用，该不变量由 `PluginScope.defer()` 唯一拥有；静态类型不能覆盖动态插件边界。
- 能力变化：合法同步/异步 cleanup、逆序排空、取消传播、task/process 跟踪不变；无效候选在 initialize/rollback 阶段提前失败。
- 性能变化：正常注册仅增加一次常数级 callable 检查，不声明性能收益。
- 测试新增：动态注册不可调用 cleanup 的边界测试。
- 测试删除及原因：无；generation/snapshot/lease/drain/abort/rollback 测试全部保留。
- 验证结果：plugin 相关测试 `145 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：`manager.py` 的候选 gate 和 watcher retry 属于更大的状态协议，本提交未改动。

### `7b4b7821` `refactor(schedule): 收紧时间展示降级边界`

- 范围：调度工具注册后的时间展示、历史任务列表展示及直接测试。
- 历史依据：PR #52 的 scheduler 后台任务隔离；PR #79/#89 的轮询与取消边界；PR #107 的 MCP 超时透传均未触及。
- 原问题：任务成功注册后，展示阶段用宽泛 `except Exception` 把内部程序错误也伪装成正常 ISO fallback。
- 为什么这样修改：只恢复 datetime/时区格式化真实会产生且当前位置能处理的 `TypeError`、`ValueError`、`OverflowError`、`OSError`；历史失效时区额外处理 `ZoneInfoNotFoundError`。
- 不变量与拥有层：调度参数结构由工具输入边界拥有；`ScheduledJob.fire_at` 的 datetime 契约由 scheduler 构造/反序列化层拥有；展示层只拥有格式降级。
- 能力变化：合法注册、循环任务和取消不变；无效展示时区/request_time 仍回退 ISO；违反内部 job 契约的错误改为显式失败。
- 性能变化：非性能提交。
- 测试新增：无效字符串/错误类型 request_time 和历史失效时区的展示回退。
- 测试删除及原因：无。
- 验证结果：定向 `39 passed`；副手完整测试 `1505 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：ToolRegistry 当前不主动调用 schema validator，错误类型参数仍可从动态调用进入工具，因此该 TypeError 恢复路径真实可达。

### `6cc15427` `fix(proactive): 暴露会话读取故障`

- 范围：`Sensor` 的普通/主动历史读取、时间戳解析、配置与返回类型及直接测试。
- 历史依据：PR #103 的 Gate→Fetch→Judge→Resolve→Deliver 次序；PR #101 的 Drift 时钟；PR #67 的 read-only 主动召回。
- 原问题：sessions SQLite 关闭、schema 或加载故障被两个入口宽泛捕获并返回空列表，普通链误判为无上下文，主动链还可能绕过去重造成重复投递。
- 为什么这样修改：Sensor 没有恢复数据库故障的能力；让错误传播到 `ProactiveLoop._tick_bound()` 现有的完整日志与重抛边界，仅保留非法旧时间戳到 `None` 的明确字段级恢复。
- 不变量与拥有层：Session 持久化错误由 SessionManager/Store 拥有；Sensor 只读取筛选；tick 级失败可观察性由 loop 拥有。
- 能力变化：无目标 session 仍返回空历史；角色、context frame、长度、主动顺序与状态标签不变；数据库故障由假空结果变为明确失败。
- 性能变化：数据库读取次数和正常筛选复杂度不变。
- 测试新增：普通筛选/截断、主动顺序/metadata、两个真实入口的已关闭 SQLite 传播。
- 测试删除及原因：无。
- 验证结果：主动相关组合 `416 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：tick 失败沿既有 supervisor 策略进入下一轮；本提交未改变重试节奏。

### `bba83b52` `perf(akasha): 合并批量删除事务`

- 范围：Akasha sidecar 节点/关联边批量物理删除和存储回归测试。
- 历史依据：PR #65 的 sidecar 存储边界；PR #66 的快速路径一致性；PR #67/#68 的 scheduler/read-only 隔离与 live/replay parity 均未触及。
- 原问题：批量接口逐项获取锁并提交事务，200 项产生 200 次 COMMIT；中途失败还会留下部分删除结果。
- 为什么这样修改：用一次锁和一次 SQLite 事务包住逐 ID `executemany`；不构造无界 `IN (...)`，避免 dashboard 批量输入触发 SQLite 参数上限。
- 不变量与拥有层：节点与全部入边/出边的一致物理删除由 AkashaStore 拥有；缺失和重复 ID 不增加删除计数。
- 能力变化：最终删除计数、缺失/重复、边清理与无关边保留不变；批次从部分提交升级为全有或全无。
- 性能变化：同一 200 项 workload、12 次测量，中位耗时 `10.208 ms → 0.926 ms`，约 `11.0x`；COMMIT `200 → 1`。
- 测试新增：成功路径覆盖计数/重复/缺失/入出边；SQLite trigger 在批次中间失败，验证节点和边全部 rollback。
- 测试删除及原因：无。
- 验证结果：`tests/test_akasha_plugin.py` `37 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：大批次仍逐 ID 执行 SQL，避免参数上限但持锁时间随批量线性增长；这是相对原实现更短的同量工作。

### `c1d37dbd` `fix(mcp): 拒绝非对象调用错误`

- 范围：`McpClient.call()` 的 JSON-RPC `tools/call` error 边界和真实 stdio 响应测试。
- 历史依据：PR #105 的 MCP generation/连接清理；PR #107 的 180 秒超时贯通；PR #89 的取消边界均未触及。
- 原问题：代码无条件对 `error` 调用 `.get()`；非对象合法 JSON 会产生无 server/tool 上下文的 `AttributeError`。
- 为什么这样修改：JSON-RPC error 必须是 object；标准对象保持既有用户可见字符串，字符串/列表等协议损坏携带 server、tool、类型和值抛出 `RuntimeError`，不归一化为普通工具失败。
- 不变量与拥有层：JSON 解码和 response id 由 `_recv` 拥有；tools/call error schema 与用户可见转换由 `McpClient.call()` 拥有。
- 能力变化：正常 content、标准远端错误、同 server 串行、timeout/cancel/disconnect 不变；损坏 error 从偶发属性错误变为有上下文的 fail-loud。
- 性能变化：仅错误路径增加常数级类型判断，不声明性能收益。
- 测试新增：标准 error object，以及字符串/列表 error 的拒绝和上下文断言。
- 测试删除及原因：无。
- 验证结果：MCP/热重载相关 `30 passed`；副手完整测试 `1508 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：标准 object 内部字段继续保持既有宽松展示，不在本提交扩大协议迁移范围。

### `8181bd51` `perf(proactive): 初始化时完成日志迁移`

- 范围：`ProactiveStateStore` tick log schema 迁移、finish 热路径及真实 SQLite 测试。
- 历史依据：PR #103/#109 的主动 tick 与事件流架构；迁移不改变 phase/order、delivery/feedback、hot reload 或 MCP poll。
- 原问题：每次 tick finish 都执行 `PRAGMA table_info(tick_log)`，但 schema 在一个 store 生命周期内只可能由初始化改变。
- 为什么这样修改：把旧库 `proactive_effects_json` 补列放入 `_init_schema()` 的建表事务；业务写入信任初始化后的 schema。
- 不变量与拥有层：finish 前列必须存在，该不变量由 `ProactiveStateStore._init_schema()` 唯一拥有；业务写入不重复验证。
- 能力变化：新库、旧库迁移、tick log JSON、dashboard 查询和提交时机不变；旧库在首次初始化即完成迁移。
- 性能变化：10 次 finish 的 schema 查询 `10 → 0`；包含初始化则 `10 → 1`，总数减少 90%，热路径减少 100%。
- 测试新增：真实旧 schema 初始化补列并写入；SQLite trace 断言连续 finish 不再查询 schema。
- 测试删除及原因：无。
- 验证结果：主动相关组合 `418 passed`、dashboard `25 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：初始化本身仍执行一次 `PRAGMA table_info`，这是兼容旧库所需的一次性成本。

### `0b916a57` `fix(mcp): 校验工具调用结果结构`

- 范围：MCP `tools/call` 成功结果的 result/content/block/text 边界及 stdio 响应测试。
- 历史依据：客户端固定协商 MCP `2024-11-05`；PR #107 的 timeout 透传和 PR #105 的连接/代际清理未修改。
- 原问题：损坏 result 有时被字符串化为“成功”工具输出，有时产生无字段上下文的属性/类型错误。
- 为什么这样修改：按已协商协议验证 result object、必需 content list、每个内容对象和 text 字符串；字段错误携带 server/tool/path/type/value 失败。
- 不变量与拥有层：`_recv` 拥有 JSON/id；`_response_result` 拥有 result object；`McpClient.call()` 拥有 CallToolResult content schema。
- 能力变化：标准 text block 仍拼接文本；合法 image/resource 等无 text 对象继续保持既有字典字符串；锁、超时、取消、断连和标准 error 不变。
- 性能变化：成功响应增加线性类型校验，与原本遍历 content 同阶，不声明性能收益。
- 测试新增：result 标量、缺失/错误 content、标量 block、非字符串 text 五条损坏路径。
- 测试删除及原因：无。
- 验证结果：MCP/热重载相关 `35 passed`；副手完整测试 `1519 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：合法非文本内容仍以 Python dict 字符串传给模型，这是既有表示协议，后续若需多模态 ToolResult 应独立设计。

### `3f4e2645` `fix(akasha): 暴露 dashboard 配置错误`

- 范围：Akasha dashboard 注册时的插件配置来源与损坏配置回归。
- 历史依据：PR #93 的 snapshot freshness/旧坐标复用；PR #105 的 candidate 初始化/回滚边界。
- 原问题：dashboard 忽略 runtime 传入的真实 `plugin_dir`，并捕获所有配置加载异常后退回默认配置，可能连接或创建错误 sidecar。
- 为什么这样修改：直接从 canonical plugin_dir 调用统一配置加载器；配置不存在仍由加载器使用默认值，配置存在但 TOML 损坏/不可读则阻止注册。
- 不变量与拥有层：外部 TOML 结构与读取由 `load_akasha_config` 拥有；dashboard 没有推导正确 DB 路径的恢复能力。
- 能力变化：合法配置和缺失配置默认值不变；损坏配置从静默换库变为原始配置错误；recall/replay/snapshot 算法未触及。
- 性能变化：删除一层 helper 和异常分支，无性能声明。
- 测试新增：真实非法 TOML 在 dashboard 注册时传播 `TOMLDecodeError`。
- 测试删除及原因：无。
- 验证结果：Akasha/dashboard 相关 `38 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：配置字段的数值转换仍有历史默认策略，需要按字段契约另行审计。

### `f45b899e` `fix(proactive): 暴露上下文组装故障`

- 范围：主动 prompt 的 MemoryProfile/workspace 规则读取、类型协议和 facade 测试替身。
- 历史依据：PR #101 的 runtime clock 和 Drift 规则；PR #103 的 Prepare→Judge→Resolve→Deliver 次序。
- 原问题：prompt builder 分别吞掉四个任意异常，把画像、长期记忆、近期上下文和 workspace 规则故障伪装成内容为空；旧测试假对象缺少真实协议方法也被掩盖。
- 为什么这样修改：MemoryProfile 是完整内部协议；workspace callback 已在文件 I/O 边界记录失败并返回旧缓存，组装层没有第二种恢复动作。
- 不变量与拥有层：profile 读取由 MemoryProfileApi/runtime 拥有；workspace 文件恢复由 loop callback 拥有；prompt builder 只组装；tick supervisor 记录并隔离整轮错误。
- 能力变化：正常区块、空内容跳过和 runtime clock 位置不变；依赖故障从缺块假成功变为明确 tick 失败。
- 性能变化：读取次数不变，删除重复异常框架，无性能声明。
- 测试新增：三个 profile 方法和 workspace callback 的失败传播；修复 facade 使其实现真实读取协议。
- 测试删除及原因：无。
- 验证结果：主动相关组合 `422 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：workspace I/O 仍按设计可降级到旧缓存并记录 warning；这是拥有恢复动作的边界，不属于静默失败。

### `f0af9b55` `fix(peer-agent): reject missing remote task id`

- 范围：A2A `message/send` 非阻塞提交响应、Poller 注册和新直接测试。
- 历史依据：现有请求固定 `configuration.blocking=false`，随后必须以服务端 Task ID 调用异步 Poller。
- 原问题：响应缺少 `result.id` 时生成从未发给服务端的本地 UUID，返回 submitted 并永久轮询不存在的任务。
- 为什么这样修改：验证顶层/result object 和非空字符串 Task ID；协议损坏进入既有公开提交失败结果，且不注册 Poller。
- 不变量与拥有层：A2A HTTP/JSON 响应由 `_submit_task` 拥有；只有服务端 Task ID 能进入 Poller；`execute()` 拥有对用户可见的提交失败转换。
- 能力变化：合法异步 Task、冷启动、channel/chat 绑定与后台通知不变；假成功被删除。
- 性能变化：仅响应边界增加常数级校验，不声明性能收益。
- 测试新增：服务端 ID 正常注册，以及数组响应、缺失/空/非对象 result、空/非字符串 id 共七条路径。
- 测试删除及原因：无。
- 验证结果：定向 `7 passed`；副手完整测试 `1528 passed`；pyright `0 errors`；`git diff --check` 通过。
- 残余风险：若未来改为 blocking 请求允许直接 Message，必须单独设计同步结果分支，不能复用异步 Poller。

### `e6187d6f` `fix(akasha): 拒绝非法显式配置值`

- 范围：Akasha 配置字符串/整数/浮点解析及真实 TOML 参数化测试。
- 历史依据：统一 `load_akasha_config` 被 candidate 初始化、replay、dashboard 和诊断命令共同使用，是唯一 schema owner。
- 原问题：显式非法值被静默替换为默认值，且 `bool` 会因 Python 是 `int` 子类而可能穿透数字判断。
- 为什么这样修改：文件或字段缺失才使用默认；合法整数、浮点和历史数字字符串继续支持；显式错误携带字段名失败。
- 不变量与拥有层：TOML 类型收敛由配置加载器拥有，上游无法保证手工文件；算法层信任强类型且有限的数值。
- 能力变化：缺失配置默认值和合法历史写法不变；错误 db_path、非数字字符串、非整数 float、nan、bool、容器改为 fail-fast。
- 性能变化：仅初始化解析路径，无性能声明。
- 测试新增：默认/合法数字字符串，以及上述显式错误类型；特别覆盖 int/float 字段的 bool 与容器。
- 测试删除及原因：无。
- 验证结果：配置定向 `8 passed`、Akasha+fast replay parity `46 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：字段领域范围未在本提交新增限制；需要先从算法和历史配置证明范围，避免武断裁剪能力。

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
