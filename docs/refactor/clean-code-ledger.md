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

### `94e9ac6a` `fix(akasha): 对齐来源引用失败语义`

- 范围：live/replay query log 的 source_ref 统计和内部共享 helper。
- 历史依据：PR #66 要求离线快速 replay 与线上单轮路径保持一致。
- 原问题：live 独立实现并两次吞掉任意 JSON 错误，写入 `source_ref_count=0` 的假成功诊断；replay 对相同内部契约则直接失败。
- 为什么这样修改：`_load_turn_card` 唯一生成 JSON list source_ref；live/replay 共用解析逻辑，内部契约违反时不应由诊断写入层恢复。
- 不变量与拥有层：card source_ref 结构由 card 构造拥有；query log 只统计并持久化，不能把损坏解释为空来源。
- 能力变化：合法引用计数和 query log 内容不变；损坏引用从假成功改为失败且不写半条诊断。
- 性能变化：同阶线性解析，删除重复实现，无性能声明。
- 测试新增：构造损坏内部 card，断言 JSON 错误传播且 query log 总数仍为零。
- 测试删除及原因：无。
- 验证结果：Akasha+fast replay parity `50 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：source_ref 仍是 JSON 字符串内部表示；若未来开放外部构造，应升级为 typed 字段而不是下游重复校验。

### `54da202c` `fix(scheduler): reject corrupt persisted jobs`

- 范围：JobStore 严格读取、schema 反序列化、原子保存和持久化测试。
- 历史依据：PR #52 的 scheduler 后台任务语义；PR #79/#89 的 timeout/cancel 行为未改。
- 原问题：坏 JSON、顶层/任务结构和时间戳损坏全部被当成空任务集；下一次 add/cancel/save 会覆盖原文件并丢失任务；非原子保存还会制造半文件。
- 为什么这样修改：文件不存在才为空；严格 read_text/json.loads 保留 I/O/JSON 原异常；成功解析后的 schema 错误带 path/index/field；保存改用既有同目录原子替换。
- 不变量与拥有层：JSON→ScheduledJob 由 JobStore 拥有，下游 SchedulerService 信任完整任务；读/解析错误不能伪装为无任务。
- 能力变化：合法 roundtrip、misfire/recovery、执行和取消不变；损坏文件阻止启动/覆盖；保存具备原子替换。
- 性能变化：写入增加一次同目录临时文件 rename，以可靠性为目标，不声明提速。
- 测试新增：原始 JSONDecodeError/PermissionError、顶层/条目 schema、缺失/损坏时间字段与 roundtrip。
- 测试删除及原因：无。
- 验证结果：定向 `33 passed`；副手完整测试 `1539 passed`；`git diff --check` 通过；worktree pyright 仅缺可选环境包产生既有 missing-import，新增路径无错误。
- 残余风险：已有损坏 jobs.json 会在启动时明确失败，需要人工修复或从备份恢复；这是防止静默丢任务的预期行为。

### `9b11ec4b` `fix(akasha): 暴露空节点向量损坏`

- 范围：Akasha sidecar 节点反序列化与损坏 DB 测试。
- 历史依据：PR #65/#66 的 sidecar/dense 图与 live/replay parity；上游 MessageEmbeddingStore 已拥有非空向量写契约。
- 原问题：空 embedding BLOB 节点被 list/get 静默当作不存在，使节点、边、fan 和诊断计数分叉。
- 为什么这样修改：sidecar DB 可来自旧版本或手工修改；读取边界没有正确修复动作，应携带节点 key 失败。
- 不变量与拥有层：正常写入的非空向量由 embedding/upsert 构造链拥有；持久化 BLOB 到 AkashaNode 由 `_row_to_node` 拥有。
- 能力变化：合法节点、召回、replay、read-only、reinforce 和 snapshot 不变；损坏节点不再被过滤。
- 性能变化：删除 list comprehension 的 None 过滤，非性能提交。
- 测试新增：真实写入节点后把 BLOB 改为空，断言 list_nodes 以节点 key 报错。
- 测试删除及原因：无。
- 验证结果：Akasha+fast replay parity `51 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：已有空向量节点会阻止整图加载，需重建 sidecar；这是避免错图运行的预期 fail-stop。

### `badc79c1` `fix(proactive): 暴露记忆优化失败`

- 范围：MemoryOptimizer pending 两阶段事务、SELF 更新、取消传播与历史测试替身。
- 历史依据：PR #75 的 memory fail-stop；后台 `MemoryOptimizerLoop` 已拥有记录异常并等待下周期的 supervisor 边界。
- 原问题：merge/provider 与 SELF 异常被吞成空内容或假成功；旧测试只提供一次模型响应，第二步 `StopAsyncIteration` 也被掩盖；marker-only snapshot 会永久遗留。
- 为什么这样修改：snapshot 成功后，read/merge/backup/write/commit/rollback 整个 MEMORY 阶段任一步失败或取消都恢复 pending 并重抛；SELF 在事务外，不能回滚已提交 MEMORY但必须报告失败。
- 不变量与拥有层：pending 两阶段事务由 optimizer 拥有；周期隔离由 loop supervisor 拥有；正常空 merge 明确 rollback；marker-only 空有效内容明确 commit 清理 snapshot。
- 能力变化：正常合并、空结果保留原记忆、SELF 更新和周期续跑不变；异常/取消可见且 pending 不丢；SELF 部分失败如实暴露。
- 性能变化：正常模型调用次数和顺序不变，无性能声明。
- 测试新增：merge RuntimeError、真实 MEMORY 写失败、CancelledError、SELF 失败、marker-only snapshot；修正旧测试两步响应。
- 测试删除及原因：无。
- 验证结果：optimizer `14 passed`，相关主动组合 `422 passed`；pyright `0 errors` 且仅一个既有 warning；`git diff --check` 通过。
- 残余风险：SELF 写入不是与 MEMORY 同一原子事务，失败会保留已提交 MEMORY；该部分成功状态现在显式可见，后续若要全局原子性需独立设计。

### `96baa0ab` `fix(proactive): 收紧时间归一化异常边界`

- 范围：主动候选时间与时区归一化、直接边界测试。
- 历史依据：PR #101 的 runtime clock；外部候选非法时间按既有契约可忽略，运行环境故障不可伪装成无时间。
- 原问题：两个 `except Exception` 同时吞掉非法输入与 tzdata/runtime 程序错误。
- 为什么这样修改：ISO 只恢复 `ValueError`；时区只恢复 `ValueError`/`ZoneInfoNotFoundError`；其他故障向 tick supervisor 传播。
- 不变量与拥有层：外部字符串解析由 contracts 边界拥有；tzdata/runtime 可用性不由归一化函数恢复；tick supervisor 负责记录和续跑。
- 能力变化：合法本地时间、非法 ISO/未知时区忽略不变；非预期时区解析故障改为明确失败。
- 性能变化：分支和调用次数不变，无性能声明。
- 测试新增：非法时间/时区继续恢复，以及注入非预期 ZoneInfo RuntimeError 的传播。
- 测试删除及原因：无。
- 验证结果：定向 `10 passed`、主动相关 `424 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：GatewayResult 动态 payload 类型仍需跨模块协议设计，不能靠局部 cast 解决。

### `94534191` `fix(akasha): 对齐只读来源引用契约`

- 范围：Akasha source_ref JSON-list 统一解析和 read-only query 回归。
- 历史依据：PR #66 的 live/replay parity；PR #67 的 read-only 查询不得写 activation/query log。
- 原问题：stateful query log 已 fail-loud，但 read-only record 构造仍把损坏 JSON 或非数组归为空 evidence，形成模式间失败语义分叉。
- 为什么这样修改：`_source_refs()` 与 `_source_ref_ids()` 共用唯一 JSON-list parser；内部生成契约违反时直接失败。
- 不变量与拥有层：source_ref 由 `_load_turn_card` 生成 JSON list；record/query-log 消费层不拥有修复动作。
- 能力变化：合法 evidence、stateful/read-only 召回结果不变；read-only 损坏引用明确失败，同时仍不产生 pending activation 或 query log。
- 性能变化：删除重复解析分支，无性能声明。
- 测试新增：同一 read-only request 先验证合法结果，再注入非数组 source_ref，断言失败且两次均 `update_state=False`、无状态写入。
- 测试删除及原因：无。
- 验证结果：Akasha+fast replay parity `51 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：历史 sidecar/query log 若含坏 source_ref 会显式失败，需要迁移或重建；这是避免空证据假成功的预期行为。

## 集成检查点

- Wave 1 主分支组合验证：`1502 passed`。
- Wave 2 中段主分支组合验证：`1516 passed`。
- Wave 2 收束前主分支组合验证：`1554 passed`。
- 三次均运行 `pytest -q tests/`，未删除测试；用例增长来自真实契约、事务和性能回归。

### `48a8768f` `fix(memory): 拒绝非法显式插件配置`

- 范围：default-memory TOML section/字段类型收敛和配置回归。
- 历史依据：PR #41 的默认记忆插件标准 TOML 写法全部保留。
- 原问题：错误 section 被归为空配置；`bool("false")` 变 True；db_path/整数/浮点错误值被强转或截断。
- 为什么这样修改：只对文件、section 或字段缺失使用默认；显式值由唯一配置 owner 严格解析并携带完整字段路径失败。
- 不变量与拥有层：外部 TOML schema 由 `load_default_memory_config`/codec 拥有，engine 信任强类型；不在算法层重复检查。
- 能力变化：标准 TOML、历史整数/数字字符串和整数值 float 保留；错误根/嵌套 section、bool 冒充数字、非整数 float、容器等 fail-fast；未新增范围限制。
- 性能变化：仅初始化解析，无性能声明。
- 测试新增：合法旧写法和九类显式错误值/section。
- 测试删除及原因：无。
- 验证结果：配置与 memory engine contract `39 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：字段数值范围仍需结合召回算法和历史配置设计，未武断收紧。

### `e2d3a7ba` `fix(bus): report admission enqueue failures`

- 范围：EventBus 热重载 admission 后台入队 task 所有权与错误日志测试。
- 历史依据：PR #105 的 snapshot admission/lease/drain；PR #109 的事件流唤醒。
- 原问题：暂停 admission 时创建的后台 task 只从集合删除，不读取异常；acquire 失败导致事件丢失并产生无人拥有的 asyncio 异常。
- 为什么这样修改：EventBus 作为 task owner，done 时统一清集合；shutdown cancellation 静默，其他失败读取原异常并记录 traceback 和事件类型。
- 不变量与拥有层：admission/acquire 由 snapshot store 拥有；task 生命周期和失败可见性由 EventBus 拥有；不新增 retry/fallback。
- 能力变化：成功入队、lease、queue、drain/close 不变；失败仍不伪装成功，但具备领域日志。
- 性能变化：成功路径多一次 `task.exception()` 常数操作，无性能声明。
- 测试新增：模拟 acquire 失败，断言原 cause、日志和 pending owner 清理。
- 测试删除及原因：无。
- 验证结果：热重载相关 `90 passed`；副手完整测试 `1557 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：失败事件不自动重试；是否持久化事件属于 durable delivery 设计，不应局部猜测。

### `7a595739` `fix(skills): 拒绝损坏的元数据配置`

- 范围：SKILL.md metadata YAML/JSON 边界、requires 可用性与 loader 测试。
- 历史依据：PR #95 的 Skill Catalog generation 与 PR #105 的候选 snapshot/hot reload。
- 原问题：损坏或非对象 JSON metadata 被归为空配置，绕过 requires 后错误标记技能可用。
- 为什么这样修改：metadata 缺失/空才无配置；YAML map/JSON object 正常；损坏 JSON、数组、null 携带具体 SKILL.md 路径失败。
- 不变量与拥有层：metadata schema 和 requirements 由 SkillsLoader 拥有；snapshot 只接收已校验 SkillRecord。
- 能力变化：合法技能、优先级、缺失 metadata 和热重载不变；损坏候选在发布前失败。
- 性能变化：索引构建增加常数级结构判断，无性能声明。
- 测试新增：空 metadata 两种写法、损坏 JSON、数组/null 非对象和路径上下文。
- 测试删除及原因：无。
- 验证结果：相关公共契约/snapshot/热重载 `224 passed`；pyright `0 errors, 0 warnings`；`git diff --check` 通过。
- 残余风险：requires 领域规则未扩展；未来字段必须在 owner 层显式设计。

### `717e61ee` `fix(bootstrap): continue cleanup after server failure`

- 范围：AppRuntime dashboard/chat task 等待与 shutdown supervisor 测试。
- 历史依据：应用 shutdown 已定义逐项继续清理、最后抛首错；PR #105 的 watcher/services/core drain 需要完整执行。
- 原问题：server task 已失败时，统一 cleanup supervisor 之前的直接 await 立即重抛，跳过 watcher、proactive、IPC、channels、core、memory 和 HTTP 资源清理。
- 为什么这样修改：把两个 server wait 纳入 `_run_cleanup_steps`；server 异常仍是最终首错，但后续资源全部获得清理机会。
- 不变量与拥有层：server should_exit/等待由 server step 拥有；跨资源继续清理和首错由 shutdown supervisor 拥有。
- 能力变化：正常顺序和 CancelledError 语义不变；失败 shutdown 不再短路后续清理。
- 性能变化：正常 shutdown 等待顺序不变，无性能声明。
- 测试新增：dashboard task 预先失败，断言最终原错、core.stop、should_exit 和 HTTP close。
- 测试删除及原因：无。
- 验证结果：相关 `40 passed`；副手完整测试 `1568 passed`；pyright `0 errors` 且无新增 warning；`git diff --check` 通过。
- 残余风险：server task 无限等待和 shutdown 外部取消需要整体 timeout/shield 契约，本提交不局部改变。

### `363b725e` `fix(chat): 限制代码高亮缓存`

- 范围：聊天前端代码块高亮缓存与并发请求合并。
- 原问题：以不完整键缓存高亮结果且无容量上限；同一输入可重复启动异步高亮。
- 为什么这样修改：缓存键覆盖语言、主题和代码全文，使用 128 项 LRU，并复用同键 pending promise。
- 不变量与拥有层：代码块组件拥有展示缓存；Shiki 仍拥有语法高亮结果，组件不伪造失败结果。
- 能力变化：高亮内容与主题切换保持；消除键碰撞和重复计算。
- 性能变化：已完成缓存从无界变为最多 128 项；同键并发计算从 N 次变为 1 次。
- 测试新增：无；该组件暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 和 build 通过；lint 仅 3 条既有 Hook warning。
- 残余风险：pending 表在任务存续期保留 promise；任务完成后立即删除。

### `27dd8f0a` `fix(ipc): reject non-object client frames`

- 范围：IPC client newline JSON 帧反序列化边界。
- 原问题：合法 JSON 标量随后以对象方法访问，产生不透明异常并断开连接。
- 为什么这样修改：JSON 解码后立即确认顶层对象；非法帧显式记录并跳过，后续合法帧仍可处理。
- 不变量与拥有层：wire JSON 结构由 IPC 边界拥有；handler 信任对象，不重复检查。
- 能力变化：合法对象不变；单个非对象帧不再破坏长连接。
- 性能变化：每帧增加一次常数结构判断，无提速声明。
- 测试新增：同一连接发送标量后发送合法对象，验证错误可见且连接继续工作。
- 测试删除及原因：无。
- 验证结果：IPC 定向测试和 pyright 通过。
- 残余风险：对象内部字段仍按各消息 handler 的协议分别校验。

### `6d4d58ee` `fix(config): 拒绝无效工具集装配配置`

- 范围：agent 工具集装配配置读取。
- 原问题：错误类型和未知工具集被静默归一化，容易在启动后表现为能力缺失。
- 为什么这样修改：缺失字段使用默认；显式空列表保留“禁用全部”；错误结构和未知名字启动期失败。
- 不变量与拥有层：外部配置 schema 由装配层拥有；运行时只接收已解析工具集。
- 能力变化：默认与显式禁用语义不变；配置错误从隐性降级变为明确失败。
- 性能变化：仅启动期校验，无性能声明。
- 测试新增：覆盖缺失、显式空、错误类型和未知工具集。
- 测试删除及原因：无。
- 验证结果：相关配置测试和 pyright 通过。
- 残余风险：工具自身的运行时外部输入仍由各自边界拥有。

### `54f2026b` `refactor(chat): 保持代码高亮渲染纯净`

- 范围：聊天代码块异步高亮状态更新时机。
- 原问题：React render 阶段触发 setState，可能引发重复渲染与陈旧结果覆盖。
- 为什么这样修改：副作用移入 effect，并把异步结果绑定到当前输入。
- 不变量与拥有层：React effect 拥有异步生命周期；render 只从状态生成视图。
- 能力变化：高亮、复制和主题效果不变；旧请求不再覆盖新代码。
- 性能变化：消除 render 阶段额外状态更新，无量化延迟声明。
- 测试新增：无；该组件暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 和 build 通过。
- 残余风险：前端缺少组件级并发测试，当前由类型、构建与代码审阅覆盖。

### `5cdff4b9` `fix(lifecycle): 拒绝未闭合的阶段依赖`

- 范围：Phase 核心模块依赖和数据 slot 启动校验。
- 原问题：核心依赖缺失、顺序错误或 slot 未产生只记录 warning，真实 turn 才以 KeyError 等不透明方式失败。
- 为什么这样修改：核心阶段构造期 fail-fast；插件模块缺失插件依赖仍由拓扑层递归禁用，保留热插拔降级。
- 不变量与拥有层：Phase 拥有核心链闭合；插件拓扑拥有可卸载插件依赖。
- 能力变化：正常 turn、snapshot、interrupt 和 hot reload 不变；核心装配错误提前暴露。
- 性能变化：仅构造期校验，无性能声明。
- 测试新增：核心依赖不存在、顺序错误和未闭合 slot；保留插件递归禁用回归。
- 测试删除及原因：无。
- 验证结果：主线生命周期/热重载组合 `137 passed`；副手相关 `148 passed`；pyright 通过。
- 残余风险：动态插件是否允许依赖核心 slot 仍由现有命名协议区分。

### `92c7addd` `fix(akasha): 串行化图快照轮询`

- 范围：Akasha graph panel 快照轮询与 disposer。
- 原问题：请求慢于 5 秒轮询周期时会无限重叠，旧响应还可能晚到并覆盖新结果。
- 为什么这样修改：每个 panel 最多一个 in-flight 请求；完成后恢复轮询；dispose 后不再应用结果。
- 不变量与拥有层：panel 拥有轮询并发；后端 snapshot version 与增量坐标协议未改。
- 能力变化：首次 refit、坐标、交互和热重载 disposer 保持；消除旧响应覆盖。
- 性能变化：并发快照请求从无界降为最多 1；不声明单次延迟提升。
- 测试新增：无；插件面板暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 与真实 esbuild 参数编译通过；未修改 static bundle。
- 残余风险：请求失败仍沿既有显式失败路径，由下一轮定时器重试。

### `64c66fb0` `fix(clock): make replay advance atomic`

- 范围：ReplayClock 单实例并发推进与持久化。
- 原问题：`advance` 的读取和写入分属两个锁区间，并发调用会丢失 delta。
- 为什么这样修改：同一锁内完成 read-modify-write；底层同目录临时文件替换保持。
- 不变量与拥有层：ReplayClock 实例拥有进程内串行化；不声明跨实例或跨进程互斥。
- 能力变化：now/set/环境选择保持；同实例并发推进不再丢增量。
- 性能变化：锁覆盖一次文件读写，以正确性为目标；无延迟优化声明。
- 测试新增：8 线程各推进 50 次，400 个返回时间唯一且最终时间累计完整。
- 测试删除及原因：无；初版审阅时删除了未被生产路径调用的无效 barrier 测试钩子，改为真实并发压力回归。
- 验证结果：Clock/wake `25 passed`；副手全量 `1574 passed`；pyright `0 errors, 0 warnings`。
- 残余风险：多个 ReplayClock 实例指向同一路径仍需文件锁或单 owner 架构，本提交不扩大承诺。

### `93de1a8a` `fix(context): 显式标记不可用媒体`

- 范围：MessageEnvelopeBuilder 本地媒体装配。
- 原问题：不存在的本地附件在多模态和文本/VL 两条路径都被静默丢弃，模型无法区分“无附件”和“附件不可用”。
- 为什么这样修改：保留文字 turn，在上下文和 warning 中明确具体不可用路径；仅缺失附件时不诱导调用读图工具。
- 不变量与拥有层：媒体文件可访问性由上下文装配边界拥有；模型调用链信任已标注媒体引用。
- 能力变化：有效本地图片、文档、远程图片和 VL fallback 不变；缺失附件变为可观察降级。
- 性能变化：缺失路径增加一条 warning 和文本标记，无性能声明。
- 测试新增：两种媒体能力路径的缺失文件，以及仅缺失附件时不生成读图调用。
- 测试删除及原因：无。
- 验证结果：副手 ContextBuilder/lifecycle `117 passed`；主线相关 `47 passed`；pyright 通过。
- 残余风险：远程 URL 的可达性仍由实际 HTTP/视觉工具边界判断，装配期不预请求。

### `f97e0eb9` `fix(dashboard): 暴露插件发现失败`

- 范围：Dashboard 插件清单启动加载与既有错误边界。
- 原问题：`/api/dashboard/plugins` 失败被转为空列表，UI 看似正常但插件能力全部消失。
- 为什么这样修改：移除空列表 fallback，把失败交给 App 统一 `run()` 边界展示；单 panel import 隔离策略保留。
- 不变量与拥有层：清单请求整体成功由启动加载拥有；单插件模块失败由 importPanel 隔离并记录。
- 能力变化：合法插件、版本 URL、CSS 注入和 hot-reload freshness 不变；发现失败明确显示。
- 性能变化：请求与加载次数不变，无性能声明。
- 测试新增：无；该启动链暂无前端测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、lint 和 production build 通过；lint 仍为 3 条既有 Hook warning。
- 残余风险：单 panel import 失败仍允许其他插件继续加载，这是插件隔离边界的既有能力。

### `8307360f` `fix(persistence): isolate atomic save temp files`

- 范围：scheduler 与 AnyAction 共用的 JSON 原子写底座。
- 原问题：同一目标的并发 writer 共用固定 `.tmp`；一个 writer 可替换另一个的内容，随后另一个因临时文件已移动而失败。
- 为什么这样修改：每次写入使用同目录唯一临时文件，再原子 replace；失败仅清理本 writer 的临时文件并传播原异常。
- 不变量与拥有层：helper 拥有 staging 文件隔离和原子替换；不声明 writer 顺序、跨进程锁或 compare-and-swap。
- 能力变化：JSON 格式、目标路径和错误契约不变；并发写不再互相窃取/删除 staging 文件。
- 性能变化：写入与 replace 次数不变，UUID 生成是常数开销；无提速声明。
- 测试新增：两个真实线程同步到 replace 后均可提交且结果完整；replace 失败保持旧目标、清理本次临时文件并传播。
- 测试删除及原因：无。
- 验证结果：主线持久化 `16 passed`；副手全量 `1583 passed`；pyright 无 error。
- 残余风险：最后写入者覆盖先写入者仍是普通文件存储语义；需 CAS 的调用方必须另设版本协议。

### `d0171f73` `fix(persistence): log atomic cleanup failures`

- 范围：JSON 原子替换失败后的 staging 清理。
- 原问题：replace 首错后的 `unlink` 使用宽泛捕获并静默 pass，残留临时文件没有路径和原因。
- 为什么这样修改：仅捕获文件清理边界的 `OSError` 并记录 domain/tmp/error，随后继续抛原 replace 错误。
- 不变量与拥有层：原事务错误保持首错；helper 只补充 best-effort cleanup 的可观测性。
- 能力变化：成功路径和错误类型不变；清理失败不再静默。
- 性能变化：仅失败路径增加一条日志，无性能声明。
- 测试新增：replace 与 unlink 同时失败，断言首错和 cleanup 上下文。
- 测试删除及原因：无。
- 验证结果：副手全量 `1584 passed`；pyright 无 error。
- 残余风险：cleanup 失败会保留唯一临时文件，需按日志人工清理。

### `0c9e8da9` `perf(core): 限制工具发现会话缓存`

- 范围：ToolDiscoveryState 跨会话和会话内解锁工具缓存。
- 原问题：每会话已有 5 项 LRU，但 session 数无限增长；仅使用 always-on/meta 工具也会制造空项。
- 为什么这样修改：增加默认 1024 session 的 LRU；访问刷新顺序；空会话不入表，淘汰后可重新 tool_search。
- 不变量与拥有层：发现缓存只保存可重建的工具名，不是业务状态；registry 仍拥有真实工具可用性。
- 能力变化：当前 1024 个活跃 session 的工具顺序与复用不变；旧 session 被淘汰后重新发现。
- 性能变化：默认最坏驻留从无限增长收敛为约 5120 个工具名。
- 测试新增：空项不创建、跨会话 LRU 访问刷新和最旧淘汰。
- 测试删除及原因：无；审阅阶段删除了两个无意义 default-factory 包装函数后才合入。
- 验证结果：副手相关 `118 passed`；主线组合 `23 passed`；pyright 无 error。
- 残余风险：容量是实例参数；显式调大时上界随配置线性增长。

### `9d54a421` `refactor(chat): 清理代码块注释`

- 范围：聊天代码块组件注释。
- 原问题：Types/Context/Token rendering 等标题式英文注释重复代码结构，必要约束也未按项目中文约定表达。
- 为什么这样修改：删除 10 条废注释；保留并中文化 Shiki 位标志、稳定键、CSS 行号、缓存和异步展示约束。
- 不变量与拥有层：仅注释变更，运行代码和 lint 指令不变。
- 能力变化：无。
- 性能变化：无。
- 测试新增：无。
- 测试删除及原因：无。
- 验证结果：目标 lint、typecheck、全量 lint 和 chat build 通过。
- 残余风险：其他前端文件的英文注释按文件继续审阅，不机械全局替换。

### `ad7f7959` `fix(dashboard): 收紧 Hook 生命周期`

- 范围：Dashboard MagicIndicator 与插件跨页事件 Hook。
- 原问题：动态依赖数组无法静态验证；goto-session 订阅闭包可能调用旧 selectView。
- 为什么这样修改：MagicIndicator 只声明真实静态依赖，DOM 选中变化继续由 MutationObserver 驱动；事件订阅用 Effect Event 读取最新跳转逻辑。
- 不变量与拥有层：观察器拥有 DOM/class 变化；全局事件只订阅一次，不因 view state 重装。
- 能力变化：指示器、插件跳转与 DOM 生命周期保持；消除 stale closure。
- 性能变化：切换状态不再为依赖变化拆装观察器，无量化声明。
- 测试新增：无；该 UI 链暂无组件测试 runner。
- 测试删除及原因：无。
- 验证结果：typecheck、production build 和 lint 全过，历史 3 条 Hook warning 归零。
- 残余风险：MutationObserver 高频 mutation 的 RAF 合并仍可进一步独立评估。

### `be2e828b` `fix(subagent): 暴露模型调用硬错误`

- 范围：SubAgent provider 调用与同步/后台失败转换链。
- 原问题：provider 硬错误被 SubAgent 捕获后返回空字符串，可能被上层解释为正常空结果。
- 为什么这样修改：owner 先标记 `last_exit_reason=error`，再传播原异常；同步 spawn/后台 runner 继续在各自边界转成明确失败。
- 不变量与拥有层：SubAgent 拥有退出原因；任务 runner 拥有面向调用方的 error status/摘要。
- 能力变化：正常完成、loop guard、预算收尾不变；provider 故障不再假成功。
- 性能变化：无。
- 测试新增：provider RuntimeError 原样传播且退出原因为 error。
- 测试删除及原因：无。
- 验证结果：副手 SubAgent/spawn/background `40 passed`；主线相关 `34 passed`；pyright 无 error。
- 残余风险：工具执行错误仍按 ToolResult 协议处理，不与 provider 基础设施故障混淆。

### `0109b65f` `fix(proactive): reject corrupt quota state`

- 范围：AnyAction 每日配额 JSON 反序列化边界。
- 原问题：坏 JSON、权限错误和缺失文件都初始化零用量，可绕过每日动作上限；字段又被不一致地强转。
- 为什么这样修改：仅文件不存在初始化；严格读取 version=1 完整 schema、window key、非负整数和 aware ISO 时间；TypedDict 固化下游类型。
- 不变量与拥有层：QuotaStore 拥有持久化 schema；drift best-effort skill state 继续保留独立降级语义。
- 能力变化：合法首版格式、空 last_action 和 rollover 保持；损坏 quota 阻止启动且保留原文件。
- 性能变化：仅启动期校验，无性能声明。
- 测试新增：缺失、合法、非对象、缺字段、version/used/window/time、JSON 与读取权限错误。
- 测试删除及原因：无。
- 验证结果：定向 `12 passed`；副手全量 `1603 passed`；pyright 无 error。
- 残余风险：未知额外字段被忽略以保留向前兼容；schema 升级需显式版本迁移。

### `fc7fae40` `fix(subagent): 拒绝空白任务结果`

- 范围：SubAgent 无工具调用的最终响应契约。
- 原问题：模型以空白 content 结束时被标记 completed，后台/同步调用方收到假成功空结果。
- 为什么这样修改：最终响应 trim 后必须非空；否则标记 error 并由既有任务边界转换失败。
- 不变量与拥有层：中间 tool-call 响应允许空 content；最终任务结果由 SubAgent owner 保证可展示。
- 能力变化：正常文本、工具循环和预算收尾不变；空白最终响应明确失败。
- 性能变化：一次常数级字符串判断，无性能声明。
- 测试新增：空白 final response 抛错且退出原因为 error。
- 测试删除及原因：无。
- 验证结果：副手相关 `41 passed`；主线相关 `35 passed`；pyright 无 error。
- 残余风险：强制收尾 helper 的空结果契约继续单独沿完整调用链审阅。

### `23b08f74` `refactor(memory): 清理面板注释`

- 范围：默认记忆 Dashboard 面板注释。
- 原问题：12 条英文标题/逐段翻译注释无信息增量，必要的全局命名和增量 DOM 约束未按中文约定表达。
- 为什么这样修改：删除废注释；保留并中文化命名冲突、计数缓存、增量 DOM、焦点保持和降级边界。
- 不变量与拥有层：仅注释修改；运行代码和 TypeScript reference 不变。
- 能力变化：无。
- 性能变化：无。
- 测试新增：无。
- 测试删除及原因：无。
- 验证结果：typecheck、lint、插件 esbuild 与 dashboard build 通过。
- 残余风险：审阅同时发现文件内 catch 降级缺少可观测性，已作为下一笔功能修复处理，不能靠注释合理化静默失败。

### 外部插件 `8aaeab3` `fix(feed): maintain cache freshness in MCP lifecycle`

- 范围：canonical Feed 插件 `/mnt/data/coding/akashic-plugin/feed-mcp`、GitHub `akashic-plugins/feed-mcp` 与安装版本 `feed@github 1.2.0`。
- 历史依据：`3b456e7b` 把 source poll 绑定到 `default_proactive` lifecycle；启用 wake package 时 manifest 会禁用 default package，但 wake 只调用 `get_proactive_events`，从而丢失 Feed 外部刷新能力。
- 原问题：Feed MCP 进程持续运行且 wake 每约 5 分钟读取一次缓存，但 `poll_state.last_polled_at` 停在 2026-07-12 14:59 UTC；Tibo RSS 已有新消息，SQLite 仍是旧列表。现有测试只证明 default lifecycle 拥有 poll，没有覆盖 wake + Feed freshness 组合。
- 为什么这样修改：缓存 freshness 归缓存拥有者。Feed MCP 使用 FastMCP lifespan 启动唯一后台 poller；首次主动读取等待首次刷新，之后按 `feed_mcp.json.poll_ttl_seconds` 刷新。插件不再声明宿主 `poll_tool`，default 与 wake 都只通过异步 MCP 调用读取稳定缓存。
- 不变量与拥有层：Feed poller 唯一拥有刷新串行、首次 ready、失败状态和重试；backend 拥有单源 TTL 与 SQLite 数据；proactive lifecycle 只消费 source snapshot。系统级刷新错误使读取显式失败，单源失败继续由 Feed `_poll_rows` 隔离并记录。
- 能力变化：default/wake 的 fetch、分页、ack 和排序不变；wake 模式恢复新消息获取。MCP 启动不等待 32 个网络源，首次 `get_proactive_events` 才等待首次刷新；手动 poll 与后台 poll 由同一异步锁串行。
- 性能变化：外部抓取由宿主生命周期耦合改为 MCP 每 300 秒自行刷新；SQLite 启用 WAL 和 30 秒 busy timeout，轮询写入期间读取稳定快照。没有增加每次 wake 的网络抓取。
- 测试新增：poller 首次刷新屏障、持续刷新、失败可见与下一轮恢复。
- 测试删除及原因：删除未接线且吞掉启动错误的 `startup_force_poll()` 死代码；未删除行为测试。
- 验证结果：Feed 插件 `11 passed`；pyright `0 errors, 0 warnings`；GitHub 已推送；`plugin-install` 安装 1.2.0；运行进程切换到 1.2.0；首次自刷新 32/32 成功，Tibo 源解析 19 条并新增 2 条，`last_polled_at` 推进到 2026-07-12 18:59 UTC。
- 残余风险：同轮审计发现 Steam 历史 snapshot 的部分同类问题，已由下一条记录修复；Calendar 每次读取实时查询 Google API，Fitbit managed service 已自行轮询，二者不存在本次旧缓存问题。

### 外部插件 `326c055` `fix(steam): refresh proactive snapshots on demand`

- 范围：canonical Steam 插件 `/mnt/data/coding/akashic-plugin/steam-mcp`、GitHub `akashic-plugins/steam-mcp` 与安装版本 `steam@github 1.1.0`。
- 历史依据：Steam proactive source 的在线状态每次实时查询，但历史游戏时长只由手动 `take_steam_snapshot` 更新；运行数据库最后快照停在 2026-06-06。
- 原问题：`get_steam_context` 每约 5 分钟读取相同旧 snapshot；仓库没有定时调用者。即使新增定时调用，空的最近游玩列表也不会写任何行，下一轮仍会判断为从未成功刷新。
- 为什么这样修改：Steam context owner 在读取前检查 snapshot run 的 TTL，超过 6 小时才调用一次 Recently Played API；独立 `snapshot_runs` 表记录包括空结果在内的成功刷新批次。
- 不变量与拥有层：Steam MCP 拥有实时状态和历史快照 freshness；wake 只读取结构化 context。配置 JSON 损坏在读取边界 fail-loud；远端快照刷新失败保留实时状态并通过 `snapshot_refresh_error` 显式降级。
- 能力变化：实时 online/in-game 查询、两周与历史时长对比、wake presence/transition 保持；旧快照自动恢复刷新，空列表不再造成重复请求。
- 性能变化：wake 仍每轮查询轻量在线状态；Recently Played API 由过去“永不自动调用”变为最多每 6 小时一次，同 TTL 内只读 SQLite。
- 测试新增：过期快照只刷新一次、空快照记录成功批次、刷新失败可见、TTL 内跳过刷新。
- 测试删除及原因：无。
- 验证结果：Steam 插件 `7 passed`；pyright `0 errors, 0 warnings`；GitHub main 已推送；安装 1.1.0；真实 context 刷新成功，freshness `0.0h`、2 个近期游戏、无刷新错误；旧 1.0.0 generation 排空后仅保留 1.1.0 MCP 进程。
- 残余风险：Recently Played 和 Player Summary 是两个独立 Steam API 请求；其中一条失败时 context 会明确区分 snapshot 与 realtime 的降级状态，不提供跨 API 原子快照。

### `05ab66b3` `fix(runtime): restore session context after turn and tick`

- 范围：被动 turn 与 proactive tick 的 `current_session_key` 生命周期。
- 原问题：两个长链路入口调用 `ContextVar.set()` 后没有 reset；同一 task 后续执行会继承上一轮 session，导致 observe 全局错误归属错误。常规消息循环为每条消息创建 task，会掩盖被动路径问题，但 `process_direct` 和 proactive 长生命周期 loop 可真实触发。
- 为什么这样修改：由设置上下文的入口保存 token，并在最外层 `finally` 恢复调用方上下文；busy 状态仍由内层 `finally` 独立释放。
- 不变量与拥有层：`AgentLoop._process` 拥有单个 turn 的 session 绑定，`ProactiveLoop._tick_bound` 拥有单个 tick 的绑定；共享 ContextVar 和 observe 只读取，不承担生命周期清理。
- 能力变化：续跑、TurnStarted、核心处理、主动 Gate → Fetch → Judge → Resolve → Deliver、异常传播和 busy 状态语义保持；成功、失败与取消离开入口后均不残留 session。
- 性能变化：删除一处未使用计时，增加两次常数级 ContextVar token 操作；无性能收益声明。
- 测试新增：被动成功恢复、核心失败恢复并释放 processing state、主动成功与失败恢复。
- 测试删除及原因：无。
- 验证结果：副手定向 `19 passed`、全量 `1619 passed`、pyright `0 errors`；主线合入后定向 `19 passed`、全量 `1619 passed`，`git diff --check` 通过。
- 残余风险：其他独立 ContextVar 设置点仍需按各自任务生命周期审阅，不能从本次两处修复推断全仓已覆盖。

### `27e1c638` `fix(mcp): enforce 2024 tool result boundaries`

- 范围：MCP `tools/list`、`tools/call` 的外部 schema 与远端失败分类。
- 原问题：缺失 `inputSchema` 会静默变成空 schema，非字符串 description 被强转；坏 content block 可能以 Python repr 当成功结果进入模型；JSON-RPC error 与 `isError=true` 都被返回为普通字符串，直接调用方无法区分成功和失败。
- 为什么这样修改：按客户端实际协商的 `2024-11-05` 严格接受 text、image、resource 三类结果；后续版本字段明确拒绝。工具声明在 `tools/list` 边界校验，远端执行失败统一抛出带 server/tool/服务端内容的 `McpToolExecutionError`。
- 不变量与拥有层：`McpClient` 拥有 MCP 反序列化和协议版本边界；`ToolRegistry` 继续拥有面向模型的错误日志与 `工具执行出错` 回填。边界之后的 `McpToolInfo` 和工具结果不再重复防御。
- 能力变化：合法三类结果、stdio 串行、连接/执行 timeout、插件 generation 与热重载不变；坏 schema 不再进入工具目录，远端失败对直接调用方 fail-loud、对模型仍明确可见。
- 性能变化：每项增加常数级字段检查，content 仍单次 O(n) 遍历；无新增 I/O、重试或缓存。
- 测试新增：工具声明缺失/坏类型、三类有效内容、各类关键缺字段、未知与后续类型、非法 `isError`、JSON-RPC error 和 tool result error 异常。
- 测试删除及原因：无；旧 MCP 夹具补齐协议要求的 `type=text`。
- 验证结果：副手定向 `212 passed`、全量 `1632 passed`、pyright `0 errors`；主线合入后 MCP/IO 定向 `52 passed`，`git diff --check` 通过。
- 残余风险：客户端仍固定协商 `2024-11-05`；未来协议升级必须单独实现版本协商与新增 content union，不能在旧版本路径静默兼容。

### `61fba5be` `fix(channels): close resources and validate message boundaries`

- 范围：WebChat 外部消息、IPC server/client 生命周期、附件降级、渠道身份索引与 Telegram live-task 索引。
- 原问题：WebChat 强转坏 text 并静默丢弃坏 media 元素；IPC 构造时永久订阅且 stop 不关闭客户端，Unix chmod 失败会遗留已绑定 server/socket；身份保存失败会留下未持久化内存路由；Telegram 每个完成任务扫描全部 session 并永久保留空集合；附件 fallback 吞掉所有异常且无降级日志。
- 为什么这样修改：外部字段在 WebSocket 边界一次性严格校验；IPC 成功启动后才提交 server/subscription，停止前同步转移并 close 全部 ownership，再等待所有资源并重新抛首个 `OSError`；identity mapping 只在持久化成功后提交；任务回调按所属 session O(1) 清理；附件仅对 `OSError` 保留有日志的 `/tmp` 降级。
- 不变量与拥有层：WebChat 拥有帧 schema；IPC channel 拥有 server、writers 与 outbound subscription；SessionIdentityIndex 拥有 metadata/mapping 一致性；Telegram channel 拥有 live-task 索引；AttachmentStore 拥有文件系统降级。
- 能力变化：合法 WebChat、IPC、Telegram、附件上传和身份路由保持；坏帧返回明确 error 且连接可继续；IPC 启停失败不留订阅或客户端 ownership，多个关闭错误完成清理后 fail-loud。
- 性能变化：Telegram 完成回调由 O(session 数) 降为 O(1)，并删除空 session 集合；其余仅边界/生命周期常数级操作，无量化收益声明。
- 测试新增：WebChat 三类坏字段与连接续用、Unix chmod 失败事务回滚、server/writer wait 失败仍清理其余资源、IPC 正常 stop、identity 保存回滚、Telegram 空索引回收、附件 fallback 日志。
- 测试删除及原因：无。
- 验证结果：副手定向 `45 passed`、全量 `1628 passed`、pyright `0 errors`；主线合入后 channels/MCP 交叉定向 `69 passed`，`git diff --check` 通过。
- 残余风险：未改变 MessageBus 既有重试、FIFO、背压与取消策略；QQ/Telegram 外部 API 的独立错误策略需按具体调用链继续审阅。

### `f30973e9` `fix(proactive): tighten source and delivery boundaries`

- 范围：默认 proactive Gateway、MCP source event 边界与 success/post-guard ACK。
- 原问题：Gateway 再次捕获共享 source snapshot 故障并伪装成三路空数据；坏 web_fetch payload 被当成普通空正文；非对象或无 ID 的 alert/content 被跳过或生成碰撞 key；仅配置独立 alert ACK 时，普通 ACK 缺失导致 helper 提前返回。
- 为什么这样修改：单 source 隔离继续由 `fetch_sources_async` 唯一拥有，Gateway 只消费聚合 snapshot；整体失败和工具协议损坏 fail-loud。WebFetchTool 明确返回的 `{error}` 仍按可选正文降级，但记录 URL/原因 warning。source payload 在 MCP 边界拒绝无法可靠 ACK 的事件；两个 ACK 通道按实际依赖分别执行。
- 不变量与拥有层：source 聚合层拥有单源失败隔离；Gateway 拥有 snapshot 与 web_fetch 结果形状；`mcp_sources` 拥有 event object/ID；resolve helper 拥有 alert/content ACK 路由。
- 能力变化：正常并行抓取、单源隔离、显式 HTTP 失败空正文、ACK 顺序、发送、wake、热重载和 Gate → Fetch → Judge → Resolve → Deliver 不变；全部 source 失败不再假装无事件，独立 alert ACK 不再丢失。
- 性能变化：三路和 URL 抓取并行度、调用次数不变；新增每 item 常数级字段检查，无性能收益声明。
- 测试新增：三路 snapshot 失败传播、web_fetch 显式降级日志与损坏协议、坏 source item/空 ID、仅 alert ACK 的 success/post-guard 路径。
- 测试删除及原因：无。
- 验证结果：副手定向 `182 passed`、全量 `1646 passed`、pyright `0 errors`；主线合入后主动链交叉定向 `113 passed`，`git diff --check` 通过。
- 残余风险：Gateway/source payload 仍使用历史弱类型 dict；本批没有扩大为跨模块 typed contract 重构。
