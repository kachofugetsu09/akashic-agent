# Core 领域 Observe 事件任务合同

- 状态：superseded by pure-V3 direct publication
- 日期：2026-08-17
- 目标分支：`codex/plugin-v3-mobile-ui-query`
- 恢复点：`backup/observe-events-pre-20260817`
- 上游：[插件 Transform 与 Observe 事件任务合同](plugin-transform-observe-task-contract.md)
- 关联：[Turn committed typed event 合同](plugin-turn-committed-event-task-contract.md)

## 1. 目标

本文记录最初的过渡设计。当前合同见[插件 V3 能力手册](plugin-v3-capabilities.md)：领域
owner 直接发布 request/generation-bound `ObserveEventKey`，`EventBus.fanout()` 不再按 payload
类型猜测并桥接到插件组合树。Core 不复制领域 DTO，也不替插件拥有数据状态。

```text
settled domain fact
        │
        ▼
┌────────────────────────┐
│ domain owner settles fact │
└────────────┬───────────┘
             ▼
┌───────────────────────────────┐
│ request-bound CompositionRoot │  ObserveEventKey
└────────────┬──────────────────┘
             ▼
        plugin observer
```

| 事实 | Key | Core owner | 当前生产入口 |
|---|---|---|---|
| `RetrievalCompleted` | `memory.retrieval.completed` | `agent/turn_events/observe.py` | Akasha 检索 owner 直接 observe |
| `MemoryWritten` | `memory.written` | `agent/turn_events/observe.py` | 当前无生产者；只保留结构合同 |

payload 使用 `core.memory.events.RetrievalCompleted` 和 `core.memory.events.MemoryWritten`；
发布时传递原对象，不重新拼装事件。

## 2. 调度与 generation 边界

- 领域 owner 调用 `observe_composition_event(KEY, payload)`，从当前 request lease 读取 exact
  composition Root；candidate、stable 和旧 generation 不从全局 latest 重新选择。
- `TurnCommitted` 由 after-turn owner 在 Core EventBus fanout 后显式发布
  `AFTER_TURN_COMMITTED`，不是通用类型桥的一部分。
- `EventBus` 只服务仍由 Core 拥有的内部事件消费者，不是插件 API。

## 3. Retrieval 事实

Akasha 插件在 MemoryEngine 成功返回后从同一
`MemoryQueryResult` 形成 `RetrievalCompleted`：

- `rewritten_query`、`aux_queries`、`hyde_hypotheses` 和 `route_decision` 只从
  engine result 的 trace/raw 读取；缺失时使用原始请求或空集合；
- 每个 `MemoryRecord` 转成现有 `RetrievalHitSummary`，保留 id、kind、score、
  summary、injected、confidence/forced signals 和 metadata；
- engine 失败或取消时不伪造 completed event，原异常或 cancellation 继续传播。

## 4. 失败、清理与持久化边界

- 没有 composition Root 时 direct publication no-op；错误 task、释放 lease 或错误
  binding 由 lifecycle snapshot owner fail-loud。
- 普通 observer failure 仍由 `EventRegistry.observe` 记录所属 Fiber Incident
  并隔离；调用方取消、进程级异常和 lease 错误不能被吞掉。
- direct publication 只在内存中 dispatch。candidate 不写正式 workspace、plugin-data、
  SessionDB、memory DB 或外部渠道；插件自身的派生写入仍必须使用 Core 分配的
  candidate data root，并由插件合同验证。
- 本变更没有删除、更新或迁移权威持久记录。Core EventBus handler、队列 lease、Root
  Effect/listener 的清理语义保持不变。

## 5. 验收与 mutant

定向测试位于 `tests/test_plugin_composition_lifecycle.py`，覆盖：

- direct V3 publication 保留原对象 identity；
- EventBus fanout 不会隐式进入插件 composition；
- 两个绑定 Root 间的 candidate/generation 选择与 wrong-task fail-loud；
- Retrieval payload 字段和原异常传播；
- leaf contract 在 fresh interpreter 中不加载 phase runtime。

验证命令：

```bash
./.venv/bin/python -m pytest -q tests/test_plugin_composition_lifecycle.py
./.venv/bin/python -m compileall -q agent/turn_events/observe.py \
  agent/lifecycle/composition.py bus/event_bus.py \
  plugins/akasha/plugin.py agent/looping/core.py bootstrap/tools.py
```

历史过渡实现可从 `backup/observe-events-pre-20260817` 查阅，但不得恢复通用桥。
