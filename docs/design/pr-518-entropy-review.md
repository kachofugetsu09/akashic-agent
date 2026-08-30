# PR 518 熵回收评审台账

本文记录 `pro/clean-code` 相对 `origin/main` 的持续评审。结论只来自当前代码、测试、Git 历史、项目合同和已定位的外部插件源码；尚未证明的删除不记为安全。

当前评审基线：`19a510404736cf1ad376eee1f47bbb5c8004840f`。

## 评审原则

- Core 只拥有 Message、Turn、Session、Loop 及真实运行边界。
- 插件只取得 `agent.plugin_composition` 发布的普通能力，不导入 Core 私有 runtime。
- 已有上位能力时，迁移消费者后删除下位入口；不能只删声明、留下调用者。
- 外部插件本身违反 v3 边界时，不把它的私有依赖永久固化为 Core 公共 API。
- PR 518 的既有任务合同标记 `semantic_delta: none`；公共插件 API、用户能力或候选发布语义的变化必须单独批准和迁移。

## 目标结构

```text
普通 v3 Channel 插件
        │
        ▼
ChannelControlPort                 agent.plugin_composition 公共能力
        │
        ▼
ChannelGenerationHost             exact binding、去重、回送 receipt
        │
        ▼
ConversationRuntime               唯一 Turn 中断 owner
```

不应长期存在的旁路：

```text
v2/legacy Channel ──► agent.looping.InterruptController ──► Core 私有对象
```

## 已确认的破坏或半迁移

### 1. lifecycle context prepared 事件被删错

- 证据：`agent/lifecycle/before_turn.py` 仍导入 `BeforeTurnCtx` 和 `CONTEXT_PREPARED_EVENT`，原 PR 只从 `agent/lifecycle/composition.py` 删除声明侧导入。
- 失败：导入 `passive_turn` 即失败，所有被动 Turn 无法启动。
- 处理：`8c17ea2d` 已恢复声明侧接线；12 个 before-turn 定向测试通过。
- 结论：这是未完成删除，不是安全熵回收。

### 2. InterruptController 被删时仍有真实生产调用者

- 证据：`infra/channels/contract.py`、Telegram、QQ、Web/Mobile channel、`bootstrap/channels.py` 都仍消费该协议；QQBot 和 Feishu 的外部 v2 源码也直接导入它。
- 失败：pytest 收集阶段出现 19 组 channel/bootstrap import error，运行时无法启动 Channel。
- 处理：`35baa062` 已恢复协议以止血；channel host/client 17 项通过。
- 上位替代：公共 `ChannelControlPort` 已存在，`bootstrap/app.py` 已把它绑定到 `ConversationRuntime.request_interrupt()`。
- 结论：恢复只是让 PR 可运行，不代表私有协议应永久保留。是否删除取决于下面的 v2 插件产品决定和 legacy Channel 迁移。

### 3. 五个插件 service 的 `.formal` 被删时仍有插件消费者

- 证据：内建 scheduler/subagent 的 `is_active(ServiceView)` 仍读取 timers、scoped turns、deliveries、continuations 的 `.formal`；外部 proactive-feedback v3 源码仍读取 `SessionReadService.formal`，用来阻止 candidate Root 启动 worker 或写插件数据库。
- 失败：scheduler/subagent 在静态准入阶段抛 `AttributeError`；proactive-feedback candidate/formal 行为失去判别入口。
- 处理：`19a51040` 已恢复五个可观察属性；SessionRead、subagent、scheduler 非 soft 路径 17 项通过。
- 上位替代：长期应由一个 runtime mode owner 表达 candidate/formal；`Context.data_access` 已覆盖 apply 阶段，`ServiceView` 应只表达能力是否可用。但外部插件和静态准入必须先同步迁移。
- 结论：本 PR 不能静默删除。跨仓库迁移完成前，保留派生只读属性比伪装删除更安全。

### 4. `ToolGrant.except_names()` 被删时 scheduler 仍调用

- 证据：`plugins/scheduler/plugin.py` 的 soft schedule 唯一在库调用点仍使用该构造器。
- 失败：soft schedule 在 fire 后抛 `AttributeError`，任务被错误禁用；3 个 scheduler 行为测试失败。
- 上位替代：没有更高层能力替代 deny-list。`ToolGrant(names=None, denied=...)` 是底层表示，但直接构造更难发现、也更容易误用。
- 当前方向：恢复这个小而明确的公共构造器，并保留 soft schedule 行为测试。
- 结论：这是只删了一半，不是有效简化。

## 已确认安全的删除

### 1. Core 私有空 package facade

- 范围：`agent.control`、`agent.core`、`agent.host_bridge`、`agent.lifecycle`、`agent.lifecycle.phases`、`agent.retrieval`、`agent.turns` 的空 `__init__.py` re-export。
- 证据：生产代码和 canonical v3 插件使用具体模块或 `agent.plugin_composition`；没有支持中的 v3 插件从这些私有 facade 导入。
- 失去能力：无可观察能力；只删除第二条导入路径。
- 注意：`agent.tools.meta` 另有陈旧测试消费，尚未归入此结论。

### 2. `PreToolCtx` legacy 残留

- 证据：外部 `shell_safety`、`shell_restore`、`tool_loop_guard` 源码仍出现该名称，但没有 v3 manifest；当前 PluginManager 明确拒绝非 v3 插件。
- 上位替代：v3 Tool 能力和 lifecycle context 从 `agent.plugin_composition` 注入，不要求插件导入 Core 私有 context。
- 失去能力：只影响已不受支持的 v2 残留，当前 v3 runtime 无可达消费者。
- 后续：应在外部插件源码仓库单独删除或迁移这些残留，避免继续暗示它们受支持。

### 3. turn rollout 的 bound-method identity cache

- 证据：同一实例上的同一 bound method 以 equality 可稳定比较；install 后 `reload_tx_id` 已保证非空。
- 失去能力：无；删除的是重复缓存和已由前置条件保证的分支。

## 仍在核验，不需要现在决定

- `estimate_history_budget`：生产零消费者，只有一个测试导入；需确认测试是否只冻结已删除实现，而不是遗漏新的预算 owner。
- `agent.tools.meta` facade：生产从 `catalog`/`register` 具体模块导入，只有测试仍走 facade；需检查它是否属于文档化公共 API，再决定改测试还是恢复。
- `AgentLoop.request_interrupt()` 与 `ConversationRuntime.request_interrupt()`：当前 bootstrap 把 Channel 绑定到后者，但旧 Loop 还维护独立 interrupt state；需完成调用链和恢复语义核验，确认是否为重复 owner。
- lifecycle string slot 对无效插件导出的处理：PR 从“记录并忽略”改为 fail-loud；这是插件边界语义变化，不应靠修改旧测试强行通过。

## 需要花月哥哥确认的产品决定

### A. v2 QQBot/Feishu 是否正式停止支持

推荐：**确认停止支持，并把它们迁移为普通 v3 Channel 后再恢复发布**。

- 如果确认：Core 不为它们保留 `agent.looping` 私有 ABI；迁移后 legacy `InterruptController` 可继续收敛。
- 如果不确认：PR 518 必须继续保留 legacy Channel 边界，且不能宣称插件已完全普通化。
- 当前安全默认：不删除用户能力，先保留止血协议。

### B. 是否在后续独立迁移中删除 service `.formal`

推荐：**是，但不放进 PR 518**。

- 目标：apply 阶段统一读取 `Context` 的 runtime mode；静态 `ServiceView` 只按能力存在与否决定 active。
- 前提：同步修改并发布 proactive-feedback 等外部 v3 插件，定义 Core/plugin 最低兼容版本和升级顺序。
- 当前安全默认：保留 `.formal`，不破坏已发布插件。

### C. 无效 lifecycle slot 导出是否改为 fail-loud

推荐：**fail-loud**，因为插件模块加载是外部扩展信任边界，无效类型不是可恢复状态。

- 如果确认：保留 PR 的严格校验，修改测试以验证精确错误，并检查外部 v3 插件没有无效导出。
- 如果不确认：恢复“记录并忽略”兼容语义。
- 当前状态：未提交结论，等待外部插件扫描和你的决定。

## 已提交修复与验证

| commit | 修复 | 验证 |
| --- | --- | --- |
| `8c17ea2d` | 恢复 context-prepared lifecycle 接线 | before-turn 12 passed |
| `35baa062` | 恢复仍被生产 Channel 使用的 interrupt protocol | channel host/client 17 passed |
| `19a51040` | 恢复正式/候选插件 service 合同 | 17 passed；另发现 3 个独立 scheduler soft 失败 |

完整 pytest 当前仍在收集阶段被两个陈旧导入阻挡：`estimate_history_budget` 和 `agent.tools.meta` facade。两项完成归属判断后再恢复全量验证。
