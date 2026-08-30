# PR 518 熵回收评审台账

本文记录 `pro/clean-code` 相对 `origin/main` 的持续评审。结论只来自当前代码、测试、Git 历史、项目合同和已定位的外部插件源码；尚未证明的删除不记为安全。

当前评审基线：`646ff15c2e401db92d27438cd22e8e28ec67aaaa`。

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
- 处理：`7a6e5a1e` 已恢复这个小而明确的公共构造器；scheduler 17 项和 scoped 10 项通过。
- 结论：这是只删了一半，不是有效简化。

### 5. readonly tool 构造器删除参数后，上游仍继续传参

- 证据：`build_readonly_tools()` 已不再接收 `tool_context_provider` 和 `agent_loop_provider`，但 `bootstrap/tools.py` 的 fresh runtime 路径仍构造并传入二者，`loop_ref` 也只为这条失效路径存在。
- 失败：fresh config 与 `serve` 启动 smoke 在组装 runtime 时抛 `TypeError`。
- 处理：`5ceef3e5` 从调用点、import 和死引用端到端删除，不恢复 no-op 参数；启动/toolset 28 项通过。
- 结论：这是调用侧未完成迁移；删除上游胶水才真正减少概念。

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

### 4. history budget helper 与 `agent.tools.meta` facade

- 证据：`estimate_history_budget` 只有测试消费者，生产预算由当前 context owner 直接计算；`agent.tools.meta` 只有测试从 facade 导入，生产已从 `catalog` 和 `register` 具体模块导入。
- 处理：`eed34b21` 删除只冻结旧 helper 的测试，并让 facade 测试跟随真实 owner。
- 失去能力：无生产、动态插件或受支持公共 API 能力。

### 5. prompt section wrapper 与 lifecycle façade 多余入口

- 证据：Prompt builder 的唯一组装结果已经直接是 `list[PromptSectionRender]`；`TurnLifecycle` 的生产 wiring 只使用 `on_after_step`，普通 v3 插件通过公开 typed event 注册其他阶段。
- 处理：`82f5a2db` 让 prompt 测试读取直接结果；`e52c8108` 删除六组只测试已删转发方法的形式测试，并把 subscription ownership 覆盖合并到存活入口。
- 失去能力：无；插件事件能力仍由 `agent.plugin_composition` 拥有。

### 6. 测试替身绕开真实端口或私有表示

- 证据：`last_debug_breakdown` 和 `peek_next_message_id` 是生产必需端口；上一条消息 ID 不能替代即将写入的用户消息 ID。Discovery 的 LRU owner 是 `ToolDiscoveryState.update()`，测试直接覆盖私有 `OrderedDict` 会制造不存在的表示。
- 处理：`94726740` 补齐真实身份端口并删除错误 fallback 断言；`07bfb2aa` 让 discovery 测试走公开 owner。
- 失去能力：无；测试改为验证实际合同。

### 7. change gate 的已删除路径仍被标为 live

- 证据：`history_route.py`、`agent/policies/__init__.py`、`agent/tool_bundles.py` 和 `.claude/**` 已不存在，catalog 仍要求它们由活跃范围覆盖。
- 处理：`f270cc49` 把应追踪的源码移入 `deleted_paths`，移除工具目录的陈旧 live 项并更新冻结 digest；change gate 21 项通过。
- 失去能力：无；门禁继续验证删除事实，而不是要求不存在的路径存活。

### 8. AgentLoop 的第二套中断和续接 owner

- 证据：正式启动把内建 Channel 的窄 controller 注入 `ConversationRuntime`；普通 v3 Channel 走 exact `ChannelControlPort`。`PassiveMessageWorker` 对正常 `InboundMessage` 也先由 `ConversationRuntime.start_turn()` 准入。Core 和可见插件没有生产调用 `AgentLoop.request_interrupt()`。
- 冲突：旧 Loop 同时维护 `_interrupt_states`、取消 task、TTL 和 `[interrupted]` Session marker；这与 RUN-008 的唯一 attempt owner、SES-007/008 的 durable predecessor 续接形成第二套事实和写库旁路。
- 处理：删除 Loop 的 request/resume/marker 全链路；只保留 `ActiveTurnState` 作为当前执行的临时 progress view。legacy Channel 的窄 `InterruptController` 暂留，实际对象仍是 `ConversationRuntime`。
- 失去能力：只删除不可达的旧续接实现；正式 `/stop`、durable continuation 和 v3 exact binding 不变。

### 9. v3 Gate、实验 fixture 与源码快照测试

- 证据：被删部分是固定 commit/digest、CI 文本、全仓关键字扫描、退役 registry tombstone、旧 Loop mock、Wake 固定 A/B 输入和阶段性 receipt fixture。
- 保留覆盖：v3 artifact AST 与锁文件、E1 disposable write-set、E2 runtime stage、WebUI 持久化/隔离/清理、Scoped Turn admission/terminal/release、Wake durable decision 和实际 tool loop 均仍由直接行为测试覆盖。
- 处理：独立复核相关 Gate 104 项通过；同时修正 `react-core-scheduler-subagent-task-contract.md` 对已删 fixture 的陈旧引用，改指现行 scoped-turn 合同测试。
- 失去能力：无产品或插件兼容能力；删除的是历史证明工具，不是当前合同 owner。

## 仍在核验，不需要现在决定

- legacy Channel 的 `InterruptController` 应在内建 Channel 全部迁入普通 v3 exact binding 后删除；当前直接删类型仍会破坏 Telegram、QQ、Web 和 Mobile 的 import/构造链。

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
- 当前安全默认：`a75d2d6f` 已恢复“忽略 `None`、记录其他坏项并继续”的既有 ABI，使本 PR 保持 `semantic_delta: none`。若确认严格模式，应在独立迁移中定义插件错误状态、升级顺序和精确测试。

### D. 持久化 tool-chain 的坏参数是否改为 fail-loud

推荐：**是，但先核对历史数据并设计显式迁移**。

- 当前 `agent/core/types.py::to_tool_call_groups()` 把非 object 的 `arguments` 静默变成 `{}`；这会把损坏的持久记录伪装成合法的无参数调用。
- 该函数位于数据库反序列化边界，正确方向是精确报错或产生调用方可区分的损坏记录状态，不应在内部悄悄归一化。
- 当前安全默认：不在 entropy PR 中顺手改变历史数据兼容；先扫描正式 workspace 的既有 rows，再决定拒绝、隔离或迁移策略。

## 已提交修复与验证

| commit | 修复 | 验证 |
| --- | --- | --- |
| `8c17ea2d` | 恢复 context-prepared lifecycle 接线 | before-turn 12 passed |
| `35baa062` | 恢复仍被生产 Channel 使用的 interrupt protocol | channel host/client 17 passed |
| `19a51040` | 恢复正式/候选插件 service 合同 | 17 passed；另发现 3 个独立 scheduler soft 失败 |
| `7a6e5a1e` | 恢复 scheduler deny-list 构造器 | scheduler 17 + scoped 10 passed |
| `eed34b21` | 删除旧 budget/facade 测试消费者 | full collection 3290 |
| `82f5a2db` | prompt 测试跟随直接 section 结果 | 7 passed |
| `94726740` | 测试替身使用必需身份端口 | 15 passed |
| `5ceef3e5` | 完成 readonly tool 调用侧清理 | 28 passed |
| `07bfb2aa` | discovery 测试使用公开 LRU owner | 4 passed |
| `f270cc49` | 对账 change gate 删除目录 | 21 passed |
| `e52c8108` | 删除 lifecycle façade 形式测试 | 65 passed（严格 slot 个案另行核验） |
| `a75d2d6f` | 保留 lifecycle slot 既有兼容合同 | lifecycle 66 passed |
| `c905348f` | 删除 AgentLoop 第二套中断/续接 owner | runtime/control/channel 176 passed；pyright 0 errors |
| `646ff15c` | 合并最新 main，并按普通插件 Web UI 解决冲突 | Python 冲突范围 230 + 32 passed；mobile Web 122 passed；typecheck/build passed |

第一次完整 pytest 暴露 `29 failed, 3255 passed, 6 skipped`；29 项已按上面的真实半迁移、测试残留和 ABI 变化分别处理。修复至 `a75d2d6f` 后完整 pytest 为 `3278 passed, 6 skipped`；删除重复中断 owner 的 `c905348f` 通过相关 176 项。合并 `origin/main` 后全量行为为 `3283 passed, 6 skipped`，唯一失败是 mobile Gate 明确拒绝尚未提交的 merge index；形成 clean merge commit 后该 Gate 与 change/release Gate 32 项通过。完整前端 build、TypeScript typecheck 和 mobile Web 122 项通过。
