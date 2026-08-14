# 插件 Turn event leaf import 任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 目标分支：`codex/plugin-ui-slots` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-event-import-contracts-before-20260814@1e0dfe62`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件 lifecycle 接入点合同](plugin-lifecycle-seam-task-contract.md)

## Goal

让外部插件可以在尚未导入 Core phase runtime 时安全取得 typed Turn event key。事件名称、payload、serial mode、触发位置、失败语义与 generation owner 均不改变。

```text
plugin.py ──import──► agent.turn_events.<phase>
                              │
                              │ same key object
                              ▼
Core phase runtime ───────► run_turn_stage_event
```

## Ownership and public seam

- `agent.turn_events.prompt_render` 只拥有 Prompt Render 的公开 key；`agent.turn_events.after_reasoning` 只拥有 After Reasoning 的公开 key。
- leaf module 只依赖 `agent.plugin_composition`。payload 类型只在类型检查时导入，因此插件 entrypoint import 不启动 phase、facade、Session 或 passive turn runtime。
- phase 实现从 leaf module 导入同一个 key object；不复制名称，不建立第二份 event catalog。
- `agent.lifecycle.composition` 仍只负责从 request-bound snapshot dispatch；Context/Fiber 仍拥有 listener、顺序与清理。
- 不增加 `LifecycleEvents` Service、priority、waterfall、动态字符串订阅或 deprecated alias。

```python
from agent.turn_events.after_reasoning import AFTER_REASONING_BEFORE_EVENT_BUS
from agent.turn_events.prompt_render import PROMPT_RENDER_AFTER_EVENT_BUS

async def apply(ctx, config):
    await ctx.on(PROMPT_RENDER_AFTER_EVENT_BUS, render_listener)
    await ctx.on(AFTER_REASONING_BEFORE_EVENT_BUS, answer_listener)
```

## Change and verification

```yaml
change_type: refactor
semantic_delta: none
capability_owner: core
protected_state:
  - event names, payload types, dispatch mode and exact positions
  - legacy phase and EventBus order
  - generation publication, workspace, plugin-data and SessionDB
allowed_effects:
  - fresh interpreter import probe
  - temporary composition roots in tests
forbidden_effects:
  - formal runtime or plugin installation
  - persistent state or external API writes
rollback: "Revert this adjacent PR or return to backup/plugin-event-import-contracts-before-20260814."
```

- fresh interpreter 证明导入 leaf contracts 后没有加载 `agent.lifecycle.phases.after_reasoning`；
- lifecycle seam tests 继续证明三个 exact position、legacy 顺序、Bail 与 listener failure；
- Citation/Meme cross-repository tests 使用 leaf import，作为首个真实 consumer；
- public Gate 绑定 exact source digest。
