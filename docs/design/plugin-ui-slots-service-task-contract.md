# 插件 UI Slots 组合能力任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 目标分支：`codex/plugin-skills` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-ui-slots-before-20260814@92dc0f99`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件元能力底座与测试先行合同](plugin-meta-capability-task-contract.md)、[Skill/Dashboard 过渡 Service 合同](plugin-assets-service-task-contract.md)

## Goal

把 Dashboard module 从过渡 `PLUGIN_ASSETS` 收到独立 `core.ui_slots` 能力。插件只登记自己 source 内的 Dashboard backend module；Core 继续拥有路径边界、路由冲突检查、module import、generation snapshot、请求 lease 与排空。

```text
┌─────────────────┐  inject core.ui_slots  ┌──────────────────┐
│ v3 plugin Fiber │ ─────────────────────▶ │ PluginUiSlots    │
│ owns UI source  │                        │ Root registration│
└────────┬────────┘                        └────────┬─────────┘
         │ register_dashboard(path)                 │ freeze after ready
         ▼                                          ▼
┌─────────────────┐                        ┌──────────────────┐
│ Fiber Effect    │                        │ generation       │
│ exact disposer  │                        │ contribution     │
└─────────────────┘                        └────────┬─────────┘
                                                  ▼
                                         Dashboard host
                                         routes + snapshot lease
```

## Ownership and invariants

- Dashboard module 真源是插件 canonical source；`PluginUiSlots` 只登记 Root 内声明，不导入 module、不挂路由、不读取插件业务数据。
- `PluginDashboardHost` 继续拥有 route conflict、module import、generation closeable 和请求 snapshot lease；本能力不成为第二个 Dashboard host。
- 注册必须来自调用 Fiber 的 `ctx.runtime.plugin_dir`。相对路径在边界解析 symlink；不存在、绝对路径、越出插件 source、非文件或非 Python module 均 fail-loud。
- 每个插件第一版最多登记一个 Dashboard module；重复登记没有覆盖或优先级语义，直接拒绝。
- 每次登记是所属 Fiber 的 Effect；依赖撤除、reload、失败或 dispose 后精确移除。Root ready 后冻结声明，后续登记拒绝。
- Core 只从 ready Root 的冻结值更新 generation contribution。candidate、stable、latest、lease、晋升、丢弃与恢复的 owner 不移动。
- v2 `dashboard_module()` 保持原行为。未进入正式发布的 `PLUGIN_ASSETS` 过渡面直接删除，不保留 deprecated alias。

## Public seam

```python
from agent.plugin_composition import UI_SLOTS

inject = (UI_SLOTS,)

async def apply(ctx, config):
    ui_slots = ctx.require(UI_SLOTS)
    await ui_slots.register_dashboard(ctx, "dashboard.py")
```

- `UI_SLOTS = ServiceKey("core.ui_slots")`
- `PluginUiSlots.register_dashboard(ctx, relative_path) -> None`
- 第一版只公开已有真实消费者需要的 Dashboard backend slot。
- Mobile UI 的 `slots` catalog、静态 asset 与只读 RPC 已由 `PluginMobileUiProvider` 拥有。它不是本 PR 的兼容别名；迁移时必须另立行为等价合同。
- Dashboard frontend panel 的发现和 bundle 注册保持现状，不通过本 Service 重画客户端协议。

## Persistence and effects

```yaml
change_type: refactor
semantic_delta: compatible
capability_owner: "Plugin owns Dashboard source; Core owns route binding and generation publication."
consumer_scope:
  - v3 plugins with a Dashboard backend module
protected_state:
  - formal workspace and plugin-data
  - installed plugin artifacts and manifests
  - stable/latest generation pointers
  - existing Mobile UI catalog, asset and RPC protocol
allowed_effects:
  - temporary plugin roots and workspaces in tests
  - Root-local registrations, Dashboard bindings and deterministic cleanup receipts
forbidden_effects:
  - canonical plugin source or installed cache mutation
  - formal plugin migration
  - formal workspace, plugin-data, Session or external API writes
rollback: "Revert this adjacent PR or return to backup/plugin-ui-slots-before-20260814; v2 declarations and the previous stable generation remain intact."
```

## Verification

- unit fixture 覆盖 freeze、duplicate、symlink escape 与 Effect cleanup；
- real namespace loader 证明 `core.ui_slots` 经 inject 编译为 Dashboard binding；
- installed candidate 证明隔离 Root 与正式 Root 得到等价 Dashboard contribution；
- leaked-disposer mutant 与正确实现运行同一 fixture，证明 owner cleanup oracle 能发现残留登记；
- public plugin generation Gate 绑定 exact source digest，并回归 v2 Dashboard、hot reload 与晋升。
