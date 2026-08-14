# 插件 Skills 组合能力任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 目标分支：`codex/plugin-agent-input` 之后的 Draft stacked PR
- 恢复点：`backup/plugin-skills-ui-slots-before-20260814@f96b78e2`
- 上游：[0037](../decisions/0037-plugin-services-name-capabilities-not-categories.md)、[插件元能力底座与测试先行合同](plugin-meta-capability-task-contract.md)、[Skill/Dashboard 过渡 Service 合同](plugin-assets-service-task-contract.md)

## Goal

把 Skill 与 Drift skill 从过渡的 `PLUGIN_ASSETS` 中拆成独立 `core.skills` 能力。插件只登记自己 source 内的 Skill root；Core 继续拥有路径边界、generation catalog、stable/latest、workspace 软链接投影与回收。

```text
┌─────────────────┐  inject core.skills  ┌──────────────────┐
│ v3 plugin Fiber │ ────────────────────▶ │ PluginSkills     │
│ owns Skill body │                       │ Root registration│
└────────┬────────┘                       └────────┬─────────┘
         │ register(path, drift=...)               │ freeze after ready
         ▼                                         ▼
┌─────────────────┐                       ┌──────────────────┐
│ Fiber Effect    │                       │ generation Skill │
│ exact disposer  │                       │ catalog          │
└─────────────────┘                       └────────┬─────────┘
                                                  ▼
                                         workspace soft links
                                         (rebuildable view)
```

## Ownership and invariants

- Skill/Drift skill 正文真源是插件 canonical source；已安装 artifact 保存不可变版本，workspace 软链接只是 active generation 的可重建投影。
- `PluginSkills` 只登记 Root 内的 source roots，不解析 Skill 正文、不写 workspace、不管理链接，也不成为第二个 `PluginSkillHost`。
- 注册必须来自调用 Fiber 的 `ctx.runtime.plugin_dir`，相对路径在边界解析 symlink；不存在、绝对路径、越出插件 source 或非目录均 fail-loud。
- 普通与 Drift 是同一能力的两个既有 catalog lane；同一插件、同一 lane、同一路径的重复登记没有覆盖语义，直接拒绝。
- 每次登记是所属 Fiber 的 Effect；依赖撤除、reload、dispose 后精确移除。Root ready 后冻结声明，后续登记拒绝。
- Core 只从 ready Root 的冻结值更新 generation contribution。只有 Skill roots 变化才重建 Skill catalog；Dashboard 变化不再触发无关 catalog rebuild。
- v2 `skill_roots()` 与 `drift_skill_roots()` 保持原行为。`PLUGIN_ASSETS.register_skill()` 是未进入正式发布的过渡 v3 seam，本 PR 直接移除，不保留 deprecated alias；Dashboard seam 随后由 [UI Slots](plugin-ui-slots-service-task-contract.md)收口。
- publication plane 不变：完整 Root 仍由 Core 生成候选回执，stable/latest、lease、晋升、丢弃与恢复 owner 不移动。

## Public seam

```python
from agent.plugin_composition import SKILLS

inject = (SKILLS,)

async def apply(ctx, config):
    skills = ctx.require(SKILLS)
    await skills.register(ctx, "skills")
    await skills.register(ctx, "drift/skills", drift=True)
```

- `SKILLS = ServiceKey("core.skills")`
- `PluginSkills.register(ctx, relative_path, *, drift=False) -> None`
- `drift=False` 写入普通 Skill catalog；`drift=True` 写入既有 Drift skill catalog。
- 第一版不直接登记单个 Skill、不定义 rank/override/provider、不监听文件变更，也不允许 snapshot 发布后的运行时新增。

## Persistence and effects

```yaml
change_type: refactor
semantic_delta: compatible
capability_owner: "Plugin owns Skill source; Core owns catalog, projection and generation publication."
consumer_scope:
  - v3 plugins with static Skill or Drift skill roots
protected_state:
  - formal workspace skills and drift/skills projections
  - plugin manifest, installed artifact and plugin-data
  - stable/latest generation pointers
allowed_effects:
  - temporary plugin roots, catalogs and workspaces in tests
  - Root-local registrations and deterministic disposer receipts
forbidden_effects:
  - formal Skill link creation, removal or adoption
  - canonical plugin source or installed cache mutation
  - formal plugin migration
rollback: "Revert this adjacent PR or return to backup/plugin-skills-ui-slots-before-20260814; v2 declarations and the previous stable generation remain intact."
```

## Verification

- unit fixture 覆盖 normal/Drift lane、声明顺序、freeze、duplicate 与 symlink escape；
- real namespace loader 证明 `core.skills` 经 inject 进入 Skill catalog，并在 reload 后保持旧 snapshot catalog 到 lease 排空；
- installed candidate 证明隔离 Root 与正式 Root 得到等价 Skill contribution；
- leaked-disposer mutant 与正确实现运行同一 fixture，证明 owner cleanup oracle 能发现残留登记；
- public plugin generation Gate 绑定 exact source digest，并回归 v2 Skill、Dashboard、hot reload 与 uninstall 投影。
