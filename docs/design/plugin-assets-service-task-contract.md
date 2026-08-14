# 插件 Skill/Dashboard 组合 Service 任务合同

- 状态：implemented / partially superseded
- 日期：2026-08-14
- 实现基线：`2dd90295dc23e1c8577aecdfa9e014cea8e350a4`
- 关联条款：PLG-001～PLG-004、PLG-006、PLG-008～PLG-010、PLG-013～PLG-014、WSP-001～WSP-002
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[v3 loader 合同](plugin-v3-loader-task-contract.md)
- 后续：Skill 部分由 [插件 Skills 组合能力合同](plugin-skills-service-task-contract.md)取代；`PLUGIN_ASSETS` 当前只暂留 Dashboard 注册，下一张 UI Slots PR 删除该过渡公开面。

## 1. 目标与能力归属

本 PR 提供 Root-scoped `PLUGIN_ASSETS` Service。v3 插件通过 `inject` 取得 Service，再在自己的 `apply(ctx, config)` 中注册 Skill root 或 Dashboard module；Core 只负责路径边界、generation catalog、Dashboard host、snapshot 发布和 lease 排空。

```text
Core Root ──provide──► PLUGIN_ASSETS
                           ▲
                           │ inject / require
                     plugin apply(ctx)
                           │
                  register_skill/dashboard
                           │ Fiber Effect
                           ▼
                  frozen asset contribution
                     ┌─────┴─────┐
                     ▼           ▼
                Skill catalog  Dashboard host
```

`semantic_delta: compatible`。v2 `skill_roots()`、`drift_skill_roots()` 和 `dashboard_module()` 保持不变；本 PR 不把这些领域声明继续增加到 v3 固定 namespace，也不迁移 Citation/Meme。

## 2. 接口与失败语义

```python
inject = (PLUGIN_ASSETS,)

async def apply(ctx, config):
    assets = ctx.require(PLUGIN_ASSETS)
    await assets.register_skill(ctx, "skills")
    await assets.register_dashboard(ctx, "dashboard.py")
```

- 注册是当前 Fiber 的 Effect；Root 失败或排空时按逆序清理。
- 相对路径在注册边界解析 symlink；不存在、绝对路径、越出 plugin root、非目录 Skill 或非 Python Dashboard 都 fail-loud。
- 每个插件最多注册一个 Dashboard module；重复声明不定义覆盖顺序。
- Core 在 Root ready 后冻结声明，再编译 generation；Root pending、错误或外部效果存在时不发布资产。
- installed candidate 在隔离 Root 验证；正式恢复后必须得到相同 snapshot identity，否则沿用既有晋升拒绝。

## 3. 状态与副作用

- Skill/Dashboard 正文真源仍是插件 source；本 Service 不写或删除 canonical source。
- Skill catalog 是 generation-scoped 可重建临时快照，随既有 scope 清理。
- workspace Skill 软链接仍由 `PluginSkillLinker` 的既有 ownership journal 管理；本 PR 不新增链接写入协议。
- Dashboard module 仍由 `PluginDashboardHost` 装载；插件领域数据由插件在既有 `plugin-data`/workspace 合同内实现，Core 不替插件创建业务 repository。
- 卸载不级联删除 plugin-data；Session、数据库、正式 workspace 业务数据均不在本 PR write set。

## 4. 验证与回滚

- targeted：真实 v3 Skill catalog、Dashboard binding、Effect 回收、symlink escape、Dashboard 重复声明。
- promotion：installed v3 candidate 的隔离加载、正式重建、Skill/Dashboard 贡献与 identity 等价。
- cumulative：composition、PluginManager、Skill links 与 hot reload 回归。
- static：compileall、Basedpyright error-level、`git diff --check`。
- Gate：`python docker/debug/gate.py run --base origin/main`。
- 回滚：`/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-assets-service-2dd90295.bundle`。
