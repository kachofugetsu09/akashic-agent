# 插件 v3 generation loader 任务合同

- 状态：implemented / verified
- 日期：2026-08-14
- 实现基线：`06f5426d9854a4bbbab9444c2196ee34b416df55`
- 关联条款：PLG-001～PLG-004、PLG-008～PLG-010、PLG-014
- 上游：[0036](../decisions/0036-plugin-composition-keeps-promotion-owner.md)、[lifecycle 接入点合同](plugin-lifecycle-seam-task-contract.md)

## 1. 目标与边界

本 PR 把 Python 组合 Root 接入真实插件发现、安装、generation、stable/latest、候选隔离、晋升和 lease 排空链。v3 插件不继承 legacy `Plugin`，也不实现 `prepare / activate / retire / terminate`；Core 只识别模块的命名导出并挂载 `apply`。

`semantic_delta: compatible`。v2 类插件和现有 contribution 收集、EventBus、phase、Job、UI、Skill、MCP 路径保持不变。本 PR 不迁移 Citation/Meme，不删除 legacy host，不写正式 workspace/plugin-data，不调用外部 API。

```text
plugin.py named exports
        │ name / inject / Config / apply
        ▼
ComposablePlugin declaration ── mount ──► generation CompositionRoot
                                             │
                            RuntimeSnapshot + TopologyView
                                             │
                      stable/latest validation + promotion
                                             │
                                lease drain ──► dispose
```

## 2. v3 模块合同

```python
from pydantic import BaseModel
from agent.plugin_composition import Context, ServiceKey

api_version = 3
name = "example"
version = "1.0.0"
inject = (ServiceKey[object]("required.service"),)

class Config(BaseModel):
    enabled: bool = True

async def apply(ctx: Context) -> None:
    config = Config.model_validate(ctx.config)
    runtime = ctx.runtime
    # runtime 由 Core 分配 plugin_id/plugin_dir/data_dir/workspace/config；
    # 插件自己实现领域读取、注册和 Effect 清理。
```

- `name / version / apply` 必需；`inject / desc / author` 可选。配置模型是插件内部实现，不属于模块导出协议。
- `inject` 只接受 typed `ServiceKey`。扫描顺序不承担依赖语义；启动期先出现的 consumer 可以等待随后出现的 provider，完整 stable 树结束仍缺 Service 时启动失败。
- `ctx.runtime` 只给出 Core 拥有的身份、路径和配置接入点，不替插件实现领域数据协议。
- `apply` 的注册、监听器、任务和子插件必须归属 Fiber/Effect；Root 退役时逆序回收。
- 装配通过实际入口、资源注册和依赖检查判断合法性；插件业务自测不再是模块导出协议。晋升职责见 [0071](../decisions/0071-plugin-composition-and-whole-runtime-updates.md)。

## 3. publication 与失败语义

- 每次 v3 拓扑变化创建完整候选 Root；MCP 只作为插件 Root registry 与 generation host 的一部分发布，不再存在独立 workspace MCP payload。
- installed candidate 使用隔离 workspace/plugin-data 挂载并验证；晋升前排空 candidate lease，以同一 generation/config 在正式路径重建 Root，identity 不一致即拒绝。
- stable/latest 快照各自持有 Root。旧快照仍有 lease 时旧 Root 不清理；最后一个引用排空后才 dispose。
- apply 失败、required Service 缺失、重复 Service、拓扑不 ready、外部效果审计失败都不能进入可租用候选。
- 启动期为解决跨插件扫描顺序只允许暂存无错误的 required-pending Fiber；`load_all()` 返回前必须形成 ready Root，否则 fail-loud。

## 4. 验证与回滚

- targeted：namespace/config/runtime、反扫描顺序依赖、缺失依赖、builtin reload、installed latest/promote、Root drain、v3 install。
- cumulative：PluginManager hot reload/install、composition kernel/events/executor/lifecycle。
- static：compileall、Basedpyright error-level、`git diff --check`。
- Gate：`python docker/debug/gate.py run --base origin/main`。
- 停止条件：v2 回调顺序变化、candidate 写正式数据、snapshot identity 漂移、Root 在 lease 前清理。
- 回滚：`/mnt/data/coding/akasic-agent/.backups/20260814-pre-plugin-v3-loader-06f5426d.bundle`。
