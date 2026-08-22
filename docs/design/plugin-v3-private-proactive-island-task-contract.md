# 插件 v3 Default/Wake Proactive 私有兼容岛任务合同

- 状态：approved / ready for implementation
- 日期：2026-08-16
- 实现起点：`7f3852df`
- 清单：C20
- 唯一 consumers：`default_proactive`、`proactive_flow`、`drift_flow`、`wake_proactive`、
  `wake_proactive_flow`、`wake_drift_flow`

## 1. 目标与边界

最终 runtime 不再公开 v2 `Plugin`、`PluginContext`、`proactive_*()` 或 `jobs()` ABI。Default/Wake 两族暂不
重写其成熟的 flow/runtime/state machine；六个 in-tree module 改成 `api_version = 3` 薄入口，通过 Core-private
bridge 把原领域对象投影到 committed proactive kernel。

```text
six exact in-tree v3 modules ── private registration ──► candidate descriptors
                                                          │ freeze/seal
                                                          ▼
committed snapshot ── Core private host ──► existing Default/Wake runtime

external plugin ── old proactive/jobs claim or private bridge ──► fail-loud
```

`semantic_delta: compatible`。保留 tick、gate、source、dedupe、ack/cursor、flow DAG、wake/drift state、文件与
数据库 schema、turn/delivery owner；不把这个 bridge 变成第三套公共 proactive API，也不顺手重写领域 runtime。

## 2. 私有 admission

1. allowlist 固定六个 `plugin_id + resolved in-tree source root + package member identity`。仅名字相同、installed
   artifact、symlink、workspace/cache copy 或 external repository 都不得取得 bridge。
2. private ServiceKey/registry 不从 `agent.plugin_composition.__init__`、插件合同或 public typing package 导出。
   同 UID Python 不是安全沙箱；Core 仍在 mount/admission 以 exact runtime identity fail-loud，supported API 不承诺
   防止恶意反射。
3. 六个模块只暴露 `api_version = 3`、精确 `apply(ctx, config)` 与必要 static dependency/workspace roots；不得再继承
   `Plugin`、构造 `PluginContext` 或实现任一 v2 fixed method。
4. private definition 只允许显式 enum family/member 与 source-relative factory export；不接受任意 module path、callable
   closure、Manager/EventBus/Session repository。candidate/formal 各自解析自己的 export，不跨 Root 携带 handler。
5. candidate 只 freeze descriptor，不实例化 runtime、不启动 tick、不拉 source、不调 LLM、不写正式 proactive data、
   不 enqueue/send。formal publication 后 private host 才 materialize。

allowlist 与 factory 顺序固定为下表。`<project_root>` 由一个 Core-owned 常量从正在执行的 Core
package 文件位置解析（等价于 Core 模块 `Path(__file__).resolve(strict=True)` 的 repository/package root），不读取
`PluginManager.plugin_dirs`、workspace、环境变量、配置或调用方参数；复制仓库即使被加入 plugin search path，也不是
当前进程的 canonical Core source root：

| family/order | package id | member/plugin id | lexical canonical root | entry | private exports |
|---|---|---|---|---|---|
| default/0 | `default-proactive` | `default_proactive` | `<project_root>/plugins/default_proactive` | `plugin.py` | `DefaultRuntimeFactory`、`DefaultModuleFactory`、`build_default_lifecycle` |
| default/1 | `default-proactive` | `proactive_flow` | `<project_root>/plugins/proactive_flow` | `plugin.py` | `ProactiveModuleFactory` |
| default/2 | `default-proactive` | `drift_flow` | `<project_root>/plugins/drift_flow` | `plugin.py` | `DriftModuleFactory` |
| wake/0 | `wake-proactive` | `wake_proactive` | `<project_root>/plugins/wake_proactive` | `plugin.py` | `WakeRuntimeFactory`、`WakeProactiveModuleFactory`、`build_wake_lifecycle` |
| wake/1 | `wake-proactive` | `wake_proactive_flow` | `<project_root>/plugins/wake_proactive_flow` | `plugin.py` | `WakeContentModuleFactory` |
| wake/2 | `wake-proactive` | `wake_drift_flow` | `<project_root>/plugins/wake_drift_flow` | `plugin.py` | `WakeDriftModuleFactory` |

admission 同时要求 lexical root 的每一级都不是 symlink、`resolve(strict=True)` 与表中 exact root 相等、package manifest
的 id/member/order 完全匹配、entry 是该 root 的直接 `plugin.py`。installed/cache copy、同名 external module、错误
package/member、symlink 和从 external wrapper re-export 上表 callable 全部 fail-loud；callable 的 `__module__` 也必须
属于 exact admitted entry，不能借 re-export 绕过来源身份。

## 3. 保留与删除

### 保留的领域实现

- `proactive_v2` 下 Default/Wake runtime 依赖的 frame/config/sensor/runtime scope；
- `plugins/default_proactive`、`proactive_flow`、`drift_flow`、`wake_proactive`、`wake_proactive_flow`、
  `wake_drift_flow` 的领域模块、工具、prompt、state 与 schema；
- proactive/wake/drift DB、`PROACTIVE_CONTEXT.md`、`proactive_pending.md` 的既有 owner 与恢复语义；
- Core proactive kernel 的 tick、presence/busy、turn enqueue、delivery、ack 与 drain owner。

Default/Wake Dashboard 的只读 reader 同属这个私有岛：Dashboard host 把对应内建 family 的路由绑定到 exact
`RuntimeSnapshot.private_proactive_catalog`，每次请求随 snapshot lease 分派，reader 实现位于 exact Core source root。
package manifest 只声明面板资源，不提供可执行 backend，也不形成 external plugin 可调用的 Dashboard ABI。

目录名 `proactive_v2` 在第一轮可以保留为私有领域实现标识；它不再代表 public Plugin API。若后续重命名，只能做
机械 import migration 并单独证明 state/schema/behavior 不变。

### full-fleet 删除 Gate 通过后删除的公共兼容

- `Plugin.proactive_modules/proactive_lifecycles/proactive_module_factories/proactive_runtime_factories/
  proactive_sources/jobs`；
- `PluginContributions`、`RuntimeSnapshot`、`PluginManager` 的 public v2 proactive/job mutable/fixed fields 与 collectors；
- bootstrap 对旧 Manager lists/factories 的固定传参；
- `PluginContext` 为这些旧能力提供的注入路径；DTO 本身只有在全体 v2 consumer 清零并迁入 C18 internal metadata
  后才删除，C20 不单独删除仍被其他 v2 使用的类型；
- external v2 proactive/job contract Gate、manifest/discovery adapter 与 deprecated aliases。

删除后生产扫描发现 external plugin 声明上述 v2 method、`api_version != 3`（含缺失、整数 `2`、整数 `4`、字符串
`"3"`），或 import public private-bridge 路径时，
install、doctor、manifest/cache discovery 与 runtime admission 都抛明确异常/返回非零；不能记录后跳过、静默忽略或
退回旧 loader。该全局边界只在其他 v2 family 也完成自己的删除批次后启用，不能由 C20 提前误杀尚在迁移的插件。

`plugin-api-v2.lock.json` 与 `plugin_api_v2_gate.py` 在最终 pure-v3 Gate 建立后删除；在此之前它们只是迁移库存，
不能作为 C20 成功证据。

## 4. publication 与生命周期

1. candidate/formal descriptor 冻结为 Core-private `DefaultWakeProactiveCatalog` 并进入
   `RuntimeSnapshot.private_proactive_catalog` 与 snapshot identity；handler/factory 绑定 exact Root/generation，
   不进入内容 hash。catalog descriptor 固定 family/member/package/order/export names/source revision。
2. `PrivateProactiveHost.prepare_components(tx_id, target_lease, target_catalog)` 只读 exact closed target catalog，生成纯
   `PrivateProactivePlan(family, runtime_scope, lifecycle, runtime_factory, module_factories, source_catalog)`；该阶段不得创建
   timer/process/model/client/subscription。只有 ActivityHost 完成 old drain 后才能调用 `materialize_closed()` 生成 binding。
   `ProactiveLoop` 只从该 binding 构建现有 runtime，不再读取 Manager mutable lists。C15 的 committed source catalog
   负责填充 `ProactiveRuntimeScope.proactive_sources`，private bridge 不复制第三套 source owner。
3. `PrivateProactiveHost` 只是 C15 `ActivityHost` 的 Core-private child adapter，不拥有 publication、stable pointer、lease、
   admission 或 drain。唯一 ActivityHost 调用其 `prepare_components/materialize_closed/stop_components/
   restore_components/close_components`，并统一执行 old kernel
   admission close、lease/in-flight drain、new materialize/start、pointer finalize 与 new admission open；private adapter
   不读取 Manager mutable list，也不能单独切 kernel。
   child adapter 实现 C15 唯一的 Core-private `ActivityChildAdapter[PrivateProactivePlan,
   PrivateProactiveBinding]`，参数、返回与阶段顺序完全相同：

   ```python
   plan = private.prepare_components(tx.id, tx.target_lease, tx.target_catalog)
   await private.stop_components(tx.id, tx.old_binding)       # ActivityHost.drain 内
   new_binding = await private.materialize_closed(tx.id, plan)  # old drain 后
   await private.restore_components(tx.id, tx.old_binding)    # only during ActivityHost rollback
   await private.close_components(tx.id, new_binding)         # cleanup/retry
   ```

   prepare 的不可变 plan 固定 exact snapshot lease/catalog identity；target plan/binding 的后续调用携带同一个
   transaction id 与 exact object。`old_binding` 保留其原 publication transaction id，跨 reload 的 stop/restore/close
   只校验 adapter ownership 与 exact object，不把新的 publication transaction id 错当成旧 binding 身份。adapter 不得
   lookup current snapshot、不得创建/释放 snapshot lease，也不得独立 pause/drain/finalize/open。
   所有 stop/restore/close 结果回到 ActivityHost journal。
4. old turn/job 继续持有 old snapshot/handler 到 terminal；new admission 只见 new binding。
5. start/cancel/restore/stop 失败保留 generation/runtime owner、Health/Incident 与 cleanup-pending retry；恢复 pointer
   不能伪装文件、DB、timer、sender 或 model request 已回滚。
6. candidate discard、failed promotion、terminate 后 clone module、timer/task/listener/handler/LLM binding 全零。

## 5. 验证与停止条件

- allowlist：六个 exact identity 全通过；同名 external、installed copy、symlink root、错误 package/member、external
  re-export 全拒绝；把完整 Core repository 复制到另一目录并加入 `plugin_dirs` 也必须拒绝；
- admission：六个 entrypoint 都是纯 v3；production scan 不再发现其 v2 base/fixed methods/PluginContext；
- candidate：runtime/timer/source/model/turn/sender 调用为 0，正式 DB/Markdown/plugin-data digest 不变；
- stable：fixed clock 下 Default 与 Wake 各跑 normal/empty/skip/failure/cancel/restart，旧 state/schema 行为等价；
- reload：old in-flight 完成，新 binding 才接新 tick；start failure 恢复旧 exact admission/pointer，或清空 kernel/lease
  并保持 fail-closed，cleanup failure 可查询/retry；
- host：snapshot private catalog identity、family/order、C15 source projection、exact lease、ProactiveLoop adapter 与
  Dashboard family 路由换代全部可观察；删除旧 lists 后 Default/Wake normal path 仍执行；
- deletion：C20 的两个私有岛 E3 只证明六个 in-tree module，不授权删除公共 v2 owner。只有其他 external proactive/job
  consumer 全部迁移、最终 full-fleet E3/J 通过且 production/cache/canonical-source zero-consumer scan 为空后，才能证明
  除 private allowlist/host 和保留领域包外，`Plugin`/Manager/Snapshot/bootstrap 无 public v2 proactive/job consumer；
  install/doctor/manifest/cache/discovery 对 external `api_version` 缺失/2/4/字符串与 v2 method matrix 全 fail-loud，
  旧 lock/Gate 已替换；
- E3 使用两个互斥子矩阵：Default package 与 Wake package 分别在独立 manifest/workspace 跑 full boot/catalog/
  promote/reload/normal/empty/skip/failure/cancel/restart，再与 C15/C21、recording MCP/model/sink 汇总成一次报告；
  E4 只在复制 workspace 做 WebUI-only rehearsal。

出现 external plugin 取得 private bridge、candidate 发送/写正式状态、旧 DB/schema 被迁移、双 kernel/timer、ack/cursor
丢失、rollback 只恢复内存或删除后仍有 public v2 consumer，均停止交付。

本合同已经三轮独立只读复审；exact identity、Core source root、host/catalog owner、external admission 与最终删除
边界均无残留 P0/P1。该结论只批准实现，不把尚未编写的 C20 代码或私有岛 E3 记为完成。

## 6. 实现顺序与回滚

1. C15/C21 先闭合 shared committed host、lease、job/LLM 与 source/module catalog。
2. 新增 Core-private allowlist/definition/host adapter并先迁 Default family，再迁 Wake family。
3. 六个模块完成领域等价回归后运行私有岛 E3；它只把 C20 标为 candidate。等全部 external proactive/job consumer
   迁移完成，再运行 full-fleet E3/J 与 zero-consumer scan；只有该最终 Gate 才授权公共 v2 proactive/job 删除批次。
4. Core 恢复点为 `7f3852df`；每个内建模块迁移前建立独立 Git backup。所有行为验证使用一次性 workspace、fixed
   clock、recording MCP/model/sink，不写 hua-home。
