# 0036 · 插件组合内核保留现有晋升 owner

- 状态：accepted / implemented phases 1–6
- 日期：2026-08-14
- 关联条款：PLG-001～PLG-014、WSP-001～WSP-005、ERR-001、TST-001～TST-007
- supersedes：无
- superseded by：无

## 背景

当前 `PluginManager` 同时理解插件加载、固定 lifecycle、七组 ReAct phase、Job、Channel、MCP、UI、Skill 和主动流程。它把各领域的贡献枚举成一个封闭总表，插件新增能力必须先修改 Core。DeepSeek Harness 使用的 Cordis 则把插件组合收敛为 Context、Service、Inject、Fiber、Effect 和 Scope，领域能力由普通 Service 自己定义。

Akashic 已经拥有 Cordis 不提供的候选隔离、stable/latest、父 Turn 授权、generation lease、自验证和回滚日志。替换组合机制不应丢弃这些发布语义，也不应把 TypeScript Loader、HMR 或 Schemastery 再复制成第二个 owner。

## 决定

Akashic 新插件采用 Python 实现的最小组合内核：Root Context 持有一代完整拓扑，Service 表达能力，Inject 表达 Fiber 的硬依赖；可选能力不进入 inject，由使用点查询或不阻塞 Root readiness 的嵌套 Fiber 承载。Fiber 管理依赖波动下的状态，Effect 在所属 Root/Fiber scope 内逆序回收资源。插件公开入口是 `apply(ctx)`；Job、Channel、Prompt、输出处理、UI、MCP、存储和外部效果都是领域 Service，不进入组合内核的固定枚举。第一阶段不另造一个只转发 Fiber 所有权的 `Scope` 类；Cordis 的独立 service isolation scope 留到真实迁移插件提出需求时再实现。

事件能力只提供三种公开 dispatch：同步串行 `emit`、支持显式 `Bail` 的异步串行 `serial`、等待全部 settle 并聚合失败的异步并发 `parallel`。同步并发由受限 `ExecutorService` 提供，不允许工作线程取得 Context/Fiber。listener 只使用 generation 内稳定注册顺序；真实能力依赖使用 inject，不引入 priority、listener DAG 或通用 waterfall。

现有 publication plane 保持唯一 owner：安装 artifact、候选隔离、generation identity、行为验证回执、stable/latest、父 Turn 终点授权、snapshot lease、晋升、丢弃和恢复日志继续由 Core 管理。候选发布单位是完整 Root Context 拓扑，不是单个回调或单个 Service。

现有插件在迁移完成前继续由一个 legacy host 承载。新内核不得为每个旧插件增加适配器，也不得改变旧 lifecycle、phase/slot 顺序、错误传播、plugin-data 或外部效果。每个领域迁移完成后删除对应旧分支，最终删除 legacy host。

第一阶段只实现组合内核、候选回执和无外部效果的实验插件，并在带 run identity 的隔离 workspace 验证。它不接管正式 manifest、正式 plugin-data、真实渠道或远程 API。

```text
┌──────────────── publication plane / Core ────────────────┐
│ artifact → candidate topology → behavior receipt → stable │
│ generation identity / lease / journal / rollback          │
└──────────────────────────┬─────────────────────────────────┘
                           │ 发布一个完整 Root Context
                           ▼
┌──────────────── composition plane ────────────────────────┐
│ Context ─ Service ─ Inject ─ Fiber ─ Effect                │
└──────────────────────────┬─────────────────────────────────┘
                           │ 普通领域 Service
                           ▼
 Prompt / Output / Tool / Job / Channel / UI / MCP / Storage
```

## 理由

这保留了 Akashic 能验证自身修改的优势，同时消除 Core 对插件领域形状的先验知识。选择性转译 Cordis 的组合语义，比完整移植 Loader、Include、HMR、Schemastery 和 CosmoKit 更少产生重复 owner；Python 边界继续使用项目现有类型与 Pydantic，不宣称与 TypeScript 配置系统完全等价。

## 影响

- 正面影响：新领域能力不再要求扩展 `PluginManager` 固定字段；依赖出现和消失可以局部激活或回收 Fiber；资源由注册它的插件拥有清理。
- 兼容性：旧插件默认行为不变；新插件只走显式的新入口。迁移必须逐插件证明行为等价。
- 数据和迁移：第一阶段只写隔离 workspace 的实验回执；Core 只分配 plugin-data 根和权限，插件拥有其 schema、迁移、保留与删除协议。
- 失败与回滚：组合失败拒绝候选，stable 不变；恢复点是实施前 Git bundle 和旧 stable generation。Effect 清理不能冒充文件、数据库或外部调用已回滚。

## 验收

- [x] Cordis 生命周期关键行为在 Python 内核测试中通过，包括依赖波动、重入卸载、epoch 防陈旧激活和逆序清理。
- [x] 实验插件在隔离 workspace 中完成候选装载、自验证、晋升、依赖撤除和资源归零，且正式 workspace 与插件清单零写入。
- [x] 旧插件回归保持通过，现有 publication plane 的 stable/latest 与 parent-Turn owner 不变。
- [x] Root 拓扑以不可变 `TopologyView` 进入 RuntimeSnapshot；内容 identity 覆盖 Fiber 依赖、Service、Effect 与 listener 顺序，晋升仍由 Core 复核。
- [x] Prompt 与回答处理通过当前 snapshot Root 的 typed serial 接入点扩展；旧 phase/slot 与 EventBus 顺序保持不变。
- [x] v3 namespace 插件由真实 install/generation loader 挂载到 snapshot Root；stable/latest、自验证晋升和 lease 排空仍由 Core 拥有。
- [x] v3 回答 listener 可声明 assistant 持久 metadata；Session 写入与字段冲突校验仍由 Core persistence phase 拥有。

## 未决问题

- Citation/Meme、GitHub Watch 和其他现有插件的迁移顺序由各自差分回放结果决定，不在第一阶段预先承诺。
