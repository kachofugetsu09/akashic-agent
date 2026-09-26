# Issue 766 · 能力依赖与执行归属

状态：Issue 766 实现完成，最终验证与独立概念 Gate 见本文末尾。

## 目标与取舍

能力只依赖自己需要的合同；界面变化只影响界面；执行期间由框架保护实际 owner。
复用 Context、Fiber、Effect、Message 和 SourceSession，不增加并行运行图或恢复账本。

维护者确认：binding 固定业务选择，使用当前兼容实现；不兼容明确失败，归档只作来源证据。
这与 PLG-018、0070 一致，不恢复历史 Root，不用代码散列替代业务兼容性检查。
维护者要求：不要为极少见场景扩散大量代码；本任务不新增或改写单元测试。
唯一获批例外：给 tests/test_tool_bindings.py 既有夹具补 inject=(TASKS,)，不改断言。

参考本地 deepseek-harness 的 `477b4f4205`：

- `packages/preset/agent-preset-registry/src/index.ts` 保留仍被 Agent 使用的实际 revision；旧 revision 在引用释放后回收。
- 同包 `README.zh.md` 与 `packages/api/session-controller/src/agent.ts` 明确区分持久 preset 标识和当前定义；重启恢复不锁历史代码。
- `packages/core/session/src/repair.ts` 与 `packages/core/agent-loop/src/index.ts` 把中断调用写成未开始或结果未知，不盲目重放外部效果。

Akashic 借鉴运行中实例与持久恢复的分工，保留自己的单 Root 和局部排空机制。

## 已实现的边界

```text
来源 ── 已提交 Input + Task + 本次授权 ── ReplyProgram
  │                                         │
  └─ 原指针、准入、业务结算                  └─ provider 捕获模型、内容、工具依赖

计算 Fiber（Models、驱动、Akasha、Wake）
└─ 可选 UI 子 Fiber ── UI 合同

工具贡献者 ── Tools 注册表 ── 本次允许名单 ── 持久业务 binding

组合内核：声明依赖、执行接纳、子作用域、释放、图查询
安装控制器：制品、selection、应用与恢复
```

1. `get/require` 只读取声明依赖或自身提供的服务。`borrow` 在一次有界调用中选择可选 provider 并保护其寿命，借得的服务不能在 scope 外调用；方法返回的独立句柄遵守自身寿命合同。Root 的宿主装配保留全局读取权限。
2. ReplyProgram 在 apply 时捕获全部程序依赖；调用方 Context 只提供消息与状态写入授权。Conversation 的消息目录是启动依赖，模型目录只在校验显式选择时借用。来源提交使用同步 `SOURCE_CHANGED` 事件，保留输入与回复活动之间没有空闲窗口的语义。
3. `Context.entrypoint` 为明确的同步/异步 callable 建立 owner scope。事件回调和 `spawn` 的执行同样取得实际 owner 许可。后台任务先取消并等待，再排空调用和逆序释放资源。需要跨任务移交或跨 yield 保持资源的既有 scope 仍保留，不代理任意对象或安装全局 task factory。
4. Models、三个模型驱动、Akasha 与 Wake 的界面注册成为可选子 Fiber。名字只在同一父级内唯一；监控使用层级 path。计算资源、ModelsStore 路径和插件身份不变。
5. Wake 从 Tools 公共目录按 `investigation_tools` 选择当前可用调查工具，默认仍为 recall_memory、web_fetch。决策工具仍是本次必需能力。选择写入原 Request.tools；恢复使用原 binding，不用当前配置重新选择。缺席的可选调查工具不阻止新 Wake；恢复中的已选工具缺失或不兼容仍明确失败。
6. Akasha、Standard Web 和 Standard Tools 仅注册工具，删除无消费者的旧工具包服务。Reply 的工具搜索展示是临时可选能力。`tool-search.presentation.v2` 返回获授 view 和展示协议，替代 v1 的展示与独立工具包组合；没有搜索 provider 时使用原生展示。不保留旧 v1 适配层。
7. Tools、Models、Reply、Content、Context、Source、Turn、Delivery、Compaction、主动来源与客户端的共享 key/Protocol 统一声明；所有提供与消费方导入同一合同。ServiceKey 值类型不变型，异构图容器才使用 Any。R6 拒绝全仓同名重复声明和公共业务合同的裸 Any；R4 检查角色登记。能力目录由 `python scripts/plugin_boundary.py catalog` 从源码生成，不执行插件。
8. Manager 的 Fiber 枚举、依赖闭包和局部 readiness 查询由 CompositionRoot 拥有。换代前沿固定 provider 边捕获消费者，换代后沿当前声明边核对就绪，未改安装提交和失败恢复语义。

## 外部 Shell 插件的精确工具依赖

部署预检发现外部 `shell_restore`、`shell_safety` 仍消费旧 `STANDARD_TOOLS`。
工具注册现在同时由贡献者 Context 发布 `tool_key(name)`，值为该次真实 `ToolRef`。
消费者声明 `(TOOLS, tool_key("shell"))`，取得 Shell 后注册参数准备或授权行为。
工具缺席时消费者等待；工具释放或换代时，组合图先排空消费者，再释放工具。
不以 `ALL_TOOLS` 的瞬时快照代替就绪依赖，不恢复旧工具包服务。
服务发布失败时，Context 撤销服务，Tools 撤销目录注册；撤销失败保留原 Effect 的清理责任，
并同时报告发布与撤销错误。持久 binding 仍只保存原业务选择。

两个外部插件直接导入共享合同，删除各自的私有合同副本。此修复只改变工具发现与依赖，
不改变命令改写、拒绝规则或数据路径；外部历史测试引用已删除的工具包与旧夹具，
按维护者要求不改写，使用临时真实组合验证部署行为。

## 资源贡献者与宿主执行授权

业务插件只声明 MCP、进程或 Workload 门面；原始宿主执行端口由对应 provider 声明。
provider 为贡献者申请 grant 时，宿主核对同一 Root、插件/generation 身份与固定代码目录，
不要求贡献者重复声明 provider 的宿主依赖。Controller grant 复用同一执行身份校验。
这样资源所有者仍是贡献者，执行依赖仍由 provider 持有，两个角色不会被一次授权混为一谈。

## Turn 与宿主职责取舍

不新增万能 Turn 服务。Wake 的 Input/领取指针、Subagent 的父会话与容量、Scheduler 的触发记录/即时投递分支不是同一份状态。来源继续拥有原事务与结算，模型分支通过同一个 ReplyProgram 入口运行。把它们参数化为大量回调会增加第二套控制模型；SourceSession 继续服务普通会话来源。

Manager 保留安装、持久选择、实际 generation、执行授权与失败恢复。宿主装配移到
`agent.plugins.host`，只接收明确端口和实时只读事实；PluginUpdates 使用三方法安装端口，
不接收完整 Manager。消息数据权限仍由原 Core owner 授予，客户端投影保持 exact Root 和调用者权限，
没有新增一层业务 provider 或第二套状态。服务执行归属集中在 Context，纯值协议不代理所有方法。

这完成 Issue 766 的 D/F 设计取舍：复用实际模型执行入口，不强行统一不同来源的提交与结算；
移动真实装配职责，不为宿主权限套普通插件外壳。安装提交边界和外部客户端压力验收分别属于 #750/#661，
本 PR 不据此宣称它们完成。

## 旧入口清点

- 删除无生产调用者的 Root freeze/frozen、候选 incident 预算及其分支；保留现有诊断字段 `incident_overflowed=False`。
- 三个安装输入准备调用统一为 `_prepare_one`，删除只会拒绝旧发布路径的 activate/stage_stable 旗标。
- `update_rollback.py` 仍拥有离线安装回退的原指针和日志阶段；candidate 列是安装输入身份，非候选 Root，不改写历史表。
- ExecutionAccess/Controller 的 candidate 模式仍属于远端资源协议与残留清理；正式 Root 固定使用 formal。
- Manager 保留单 operation 和 draining owner 的串行管理，不引入并发更新。

## 任务与持久化合同

- change_type：refactor；semantic_delta：扩展合同收紧、可选能力缺席时仍可计算；默认安装下提示词、工具 schema、消息格式和业务恢复选择保持。
- capability_owner：组合内核拥有依赖和执行保护；普通插件拥有业务、工具选择与 UI。
- consumer_scope：Core 与内置插件；外部插件兼容性另行验证。
- runtime_patch：required；PLG-003/006/014 的 owner、排空和可选分支由内核保证。
- authoritative_state_owner：MessageLog、原插件数据 owner、PluginSelection 保持。
- client_only_alternative：客户端无法修复服务端 activation 和资源寿命。
- 唯一 writer：本任务 Codex；目标 main；基线 ad40a70d；独立 worktree。
- 允许副作用：源码、合同文档与隔离验证；按维护者明确授权提交、推送并创建 Draft PR；不合并、部署或发送外部业务消息。
- 恢复点：`/mnt/data/coding/issue766-backup-20260925/base-ad40a70d.tar`；正式 workspace 未操作。

| 对象 | 增加、更新、逻辑失效 | 物理减少与恢复 |
|---|---|---|
| Message | 原 MessageLog 只追加；不改 Input/正文格式 | 本任务无删除权；原撤销/会话删除协议保持 |
| 来源指针 | 原 owner 与 Input 同事务保存，按原终态推进 | 不迁移、不重编码、不删除，原记录继续可读 |
| ModelsStore | 原数据路径、连接选择和调用记录保持 | UI 卸载不关闭计算 owner，不减少记录 |
| binding/归档 | 原不可变业务选择与来源证据保持 | 不 GC；当前实现解释原选择，不兼容失败 |
| selection | 原安装控制器唯一提交 | 不自动回退，不把 accepted 当作 active |

取消与卸载继续保留清理失败的 owner。内存恢复不能代表外部效果回滚。

## 验证

既有 41 项概念测试通过；Core/tests 类型检查、边界、迁移、两组协议生成物与前端类型检查通过。
未新增或改写单元测试，夹具仅有获批的一行依赖声明。额外全插件类型扫描为 21 项既有错误，
同一基线扫描为 24 项；没有新增错误，不能描述成全插件零错误。
能力目录生成成功，R1/R2/R3 为 0，R6 未发现重复服务声明。

真实临时 workspace 启动 Models 与三个驱动，无 UI 时四个计算 Fiber 全部 ACTIVE。
带 UI 的组合十个 Fiber 全部 ACTIVE；卸载准确 path 为 ui 的 provider 后，模型对象和 activation Context
均保持同一实例，embedding 服务也保持同一实例；四个计算 Fiber 仍 ACTIVE，五个界面分支 PENDING。
新宿主装配的 runtime catalog 由真实 Models 子 Fiber 声明并读取成功。未调用外部模型，未写正式 workspace。

独立概念 reviewer：concept_review，gpt-5.6-terra / xhigh。最终审查 head、结论和 must-fix 处置记录于 PR。
旧批次的 Gate 不代替最终差异审查。正式发布、外部安装插件和真实客户端不在本地验证范围。
