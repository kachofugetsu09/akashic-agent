# 插件边界地基：防止新增耦合，为外置验收准备

- 状态：第 1、2 步保留；维护者已授权完整外置与正交化实施，按 stacked PR 交付。
- 更新：2026-09-12。
- 决策：[0065](../decisions/0065-plugin-boundary-checks-do-not-grant-core-ownership.md)，取代 [0064](../decisions/0064-plugin-boundary-is-machine-enforced.md) 的机械迁移路线。
- 长期约束：PLG-003、PLG-006、PLG-008、PLG-010、PLG-014、PLG-016、STA-001、CAP-001。

## 1. 目标与判断标准

新增一种业务只增加和装配插件；替换一种实现只调整装配；Core 不需要知道新增业务叫什么。
Core 保留插件基座、具有明确事实或机制 owner 的原子能力，以及这些能力自己的窄合同。
业务协作仍需要合同，但合同不因为被多个插件消费就进入 Core。

第一方和外部插件使用同一套入口。允许声明能力依赖，不允许 import 另一个插件实现。
包内相对导入保留同一 generation；默认装配不取得额外权限。

## 2. 已核对事实与局限

- `FreshPluginImporter` 为 artifact 建立独立模块身份并检查路径；它不是同进程 Python 安全沙箱。
  `plugins.<自身包>` 的绝对导入仍可能走仓库命名空间，必须作为欠账暴露。
- `ServiceKey` 按名称相等，只说明双方可以声明同名 key；它不证明协议相同、类型完整、
  生命周期正确，也不证明移动定义没有影响。移动仍须检查类型身份、导入副作用和真实消费者。
- `session/message.py` 原本只包含值类型及构造边界；#593 搬至
  `agent/plugin_contracts/message.py` 并保留同对象再导出。本次不扩大它的业务含义。
- `bootstrap/channels.py` 的渠道装配、`run_reply` 的跨插件组合以及宽泛 snapshot 访问仍待设计。
  即使 import 数量归零，这些语义耦合也不自动消失。
- 角色表只盘点 Core 文件中出现的字面 ServiceKey。它不是运行时注册表，
  不检查所有动态表达式，不证明能力归属或替换成功。

## 3. 本次允许与禁止的变化

#593 完善静态检查、测试与设计依据；#594 核实并删除原有 11 个遗留模块。
两层分别提交与验证，不合并 PR，不部署，不迁移正式 workspace。
#595 的现有路线停止；保留分支与提交作为参考，不以其剩余 import 数量继续推进。

持久状态增、改、逻辑失效、物理减少均无变化。Message、Session、memory、plugin-data、
迁移文件和外部效果协议全部保留。测试只创建一次性数据。
恢复点由原 PR commit 和任务开始前的 Git bundle 提供；代码回退不冒充运行数据回退。

## 4. 静态门究竟证明什么

| 规则 | 检查内容 | 不证明什么 |
|---|---|---|
| R1 | Core（含 host_bridge 和历史 migrations）不得新增对仓库插件的静态依赖 | 没有业务名称分支或硬编码装配 |
| R2 | 插件对 Core 的导入只能使用冻结的既有公开模块清单 | 清单内所有对象都已是窄能力 |
| R3 | 不得跨插件导入实现，也不得通过自身绝对路径绕过 generation | 运行时热重载、卸载和归档正确 |
| R4 | 字面 ServiceKey 与角色清单双向一致，角色值合法 | 角色语义、权限或提供者可替换 |
| R5 | 尚未实现的文档名称进入实现后，必须同步清理旧声明 | 所有文档都与实现一致 |

公开模块清单位于 `scripts/plugin_boundary.py::PLUGIN_ALLOWED_MODULES`。
它冻结兼容面，不给未来新建的 `plugin_composition/*` 自动放行。
增加公开模块、搬运业务实现、转导出全功能 store 都须单独说明 owner 和消费者权限，
不能仅凭 R2 通过接纳。值合同依赖测试另行禁止引入存储、网络及第三方实现库；
纯函数中的业务分支仍需概念评审，AST 无法证明职责正确。

扫描包括 `import`、`from package import module`、相对导入、星号导入和 TYPE_CHECKING 中的导入，
以及字面 `import_module` / `__import__`（含导入别名）。R4 包括位置与 `name=` 参数、
导入别名和小写声明，不追踪 `Key = ServiceKey` 这类赋值数据流。
同一文件对同一模块的重复导入只计一条。扫描仓库已跟踪的 Python 文件，新增文件先 stage。
计算式动态导入、反射、其他语言、外部插件仓库及 Python 执行沙箱不在本门覆盖范围；
新增动态加载入口必须在独立安装验收中展开，不能声称本门提供运行时隔离。

### 债务不能通过改账本消失

```bash
python scripts/plugin_boundary.py check --base origin/main
python scripts/plugin_boundary.py baseline
```

`check` 要求当前违规与账本完全一致；陈旧条目也失败。
`--base` 以**同一版检查器**扫描 Git 基线和当前源码，新增依赖即使写进账本仍失败。
因此修复扫描漏报可以补录基线原有事实，不等于允许新源码增加依赖。
比较基线必须是评审目标 commit，不能选择自己的 HEAD 来冒充无新增。
CI 对所有 PR（包括 stacked 分支）使用事件中的 base SHA。

`baseline` 只向标准输出打印待评审内容，不写文件。没有 `--base` 的本地 check
只证明账本一致，不能证明只减不增。检查器和清单自身变化仍需要可信评审，不能自证不可绕过。
历史迁移的 import 保留为精确债务；不提供整个目录永久豁免，也不修改已发布迁移来获得绿色。

## 5. 第 2 步删除边界

候选范围固定为 `agent/tools/` 中：
`forget_memory.py`、`memorize.py`、`message_lookup.py`、`message_push.py`、
`recall_memory.py`、`skill_loader.py`、`tool_search.py`、`vision.py`、
`web_fetch.py`、`web_search.py`、`shell.py`。

逐项核对静态消费者、相对与包入口导入、配置字符串、动态装载、安装插件消费者与兼容义务。
删除同时更新 Gate 目录登记和活文档链接，不扩大到 `agent/tools/` 整个目录。
`snapshot.tool_registry` 在生产中实际赋值；它及存活消费者继续保留。
`executor.py` / `events.py` 有合同和测试消费者，本次不删除，不预先指定未来迁移位置。

第 2 步已删除上述 11 个文件，并同步 Gate 路径登记及精确债务账本。
本仓库静态消费者为零；本地外部源码/cache 检查发现旧测试引用，未发现运行时代码引用。
旧测试同时依赖已退役 API，不能当作当前兼容验收；具体快照与保留理由见
[删除账本](../refactor/clean-code-ledger.md#2026-09-12-删除范围复核)。

已安装外部插件的消费者证据只适用于记录过的本地快照；未取得正式运行 fleet 证据时，
不能把全仓搜索写成“所有外部消费者为零”。发现真实外部依赖先处理兼容义务，不修改 cache。

## 6. 后续设计与装配

```text
┌──────────────────────────────────┐
│ Core：加载、作用域、生命周期、原子能力 │
└────────────────┬─────────────────┘
                 │ 同一套插件入口
        ┌────────┴────────┐
        ▼                 ▼
┌──────────────┐  ┌────────────────┐
│ 普通能力提供插件 │  │ 普通业务/装配插件 │
└──────┬───────┘  └───────┬────────┘
       └── 明确合同与依赖 ───┘
```

先选择一个完整业务切片验证通用边界，再迁移其他能力；不预先冻结所有业务 Protocol。
例如新内容的结构、模型解释和 UI 展示是不同操作，各自由对应插件协作拥有，
Core 不维护新内容名称的中央分支；没有解释器与解释器明确返回空必须可区分。
这只是候选验证案例，不授权本轮实现新 renderer 注册表或改变现有内容语义。

## 7. 验收标准

### 第 1、2 步的交付

1. 原来能绕过的包入口、跨包相对导入和自身绝对导入能被检测。
2. 新依赖写入账本仍被 `check --base` 拒绝；已还清债务不能继续留在账本。
3. 非法角色和别名声明受检；公开模块不能靠放入允许目录自动增加。
4. Message 两条路径保持同一对象，消息与 metadata 现有行为回归通过。
5. 第 2 步只删除有证据的模块，存活工具、迁移导入与 Gate 目录继续有效。
6. targeted tests、类型检查、change-impact Gate 与独立概念审查分别记录真实结果。

### 最终目标，尚未完成

| 验收 | 独立可观察证据 |
|---|---|
| 空载 | Core 制品不含业务插件源码；启动和插件管理可用，缺失能力明确报告，不假装能够聊天 |
| 外置 | 正式安装链加载外部插件；不借仓库 `plugins/` 或源码 PYTHONPATH 兜底；验证 apply 与真实能力调用 |
| 新增 | 新增业务及其内容解释只增加插件与装配；Core、无关消费者不改，不增加第一方特殊分支 |
| 替换 | 独立实现不委托原提供者，只改装配即可满足同一消费者的行为合同 |
| 生命周期 | 缺依赖、卸载、失败、取消与 generation 切换明确；旧任务/历史绑定继续使用原实现，原始事实保留 |

只要求依赖满足的合法组合能运行，不要求任意插件脱离所有依赖单独运行。
R1～R3 归零是结构条件之一，不是上述五项验收的替代品。

## 8. 停止条件

需要改变正式数据、持久语义、外部效果协议或既有插件生命周期时，转入独立设计与批准。
不得通过转导出实现、扩大目录豁免、修改受保护测试或仅修改角色标签制造架构完成。


## 9. 完整外置实施合同（2026-09-12）

维护者已授权实现、提交、推送和逐层 draft PR。主线程负责总设计、跨组接口与集成，
三个 Luna Max 副手在各自 worktree 执行独立任务。范围包含全部业务插件及内部支持包，
不以代表性插件代替完整清单。合并、部署和正式运行数据迁移不在本任务授权中。

- `change_type`: architecture；`semantic_delta`: 插件安装入口、业务扩展与包依赖改变。
  Message、工具与投递回执、generation/lease 及历史恢复语义保持不变。
- `capability_owner`: Core 拥有加载、作用域、生命周期与来源中立原子机制；
  普通插件拥有业务状态、算法、贡献接口及默认组合。
- `consumer_scope`: 所有第一方业务包，实际检测到的外部消费者及 Web/Mobile/控制端。
- `runtime_patch`: yes；必须从宿主移除业务实现导入及 checkout 兜底，客户端修改不能修复宿主依赖。
- `authoritative_state_owner`: 保持既有 Message、plugin-data、receipt 和归档 owner。
  本次只在一次性测试数据上运行；正常增加、允许更新、失效与物理减少协议均不改变。
- 恢复点：以 #594 的 `5436c3f6` 为实现基线；开始前保存 Git bundle 和主 checkout tracked diff。
  各 writer 修改前保存文件副本；已发布提交通过后继修复或 revert 恢复，禁止强推。

实施依次覆盖：全量清单和外置验收环境；真实材料组合切片；回复与工具收尾全链；
依赖顺序迁移剩余插件；Core/默认分发/历史升级依赖收口；累计替换与生命周期验收。
不同 writer 不同时修改公共接口，交接必须给出不可变提交及 dirty 状态。

完整完成要求：空 Core 制品可管理与安装；每个业务包从正式 artifact 加载并调用真实能力；
独立替代实现不存在原实现兜底；新增业务不改 Core；多贡献、缺失和冲突有明确语义；
新旧代请求不串代；历史绑定与未知效果保留原恢复约束；默认产品全链经过行为验证。
静态 import 门只是其中一个条件，未验证项目必须如实记录。

按维护者最新执行顺序，每层先做必要针对性检查并发布 draft PR，随即实施下一层。
全部实现后集中执行和修复 Gate、CI 与独立概念审查，不在每层驻留等待远端检查。
中间 PR 明确标记未完成的验证，不能据此声称可合并。

### 9.1 控制端扩展入口

RPC 的参数模型和处理函数由声明该方法的插件拥有。Core 只拥有 JSON-RPC 传输边界，
通过既有 ServiceKey/Context.provide 解析单个方法；不建立中央业务方法表或第二个注册系统。
插件不能覆盖宿主保留方法；重复 provider 沿用组合层的冲突处理。
解析、参数校验和调用持有同一 snapshot lease，连接不缓存业务插件参数表。
旧请求在原 generation 完成；新请求使用当前 provider。缺失方法返回 METHOD_NOT_FOUND。
当前 programmatic 方法名称、参数和调用语义不变；方法在插件实际启用期间可用。


### 9.2 独立分发制品

`scripts/build_plugin_distribution.py` 从明确 Git commit 构建 `core.tar` 和每个插件自己的
Git bundle。所有第一方插件均提供静态 manifest；没有 manifest 的入口会阻止分发。
Core 的路径清单不包含 `plugins/`，每个插件源只含自身子树及构建来源证明。
Git bundle 由原 `install_git_plugin` 安装，不给第一方插件增加 source loader 特权。
构建报告分别记录源码 commit/path、独立 Git revision 与产物 SHA256；它们不是同一个身份。
输出目录必须新建，失败保留已有内容，不覆盖旧发布制品。

这一步只证明包边界及安装输入，不能证明每个包已经可以独立 apply。
跨插件实现导入、宿主业务入口和默认组合仍需后续迁移；Core archive 尚不宣称是完整
可启动发布镜像。Docker、前端资源和历史升级的完整验收在累计收口中完成。


### 9.3 空载与客户端预览

生产默认只加载已安装源；开发 checkout 通过 `plugin_dirs` 或
`AKASHIC_EXTRA_PLUGIN_DIRS` 显式选择。空组合沿正常编译和发布路径形成真实 Root，
而不是让管理端遇到 None。该变化不把缺失业务能力当作已具备聊天能力。

回复插件内部继续保存短命 `ReplyActivity`。客户端订阅改用 `reply.status.v2`，
由插件直接提供序列化状态；控制 adapter 只声明自己的窄 `follow` 输入，不 import
reply 或 react 实现。客户端线上 JSON 字段不变；Python 服务 ABI 从内部 dataclass
变为已投影数据，因此明确升级 key。旧 generation 的预览仍在切换时撤下，
订阅不长期占用执行 lease，历史 Message 与回执不变。


### 9.4 回复收尾输入

`run_reply` 的工具收尾由调用者显式注入；标准工具插件提供 `tools.cleanup.v1`。
回复、调度、子任务、Wake 和候选验证各自在其消费位置声明窄 callable，不再由回复程序
导入 shell 实现。资源清理仍由标准工具原 owner 执行，取消和失败沿既有 finally 路径收尾。
来源插件只使用本地 SourceSession 结构，不导入 conversation 实现。
`MessageReader` 通过既有公开 messages 模块提供；它读取 Core 权威 Message 日志，
不授予任意 SQL、删除或外部发送能力，也不包含业务投影。


### 9.5 消息展示由内容 owner 提供

客户端页面通过 `message.display:<kind>` 消费该内容的只读展示函数。内容插件分别
提供自己拥有的 kind，不增加 Core 业务列表；同一 key 的冲突由已有组合层拒绝。
插件选择可以公开的字段，未知内容明确标记 unavailable，不把内部 continuation 直传客户端。
工具名称沿用工具插件的只读 binding 描述能力，不重开工具或按当前名称重新绑定。

Web、Mobile 和控制端每次页面或正文投影在同一个 snapshot lease 内解析并调用，
完成后释放；长连接、下载票据和频道对象不缓存业务 provider。新一页使用当前代，
单页不串代。历史事实与下载摘要检查保持原约束；提供者变化导致表示不匹配时仍明确拒绝，
不放宽摘要条件。展示测试覆盖新 kind、提供者替换、提供者移除和每页 lease 释放。


### 9.6 独立出站 provider

Telegram、QQ 和 Akashic sender 在自己的包中声明注册输入及发送结果，不导入
Delivery 实现。Delivery 在 sender 边界严格校验结果为其自身回执；字段、状态和
provider IDs 的持久语义不变，未知效果不变成可重试的成功。实际网络调用及全部附件
预读仍由 sender 执行；只读日志和 artifact 能力不扩大权限。

最终 Output 的等待只读取来源与结束消息 ID，不依赖 turn_projection 类身份。
注册表、发送记录、原 binding、lease 与恢复仍归原 owner；本层没有迁移正式数据。
公开 Message 输入沿用已有 Core 值合同，不新建业务 schema 中心。


### 9.7 调度检查由 scheduler 拥有

调度列表、排序、启用过滤和详情文档由 `scheduler.inspection.v1` 的普通插件 provider
生成。Core 不打开 schedules.json，不 import JobStore 或 ScheduledJob。真实 store
与调度执行共用原 owner 的实例，不增加并行存储。每次 Web/Mobile 查询在一个 snapshot
lease 内完成；缺 provider 明确返回 scheduler_unavailable，空列表只表示查询成功且无任务。
本层只读取调度事实，不创建、改写、失效或减少计划。


### 9.8 Core 消息原子输入

公开 `agent.plugin_contracts` 保留 Core 自己的 Message 值、冻结 JSON 和当前 body
表示，不接纳模型、调度、投递或材料 schema。`json_value` 和 `body_to_dict` 与这些
现有值同属一个 owner；历史持久编码与解码仍由 session 存储层拥有，旧表示不改写。

既有 messages 能力模块显式公开其已经签发的 reader、writer、catalog、固定 owner
记录与事务类型，以及冲突和分页错误。它们不提供 MessageLog、SQL、任意 owner
选择或删除；writer 的授权、事务 CAS 与追加检查保持原路径。状态依旧由原插件
保存自身记录，只允许原 CAS 更新；Message 正常只追加，没有新增物理减少路径。

消费者可依赖这一组来源中立原子输入，不必为了类型注解导入 session 私有实现。
这是公开已有 Core 原子合同，不是把业务对象搬入共享目录；业务协作继续使用局部输入。


### 9.9 材料贡献使用结构输入

`context.materials.v2` 的贡献者返回 system_prompt、reminders、summary、references
结构映射；Context 边界校验并转成其内部材料。未知字段、错误类型、未授权 Prompt/摘要、
重复引用证据冲突和跨 Root 注册均明确失败。没有旧 dataclass provider 的兼容分支；
旧 key 不与新 ABI 混用。Prompt、Markdown、Akasha 和 compaction 分别声明局部注册输入。

来源事实、召回查询记录、摘要来源范围与权限保持原 owner；召回工具仍从已保存查询
重建同一呈现出处。本层不减少 Message、不重写旧摘要或学习数据。
`ContextBuilder.build_attempt` 以容量拒绝结果承接外部消费者，不要求 import 内部异常类。
材料消费端及 Content 引用解释的剩余实现依赖继续在后续层迁移；本层不宣称 Context
已可完全脱离 Content 独立运行。


### 9.10 来源接纳与控制

`sources.v2` 接受来源局部声明的 open、needs_reply、accept 和 channels，注册随
同一 effect 生灭。回复跟随器消费来源自己的待回复判断，不 import conversation。
普通会话与 programmatic 通过 `source.session.v1` 取得来源控制实例；来源插件拥有
接纳、暂停、恢复与 Task 准入算法，conversation 只保留普通会话的输入和模型选择。

Message 追加、幂等接纳、撤权与物理排空保持原顺序；没有第二份任务或会话状态。
Core tasks 公开已有重启准入原子，不拥有业务回复策略。最终输出等待的局部输入
包含消息前缀 ID，保证 programmatic 仍能等待原连接的真实 writer flush。


### 9.11 工具 provider 与回执生命周期

standard_tools 与 standard_web 使用局部注册输入和结果值，tools 在实际执行边界
校验结果。内部相对 import 允许归档包持有自己的实现；首次返回与持久回执重读比较
结果值，不要求跨代 Python 类身份。工具 ID、原 binding、查询结果和未知效果语义不变。
Skills 同时改为材料 v2 结构输入，关闭上一层留下的旧材料贡献 ABI。

后台 abandon 监听只在实际结算一条回执时打开 Content 与 runtime scope；提交完成或
异常后释放 writer 和租约。空闲监听不固定业务 provider，不阻止新代替换旧代。


### 9.12 模型解释由 models 提供

models 分别提供本次投影构造、消息事实校验、模型正文与附件读取、选择解释能力。
普通会话、Wake 与回复程序声明自身需要的窄输入，不 import models 实现。
模型投影接收 Context 的摘要校验函数，models 不再反向 import Context。
提供者直接发布原实现，不增加第二份模型请求、事实或存储。

定向程序回归确认这些能力在真实组合内调用。外部整组加载仍被其余跨插件依赖阻断；
工具参数拒绝曾依赖跨包异常类身份，材料消费者仍有旧类型注解，均在紧接的层继续迁移。
这些已知缺口不计作独立组合通过。


### 9.13 回复资源组合与跨归档输入

普通 `reply_program` 插件提供 `reply.execute.v1`，统一持有一次回复的模型、内容、材料、
工具菜单和 writer；默认回复、Wake、scheduler、subagent 与候选验证只保留来源策略。
`sources` 提供 `source.check.v1`，命令和回复在效果开始前使用同一输入撤权检查。
工具 owner 提供菜单与回执工厂；拒绝使用显式结果，不依赖另一归档的异常类身份。

Content v2、Context v2、材料 v3 与 react v2 接受局部结构输入。Content 接纳声明映射并
在 owner 内验证，历史用户身份与学习资格仍由其解释。Context 容量不足返回明确拒绝原因，
react 保留已有摘要直到 reducer 返回新摘要，不能因“没有新摘要”撤回已有覆盖区间。

models 拥有会话选择的读写；Core 每次查询通过当前 snapshot lease 取得模型 owner。
Core-only 缺少模型能力时返回 unavailable，不能以空目录冒充成功。投递和工具搜索消费者
使用局部结构，原 binding、不可确认发送与恢复查询语义继续保留。

本层仍是分阶段实施：记忆插件局部边界中的摘要区间与结算前缀重复算法需在下一层归还
唯一 owner；其余渠道与 Core 迁移导入尚未收口。当前结果不代表完整外部组合验收。


### 9.14 来源策略与原效果选择

scheduler、subagent、plugin_update 与 Computer 消费各自实际需要的局部结构，不导入
Tools、Delivery、Conversation 或 Turn projection 实现。Tools 提供原 binding 的配置派生，
调度只提交目的地映射；子任务与更新请求保存自己的输入记录，发送时由 Delivery 校验。
原任务 ID、工具目录、发送 binding、未知回执及内部学习排除语义保持原样。

`delivery.input-origin.v1` 发布原策略的已接纳输入读取；`tools.bind-saved.v1` 发布原归档
派生操作；工具调用前缀暴露既有 effect key，restart 不自行重造该身份。
Tools 的 abandon 监听明确声明 Content 依赖，缺少 owner 在装配时失败，不能延迟到停止时。

测试比较跨归档结果的 outcome 与 parts，不要求两次加载拥有同一个 Python class。
Markdown 只读材料不应强制依赖摘要学习能力；该合法组合的现有回归仍需随下一层记忆
owner 修复关闭。Core 历史升级与通道入口、Wake 和完整外部分发验收继续实施。


### 9.15 Wake 局部能力与预算失败

Wake 只提交目的地记录、固定工具引用和已校验内容，兴趣、历史读取与投递能力均由
对应 owner 提供。Wake 不构造其它插件的 ToolView，也不 import 它们的结果与异常类。
模型请求预算耗尽属于不可重试的 ModelError；react 保留自身 StepLimit 类和原错误说明，
Wake 使用同一失败通道保存 Control 并结算原职责。完整请求预算测试确认无发送、不自动
重试且原职责只关闭一次。删除只比较字面字段或重复静态扫描的临时测试，保留真实链路断言。


### 9.16 公开原子与执行租约

Message 编解码实现归入现有公开 Message 值模块；Core 旧路径仅保留兼容引用，
编码字节、旧 unknown 重放和新消息拒绝规则不变。附件值、timer 回执和 restart
拒绝是已有 Core 原子的公开输入输出，不增加业务模型或另一份状态。

react 通过 Context 的公开 capture_runtime_scope 取得真实调用租约，不再读取宿主
snapshot 私有全局。已提交调用的取消、排空和 generation 归属保留原路径。
restart 回归组合安装真正的 standard_tools 清理 owner，并明确授予其 skills 材料权；
测试在 reply.execute 的注入点协调清理与发送，验证关闭完成前不能提交 restart。


### 9.17 摘要范围的唯一 owner

Context 提供已有摘要校验、连续来源范围和已结算前缀算法；Compaction 与 Markdown
通过回调消费，删除局部边界中复制的算法。SummaryRecords 仍在原事务中校验来源，
只追加摘要并推进原指针；权威消息正文没有写入或减少。

Markdown 只读材料组合不要求安装 Compaction；真正遇到需要学习的摘要时必须取得
原摘要读取能力，缺失明确失败，不跳过或伪造来源。Akasha 使用公开只读 embedding
接口与实际 Message 编码，重建写入器的权限归属继续单独处理。


### 9.18 Dashboard 的请求能力入口

Dashboard 模块通过自己的 inject 声明只读查询能力；async 路由使用 DashboardContext.require
在实际请求租约中取得该能力。Core 不保存业务 key 清单。未声明、没有请求作用域或
请求 generation 与页面不符均明确拒绝。页面不能取得宿主 Root、任意 SQL 或 writer。
Akasha、Wake 与工作台使用这一入口；真实 HTTP、持久事实和编译页面内容共同验证。

### 9.19 参考系统与有意的取舍

| 参考 | 采用的机制 | Akashic 的取舍 |
|---|---|---|
| 本地 deepseek-harness 的 docs/architecture.md | Core 上安装普通能力插件，profile/bundle 明确组合，可撤销注册随卸载释放 | 分发与组合不能成为 Core 内另一份业务装配表；每个包仍走正式安装链 |
| [JupyterLab 扩展](https://jupyterlab.readthedocs.io/en/stable/extension/extension_dev.html) | 提供/需要能力决定激活顺序，缺失必需能力明确失败，默认产品也由扩展组成 | 不要求提供者与消费者 import 同一业务 token 包；本地结构与有版本的名字组成 ABI，避免对象身份与包去重要求 |
| [Backstage 扩展点](https://backstage.io/docs/backend-system/architecture/extension-points/) | 插件拥有自己的小扩展面，演进时使用新名称，失败归属具体 owner | 不建立 Core 业务接口目录；普通插件可声明局部输入，而不是共享所有业务 Protocol |
| [VS Code 贡献点](https://code.visualstudio.com/api/references/contribution-points) | 包元数据声明 UI 与命令贡献，由宿主接纳 | 保留通用接纳与生命周期；业务展示和业务操作继续由外部插件拥有 |

这些系统都需要协作合同。这里去中心化的是业务语义的所有权与发布路径，而不是消除
可检验的协作语义。允许依赖公开能力，不允许通过 Python class 身份、兄弟源码或
宿主私有对象偷渡实现依赖。真正的成功还必须由独立提供者替换和 checkout 缺席验收证明。


### 9.20 普通渠道 owner 与可恢复配置迁移

Telegram 与 QQ 入站适配器、协议过滤和发送实现移到各自可安装包；Core bootstrap
只启动已声明的渠道。静态 channel_credentials 是凭据权限上限，配置可以不注册渠道；
未注册时没有 provider client，通用凭据读取不会因此获权。实际注册仍须精确匹配声明。

QQ 必须等 SDK 真实 startup 和 API 后才报告 ready；SDK 把事件回调放进短命任务，
因此插件在真实 connect_websocket 与 start 入口记录连接任务和线程。停止在 provider
loop 卸载资源、取消长连接并 join 实际线程，不能取消 to_thread 等待者就声称退出。
重复停止共享实际清理结果。Telegram 保留多行转换、被回复附件和带 bot 名的 stop 命令。

配置迁移显式指定 marketplace，先保存原配置，再发布全部目标，最后去掉旧表；
故障后同内容允许续做，异内容拒绝覆盖。测试注入第二目标发布失败，确认原配置与
备份完整，并能续做至外部实例数据目录。脚本没有在正式 workspace 执行。

本层不把安装默认禁用渠道等同于真实联网验收。QQ 的 provider 超时配置和完整外部
入站→Message→回复→回执组合继续核对；原 Telegram/QQ 旧 bus live handlers 在当前
源码没有事件生产者，不据此声明恢复了 live preview。

### 9.21 源码不可见的整组安装与启动证据

验收 runner 从固定 revision 的 Core tar 和独立 Git bundle 开始，使用正式安装链，
从 import 路径移除所有 checkout 与 editable 路径，并记录实际模块来源。当前 40 个
包均安装、apply 并进入 generation；Core 空根和整组 AppRuntime 均实际启动、退出，
退出时检查后台任务、socket 和 snapshot 释放。

能力枚举不计为调用。runner 对选定的已发布服务执行带输入与预期输出的调用；当前
已验证 models 的 Message 展示事实转换。测试 workload controller 只提供本地协议
夹具，不能据此宣称真实容器、渠道发送或所有业务组合通过。

这批证据固定在 1d4ede3af3e7e412d615d7f074508737def9f837 构建的产物，Core tar
SHA-256 为 0ae58a6601e30077a95dd4a721dc5398286298a84f840b14a8fa0cf1da8307a0。
后续仍须覆盖真实 CLI 迁移入口、合法子集、原 provider 不在场的替换及最终业务结果。
历史 Yoyo Message 元数据迁移仍引用旧 codec 的私有解析 helper；保留指向同一实现
的 Core 内部别名，迁移包拆分前不能因移动公开 codec 破坏旧数据入口。

按用户指定顺序，逐层发布 draft PR 并继续实施，全部实现后统一完成 Gate、CI 与
概念评审；本节的本地证据不代表这些检查已通过。

### 9.22 进程、文件与归档实际 owner 的公开入口

进程组生命周期、主机文件访问、不可变插件归档和工具 schema 校验的实际实现移到
明确的宿主模块；业务工具只消费这些能力，shell 命令安全策略归 standard_tools。
Models 执行通过 Context 捕获同一 runtime scope，保留嵌套执行、子 task 拒绝和
driver 关闭语义。RuntimeScope 的身份直接来自真实 lease，不以 getattr 猜测内部结构。

技能读取端口每次调用读取已发布 snapshot 的安装目录，不能在 apply 时捕获旧目录。
绑定先归档原资源，恢复只打开原归档；安装目录不是任意 workspace 技能扫描授权。
现有 Core skill host 仍拥有解析与目录生命周期，这项来源特定职责尚须在最终原子性
审查中收敛；公开入口本身不作为完成去中心化的证明。

进程、文件、归档等定向测试 124 项通过、1 项平台跳过；发现的旧材料 ABI 与默认
加载 fixture 已修正，随后 Prompt 与真实模型执行 31 项通过。定向类型检查 0 errors。
当前静态债务 R1=29、R2=18、R3=0；Gate、CI 与累计概念审查留在实施完成后。

### 9.23 人格与初始化的业务 owner

Prompt 的完整模板、读取校验、默认恢复和原始字节备份由同一个安装包拥有。
Core 不再提供 veda-reset，也不在启动前读取 VEDA；用户显式运行安装包的
`persona.py --workspace PATH` 维护入口。正常读取仍对缺失、空白、损坏报错，不能
借插件加载自动重置人格。测试实际执行外置复制包的维护命令，验证非法 UTF-8
原文备份、默认结果与重复执行不重写。

Core init 只准备配置和空 workspace；不再写 VEDA、Context 默认授权、meme 清单
或业务目录，已有文件即便 --force 也不会被这些已移出的动作改写。新安装组合须
由产品配置与各包维护入口完成业务初始化；这是显式行为变化，不保留隐式 builtin
产品组装作为兼容 fallback。

Markdown JSON 解析属于自身；日期、原子文件写入、Turn effect 持久编码和 Message
embedding writer 使用实际 owner 的精确公共入口。没有另建业务接口全集，writer
没有被伪装成只读检索端口。Prompt、Core 初始化及公开合同相关 63 项测试通过，
定向 pyright 0 errors；静态边界 R1=29、R2=6、R3=0。最终 Gate 与 CI 尚未运行。

### 9.24 控制连接与页面协议的实际公共原子

控制连接 drain 路由、结束 claim 与取消回收的同一实现位于公开 control_frames；
Core 连接与普通插件共用实际 owner，没有另建投递账本。programmatic 的 RPC 参数
在本包校验，通用 RpcMethod 接受 Pydantic BaseModel，不要求继承宿主业务参数类。

Web、Mobile、控制 API 与 Workbench 共用 Message 页面协议的纯投影实现；公开
message_view 不读取业务配置或解析模型内容，特定 ContentPart 展示仍由已发布
回调提供。旧 Core 路径只是同一实现的内部兼容入口，没有业务插件 reexport。

程序请求、结果、MessagePush、页面、元数据与重启定向 58 项测试通过，RPC 输入
调整后再验证 9 项通过；定向 pyright 0 errors。静态 R1=29、R2=0、R3=0，剩余
Core 业务 import 是历史迁移链。静态清零不代替外部替换与概念审查。

### 9.25 跨代 SDK 共享资源与可完成的清理重试

QQ 的 SDK 连接函数按实例绑定实际 websocket 超时；不支持的 SDK 入口形状明确报错，
不因测试 provider 缺少真实入口而静默忽略配置。NcatBot 配置在占用前快照、资源
关闭后恢复。排他 claim 位于宿主进程原子，不能使用每代独立插件模块各自的锁。
Core 只管理任意命名资源的排他事实，不认识 NcatBot 名称或 SDK 配置。

SDK 的空插件发现目录使用本代独有临时树，停止后清理；不再扫描或创建全局 HOME
下的 SDK 插件目录。原正式目录没有被迁移或删除。SDK unload 失败时保留连接和
进程 claim，重试成功后才能退出线程、清理目录并交给新代。

启动失败照常传播；stop 回执只回答实际资源是否关闭，不把已传播的旧错误永久
当作未完成资源。Telegram 复用同一条清理路径，失败保留资源句柄；取消某个 stop
等待者不取消共享清理。测试以 FreshPluginImporter 加载两份实际插件 namespace，
证明它们不能同时改共享 SDK 配置，旧代关闭后新代可进入。渠道 9 项和宿主相关
50 项测试通过，定向 pyright 0 errors；完整 Gate 与 CI 仍按用户顺序最后处理。

### 9.26 历史迁移外置的当前证据与未完成项

48 个历史步骤及其冻结 helper 已进入普通 legacy_upgrade artifact，Core 保留起点
和 ID/依赖索引；安装时验证 catalog 与全部包文件摘要，不从 checkout 猜测实现。
在独立临时 workspace 显式加载 bundle，49 个步骤实际完成，第二次执行返回 current。
原步骤 ID 与共用 Yoyo ledger 保留；这次验证未使用 monkeypatch 补齐 codec。

尚未完成：新空 workspace 不安装历史业务包时的真实 CLI 启动、历史 helper 的
Core 残留、迁移包独立导入身份及精确公共依赖。初次集成新增的迁移专用 R2 尚未
清零，不能把此前常规插件的静态结果套用于这批新增源码。

### 9.27 分开的组合、替代和旧 lease 实验

验收分别在两个独立安装根运行 content+Message 消费者，以及 content+另一包名
provider+原消费者；第二组没有原 provider 安装，也没有更改消费者源码来识别
替代包名字。实际经正式 installer、PluginManager 和 generation lease 写入 Message，
再由独立 reader 回读最终文本。

同名热更新另做一组实验：观察真实 snapshot drain 等待，旧 lease 未释放时 publication
不得结束，旧消费者仍读原结果；释放后新 snapshot 消费者读替代结果。两类证据不
混称。源码暂时移至命名恢复点，退出后恢复原路径；验收不删除调用者的源码。

固定 00cfea41 的 Core tar 外置实跑，异名 provider 的结果为
`independent:same-consumer`，Core 摘要不变且 Core-only AppRuntime 启停通过。
该结果不代表 CLI 历史迁移已经闭环。8 项定向测试通过；未提供 Core 制品或未请求
热更新时，报告不再为未执行的验证生成通过标记。完整 Gate、CI 与概念审查待实施
完成后统一运行。

### 9.28 声明资产与 Skill 解析的归属

Core 删除 SkillIndex、SkillsLoader 和技能 frontmatter 解析，只发布当前任务租约固定的
InstalledAsset(owner_id, category, root_dir)。插件使用 asset_roots 声明目录类别；旧
skill_roots 在加载边界转换，不能同时使用两种声明。standard_tools 拥有技能解析、
可用性、提示、读取工具与检查投影。工具绑定仍单独归档原始资源树，恢复时不补读
当前安装。Core 的候选状态删除没有真实数据来源的空技能字段与计数。

解析缓存每次先读取当前租约的资产，再按实际资产目录复用；不得绕过任务边界或
缓存已经释放的旧目录。73 项定向回归通过，包含缓存命中后跨任务及释放租约拒绝、
原安装消失后的工具归档恢复。类型检查零错误。Prompt 测试直接通过 MessageLog
初始化新库，历史升级路径由迁移测试单独验证。

未完成的归属：Core 启停与发布仍有旧 Skill 软链接投影，Mobile 检查仍消费专用技能
服务；这两项仍需继续收口。此层不宣称所有 Skill 业务已经离开 Core。

### 9.29 正式安装的回复与送达闭环

16 个真实业务包分别走正式 Git 安装链，原安装源码目录移至命名恢复点。运行时
逐包核对实际模块来自该 generation 的归档树；不替换 Models、reply_program、
delivery 或 TelegramSender。用本地 HTTP 服务实现 OpenAI 与 Telegram 外部协议，
通过实际模型配置入口和 CHANNEL_INPUT 接纳一条输入。

验收得到一条完整回复和一次发送请求，再通过实际最终 Output 等待能力取得
delivered 回执。关闭所有 runtime 后重新打开消息库，原 Input/Output 字节语义未变，
送达记录仍保留 provider ID 731。该测试通过；不声称测试了真实 Telegram 网络、
Bot 收件或 checkout 完全不可见的进程（后二者有各自独立验收）。

### 9.30 资产的唯一生命周期与旧投影退役

删除 Core SkillLinker、启动/发布/回滚自动软链接路径，doctor 只检查任意类别的
声明资产目录，不再要求 workspace skill 链接，也不从 checkout 补齐未安装的插件。
旧模块字段只在读取模块时转换；ComposablePlugin、PluginContributions 和
ActivePluginInfo 不再保存重复的技能专用字段。

候选升为正式版本的实际测试暴露原快照资产 ID 漏接。修复采用一份 generation
资产树：快照从自己固定的 generation 集合派生目录，不再复制第二份拓扑目录或
保存独立目录 ID。历史归档重开也准备同一生命周期的资产；关闭失败保留 owner，
重试完成后才注销。持久 SkillTool 的原绑定文件树协议保持不变。

原 workspace 用户文件、同名目录、旧软链接和 ownership journal 都不改变；同名
用户目录不再阻止插件升级。旧恢复记录若仍依赖已删除投影 owner，启动与显式
retry 均在外部效果之前阻断，journal 保持 pending。没有正式数据迁移。

68 项资产/候选/归档/外置组合回归通过；补查清理重试与运行入口的 88 项测试通过，
定向类型检查零错误。完整 Gate 与 CI 仍在所有实施层完成后统一执行。
