# 插件 Onboarding 投影设计

- 状态：内置实现、运行验收与独立评审完成。维护者于 2026-09-28 授权完成内置插件、CDP 场景、独立评审与 PR；不包含生产部署。
- 核对基线：d99dd2b4b4ae2fdb146d9265554262cb39fdcdb5。
- 关联需求：ONB-001、PLG-003、PLG-006、PLG-010、PLG-014、PLG-016、STA-001～STA-003、ERR-001。
- 关联设计：[能力与执行归属](issue-766-orthogonal-capabilities.md)、[单图插件系统](issue-750-plugin-publication-simplification.md)、[模型配置](runtime-model-registry-and-onboarding.md)、[WebUI 组合](web-ui-plugin-composition.md)、[持久化状态地图](persistence-state-map.md)。

本设计将模型、渠道、Akasha 与 Wake 的初始配置组合为普通插件页面。
业务插件拥有开关、配置、校验与运行状态；onboarding 只拥有注册、排序和展示；
Core 提供真实依赖图、正式配置应用与无配置启动能力。
实现位置、关键决断与验证边界见 §11；旧入口清单保留为迁移对账。

## 1. 最小范围

| 配置区域 | 参与插件 | 本次操作 |
|---|---|---|
| 模型连接 | models、codex、opencode_go、openai_compatible | 选择连接方式，校验连接并选默认模型；按需配置向量模型 |
| 渠道 | channels、telegram_channel、telegram_sender、qq_channel、qq_sender | 各自开启或关闭；开启时配置凭据、接收范围或发送连接 |
| Akasha 情景记忆 | akasha | 开启或关闭；开启需要可用向量模型 |
| 主动联系 | wake | 前置满足时开启或关闭，选择发送目标；前置关闭时只允许下一步 |

当前内置源码没有独立 proactive 插件；本次使用 wake，不新建第二套主动流程。
模型驱动是连接方式，不要求逐个登录；不为 models 增加与连接库重复的总开关。
无模型时离开页面不算完成或跳过，Chat 保持真实空状态。

Computer、手机配对、Markdown 记忆、Scheduler、人格编辑不新增引导步骤。
Akasha 只控制自己的学习、召回与兴趣能力，文案必须写“Akasha 情景记忆”，不能声称关闭所有记忆。
Markdown 记忆保持原行为；prompt 只接住旧 setup 中创建缺失人格文件的工作。
基础插件保留在图里，只有需要用户配置的项显示为步骤。

不做通用表单 schema、跨插件一键事务、引导进度数据库、首次成功 Turn 弹窗、
后台新插件通知、热更新专用向导或协作编辑。打开页面、提交后和下一步时重读即可。
关闭和卸载均不删除会话、记忆、凭据、调度或主动流程恢复记录。

## 2. 实现前的基线事实与差距

| 已核对事实 | 代码入口 | 需要补齐 |
|---|---|---|
| Core 不解释 Config，准备 generation 时读取并归档固定输入 | agent/plugin_composition/config_input.py、agent/plugins/input_preparation.py | 不能把写文件当成运行已采用 |
| 模型使用自己的库、revision 和连接表单扩展点 | plugins/models/plugin.py、plugins/codex/plugin.py | 直接复用，不搬入引导状态 |
| Telegram/QQ 的 channel、sender 各有 enabled 与独立配置 | 对应 plugin.py、config.py、configure.py | 迁移 Web 表单，不发明彼此的硬依赖 |
| Akasha、Wake 没有本次要求的完整开关 | plugins/akasha/plugin.py、plugins/wake/api.py | 实现真实停用，不只隐藏 UI |
| Wake 依赖 SEMANTIC_INTEREST，具体 sender 到执行时才按名称绑定 | plugins/wake/plugin.py、plugins/wake/runtime.py | 补具名发送能力与实际选择依赖 |
| UI 已能成为可选子 Fiber | docs/design/issue-766-orthogonal-capabilities.md | 沿用相同方式接 onboarding |
| runtime catalog 未完整公开 provider 边 | agent/plugin_composition/runtime_catalog.py、context.py | 补窄只读图查询 |
| PluginUpdates 只公开安装，缺少仅改自身配置的端口 | agent/plugin_composition/plugin_updates.py | 明确新增宿主配置应用边界 |
| supervisor 在 config.toml 缺失时退出 | agent/supervisor.py | 补齐 ONB-001 无配置 Web 存活 |

[模型配置设计](runtime-model-registry-and-onboarding.md) §7.1 的固定连续步骤与新模型有差异；
其首跑流程由本稿替换；模型目录、连接、revision 与运行绑定仍归 models。

## 3. 职责与 Core 增量

### 3.1 Core 只补三个真实边界

1. **只读组合图。** 从现有 runtime catalog 提供节点、父子 Fiber、声明依赖、provider 归属、
   缺失原因；包含已挂载但等待依赖的功能分支。调用者拿不到服务对象、Root、SQL 或安装器。
   图归 Core，排序与界面归 onboarding。
2. **自身配置应用。** 受当前 Context 限制，只能读自身输入版本、提交自身配置、读自身回执。
   宿主复用已选制品、固定输入归档、selection 提交、局部排空和既有恢复 owner。
   不授予任意 plugin_id、代码来源、卸载能力或完整 Manager。
3. **无配置启动。** Supervisor 以合法最小宿主配置启动同一套普通插件组合，保持 2236 壳层。
   业务缺配置不能关掉设置入口。默认插件来自发行安装清单；Core 不识别 models/wake/onboarding
   名称，不在用户卸载 onboarding 后强行补装它。

不增加 Core 向导管理器、业务开关表或业务字段 schema。公共合同位于 agent/plugin_contracts，
provider 与策略仍在普通插件。配置应用涉及正式输入提交和调用排空，无法只由前端完成；
真实图的提供者归属也不能由前端扫描源码猜测。这是两项 runtime patch 的归属依据。

### 3.2 onboarding 是普通注册表、投影和页面

| 部分 | 职责 |
|---|---|
| plugin.py / 注册表 | 提供 ONBOARDING；收集 group、step；贡献者 Effect 持有注册 |
| projection.py | 从图排序并合并 owner 状态；不写业务数据 |
| dashboard 与 Web 源码 | 返回视图，显示表单和下一步；复用 UI.register、模块摘要与同源路由 |

必需依赖只允许通用 UI、只读图等能力。onboarding 不依赖 models、channels、Akasha、Wake、
任一 sender、Message writer 或安装器；不维护数据库。页面位置只是临时 UI 状态。

注册最小信息：稳定 key、group、显示文案、对应功能 Fiber 引用、只读状态入口、既有表单模块/路由。
owner 由贡献 Context 确定，不能自报其他身份；不另填 depends_on、priority 或完成 flag。
状态读取以贡献者的受保护入口运行，步骤保存调用其自有 API，不把 writer 交给引导。
目录先返回注册元数据，各项状态分别读取。业务探测的已知失败由 owner 返回 fault/unknown；
贡献者换代或卸载造成的入口失效只标记该项不可用并重读目录。其他项照常显示。
不在投影外层用宽泛异常把程序错误转换成“完成”或空目录；契约错误保留 owner 和错误诊断。

### 3.3 设置入口独立于运行条件

~~~text
功能插件常驻入口
├─ 自有配置与状态
├─ 独立设置页（可选 UI 子 Fiber）
├─ 引导贡献（可选 ONBOARDING 子 Fiber）
└─ 功能运行分支（真实业务依赖）
~~~

通过 ctx.inject 接入引导，不在业务主 apply 中硬 require ONBOARDING。
配置/引导分支不能放在 enabled 判断或全部业务依赖之后，否则关闭后无法再次配置。

关闭功能仍保留配置、运行分支的声明和稳定业务接口。例如 Akasha 兴趣接口明确报告
disabled/unavailable；实际运算仅在运行分支就绪时通过该分支的受保护入口执行。
未启用的执行调用必须明确失败，不返回假成功。这让冷启动也能从同一图找到 provider owner，
不依赖旧活跃图缓存、静态扫描或另一张手填依赖表。

同一个业务可用性检查用于运行入口和设置页。Wake 启动及接纳新职责前核对模型、兴趣能力
与已选目标；onboarding 只显示该结果。上游配置换代仍由 Core 处理实际调用与资源寿命，
再按新状态决定后续工作。设置模块不接管发送、记忆或主动流程事务。

各插件的具体拆分如下；功能分支即使关闭也保留声明，回调只有 enabled=true 才启动工作：

| 插件 | 常驻入口 | 功能分支与前置 |
|---|---|---|
| models / 驱动 | 保留现有模型目录、连接与状态接口 | 连接可用性沿现有实现；不为引导再造执行接口 |
| Telegram/QQ channel | 自身配置、状态、可选设置与引导 | 将现有 CHANNELS、CHANNEL_INPUT、来源会话等业务依赖移入 child；仅开启时注册真实 channel |
| Telegram/QQ sender | 自身配置、状态、可选设置与引导；向 delivery 可选贡献候选 | 依赖原发送目录与宿主凭据/消息/附件接口；仅开启时注册实际 sender 与具名能力 |
| Akasha | 自身配置/状态；稳定的 SEMANTIC_INTEREST 接口 | 原模型向量、消息投影、材料、工具等依赖移入 child；学习、召回、兴趣运算和资源归 child |
| Wake | 自身配置/状态、可选设置与引导 | 原 models、reply、delivery、eventmail、drift、SEMANTIC_INTEREST 等依赖移入 child；选定后再加入具名 sender；原程序/工具/监听器归 child |

这里稳定接口只用于确有消费者的能力；Wake 不为引导新建一个空执行门面。
disabled：设置存在、状态明确、拒绝新业务；missing：设置存在、报告缺失 key/provider；
uninstalled：该 owner 的接口和贡献均撤回。Akasha 的稳定兴趣接口不持有脱离 child 寿命的
旧 worker；调用必须经当前 child 的受保护入口，无 worker 时明确不可用。

## 4. 子层级与排序

### 4.1 分组是入口归属

~~~text
onboarding
├─ channels（channels 提供普通子入口注册点）
│  ├─ QQ：渠道接入 / 独立发送
│  └─ Telegram：渠道接入 / 独立发送
├─ models（复用 models.connection-types.v1）
│  ├─ Codex
│  ├─ OpenAI-compatible
│  └─ OpenCode
├─ Akasha 情景记忆
└─ Wake 主动联系
~~~

Telegram/QQ 自己贡献表单，引导不写死名称。渠道接入与独立发送分别提交、分别显示结果；
不新增跨插件总开关。共享分组不等于共享配置 owner。
models 子项是连接方式，不是必须逐个完成的步骤。Akashic 已有发送能力可供 Wake 选择，
本次不新做 Web/Mobile 设置页。

分组不是功能开关，不保存“channels 已开启”。sender 依赖 delivery，仍可显示在渠道分组。
缺少所属分组只撤去相关可选 UI 贡献或显示局部诊断，不使整个 onboarding 失败。

### 4.2 唯一排序规则

~~~text
level(node) = 0                         没有内置插件前置
level(node) = 1 + max(level(parent))    有前置
排序键 = (level, 直接依赖的不同插件数, 稳定 plugin_id, 插件内 step_key)
~~~

宿主能力不计一个插件，同 provider 的多个 key 只计一次。
只沿功能分支的业务边计算，排除可选 UI/onboarding 分支，避免把注册关系混进业务图。
GraphReader 的边为“功能 Fiber → ServiceKey → 当前 provider plugin”，保留 Fiber 的 owner
和父子关系。step 引用实际功能分支，投影收集这些分支的声明边再折叠为 plugin 边；
经过同插件的稳定接口时也展开它所对应的功能分支，不能只读取无依赖的 root。
没有引导项的中间插件沿其真实必需分支继续展开。功能分支由注册引用明确指定，
不靠子 Fiber 名称猜测，也不由插件重填一份依赖名单。
先在完整相关图算层级，再隐藏没有步骤的中间节点。分组不要求连续做完所有子项；
下一步沿全局排序前进。错误边或环不能用名字排序掩盖，保留具体 owner 诊断。
provider 缺席的边返回 unresolved key/reason，不能猜测卸载插件的 ID；依赖它的层级也标记
未解析，放在可解析层级之后，以已知直接依赖数量、plugin_id、step_key 稳定排列并显示原因。
缺失能力恢复后重新计算。该状态不宣称获得了完整层级，也不阻断其他步骤。

基线静态盘点相关顺序：

| 层级 | 项目（直接插件依赖数） |
|---|---|
| L0 | channels(0)、delivery(0)、models(0)，以及其他基础插件 |
| L1 | codex(1)、openai_compatible(1)、opencode_go(1)、qq_sender(1)、telegram_sender(1)、qq_channel(2)、telegram_channel(2) |
| L2 | akasha(6) |
| L4 | wake(8) |

基线共 50 个内置插件、97 条顶层插件依赖边，排除可选 UI 分支。
这是声明结构，不代表运行可用性；拆开功能分支和补 sender 边后重算，不能固定成 priority。
同层同数量按名字排时 channels 在 models 前，不增加“模型永远第一”特判。
这指分组/插件目录的平局规则。channels 本身没有配置步骤，驱动也只是 models 的表单选项；
本次实际可执行步骤因此是 models → L1 的 sender/channel → Akasha → Wake。
左侧分组负责定位，页脚下一步只服从实际步骤序，不按分组标题强迫连续填写。

### 4.3 选择具体 sender 后建立真实依赖

当前 Wake 只声明 DELIVERY_SENDERS，执行时才绑定具体名称。
参考已有 tool_key(name)，delivery 在同一 sender 注册时发布具名能力，撤销时一并回收。
Wake 配置目标后，其运行分支依赖选中的能力；原 Bindings、身份和发送回执不变。

现有 registered_names() 只含正在注册的 sender，不能用它发现尚未配置的候选。
delivery 的发送目录增加 candidate 注册：sender 的常驻设置分支用 Effect 贡献名称、owner、
状态入口、设置路由与拟提供的具名 key，不打开网络连接，不授予发送权。
实际 sender 和具名能力仍由功能分支共同注册/回收；候选记录不冒充可运行能力。
Wake 从这个目录展示候选，只为已选目标建立动态 ctx.inject 依赖，不把所有 sender 当前置。
目录与绑定均归 delivery，onboarding 不读取它或维护第二份发送目录。
有未配置的候选就返回其配置入口；没有候选就说明缺少发送能力。
关闭一个 sender 只使该目标不可选，不阻断其他渠道。
关闭 Telegram 收件不自动关闭独立发送；关闭 Akasha 会使需要兴趣能力的 Wake 不可开启。
这些前置由业务 owner 判断，引导不维护插件名称规则。

## 5. 状态与配置应用

### 5.1 无“跳过”，保留未配置状态

owner 分别报告：
- 选择：尚未决定 / 开启 / 关闭。尚未决定不是第三个按钮。
- 实际状态：可用 / 前置关闭 / 缺配置或能力 / 应用中 / 故障或未知，附具体原因。

本次带开关的插件统一在各自配置中使用 enabled: bool | None：null 为未决定，true 为开启，
false 为明确关闭。新安装默认 null，只有 true 才运行；不增加独立完成表或 decision 字段。
旧输入先按 §7.2 升级再解释，不能直接拿旧 enabled=false 默认值推断新用户已选择关闭。
模型仍由自己的连接/绑定状态判断是否可用，不新增总开关。

| 情况 | 页面动作 |
|---|---|
| 前置满足，本项未决定 | 开启或关闭；开启需完成表单 |
| 用户明确关闭 | 已决定，可以下一步 |
| 开启且正式应用成功 | 已决定，可以下一步 |
| 前置明确关闭或未安装 | 显示受阻，只允许下一步；不写成用户关闭本项 |
| 前置未决定 | 先处理前置 |
| 校验、探测或应用失败 | 显示错误，可修复；允许关闭的功能可明确选择关闭，不冒充完成 |
| 刷新或取消填写 | 不提交选择，重开后按真实配置继续 |

完成表示当前可决定项已处理，其他受阻项有明确关闭/缺失前置；不代表所有功能已开启。
故障不抹去原开关。开启前置后，从未选择的后续项重新待决定；明确关闭的保持关闭；
原已开启但受阻的按原配置恢复，不重放已结算的外部效果。

### 5.2 提交与生效有一个宿主 owner

模型继续沿 models 原 revision 协议。其他本次固定配置按以下链路：

~~~text
插件表单 → 插件校验/探测 → 自身配置应用端口
                               │
                          返回受理回执
                               │ 原请求作用域退出
                               ▼
                   固定原制品与新配置，提交/应用
                               │
                               ▼
                    重读正式采用状态，再显示成功
~~~

端口只允许读自身输入版本、提交自身已校验配置、查询自身回执。
插件解释字段；Core 只检查归属、编码、凭据引用权限、输入版本和提交合法性。
凭据复用 CredentialRef，查询不返回明文。改开关使用当前已选制品，不拉取最新代码。

请求包含 request_id、expected_input_ref、expected_config_revision 和已校验的新配置。
身份由 Context 绑定，回执包含请求号、旧/新输入引用、阶段与明确错误；跨 generation 的
同一 owner 可读自身回执。复用宿主串行控制面，在既有更新记录中增加配置操作类型，
不新增引导队列。具体提交阶段如下：

| 阶段 | 宿主动作与可观察结果 |
|---|---|
| accepted | 核对自身版本，持久记录请求，返回回执；由宿主任务继续，原请求退出 |
| prepared | 复用当前选中制品的 code/environment，归档新配置和固定输入；此时不改 config.input.json、不排空旧实例 |
| selected | 串行提交前再次核对输入版本，以 selection 的 expected_ref 做 CAS；提交新的正式输入引用并记录阶段 |
| applying | 从正式输入备份并发布 config.input.json，再排空旧 generation、启动新 generation；依既有 Manager 的 CAS→drain→activate 次序 |
| active / failed | 正式选择、配置文件和新 generation 就绪一致才报告 active；失败给出阶段和原因，不把受理当成功 |

需扩展输入准备入口以接收新配置并复用选中制品，不能先调用会覆盖文件的 save_config()。
CAS 前失败保持旧选择/文件/运行；CAS 后失败保留已提交事实，不假称回滚。
恢复使用同一更新记录和正式输入修复配置文件、继续或报告启动失败，不重新执行业务发送；
启动时先处理未完成的配置发布，再装载正式选择。查询根据正式引用核对阶段，不能只信
最后一次阶段文字。请求退出不取消已受理任务，重复 request_id 读取同一结果。

请求不能持有自身调用许可等待自身排空；受理后只通过短查询读取回执。字段含义归插件，
正式 selection 中的固定输入是运行依据，config.input.json 的发布与恢复归同一提交 owner。
这些端口在本次实现中通过既有固定输入、Root 选择和配置回执完成。旧输入保留，无新增 GC；
不同插件分别提交，不做跨插件原子操作或专门处理高频更新的机制。

## 6. 每个插件改什么

| 对象 | 改动与边界 |
|---|---|
| models / 三驱动 | 贡献分组与步骤；复用目录检测、默认模型与保存协议；独立模型页保持可用 |
| channels | 可选 UI 分支贡献渠道分组和子注册点，不拿其他渠道的配置 writer |
| Telegram/QQ channel、sender | 独立设置/引导分支，迁移 configure.py 校验，通过自身端口应用配置；关闭不擦凭据 |
| akasha | 明确启用选择，配置入口先于运行依赖；开启检查 embedding，关闭停止新的学习/召回/兴趣运算；保留图与消息 |
| wake | 明确启用选择，独立设置入口，复用真实前置检查，选具名 sender 和目标；不增加 proactive 插件 |
| delivery | 同一发送目录增加候选描述，实际具名能力与 sender 注册同寿命；不建立新发送或回执 owner |
| prompt / setup | 缺失人格文件由 prompt 初始化；所有替代入口完成后移除 CLI 插件问答；不覆盖已有 VEDA.md |
| shell / Chat | 普通设置/引导入口和真实不可用原因；onboarding 卸载后仍有独立设置；Core 不按业务名称分支 |

Wake 优先从现有合法会话/收件目标选择，复用身份与 Session admission 规则。
不要求用户填写内部 session_id，不编造已有对话。新用户没有合法目标时可返回渠道/对话
建立目标，或明确关闭 Wake；sender 注册成功不能当作目标已送达。
连接校验不擅自发消息，真实送达验收使用明确测试目标。

### 6.1 页面入口与旧 setup 收尾

onboarding 通过普通 shell 页面合同贡献“初始配置”入口，首次缺配置时显示有序清单与
当前待处理项，不再区分 BLOCKING/SHAPING/OPTIONAL，不等待首次 Turn 再弹记忆决策。
是否仍需配置由当前 owner 状态派生；退出页面不写跳过，也不把清单标成完成。
用户已做出的关闭决定不会在刷新、升级或重装后反复询问。
shell 只使用普通页面/动作注册，不按模型或 onboarding 的插件名分支；onboarding 未安装时
仍显示各插件独立设置入口。Chat 的不可用原因仍由真实模型/运行状态给出，链接使用 owner
贡献的配置动作，不让 onboarding 承担“对话是否可用”的判断。

现有 models first-run 表单收敛为同一独立表单，由引导挂载，不再自己维护另一套首跑流程。
main.py setup 最后收窄为非交互最小配置与 workspace 初始化；删除插件 configure.py 子进程
问答链之前，先迁移四个渠道表单的校验，并验证 prompt 仅创建缺失 VEDA.md。
Web 与现有使用同一 Web 壳的客户端共享入口；本次不添加原生移动端专用引导或配对流程。

### 6.2 与新引导重合的旧步骤：本次移除清单

以下均是当前源码已核对、后续实现时处理的对象，本次只清理设计文档。

| 当前入口 | 现有行为 | 后续处理与接替者 |
|---|---|---|
| [setup_wizard.py](../../bootstrap/setup_wizard.py) 的 run_setup_wizard / _run_plugin_setups | 询问覆盖 Core 配置，然后逐个执行已安装插件 configure.py | 删除交互问答、子进程遍历和其专用 _setup_runtime/env 注入；setup 保留缺失配置创建、校验、workspace 初始化，已有配置不覆盖 |
| `telegram_channel/configure.py`（已删除） 的 main | 是否配置 Telegram、BotFather 提示、token/getMe、用户名允许范围 | 迁到 Telegram channel 的自有 Web 表单/API；保留连接校验和凭据保护 |
| `telegram_sender/configure.py`（已删除） 的 main | 是否启用独立发送、Bot token | 迁到 Telegram sender 的自有表单/API，独立提交其开关和凭据 |
| `qq_channel/configure.py`（已删除） 的 main | 是否配置 QQ、Bot UIN、允许用户、WebSocket 启动超时 | 迁到 QQ channel 的自有表单/API；超时归高级配置，不在主步骤强制问一遍 |
| `qq_sender/configure.py`（已删除） 的 main | 是否启用独立发送、可选 token、OneBot WS endpoint | 迁到 QQ sender 的自有表单/API |
| `prompt/configure.py`（已删除） | setup 子进程中创建缺失人格文件；实际没有用户问答 | 删除单独 setup 脚本入口，initialize_veda_if_missing 由 prompt 初始化接住，已有文件字节保持不变 |
| [models/web_module.js](../../plugins/models/web_module.js) 的 renderCatalog | 以 connections.length 判断首跑，切换标题、样式与区块 | 收掉独立首跑展示分支；同一连接表单供独立设置页和 onboarding 使用，保留正常空目录展示 |
| [desktop-chat-view.tsx](../../frontend/chat/src/desktop-chat-view.tsx) 的 DesktopEmptyState | 将 needs_setup/starting 解释为未连模型/模型已保存，并硬编码 /#models | 改为真实不可用原因和 owner 贡献的配置动作；保留空状态、禁止无模型发送和启动状态 |
| [README.md](../../README.md) 的两次 setup 教程、main.py setup 帮助、[prompt/README.md](../../plugins/prompt/README.md) | 安装默认插件后再次跑 CLI 问答、描述 configure.py 初始化 | 替代入口验收后一起改为非交互初始化与 Web 配置，不提前把当前教程写成已实现 |

上述四个旧渠道 configure.py 原先含 `--upgrade`：转换旧 TOML，涉及凭据的还会将明文转为
CredentialRef。删问答不等于删升级能力。实施时将转换函数及明确的离线升级入口迁到所属
插件的升级模块，再删除旧文件；同步 [upgrade_plugin_config.py](../../scripts/upgrade_plugin_config.py)
对“含秘密配置由插件自有 upgrade_config.py 升级”的说明。保留备份、权限与失败语义，不在引导中自动跑旧迁移。

### 6.3 不属于重复引导的能力

- 模型的连接 CRUD、登录/授权、目录发现、同步、默认角色与向量绑定继续保留；三种驱动
  表单是业务配置本身，onboarding 复用它们，不再复制一套。
- Telegram/QQ channel 与 sender 是不同业务 owner，不能因相同渠道名合成一个配置 writer。
- 配置/凭据读写和升级通用能力不随 CLI 删除；旧输入、凭据与用户决定不清空。
- `main.py setup` 的非交互初始化与默认插件安装继续保留；不因引导存在就省掉安装链。
- `/api/shell/state` 当前只报告配置文件存在性与 Chat health，没有读取 models 目录，也没有
  跳转实现；不能照旧设计的文字删除一个不存在的模型跳转 adapter。真实误判在上表的 Chat 消费端。
- [settings_api.py](../../bootstrap/settings_api.py) 的 `/settings → /#models` 是旧 URL 重定向，
  web_shell 的旧设置写接口已经返回 410；它们不是仍在运行的配置问答，本轮不顺带移除。
- Akasha/Wake 当前没有对应 configure.py；本次给它们补开关与配置页，不虚列旧向导删除项。

### 6.4 退役顺序

~~~text
自有 Web 表单与正式配置应用可用
                │
                ▼
旧凭据/配置升级能力有明确承接
                │
                ▼
移除四套 CLI 问答与 prompt setup 入口
                │
                ▼
删除宿主遍历/子进程辅助代码，收敛首跑 UI，更新教程
~~~

验收同时覆盖首次安装与已有配置：新入口可配置、关闭后可再开启、重启后采用正式输入，
已有 VEDA/凭据保持不变；卸载 onboarding 后独立设置仍可用。完成这些场景前不退役旧入口。

## 7. 卸载、新旧用户和持久状态

### 7.1 卸载 Wake 只撤去其引导贡献

~~~text
onboarding ──提供注册能力──▶ Wake 可选引导分支
                                  │
                             Effect 持有步骤
                                  │
                              卸载 Wake
                                  ▼
                         仅回收这一项注册
~~~

onboarding 不 import Wake、不 inject WAKE_*、不要求固定步骤数量或 Wake 探测成功。
卸载后目录无 Wake，模型与渠道仍能配置；旧表单返回目标已移除并重读现有步骤。
不需要卸载补偿表。反向卸载 onboarding 只回收引导分支，功能及独立设置继续可用。
关闭 Wake 保留步骤并显示关闭，卸载 Wake 才让步骤消失。

### 7.2 新旧状态由真正 owner 解释

- 新安装功能缺启用决定时不运行，报告待决定，不把默认值冒充用户选择。
- 对已有正式选择，由插件显式升级配置：保留原开关，缺新字段时按旧版本已定义行为补齐。
  旧 Telegram/QQ 显式布尔值原样保留，旧字段缺失沿旧默认 false；旧 Akasha/Wake 缺字段
  补 true，保留原有自动工作意图；其缺向量/缺目标等实际状态仍单独报告，不假称可用。
- 安装/输入升级 owner 明确提供新安装或升级来源，不用文件时间、凭据残留或聊天记录猜测。
- 已有决定无需反复问；新业务决定才新增待配置项。改 step key 不重置业务配置。
- 卸载保留数据，重装沿用原决定。无需 first_seen_at 或引导数据库。

### 7.3 增、改、减与恢复

| 对象 | 增加与更新 | 逻辑失效 | 减少与恢复 |
|---|---|---|---|
| 插件开关/配置 | 首次明确选择或升级补齐；插件校验后请求新输入 | 旧输入不再被当前 selection 采用 | 关闭/卸载不删；既有备份与固定输入保留；数据删除另行授权 |
| 模型连接/绑定 | models 原保存与 revision 协议 | 原 revision 规则 | 沿现行合同，本次无新清理 |
| 凭据 | 私有目录增不可变版本，配置改引用 | 撤销沿现行协议 | 关闭不擦除；备份必须包括私有凭据，不能只备份 plugin-data |
| 正式选择/应用回执 | 宿主原归档、selection 与更新记录 | 按原状态推进，失败可见 | 本次无 GC，依既有恢复证据核对 |
| 步骤/排序/受阻状态 | Effect 注册与即时计算 | 随图和配置重读 | 卸载回收内存，无业务行需要删除 |
| 消息/记忆/主动记录 | 配置模块不新增正文写入路径 | 原领域 owner 继续恢复、结算 | 本次无删除或裁切权，不丢已接纳工作 |

回滚 UI/onboarding 不回退用户配置。恢复旧配置通过同一入口生成新输入；
已有发送、学习和 Message 不因指针回退而撤销。普通失败、取消、重启沿原 owner 处理，
不增加罕见热更新专用分支。

## 8. 实施与验收

1. 补 Supervisor 无配置存活和自身配置应用端口，证明设置可打开、配置真正采用。
2. 做普通 onboarding 注册表/投影/页面，先接 models 和 Telegram sender。
3. 接渠道分组与剩余 Telegram/QQ 表单，验证独立开关、校验与重启恢复。
4. 接 Akasha、Wake 的真实开关和具名发送目标，验证依赖阻断与恢复。
5. 所有替代入口完成后删除 CLI 插件问答，接住 prompt 初始化和 Chat 设置入口。

本次交付包含实现、commit、push 和 PR，不包含合并或部署。按 WORKFLOW 完成概念验证与静态检查，
功能使用隔离 workspace 的真实场景；不新增镜像实现细节的单元测试。

| 场景 | 通过条件 |
|---|---|
| 无 config.toml | 2236 壳层可配置模型和渠道，不被未就绪业务杀死 |
| 排序 | provider 在前，数量/名字打破平局，输入枚举顺序不影响结果 |
| Telegram sender 开启 | 无效凭据失败，正式采用后才显示开启，重启保留 |
| 关闭 Akasha | 停止新学习/召回/兴趣运算；Wake 不接纳新职责，页面仅下一步；原数据保留 |
| 恢复前置 | 未决定项重新待选，明确关闭项仍关闭，原开启项按原配置恢复 |
| 关闭所选 sender | 对应 Wake 目标不可用，其他发送者和无关插件不受影响 |
| 卸载 Wake | 引导仍可配置模型和渠道，只有 Wake 步骤消失，数据保留 |
| 卸载 onboarding | 已有业务及独立设置继续工作 |
| 配置受理后失败/请求退出 | 不误报生效、不自身排空死锁，按既有回执恢复或报错 |
| 单项状态读取失败 | 保留该项故障诊断，其他插件仍可配置；不输出空目录或假完成 |
| 老用户升级/重装 | 原状态与数据保持，不重走全屏流程，不凭空开启功能 |

## 9. 已有验证与待验收边界

本基线静态盘点覆盖顶层必需依赖、提供者和 RPC 声明：50 节点、97 边无环，
全部 provider 先于消费者，100 次打乱输入排序一致。它不证明新设计运行通过、正式迁移完成
或真实送达。

核心设计经独立只读审查后实现。实际检查和仍未覆盖的外部服务边界见[验收记录](plugin-onboarding-validation.md)。

## 10. 基线完整层级清单

下表用于核对当前 50 个内置插件；括号为去重后的直接插件依赖数量。
同一行已按数量、名字排好。新设计拆分功能分支后应重新从真实图计算。

| 层级 | 当前顺序 |
|---|---|
| L0 | assets(0)、channels(0)、commands(0)、content(0)、context(0)、delivery(0)、drift(0)、eventmail(0)、managed_processes(0)、mcp(0)、models(0)、react(0)、runtime_inspection(0)、sources(0)、turn_projection(0)、ui(0)、workloads(0) |
| L1 | akashic_sender(1)、codex(1)、conversation_ui(1)、openai_compatible(1)、opencode_go(1)、projects(1)、prompt(1)、qq_sender(1)、shell_ui(1)、stable_view(1)、telegram_sender(1)、tools(1)、workbench_ui(1)、qq_channel(2)、telegram_channel(2)、compaction(3)、conversation(4) |
| L2 | standard_web(1)、tool_search(1)、delivery_policy(2)、message_push(3)、standard_tools(3)、programmatic(4)、markdown_memory(5)、akasha(6)、computer(6) |
| L3 | plugin_update(5)、reply_program(8) |
| L4 | reply(4)、scheduler(4)、wake(8) |
| L5 | subagent(6)、akashic_clients(7) |


## 11. 实现决断与交付边界

以下决定由维护者“自行判断、不要询问”的授权完成。

### 11.1 真正的职责边界

```text
┌ Core：当前 Context → 自身配置输入 → selection CAS → 排空/启动 → 回执 ┐
│ 不识别 models、Telegram、Akasha、Wake，也不保存引导进度               │
└───────────────────────────────────────────────────────────────────┘
      ▲ 插件提交自己的配置                 │ 只读当前依赖提供者
┌─────┴──────────────────┐      ┌─────────▼───────────────────────┐
│ 每个业务插件            │      │ 普通 onboarding 插件             │
│ 开关 / 字段 / 校验 / UI │ ───▶ │ Effect 目录 / 拓扑排序 / 当前状态 │
└────────────────────────┘      └─────────────────────────────────┘
```

- `agent/plugin_composition/plugin_config.py` 只接受真实 Context，自身端口没有目标 plugin ID 参数。
- `agent/plugins/manager.py` 归档同一代码制品的新配置，沿原局部换代路径采用；请求返回后由宿主任务继续排空，避免请求等自身卸载。
- `config_updates` 位于原 reload journal，持有请求身份、旧/新 input ref、阶段与错误；是应用回执，不是引导进度。新增 Core yoyo migration 先备份已有 journal，只增表。
- `onboarding` 不导入其他插件实现。业务插件可选注入 `ONBOARDING`；卸载引导只回收这些贡献子分支。
- 配置根分支持续提供设置；实际业务放在有依赖的子 Fiber。关闭 Akasha 停止学习、召回与兴趣运算；稳定兴趣入口只报告不可用。Wake 不再接纳新职责，原始消息和回执保留。
- Delivery 区分可配置候选与实际 sender 注册。具名 sender ServiceKey 随实际注册 Effect 回收；Wake 只依赖自己选择的 sender。

### 11.2 界面与正常状态

- Shell 提供通用 `pages` 渲染视图；引导复用各插件自己的设置页面。没有在 Core 或 Shell 硬编码引导步骤、业务名字或必装插件。
- 每页只有开启/关闭，保存后才允许下一步；前置关闭或缺失时展示原因，只能下一步，不伪造关闭决定。
- 下一步重新读取真实状态；浏览器只记当前页面。无完成数据库、永久跳过、版本提醒或首次 Turn 弹窗。
- 表单保留已存秘密但不回显，空密码保留旧值；验证失败保留输入，按钮在应用中禁用。未保存离开有确认；窄屏、键盘、焦点与错误文案纳入 CDP 检查。
- 自身配置换代后旧 Web catalog 失效。通用宿主在已提交配置后等待新 catalog 并刷新，保留当前路由；新页面读取请求回执后显示生效。`accepted` 不显示为成功，普通后台插件更新仍要求显式刷新。
- 模型驱动继续通过 `models.connection-types.v1` 提供 Codex、OpenCode 和兼容 API 选项，不变成互相依赖的强制步骤。
- 原模型页缺少新增向量模型入口。本次补齐“添加向量模型”：选现有连接、模型名、服务文档中的维度，真实发送固定测试文本并校验输出，再设为默认。维度不从模型名字猜测；保持原模型存储和验证 owner，不新建 Core 配置协议。
- Wake 只列出已有、可列举会话的真实 channel origin 和已注册 sender；不要求用户填写内部 session ID。没有会话时明确提示先建立对话，不擅自发送测试消息。
- Wake 关闭后仍能只读历史。新安装尚无 Akasha 图时返回明确的“尚未建立”状态，不制造空学习成功。

### 11.3 安装与旧入口退役

- 新安装制品可以提供 `initial_config.json`。内置可选功能用 `enabled: null`；只有首次固定输入采用它。选中输入在业务启动前写入缺失的配置文件，防止后来更新把“未决定”误读成老用户默认开启。
- 旧输入缺少 enabled 时保留既有默认：渠道 false，Akasha/Wake true。已有配置始终优先；关闭和卸载均不删除数据或凭据。
- `setup` 收窄为幂等 Core 初始化，已有配置不覆盖；默认启动入口在迁移之前调用同一初始化路径，Supervisor 接收已准备好的配置。Core 不擅自安装业务插件，插件组合仍由 distribution profile 拥有。
- 默认 profile 加入内置渠道、记忆、Wake、引导及模型连接方式；未决定的功能不运行。安装操作与业务配置不再混为一套 CLI 问答。
- 删除四个渠道 `configure.py` 的问答，旧 TOML 转换移到各自 `upgrade_config.py --data-dir PATH`；不在引导中自动迁移。
- 删除 prompt 的 configure.py；首次 apply 只创建缺失 VEDA，已有合法字节不变，损坏内容仍明确报错。
- models 的首次配置专属布局退役；Chat 不再把所有启动等待都解释成“模型已保存”。README 去掉第二次 setup。

### 11.4 运行验收后的修正

- 原有启动入口先做迁移，再进入 Supervisor；把初始化放在 Supervisor 已经太迟。本次由 `main.py` 在首次迁移前建立空选择，防止新目录误判为旧 workspace。既有 workspace 缺 stable 仍明确失败，不猜测或重建权威选择。
- Shell 保留隐藏页面，所以表单在真正可见时重读配置与前置；有草稿或应用中的请求不被刷新覆盖。这样“先聊天、再设置 Wake”能立即看到新会话。
- 欢迎弹窗通过 React portal 挂在 body，避免隐藏引导页面使一个不可见 modal 锁住其他页面。弹窗仍由普通 onboarding 模块创建和回收。
- 新表单和弹窗使用现有 Material 主题语义色；深色模式与浅色模式分别做对比度检查。
- 已选 sender 缺席无条件标记 Wake blocked，即使没有可替代目标；其他目标可用时，独立设置允许改选。
- 收到应用失败回执后恢复当前草稿的可提交状态，用户可以直接重试。

### 11.5 验证记录

实现与实际验收记录保存在同目录的 `plugin-onboarding-validation.md`。测试只用一次性 workspace、安装目录与 CloakBrowser profile；未写正式 workspace、未登录用户账号、未向真实 Telegram/QQ 发送消息。外部 Provider 的协议替身证据与真实外部认证/送达严格分开。
