# 0168 · 用户界面与 `self` 一一对应；助手开会话、跨会话对话走自动作者窄入口

- 状态：accepted / implemented
- 日期：2026-11-27
- 规格：Self-CLI / Operation Bus 规格（`OPERATIONS_V1`），顾问评审结论见 PR 描述

## 背景

助手要能做到用户在界面上能做的事：建项目、配项目属性、开一个会话把问题交代好让用户接着聊、给另一个会话发消息。过去每多一种能力就多一个工具，用户界面和工具各写一套，漂移只能靠人记。

## 决定

**一个工具，一份操作表。** 插件把操作注册到 `OPERATIONS_V1`，模型只看到一个工具 `self(argv)`，按 `help`/`describe` 现查。界面入口与 operation 调用同一段领域代码，不准 HTTP 自调用或复制业务逻辑。依赖插件用命名空间参数（`--akasha.learn`）挂到主操作上，需要原子性的经 Ledger 联合事务（`OwnerState.transact_joint`）与主记录同一次提交。

**一对一由 CI 保证。** `scripts/check_interaction_parity.py` 从源码独立采集用户入口（HTTP/WS 路由、WS 帧、Plugin UI RPC、Slash、Gateway RPC）与 operation 声明（`OperationSpec.mirrors`、`QueryMirror`）。每个入口恰好一个 operation 镜像，或写进 `explicit_exemption` 并给出类型与理由；每个 operation 至少镜像一个入口，或写进 `agent_only` 并给出理由。不允许通配。Core 与 extra 各有一份清单，在合成树里合并检查。没有欠账清单：新入口在同一个 PR 里镜像或豁免，否则 CI 失败。

一个入口里按 `type`/`kind` 分出多个动作的（如模型设置的 `/command`），门禁按 `<入口>#<动作>` 逐个计数；Slash 命令的子动作由 `CommandDefinition.subcommands` 显式声明，同样逐个计数。`check_interaction_parity.py runtime --workspace W` 用 `self describe` 对账运行中实例：运行期注册了、静态采集看不到的 operation 判失败。

**三个入口，一张表。** 模型工具 `self`（`self_cli` 插件，actor=agent）、终端 `akashic self …`（Gateway `self/invoke`，actor=user）、界面按钮（各插件领域函数，与 operation 同一段代码）。`user_only` 的 operation 只对用户身份可见。处理函数在注册插件自己的执行作用域里运行，不经总线转手 owner 身份；意外异常回成 `failed`，不拖垮工具循环。

**豁免只有四类。** transport（心跳、订阅流、视频/信令、静态资产）、view_only（页面壳、只改一个客户端视图的状态）、user_only（代表人本人的动作或既有决定禁止模型读取，如 0120 运行诊断、0122 交互终端）、auth_security（口令、登录票据、OAuth 回填）。

**助手的会话动作不冒充用户。**

- `session open` 经 `SESSION_ADMISSION` 接纳正式 listed 会话，交接记录与侧栏可见的交接卡（`handoff` 来源的 Output）在同一事务提交，不造用户消息、不调用模型；用户首句说「继续」时，交接内容由上下文材料交给模型。`--mode start` 再提交一条自动输入开始处理。
- `session send/steer` 由 `self-cli` 作者写入对话来源，不走用户输入登记，也不经过 `/stop`、`/abandon` 文本快路径。
- 用户优先：目标正在回复时 `send` 返回 busy，不抢占；`steer` 暂存到下一步，用户在此期间停止或放弃则作废，不在 `/stop` 之后再唤醒目标。
- 防死循环：自动输入带 `cross_session` 片段（来源会话、根请求、剩余跳数），从用户原话起最多 3 次自动往返，禁止发给自己。
- 记忆边界：交接只带显式写进交接卡的内容；来源会话只记 ID，不授予新会话回读权。
- 子任务看不到 `self`（`self` 能触达插件安装、部署等子任务被禁的能力，权限未落到 operation 粒度前一律不给）。
- 会话作用域的操作（消息读取与查找、子任务工作台、终端、设计文档）经 `session_of(req)` 取目标：用户可指定任意会话，助手只能作用于自己所在的会话；跨会话协作走 `session send --wait`。
- 回执绑定调用指纹（身份、会话、argv）：同一请求 ID 换身份或换参数返回 `conflict`，并发的同一调用只执行一次；`receipt` 只对原调用者可见。
- `send` 的空闲判定与提交在同一次来源准入里（`accept(when_idle=True)`）；已提交的同 ID 消息直接重放。`--retry` 同样消耗跳数，用户停止或放弃过的输入不由助手恢复。
- 自动输入不带渠道身份，不执行 Slash 命令；以 `/` 开头的文本交给模型。
- 没给标题时，会话标题取 goal 首行。

**附加设置随主记录原子提交。** 新建项目时，akasha 等插件在 `scope.fields` 里只登记取值（`contribute(namespace, read)`），不自己写；项目插件把它们翻成 `--<ns>.<opt>`，走与 `self project create` 相同的 operation，扩展的写入与项目记录经 `transact_joint` 同一次提交。

## 已知限制

- 终端 `akashic self` 经 Gateway 以 user 身份调用：能到达 Gateway 的进程都被当作用户。助手的 shell 工具若能访问本机 Gateway，就能以用户身份调用 `user_only` 操作；这与 shell 本身能读写工作区属于同一级风险，Gateway 调用凭据的身份区分留待后续。
- 静态门禁只证明声明存在：写在从不执行的函数里的 `OperationSpec` 也能通过；`runtime` 子命令只能发现运行期多出来的 operation，未启用的插件无法区分。
- HTTP 路由只认 `app`/`router` 接收者，Gateway 方法只认 `RpcMethod.key(大写常量)` 与 `def rpc_methods()` 字典；其他写法会漏采，新增写法时须同步采集器。

- operation 回执只在进程内保留；重启后重放依赖各写操作自身的稳定 ID（会话 ID 由请求 ID 派生）。
- `session send` 不在目标回复中途注入；需要中途插话用 `session steer`。
- 门禁按源码静态采集；运行期动态生成的入口（如按配置生成的设置表单字段）以其宿主路由为单位计数。
