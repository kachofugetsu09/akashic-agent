# 普通资源 provider

本层按 0071 把资源操作移出 Manager 与 Snapshot。仓库默认发行 profile 显式选择
`host_execution`、`workloads`、`managed_processes`、`mcp`。贡献插件通过 `inject` 声明所需服务；
缺少服务与服务冲突按普通组合规则失败。Core 不按 provider 名称或资源类别启动插件。

```text
┌──────────────────────────────────────┐
│ 内核绑定代码授权；执行插件发布 Controller │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ provider apply → provide(窄服务)       │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ 贡献 apply → Scope 登记关闭责任         │
│           → await 取得实际资源句柄     │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│ 先 Workload/Process，后依赖它的 MCP     │
│ 关闭反序执行；失败保留原 owner 与依赖   │
└──────────────────────────────────────┘
```

## 服务与实际 owner

- `WORKLOADS.register(ctx, definition)` 等待 Controller 回执与健康检查，返回真实
  Workload 句柄。`url(ctx, port)` 与 `borrow(ctx)` 检查实际 Root、Context owner
  和 activation。MCP 接受 `WorkloadEnv(..., handle, port)`，不再让编译器按字符串找资源。
- `MANAGED_PROCESSES.register(ctx, definition)` 等待实际进程就绪。端口、借用、
  恢复与终止归该 provider。借用未释放时不终止进程。
- `MCP_SERVERS.register(ctx, definition)` 固定按调用打开的目标，并立即检查资源引用。
  `open(ctx, name)` 保留原有每调用独立会话语义：先把会话关闭责任登记到贡献 Context，
  再取得借用并等待进程/协议握手。退出后 route 失效；不会自动重放工具调用。
  `failures()` 只读该 provider 保留的真实会话；清理失败保留同一 Effect，
  随所属 Fiber 或 Root 再次排空而重试，不创建新会话或重放工具调用。

三个 provider 的注册限于 `apply`。MCP 的 `open` 是运行期操作，取得 exact Root scope。
Runtime catalog 只列已注册的 MCP 目标并标记 `declared`，不把定义当成已发现的工具。
客户端显式请求目标详情时，当前请求许可经 Core 窄服务验证，再用原贡献 Context
按调用打开会话；工具来自本次握手，会话关闭失败或取消如实传播，不建立常驻目录。
每次实际连接检查必需工具和候选只读范围，不承诺两次远端连接返回相同目录。
无调用者的 `expected_catalog_digest` 参数、摘要副本及比较链已删除；可调用工具仍来自
本次握手，并保持 schema 冻结、route 失效和只读执行边界。
Dashboard 从实际贡献 Context 取得 Workload 服务，runtime catalog 从所选 Root 取得 MCP 服务。
Snapshot 不再存储三份 registry 或 identity；编译器不校验资源定义与依赖路径。

## 宿主授权

`host.execution.v1` 绑定实际 Context 的固定代码 owner 与 Root 权限。
命令解析使用安装 owner 准备的实际 Python 环境目录，cwd 必须位于所属安装代码目录。
环境引用直接指向目录身份，不依赖代码或环境 descriptor 归档；见 [0094](../decisions/0094-plugin-runtime-uses-installed-files.md)。
安装准备与运行时读取按 PLG-002 分开；[0077](../decisions/0077-trust-installed-runtime-inputs.md)
记录删除摘要复验的理由和恢复边界。
插件 API 不接受 formal/candidate 参数、正式 workspace 路径或自报的代码 owner。

环境只继承明确列出的 PATH、语言/时区项与 Supervisor 身份。候选只采用显式
`candidate_env`，不合并正式 `env`。HOME、plugin-data 与 workspace 由实际 Context 固定；
子进程不再第二次合并整个宿主环境。Supervisor 身份仍由进程组 helper 固定。
这不是 OS 沙箱，也不声称可以隔离同 UID 的任意 Python 代码。

`host_execution` 的 `host.workloads.v1` 为实际 Context 发出独立授权，固定 workspace、插件 owner、
候选/正式模式与请求身份。插件不能把候选请求改成正式请求，也不能用其他 owner 的 lease
请求停止资源。Controller 仍拥有容器与挂载的原子操作及回执。

## 新进程 boot 的候选清理

`host_execution` 激活时按当前 workspace 的固定身份调用 Controller
`cleanup_candidates`，逐份确认回执属于该 workspace 的 candidate，且
`container_absent` 和 `mounts_released` 都成立，随后才提供 `host.workloads.v1`。
失败或未知结果使本 provider 失败，其硬依赖无法取得资源；无关插件不受阻断。
Core 不再构造 Controller，也不在 `load_all` 中发起该外部操作。

```text
┌────────────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
│ 执行插件核对本 boot 的清理事实 │ → │ 清理并核对全部回执 │ → │ 提供 Controller 授权 │
└────────────────────────────┘   └─────────────────────┘   └─────────────────────┘
```

执行插件的数据根中 `last-cleaned-boot.json` 只记录已确认完成清理的 boot ID。
同进程换代不重发清扫，新 boot 在回执完整后原子替换旧标记；未配置 Controller 时不写标记。
该派生文件损坏时显式失败，缺失时重新核对清理回执；它不代替 Controller 的 lease、
停止账或未决外部工作。标记只随新 boot 更新，不自动删除；恢复依据是 Controller 的真实回执。
清理只释放旧候选容器和挂载，不删除持久业务数据，不重发未知 start，
也不把连接取消解释为 Docker 已回滚。

## 失败与持久数据

Scope 在外部 await 之前持有关闭回调。取得失败后仍沿原 owner 清理；只有确认成功才解除
责任。进程 spawn 的等待被取消时，先接住实际进程回执，再交给已有进程组关闭链传播取消。

Workload 发出 start 后若缺少有效回执，包括普通错误、取消和身份不匹配，保留完整原请求。
现有 Controller 没有按请求核实/取消未知 start 的接口；清理明确失败并报告需要宿主或
Controller 处理，不重发 start，不把无回执解释为未启动。未新增查询协议或持久 request ledger。
跨进程残留仍由宿主/Controller 处理，不能用内存状态恢复伪造外部效果已回滚。

本层不迁移或裁切数据。Workload/Process/会话关闭只释放所属临时运行资源；不删除正式
plugin-data、消息、回执或归档。代码更新后的数据解释仍由新插件负责。

## 与整 Root 层集成

本层删除中央资源启动阶段，因此正式资源会在普通 `apply` 中取得。整 Root 层必须在旧
正式 Root 关闭成功后才挂载新正式 Root；候选使用宿主绑定的候选权限。该构造、提交、
operation 与外部 Channel admission 协议由并行切片负责，本层不建立第二套发布协议。

Controller 的真实验证入口是 [host_controller_scenario.py](../../scripts/host_controller_scenario.py)：
隔离安装执行与 Workloads provider，在独立 Docker 网络中读取实际 HTTP 响应，
更新服务与执行 provider，再重启、卸载；核对租约排空、数据保留、Controller 关闭回执
与每个 boot 的清理请求次数。MCP、managed process 和 UI 的验收分别由其真实场景负责；
这一场景不代替整发行版或生产部署验收。


## Bridge 状态 owner

`host_execution` 的 Effect 拥有实际探测任务与连接，并提供 `host.status.v1`。
探测不创建 execution manager；瞬时断连明确降级，恢复后重置故障计数。
身份等永久拒绝保留错误码、degraded health 与 Incident，停止本次探测；
不会用诊断失败结束无关插件，新的 generation 可重新取得连接。
Effect 关闭先取消并等待原任务及连接退出，不留下旧 generation 的探测任务。

执行插件通过可选 UI child 注册 `/api/runtime/host-bridge`。路由只读取该 generation
的状态，不把监控任务或宿主内存状态交给 UI；没有 UI 时执行服务仍可激活。
禁用执行插件只撤下其路由与监控，不关闭 UI 或重新 apply 无关观察者。
状态和 health 均为内存事实，本步没有增加文件写入或修改业务数据。

`claim_host_bridge_boot` 目前仍在进程启动、业务数据库构造之前执行。
这一外部认领不由状态探测代替；完整后台迁移要保留认领先于持久 owner 开放的顺序。
实际验证见 [host_monitor_scenario.py](../../scripts/host_monitor_scenario.py)。

## Dashboard listener 的 owner

UI 插件持有 Dashboard 的静态资产、中间件、请求许可和实际 Unix listener。
资产来自所选 UI 代码制品的 `static/dashboard`；缺少构建入口返回 503，
启动不创建或修改代码目录。构建分发把同次生成的 Dashboard 资产放进 UI 插件 bundle。

```text
┌─────────────────────┐   ┌─────────────────────┐   ┌──────────────────────┐
│ UI Effect 保留关闭责任 │ → │ 实际 listener 完成启动 │ → │ 发布 Context endpoint │
└─────────────────────┘   └─────────────────────┘   └──────────────────────┘
          关闭时：先撤下 endpoint，再排空连接、停止 listener、核对并移除原 socket
```

`runtime/dashboard.sock` 保持原节点位置；短 Unix 别名由共享 OS helper 计算。
派生端点计划增加本 listener 的记录，正常关闭减少这一条记录；它不改写权威状态。
UI 关闭只清理其原 socket，节点被替换时明确拒绝删除；监听器未停止时不报告释放成功。
中间件只持有 UI 的实际 Context 与 registry，不取得 Manager、Root 或内部 provider 查找。
旧 WebSocket 在所属 owner 撤下时关闭为 1012；旧 catalog 身份返回明确 stale 拒绝。
`dashboard` 独立命令由所选 UI 插件声明，命令不启动 Root 或打开业务数据库。
真实 listener 验证见 [ui_listener_scenario.py](../../scripts/ui_listener_scenario.py)。

## UI 注册合同归属

`plugins/ui/contract.py` 拥有 Web/Dashboard 注册、Plugin UI slot、目录和失败类型。
消费插件直接导入该模块；Core 的聚合入口和旧 UI 模块不再再导出。
`core.web_ui.v1`、`core.ui_slots` 原子改名为 `ui.web.v1`、`ui.slots.v1`，没有旧 key 别名。
Web 目录编码和 route matching 留在 UI 实现，公开数据保持 frozen。
这层只改变源码合同和 key；资产、query、请求作用域和持久状态行为不变。
消息展示合同和投影仍有 App Server 消费者，随 Gateway owner 迁移一并移走。

## Web Shell 读取派生端点

```text
┌─────────┐    ┌───────────┐    ┌──────────────────────┐
│ Browser │ ─→ │ Web Shell │ ─→ │ 最长匹配的插件 listener │
└─────────┘    └─────┬─────┘    └──────────────────────┘
                    │ 只读
             ┌──────▼───────┐
             │ endpoint plan │
             └──────────────┘
```

Web Shell 只按 `runtime/endpoints.json` 的最长完整路径前缀转发 HTTP/WebSocket，
不识别插件名、业务健康接口、模型设置或固定业务 socket。
路由缺席/监听器不可达返回明确的 503；损坏计划单独返回 `endpoint_plan_unavailable` 并记录原错误。
浏览器 HTML 请求在 runtime 停止后仍得到外壳的不可用页面。

客户端 listener 在就绪后登记自己的 API、WebSocket 和资产前缀，关闭时先撤下端点。
聊天状态、静态缓存和 settings 导航仍由客户端响应；UI listener 负责 Dashboard 及其插件路由。
目录读取使用有界文件任务；查询编码、redirect Location、WebSocket 关闭码都经过真实代理验证。

Dashboard/Chat 的构建产物只进入各自插件 bundle，不再进入 Core tar。
当前聊天区域实现仍在 frontend/chat；区域 slot 化由独立的前端阶段处理。
端点计划和监听器都是可重建状态，本层无权减少消息或插件数据。

## Gateway 远程命令

Gateway 的命令声明提供 `exec`、`plugin-install`、`plugin-status` 和 `plugin-uninstall`。
Core 的通用分派只读取当前选择并调用唯一 provider；命令不取得 workspace lock，
不启动第二个 Runtime，不从配置或固定 socket 路径猜测监听地址。

```text
┌────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ 所选 Gateway CLI │ → │ 只读 endpoint plan │ → │ 实际 JSON-RPC listener │
└────────────────┘   └──────────────────┘   └──────────────────┘
```

JSON-RPC listener 就绪后发布实际绑定地址，停止时撤下；当前 listener 由 AppRuntime 拥有。
命令只接受唯一 `jsonrpc+unix` / `jsonrpc+tcp` 端点，计划缺席、损坏或冲突明确失败。
`exec --endpoint` 仍允许显式选择连接地址。TCP 命令只读现有 `.app-server-token`，
不创建、重置或自动修复 secret。安装、卸载和消息提交都由运行实例的原 owner 执行。

`exec` 的 SIGINT 提交原程序来源的 pause；普通断连只关闭本地读取。
命令 generation 换代不重新 apply 无关插件；卸载命令 provider 后通用分派明确报缺席。
本层只新增可重建的 JSON-RPC 端点投影，既有消息、表和 secret 的写入规则不变。
实际验证见 [gateway_cli_scenario.py](../../scripts/gateway_cli_scenario.py)。

Gateway 的离线 bundle 先把旧 `[app_server]` 完整复制到自身 `config.input.json`。
复制保留旧 Core 的数值转换规则，固定输入之后使用严格 schema；默认值与自定义上限均保留。
这一阶段源配置仍由旧 listener 使用，原字节、注释和表保持不变，作为恢复依据。
目标缺席时只增加 Gateway 配置；已有目标等价则不写，不同或损坏则失败且不记成功。
共享 schema 属于 Gateway 的 migration helper，运行入口与离线迁移使用同一份定义。
本阶段无自动删除、覆盖或减少协议；实际复制、冲突恢复和两次 boot 见
[gateway_config_scenario.py](../../scripts/gateway_config_scenario.py)。

## 进程停止意图

进程外壳通过现有 RestartGate 等待插件的 `request_shutdown`，不新增 ServiceKey。
首次意图立即关闭新 work 准入；外壳按正常路径排空 Fiber、监听器和存储后结束。
正常 EOF 返回 0；带错误的意图在清理后重新抛出原原因，进程返回非零。
取消或清理失败沿原错误链返回，不把停止请求当作物理资源已经释放。
验证见 [process_shutdown_scenario.py](../../scripts/process_shutdown_scenario.py)：真实命令、
失败、EOF 和失败后的新 boot 都核对原 Message 全行；管道作为协调边界。
通用命令分派同时固定实际 `AKASHIC_CORE_ROOT`，命令不根据自身插件路径猜测 Core。
