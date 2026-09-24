# 普通资源 provider

本层按 0071 把资源操作移出 Manager 与 Snapshot。仓库默认发行 profile 显式选择
`workloads`、`managed_processes`、`mcp`。贡献插件通过 `inject` 声明所需服务；
缺少服务与服务冲突按普通组合规则失败。Core 不按 provider 名称或资源类别启动插件。

```text
┌──────────────────────────────────────┐
│ 宿主绑定当前 Root 的执行/Controller 授权 │
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
  `failures()` 和 `retry_cleanup(ctx, identity)` 只操作该 provider 保留的真实会话。
  Root 关闭也重试同一 Effect，而非创建新的会话。

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
命令解析继续使用归档中的 Python 环境，cwd 必须位于所属固定代码制品。
插件 API 不接受 formal/candidate 参数、正式 workspace 路径或自报的代码 owner。

环境只继承明确列出的 PATH、语言/时区项与 Supervisor 身份。候选只采用显式
`candidate_env`，不合并正式 `env`。HOME、plugin-data 与 workspace 由实际 Context 固定；
子进程不再第二次合并整个宿主环境。Supervisor 身份仍由进程组 helper 固定。
这不是 OS 沙箱，也不声称可以隔离同 UID 的任意 Python 代码。

`host.workload_controller.v1` 为实际 Context 发出独立授权，固定 workspace、插件 owner、
候选/正式模式与请求身份。插件不能把候选请求改成正式请求，也不能用其他 owner 的 lease
请求停止资源。Controller 仍拥有容器与挂载的原子操作及回执。

## 新 Core boot 的候选清理

宿主在真实新 Core boot 的 `load_all` 边界先读取有效 selection，再按该 workspace
的固定身份调用 Controller `cleanup_candidates`，最后才允许任何新插件 `apply`。
宿主原子模块逐份确认回执属于同一 workspace 的 candidate，且 `container_absent`
和 `mounts_released` 都成立；任一失败或未知结果阻止 boot，不提前结算恢复记录。
未配置 Controller 的宿主没有该外部清理调用。

```text
┌──────────────────┐   ┌────────────────────────┐   ┌─────────────────────┐
│ 读取有效 selection │ → │ 清理旧候选并确认全部回执 │ → │ 检查 operation 许可 │
└──────────────────┘   └────────────────────────┘   └──────────┬──────────┘
                                                              ▼
                                                       新 Root apply
```

此调用由已有启动 operation 的实际任务持有；取消或截止撤销许可，迟到的成功回执
也不能继续 `apply`。Controller 保留其原有 lease/停止账及未决外部工作，Manager 保留
实际 operation；此边界不重发未知 start，也不把连接取消解释为 Docker 已回滚。
普通 Root 换代、候选子 Manager、Controller 首次连接或首次 start 均不触发清扫。
`owner_container` 可选，进程组 guardian 不负责 Docker，二者不能替代该 boot 清理。
清理仅释放候选容器和挂载，不删除持久业务数据，也不改变实际容器协议。

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

验证遵从本次授权：只静态阅读与 `git diff --check`。测试源码覆盖 Scope 预登记、真实句柄
依赖、未知请求保留、跨 Root/owner、候选环境、每调用 MCP、EOF/进程组关闭和发行安装
fixtures；未执行 tests、Gate、CI、build、lint、AST 或产品运行，不能据此声称运行验收通过。
