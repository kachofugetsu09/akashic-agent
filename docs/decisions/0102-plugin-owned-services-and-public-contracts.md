# 0102 · 服务与公共合同由提供方插件拥有

- 状态：accepted target / implementation
- 日期：2026-10-10
- 依据：[#1179](https://github.com/kachofugetsu09/akashic-agent/issues/1179)
- 补充：0065、0082、0094；不改变持久事实与外部回执合同。

## 决定与理由

Core 的目标是组合内核、插件宿主和进程外壳。服务拥有的领域事实、控制流和
生命周期交给提供方插件，公共合同位于该插件的 `contract.py`，没有中央业务
contracts 包。消费者只依赖公共值类型、Protocol 与 ServiceKey，不导入兄弟实现。
同一发行版的内置合同同步版本；公共 key 一次性使用领域名，不保留 `core.*` 别名。

公共值类型在同一进程保持身份，实例实现仍按 Fiber/generation 独立加载、排空和
清理。宿主从已解析的实际源码登记 `plugins.<name>.contract`，不执行包入口，
不复制文件到 Core，不恢复历史接口，不根据插件是否 active 推断类型可用性。
公共 namespace 的搜索目录只来自这些源码，不混入 checkout 的同名 plugins 目录。
消费公共合同需要安装其真实 owner 的源码制品，安装 API 不要求启用该实现；
合法子组合显式准备这些制品，不从 checkout 补接口或自动选择 provider。
源实现可以热换代；公共合同源码改变时明确要求重启，不在同进程混用两套类型。
登记前按名称采用已安装源码，内置同名源码不参与登记；停用实现不撤下已安装 API。
合同更新先完成安装并固定新输入，journal 按 update ID 追加重启要求，再提交下一次
启动的 selection；旧 generation 继续 active，状态查询、停用与卸载照常可用。
更新状态为 `restart_required`，新进程从已提交输入加载后才显示 active，不能热混用
类型或把重启要求报告为安装失败。重启事实不复制输入，不改变安装恢复 phase。
这是 API 版本边界，不是插件沙箱；静态门仍禁止兄弟实现依赖。

UI 查询目录、配额和线程池由 UI 插件的 Effect 拥有；查询只持自己的 Context
和贡献登记，不读取 Root 内部状态。取消尚未进入子任务时释放捕获许可，已运行
线程在物理完成后才释放 scope 和名额；插件清理结束后不残留查询线程。

插件在真实监听器就绪后用 `Context.endpoint` 发布协议、地址与路由前缀。
Root 独占登记，runtime catalog 与 `runtime/endpoints.json` 只读取派生视图；
Core 不根据业务路径推断 owner。计划文件完整写入并替换成功后才提交内存登记，
撤下失败保留 Effect 和监听器，成功撤下后按资源取得的逆序关闭监听器。
该文件不承诺掉电持久或替代外部效果回执，启动从当前选择重建。

Gateway 独占控制帧 route、暂存替代 route 和 ToolCall claim。父 Fiber 发布
`gateway.frames.v1`，其 Effect 在依赖消费者和监听子 Fiber 排空后关闭唯一 FrameBook；
Core 不再构造、提供或关闭它。关闭自动 socket 配置不关闭帧端口；卸载 Gateway
只挂起真实依赖消费者，普通 MessagePush 工具不依赖该端口，重启子 Fiber 才依赖。
公共 Protocol 只给出回执与动作，不返回内部可变 route，不开放 FrameBook 的关闭权。

```text
┌────────────────┐  页面 + writer future  ┌─────────────────────┐
│ Gateway 连接   ├──────────────────────►│ Gateway FrameBook   │
└────────────────┘                       └──────────┬──────────┘
                                                   │ 实际 drain 回执
                             ┌─────────────────────▼───────────┐
                             │ Programmatic / 重启与停止等待   │
                             └─────────────────────────────────┘
```

帧 route 是进程内事实，不写 SQLite，不存入 workspace，不承诺跨 generation 恢复。
断连、卸载和操作取消使等待明确失败；消息正文、身份、顺序与持久 pause 不变。
缺失或结束的 route 使用标准 LookupError；Programmatic 将该等待错误转换为连接失败，
重启 claim 等待转换为显式拒绝，不把未写出的最终 Output 当成送达成功。

Delivery 的发送、回执与历史合同归实际 Delivery owner；默认输入目的地查询
归 DeliveryPolicy。具名 sender 使用冻结 SenderDefinition 发布 owner 与幂等事实，
Wake 按选中渠道名取得动态 key；不公开无类型 object key 或中央 helper。
归档仍保存 name、owner、idempotent 三个原字段，读取边界严格验证后构造该值；
不改写既有 binding，不重新选择旧消息的发送目标。

模型失败使用冻结值保存分类、服务端等待期限、流进展和发送证据，标准
RuntimeError / TimeoutError 只传递该值。消费者显式读取模型失败；普通程序错误
不能取得重试语义，asyncio 取消保留原类型。外层增加诊断或结算事实时构造新值，
不改变 driver 原错误、其他等待者的响应或耐久调用回执。模型控制缺席也由这套
合同表达，HTTP 与 RPC 保留原状态码；不再由 Core 单独定义其异常类型。

Models 公共合同真实拥有请求、响应、失败、usage、continuation、目录、
能力与 driver 协议；Core 不保留模型模块或再导出。消息与模型共用中立 JSON
冻结算法。Models 的 ToolCall 是 provider 返回的请求值；消息 ToolCall 是已提交
调用事实，两者保持独立类型，投影边界显式转换。Tools 自己拥有模型调用的呈现、
菜单与程序消费接口，Core 不为这些消费接口反向依赖 Models。

独立命令由插件用字面 `entrypoints = {"command": "module.function"}` 声明。
安装输入 v6 将该声明纳入唯一 selection；v5 原记录仍原样读取，不改身份，
没有凭空补出的命令入口。Core 只按当前选择分发唯一 provider，不执行包入口或
`apply`，不启动 Root、不争抢运行锁、不打开业务数据库。无关实现损坏不阻断
所选命令，命令实现负责自己的连接、进程与失败回执；模块作用域在退出后释放。

命令来源事先分类，不在缺席时自动回退：`dashboard`、`workload-controller` 固定
使用当前发行版/镜像源码（开发 checkout 使用随 Core 的 `plugins/`）；不依赖
workspace selection，不因此启用任何业务实现。`plugin-install`、`plugin-status`、
`plugin-uninstall` 仍是当前选择中的 Gateway RPC 客户端；`exec`、`app-server`
只从选择记录分发。管理 provider 或已发布端点缺席时，错误给出恢复路径：
用 `plugin-doctor` 查看 Gateway 的完整 ID，执行 `plugin-enable gateway@<marketplace>`，
然后重启实例。裸 `gateway` 不是已安装身份；例如实际 ID 为 `gateway@release` 时
使用 `plugin-enable gateway@release`。Core 不新增管理端口，不静默启用 provider；
彻底独立的管理通道留在边界②讨论。

持久 binding 中的 service 名属于原选择证据，不能随 key 改名原位改写。
存储读取边界解释实际存在的旧 Commands service 名，继续打开当前领域 provider；
这不注册旧运行 key，也不改变 descriptor、hash、binding 身份或原消息。未知版本、
服务不匹配和缺少领域恢复回执仍明确失败，禁止借改名重跑未知外部效果。

```text
┌──────────────────────────┐   ┌──────────────────────────┐
│ 提供方 contract.py        │ ← │ 消费方声明能力依赖       │
│ 进程内固定公共类型身份    │   │ 同一公共类型             │
└────────────┬─────────────┘   └──────────────────────────┘
             ▼
┌──────────────────────────┐
│ 提供方实现 / Fiber       │
│ 按 generation 更新、清理 │
└──────────────────────────┘
```

组合输入采用发行版 base、mode、用户 patch 的顺序；同 row 后写者生效，config 整行
替换，不做深合并或自动 provider 求解。唯一运行选择仍由现有 selection owner 保存。

## 持久化边界

Ledger 插件目标独占现有 `sessions.db`，保留跨消息、接纳、入站、回执、owner_state
和 bindings 的原子事务。权威表不拆库，不改字段，不 DROP 退役表，也不改写既有
Message。嵌入向量等派生数据另存 ledger 的派生库，复制迁移保留原表；派生丢失不
授予权威数据减少权限。新增校验、迁移与恢复必须继续沿原数据 owner 的明确入口。

本记录描述分步实现的目标，不代表所有服务已迁走或验收完成。各层 PR 给出实际
write set、取消、重启、换代证据，最终核对 Core 白名单与禁用、替换场景。
没有操作正式 workspace；源码由 Git 基线恢复，运行数据仍需自己的恢复证据。
