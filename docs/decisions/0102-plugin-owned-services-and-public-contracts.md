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

独立命令由插件用字面 `entrypoints = {"command": "module.function"}` 声明。
安装输入 v6 将该声明纳入唯一 selection；v5 原记录仍原样读取，不改身份，
没有凭空补出的命令入口。Core 只按当前选择分发唯一 provider，不执行包入口或
`apply`，不启动 Root、不争抢运行锁、不打开业务数据库。无关实现损坏不阻断
所选命令，命令实现负责自己的连接、进程与失败回执；模块作用域在退出后释放。

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
