# 0097 · Host Bridge 在复用的 Unix socket 上传输 Protobuf

- 状态：proposed；性能实验分支已实现，待维护者评审采用。
- 日期：2026-10-08
- 若采用，supersedes [0055](0055-host-bridge-uses-typed-protobuf.md) 的 gRPC 传输实现与生成 stub 要求；字段和执行语义不变。
- 关联：RUN-013～RUN-015、SH-001～SH-003。

短命令通过 Host Bridge 执行时，Python gRPC 路径的固定开销占比较高。
目标是缩小相对本地 shell 的额外等待，同时保留既有进程、日志与 lease owner。
真实 UDS 测量中，仅调整编码、gRPC 参数或事件循环未获得足够且稳定的收益；
复用 asyncio Unix socket 可以减少 Python 调用边界和响应调度的成本。

候选使用一个复用的 Unix stream socket 承载并发 unary 请求。方法与消息类型由现有 proto
service 唯一声明；帧只携带长度、类型、方法或状态码和连接内调用编号。客户端把响应交给
对应等待者。每个请求仍独立认证与检查实时 lease；token 只在连接握手发送，不进入业务载荷。
执行级环境覆盖只传既有白名单，宿主环境模板在服务启动时建立。诊断事件保留上下文、异常
和顺序，在当前事件循环的下一次调度输出，让响应先进入 socket。

┌──────────────────┐   typed request / reply   ┌──────────────────┐
│ Core Bridge client│ ═══ 复用 Unix socket ═══ │ Bridge service   │
│ 独立等待与取消    │                           │ 认证与 lease     │
└──────────────────┘                           └────────┬─────────┘
                                                       ▼
                                              ┌──────────────────┐
                                              │ ShellProcessManager│
                                              │ 进程与输出的 owner │
                                              └──────────────────┘

取消、deadline 和断线只结束 RPC 等待；已登记进程仍由原 manager 查找与清理。任何请求均不
自动重发。业务背压不能阻塞心跳、探测和停止。连接关闭必须失败所有在途等待，不伪造清理成功。
单条消息仍限制 16 MiB，字段校验、错误分类、boot fencing、完整诊断与文件线程排空继续保留。

代价是项目承担帧、并发和连接清理代码；验收必须覆盖并发响应配对、真实断线、丢失响应、
deadline、取消、畸形输入、同步能力探测和长短命令。保留 gRPC 双协议或另外引入 Go executor
会增加运行与发布 owner；仅为微小常数收益增加第二套输出模型也不合适。若将来需要远程服务、
多语言消费者或标准 gRPC 生态，应重新评估此选择。

此候选 wire 不兼容旧 gRPC；Core 与 Bridge 仍按同 commit 成对发布和恢复。实验不授权部署，
不声称能回滚已执行命令或正式数据。字段和恢复合同见 [协议设计](../design/host-bridge-protocol-v2.md)。
