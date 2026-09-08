# ReAct 链路性能

本次减少消息解码、请求准备和每轮建连接的开销。消息仍耐久追加，工具仍先结算本地 start 再执行，插件归档仍逐次建立独立 Root。极大未压缩历史仍有明显的组装成本；这些测量不能证明所有负载都达到零延迟。

## 范围与 owner

```text
┌───────────────────────────────────┐
│ ChannelInput → Input 耐久提交     │
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────────┐
│ Source 唤醒 → 命令 → 回复读取视图 │
└─────────────────┬─────────────────┘
                  ▼
┌───────────────────────────────────┐
│ ReAct：结算工具 → 顺序准备材料    │◀────────────────┐
│ → 请求投影/容量估算 → 调用账      │                 │
└─────────────────┬─────────────────┘                 │
                  ▼                                   │
┌───────────────────────────────────┐                 │
│ HTTP → provider → SSE → Output   │                 │
└─────────────────┬─────────────────┘                 │
                  ▼                                   │
       ┌──────────┴──────────────┐                     │
       │ 包含工具调用           │                     │
       ▼                        │                     │
┌────────────────────────────┐  │                     │
│ 归档 scope → prepare       │  │                     │
│ → 授权 → start → invoke    │  │                     │
│ → ToolResult 耐久提交       │──┼─────────────────────┘
└────────────────────────────┘  │ 完成
                               ▼
                  ┌───────────────────────────┐
                  │ Delivery 固定选路/回执    │
                  │ → 归档 sender → 发送入口  │
                  └───────────────────────────┘
```

| 组件 | 观察与处理 | 保留的边界 |
|---|---|---|
| ChannelInput / Source | 唤醒只读最新同来源 Input 及后续消息 | 输入先落盘，来源与作者不从 Session 名称猜测 |
| Conversation / MessageReader | 一次回复共用增量读取视图；不同 reader 用完整 SQLite 行键共享弱引用解码结果 | 每次仍读数据库；外部管理改写、删除及事务回滚不能被旧缓存遮住 |
| 附件读取 | 一次 SQL 读取一组 Message 的引用，避免按历史逐条重新解码 | Session 归属、缺失消息与附件顺序仍校验，不取得新写权限 |
| Turn / Tools menu | 继续处理完整事实视图 | 不改变工作单元、工具发现、历史 binding 和 abandon 语义 |
| Context materials | 继续按已固定的 contributor 顺序运行 | Akasha 的锁、图提交、取消排空和材料因果顺序不变 |
| Models projection | 批量读取调用账窄字段；重新渲染后，相同行复用上一份冻结表示 | 每轮仍检查真实调用状态并执行动态内容 renderer、工具名称查询 |
| ModelRequest / 账本 | 已深冻结的值直接复用；请求摘要直接编码已验证 JSON | 可变 Mapping 仍复制；外部输入不能借 MappingProxyType 绕过深冻结 |
| 模型 HTTP driver | 三个内置 driver 在 execution 内共用延迟创建的客户端；合并 system 行不再复制整份正文 | 凭据逐请求读取，Cookie 不跨请求继承，重试与部分流失败语义不变 |
| Tools / Shell | 等待结果与清理从本次调用附近订阅 | 原 CallRef、取消后的真实效果、late result 与进程归属不变 |
| 插件归档 | 最多复用 128 个相同路径、相同源码的编译结果 | 每次仍读源码、执行新模块字典、验证归档、建立/关闭独立 Root |
| Delivery | 收益主要来自上述读取与归档准备 | 目的地选择、prepared 回执、实际发送时机不变 |
| Markdown 后台读取 | 最新 main 的 #572 已提供增量读取及线程转移，本 PR 不重复实现 | 继续等待 project 成功后推进 cursor |

HTTP 客户端由 `DriverConnection.close` 归还。Models 的 chat/embedding scope、设置检查和 seal 都明确处理资源；部分绑定失败也关闭之前已打开的连接。不同 execution 不共用可变客户端状态。SSE 的结束标记只证明模型结果完成，HTTP 正文仍需收尾才能复用 socket；额外尾流最多等 10 ms，超时或传输失败会记录放弃复用，由 response scope 关闭连接，不重放已完成响应。该做法遵循 [HTTPX 的作用域客户端与响应关闭说明](https://www.python-httpx.org/async/)。

增量视图只在自己的连接未处于外部事务时推进缓存；`PRAGMA data_version` 检测其他连接的管理改写，同一个短读取事务固定 head 与该版本。正常 Message writer 在本连接只追加；直接任意 SQL 改写历史不属于这个窄接口。参见 [SQLite data_version](https://sqlite.org/pragma.html#pragma_data_version) 与 [读取隔离](https://www.sqlite.org/isolation.html)。

## 测量方法

脚本位于 [`scripts/react-performance/`](../../scripts/react-performance/)，复用现有 application fixture，调用真实 ChannelInput、Source、Conversation、ReAct、Tools、Models 调用账和 Delivery。外部边界只有本地 SSE provider、两次实际文件工具写入和 fixture sender。可选 `--materials` 载入真实 Akasha/Markdown，embedding 由固定二维 fixture 提供。

测量起点为 ChannelInput 入口，终点为 sender 入口。分别记录输入/输出/工具结果提交、provider 进出、工具进出、HTTP 收到完整请求、CPU 时间和实际消息解码次数。启动追赶不计入；线程任务有明确的完成屏障，避免把尚未完成的后台读取当成稳定状态。计时不包括手机/Web 网络上传，也不包括 sender 入口之后的外部送达。

每次运行只有一个 Session、三轮模型响应、两次实际工具效果和一次发送；脚本检查实际 Message 顺序与效果数量。历史交替包含 512 字符 Input/Output，历史 Output 有对应成功模型调用账。全部历史不压缩，容量上限仅在一次性 fixture 放大，用于观察最坏的大请求成本。每份约 11.37 MB 的 20,000 条历史请求都真实经过 HTTP 编码、发送和 SSE 接收。

curl 使用一个进程依次重放捕获的三份请求，每轮实验重放四次。脚本逐字节断言重放正文一致，另外核对跨版本的请求摘要与精确 HTTP 正文摘要。curl 从已编码文件发请求，因此它刻意不支付历史读取、材料、组装和持久回执成本；两者差值就是本次需要解释的 harness 开销。计时采用交替先后顺序，普通 HTTP 场景各两次；包含记忆的压力场景各一次，不宣称 p95。

`--materials` 使用新建的记忆图并在已种入历史的 head 建立 cutover，随后学习本次新输出。它覆盖真实材料与消费路径，但不代表已有多年图状态、真实 embedding provider 或已安装外部 MCP 的延迟。没有改动 Akasha 算法或锁来换取速度。

## 最终结果

基线为最新 `main` 的 `fbcbc35f`。原始分段数据与请求摘要保存在 [`results.jsonl`](../../scripts/react-performance/results.jsonl)。

| 未压缩历史 | 整段链路：main → 本 PR | 两次工具轮次间隙的中位数：main → 本 PR | curl 三请求中位数 |
|---|---:|---:|---:|
| 0 条 | 169.0 → 113.0 ms | 44.0 → 27.7 ms | 10.7 ms |
| 1,000 条 | 1346.3 → 248.1 ms | 318.8 → 63.9 ms | 23.2 ms |
| 20,000 条 | 26860.3 → 2847.2 ms | 7612.4 → 651.8 ms | 207.0 ms |

20,000 条历史的整段链路约减少 89%。新输入到达前，后台消费者已持有历史 Message；弱引用允许复用这些对象，所以普通 HTTP 场景的测量区间只新增解码 8 条消息。这不代表从冷数据库读入 20,000 条消息只需要解码 8 条。

| 真实 Akasha/Markdown，单次样本 | 整段链路：main → 本 PR | 本 PR 的两次轮次间隙 |
|---|---:|---:|
| 1,000 条历史 | 2611.6 → 852.5 ms | 192.3 / 176.3 ms |
| 20,000 条历史 | 35427.6 → 5307.2 ms | 1254.3 / 1253.4 ms |

**达到的范围：** 短历史中，本地工具轮次间隙约 27–34 ms，1,000 条历史约 63–66 ms；20,000 条完整历史及记忆材料仍明显高于 curl，不能称为无感。剩余时间包含全事实投影、实时账本检查、token 估算、JSON 编码和 Akasha 材料处理。

上述小样本只说明这台机器、该 fixture 的实测分布。短上下文的工具动作本身通常低于 0.2 ms；其余时间主要是耐久回执、独立归档 scope 和下一次请求准备。把这些 owner 的提交或关闭移到后台，或共享原本独立的 Root，会改变故障与隔离语义，本次保留这些成本。

### 实际 provider

使用用户授权的 CommandCode `deepseek/deepseek-v4-flash`，每轮最多输出 256 token，依次调用工具两次后结束。共使用三组小实验，每组 harness 三次请求和 curl 三次精确正文重放，总计 18 次 API 请求；之后只用本地 fixture。

修复后的实测只发生第一次 TLS 握手；第二、三轮没有新的 TCP/TLS 建连。两次 provider 返回至下一次 provider 调用为 38.955 / 34.386 ms，工具结果提交至下一次调用为 5.584 / 5.796 ms，最终返回至 sender 入口为 25.338 ms。该实测发生在公共 HTTP helper 提取之前；最终三个 driver 的 socket、凭据与关闭行为另有本地协议回归覆盖。

这组 harness 总耗时 7.543 秒，curl 为 4.198 秒，但不能把约 3.35 秒直接归因于 harness：第一轮 HTTP 请求发完至响应头，harness 约 3.387 秒，curl 的首字节时间约 1.098 秒；生成的 reasoning/output token 数也不同。只用分段本地时间判断 harness 开销，不能据此宣称端到端总耗时与 curl 相等。

## 复现与验证

在项目既有 Python 环境运行，无需真实 API key：

```bash
.venv/bin/python scripts/react-performance/benchmark.py \
  --history 1000 --output /tmp/react-1000.jsonl
.venv/bin/python scripts/react-performance/benchmark.py \
  --history 20000 --materials --output /tmp/react-materials.jsonl
.venv/bin/python scripts/react-performance/benchmark.py \
  --history 20000 --profile /tmp/react.prof --output /tmp/react-profile.jsonl
```

输出路径必须是新路径。性能对照不与 pytest 或类型检查同时运行。`--profile` 只剖析测量区间，cProfile 会放大 CPU 时间；不能将该值混入普通计时表。设置 `AKASHIC_PERF_ROOT` 可从另一份只读 checkout 导入代码；测旧版 driver 必须显式添加 `--legacy-driver`，不会静默替换失败的实际 driver。

相关组合曾通过 245 项测试。接到最新 main 并补上 Cookie 隔离后，重跑 MessageReader、Tools abandon、Markdown 和三个模型 driver 的相关组合，73 项通过。覆盖外部数据库改写/删除/回滚、增量快照一致性、非合作工具的迟到结果、动态请求内容、同尺寸同时间戳源码更新、真实 socket 复用、凭据轮换、响应尾流以及部分打开失败/取消时的资源关闭。模型发现的成功与失败路径也检查临时客户端关闭。

17 个改动生产文件的类型检查为 0 errors / 111 warnings；相同范围的 main 为 0 errors / 114 warnings，没有新增诊断。另用 250 组包含工具、图片、多 system 行及转义文本的输入核对请求 JSON 与 token 估算一致。按用户要求未运行 CI 或 Gate，也未启用合并。

没有正式 workspace 写入、schema 迁移、插件 cache 编辑、部署或合并。缓存只影响临时对象，不减少持久事实。源码恢复点为 `backup/react-performance-before-delivery-20260909`（`fbcbc35f`）及 `/tmp/akashic-incremental-20260909/` 内按修改命名的备份。恢复代码可以逐个 revert 本 PR 的提交；无需逆向迁移用户数据。
