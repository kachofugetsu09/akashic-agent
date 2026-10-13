# Web Shell HTTP 复用爬坡

当前 Web Shell 每次请求创建并关闭一个 HTTPX 客户端，无法复用 UDS 连接。
候选由 Web Shell 生命周期持有池；键使用现有 Endpoint（含 owner、generation 与地址）。
单个响应只拥有自己的借用，旧端点撤下后排空在途响应再关闭旧池。
Cookie 不存入共享客户端；鉴权和请求内容逐次透传，不增加自动重试。

```text
浏览器请求 ── Web Shell ── 当前 Endpoint 连接池 ── 实际 UDS listener
                  │              └─ 单个响应结束：归还借用
                  └─ 端点撤下：后续路由不再选择，旧响应排空后关闭
```

在首次运行前固定标准：

- 使用 `scripts/web_shell_http_scenario.py` 的真实 TCP→UDS 链路；每层并发 1/10，各 5 批 × 100 个请求。
  对原始、候选、原始复核分别记录源码摘要、墙钟、进程 CPU、每批请求中位数/p95、事件循环最大调度延迟和实际 UDS 建连次数。
- 候选的建连数至少降低 90%，并发档位的批次墙钟与 CPU 中位数至少降低 15%；失败请求不能从分母中移除。
- 独立验证 Cookie/Authorization 隔离、101 条长流期间的短请求、响应头之前的取消、POST 已接纳后的断线不重发、同 socket 路径换代、旧流排空、计划撤下和停机释放。
- 完整 Runtime 的浏览器筛查沿用连接基准：每档 3 对冷/热样本，固定同一插件实例、workspace、聊天构建和目录摘要，仅替换 Shell。
  所有样本通过，各档 ready 中位数不退化超过 `max(100ms, 10%)`。该小样本只作回归筛查，不声称生产 p95。
- 用 `serve-shell.py --source ... --workspace ... --port ...` 在同一端口依次运行原始和候选 Shell。
  Runtime 自身保持运行；A/B/A 期间不换代插件、不修改正式 workspace。

连接池的并发上限保持原行为，不让 SSE 占满新引入的总配额；只限制每端点闲置连接数量。
线程数不增加。方法遵循 [HTTPX 的客户端复用与响应释放合同](https://www.python-httpx.org/async/)。

首次预检发现夹具用 `proto=0` 创建 TCP listener，使 asyncio 跳过 `TCP_NODELAY`。
系统调用记录显示响应正文被延迟约 40ms；正式单进程 Uvicorn 的 host/port 路径使用 `IPPROTO_TCP`。
修正夹具后重新执行全部 A/B/A，接受阈值和生产候选未变；此前记录保留为预检，不能混入最终结果。

串行完整页面筛查中，候选 18/18 功能通过，但 5/6 档耗时超过门槛；同期其它任务约占 660% CPU。
因此继续保留串行失败结果，补做每档 3 对冷/热的相邻交错 A/B，逐轮旋转档位与 A/B 顺序。
Runtime、目录、页面、浏览器和 CPU 亲和性保持不变；只切换 Shell，记录每对前后负载。
补测仍要求 36/36 功能通过，且各档相邻 ready 差值中位数不退化超过 `max(100ms, 基线中位数 10%)`。
补测前冻结上述判定；不根据个别样本重跑或筛掉失败。驱动与报告随原始结果保存在 `compare-http-browser.py` 与 `http-paired/`。

最终测量和已知限制见 [HTTP_POOL_RESULTS.md](HTTP_POOL_RESULTS.md)。
