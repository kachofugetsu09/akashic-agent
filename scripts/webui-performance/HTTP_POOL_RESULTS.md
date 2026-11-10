# Web Shell 连接池爬坡结果

2026-10-14；冻结方法与阈值见 [HTTP_POOL.md](HTTP_POOL.md)。

原版、候选、原版复核各 1000 个真实 TCP→UDS 请求，功能检查全部通过。
候选实际建连 9 次，两个原版各 1000 次，减少 99.1%。

| 并发 | 原版 / 候选 / 复核批次耗时 | 原版 / 候选 / 复核进程 CPU |
|---|---:|---:|
| 1 | 1295 ms / 354 ms / 999 ms | 1220 ms / 303 ms / 917 ms |
| 10 | 1406 ms / 493 ms / 1164 ms | 1310 ms / 478 ms / 1058 ms |

以上为每档 5 批、每批 100 次的中位数。对两次原版分别比较，耗时降低 58%–73%，CPU 降低 55%–75%。
这些是隔离代理链路测量，不是完整页面提速百分比；宿主有其它工作，不能外推生产 p95。

通过的真实故障与边界验收：

- 三个浏览器身份的 Cookie/Authorization 隔离，上游 Cookie 不写入共享 jar。
- 101 条长流期间短请求仍完成；补强验收确认全部 101 条上游流结束，响应头前取消可恢复。
- POST 已接纳后强制掉线，返回 503 且仅接纳一次；随后请求恢复。
- 同 UDS 路径换代后新请求到新 listener，旧流排空；计划撤下与停机释放连接。
- 既有 `web_shell_endpoint_scenario.py` 通过真实插件安装、缓存/安全头、WS 换代、损坏计划显式失败、Runtime 停止和完整消息行保留。
- Web Shell 类型检查和插件边界检查通过。

旧 `chat_websocket_proxy_scenario.py` 在夹具初始化时失败：`SessionAdmin` 已是 Protocol；候选与原版均复现，本次不修改该无关夹具。
完整 Runtime 的串行 A/B/A 共 54/54 功能通过，但候选 5/6 档的耗时超过首次基线门槛；切回原版也整体变慢。
保留这次失败筛查，并按预先补充的方法交错 A/B：36/36 功能通过，全部 6 档的相邻差值满足原耗时门槛。

| 网络 / 缓存 | 相邻 ready 差值中位数 | 相邻百分比中位数 |
|---|---:|---:|
| lan/cold | -290 ms | -5.9% |
| lan/warm | -158 ms | -6.8% |
| latency/cold | -124 ms | -2.5% |
| latency/warm | -13 ms | -0.3% |
| limited/cold | -126 ms | -1.8% |
| limited/warm | -144 ms | -4.8% |

只有每档 3 对；这是受宿主负载影响的回归筛查，不能称为稳定的生产页面提速。
两组浏览器实验的实际目录、聊天与首页入口、浏览器、harness 和锁文件摘要一致；原始报告分别在 `http-browser-*` 和 `http-paired/`。

恢复点：父提交 `d4715e4e`；未部署，正式 workspace 未写入。
原始数据：`/mnt/data/coding/benchmark-data/connections-climb-20261013/` 下的 `http-final-{before,after,control}.json`、`http-experiment.json` 和各验收日志。
预检的 TCP listener 配置问题及系统调用记录保留在 `http-trace.log`，不混入最终数据。

隔离 Runtime 收到 SIGINT 后已退出（状态 130），finally 中写出的完整消息行摘要前后相等；Shell 进程均已结束。
换代测试修改的两个隔离插件文件均已与备份逐字节核对恢复。
