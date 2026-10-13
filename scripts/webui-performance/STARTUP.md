# 首页启动基准

本基准从首页经过真实插件目录、聊天 iframe、历史 API 和 WebSocket，测量高延迟链路上的启动依赖。它运行完整本地 Runtime，使用一次性配置、插件副本和 100 条固定消息；不配置模型或外部账号，不发送消息。因此“可用”指历史可读、草稿可编辑且已收到实时数据，不代表模型回复完成。

```text
生产构建 → 隔离 Runtime / Web Shell → Chromium 网络与 CPU 限速
                                       ├─ 冷缓存首次进入
                                       └─ 同一 context 的热缓存再次进入
```

## 固定输入

- 基线取最新目标分支的固定 SHA；本轮为 `91275900429e472664e4f7af6bbb4d5e8f4f6a2d`。
- 使用 `npm ci` 的锁定依赖、同一个 Chromium 可执行文件和 412×915 viewport；CPU 降速倍率固定为 4。
- Session 固定为 `akashic:startup-benchmark`，消息 ID `startup-000` 到 `startup-099`，正文来自 `serve-startup-runtime.py`。整个爬坡过程复用同一个服务、数据和插件目录；退出时比较完整消息行 SHA-256。
- Computer 依赖独立 Workload Controller，本基准显式停用它。其余按默认本地组合加载，不用假的 API 成功响应。
- runtime 开始后停止源码 watcher，避免构建触发换代；只在浏览器采样间替换本次隔离目录中的 dashboard 构建。业务插件与聊天构建保持不变。
- 静态制品的 mtime 固定为 `1767225600`（2026-01-01 UTC）；否则浏览器的启发式缓存可能让刚构建的候选多出 304 往返。每次替换候选后都重新固定这项输入；此设置只适用于基准副本。
- 网络参数使用 CDP：`lan` 为 20ms、100Mbps；`latency` 为 600ms、100Mbps；`limited` 为 300ms、4Mbps。上行分别为 100、10、1Mbps。参数表达 CDP 的延迟和吞吐上限，不模拟丢包、运营商路由或 Cloudflare。
- 源站直接提供未压缩内容。这能隔离依赖等待，但绝对耗时不能当作公网预测。真实公网只作为旁证，不参与接受或拒绝候选的门槛。

## 测什么

| 指标 | 定义 |
|---|---|
| `readyMs`（主指标） | 从开始导航到尾页最后一条消息正文可见、输入框写入并读回草稿、WebSocket 已收到数据；不发送消息 |
| `fcpMs` | 顶层和聊天 frame 中最早 FCP，按各自 timeOrigin 换算；不把“有加载提示”等同于可用 |
| 成功率 | 每个计划样本进入分母；45 秒未可用、脚本异常或交互失败单列，不能只统计成功样本 |
| `transferBytes` | 到可用时已完成资源的 transferSize 总和，不包含仍在下载的字节 |
| 辅助证据 | 每个 frame 的资源瀑布、长任务、请求失败、WebSocket 数量与首次数据、截图 |

冷样本使用新 context；热样本在同一 context 内第二次导航。前一次字体加载完成后才开始热样本。正常性能采样不使用 Playwright route，避免拦截自动关闭 HTTP 缓存。各网络档位按轮次旋转顺序，避免一直让某档位先运行。

报告保存 `plannedSamples`。截图、字体等待或浏览器取证失败也记录为该样本失败，关闭页面后继续采样；不能因取证异常丢掉整批结果。整轮验收仍要求计划样本全部成功。

## 爬坡与接受标准（改优化代码前固定）

1. 先用每档 3 对冷/热样本筛查，保留原始 JSON 和失败截图。
2. 每轮只增加一项优化。先对前一轮比较，再对原始基线比较；没有收益的复杂度撤回。
3. 最终对基线和候选各跑每档至少 7 对样本，并补跑原始基线复核漂移。小样本只报告中位数、最大值和完整样本，不包装成稳定 p95。
4. 接受门槛：所有样本功能成功；高延迟冷启动 ready 中位数至少改善 15%；其他档位及热启动退化不超过 `max(100ms, 10%)`。异常和失败不能用更快的成功样本抵消。
5. 只改善 FCP 的加载提示归为反馈改善，不能计为 ready 加速。插件激活顺序、注册截止、CSS 顺序、错误隔离、snapshot 校验和消息内容保持现行合同。
6. 任何比较必须核对 harness、浏览器版本、锁文件、bootstrap 摘要和模块列表一致。报告另保存实际入口脚本摘要，防止误测旧构建。候选 commit/patch 与构建输出一起保留。

## 运行

在独立 worktree 中准备，不使用正式 workspace：

```bash
npm ci
npm run build
.venv/bin/python scripts/webui-performance/serve-startup-runtime.py \
  --state /absolute/new/benchmark-state --port 24139
```

另一个终端运行（输出目录必须尚不存在，父目录已存在）：

```bash
node scripts/webui-performance/measure-startup.mjs \
  --url http://127.0.0.1:24139 \
  --browser /absolute/path/to/chromium \
  --output /absolute/results/baseline --runs 7 --label baseline
```

Runtime 复制插件及前端制品到 `benchmark-state/plugins`。下一轮仅替换 `plugins/ui/static/dashboard`，先保留它的基线副本；不替换权威 workspace 或重新启动 Runtime。测试完对服务发送 SIGINT，检查 `integrity.json` 的 before/after 相同。测试数据与报告保留给评审；脚本不自动删除目录。

替换后的 dashboard 副本用同一时间戳（下例仅指测试目录）：

```python
import os
from pathlib import Path
for path in Path("/absolute/benchmark-state/plugins/ui/static/dashboard").rglob("*"):
    if path.is_file():
        os.utime(path, (1767225600, 1767225600))
```

## 方案依据

- [MDN modulepreload](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Attributes/rel/modulepreload)：可以提前下载和编译模块，执行仍由 import 决定。
- [Vite 异步 chunk 优化](https://v6.vite.dev/guide/features#async-chunk-loading-optimization)：消除依赖发现造成的额外网络往返。运行时 Blob 模块需要单独核对，不能假设构建器已覆盖。
- [web.dev 预加载取舍](https://web.dev/articles/preload-critical-assets)：过量预加载会争抢弱网带宽，所以逐项测量，不默认并发越多越好。
