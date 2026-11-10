# 流式回复渲染基准

从按下回车到回复落定富渲染的端到端测量：harness 构建的聊天页 + 夹具 HTTP/WS 服务端按固定节奏推流式预览，浏览器侧记录长任务、帧间隔与关键里程碑。回答切换基准（[SWITCH.md](SWITCH.md)）覆盖不到的「生成中」路径：逐次增量重渲、代码块在流式期间闭合后随即走高亮链、草稿到正式行的换手。

## 固定输入

- 夹具会话：30 对历史问答（每条回复带一个 ```python 围栏块），回合正文 3183 字符、8 个 ```python 块。
- 推流节奏：25ms 一跳、每跳 15 字符（≈600 字/秒，快模型稳态），推完追加正式行并撤下草稿。
- Chromium 无头、1365×860 视口、CDP 网络 20ms/12.5MBps、CPU 降速 4 倍；每轮新 browser context。
- 发送后先 dwell 6 秒再发：挂载期的空闲预热链（rIC + 3s timeout 兜底）若与出字窗口重叠，两组都会把预热求值算进草稿延迟。

## 测什么

| 指标 | 定义 |
|---|---|
| `activityMs` | 夹具收到 `message.send` → 草稿行（`.reply-activity`）出现 |
| `draftMs` | 同上 → 草稿开始出字（放字节奏自身的缓冲含在内） |
| `settleRowsMs` | 同上 → 含完整正文的正式行出现且草稿撤下 |
| `settleRichMs` | 同上 → 正式行 8 个代码块全部进入高亮路径 |
| `longTaskCount/TotalMs/WorstMs` | 发送→落定富渲染窗口内的 longtask 统计 |
| `frameP95Ms/WorstMs` | 同窗口 rAF 帧间隔 |

发送时刻取夹具收到 `message.send` 的 `Date.now()`（与页面同机同时钟），**不是**页面侧按下回车的时刻：CPU×4 节流下 Playwright 的 fill/Enter 自身有数百毫秒的 CDP 调度开销，从页面侧计时会把它误算成应用延迟（实测 activityMs 虚高约 600ms）。同一起因在切换基准里表现为「点击→请求 320-700ms」。

## 运行

```
node scripts/webui-performance/measure-stream-reply.mjs --browser <chromium> --runs 5 --output <dir> --label main
# 测变体源码（另一 worktree）：
node scripts/webui-performance/measure-stream-reply.mjs --source /path/to/worktree --label variant
```

每轮把 CDP Profiler 样本写到 `<dir>/<label>-run<N>.profile.json`，可用 `dbg-map-profile.mjs` 归因（需带 sourcemap 的构建）。

## 基线（main @ e4ba3e29，2026-10-21，5 轮中位）

| 指标 | 中位 |
|---|---|
| activityMs | 408 |
| draftMs | 493 |
| settleRowsMs | 5726（流式时长 5.3s，落定尾延 ~0.4s） |
| settleRichMs | 5823 |
| longTask | 4 个 / 共 444ms / 最大 223ms |
| frameP95Ms | 68 |

归因要点（CPU profile）：流式窗口 JS self time 前三为 markstream 主渲染函数（逐次增量重渲）、stream-diffs `getContentHeight`、高亮 Worker 回包主线程处理；native 大头是代码块闭合后逐块走高亮链引发的样式/布局读取。修改前先看 [#1581](https://github.com/kachofugetsu09/akashic-agent/issues/1581) 的已否定方向（逐行串行升级、隐藏渲染预热等）。
