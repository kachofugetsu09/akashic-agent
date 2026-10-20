# 会话切换性能基准

测量从短会话切换到长历史会话的端到端延迟,并分解网络、渲染与绘制段。

## 口径

- fixture:`serve-startup-runtime.py --switch-session-messages 2000` 额外灌入
  `akashic:switch-benchmark` 会话(2000 条,markdown 富文本:代码块、列表、链接、
  内联代码),默认不灌入,既有启动基准数据与摘要不受影响。
- 测量:`measure-session-switch.mjs` 每轮新 browser context、412×915 视口、
  CPU×4 节流,先在启动会话就绪(与启动基准同一口径),再 `dispatchEvent("click")`
  点击侧栏目标行——不经过 hover,排除悬停预取,测冷切换上界。
- 计时点:点击 → messages 响应(服务端尾页 50 条)→ 末条消息行出现 → 双 rAF 绘制。
- 运行:

```bash
.venv/bin/python scripts/webui-performance/serve-startup-runtime.py \
  --state /tmp/switch-state --port 24142 --switch-session-messages 2000
node scripts/webui-performance/measure-session-switch.mjs \
  --url http://127.0.0.1:24142 --browser /usr/bin/chromium \
  --output results/switch --runs 5 --profiles lan,limited --label <label>
```

## 基线(origin/main e1628ddc,5 轮中位)

| profile | 启动就绪 | messages 响应 | 行渲染 | 绘制完成 |
|---|---:|---:|---:|---:|
| lan | 1494ms | 231ms | 467ms | 538ms |
| limited | 8417ms | 996ms | 1406ms | 1431ms |

## 构成与已排除的候选

CDP 请求时间线(CDP 的 requestWillBeSent 为准;Playwright 的 request 事件
在节流下到达延迟,不能用来推断请求发出时刻):

1. 点击 → messages 请求 51-280ms,方差大,受主线程长任务调度影响;
   响应本身 LAN 约 6-15ms(read_tail_50 服务端 1.5ms)。limited 下 ~1000ms
   主要是带宽与延迟,真实用户有 hover 预取(`onPrefetchSession`)遮挡一部分。
2. 冷切换(每 context 首次)伴随 plugin-ui/asset ×15 与 plugin-ui/query ×7
   的插件界面扇入;资产 URL 内容寻址 + immutable 年缓存,同页第二次切换消失。
3. 懒加载 chunk `chat-app-message-response-*.js`(markdown 渲染器)出现在
   切换窗口。**优化尝试与证伪**:就绪后 `requestIdleCallback` 预热该模块
   (desktop-chat-app.tsx,已还原未提交)消除了切换窗口内的 chunk 请求,
   但行渲染与绘制耗时不变(5×2 轮,lan 467→467ms、limited 1406→1422ms)——
   chunk 取数与其他工作异步重叠,不在门径上。行渲染段是 CPU 支配的 React
   提交与 markdown 行渲染,常数项无单点杠杆。
4. 热切换(尾页缓存新鲜)不再发 messages 请求;仍有 plugin-ui/query ×5-14
   的逐切换查询风暴,属各插件对会话变更的正当数据查询,合并需要插件边界
   语义审查,不在本基准范围内单方面改动。

## 结论

会话切换在网络层没有可摘的果实:数据请求早且快,尾页缓存对热切换有效。
剩余成本是 CPU 支配的渲染段与插件扇入;前者依赖渲染架构(懒增强视口外行
已存在),后者需要在插件 UI 生命周期层面设计(如切换时保持挂载、查询合并),
属于设计议题而非单点优化。
