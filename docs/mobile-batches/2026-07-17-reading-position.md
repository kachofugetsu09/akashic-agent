# 2026-07-17 移动端阅读位置闭环

本批补完会话进入、历史分页、插件看板返回、输入法开合、流式跟随和进程重启之间的视口所有权。只复用既有 Room 阅读状态与 WebView 消息投影，没有修改消息协议或 Akashic Agent 核心。

## 问题与根因

1. `markReadThrough` 只推进已读时间，没有清除 Room 中的消息锚点；用户明确回到底部后，重启仍会回到旧位置。
2. WebView 曾按消息数组顺序寻找首个可见元素，而 canonical/history 合并后的数组顺序不等于真实 DOM 几何顺序。
3. 锚点只恢复一次；随后到达的历史分页会把视口再次推走，浏览器的 history restoration 也可能覆盖应用恢复结果。
4. 页面底部的视觉状态与 `useStickToBottom` 的状态可能短暂不同，导致真实到达底部却没有清除原生锚点。
5. 投影重建后，服务端可能不再返回旧锚点消息；无限等待旧消息会让该会话停止保存新的阅读位置。

## 视口所有权

```text
┌─ 进入会话
│
├─ 有有效锚点 ── 同一投影代次重复校准 ── 同步完成 ── 320 ms 布局稳定 ── 最终校准
│       │                                                   │
│       └─ touch / wheel ──────────────── 立即交给用户 ─────┤
│                                                           ▼
├─ 旧锚点失效 ─────────────────────── instant 到最新消息 ──┤
│                                                           │
└─ 无锚点 ─────────────────────────── instant 到最新消息 ──┘
                                                            │
                                  用户上滑 ── 保存 DOM 几何锚点
                                                            │
                                  到达底部 ── 清锚点并推进已读
```

- `window.history.scrollRestoration = "manual"` 只在移动聊天应用存活期间生效，卸载时恢复原值。
- Room 是阅读锚点的唯一事实源；插件看板往返、进程死亡和 canonical identity 迁移都读取同一状态，不在 WebView 另存可能过期的副本。
- 恢复期间禁止写回中间位置；稳定后主动派发一次滚动事件，把最终位置交给既有持久化链路。
- 首个可见消息按实际 DOM `rect.top` 选择，不信任上游数组顺序。
- 到底部以真实 `scrollHeight - scrollTop - clientHeight <= 2` 为准，同一会话和已读时间通过幂等 key 去重。
- 只有原生明确结束该投影代次的重同步后，旧锚点仍不存在才回到底部；320ms 只等待 DOM 布局稳定，不猜测网络分页何时完成。
- 原生保存、清除、投影重建和 canonical identity 迁移由 `LocalDeliveryStore` 的唯一 Mutex 串行拥有；所有路径保持 `reading mutex → Room transaction` 的固定锁序。WebView 回调在 UI 分发栈内以 `UNDISPATCHED` 进入 owner，旧消息 ID 的迟到写入通过有界短期 alias 落到 canonical 消息，不持久化第二份映射。
- 普通断线重连即使不增加投影代次，也会在 `resyncing` 开始时撤销已稳定标记；只有当前会话与投影 key 的 timer、animation frame 和 promise 才能完成恢复。
- effect 卸载只取消尚未稳定的 debounce，不再拿旧 session 闭包读取父级共享的新会话 DOM。
- 重同步期间消息投影可能暂时为空；该窗口禁止保存或清除阅读状态，不能把空列表误判为真实会话尾部。

## ExtraGram 与五项 UI 复核

- ExtraGram/Telegram 的 `ChatActivity` 把手动滚动与程序定位分开，并只在明确动作后定位最新消息。本批沿用这条所有权原则，没有照搬视觉组件。
- Better UI：恢复为 instant，不播放无意义的长距离动画；touch/wheel 可中断，操作响应不等待定时器结束。
- Better Colors：没有新增颜色。回到底部继续使用既有蓝灰 surface，运行态继续由亮紫承担，避免新增一套滚动状态色。
- Better Typography：没有新增标签、徽章或辅助文案；正文、时间与输入栏继续使用现有系统字体层级。
- Material 3：复用现有 48dp 回到底部圆形动作；阅读位置是行为状态，不新增卡片或浮层。
- Kill AI Slop：本批没有渐变、玻璃拟态、发光点、胶囊提示或装饰性 surface；所有变化都服务于阅读任务。

## 自动验证

- `npm run typecheck`：通过。
- `npm run lint`：通过。
- `npm run test:mobile-web-state`：15 passed。
- Pixel 7 执行完整 `LocalDeliveryStoreTest`：28 passed，覆盖旧消息 ID 的单段与两段迟到写入都解析到最终 canonical 锚点；原始输出保存在 `/tmp/pixel7-local-delivery-store-20260717.txt`。
- 签名 release 构建、R8 和 APK v2 签名验证：通过。

## Pixel 7 隔离真机闭环

设备连接 Docker Mobile Lab，正式 workspace 未参与本批请求。

| 场景 | 结果 | 证据 |
| --- | --- | --- |
| 中段强停并重启 | 首屏恢复到相同消息与偏移 | `/tmp/pixel7-v5-before-restart.png`、`/tmp/pixel7-v5-after-restart.png` |
| 插件目录与 Observe 看板往返 | 返回聊天后保持原阅读位置 | `/tmp/pixel7-v7c-launch.png`、`/tmp/pixel7-v7c-directory.png`、`/tmp/pixel7-v5-dashboard.png`、`/tmp/pixel7-v7c-after-plugin.png` |
| 回到底部后强停并重启 | 仍在最新回答，旧锚点没有复活 | `/tmp/pixel7-v8-tail-before-restart.png`、`/tmp/pixel7-v8-tail-after-restart.png` |
| 输入法开合 | composer 紧贴 IME；关闭后只增加可见内容，不跳历史 | `/tmp/pixel7-v8-ime-open.png`、`/tmp/pixel7-v8-ime-closed.png` |
| 真实模型流式输出 | token 增长时持续跟随尾部 | `/tmp/pixel7-stream-autofollow-1.png` 至 `/tmp/pixel7-stream-autofollow-3.png` |
| 流式期间手动上滑 | 间隔 3 秒的两帧保持相同视口，并显示回到底部动作 | `/tmp/pixel7-stream-manual-hold-1.png`、`/tmp/pixel7-stream-manual-hold-2.png` |
| 同会话清缓存重同步 | 空投影和逐页历史期间不写阅读状态；同步完成后回到 `remoteok12345` 与同一附件附近 | `/tmp/pixel7-reading-resync-v2-before.png`、`/tmp/pixel7-reading-resync-v2-early.png`、`/tmp/pixel7-reading-resync-v2-mid.png`、`/tmp/pixel7-reading-resync-v2-after.png` |
| 分页同步中手动滚动 | touch 后移动到 `unreadgamma` 附近；后续分页与同步完成没有夺回视口 | `/tmp/pixel7-reading-round3-sync-before-touch.png`、`/tmp/pixel7-reading-round3-sync-after-touch.png`、`/tmp/pixel7-reading-round3-sync-settled.png` |
| 普通网络断线重连 | Wi-Fi 关闭、连接重置、恢复同步到连接正常，全程保持同一消息与偏移 | `/tmp/pixel7-reading-final-before-reconnect.png`、`/tmp/pixel7-reading-final-offline.png`、`/tmp/pixel7-reading-final-after-reconnect-ready.png` |
| 最终 release 重启与插件往返 | 中段强停重启保持位置；进入 Observe KV Cache 看板再返回仍保持位置 | `/tmp/pixel7-reading-final-before-restart.png`、`/tmp/pixel7-reading-final-after-restart.png`、`/tmp/pixel7-reading-final-dashboard.png`、`/tmp/pixel7-reading-final-after-plugin.png` |

第一次重同步实测曾暴露空投影在 180ms 后被误判成尾部，修复为 `resyncing` 全程停写后重跑通过。最终签名 APK SHA-256 为 `391fefeec0c4a44dabfc3fdff69093fa055ce5c6e10cb48bfb4c0958b89e987f`；logcat 未出现应用 FATAL、阅读锚点跨会话异常、WebView render crash、event sequence gap 或协议错误。网络断线只暴露真实 `Connection reset` 并成功恢复，流式快照持续交付，回到底部后最终回答可见。
