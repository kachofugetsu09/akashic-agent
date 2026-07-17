# 电脑端已不存在会话的本地收口

## 问题与边界

手机会长期保存已同步的对话投影。电脑端删除会话或更换 workspace 后，旧会话仍可能留在手机上；如果客户端继续发送，旧实现会得到“会话不可用”，更危险的路径还可能重新创建已经删除的会话。

本组不在 Agent 核心增加移动端 tombstone。Android 用持久化的 `remoteKnown` 记录“这段会话曾被服务端确认存在”，再用当前连接完整的 `session.list` 判断它是否已经从电脑端消失：

1. `remoteKnown = true`，且当前服务端目录不再包含该会话时，立即停止消息、附件和插件 Turn 请求；
2. 有 pending、retryable、outcome unknown 消息或未发送附件时，历史和本地工作全部保留，只允许新建聊天；
3. 没有本地工作和运行中 Turn 时，允许从手机移除本地投影；
4. 本机新建、尚未被服务端确认的会话不参与缺失判断；重连尚未取得完整目录时也不推断。

## Task-first 交互

```text
完整 session.list
        │
        ├─ 服务端仍存在 ───────────────→ 正常聊天
        │
        └─ remoteKnown 且目录缺失
                    │
          ┌─────────┴─────────┐
          │                   │
       有本地工作          无本地工作
          │                   │
┌─────────▼──────────┐  ┌─────▼────────────┐
│ 历史与草稿保持可读  │  │ 历史保持可读      │
│ 已停止发送          │  │ 从本机移除        │
│ 新聊天              │  │ Material 3 确认层 │
└────────────────────┘  └──────────────────┘
```

- 抽屉保持平面列表，只用 warning 文案替换异常会话的普通预览，不增加卡片、badge 或发光状态点。
- 底部状态面原位接管 composer 的任务位置。有本地工作时给出“新聊天”；只有可安全删除时才出现“从本机移除”。
- warning 暖橙只表达远端目录缺失，主蓝只表达普通导航和下一步操作，error 只用于删除确认。
- 删除确认复用既有 Radix Dialog，使用 Material 3 `28dp` modal、32% scrim、48dp 文字动作和单层容器，没有手搓第二套焦点管理。
- warning token 为 `oklch(0.54 0.14 72)`，在 `surface-container-low` 上约 `4.55:1`。正文继续沿用现有中文系统字体和舒展行高，没有用胶囊或等宽字体制造层级。
- 已失效会话不再挂载本轮插件 slot。插件目录和全局面板仍可用，但 Observe 等 Turn 级 UI 不会拿不存在的服务端 owner 发请求，也不会渲染误导性的红色错误。

## 实现所有权

- Room schema v6 由 `ConversationEntity.remoteKnown` 持久拥有服务端身份；5→6 migration 用既有 `serverSeq`，以及可证明已进入实时链路的 assistant streaming/complete/interrupted、user sent 投影回填，pending/failed 本地工作不会被误标。此后由 `session.list`、history 和实时 session/turn/message 事件显式确认。
- `ConversationSummary` 同时暴露 `remoteKnown` 和 `hasLocalWork`。前者决定是否缺失，后者只决定能否删除；两者不再混为一个 `isAvailable` 条件。
- `RealtimeSession` 在每个连接 generation 内先获取 `session.list` 和所有历史页，完成前不进入 READY；重连后不恢复上传、下载、stop 和 outbox 队列，READY 前不 flush outbox。目录同步失败直接重连，不拿旧目录继续发送。
- send、retry、附件导入/重试、通知快捷回复、Turn 插件 RPC 和 outbox 都在同一个 Android owner 上核对目录。outbox 遇到旧会话时用独立的 pending/retry 状态迁移把该消息标记为可重试并跳过，不借用 in-flight 失败路径，也不阻塞其他会话；下载队列会跳过失效会话，继续处理其他会话的附件。
- WebView snapshot v4 显式携带 `isAvailable` 与 `canRemove`；React 只表达状态、确认和任务动作，不自行推断服务端目录。
- 服务端主要在 mobile 边界保护：已有 mobile claim 但 canonical `SessionManager` 中不存在的会话返回 `session_not_found`。Agent 生命周期只增加可复用的 existing-only admission，不增加移动端 tombstone，也不改变 session 持久化和插件协议。
- 多设备首次看到另一台手机的新会话时，Android 在同一 Room 事务先建立 `remoteKnown` conversation，再应用 `turn.started` 并推进 cursor，避免外键失败后无限重放。发送 ACK、history、实时事件和 `session_not_found` 都由同一 Room owner 确认该持久身份；已进入 mobile channel 的入站消息统一携带“必须已存在”不变量，before-turn 消费后只走 `SessionManager.get_existing`，首次 claim 排队后或检查后并发删除都不会进入创建路径。
- 隔离数据库回滚可能让客户端 `last_ack` 高于服务端 durable cursor。检测到已认证客户端 ACK 高于服务端 cursor 时，Gateway 选择以客户端已应用序号为恢复基准，原子清理回退 inbox、前移 cursor，并在下一个精确序号持久化 `sync.reset_required`；协议与存储边界把恢复 ACK 限在 SQLite 64 位序号空间的一半，为 reset、completed 和后续日常事件保留充足余量。普通 resume 和服务端领先客户端的路径不变。

## 验证

### 自动门禁

- `npm run typecheck`、`npm run lint`：通过。
- `npm run test:mobile-web-state`：20 passed。
- Kill AI Slop 扫描 `frontend/chat/src` 为 38 个文件、10 组、58 个机械命中；本组没有新增渐变、玻璃拟态、发光点、卡片墙或胶囊状态，命中均来自既有组件、圆形图标按钮、代码等宽字体和 `touch-callout` 误报。
- Android `:app:testDebugUnitTest`、`assembleDebug`、`assembleDebugAndroidTest`：通过。
- Pixel 7 Room instrumentation：39 passed，包含 schema 5→6 live-only 回填、首个远端 Turn 建立 conversation、持久化 `remoteKnown` 和本地工作分流。
- Pyright 0 error/0 warning；mobile gateway、channel、storage、protocol 与 lifecycle 定向测试合计 131 passed，包含首次 claim 排队后删除、检查后并发删除、客户端 ACK 超前的原子恢复、进程重启、过期 inbox 组合、最大合法 ACK 完整恢复和 existing-only lifecycle。
- `clients/android/scripts/build-release.sh`：release unit、Lint、R8、assemble 与 APK v2 签名通过；最终 `0.7.11 (20)` APK 为 8,326,778 bytes，SHA-256 `fca7ad58fe9880543116b4e12386b33643445cdc35edc0d127f5c27f015dc000`。

### Pixel 7 / 隔离 Mobile Lab

1. 正式 workspace 全程未写入；隔离 `sessions.db` 与 `mobile_realtime.db` 在破坏性测试前备份到 `/mnt/data/coding/backups/mobile-lab-stale-session-review-fixes-before-20260717-091000/`，最终删除前的已知完好副本另存到 `/mnt/data/coding/backups/mobile-lab-final-stale-before-20260717-094443/`。
2. 签名 release APK 覆盖安装后，真实 Room 5→6 migration 成功；客户端先完成当前 generation 的目录与历史同步，再进入 READY。
3. 隔离服务端删除 `mobile:74cf8e16-a7ab-4077-b58e-b057480b91ac`，同时保留手机上的待发消息和附件。重连期间实际经历 tunnel 502，恢复后客户端没有发送 `message.send`；服务端 `sessions` 与 `messages` 均为 0。
4. 手机历史保持可读，底部显示“电脑端已不存在 / 未发送的消息或附件仍保留在本机；已停止发送，避免重新创建会话 / 新聊天”，截图 `/tmp/pixel7-stale-session-local-work-after-reconnect.png`。
5. 恢复隔离数据库时真实触发“客户端 ACK 高于服务端 durable cursor”；旧循环断线被 `sync.reset_required` 收口，Pixel 7 回到“连接正常”，截图 `/tmp/pixel7-stale-session-final-rebased-healthy.png`。独立复核随后要求把 cursor 前移与 reset 入箱合并为同一事务，并补齐提交后立即进程退出、重启仍只先发送 reset，以及回退事件同时超过保留期的组合回归。
6. 再次删除服务端会话后，最终 APK 冷启动保持连接正常，Turn 级 Observe slot 已隐藏，不再出现“Token 统计不可用：会话不存在”；最终截图 `/tmp/pixel7-stale-session-final-reviewed-v2.png`。随后 Mobile Lab 明确改挂本 worktree 最新代码并重建三个容器，Pixel 7 再次冷启动仍保持同一状态，截图 `/tmp/pixel7-stale-session-latest-server.png`；容器内外 gateway/storage SHA-256 一致，服务端再次读回 `sessions=0 / messages=0`。应用 PID 与容器日志无 FATAL、RenderProcessGone、event gap、协议校验、ASGI 异常或旧会话 `message.send`。
7. 无本地工作的另一条隔离会话完成“抽屉提示 → 保留历史 → Material 3 确认 → 从本机移除 → 自动切换 → 服务端恢复后重新同步”闭环，证据为 `/tmp/pixel7-stale-drawer-detected.png`、`/tmp/pixel7-stale-session-dialog-final.png`、`/tmp/pixel7-stale-session-drawer-after-remove2.png` 和 `/tmp/pixel7-stale-session-restored-server.png`。

## 五项 UI 审阅结论

- Better UI：状态面放在原 composer 任务位置，动作随可恢复性变化；没有叠加 toast、banner 和卡片。
- Better Colors：主蓝、warning、error 和 Agent 紫保持单一语义，不用默认“彩色状态大全”。
- Better Typography：标题、解释和动作三层足以完成决策；数字与代码不存在时不强加 tabular 或 monospace。
- Material 3：复用 dialog、state layer、48dp action 和既有 surface 层级，返回与焦点语义由成熟组件承担。
- Kill AI Slop：没有新增渐变、玻璃拟态、发光点、圆角卡片墙、图标彩色方块或胶囊堆叠。

状态：独立复核发现的 crash-consistency 与整数边界已修复并重新送审。本组已修复 live-only migration、多设备首事件外键、pending outbox 状态所有权、下载队列阻塞、所有 mobile admission 的 existing-only 约束、claimed-session 并发删除和 durable cursor 回滚边界，并通过自动门禁、隔离弱网重连和 Pixel 7 签名 release 验收。
