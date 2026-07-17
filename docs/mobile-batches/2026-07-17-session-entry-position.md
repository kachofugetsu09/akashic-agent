# 显式进入会话的阅读位置

## 问题

Android 会为冷启动持久化每个会话的阅读锚点，但抽屉中主动选择一个会话时也恢复该锚点。用户明确进入会话后仍停在旧消息，并需要再次点击“到底部”。

## 交互语义

```text
应用冷启动 ────────────────→ 恢复上次阅读锚点

抽屉主动选择另一个会话 ───→ 打开最新消息
                              └─ 不提前改变未读水位
```

这是导航状态修复，不增加按钮、提示、卡片、颜色或动画。Material 3 的任务优先原则在这里体现为：让现有进入动作产生用户预期的结果，而不是再叠一层补救控件。

## 实施

- Room 只清除目标会话的 `anchorMessageId` 与 `anchorOffsetPx`，保留 `lastReadAt`，因此不会把尚未看到的助手消息伪装成已读。
- `RealtimeSession.selectSession` 仅在用户确实切换到另一个会话时清除锚点；应用恢复当前会话不走该动作，继续保留冷启动恢复。
- WebView 阅读位置 hook 现在接受原生快照把锚点明确清为 `null`。修复前，恢复已开始后只接受另一个非空锚点，导致 Room 已要求到底部，React 仍按旧锚点定位。
- 没有新增协议字段、数据库 migration、Akashic Agent 核心能力或持久化副本。

## 验证

- `npm run test:mobile-web-state`：21 项通过。
- `npm run typecheck`、`npm run lint`、`npm run build:mobile-web`：通过。
- Android debug unit 与 androidTest 编译、release unit/Lint/R8/assemble、v2 签名：通过。
- Pixel 7 安装最终签名 APK，SHA-256 为 `eec2630fc8d19cb9e59eccf2f3c5f6a16b5e022041b67e93e09043573e2562e9`。
- 真机先停在旧锚点并重启，仍恢复原位置；随后从抽屉切到 `cycle2alpha`，再主动进入 `attachment_test_session`，最新一轮完整可见且“到底部”按钮消失。
- 真机证据：`/tmp/pixel7-entry-fix-launch-settled.png`、`/tmp/pixel7-entry-fix-cycle-real.png`、`/tmp/pixel7-entry-fix-explicit-tail-real.png`。
- 最终 logcat 没有 Akashic 应用 FATAL、RenderProcessGone、协议校验或 event sequence 错误。`uiautomator dump` 自身曾因 `Bad file descriptor` 崩溃，进程属于 Android UiAutomation，不属于 Akashic，后续验收改用截图和应用日志。

## 设计复核

- Better UI / Material 3：修正入口行为，不增加补救 UI。
- Better Colors / Better Typography：没有新增视觉语义，因此不引入新 token 或排版层级。
- Kill AI Slop：没有新增卡片、胶囊、阴影、渐变或状态装饰。
