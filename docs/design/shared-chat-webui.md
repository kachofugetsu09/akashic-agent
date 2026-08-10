# 共享对话 WebUI 试点设计

- 状态：implemented pilot
- 日期：2026-08-01
- 决策：[0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)
- 关联条款：WEBUI-001～WEBUI-007、MOB-001、TST-007～TST-008
- 视觉系统：[0023](../decisions/0023-akashic-tokens-own-material-3-semantics.md)

## 1. 用户意图

两端采用移动端现有的浅蓝主题与界面质感，同时保留桌面 Web Chat 更自然的流式正文生长。以后对话前端只在 `akasic-agent` 修改；Android 仍保留比浏览器更丰富的原生能力，桌面仍可提供扫码配对等仅 Web 能力。

## 2. 当前事实与边界

- **F：** 两端都使用 React、Vite、`ChatMessageView` 和 `MessageResponse`。
- **F：** Android 通过 `WebViewAssetLoader` 加载 APK 内静态资产，以 Native bridge 发送完整 snapshot 和 streaming patch。
- **F：** Android 原生层拥有 Room、outbox、附件传输、通知、Keystore、相机扫码和生命周期。
- **C：** 共享 WebUI 源码迁入本仓库；视觉采用移动端色调，流式正文使用 Web 端呈现路径。
- **C：** 桌面扫码配对继续存在，移动端不需要伪装支持它。
- **U：** 本试点不定义 iOS 容器、远程动态下发 WebUI 或线上灰度更新协议。

## 3. 目标结构

```text
┌──────────────────────── akasic-agent ─────────────────────────┐
│ frontend/theme                                               │
│ ├─ theme-catalog.json       Akashic Material 与领域颜色目录   │
│ ├─ material-tokens.css      共享形状、排版、间距和动效 token  │
│ └─ material-react.tsx       Material Web 的 React 适配器      │
│ frontend/chat                                                │
│ ├─ theme.css                共享 WebUI token 入口             │
│ ├─ message-view.tsx          共享消息、工具、流式正文          │
│ ├─ message-view.css          共享消息、工具与引用视觉          │
│ ├─ message-actions.tsx       共享引用、复制与引用预览          │
│ ├─ conversation-navigation.* 共享功能入口、会话与底部操作      │
│ ├─ main.tsx                  桌面适配器 + QR 配对能力          │
│ └─ mobile-native.tsx         Android 适配器 + Native bridge   │
└───────────────┬──────────────────────────────┬─────────────────┘
                │ desktop Vite build           │ clean commit build
                ▼                              ▼
       ┌─────────────────┐          ┌──────────────────────────┐
       │ static/chat     │          │ akashic-mobile-web.zip   │
       │ HTTP + WebSocket│          │ manifest + SHA-256       │
       └─────────────────┘          └────────────┬─────────────┘
                                                │ pinned consumer
                                                ▼
                                   ┌──────────────────────────┐
                                   │ akashic-mobile           │
                                   │ Gradle verify + unzip    │
                                   │ WebViewAssetLoader       │
                                   └──────────────────────────┘
```

## 4. 能力矩阵

| 能力 | 共享 WebUI | 桌面适配器 | Android 适配器 / 原生层 |
|---|---|---|---|
| 主题、消息、Markdown、工具轨迹 | 拥有 | 使用 | 使用 |
| 流式正文生长 | `isAnimating` 驱动 | WebSocket delta | rAF 合并的 native patch |
| 会话侧栏、引用、复制 | 拥有 | 使用 | 使用 |
| 知识与插件入口 | 共享导航结构 | 跳转 Dashboard 公网端口 | 打开 Native bridge 页面 |
| 新聊天 | 共享导航结构 | Web session | Native bridge session |
| 扫码配对展示 | 复用视觉组件 | 生成 QR、确认设备 | 不挂载 |
| 相机扫码 | 无 | 无 | 原生 CameraX / ZXing |
| 设置、诊断、清理同步、重新扫码 | 无 | 不挂载 | Native bridge 拥有 |
| 离线队列、重试、阅读位置 | 只展示已验证状态 | 无 | Room 与 Native bridge 拥有 |
| 通知、分享、Keystore | 无 | 无 | Android 原生拥有 |

## 5. 性能合同

1. 历史消息保持 `content-visibility: auto`，streaming 行不启用该隔离，避免正在增长的消息高度估算错误。
2. Native patch 继续按 `requestAnimationFrame` 合并；React 消息行继续 memo，未变化历史行不重渲染。
3. `MessageResponse` 只在正文或 `isAnimating` 改变时更新；流式结束后保留最终静态 Markdown。
4. 不为动效新增依赖；交互状态使用可中断 transition，`prefers-reduced-motion` 保留无动画路径。
5. 产物按构建入口分离，桌面不会加载 Android bridge、Room 投影或移动插件目录代码。

## 6. 产物、失败和回滚

- `npm run package:mobile-web` 只接受干净 Git tree，ZIP 内写入 source repository、commit、tree 和资产摘要。
- Android 在解包前核对外部 SHA-256；不匹配时 Gradle 失败，不使用旧缓存或网络 fallback。
- WebUI 构建失败不会改变移动端原生状态；产物升级只替换 APK 构建输入。
- 回滚主仓库到上一个 WebUI commit并重新打包；移动仓库恢复上一个 ZIP、摘要和 source lock。两边都不需要迁移数据库或 workspace。

## 7. 试点验收

- 主仓库：typecheck、chat build、mobile web build、mobile state tests、lint。
- 视觉：桌面和 mobile showcase 同时渲染，核对主题 token、布局、流式状态与 reduced motion。
- 离线共享组件验收：打开 `/?preview=chat`，直接观察生产 `ChatMessageView` 的 thinking、工具开始/完成和正文生长；该入口不连接 Runtime、不读取正式会话。
- 移动仓库：ZIP 正向校验、篡改失败测试、Gradle debug build。
- 报告两仓库 commit/tree、ZIP digest；真机 WebView、内存、掉帧和冷启动单独列为未验证或设备证据。
