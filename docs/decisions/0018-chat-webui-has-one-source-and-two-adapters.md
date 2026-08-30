# 0018 · 对话 WebUI 使用一个源码真源和两个平台适配器

- 状态：accepted
- 日期：2026-08-01
- 关联条款：WEBUI-001～WEBUI-003、MOB-001、GOV-001～GOV-005、TST-007～TST-008
- 部分勘误：[0022](0022-mobile-webui-uses-server-selected-generations.md) 将固定 ZIP 收窄为 embedded baseline，并增加服务端选择的不可变 generation
- 扩展决定：[0044](0044-akashic-channel-uses-web-and-mobile-adapters.md) 在本决定的 UI 源码与平台入口边界之上，统一 Web/Mobile 的逻辑 Channel 与 Session 身份
- 桌面入口勘误：[0051](0051-web-ui-composes-ordinary-plugin-modules.md) 将桌面顶层页面注册与 adapter 交给普通 `conversation-ui` 插件；共享消息实现和 Mobile owner 不变

## 背景

桌面 Web Chat 与 Android WebView 已经使用 React、Vite 和同一批消息组件，但源码分别保存在两个仓库。两份实现会独立漂移：移动端拥有更完整的触摸、离线和原生桥交互，桌面端的 WebSocket 增量呈现更自然，主题也不一致。只同步 CSS 或定期复制文件不能给出唯一 owner，也无法从一个 commit 重建两端组合。

## 决定

1. `akasic-agent/frontend/chat` 是共享对话实现的唯一源码真源，同时构建浏览器内容与 Android WebView 入口；桌面顶层页面的注册和 adapter 由 [0051](0051-web-ui-composes-ordinary-plugin-modules.md) 的普通 `conversation-ui` 插件拥有。
2. 共用消息、Markdown、流式呈现和主题 token 只实现一次；桌面扫码配对与 Android 原生桥通过两个显式入口组合。
3. Android 的 Room、outbox、附件、通知、Keystore、配对扫描和生命周期继续由 `akashic-mobile` 原生层拥有。共享 WebUI 只消费经过校验的 snapshot、patch 和命令接口。
4. `akashic-mobile` 不通过本机路径、submodule 或浮动分支读取源码。它保存一个由干净源码 commit 构建的静态 ZIP 作为 embedded baseline，并在 Gradle 解包前验证 SHA-256；ZIP 内 manifest 固定 repository、commit、tree 和资产摘要。已配对运行时还可按 [0022](0022-mobile-webui-uses-server-selected-generations.md) 消费服务端选择的不可变 generation。
5. Web 默认改用移动端现有浅蓝色主题；正文流式呈现由共用 `MessageResponse` 的 `isAnimating` 状态驱动。`prefers-reduced-motion` 继续关闭非必要动画。

## 理由

源码集中可以消除视觉与消息组件漂移；双入口保留平台能力差异，不需要把原生状态搬进浏览器运行时。固定产物让移动仓库在没有父仓库 checkout 时仍可离线构建，并使发布证据绑定不可变组合。

## 影响

- WebUI 改动只在 `akasic-agent` 编写和测试；baseline 变化更新移动仓库的固定 ZIP 与摘要，兼容的运行时 UI 变化也可按 0022 发布服务端 generation。
- Android 原生接口变化仍需先更新移动端桥合同，并通过固定组合验证；本决策不授权修改核心协议或持久状态。
- 移动仓库删除 `frontend/chat` 源码和 Node 构建链，CI 改为校验固定 WebUI 产物并构建 Android。

## 验收

- 主仓库能分别完成桌面与移动 WebUI 构建，移动状态测试通过。
- Web 与移动产物使用同一主题 token 和同一消息呈现组件。
- 移动仓库在不存在主仓库 checkout、没有 `node_modules` 时能验证并解包固定产物，再完成 Android debug 构建。
- 篡改 ZIP 或摘要时构建 fail-loud。
- 两仓库报告包含各自 commit/tree、WebUI source commit/tree 和 ZIP SHA-256。
