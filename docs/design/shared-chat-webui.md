# 共享对话 WebUI 试点设计

- 状态：implemented pilot
- 日期：2026-08-01
- 决策：[0018](../decisions/0018-chat-webui-has-one-source-and-two-adapters.md)
- 关联条款：WEBUI-001～WEBUI-007、MOB-001、TST-007～TST-008
- 视觉系统：[0043](../decisions/0043-paper-brand-tokens-replace-material-visual-semantics.md)；[纸张品牌系统](akashic-paper-brand-system.md)

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
│ ├─ theme-catalog.json       主题色值与领域状态目录             │
│ ├─ brand-tokens.css         paper / ink / rule / type          │
│ └─ material-tokens.css      迁移期兼容与既有适配器             │
│ frontend/chat                                                │
│ ├─ theme.css                共享 WebUI token 入口             │
│ ├─ message-view.tsx          共享消息、工具、流式正文          │
│ ├─ message-view.css          共享消息、工具与引用视觉          │
│ ├─ message-actions.tsx       共享引用、复制与引用预览          │
│ ├─ conversation-navigation.* 共享功能入口、会话与底部操作      │
│ ├─ main.tsx                  桌面适配器 + QR 配对能力          │
│ ├─ mobile-native.tsx         Mobile React 应用                │
│ └─ mobile-entry.tsx          Android transport + 挂载入口     │
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
| 流式正文生长 | 单消息 rAF 发布器 | WebSocket delta 提交权威目标 | native patch 提交权威目标 |
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
3. 桌面与 Android 的权威增量都先进入单消息展示投影。同一帧内的多次更新合并为最新 target，每帧只通知一次对应消息行；不创建逐字队列，也不扫描或重建稳定历史行。
4. `message.final` 和 Android `streaming=false` 立即显示权威终稿并取消剩余展示帧；已经调度的旧帧不得在 terminal 后再次通知或覆盖终稿。
5. `MessageResponse` 只在正文或 `isAnimating` 改变时更新。流式 Markdown 由 Markstream 的 append-tail parser 接管并复用稳定顶层节点；terminal 使用同一组件完成最终解析，不再维护第二套 block 冻结和未闭合修复补丁。
6. 代码块、数学公式与 Mermaid 在 streaming 阶段保持轻量源码节点，terminal 后交回 Markstream 内建 renderer 完成富化；历史行继续按 viewport 延后富化。
7. 桌面打开会话先按 `seq` 游标读取最新尾页；读取更早页时按稳定消息 identity 恢复阅读锚点。分页只读 SessionDB，不修改、压缩或删除权威消息。
8. 不为动效新增依赖；交互状态使用可中断 transition。产物按构建入口分离，桌面不会加载 Android bridge、Room 投影或移动插件目录代码。

### 5.1 已验证的更新放大故障

两端共享 `Message`、React 组件和最终视觉，只保证展示语义同构，不保证执行成本同构。旧 Android 流式路径在每个 delta 写入 Room 后，使活动会话的完整 `MessageWithBlocks` 查询失效；客户端随后重新物化稳定历史并跨 Native bridge 提交 WebView。桌面 adapter 直接把 WebSocket delta 写入单消息展示投影，不经过 Room 与 bridge，因此相同 TPS 隐藏了完全不同的单次更新成本。旧 Markdown 渲染增加了活动消息的主线程工作，但不是这次全量更新放大的 owner。

高频局部变化经过每个 adapter 后都必须保持局部：正文 delta 不得重新查询、物化、序列化或提交未变化历史；稳定消息保持对象身份；只有 terminal、history heal 或明确的会话切换可以用权威 snapshot 校准展示。性能验收除 TPS 与总耗时外，还要记录每个 delta 触发的查询行数、bridge 字节数、React 通知次数和长任务，避免把“结果一样”误判成“成本一样”。

### 5.2 观测与归因

观测以 `session_id + turn_id + client_message_id` 为主身份，日志只记录阶段、耗时、计数与 outcome，不记录 prompt、正文或工具参数。Provider 原始首块、Core 首增量、Mobile durable inbox、真实 socket、Room、React commit、下一帧与 composer-ready 分层记录，不能用下游首字倒推 Provider TTFT。

```text
用户发送
  │
  ├─ send.received → send.ack → reply_sent
  │
  ├─ Akasha query → Provider raw first → Core first delta
  │                                      │
  │                                      ▼
  │                         durable queued → socket sent
  │                                      │
  │                                      ▼
  │                         Room → React commit → next frame
  │
  └─ Provider done → Akasha turn commit → runtime terminal
                                           │
                                           ▼
                              durable final → composer ready
```

定位规则：`send → provider.call.start` 属于准入、上下文与 Akasha 前置段；`provider.call.start → raw.first` 才是供应商首块段；`raw.first → next frame` 属于 Core、网络、Room 与 WebView 消费段。尾部同理拆成 Provider 完成、Akasha/AfterTurn、worker durable terminal 和客户端 composer 四段。

## 6. 产物、失败和回滚

- `npm run package:mobile-web` 只接受干净 Git tree，ZIP 内写入 source repository、commit、tree 和资产摘要。
- Android 在解包前核对外部 SHA-256；不匹配时 Gradle 失败，不使用旧缓存或网络 fallback。
- WebUI 构建失败不会改变移动端原生状态；产物升级只替换 APK 构建输入。
- 回滚主仓库到上一个 WebUI commit并重新打包；移动仓库恢复上一个 ZIP、摘要和 source lock。两边都不需要迁移数据库或 workspace。

## 7. 视觉语法

桌面与 Mobile 使用同一纸张品牌 token、`ChatMessageView` 和 `message-view.css`。用户气泡、Akashic 正文、Markdown 与工具过程由共享 WebUI 拥有；Mobile 只补 viewport、触摸、抽屉、Bridge、草稿、outbox 与离线状态，不增加装饰性角色标题或平行消息组件。Android 原生能力和 Bridge owner 不随视觉变化。

## 8. 试点验收

- 主仓库：typecheck、chat build、mobile web build、mobile state tests、lint。
- 视觉：只在生产桌面 Chat 与移动 Web 构建中核对主题 token、布局、消息和工具轨迹；不得为验收复制第二套消息 DOM、静态数据或交互状态。
- 离线与降级场景：通过生产 Chat 的静态响应、fixture transport 或现有状态测试注入输入，继续使用正式 `ChatMessageView`；不维护平行方案页或独立消息实现。
- 移动仓库：ZIP 正向校验、篡改失败测试、Gradle debug build。
- 报告两仓库 commit/tree、ZIP digest；真机 WebView、内存、掉帧和冷启动单独列为未验证或设备证据。
