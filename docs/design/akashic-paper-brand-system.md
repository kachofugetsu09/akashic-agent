# Akashic 纸张品牌系统

- 状态：implemented first slice
- 日期：2026-08-26
- 决策：[0042](../decisions/0042-paper-brand-tokens-replace-material-visual-semantics.md)
- 关联条款：WEBUI-001～WEBUI-007、MOB-001

## 1. 问题和用户意图

PR #500 已经统一桌面 Chat、Dashboard 的暖纸颜色和霞鹜文楷，但生产 Mobile 仍有独立解释的颜色和组件外观。后续界面应直接复用成熟 WebUI 的消息、Markdown、工具过程与视觉变量，而不是为 Mobile 再发明一套内容语言。

本阶段固定 token 合同，并让生产 Mobile 对齐共享 WebUI；不改变 Message、Turn、Session、Native Bridge 或 WebUI 发布协议。

## 2. 品牌语法

```text
┌──────────────────────── 一张连续纸面 ────────────────────────┐
│ paper      纸张层级：canvas / sheet / quiet / raised / inset │
│ ink        阅读层级：primary / secondary / muted / inverse   │
│ rule       结构层级：subtle / default / strong / focus        │
│ typography 阅读、技术、正文与辅助信息                        │
│ annotation 引用、选中、搜索、流式边、工具批注                │
│ status     success / warning / error / trace / info          │
└──────────────────────────────────────────────────────────────┘
```

品牌感觉由排版、留白、墨色关系和细规则线共同形成。不得用位图噪点、`feTurbulence`、随机纹理或持续动画冒充纸张；去掉装饰后，阅读层级和交互状态仍必须完整。

## 3. Token 合同

| 轴 | 公共前缀 | 负责 | 不负责 |
|---|---|---|---|
| Paper | `--ak-paper-*` | 页面和局部纸片的表面层级 | 文本、状态色 |
| Ink | `--ak-ink-*` | 正文、次要文字、弱化文字 | 背景和边框 |
| Rule | `--ak-rule-*` | 结构线、强边界、焦点 | elevation |
| Typography | `--ak-type-*` | 字体、字号、行高与辅助信息节奏 | 组件尺寸 |
| Annotation | `--ak-annotation-*` | 选择、引用、搜索、流式活动 | success/error |
| Status | `--ak-color-status-*` / `--ak-sys-color-*` | 成功、警告、错误、轨迹和信息 | 品牌强调 |

组件只能消费语义 token。若缺少角色，先补角色，不借用当前颜色恰好相同的 border 或 status token。

## 4. Mobile 复用边界

```text
共享 WebUI
├─ ChatMessageView / message-view.css ── 用户气泡、回答、Markdown、工具过程
├─ theme.css / brand tokens ──────────── 字体、颜色、边界与状态
└─ Mobile adapter
   ├─ 复用上面两层
   └─ 只拥有 viewport、触摸、抽屉、Bridge、草稿和离线状态
```

- 用户消息继续使用桌面 WebUI 已有的中性气泡，Akashic 回答直接显示正文，不添加角色标题。
- Markdown、代码、附件和工具过程继续使用共享组件，不复制 Mobile 专用 DOM。
- Mobile composer 保留自己的 Native 草稿、附件和 outbox 行为，只复用桌面 composer 的视觉变量与 focus 语言。
- 流式正文继续使用共享投影，不增加逐字动画或 Mobile 专用解析器。
- 原生能力在 Browser Lab 中只记录、实现或明确拒绝，不能用 mock success 隐藏边界。

## 5. 字体

- 阅读正文使用仓库随附的 `LXGW WenKai GB Screen` WOFF2；正文不低于 16 px，三行以上正文行高不低于 1.4。
- 代码、时间、运行身份和短技术标签使用 `JetBrains Mono`。
- 最多同时出现阅读与技术两种字体；不能为单一组件增加第三种品牌字体。
- 中文字体约 9.7 MiB，是当前 Mobile 首屏最大单项静态资源。后续优化必须先建立子集字符覆盖与 fallback 视觉 Gate，不能直接换回系统字体伪装优化成功。

## 6. 主题与兼容边界

Theme Runtime 仍从同一 Catalog 选择浅色和深色主题。`brand-tokens.css` 是新组件入口；Material 和旧 Akashic namespace 只提供迁移兼容。迁移期间的方向是：

```text
Theme Catalog → brand tokens → product components
             └→ legacy aliases → un-migrated consumers
```

新组件不得增加 `--md-sys-*` 直接依赖。迁移完成前不删除旧 namespace，避免破坏 Dashboard、插件和第三方公开控件。

## 7. 验收

1. 真实生产 `mobile-native.tsx` 在 Browser Lab 使用共享消息组件，而不是 Lab 自己复制 DOM。
2. conversation、stream、long、reconnecting 四个 fixture 均可操作。
3. 320 px 不发生横向溢出；200% 缩放仍可到达输入、发送和恢复动作。
4. light、dark、focus、selected、error、stopped 和 reduced-motion 均保持文字或图标信号。
5. WCAG 2 A/AA 自动检查通过；字体、色值和截图只在固定 Chromium 环境内比较。

## 8. 回滚

回滚 Mobile 对齐 CSS 和 `brand-tokens.css` 引用即可回到 PR #500 原视觉。状态、协议、数据库、原生客户端和正式 workspace 没有迁移。
