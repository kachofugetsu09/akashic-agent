# 0041 · 纸张品牌 Token 取代 Material 视觉语义

- 状态：accepted
- 日期：2026-08-26
- supersedes：[0023](0023-akashic-tokens-own-material-3-semantics.md)
- 关联条款：WEBUI-001～WEBUI-007

## 背景

0023 让 Chat、Mobile WebUI、Dashboard 和插件从同一目录取得颜色，解决了多个页面各自解释主题的问题。但它同时把 `primary`、tonal surface、圆角胶囊和 Material 组件词汇带进品牌层。PR #500 已经引入暖纸底色和霞鹜文楷，实际组件仍大量依赖卡片、气泡、胶囊与 elevation，纸张只停留在配色，不足以成为可复用的 Akashic 品牌语言。

用户确认后续不再以 Material Design 作为视觉目标。新的纸张品牌必须继续保留主题唯一 owner、状态色语义和平台能力边界。

## 决定

1. `frontend/theme/src/brand-tokens.css` 是组件使用的品牌 token 入口，公开五条正交轴：`paper`、`ink`、`rule`、`typography`、`annotation`；success、warning、error、trace 和 info 继续使用独立状态角色。
2. token 以角色而不是组件命名。允许 `paper-canvas`、`ink-secondary`、`annotation-paper`，不新增 `card-background`、`chip-radius` 或 `button-blue`。
3. 页面默认是连续纸面。留白、字级、缩进和细规则线建立层级；只有输入、附件、错误、选择和需要隔离的工具详情形成局部纸片。卡片、气泡、胶囊、阴影和纹理不得成为默认容器。
4. Mobile 对话按连续手稿编排：用户输入是题记，Akashic 回复是正文，工具过程是可展开批注。流式内容只向后生长，停止、失败和重连保留已落下的正文并提供文字恢复路径。
5. `--md-sys-*`、`--ak-color-*` 和 `@material/web` 只作为迁移兼容接口，不能继续拥有品牌含义。尚未迁移的页面可以通过兼容别名取值，新实现只消费 `--ak-paper-*`、`--ak-ink-*`、`--ak-rule-*`、`--ak-annotation-*` 和 `--ak-type-*`。
6. 本决定只改变 WebUI 展示和 token API，不取得 SessionDB、Room、outbox、Bridge、配对、附件传输、原生生命周期或 WebUI generation 的所有权。

## 目标结构

```text
┌──────────────── Akashic Theme Catalog ────────────────┐
│ 主题色值 + status roles                               │
└────────────────────────┬──────────────────────────────┘
                         ▼
┌──────────────── Paper brand contract ────────────────┐
│ paper │ ink │ rule │ typography │ annotation │ status │
└───────────────┬───────────────────────┬──────────────┘
                ▼                       ▼
       Chat / Dashboard          Mobile manuscript
                │                       │
                └──────────┬────────────┘
                           ▼
           legacy Material aliases during migration
```

## 理由

纸张、墨色、排版和批注是彼此独立的变化轴，可以跨 Chat、Mobile、Dashboard 和插件组合；Button、Card、Chip 是组件形态，把它们写进品牌 token 会让主题依赖眼前组件。兼容别名把视觉迁移与 Theme Runtime、插件和发布链解耦，避免一次性重写所有消费者。

## 影响与回滚

- 新增品牌 token 与 Mobile 手稿样式，现有主题目录、深浅主题选择和领域状态色保持可用。
- 0023 变为 superseded；Material namespace 在消费者迁移完成前保留，不再接受新的直接依赖。
- 回滚本决定和对应 CSS/TSX 即可恢复旧视觉；没有数据库、workspace、Android 或协议迁移。

## 验收

- Mobile 的页面、消息、输入、模型选择、工具状态和流式回答只由品牌 token 建立视觉层级。
- 关闭阴影和背景装饰后，角色、状态和交互边界仍可辨认。
- 320 px、常用手机宽度、200% 缩放、浅色、深色和 reduced motion 保持可用。
- Browser Lab 覆盖 snapshot、stream、terminal、send 和原生能力拒绝；自动可访问性检查无 A/AA 违规。
