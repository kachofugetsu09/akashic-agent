# 0140 · 设计工具用 scene graph 做原型，投影成 show_site 收的 HTML

- 状态：proposed
- 日期：2026-10-19
- 关联：0124（show_site 在用户浏览器打开）、0117（可视化附件）、0121（工作台标签）、PLG（插件边界）

## 背景

`show_site`（0124）已经解决了"把网页给用户看"：agent 在 workspace 写 HTML，
`show_site` 内联资源成不可变快照，右侧栏打开。

瓶颈移到了"agent 用什么做"。手写 HTML 时，间距、字号、颜色每次都会漂；
agent 也无法复用同一份设计做多目标输出。设计工具的通行做法是让调用方操作
**scene graph**（frame/文本/填充/自动布局），渲染与导出由引擎负责。

## 决定

新增 `plugins/design`，让 agent 操作 scene graph，投影成 HTML 后交给既有的 `show_site` 链路。

```text
Agent ──design_* 工具──▶ scene graph（权威）
                              │
                              └─▶ exportHTMLBundle({html:'standalone'}) ─▶ workspace/x.html
                                                                            │
                                                              show_site（0124，不改）
```

三条边界：

1. **scene graph 是权威，HTML 是投影。** 与 projectneed「保存事实，按需投影」一致。
   反向（HTML 权威、scene 解析而来）会让渲染结果承担设计意图，且 `snapshot.py` 的
   内联是单向的，无法无损反解。
2. **投影在浏览器 JS 环境执行，不在 Python 里。** 发行镜像
   （`Dockerfile.distribution`）没有 Node，无法运行 `@open-pencil/dom-css`；
   但它装了 Chromium。`plugins/visuals/preview.py` 已有 Playwright 浏览器池范式。
3. **不新增存储、鉴权或宿主执行能力。** 发布仍走 0124 的 Ledger 附件与隔离 frame；
   `sites` 的工具、快照格式与标签行为都不变。

## 复用范围

`@open-pencil/*` 为 MIT。复用 `scene-graph`（文档模型）、`dom-css`（DOM/CSS 投影）；
**不复用** `@open-pencil/vue`（Vue UI 绑定，本项目前端是 React）与
`@open-pencil/mcp`（独立 HTTP/WS 服务，会引入第二个控制面）。

## 为什么不选另一方案

- **继续让 agent 手写 HTML**：不解决样式漂移，也无法多目标导出。
- **Python 侧重写投影**：等于重做 DOM/CSS 投影与 Tailwind 序列化，收益不抵成本。
- **引入 Node 到发行镜像**：扩大运行期依赖与攻击面，只为一次投影不值得。

## 什么前提改变后值得重议

- 发行镜像开始带 Node，投影可移回 Python；
- `dom-css` 的浏览器导出路径出现阻碍（如必须 Node 才能保证保真）；
- 前端技术栈改为 Vue，则 `@open-pencil/vue` 的 UI 绑定可一并复用。
