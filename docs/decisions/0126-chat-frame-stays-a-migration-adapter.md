# 0126 · 聊天 iframe 暂时保留为迁移适配层

- 状态：proposed
- 日期：2026-10-19
- 关联：WEBUI-008、0051（Web UI 由普通插件模块组合）、0121（工作台标签）、`docs/design/web-ui-plugin-composition.md` §4.4 与 §13.3

## 背景

Web 外壳（`frontend/dashboard`，Host SDK `@akashic/web-ui-v1`）用递归挂载组合插件：页面、设置页和工作台面板都是普通插件
`mount(host, context)`，父插件给子插件开挂载点。聊天页（`frontend/chat`，SDK `@akashic/chat-ui-v1`）由 conversation_ui 插件
放进外壳；它在 iframe 里另起一个文档，用平铺的插槽（`turn.after_answer`、`tool.result`、`composer.toolbar` 等）组合插件。

两边实际是两个单页应用：

| | 外壳 | 聊天 |
|---|---|---|
| import map | `/dashboard/assets/sdk/*`，含 `@akashic/web-ui-v1` | `/assets/sdk/*`，含 `@akashic/chat-ui-v1` |
| React 实例 | 一份 | 另一份 |
| 全局样式与启动壳 | `frontend/dashboard` | `frontend/chat/index.html`、`styles.css` |
| 插件组合 | 递归挂载 | 插槽，按插槽给出有类型的事实（#1508） |

conversation_ui 的 `web_module.js`（约 1200 行）负责两边之间的桥。它使用 12 种 postMessage
（主题、页面重新可见、会话与覆盖层状态、工作台标签与意图、底栏动作、引用到输入框），以及 2 个 localStorage 键。

`web-ui-plugin-composition.md` §13.3 已经写明：iframe 不作为长期默认的插件隔离，只当迁移适配层。

## 决定

1. 这一轮插件化收口（#1478–#1509）保留 iframe。业务插件只通过 chat-ui-v1 插槽或 web-ui-v1 挂载点接入，桥只认与业务无关的
   宿主消息：#1509 去掉了最后一条带业务名的 `akashic.models.changed`。今后新增桥消息必须与业务无关；插件需要的宿主事件放进插槽上下文（例如 `onShown`）。
2. 拆掉 iframe 另立一组栈，不混在收口 PR 里。拆的方向：
   1. 外壳 import map 增加 `@akashic/chat-ui-v1`，两边共用一份 React；
   2. 聊天应用改为 conversation_ui 的普通 `mount`，全局样式收进聊天根节点之下；
   3. 桥消息逐条换成父子挂载的属性与回调（主题读 `currentTheme`，工作台直接调用 `context.workbench`）；
   4. 最后删除独立的 `/chat` 文档与桥。
3. 插槽与递归挂载两种组合方式继续并存：聊天页是一个插件页面，内部的平铺插槽属于这个页面自己的扩展点，不升格为外壳的全局插槽。

## 后果

- 收益：现有插件不受影响；拆 iframe 时的回归面（全局样式、路由、Android Shell 加载、编辑器状态在切页时保留）可以单独验收。
- 代价：期间保留两份 React 和两份 SDK 产物，聊天首屏要多加载一个文档；桥协议仍需维护。
- 不做：在 iframe 内外各放一套插件注册表让插件自选；跨文档共享插件实例。
