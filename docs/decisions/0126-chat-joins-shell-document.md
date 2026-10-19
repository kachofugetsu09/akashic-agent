# 0126 · 聊天页并入外壳文档，拆除 iframe

- 状态：accepted / implementing
- 日期：2026-10-19
- 关联：WEBUI-008、0051（Web UI 由普通插件模块组合）、0121（工作台标签）、`docs/design/web-ui-plugin-composition.md` §4.4 与 §13.3；参考 DSH `client-plugin-loading-model`

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

conversation_ui 的 `web_module.js`（约 1200 行）负责两边之间的桥，共 11 种 postMessage：主题、页面重新可见、
会话与覆盖层状态、工作台标签与意图（含结果回传）、底栏动作（含激活回传）、引用到输入框。

`web-ui-plugin-composition.md` §13.3 已经写明：iframe 不作为长期默认的插件隔离，只当迁移适配层。

## 参考

DSH 的浏览器侧只有一个文档：外壳提供 React、Cordis 与静态 UI 库的唯一身份，各插件包（包括对话界面 `ui-conversation`）
在同一套模块系统和同一棵组件树里装载，没有用 iframe 隔离第一方插件。我们的 Web 外壳本来就是 import map 加共享运行时，缺的只是让聊天也走这条路。

## 决定

1. 聊天应用改为 conversation_ui 的普通挂载：`mountChat(host, 宿主接口)`。外壳文档里只保留一份 React，聊天和聊天插件模块都经外壳的
   import map 取得 `react`、`react-dom`、`@akashic/web-ui-v1` 与 `@akashic/chat-ui-v1`。`mountChat` 本身也是 Host SDK
   `@akashic/chat-app-v1`：插件 web module 仍只能静态 import Host SDK，conversation_ui 不靠动态加载取得聊天。
2. 原来的 11 种桥消息改为有类型的宿主接口：宿主到聊天的方向用句柄方法（底栏动作、工具区是否在场、工作台意图与引用），
   聊天到宿主的方向用回调（当前会话、覆盖层、意图请求、底栏动作激活）。主题直接读外壳的 `@akashic/web-ui-v1`，不再同步。
3. 同一文档里的约束：
   - 聊天的全局样式（`html/body/#root`）收进聊天根节点之下；
   - `window.__akashicRuntime` 只追加 `ChatUi`，不覆盖外壳已登记的运行时；
   - 聊天的 document 级快捷键不处理来自其他区域（工作台面板）的按键；
   - 聊天的响应式断点按聊天根节点自身的宽度判断，不按窗口宽度。
   - 锁住页面的聊天模态层（对话框遮罩）带 `data-chat-modal`；宿主据此在隐藏聊天时先保留模态可见，
     不从 `aria-hidden` 等通用属性反推。聊天里的菜单一律非模态，不锁页面。
4. 拆除顺序（每步一个 PR）：
   1. 聊天可挂载：增加模块入口与共享 React 的构建，桥改成接口；iframe 仍用同一接口的消息适配；
   2. conversation_ui 直接挂载聊天，删除 iframe 与桥；
   3. 删除独立 `/chat` 页面、embedded 模式和 iframe 构建（`/chat` 带查询参数 308 回外壳）；夹具脚本改用
      `vite.harness.config.ts` 的独立夹具页面，它同样经 `mountChat` 挂载，只用于本地开发与验收，不发布。
      插件共享运行时的垫片（含 `chat-ui-v1.js`）归外壳的 `frontend/dashboard/public/sdk`。
5. 插槽与递归挂载两种组合方式继续并存：聊天页是一个插件页面，它的平铺插槽属于这个页面自己的扩展点，不升格为外壳的全局插槽。

## 后果

- 收益：只有一个文档、一份 React，聊天插件与外壳插件共享同一套运行时；不再有跨文档的消息协议要维护和做兼容。
- 代价：聊天与工作台不再被 iframe 隔开，样式和按键要靠根节点作用域与区域判断来约束；聊天的弹层以整个窗口为范围居中。
- 不做：在 iframe 内外各放一套插件注册表让插件自选；用 iframe 隔离第一方插件。不可信 UI 若出现，另立隔离设计。
