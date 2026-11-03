# 0140 · 设计工作台：Agent 和用户在同一块 open-pencil 画布上做设计，取代“网站”

- 状态：accepted / implemented for review
- 日期：2026-11-04
- 关联：0121（工作台标签）、0124（show_site，本记录取代）、0117（可视化附件）
- 修订：0124 整体退役：`show_site`、对话里的网站卡片、`/sites/frame.html` 与“网站”标签删除，已有的 `akashic.site.v1` 附件只剩普通文件下载

## 背景

0124 让 Agent 把 workspace 里的网页发到工作台“网站”标签。网页只能看、不能改：Agent 改一次要重新写文件、再发一次；
用户想挪一个按钮，只能用文字描述给 Agent。设计稿更适合放进专业的设计软件：图层、属性、撤销，双方在同一份文档上动手。

open-pencil（MIT）是开源的 Figma 式编辑器，npm 上有拆好的包：`@open-pencil/core`（场景图、CanvasKit 渲染、约 130 个工具）、
`@open-pencil/vue`（无样式的画布与面板原件）、`@open-pencil/dom-css`（HTML 转图层）。它自带的 AI 聊天、云端协作、遥测和更新都不需要。

## 决定

1. **内置插件 `design` 提供“设计”标签。** 文档按会话存在插件数据目录（`.fig` 字节 + 修订号），浏览器自动保存；
   修订号不一致时拒绝覆盖，提示用户重新加载。编辑器（约 5 MB 脚本、7 MB WASM、字体）是单独的构建产物，
   第一次打开标签或 Agent 第一次调用时才加载。
2. **编辑器跑在同源 iframe 里。** 宿主不允许 Web 模块动态加载代码；iframe 没有 Web 模块身份，
   它的接口请求和实时连接都经标签模块代发（`ctx.http.request` / `webSocketUrl`）。
3. **Agent 的设计工具就是 open-pencil 的核心工具。** 构建时从 `CORE_TOOLS` 生成名字、描述和 JSON Schema（去掉联网取图库的
   `stock_photo` 和在页面里执行任意代码的 `eval`），注册成 `design_*` 一组（按需加载）。调用经 WebSocket 交给用户打开的画布执行：
   用户和 Agent 改的是同一份场景图，Agent 的每次编辑是一步撤销，画布上实时可见；桌面端 Agent 第一次调用时自动展开标签（每轮一次）。
4. **Agent 写的 JSX 只在不透明源沙箱里求值。** open-pencil 的 `render` 用 `new Function` 编译 JSX；构建时把这一处换成
   交给 `sandbox allow-scripts` 的独立页面求值，交回的只是纯数据的图层树。编辑器页面自己的 CSP 不放行 eval 与内联脚本。
5. **网页预览并入设计文档。** `design_show_html(path, title?, import_layers?)` 取代 `show_site`：沿用 0124 的快照（同目录资源内联），
   快照成为会话里的一份设计文档，“预览”在 CSP sandbox 页面里原样运行；`import_layers` 用浏览器的 CSS 引擎把页面转成图层，放进新的一页。
6. **回退字体经后端代取。** 内置字体只有 Inter；中日韩文字从 Google Fonts 取按字子集，Google 的字体目录不带 CORS，
   由插件后端只对三个 Google 域名代取，并按 URL 缓存在数据目录。

## 已知限制

- 设计工具只在网页端可用：会话没有打开的网页画布时，工具返回错误并让 Agent 请用户打开网页（`design_show_html` 仍会保存预览）。
  不在服务端无头执行：Core 容器里没有 Node，执行宿主上跑一份无头编辑器换来的只是“用户不在时也能画”，暂不值得。
- 同一会话开着多个浏览器标签时，Agent 的调用只交给其中一个（优先正在显示的），其他标签在保存后收到新修订再重新加载；
  没有 CRDT，两边同时改同一份文档时后保存的一方会看到冲突提示。
- JSX 里的变量引用跨沙箱时按标记还原，其他非 JSON 值（函数）会被拒绝。
- HTML 导入只带页面 `<head>` 里的内联样式表，外链样式表不加载。
