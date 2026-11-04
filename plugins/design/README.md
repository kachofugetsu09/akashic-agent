# design

工作台里的「设计」标签：基于 [open-pencil](https://github.com/open-pencil/open-pencil)（MIT）的设计画布，用户和 Agent 在同一份文档上编辑。决策见 ADR 0140。

- 文档按会话存放在本插件数据目录，格式是 `.fig`；修订号单调递增，写回时带上自己基于的修订，不一致就拒绝，不静默覆盖。
- 编辑器（Vue、`@open-pencil/core`/`vue`、CanvasKit WASM、字体）由 `npm run build:web-plugins` 打进不入库的 `static/`（发布构建随插件制品打包），
  按内容哈希命名、长期缓存；标签在同源 iframe 里第一次可见时才加载 `editor.html`。
- Agent 工具：`design_document` 与 open-pencil 的核心工具（`design_render`、`design_update_node`……，按需加载的一组），
  在服务端执行：插件用部署里的 Playwright Chromium 打开 `static/worker.html`（同一份编辑器代码的无头入口），每个会话一块，
  直接读写存储，用户不在线也能做设计；开着的画布经 `/api/dashboard/design/live` 收到新修订与 Agent 正在改的文档。
  `design_show_html` 把 workspace 里的网页发成一份带预览的设计文档。
- `static/jsx.html` 是 Agent JSX 的求值沙箱，`static/preview.html` 运行网页预览，两者都由 dashboard 按文件名加上独立的 CSP sandbox。
- 中日韩回退字体经 `/api/dashboard/design/fonts` 从 Google Fonts 代取，缓存在数据目录的 `fonts/`。
- open-pencil 自带的 AI 对话、云同步、协作房间、图库、统计上报与更新检查都不接入。

依赖：`ui`（`conversation.tools.v2` 工作台标签）、`tools`、`working_directory`。
