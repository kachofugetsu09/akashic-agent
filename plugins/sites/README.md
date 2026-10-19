# Sites

`show_site` 把 workspace 里的一个 HTML 发布成不可变快照，在网页聊天右侧边栏的“网站”标签里打开，
用户在自己的浏览器里直接操作；标签由 `sites_tab` 提供。边界与隔离见 [0124](../../docs/decisions/0124-show-site-opens-in-workbench.md)。

- 依赖 `tools.v1`、`ledger.artifact_import.v1` 与 `working_directory.v1`；UI（卡片与标签）可选。
- 入口同目录或子目录引用的资源内联为 data URI，合计 10 MB 以内；外部 https 资源原样加载。
- 附件是 `schema: "akashic.site.v1"` 的 JSON：`title`、`path`（相对 workspace）、`published_at`、`html`。
- 卡片源文件在 `frontend/plugins/sites/src`，`npm run build:web-plugins` 生成 `message_ui.*`；不要直接改生成产物。
