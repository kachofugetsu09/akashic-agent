# design

工作台里的「设计」标签：基于 [open-pencil](https://github.com/open-pencil/open-pencil)（MIT）的设计画布，用户和 Agent 在同一份文档上编辑。

- 文档按会话存放在本插件数据目录，格式是 `.fig`；修订号单调递增，浏览器保存时带上自己基于的修订，不一致就拒绝，不静默覆盖。
- 接口：`/api/dashboard/design/documents`（列出、新建、改名、读写；读取时修订号放在 `ETag` 里）。

依赖：`ui`。
