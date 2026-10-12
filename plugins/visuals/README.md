# Visuals

通过 `render_visual` 生成普通消息附件；可选聊天 UI 显示可交互的 HTML、SVG 和 Canvas。
架构与保留边界见 [0117](../../docs/decisions/0117-visuals-are-message-attachments.md)。

## 运行条件

插件依赖 `tools.v1` 和 `ledger.artifact_import.v1`；`ui.slots.v1` 仅用于可选展示。
安装链读取 `requirements.txt` 建立 Playwright 环境。浏览器由操作者准备，插件不在运行中下载程序：

- 默认使用 Playwright 1.59.0 对应的 Chromium，可在插件环境中运行 `python -m playwright install chromium`。
- 已有受管理 Chromium 时，插件配置 `executable_path` 指向它的绝对路径。
- 容器需提供 Chromium 的系统依赖并允许 Chromium sandbox；不通过关闭 sandbox 绕过环境错误。

源码构建运行 `npm run build:web-plugins`，生成 UI 入口、frame 和从主题目录派生的预检 CSS。
只修改 `frontend/plugins/visuals/src` 的展示源文件。插件不修改任何默认 bundle 或自动启用自身。

## 可观察合同

输入包含标题、纯文本摘要和完整 body HTML，最多 200,000 个字符。
使用内联脚本和样式，不依赖 CDN、外部请求、导航、宿主接口或动态导入。
文档自行编写 HTML/CSS/JavaScript，按内容选择 SVG、Canvas 或普通 DOM。
不内置绘图库或专用图表 API；坐标、刻度、图例、读数与交互逻辑由文档自己实现。
图表使用 `--ak-chart-1` 至 `--ak-chart-6`；纸面、文字和规则线使用工具 schema 列出的公共 token。
布局需要适应 280px 和 720px，控件至少 44px；每图最多 12 个滑块。

预检复用浏览器，最多两次并发，繁忙直接返回工具错误。单次检查预算 20 秒。
脚本异常、空白内容、横向溢出或数值输出中的非有限数值会阻止发布；滑块会触发真实 input/change。
错误交回当前模型的正常工具循环；插件不会另调模型或伪造成功。

成功返回标题、摘要和普通 `artifact_ref`；下载文件名使用标题。JSON 文档包含 `schema: "akashic.visual.v1"`、
`title`、`summary` 和 `html`，由 Ledger 以不可变文件保存。
历史页面只读文件，不重新运行工具或请求模型。图中操作只改变当前页面的临时状态。
禁用后摘要与下载链接保留；重新启用、刷新或打开另一页面会从文档初始参数开始。

宿主只把真实工具结果交给 renderer；用户上传同后缀文件不会自动执行。
加载失败、禁用或缺少插件时，宿主保留消息中的标题、摘要和下载入口。
已打开的图发生交互错误时保留原 iframe 与参数，并在图旁说明错误；原始文件仍可下载。预检不是数学正确性证明，也不能保证任意
后续交互都没有错误。服务端断网与客户端 sandbox/CSP 的保证不同，详见架构决定。

入选模型上下文的工具调用仍携带原 HTML 参数；不另建历史裁剪机制。
模型收到标题、摘要与附件引用，不会自动看到截图；预检耗时不代表完整模型生成耗时。

## 验收

在安装依赖和构建资产后，使用一次性目录：

```sh
.venv/bin/python scripts/visuals_scenario.py --output /tmp/visuals-acceptance
node scripts/visuals/browser-server.mjs /tmp/visuals-acceptance
```

浏览器打开 `http://127.0.0.1:4330`。该页用实际聊天组件回放上一命令导出的真实消息、附件与安装资产。
调节参数后追加文字、完成回复、切换主题，确认参数保留；禁用确认摘要和文件仍在；重新启用确认初始参数。
在 320px 窄屏和深色主题核对图形、控件与下载链接。此页不是正式 runtime，也不证明部署完成。
