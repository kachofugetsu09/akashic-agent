# Visuals

通过 `render_visual` 生成普通消息附件；可选聊天 UI 显示可交互的 HTML、SVG 和 Canvas。
工具组不常驻：插件随附 `visualize` 技能（`skills/visualize/`），说明什么时候画、按意图选形式和各类写法；
模型按需 `load_skill` 读取，再用 `load_tools` 解锁 `render_visual`。写法按形式拆成 `charts.md`、`diagrams.md`、
`interactive.md`，只读用得上的那份。
架构与保留边界见 [0117](../../docs/decisions/0117-visuals-are-message-attachments.md)。

## 运行条件

插件依赖 `tools.v1` 和 `ledger.artifact_import.v1`；`ui.slots.v1` 仅用于可选展示。
插件在产品进程内导入 Playwright，因此版本由产品 `requirements.txt` 与发行依赖锁拥有。
正式发行镜像随默认组合启用插件，构建时安装匹配版本的 Chromium headless shell、系统库与中英文字体；插件不在运行中下载程序。原生运行由操作者准备浏览器：

- 默认使用 Playwright 1.59.0 对应的 Chromium，可在产品环境中运行 `python -m playwright install --only-shell chromium`。
- 已有受管理 Chromium 时，插件配置 `executable_path` 指向它的绝对路径。
- 容器需提供 Chromium 的系统依赖并允许 Chromium sandbox；不通过关闭 sandbox 绕过环境错误。

源码构建运行 `npm run build:web-plugins`，生成 UI 入口、内联图表工具包的 frame 和从主题目录派生的预检 CSS。
只修改 `frontend/plugins/visuals/src` 的展示源文件。启停仍由 workspace 插件选择拥有；已有显式禁用选择不会被默认组合覆盖。

发行容器沿用非 root、只读根目录、移除全部 capabilities 和 no-new-privileges。Core 使用
`docker/host-runtime/chromium-seccomp.json` 允许浏览器创建用户命名空间，保留 Chromium sandbox。
该策略来自 [Docker 上游默认策略](https://github.com/moby/profiles/blob/4b8214fc4971e06172938a1f1554ccff6647ee80/seccomp/default.json)，
只保留本发行版 x86_64 与无 capabilities 的规则，并按 [Playwright 指导](https://playwright.dev/python/docs/docker)
允许 clone、setns、unshare，以及无宿主权限时仅在用户命名空间内可用的 chroot。
Workload Controller 仍用 Docker 默认策略。上游 Apache-2.0 许可保存在
`docker/host-runtime/chromium-seccomp.LICENSE`，本地策略的修改范围仅为上述过滤和沙箱调用。

客户端通过插件的静态入口 `/visuals/frame.html` 打开图表，响应 CSP 强制 `sandbox allow-scripts`，
不继承聊天页禁止内联脚本的策略，也不放宽聊天页权限。父页面只交付一次已读取的附件文档；
后续通信只包含主题、尺寸、就绪和错误，不提供宿主查询或执行能力。
静态文档不放在要求 Web UI 身份头的 `/api/dashboard/` 下：浏览器 iframe 导航无法携带这些头。

## 可观察合同

输入包含标题、纯文本摘要和完整 body HTML，最多 200,000 个字符。
使用内联脚本和样式，不依赖 CDN、外部请求、导航、宿主接口或动态导入。
文档自行编写 HTML/CSS/JavaScript，按内容选择 SVG、Canvas 或普通 DOM。frame 预置示意图类（`t`/`th`/`ts` 文字、
`box`、`arr`、`leader`、`node`、按系列色上色的 `d1`…`d6`/`d-gray`）、卡片与指标类（`cards`/`card`/`metric`）
和颜色随连线的 `url(#arrow)` 箭头，颜色都取主题 token。
数据图调用 frame 内联的 `window.ak`（源码 `frontend/plugins/visuals/src/charts.ts`）：柱、折线、份额、散点、哑铃、
热力图、矩形树图、排行和迷你趋势线，配合 `.headline`、`.hero`、`.dtable`、`.foot` 等编辑式版式类。
其余图由文档自己实现坐标、图例、读数与交互。API 只增不改，因为已保存附件会在新 frame 中重新执行。
系列色使用 `--k1` 至 `--k6`（映射到 `--ak-chart-*`）；纸面、文字和规则线使用工具 schema 列出的公共 token。
布局需要适应 280px 和 720px，控件至少 44px；每图最多 12 个滑块。

预检复用浏览器，最多两次并发，繁忙直接返回工具错误。单次检查预算 20 秒。
脚本异常、空白内容、横向溢出或数值输出中的非有限数值会阻止发布；滑块会触发真实 input/change。
错误交回当前模型的正常工具循环；插件不会另调模型或伪造成功。

成功返回标题、摘要、workspace `visuals/` 下的文件路径、HTML 文件的 `artifact_ref`，以及一张 720px 宽的预检截图
（IMAGE 附件）。截图只给模型自检；宿主把同一结果里的图片留在调用卡片，正文只显示交互图。
`draft: true` 只预检并回传截图，不写文件、不交付附件，用户界面不出现新图；模型看截图改好后再正式发布，
避免发布后为外观重画而出现第二张图。文件是自包含 HTML
（frame 模板、内联 `ak` 与默认主题），由 Ledger 以不可变附件保存，单独用浏览器打开也能运行；
模型可以把同一路径交给 `message_push` 推送。旧版 `.akashic-visual.json` 仍可展示。
历史页面只读文件，不重新运行工具或请求模型。图中操作只改变当前页面的临时状态。
禁用后摘要与下载链接保留；重新启用、刷新或打开另一页面会从文档初始参数开始。

任何消息中的 HTML 文件附件都由本插件在同一 sandbox 中展示；用户自己上传的网页点击“运行网页”后才执行。
加载失败、禁用或缺少插件时，宿主保留消息中的标题、摘要和下载入口。
已打开的图发生交互错误时保留原 iframe 与参数，并在图旁说明错误；原始文件仍可下载。预检不是数学正确性证明，也不能保证任意
后续交互都没有错误。服务端断网与客户端 sandbox/CSP 的保证不同，详见架构决定。

入选模型上下文的工具调用仍携带原 HTML 参数；不另建历史裁剪机制。
模型收到标题、摘要、附件引用和预检截图（模型不接收图片时宿主只给说明）；预检耗时不代表完整模型生成耗时。

## 验收

在安装依赖和构建资产后，使用一次性目录：

```sh
.venv/bin/python scripts/visuals_scenario.py --output /tmp/visuals-acceptance
node scripts/visuals/browser-server.mjs /tmp/visuals-acceptance
```

浏览器打开 `http://127.0.0.1:4330`。该页用实际聊天组件回放上一命令导出的真实消息、附件与安装资产。
调节参数后追加文字、完成回复、切换主题，确认参数保留；禁用确认摘要和文件仍在；重新启用确认初始参数。
在 320px 窄屏和深色主题核对图形、控件与下载链接。此页不是正式 runtime，也不证明部署完成。
