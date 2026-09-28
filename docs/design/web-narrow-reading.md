# 共享 Web 窄屏阅读设计

状态：实现；实际运行证据随 PR 与发布记录保存。

## 问题和目标

Pixel 9 Pro XL 的实际页面暴露出三项问题：模块菜单关闭后仍覆盖内容，固定高度目录挤压消息区，桌面六列表格在窄屏裁掉正文和字段。CSS 的 `display: grid` 覆盖了菜单的 `hidden`；减少列宽没有解决阅读路径。

共享 Web 页面继续遵循 [WEBUI-001～WEBUI-004](../projectneed.md) 和[纸张品牌](akashic-paper-brand-system.md)。窄屏一次展示当前任务，目录、列表和详情逐层进入，返回动作明确。代码和真实二维图表可以在自己的区域横向滚动；普通正文按可用宽度换行。

```text
┌─────────────────────────────┐
│ Akashic       功能设置  主题 │
│ 对话       工作台       模型 │
├─────────────────────────────┤
│ 原生模块选择                │
├─────────────────────────────┤
│ 会话目录                    │
│   选择 → 消息列表           │
│           打开 → 完整详情   │
│           ← 返回列表        │
│   ← 返回会话目录            │
└─────────────────────────────┘
```

## Owner 与调用路径

Shell 从 `shell.pages.v1` 发现页面，保留同一导航 guard 和路由；普通按钮替代滚轮的模糊、缩放和动画状态。Workbench 从 `workbench.panels.v2` 发现插件，用原生选择器选择模块。会话、消息、记录和详情仍使用原查询与状态，不创建移动端 DTO。记录摘要的字段标签来自插件已有列定义，排序、分页和批量操作保留。

窄屏通过 CSS 切换可见阅读区；插件仍按原生命周期挂载。分类导航按需展开，插件自定义主面板独占内容区。主面板自己拥有实际数据时，Workbench 不显示自己的空查询计数。

Onboarding 的嵌入页面按步骤销毁。子插件可能拥有独立 React root，清理在父页面提交后的微任务中执行，避免 React 尚在提交时提前清空子节点。每个旧 host 随步骤的 key 更换，清理不会触碰新 host。外部 Observe 的错误列表与详情使用同样的阅读方向，其数据、错误状态和发送操作仍由 Observe 拥有。

长正文至少 16px、行距至少 1.4；元数据可以更紧凑。连接名称、标识和正文按可用宽度换行。短屏让当前内容滚动，标题和操作不抢占全部高度。聊天保留原消息区的滚动 owner，短屏输入区进入文档流；模型选择面板整体滚动、保留关闭动作，首尾模型都能到达。模型表单的网格不从按钮长文字推导最小宽度。

本次只改变展示、导航和读取状态。Core、Message/Session、配置、凭据、插件输入选择、data_dir 和插件发布合同不变；没有迁移或业务写入。更新外部插件从其源码仓库交付，再走正式安装链。

## 在电脑上重复验收

安装仓库 Node 依赖并提供本机 Chromium。脚本使用现有 `playwright-core`，不安装或控制手机。

```bash
npm run check:narrow-ui -- --url http://localhost:2236 --output /tmp/akashic-narrow-ui
```

默认检查 320、360、390、412、448、760、1024、1365 CSS px。页面入口来自实际模块、工作台选项、配置目录和模型连接，不以固定页面数量代替发现。验收包含会话→消息→详情→返回、插件列表/详情/分类、Observe 错误标签、Fitbit 判断记录、Meme 分类、全部可进入的配置步骤、高级选项和未保存确认、全部模型连接/模板/手动配置、Embedding、聊天历史/导航/项目弹窗/模型与思考强度选择、实际工具标签和 Computer 剪贴板。

```bash
# 短屏、字体放大和 RTL 分开观察，报告保留各自输入。
npm run check:narrow-ui -- --url http://localhost:2236 --widths 320 --height 240 --output /tmp/akashic-short
npm run check:narrow-ui -- --url http://localhost:2236 --widths 320 --large --output /tmp/akashic-large
npm run check:narrow-ui -- --url http://localhost:2236 --widths 320 --rtl --output /tmp/akashic-rtl
# 指定 Chromium 路径：--browser /path/to/chromium 或 CHROME_BIN。
```

候选构建先运行 `npm run build:chat` 和 `npm run build:web-plugins`，再用 `--candidate` 替换本地 Chat 和 Web 插件制品；`--plugins` 指定内置插件，`--bundles observe=/path/to/source,proactive_feedback=/path/to/source` 指定已构建的外部源码目录。它只替换本次浏览器响应，不写服务器。候选页面使用 HTTPS 或 localhost；远端 LAN 服务可以用 SSH 本地转发，例如 `ssh -N -L 18836:127.0.0.1:2236 hua-home`，验收 URL 使用 `http://127.0.0.1:18836`。浏览器拦截生成的 HTML 会触发 Chromium 的本地网络检查，脚本只向该服务 origin 授予临时本地网络访问权限。部署后必须去掉 `--candidate`，检查正式制品。

脚本拒绝 HTTP 写请求，只放行已核对为只读的 `project.list` 查询；不发送消息、不保存配置、不登录、不删除记录、不向远程桌面发送输入。表单只作临时本地编辑，并通过原确认弹窗放弃。弹窗、iframe、正文与操作分别检查，失败返回非零。截图和 `report.json` 供逐页人工复核；截图可能包含个人历史，保留在私有目录，不上传 CI 或 PR。

## 验收边界和恢复

当前脚本要求服务已有可读取的会话、插件记录和模型连接；实际没有数据时不能假装完成对应阅读场景。新安装、缺失依赖、网络故障、登录中间态、特殊附件和不同 Markdown 内容需按其实际条件补充场景。脚本验证电脑 Chromium；它不证明 Android WebView 的系统生命周期、软键盘和通知行为，也不能证明今后所有插件或所有内容永远正确。

源码恢复使用任务基线；发布通过正式安装旧源码版本完成，不回写持久业务数据或插件 cache。失败报告保留页面、视口和截图，先修复实际路径，再重复受影响的场景。
