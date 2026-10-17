# 0124 · show_site：Agent 做的网页在用户自己的浏览器里打开

- 状态：proposed / implemented for review
- 日期：2026-10-18
- 关联：0117（可视化附件）、0121（工作台标签）、0123（电脑只剩一个浏览器）、WEBUI-008
- 修订：0121 第 2 条“只有用户的点击可以打开工具区”增加一个例外，见下文

## 背景

Agent 经常在 workspace 里写原型、设计稿、showcase。要给用户看，只能让远程 Computer 浏览器打开它，
画面卡，而且要先起 http 服务；遇到防火墙时 Agent 自己 `docker run` 了一个容器才打开，前后 30 分钟。
`render_visual`（0117）解决的是聊天里的小图：HTML 要写进参数、≤200k 字符、只适配 280–720px、禁止外部资源，
都不适合整页原型。

## 决定

两个插件，默认组合都启用：

- `sites` 拥有事实：`show_site(path, title?)` 工具、快照格式、对话里的网站卡片、隔离入口 `/sites/frame.html`；
- `sites_tab` 只提供工作台“网站”标签（0121 的“标签由 owner 或独立插件提供”）。
  分开的直接原因：带 Web 模块的插件，其 Dashboard 路由都要求 Web 身份头，iframe 导航带不上，
  隔离入口只能放在没有 Web 模块的插件里。标签缺席时卡片的“打开预览”自动隐藏。

复用 0117 的三件事：存储走 Ledger 附件，展示走插件自己的静态 frame 入口，父页面把文档 postMessage 给 frame。
不新增存储、鉴权或宿主执行能力。

```text
workspace/x.html ─▶ show_site：解析路径、内联本地资源 ─▶ akashic.site.v1 附件（不可变）
                                                          │
          对话：网站卡片（message.attachment）◀───────────┤
                │ site.open.v1 {artifact_id}              │
                ▼                                         ▼
          工作台“网站”标签 ── fetch 附件 ── /sites/frame.html（CSP sandbox）← 交付一次文档
```

1. **输入是路径，不是 HTML。** 相对路径按会话工作目录解析，与文件工具一致；`resolve()` 跟随符号链接后必须仍在
   workspace 内。Core 直接读 workspace（Host Bridge 部署里 workspace 挂在同一路径），不经执行桥。
2. **快照是单文件。** 入口同目录或子目录里被 `src`、`href`、`srcset`、`poster`、CSS `url()`/`@import` 引用的文件
   换成 data URI（外链脚本、样式也用 data URI，`defer`、`media`、`type` 语义不变）；CSS 递归处理。越界（`../`、
   根路径、链接到外面）与合计超过 10 MB 是工具错误；文件不存在只保留原引用并在结果里列出。外部 https 原样保留。
   脚本里拼出来的地址、`import()` 的相对模块、页面之间的链接不处理。
3. **每次调用一份附件。** 历史消息里的卡片打开的永远是当时的快照；没有“最新版”指针，也没有删除。
4. **标签单开、按会话记住网页。** 标签按路径记住本会话发布过的网页（sessionStorage，读写失败退回内存），
   同一路径再次发布时原地换成新版本，宽度（手机 390 / 平板 768 / 自适应，localStorage）不变。
5. **隔离。** frame 响应头 CSP `sandbox allow-scripts allow-forms allow-modals allow-popups`，不给
   `allow-same-origin`，页面是 opaque origin：读不到父窗口、akashic 的 cookie 与存储；不能导航顶层窗口。
   CSP 放行 https 的脚本、样式、字体、图片、连接，便于用 CDN 与网络字体；`form-action 'none'`、`base-uri 'none'`。
   “新标签页打开”同样打开这个 frame 页，经 opener 交付文档后先断开 `window.opener`；不用 blob URL（它与
   akashic 同源）。聊天页 CSP 不变。卡片只认 `show_site` 工具结果里的 `.akashic-site.json`，用户上传的同名文件不执行。
6. **自动打开的例外。** `show_site` 本身就是 Agent 明确的“给用户看”，所以刚发布的结果自动打开一次标签：
   仅当附件所在消息的时间在 2 分钟内、本页还没自动打开过这份快照、顶层窗口宽度 ≥ 768px。刷新、回到会话、
   历史卡片、手机宽度都只留卡片，由用户点开。宿主为此只多给附件来源一项 `at`（所在消息时间），不新增工具区能力。
7. **缺少 UI 时降级。** 工具不依赖 UI：没有网页聊天或插件 UI 时，结果文字与快照 JSON 下载照常存在。

## 不做

- 多页面站点、开发服务器、热更新；改完文件由 Agent 再调用一次。
- 预检（不跑浏览器、不检查溢出）：页面运行在用户浏览器里，出错用户当场能看到，Agent 也能按反馈修改。
- 把 visuals 迁进侧栏，或让 Agent 读用户在页面里的操作。

## 已知限制

- `connect-src https:` 也匹配 akashic 自己的 https 地址。opaque origin 的请求按跨站处理，SameSite=Lax 的会话 cookie
  不随请求发送，响应也没有 CORS 头，页面读不到；写请求另由 web-auth 的同源校验拒绝。
- 页面里相对链接的导航会在 sandbox frame 里打开 akashic 的同名路径（通常 404），仍在 sandbox 内。
- 快照 JSON 最大约 10 MB，打开时整份下载；附件没有物理清理（与其它附件相同）。
- 时钟差超过 2 分钟时不会自动打开，只留卡片。
