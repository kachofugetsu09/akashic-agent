# WebUI 插件

插件在 `plugin.py` 里 `UI.register(ctx, web="web_module.js", requires=("挂载点",))`，浏览器加载插件根目录的
`web_module.js`，同名 `web_module.css` 存在时一并加载。安装时宿主会静态校验，不满足直接安装失败、`error` 会写明原因。
本文件和 `templates/page/` 已经覆盖校验规则与 entry 字段，**不需要再去读宿主的校验器、`shell_ui` 打包代码或前端资源**。

## 硬性限制（安装时校验）

- `web_module.js` 是一个普通 ES module，必须**同步**导出 `activate(ctx)`，返回清理函数。不需要构建步骤，直接手写。
- 只能 `import` 这几个：`react`、`react/jsx-runtime`、`react-dom/client`、`@akashic/web-ui-v1`。不能动态 `import()`，不能引外部 URL。
- `web_module.css` 不能出现 `@import`（注释里也不行），`url()` 只能是 `data:` 或 `#`，不能声明 `@keyframes`、`@font-face`、`@property`、`@layer` 等全局 at-rule。

样式约定：类名加插件前缀（如 `.hello-page`），不写 `body`、`:root` 等全局选择器。安装校验不代替这项人工检查。

## 浏览器侧 API

`activate(ctx)` 拿到的 `ctx` 只有下面这些，没有别的类型文件要找：

- `ctx.ui.inject(挂载点, mount => mount.register(entry))`：向挂载点登记一项，返回清理函数。
- `ctx.http.request(path, init)`：请求本插件后端路由；`ctx.http.webSocketUrl(path)` 构造 WebSocket 地址。
- `ctx.data.resource(key, load, { warm })`：模块自己拥有的数据，在 `activate` 里创建，生命周期与模块相同。`render` 先画 `resource.current`，再调 `resource.refresh()` 在后台核对；`subscribe` 在值变化时通知（内容没变不通知）。`warm: true` 让宿主空闲时预取，首次打开也不必等网络；成功值留在本标签页，刷新后先画上次的结果。不要在 `render` 里直接 `fetch` 后再画，见 [0127](../../../docs/decisions/0127-web-modules-own-their-data.md)。
- `entry.render(host, view, props)`：把内容画进 `host`，返回清理函数。

## 挂载点

| 挂载点 | 出现在哪 | entry 必填字段 |
|---|---|---|
| `shell.pages.v1` | 顶部主导航的页面 | `id`、`label`、`route`（唯一）、`iconSvg`（以 `<svg` 开头）、`render` |
| `shell.rail-actions.v1` | 侧栏底部动作按钮 | `id`、`label`、`iconSvg`、`onActivate()` |
| `shell.settings.v1` | 设置里的分节 | `id`、`label`、`route`、`iconSvg`、`render` |

entry 可带 `order`（数字，越小越靠前，缺省 0）；相同时按 `id` 字母序。不想排到「对话」前面就给一个正数。

## 视觉一致性

- 颜色、字体、边框只用主题 token：`--ak-paper-canvas/sheet/raised/quiet`（底色）、`--ak-ink-primary/secondary/muted`（文字）、
  `--ak-rule-subtle/default/strong/focus`（线和焦点）、`--ak-type-reading/technical`（字体）。不写死颜色，亮暗主题就会自动跟随。
- 不 import 宿主前端源码里的组件；需要按钮可用 `@akashic/web-ui-v1` 导出的 `MaterialButton`（需要 React 渲染）。
- 验收要在亮色和暗色主题下各看一次，窄屏（约 390px）不出现横向滚动。
