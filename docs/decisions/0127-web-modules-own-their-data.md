# 0127 · Web 模块拥有自己的数据，页面挂载只读不取

- 状态：accepted / implemented for review
- 日期：2026-10-21
- 关联：WEBUI-008、0051（Web UI 由普通插件模块组合）、0126；参考 DSH 插件级 store（`ui-settings-models` 的 `ModelsSettingsStore`、`ui-settings` 的 describe mirror）

## 背景

`@akashic/web-ui-v1` 给插件的只有 `ctx.http` 与 `ctx.ui.inject`，入口是 `entry.render(host)`。插件只能在 render 里取数，数据随 DOM
一起创建、一起销毁。设置对话框换节或关闭就 dispose 分节，于是每次进入"插件列表""缓存效率""模型"都从加载文案开始，至少付一个网络往返；
模型页还因为挂载时读一次、IntersectionObserver 再读一次，每次打开发两个 catalog 请求。

DSH 同样由插件提供设置分节，但 store 建在插件的 `apply()` 里，生命周期与插件相同；分节组件只订阅 store，再次打开不发请求。

测量口径：本机 Chromium，经单向 60ms（RTT 120ms）TCP 代理访问两边；在页面内对导航按钮调用 `click()`，
记录到设置对话框内最后一次 DOM 变化（之后 800ms 无变化）的时间，取 5–7 轮中位数。

基线（origin/main 94027a5c）：

| 场景 | 模型 | 插件 | 缓存效率 | DSH 模型 | DSH 插件 |
|---|---|---|---|---|---|
| 页面加载后首次进入 | 140ms | 157ms | 141ms | 261ms | 4ms |
| 同一对话框再进 | 142ms | 153ms | 137ms | 2ms | 2ms |
| 关闭后重开 | 138ms | 152ms | 137ms | 2ms | 2ms |
| 同一标签页刷新后立即进入 | 150ms | 150ms | 138ms | 259ms | 2ms |
| 新标签页立即进入 | 146ms | 155ms | 138ms | 262ms | 5ms |

插件与缓存效率每轮都出现"正在读取…"。

## 决定

1. `WebHostContextV1` 增加 `ctx.data.resource(key, load, { warm })`，返回 `WebResource<T>`：
   - 在 `activate(ctx)` 里创建，生命周期与模块激活相同，模块释放时中止读取；
   - `current` 同步可读，`subscribe` 订阅；`refresh()` 合并进行中的读取，读取期间保留旧值；`set(value)` 接受写操作已经拿到的值；
   - 读到的内容与当前值相同（JSON 相等）时保留原对象、不通知，后台核对不引起重绘；
   - 成功值按 `插件 + 模块字节摘要 + key` 写入本标签页 sessionStorage（单值上限 256 KiB），刷新页面先画上次的结果；
   - `warm: true` 的资源在全部模块激活后由宿主在空闲时依次预取。
2. 分节 render 先画 `current`，再 `refresh()` 在后台核对；只有还没有任何值时才显示读取状态。
3. 首批迁移：插件列表（Shell）、模型目录与缓存效率（models）。插件列表是只读页，首次进入后在对话框里保活，隐藏时暂停 5 秒轮询；
   带草稿的插件表单仍在换节时销毁，守卫语义不变。
4. 顺带去掉 Shell 每次重渲染都重新解析导航 SVG：React 19 按对象身份比较 `dangerouslySetInnerHTML`，同一段图标复用同一个对象。

## 结果

同一口径：

| 场景 | 模型 | 插件 | 缓存效率 | DSH 模型 | DSH 插件 |
|---|---|---|---|---|---|
| 页面加载后首次进入 | 5ms | 5ms | 4ms | 261ms | 4ms |
| 同一对话框再进 | 5ms | 3ms | 3ms | 2ms | 2ms |
| 关闭后重开 | 4ms | 3ms | 4ms | 2ms | 2ms |
| 同一标签页刷新后立即进入 | 4ms | 6ms | 5ms | 259ms | 2ms |
| 新标签页立即进入 | 7ms | 5ms | 4ms | 262ms | 5ms |

热切换与 DSH 同在一帧内（剩余 1–3ms 是分节重建 DOM 的常数开销）；涉及网络的场景不再等往返。三处都不再出现读取文案，模型页每次打开只发一个请求。

## 后果

- 收益：插件拿到与页面解耦的数据归属，冷开、刷新、重开都不等网络；其他插件页面可以按同样方式迁移。
- 代价：显示的可能是上一次的值，直到后台核对返回（本机与远程都在一次往返内）；写操作依赖服务端 revision 校验兜底。
- 不做：服务端失效推送（现有可见期间轮询与打开时核对已够用）；对带草稿的插件表单做保活。
