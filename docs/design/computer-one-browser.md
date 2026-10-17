# Computer 精简为一个浏览器

决策见 [0123](../decisions/0123-computer-is-one-browser.md)。本文写调用链、改动范围和验收方法。

## 1. 容器里还剩什么

| 进程 | 作用 | 何时运行 |
|---|---|---|
| gateway（node） | 控制服务、OpenCLI 定时器、Agent 调用入口 | 常驻 |
| OpenCLI daemon | 宿主 `opencli` 的连接端 | 常驻 |
| Xvnc :99 | 画面缓冲，初始 1280×800 | 唤醒后 |
| matchbox | 窗口铺满屏幕、跟随 RandR，无标题栏 | 唤醒后 |
| Chromium | 个人 profile；匿名 Context 也在这里 | 唤醒后 |
| selkies | H.264 编码；有人看才编码 | 唤醒后 |

删掉的：桌面会话、桌面输入 driver（`sky`）、headless Chromium、`/view/frame` 截图轮询、
`/view/input` 匿名输入、`display` 端口（6080）。

## 2. 标签与画面

```text
Agent: browser.tabs.new()
  └─ Target.createTarget {newWindow, background}
       └─ Screen.place：setWindowBounds fullscreen → 若有观看中的标签，重新激活它

面板标签条点击
  └─ POST /api/dashboard/computer/show {target}
       └─ gateway POST /view/show → Target.activateTarget → Screen.front = target
            └─ 状态流推送 targets[].front，面板以它为准（乐观更新，失败回滚）
```

- `GET /targets` 返回 `{id, title, url, anonymous, agent, kept, front, sessionId}`；
  `id` 为 `main:<tabId>` 或 `<browserId>:<tabId>`。
- 关掉最后一个个人标签时 `recoverFront` 补开空白页；Chromium 带 `--keep-alive-for-test`，
  不因窗口全关而退出。
- 面板跟随 Agent：Agent 新开标签时画面切过去；用户手动选过标签后，本轮不再跟随，
  Agent 开始下一轮时恢复。同轮后续工具调用不恢复跟随；隐藏或被顶下来的面板不自动切页。
  空白页显示“新标签页”。

## 3. 输入与只读

```text
面板 ──WebSocket──> Dashboard 代理 ──> selkies（总是 role=controller）
                     │
                     ├─ 已接管：原样转发
                     └─ 未接管：只放行 SETTINGS、r,、s,、_arg_fps、_f、START/STOP_VIDEO、
                                REQUEST_KEYFRAME、CLIENT_FRAME_ACK、LOST_FRAME、_gz；其余丢弃
```

- 屏幕尺寸由 selkies 按面板尺寸调整（`r,宽x高`），matchbox 让窗口跟着铺满。
- 视口截图前记下 CSS 视口尺寸和像素比例（`innerWidth`、`innerHeight`、`devicePixelRatio`）。坐标动作执行前这些值已变，
  就报错并要求重新截图；没截过图的标签和按元素的动作不检查。尺寸查询失败也报错，
  不把未知尺寸当作安全坐标。内部诊断截图不更新这条记录。
- 第二个窗口连上后，selkies 断开前一个；前一个面板收到 `superseded`，
  提示“另一个窗口正在看这个浏览器”，不自动抢回。
- 人工接管、暂停 Agent 的规则沿用 0092。人工键鼠经 selkies 送到屏幕最前的窗口；
  网关旧的 `/input`（driver 基线用）把键鼠转为 CDP 输入，发到 `Screen.front` 对应的标签。
- 接管期间切到另一个匿名浏览器的标签，接管改为持有那个实例，不被闲置回收。
- 页面自己弹出的窗口（`window.open`、OAuth）成为 `Screen.front`。

## 4. 匿名浏览器

`agent.browsers.create()` → 主连接上 `Target.createBrowserContext` → 新 `BrowserBackend`
共用同一块屏幕。主浏览器的目录按 `contexts` 排除匿名标签。关闭时
`Target.disposeBrowserContext`，标签一起消失。上限 8 个，闲置 10 分钟回收，Turn 结束关闭。

## 5. 并发

`actives` 按 Session 记录正在执行的调用：不同 Session 并行；同一 Session 第二个调用直接报
“This Session already has a running Computer call”。暂停与恢复作用于全部调用。
输入按 `owner = session_id` 记录，取消只释放该 Session 按住的键鼠。
活动文件由一个写入队列保存，任何调用仍在执行时 `active` 都为真。

## 6. OpenCLI 刷新

刷新经 `agentOperation` 等人工交还后开始，拿到后记下个人页面和接管版本。结束时清理新增、
且不属于 Agent 的普通页面，保留弹窗。刷新期间有人接管（即使已经交还）、或收尾时仍有
Agent 调用时不清理。刷新与 Agent 调用可以并行。此前实测不关时，刷新留下 8 个站点页面，
常驻内存从约 300MB 涨到 1.45GiB。

## 7. 验收方法

1. 容器内：`docker/computer/test/lifecycle.test.mjs`（每例前重启运行时）。
2. 真实应用：本地应用 → `scripts/computer_ui_fixture.py --endpoint` → ssh 隧道 → 测试容器；
   Playwright 打开工作台“浏览器”，检查标签条、尺寸跟随、只读、跟随 Agent。
3. 真实模型任务：dsv4.1 在应用里打开网页并读取内容、点进二级页面。
4. MiniWoB 配对：`plugins/computer/bench/run.py`，同模型、同任务、同 seed，main 与本栈交替运行。

`scripts/computer-e2e/run.mjs` 验证并发接纳、匿名隔离、真实 H.264 切页、只读过滤、接管、
同 Turn 手动选择、窗口缩放和 profile 重启保留；`input.mjs` 验证拖拽、释放和截图，
`idle.mjs` 验证闲置回收。运行方式见该目录的 README。
`benchmark/computer_driver` 的 `sky` 适配只适用于 0123 之前的镜像。

## 8. 回退

回退使用上一版插件代码及其固定镜像。镜像同时升级 Chromium 时，旧浏览器可能不能安全读取
新 profile；先停写并备份 Computer 的 `/data`，恢复旧镜像时按备份恢复该目录，不能在线覆盖。
容器停止由 shell 统一收尾：tini 只通知 shell，shell 等 gateway 关闭 Chromium 并落盘后，
才停止 OpenCLI 和 D-Bus。不能把停止信号同时发给整组进程，否则外层提前退出会丢失刚写入的登录数据。
