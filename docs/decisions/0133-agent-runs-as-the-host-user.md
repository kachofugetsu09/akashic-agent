# 0133 · Agent 的执行与网络等同宿主用户本人

- 状态：proposed / implemented for review
- 日期：2026-10-24
- 关联：延续 [0032](0032-host-bridge-preserves-host-equivalent-execution.md) 的"等同 SSH 登录"；修订 [0014](0014-shell-uses-unified-execution.md) 中"不改变命令安全校验、网络限制"一句；与 [0017](0017-one-person-companion-security-boundary.md) 一致

## 背景

Shell 经 Host Bridge 在宿主上以用户身份执行（0032），能读写用户的全部文件、起任意进程。在这个前提下，Shell 前面还挂着一层按命令首词的文本检查：

- `curl`/`wget` 访问 `127.0.0.1` 和私网地址被拒，`localhost` 却能过；
- 带 `@文件`、`-F`、`-T` 的请求被拒；
- `nc` 整个禁用；
- 子任务 `scripting` 的 Shell 禁止全部 `curl`/`wget`，但 `pip`、`python` 照常联网；
- 只检查第一个词，`cd x && curl …`、`python3 -c …` 都能绕过。

这层检查没有记录理由。`web_fetch` 上同样的限制早已撤掉（0017）。它挡住的是自测本机开发服务、调本地 API 这些正常操作，防不住任何东西。

## 决定

- 删除 Shell 的命令文本检查（`plugins/shell/shell_security.py`）：没有禁用命令，不限制网络目标与上传参数。
- 删除 Shell 设置里的 `allow_network` 与 `restricted_dir`；子任务不再按 profile 关网。已保存的旧工具配置读取时丢掉这两个键。
- 子任务 `scripting` 的约束只剩"写入仅限任务目录"。
- 后台运行（`run_in_background`）默认不设硬截止：dev server、长任务一直活到 `task_stop`、会话收尾或宿主停止。前台命令仍默认 4 小时，显式 `timeout` 不再有上限；协议里 `hard_timeout_s = 0` 表示不设截止。
- Computer 地址栏可以打开 `file://`；输入以 `/` 开头的绝对路径时按本地文件打开。`javascript:` 等仍不接受。
- Computer 浏览器等同宿主上的一个浏览器：Workload 可以声明 `host_egress`，Controller 把 Host Bridge 的 SOCKS5 unix socket（`<Workload 运行目录>/host-egress/socks.sock`）所在目录挂进容器；computer 的网关把它转成容器本机端口，Chromium 以 `<-loopback>` 让 localhost 也走它。连接由宿主上的 Bridge 发起，`localhost`、内网和域名解析都和宿主一致。容器里只有 OpenCLI 守护进程留在本机直连。
- 出口 socket 由 Bridge 里的唯一 owner 看守：每 2 秒按 inode 核对监听路径，路径丢失就在原目录重新绑定（已建立的连接不受影响，Bridge、Core、Workload 都不重启）；路径被普通文件、其他用户或仍在监听的 socket 占用时不删除不覆盖，报告不可用并等它空出。Probe/Inspect 的能力里用 `host-egress` / `host-egress-unavailable` 单独报告出口状态，RPC 健康不代表出网入口可用。
- 不改用 Docker 的宿主网络：那会占用宿主端口、和用户已有的服务冲突，core 访问 computer 和画面直连还要靠宿主防火墙放行。unix socket 不占端口、不经防火墙，换一台机器、换一套防火墙都不用额外配置。开发服务不再需要监听网关地址，也不再需要防火墙为容器网段开放行段。

## 代价

声明了 `host_egress` 的 Workload 能访问宿主用户能访问的所有网络服务，包括只监听 localhost 的服务；只有 computer 声明它。Bridge 停止时 computer 浏览器断网（Shell 也同时不可用）。浏览器的外网流量多经宿主一跳。

后台进程不会自己到时结束，忘了停的 dev server 会一直占着端口和内存，直到会话收尾或用户在工作台里停掉。

模型可以让 Shell 访问本机和内网的任意服务，这与用户本人登录宿主后能做的事相同。需要更强的隔离时，应把执行放进单独的主机或容器，而不是在命令文本上加规则。
