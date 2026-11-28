[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# Akashic Agent

Akashic 是一个会主动找你的 AI 伙伴：能聊天，能根据你订阅的信息主动发消息，空闲时还会在后台帮你做事。

## 安装

Linux 和 Windows 的 WSL2 推荐方式一；macOS、或者只想让 Akashic 待在容器里，用方式二。

### 方式一：Linux / WSL2 用户级安装（推荐）

Akashic 的核心跑在容器里，Shell 和文件工具以你的身份直接在这台电脑上执行，所以能读写你的文件、用你装好的命令。需要：当前用户能用的 Docker、Git、[uv](https://docs.astral.sh/uv/)，以及 systemd 用户服务（WSL2 要在 `/etc/wsl.conf` 里开启 systemd）。不需要 root。

```bash
curl -fsSLO https://raw.githubusercontent.com/kachofugetsu09/akashic-agent/main/scripts/install_host_bridge.py
python3 install_host_bridge.py install
```

最后会打印一个登录链接。要装扩展包时需要访问 GitHub，可以加 `--proxy http://host.docker.internal:7897`（端口换成你代理软件的，并打开“允许局域网连接”）。

- 数据在 `~/.local/share/akashic`，两个用户服务是 `akashic-host-bridge` 和 `akashic`，开机自动启动。
- 再拿登录链接：`python3 install_host_bridge.py login`。停止：`systemctl --user stop akashic akashic-host-bridge`。
- 升级：重新运行 `install`，数据保留。
- **命令不经拦截**：模型让它执行的 Shell 命令会直接以你的身份运行，和你自己在终端里敲一样。在意的话用方式二。

### 方式二：Docker 沙箱（Windows、macOS、Linux 都可以）

先安装 [Docker Desktop](https://docs.docker.com/get-docker/)，并打开它。然后在终端里依次运行：

```bash
mkdir akashic
cd akashic
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/compose.yaml
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/chromium-seccomp.json
docker compose up -d --wait
docker compose exec akashic python main.py web-login
```

Windows 的 PowerShell 里把两行 `curl` 换成 `curl.exe`。

第一次启动只装核心，大约 1 分钟，主要是下载镜像；期间终端只显示等待；想看进度，另开一个终端运行 `docker compose logs -f`。其余功能在网页引导里挑选安装。最后一条命令会打印一个登录链接，在浏览器里打开它就能进入。

如果报错 `address already in use`，说明 2236 端口被别的程序占了：在同一目录创建 `.env` 文件，写一行 `AKASHIC_PORT=2237`，再运行一次 `docker compose up -d --wait`。

几件要知道的事：

- **代理**：装扩展包要访问 GitHub。需要代理时，在同一目录的 `.env` 里写一行 `AKASHIC_PROXY=http://host.docker.internal:7897`（端口换成你代理软件的），并在代理软件里打开“允许局域网连接”，再运行一次 `docker compose up -d --wait`。不要写 `127.0.0.1`，那在容器里指的是容器自己。
- **时区**：容器默认用 UTC。在 `.env` 里写一行 `TZ=Asia/Shanghai`（换成你所在的时区）。
- **电脑重启后不会自动启动**：先打开 Docker，再在 `akashic` 目录里运行 `docker compose up -d --wait`。
- **Akashic 能碰到的范围**：它的 Shell 和文件工具只在容器里运行，看得到的是容器和数据卷，看不到你电脑上的其他文件。命令不经过拦截直接执行，所以不要把重要目录挂进容器。

### 方式三：不用 Docker（Linux 或 WSL2）

需要 Git、Python 3.12+ 和 Node.js 20+。

```bash
git clone --branch v0.3.0 https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
./start
```

第一次运行会在终端里问你装哪些功能，直接回车就是全部安装，大约需要 10 分钟。装好后浏览器会自动打开，并且已经登录。按 Ctrl+C 停止。

## 第一次打开

1. **设置密码**：页面会让你设置一个自己的密码，至少 10 个字符。以后在其他设备上登录时使用。
2. **跟着引导走**：
   - **连接模型**（必须）：选一个模型服务，填入它的 API Key：内置 DeepSeek、智谱 GLM 等预设；其他兼容 OpenAI 接口的服务选“自定义 API”，再填地址。
   - **选择功能**（用 Docker 时才有这一步）：列出官方扩展包，每个都写明大小和依赖。默认全部安装，不需要的取消勾选即可。安装要几分钟，页面会显示装到第几个；也可以先关掉引导去聊天。
   - 聊天软件、记忆等后面几步都可以先跳过，以后在“功能设置”里随时开。
3. **开始对话**。

想要更多功能，可以直接对 Akashic 说“帮我装上某某功能”，也可以在“功能设置”→“插件”里安装。

## 插件与扩展

Akashic 把“核心闭环”和“功能扩展”拆开维护：

- **核心仓库（本仓库，Core）**：只保留长程主循环（账本、上下文、压缩、基础文件与 Shell）、Web 宿主与包管理，更新 Core 需要重新部署并重启一次。
- **官方扩展仓库（[akashic-plugins](https://github.com/kachofugetsu09/akashic-plugins)，`extra`）**：浏览器、记忆、值班、子任务、画布、手机端侧等非核心功能都在这里，按单个插件的 `release/<插件名>` 分支发布，Core 镜像默认不内置。
- **安装与热更新**：在网页端“功能设置”→“插件”→“可安装”里可一键安装；也可以让 Akashic 帮忙装，或在容器/终端里用 `akashic pkg` 管理（`list` / `groups` / `install <名|@组>` / `upgrade`）。官方扩展插件与私人插件都支持**热安装与热更新**，改完或升级不需要重启 Core。
- **私有源**：自己的插件可以放进一个和官方扩展仓库同样结构的仓库（私有仓库也行），用 `pkg source add <名称> <地址>` 登记成来源，可以登记多个；安装和更新跟官方插件走同一套命令，详见[使用手册](./docs/user-guide.md#私有源)。
- **自建插件**：想让 Akashic 自己扩展新能力时，它会按内置 Skill [`develop-akashic-plugin`](./skills/develop-akashic-plugin/SKILL.md) 写插件源码并热加载。

## 手机

- **[Akashic Android Shell](https://github.com/kachofugetsu09/akashic-android-shell/releases)**：Android 原生外壳，在手机上用 Akashic，不依赖 Google 推送也能收到消息通知。
- **让 Akashic 操作手机**：在 Shell 里开启后台手机控制，再在「选择功能」里装上「手机端侧」（phone-use、phone-assist）。它在一块独立的后台虚拟屏上操作，不占用你正在看的屏幕；首次启用需要在 Wi-Fi 下开一次无线调试。

## 常用操作

| 想做的事 | Docker（在 `akashic` 目录里运行） | 不用 Docker（在 `akashic-agent` 目录里运行） |
| --- | --- | --- |
| 停止 | `docker compose stop` | 按 Ctrl+C |
| 再次启动 | `docker compose up -d --wait` | `./start` |
| 看日志 | `docker compose logs --tail=200` | 终端里会显示日志文件的位置 |
| 拿登录链接 | `docker compose exec akashic python main.py web-login` | `./start --login` |

**不要运行 `docker compose down -v`**，它会删除你的全部数据。

更新、备份、换端口、从其他设备访问、忘记密码等操作，见[使用手册](./docs/user-guide.md)。

## 出问题了

1. 先看日志（见上表）里最后的报错。
2. 处理后重新运行启动命令。中途失败了，直接再运行一次即可，已完成的步骤不会重做。
3. 还是不行，就带上日志的最后 200 行，到 [Issues](https://github.com/kachofugetsu09/akashic-agent/issues) 提问，或者加[交流群](./COMMUNICATION.md)。

## 给 AI 助手的安装说明

如果你是正在帮用户安装 Akashic 的 AI 助手，按下面做：

- **Linux / WSL2 用户级安装**：按“方式一”运行 `python3 install_host_bridge.py install`。
  - 返回 0 并打印登录链接就表示已就绪；非 0 时它会说明缺什么（Docker 权限、uv、systemd 用户服务）。没有 systemd 时加 `--no-systemd`。
  - 换端口用 `--port 2237`。
- **Docker**：按“方式二”的命令依次运行。
  - `docker compose up -d --wait` 返回 0 就表示已就绪。如果返回非 0，运行 `docker compose logs --tail=200 akashic` 查看原因。
  - 把 `web-login` 打印的链接交给用户。这个链接 2 分钟内有效，只能用一次。
- **源码**：运行 `./start --no-browser --yes`。
  - `--yes` 表示首次安装不提问，只装核心，扩展包留给用户在网页引导里挑选。要一次装全，用 `./start --no-browser --yes --profile full`。
  - 看到 “WebUI 已就绪” 就表示完成。进程会一直运行，需要放到后台。
  - 登录链接用 `./start --login` 获取，交给用户。
- **换端口**：
  - Docker：在 `compose.yaml` 同目录创建 `.env`，写入 `AKASHIC_PORT=2237`。
  - 源码：运行 `./start --port 2237`。
- **不要替用户做的事**：
  - 不要替用户填写 API Key 或密码，这两步留给用户在网页上完成。
  - 不要运行 `docker compose down -v`。

## 文档

| 主题 | 文档 |
| --- | --- |
| 使用手册（更新、备份、远程访问、登录） | [使用手册](./docs/user-guide.md) |
| 官方插件仓库（extra） | [akashic-plugins](https://github.com/kachofugetsu09/akashic-plugins) |
| 插件开发与安装 | [插件能力手册](./docs/design/plugin-v3-capabilities.md)、[`develop-akashic-plugin`](./skills/develop-akashic-plugin/SKILL.md)、[`plugin-system`](./plugins/plugin_update/skills/plugin-system/SKILL.md) |
| 手机接入 | [Android Shell](https://github.com/kachofugetsu09/akashic-android-shell) |
| Python 客户端与协议 | [Python SDK](./sdk/python/README.md)、[协议 schema](./schema/app-server-v2.json) |
| 源码构建镜像、开发与部署 | [启动与部署说明](./docs/design/product-startup.md)、[部署操作手册](./docs/design/operator-deployment.md) |
| 项目需求与开发流程 | [工作手册索引](./docs/INDEX.md) |
