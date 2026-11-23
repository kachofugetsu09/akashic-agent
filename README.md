[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# Akashic Agent

Akashic 是一个会主动找你的 AI 伙伴：能聊天，能根据你订阅的信息主动发消息，空闲时还会在后台帮你做事。

## 安装

两种方式选一种。不确定选哪个，就用 Docker。

### 方式一：Docker（Windows、macOS、Linux 都可以）

先安装 [Docker Desktop](https://docs.docker.com/get-docker/)，并打开它。然后在终端里依次运行：

```bash
mkdir akashic && cd akashic
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/compose.yaml
docker compose up -d --wait
docker compose exec akashic python main.py web-login
```

第一次启动要下载和安装，大约需要 10 分钟，期间终端只显示等待；想看进度，另开一个终端运行 `docker compose logs -f`。最后一条命令会打印一个登录链接，在浏览器里打开它就能进入。

如果报错 `address already in use`，说明 2236 端口被别的程序占了：在同一目录创建 `.env` 文件，写一行 `AKASHIC_PORT=2237`，再运行一次 `docker compose up -d --wait`。

### 方式二：不用 Docker（Linux 或 WSL2）

需要 Git、Python 3.12+ 和 Node.js 20+。

```bash
git clone --branch v0.3.0 https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
./start
```

第一次运行会安装依赖，大约需要 10 分钟。装好后浏览器会自动打开，并且已经登录。按 Ctrl+C 停止。

## 第一次打开

1. **设置密码**：页面会让你设置一个自己的密码，至少 10 个字符。以后在其他设备上登录时使用。
2. **跟着引导走**：只有“连接模型”是必须的。选一个模型服务，填入它的 API Key：内置 DeepSeek、智谱 GLM 等预设；其他兼容 OpenAI 接口的服务选“自定义 API”，再填地址。聊天软件、记忆等后面几步都可以先跳过，以后在“功能设置”里随时开。
3. **开始对话**。

想要更多功能，可以直接对 Akashic 说“帮我装上某某功能”，也可以在“功能设置”→“插件”里安装。

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

- **Docker**：按“方式一”的命令依次运行。
  - `docker compose up -d --wait` 返回 0 就表示已就绪。如果返回非 0，运行 `docker compose logs --tail=200 akashic` 查看原因。
  - 把 `web-login` 打印的链接交给用户。这个链接 2 分钟内有效，只能用一次。
- **源码**：运行 `./start --no-browser --yes`。
  - `--yes` 表示首次安装不提问，装上全部扩展包。
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
| 插件开发与安装 | [插件能力手册](./docs/design/plugin-v3-capabilities.md)；也可以让 Akashic 用内置 Skill [`develop-akashic-plugin`](./skills/develop-akashic-plugin/SKILL.md) 自己扩展能力 |
| 手机接入 | [Android Shell](https://github.com/kachofugetsu09/akashic-android-shell) |
| Python 客户端与协议 | [Python SDK](./sdk/python/README.md)、[协议 schema](./schema/app-server-v2.json) |
| 源码构建镜像、开发与部署 | [启动与部署说明](./docs/design/product-startup.md)、[部署操作手册](./docs/design/operator-deployment.md) |
| 项目需求与开发流程 | [工作手册索引](./docs/INDEX.md) |
