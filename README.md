[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# Akashic Agent

Akashic 是一个会主动找你的 AI 伙伴。它可以对话，也能根据订阅的信息源判断何时主动发消息，并在空闲时执行后台任务。

## 快速开始

只需要 [Docker](https://docs.docker.com/get-docker/)（Compose 2.20+）。Windows 和 macOS 使用 Docker Desktop 即可。

```bash
mkdir akashic && cd akashic
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/compose.yaml
docker compose up -d --wait
```

首次启动会拉取镜像并安装默认功能，需要几分钟。命令返回后打开 <http://localhost:2236>，在“功能设置”→“初始配置”里连接模型，然后开始对话。

启动失败时查看 `docker compose logs --tail=200`，处理后重新运行同一条启动命令。

## 日常使用

```bash
docker compose stop            # 停止
docker compose up -d --wait    # 启动
docker compose logs -f         # 查看日志
```

- 配置、会话和插件都保存在 `data` 数据卷里。**不要运行 `docker compose down -v`，它会删除所有数据。**
- 端口 2236 被占用时，在同一目录创建 `.env`，写入 `AKASHIC_PORT=2237`，再重新启动。
- 更新：先停止并备份数据，再下载新版 `compose.yaml` 覆盖旧文件，运行启动命令。

```bash
docker compose stop
docker run --rm -v akashic_data:/data -v "$PWD":/backup alpine tar czf /backup/akashic-data.tgz -C /data .
curl -fsSLO https://github.com/kachofugetsu09/akashic-agent/releases/latest/download/compose.yaml
docker compose up -d --wait
```

数据卷名是“目录名_data”，上例目录为 `akashic`。

## 通过域名访问

网页默认只在本机开放：在运行 Akashic 的机器上打开 `http://localhost:2236` 即可，不需要额外配置。

想在其他设备上通过域名访问时，用 Cloudflare Tunnel、nginx、Caddy 等工具把域名转发到这台机器的 `localhost:2236`。

> [!WARNING]
> 网页本身**没有登录**，并且可以执行 Shell 和读写文件。公开到域名时，必须在前面加一层认证，例如 Cloudflare Access 或反向代理的 basic auth。不要把 Compose 的端口改成 `0.0.0.0:2236` 直接开放。

## 其他启动方式

<details>
<summary>不使用 Docker：<code>./start</code>（Linux / WSL2）</summary>

需要 Git、Python 3.12+ 和 Node.js 20+。

```bash
git clone --branch v0.2.0 https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
./start
```

首次运行会安装依赖并构建网页，就绪后自动打开 <http://127.0.0.1:2236>。按 Ctrl+C 停止；`--no-browser` 不打开浏览器，`--port 2237` 换端口。运行数据在 `~/.akashic`。

</details>

源码构建镜像、开发调试与升级细节见[启动与部署说明](./docs/design/product-startup.md)；已有 Host Bridge 部署见[部署操作手册](./docs/design/operator-deployment.md)。

## 文档

| 主题 | 文档 |
| --- | --- |
| 插件开发与安装 | [插件能力手册](./docs/design/plugin-v3-capabilities.md)；Agent 自己扩展能力用内置 Skill [`develop-akashic-plugin`](./skills/develop-akashic-plugin/SKILL.md) |
| 手机接入 | [Android Shell](https://github.com/kachofugetsu09/akashic-android-shell) |
| 记忆 | [记忆手册](./_handbook/memory-markdown.md) |
| Python 客户端与协议 | [Python SDK](./sdk/python/README.md)、[协议 schema](./schema/app-server-v2.json) |
| 数据位置与恢复 | [持久状态地图](./docs/design/persistence-state-map.md) |
| 项目需求与开发流程 | [工作手册索引](./docs/INDEX.md) |
