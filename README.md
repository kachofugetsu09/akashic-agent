[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# Akashic Agent

Akashic 是一个会主动找你的 AI 伙伴。它可以对话，也能根据订阅的信息源判断何时主动发消息，并在空闲时执行后台任务。

## 普通启动

需要 Git、Python 3.12+ 和 Node.js 20+。适用于 Linux；Windows 使用 WSL2。

```bash
git clone https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
./start
```

启动器会打开 <http://127.0.0.1:2236>。首次启动自动安装 Python 依赖、构建网页和插件发行包、创建数据目录并安装默认功能。网页显示进度；失败后可以查看日志和重试。无需手动运行 `init`、`npm ci` 或插件安装命令。

以后仍运行 `./start`，同一版本会复用构建结果。终端按 Ctrl+C 停止服务；关闭浏览器不会停止服务。无桌面环境或 Agent 使用 `./start --no-browser --non-interactive`，失败返回非零退出码。端口被占用时可用 `--port 2237`。

运行数据默认在 `~/.akashic`（配置、workspace 和 plugin-home）；构建缓存位于 checkout 的 `.akashic-start`。试用独立实例可运行 `./start --state /path/to/empty-directory --port 2237`。启动器不会接管已有的手工安装，也不会覆盖关闭或卸载选择。不要删除运行数据来解决构建问题。

## Docker Compose 启动

只需 Docker 和 Compose v2 或更新版本。在仓库目录运行：

```bash
docker compose up -d
```

然后打开 <http://localhost:2236>。首次会构建镜像，因此需要等待镜像构建完成；网页出现后可查看安装默认功能的进度。无需模型 Key、宿主 Python/Node、Host Bridge、systemd 单元或手动创建网络。

构建默认从本仓库 `main` 取得固定提交；本地未提交改动不会进入镜像。需要复现某个版本时设置 `AKASHIC_REVISION=<完整的 40 位 commit>`。镜像内包含已构建的网页和插件包，重启不重新编译。`AKASHIC_PORT=2237 docker compose up -d` 可更换本机端口。

数据保存在此 Compose project 的 `data` 卷。独立实例使用不同的 `docker compose -p <名称>`；同一项目名称表示管理同一个实例。用 Docker Desktop 的启动、停止和日志操作管理服务，或运行 `docker compose stop` / `docker compose up -d`。`docker compose down` 保留数据；**不要加 `-v`，它会删除数据卷**。默认仅向本机开放网页，文件和 Shell 工具在容器内部运行。

## 在网页中开始使用

1. 在“初始配置”连接模型：选择服务商或登录方式，验证后保存。
2. 根据需要明确开启或关闭 Telegram、Akasha 情景记忆、Wake 主动联系；依赖未满足时页面会说明原因。
3. 进入对话。以后可在“功能设置”中修改配置，凭据不会在页面回显。

默认组合包含 Web 壳、工作台、对话和首次配置插件。尚未提供模型凭据时也能打开配置页面。

## 开发、升级和宿主机集成

启动器运行当前**已提交版本**的发行制品；有未提交改动时会提示使用开发入口。源码调试、版本升级及 Host Bridge 正式部署见[启动与部署说明](./docs/design/product-startup.md)。普通启动不隐式更新已有插件或批准数据迁移；换版本试用时使用独立数据目录，现有安装通过正式发布流程升级。

已有 hua-home / Host Bridge 发行继续按[部署操作手册](./docs/design/operator-deployment.md)管理。它与本页面向首次使用的 Compose 是不同的执行环境，不应同时操作同一份运行数据。

| 主题 | 文档 |
| --- | --- |
| 插件开发与安装 | [插件教程](./_handbook/plugins-tutorial.md) |
| 手机接入 | [Android Shell](https://github.com/kachofugetsu09/akashic-android-shell) |
| 主动推送 | [Proactive 指南](./_handbook/proactive-guide.md) |
| 记忆与 Drift | [记忆手册](./_handbook/memory-markdown.md)、[Drift 指南](./_handbook/drift-guide.md) |
| Python 客户端与协议 | [Python SDK](./sdk/python/README.md)、[协议 schema](./schema/app-server-v2.json) |
| 数据位置与恢复 | [持久状态地图](./docs/design/persistence-state-map.md) |
| 项目需求与开发流程 | [工作手册索引](./docs/INDEX.md) |
