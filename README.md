[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# Akashic Agent

Akashic 是一个会主动找你的 AI 伙伴。它可以对话，也能根据订阅的信息源判断何时主动发消息，并在空闲时执行后台任务。

## Docker Compose：直接使用发行版

只需 Docker 和 Compose 2.20 或更新版本，无需模型 Key、Git、宿主 Python/Node 或 Host Bridge。Windows 和 macOS 使用 Docker Desktop 的 Linux 容器模式；发行镜像支持 AMD64 和 ARM64。

1. 在[最新稳定发行版](https://github.com/kachofugetsu09/akashic-agent/releases/latest)的 **Assets** 下载 `compose.yaml`。
2. 创建一个专用目录（例如 `akashic`），将下载文件放入该目录。在该目录打开终端。
3. 运行：

```bash
docker compose up -d --wait --wait-timeout 600
```

首次自动拉取发行镜像、准备数据卷并安装默认功能。命令成功返回后打开 <http://localhost:2236/#onboarding> 进入“初始配置”。如果先打开了对话页，点击右上角“功能设置”→“初始配置”。配置页面能显示，即可返回“对话”；首次欢迎提示可选择“稍后再说”，然后在输入框写入草稿。完成这些步骤即启动成功；没有模型凭据时先配置模型，再开始对话。容器显示 `running` 或网页返回 HTTP 200 都不单独代表启动成功。

失败时运行 `docker compose logs --tail=200` 查看原因；安装过程也可在 Docker Desktop 日志中查看。解决错误后重复同一启动命令，**不要删除数据卷来重试**。下载的 Compose 固定到此次发行镜像的 digest，重建容器不会取得变化中的 `main`。

### 停止、重启与数据

```bash
docker compose stop
docker compose up -d --wait --wait-timeout 600
```

配置、会话、附件和插件选择保存在此 Compose project 的 `data` 卷。`docker compose down` 只删除容器和网络；**不要加 `-v`，它会删除数据卷**。保持目录和 project 名称不变，才能继续使用同一实例。独立试用使用新目录及 `docker compose -p <名称>`。

默认仅向本机开放网页，文件和 Shell 工具在容器内部运行。端口被占用时，在同一目录创建 `.env`，写入 `AKASHIC_PORT=2237`，重跑启动命令后访问 <http://localhost:2237/#onboarding>。无需手工创建 Docker 网络。

### 更新发行版

先停止服务并备份此实例的数据卷，再用新发行版的 `compose.yaml` 替换同一目录中的旧文件，运行上述启动命令。固定的镜像身份会触发新版本拉取；仅重启旧 Compose 不会更新代码。

新版内置代码会替换旧版本，外置版本、停用选择、配置和运行数据保留。启动前执行必要的 Core/内置 Yoyo；外置插件自行负责数据迁移。迁移失败时停止启动，按[部署手册](docs/design/operator-deployment.md)检查状态。恢复旧镜像无法撤销已经发生的数据迁移，恢复时使用升级前的数据卷备份。

## Docker Compose：在本地仓库构建

需要 Git、Docker 和 Compose 2.20+。仓库中的 `compose.yaml` 保留源码构建；它不依赖发行镜像。下列步骤构建 v0.2.0 的已推送源码提交，未提交改动不会进入镜像。其他源码版本同样需要明确指定完整提交。

Linux、macOS 或 WSL2：

```bash
git clone --branch v0.2.0 https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
printf 'AKASHIC_REVISION=%s\n' "$(git rev-parse HEAD)" > .env
docker compose up -d --build --wait --wait-timeout 600
```

Windows PowerShell（已安装 Git 和 Docker Desktop）：

```powershell
git clone --branch v0.2.0 https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
"AKASHIC_REVISION=$(git rev-parse HEAD)" | Set-Content .env
docker compose up -d --build --wait --wait-timeout 600
```

首次构建包含已完成的网页和插件包，后续同一镜像启动不重新编译。`.env` 保存源码提交，之后在新终端也能停止和重启。成功标准、日志、端口和数据管理与发行版相同。构建新发行版时先 `git fetch --tags`、`git switch --detach <新 tag>`，修改已有 `.env` 中的 `AKASHIC_REVISION` 为新的完整提交，再运行构建启动命令。保留 `.env` 中已有的端口设置；不要把提交设为浮动的 `main`，它可能复用旧构建层。源码调试见[开发入口](docs/design/product-startup.md#development-entry)。

## 普通启动

需要 Git、Python 3.12+ 和 Node.js 20+。适用于 Linux；Windows 使用 WSL2。

```bash
git clone --branch v0.2.0 https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
./start
```

终端会显示启动进度：首次自动安装 Python 依赖、构建网页和插件发行包、创建数据目录并安装默认功能。等服务和插件界面就绪后，启动器会打开 <http://127.0.0.1:2236>，并在终端打印访问地址。失败时会显示原因和日志路径，处理后重新运行同一命令。无需手动运行 `init`、`npm ci` 或插件安装命令。

以后仍运行 `./start`，同一版本会复用构建结果。终端按 Ctrl+C 停止服务；关闭浏览器不会停止服务。无桌面环境或 Agent 使用 `./start --no-browser`，失败返回非零退出码。端口被占用时可用 `--port 2237`。

运行数据默认在 `~/.akashic`（配置、workspace 和 plugin-home）；构建缓存位于 checkout 的 `.akashic-start`。试用独立实例可运行 `./start --state /path/to/empty-directory --port 2237`。启动器不会接管已有的手工安装，也不会覆盖关闭或卸载选择。不要删除运行数据来解决构建问题。

## 在网页中开始使用

1. 从“功能设置”→“初始配置”连接模型：选择服务商或登录方式，验证后保存。
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
