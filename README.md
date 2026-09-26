[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# Akashic Agent

Akashic 是一个会主动找你的 AI 伙伴。它可以对话，也能根据订阅的信息源判断何时主动发消息，并在空闲时执行后台任务。

## 启动方式

下面两种方式适用于 Linux；Windows 用户请在 WSL2 中运行。启动后访问 <http://127.0.0.1:2236>，在 WebUI 中添加模型和认证信息。首次安装仍有终端步骤；[issue #774](https://github.com/kachofugetsu09/akashic-agent/issues/774) 跟踪将其收敛到少量命令和 WebUI 配置。

### 1. 普通启动：本地源码

需要 Git、Python 3.12 或更新版本、[uv](https://docs.astral.sh/uv/getting-started/installation/) 和 Node/npm。下面是**全新 workspace** 的流程；已有数据请先确认实际 workspace 和插件安装目录，不要当作首次安装重复执行。

```bash
git clone https://github.com/kachofugetsu09/akashic-agent.git
cd akashic-agent
uv venv
uv pip install -r requirements.txt -e sdk/python
npm ci
npm run build

# 第一次 setup 创建 Core 配置、workspace 和空插件选择。
uv run python main.py setup

# 安装默认插件组合；distribution 从已提交的 HEAD 构建。
dist_parent="$(mktemp -d)"
uv run python scripts/build_plugin_distribution.py \
  --revision HEAD --output "$dist_parent/distribution"
uv run python scripts/install_plugin_distribution.py \
  --distribution "$dist_parent/distribution" \
  --profile "$dist_parent/distribution/profiles/default.json" \
  --workspace "$HOME/.akashic/workspace" \
  --plugins-home "$HOME/.akashic-plugin" \
  --config "$PWD/config.toml"

# 第二次 setup 运行已安装插件的首次配置；保留现有 Core 配置。
uv run python main.py setup
uv run python main.py
```

两次 `setup` 之间要安装默认 profile：第一次只建立 Core 起点，第二次才会运行已安装插件的配置命令。向导询问是否覆盖已有 `config.toml` 时，按回车保留它。`main.py` 无参数时由 Supervisor 启动 Gateway 和 Web Shell。没有模型时，打开 2236 后按页面提示连接模型即可。

### 2. Docker Compose：已准备的正式发行

现有 Compose 文件运行**已经构建并激活的 release**。先按[部署操作手册](./docs/design/operator-deployment.md)准备镜像、`runtime.env`、Host Bridge、workspace 和外围服务网络；全新 checkout 目前不能直接运行 `docker compose up`。Docker Compose 需要 v2。

正式服务器通常由 `akashic-core.service` 管理，它内部运行 Compose。仅在该单元没有运行、Host Bridge 与外围服务已经就绪、需要手动启动同一 release 时，在宿主执行：

```bash
runtime_env="$HOME/.config/akashic-container/runtime.env"
runtime_checkout="$(sed -n 's/^AKASHIC_RUNTIME_CHECKOUT=//p' "$runtime_env")"
test -n "$runtime_checkout"
docker compose --project-name akashic-core --env-file "$runtime_env" \
  --file "$runtime_checkout/docker/host-runtime/compose.experiment.yaml" \
  --file "$runtime_checkout/docker/host-runtime/compose.external-services.yaml" \
  up
```

不要让手动 Compose 和 `akashic-core.service` 同时管理同一 workspace。正常由 systemd 托管时，使用 `systemctl start akashic-core.service`；首次发布和升级使用部署工具，不用手动 `compose up` 代替发行激活。

## 在 WebUI 中配置

打开 <http://127.0.0.1:2236>，进入“模型与认证”，添加 Provider、凭据和模型，并发送一次请求验证。模型连接保存在 workspace 中；页面不会回显已保存的密钥。安装更多插件时，先看[社区插件仓库](https://github.com/orgs/akashic-plugins/repositories)和[插件教程](./_handbook/plugins-tutorial.md)。

源码工作区保存代码；运行时 `<workspace>` 保存会话、记忆、附件和插件数据，默认在 `~/.akashic/workspace`。切换代码分支或重建容器时，不要把运行时 workspace 当作临时文件删除。[持久状态地图](./docs/design/persistence-state-map.md)说明各类数据的 owner 和恢复边界。

## 更多文档

| 主题 | 文档 |
| --- | --- |
| 正式发行、升级和恢复 | [部署操作手册](./docs/design/operator-deployment.md) |
| 插件开发与安装 | [插件教程](./_handbook/plugins-tutorial.md) |
| 手机接入 | [移动端接入手册](./_handbook/mobile-access.md) |
| 主动推送 | [Proactive 指南](./_handbook/proactive-guide.md) |
| 记忆与 Drift | [记忆手册](./_handbook/memory-markdown.md)、[Drift 指南](./_handbook/drift-guide.md) |
| Python 客户端与协议 | [Python SDK](./sdk/python/README.md)、[协议 schema](./schema/app-server-v2.json) |
| 项目需求与开发流程 | [工作手册索引](./docs/INDEX.md) |
