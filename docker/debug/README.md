# Docker 调试沙盒

这个目录用于临时调试真实入口，例如 Telegram 图片、多模态链路、独立 bot 配置。沙盒不会挂载宿主机 `HOME`，也不会挂载正式 `~/.akashic/workspace`。

```
host
  |
  +-- akashic-agent
      |
      +-- docker/debug
          |
          +-- Dockerfile
          +-- docker-compose.yml
          +-- entrypoint.sh
          +-- profiles
              |
              +-- default
                  |
                  +-- config.toml
                  +-- workspace
                  +-- home
                  +-- akashic.sock

container
  |
  +-- /app                 -> 当前代码
  +-- /sandbox/config.toml -> 调试 bot 配置
  +-- /sandbox/workspace   -> 调试 workspace
  +-- /sandbox/home        -> 容器 HOME
```

## 安全边界

- 默认调试配置只在 `docker/debug/profiles/default/config.toml`。
- 默认调试 workspace 只在 `docker/debug/profiles/default/workspace`。
- 容器内 `HOME` 是 `/sandbox/home`，不是宿主机 HOME。
- 启动脚本会拒绝 `/sandbox` 外的 config/workspace 路径。
- `profiles/` 已加入 `.gitignore`，不要提交调试 bot token 和测试记忆。

## 第一次配置

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup
```

这里填写专用 Telegram bot、模型 key 和多模态配置。不要填正式 bot。

## 启动调试 Agent

```bash
docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

此时向调试 Telegram bot 发消息或图片，所有会话和记忆都会进入 `docker/debug/profiles/default/workspace`。

## 多套调试配置

不同功能可以用不同 profile 保存配置和 workspace：

```bash
AKASHIC_DEBUG_PROFILE=multimodal docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug setup
AKASHIC_DEBUG_PROFILE=multimodal docker compose -f docker/debug/docker-compose.yml up akashic-debug
```

对应目录是 `docker/debug/profiles/multimodal/`。

## 连接调试 CLI

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug cli
```

CLI socket 固定为 `/sandbox/akashic.sock`，不会连接正式实例。

## 打开调试 Dashboard

```bash
docker compose -f docker/debug/docker-compose.yml run --rm --service-ports akashic-debug dashboard
```

宿主机访问 `http://127.0.0.1:2237`。

## 停止调试环境

```bash
docker compose -f docker/debug/docker-compose.yml down
```

这只会停止容器，不会删除当前 profile 目录。

## 清空调试 workspace

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug reset-workspace
```

这个命令只删除并重建当前 profile 下的 `workspace`，会保留当前 profile 下的 `config.toml`。

## 完全清理

```bash
docker compose -f docker/debug/docker-compose.yml down --remove-orphans
rm -rf docker/debug/profiles/default
```

完全清理后，下次需要重新运行 `setup`。
