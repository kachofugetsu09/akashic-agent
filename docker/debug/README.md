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

## 上下文连续性探针

`context_probe.py` 用于复现一段固定纯聊天场景，自动记录用户输入、LLM 回复、工具调用、`RECENT_CONTEXT.md` 和 `memory2.db` 写入结果。

```
context probe
  |
  +-- profile
  |     |
  |     +-- config.toml
  |     +-- workspace
  |
  +-- phase1 chat
  |
  +-- manual consolidate
  |
  +-- phase2 chat
  |
  +-- final question
        |
        +-- markdown report
        +-- json report
```

从已启动的沙盒运行：

```bash
python docker/debug/context_probe.py \
  --profile default \
  --messages docker/debug/scenarios/sleepy_study_plan.json
```

自动重置、启动、运行并停止：

```bash
python docker/debug/context_probe.py \
  --profile v4flash-memory-window \
  --messages docker/debug/scenarios/sleepy_study_plan.json \
  --reset-workspace \
  --start-agent \
  --stop-agent \
  --quiet-agent \
  --disable-qq
```

`--disable-qq` 会在运行期间临时给当前 profile 的 `[channels.qq]` 加 `enabled = false`，结束后恢复原配置，适合只测 CLI 但该 profile 配了 QQ 的情况。

默认报告写到：

```text
docker/debug/profiles/<profile>/workspace/context-probe-<profile>.md
docker/debug/profiles/<profile>/workspace/context-probe-<profile>.json
```

自定义场景 JSON 格式：

```json
{
  "name": "sleepy_study_plan",
  "turns": [
    {
      "role": "user",
      "content": "前置闲聊"
    },
    {
      "action": "consolidate",
      "label": "after_signal",
      "force": false,
      "archive_all": false
    },
    {
      "role": "user",
      "content": "consolidate 后的杂音"
    },
    {
      "role": "user",
      "content": "最后问题",
      "final": true
    }
  ]
}
```

场景 JSON 只描述输入和流程，不写结果要求。报告只记录 observe 结果，不判定通过/失败。

内置样例在：

```text
docker/debug/scenarios/sleepy_study_plan.json
```

公开场景和 schema 都放在：

```text
docker/debug/scenarios/
```

这里的文件是稳定输入，可以提交；`docker/debug/profiles/<profile>/workspace/` 里的报告 JSON / Markdown 是运行产物，默认不提交。

兼容旧格式：

```json
{
  "phase1": ["第一段闲聊"],
  "phase2": ["consolidate 后的杂音"],
  "final_question": "最后问题"
}
```

## 完全清理

```bash
docker compose -f docker/debug/docker-compose.yml down --remove-orphans
rm -rf docker/debug/profiles/default
```

完全清理后，下次需要重新运行 `setup`。
