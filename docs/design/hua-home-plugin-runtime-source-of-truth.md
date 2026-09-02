# hua-home 插件运行事实

状态：2026-09-02 已在 `hua-home` 实机核验。

## 1. 只认 hua-home

插件线上事实按下面顺序取证：

1. `/srv/data/services/akashic/activation/active.json`：当前 Core exact release commit。
2. `/srv/data/services/akashic/state/plugin-home/manifest.toml`：安装身份和 enabled 状态。
3. `plugin-home/cache/<marketplace>/<plugin>/.pointers.json`：外部插件的 exact stable/latest artifact。
4. `plugin-home/cache/.../.artifacts/<identity>/akashic.plugin.toml` 与 entrypoint：V3 admission
   identity 和真实模块。
5. `/srv/data/services/akashic/runtime-sources/<active-commit>/plugins/`：该 release 实际携带的
   builtin 插件源码。
6. `akashic-core` 容器进程、日志、snapshot/generation 证据：证明声明已启动，不用安装事实代替
   live 行为。

开发机路径都不是运行事实：

- `/home/huashen/.akashic-plugin` 只是从 hua-home 拉取的可删镜像；不能反向同步到服务端，
  不能用其 mtime、lock 或进程状态证明线上状态。
- `/home/huashen/.akashic` 是历史开发 workspace；不得用于 fleet、manifest、plugin-data、MCP
  或 generation 审计。废弃的本地 `workspace/mcp` 已删除。
- 当前 Git worktree 的 `plugins/` 是待发布源码。只有 exact commit 激活到 hua-home 后，才成为
  builtin 运行事实。
- `/mnt/data/coding/akashic-plugin/<name>` 是外部插件 canonical 开发仓库；它拥有修改，但不代表
  stable artifact 已安装或已启用。

## 2. 固定查找方法

先查 release 和 manifest：

```bash
ssh hua-home 'cat /srv/data/services/akashic/activation/active.json'
ssh hua-home 'sed -n "1,240p" /srv/data/services/akashic/state/plugin-home/manifest.toml'
```

再查 exact pointer，不扫描开发机旧 cache：

```bash
ssh hua-home 'find /srv/data/services/akashic/state/plugin-home/cache \
  -name .pointers.json -type f -print | sort'
ssh hua-home 'sed -n "1,80p" \
  /srv/data/services/akashic/state/plugin-home/cache/github/feed/.pointers.json'
```

最后核对真实 runtime：

```bash
ssh hua-home '~/.local/bin/akashic-release doctor'
ssh hua-home 'docker ps --filter name=akashic-core --format "{{.Names}} {{.Status}}"'
ssh hua-home 'docker exec akashic-core ps -eo pid,args'
ssh hua-home 'docker logs --since 30m akashic-core 2>&1 | tail -200'
```

需要在开发机只读分析 artifact 时，先重建镜像：

```bash
rsync -a --delete --exclude=.publication.lock \
  hua-home:/srv/data/services/akashic/state/plugin-home/ \
  /home/huashen/.akashic-plugin/
```

该命令的方向只能是 `hua-home → 开发机`。镜像同步后至少核对两端 `manifest.toml` SHA-256，
再读取 pointer；不要把 rsync 成功当作 runtime ready。

## 3. 2026-09-02 exact snapshot

- Core release：`8304a02420a98ba3cd4600d983f552422186e5b3`
- manifest SHA-256：`7c9f8f274a0ea4b274d1a1227c6d978d53801259e8d7e37095f1ea0934d612bf`
- enabled manifest entries：33（17 builtin + 16 external）
- external artifact directories：40；逐个 static manifest + entrypoint 扫描后 non-V3 为 0
- 所有 24 个外部 plugin identity 的 stable 与 latest 当前相同

### 3.1 Enabled builtins

| Plugin | Release source |
|---|---|
| akasha | `plugins/akasha` |
| codex | `plugins/codex` |
| compaction | `plugins/compaction` |
| computer | `plugins/computer` |
| conversation-ui | `plugins/conversation_ui` |
| drift | `plugins/drift` |
| eventmail | `plugins/eventmail` |
| markdown_memory | `plugins/markdown_memory` |
| models | `plugins/models` |
| openai-compatible | `plugins/openai_compatible` |
| opencode-go | `plugins/opencode_go` |
| runtime-ui | `plugins/runtime_ui` |
| scheduler | `plugins/scheduler` |
| shell-ui | `plugins/shell_ui` |
| subagent | `plugins/subagent` |
| wake | `plugins/wake` |
| workbench-ui | `plugins/workbench_ui` |

表中的相对路径必须接在 active release source 后面，不能接当前开发 worktree。

### 3.2 Enabled external plugins

| Plugin | Version | Stable artifact | Runtime declarations |
|---|---:|---|---|
| calendar@github | 3.2.1 | `3.2.1-9997353cdebc1885-restaged-c38b4ef82a0d4980` | MCP 1, process 1 |
| citation@github | 1.0.0 | `1.0.0-a886c74c55c4ef40-restaged-7a161afaf9af462d` | - |
| emotion@github | 3.0.4 | `3.0.4-d828fd7ec97e027b` | - |
| feed@github | 3.1.4 | `3.1.4-fd74018c2a397fcc` | MCP 1 |
| fitbit@github | 3.2.2 | `3.2.2-e0eda11d822e2ca0` | MCP 1, process 1 |
| github-watch@github | 3.0.0 | `3.0.0-b9266ab3ca9932c0` | - |
| huayue-skills@github | 1.1.0 | `1.1.0-65273781113a2305` | Skill |
| meme@github | 1.0.1 | `1.0.1-c185ea7a3847d67a` | - |
| observe@github | 1.4.1 | `1.4.1-09214c23f287f659` | - |
| plugin_undo@github | 2.0.0 | `2.0.0-86941208ea931308-restaged-d023109d29594540` | - |
| proactive_feedback@github | 3.0.1 | `3.0.1-d9d90fd4d3027d44` | - |
| setup_helper@github | 2.0.0 | `2.0.0-3d9671bfee523e78-restaged-64ee4474b9094010` | - |
| shell_restore@github | 2.0.0 | `2.0.0-d9b9e17c7e783463-restaged-f9d6e86f61d74035` | - |
| shell_safety@github | 2.0.0 | `2.0.0-5230f8ac8aec5216-restaged-821e8a5103414ea0` | - |
| status_commands@github | 2.0.0 | `2.0.0-8d119e8cfa53bd91-restaged-300683768df04d9a` | - |
| steam@github | 3.2.1 | `3.2.1-a0fda0602185a0a4-restaged-959c3b8e1d654b84` | MCP 1 |

### 3.3 Installed but disabled external identities

`content-wake-formal` marketplace 中以下 8 个 identity 已禁用，但 artifact 仍是 V3：

- calendar 3.1.0
- emotion 3.0.0
- feed 3.1.1
- fitbit 3.1.0
- github-watch 3.0.0
- observe 1.4.0
- proactive_feedback 3.0.0
- steam 3.1.0

禁用不等于 V2，也不授权删除。只有 pointer/reference、回滚集合和恢复需求都核清后，才能把它们
作为独立 GC 任务处理。
