# hua-home 插件运行事实

状态：当前取证规则；第 3 节只保留 2026-09-02 的历史记录。

## 1. 维护范围只认 fleet，运行事实只认 hua-home

维护者明确保留的外置插件集合由 [akashic-fleet](https://github.com/kachofugetsu09/akashic-fleet)
当前分支的 `.gitmodules` 与 `plugins/` gitlink 定义。先固定 fleet commit，再沿其中的仓库和
版本核对消费者。查询、升级、API 清理或“全部外置插件”都只覆盖这个集合；内置插件另由
当前 Core 分发源码定义。目录、cache、历史安装清单和旧 PR 不能扩充维护范围。

`/mnt/data/coding/akashic-plugin/<name>` 只是可能保留历史源码的本地目录。只有 fleet
列出的仓库才是本次维护对象；非 fleet 仓库不因仍存在而需要迁移、兼容或重新设计。
保留历史源码和 plugin-data 不等于保留插件功能，也不授权删除这些文件。

线上运行证据与维护集合分别取证：

1. `/srv/data/services/akashic/activation/active.json` 与实际容器挂载：当前 Core release。
2. `state/workspace/runtime/plugin-stable.json` 的 `root_ref`：当前完整插件选择。
3. `state/workspace/runtime/plugin-archives/<root_ref>.json` 及其组件 descriptor：精确代码、
   配置引用和环境；由组件的 `code` 定位对应归档树，再读取 `plugin.py`。
4. 实际 live Root/Fiber、容器和工作流结果：证明选择已激活且消费者正常工作。

`state/plugin-home/manifest.toml` 只证明安装意图，cache 只证明制品存在；它们不能代替
完整 selection 和 live 行为。旧 stable/latest 指针与旧 `akashic.plugin.toml` 只作历史证据。
若生产环境存在非 fleet 残留，单独报告，不自动把它纳入受支持插件集合。

## 2. 固定查找方法

先取得当前 fleet 清单，不修改本地 checkout 或递归更新子模块：

```bash
git -C /mnt/data/coding/akashic-fleet fetch --no-recurse-submodules origin main
git -C /mnt/data/coding/akashic-fleet rev-parse origin/main
git -C /mnt/data/coding/akashic-fleet show origin/main:.gitmodules
git -C /mnt/data/coding/akashic-fleet ls-tree origin/main:plugins
```

再使用服务器上的 release、selection、归档 descriptor 和 live Root 取证。正式 workspace
路径为 `/srv/data/services/akashic/state/workspace`；必要时从容器内只读访问挂载目录。
只输出身份、路径和能力信息，不打印固定配置或凭据内容。版本变化后重新固定证据。

开发机 worktree 是待发布源码；`~/.akashic`、`~/.akashic-plugin` 和旧源码目录不是生产事实。
离线分析只复制已固定选择引用的制品，不通过扫描旧 cache 或本地目录推断支持范围。
安装、正式选择、live 激活和真实工作流验收分别报告；源码通过不能代替部署。

## 3. 历史快照：2026-09-02（不定义当前维护或安装范围）

- Core release：`9f30b079619523cafb2c49374260c9ea9ea9a180`
- Core image：`sha256:f1acf299d15ee2dcacbd025fe8074ad0574aac98aea834f8a9796a7483f56b17`
- 激活状态：`activation/active.json` 为 `active`，Core/Host Bridge active，Core/workload 容器与
  `akashic-release doctor` 均为 healthy
- manifest SHA-256：`7c9f8f274a0ea4b274d1a1227c6d978d53801259e8d7e37095f1ea0934d612bf`
- enabled manifest entries：33（17 builtin + 16 external）
- external artifact directories：40；逐个 static manifest + entrypoint 扫描后 non-V3 为 0
- 所有 24 个外部 plugin identity 的 stable 与 latest 在当时相同

### 3.1 当时启用的内置插件

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

### 3.2 当时启用的外置插件

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

### 3.3 当时已安装但禁用的外置身份

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
