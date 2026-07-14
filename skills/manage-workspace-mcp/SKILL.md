---
name: manage-workspace-mcp
description: 安装、注册、更新、移除和诊断 Akashic 的独立 workspace MCP server。用户要求把 binary、CLI、脚本或本地项目做成常驻 MCP，询问 MCP 热重载、MCP 工具为何没出现，或需要查看非插件 MCP 状态时使用。插件自带 MCP 改用 plugin-system。
---

# 管理 Workspace MCP

使用声明式管理工具完成非插件 MCP 的全生命周期，不直接编辑运行时配置。

## 路由

- 独立 binary、CLI、脚本、本地项目：使用本 skill。
- 已安装插件自带的 MCP：加载 `plugin-system`，修改插件源码并重新安装。
- 不确定归属时，先用 `workspace_mcp_status` 查看；不要根据现有 MCP 工具名前缀猜注册方式。

## 工作流

1. 确认入口命令可以非交互启动，并通过 stdio 实现 MCP；不要把单独启动后等待 `initialize` 当成失败。
2. 用 `tool_search` 解锁 `workspace_mcp_apply` 和 `workspace_mcp_status`。
3. 调用 `workspace_mcp_apply`；优先使用稳定的绝对可执行路径。`cwd` 与 `watch_paths` 必须位于 workspace 的 `mcp/` 根内。
4. 检查返回的 generation、server 和工具列表。发布失败时保留真实错误，工具会恢复原声明。
5. 告知用户新 MCP 工具从下一轮开始可用；当前 turn 固定使用进入时的旧快照，不要在本轮反复搜索新工具。
6. 下一轮用 `workspace_mcp_status` 确认 active，再按返回的准确工具名执行真实调用。

删除时解锁并调用 `workspace_mcp_remove`。旧 generation 会等已有 turn 释放 lease 后排空，不要额外重启。

## 禁止旧路径

- 不要添加 `[mcp_servers]` 到 `config.toml`。
- 不要读写 `mcp_servers.json`。
- 不要搜索或调用 `mcp_add`、`mcp_list`、`mcp_remove`；这些旧工具不存在。
- 不要为了独立 MCP 创建 `plugin.json`、插件目录或修改插件缓存。
- MCP 和插件改动可热重载，不要调用 `agent_restart`。

## 参数原则

- `name` 使用稳定的小写名称，可含数字、`_`、`-`。
- `command` 是 argv 数组，不是 shell 字符串；不要嵌入 `&`、管道或重定向。
- Node wrapper 若依赖不稳定的 `PATH`，使用绝对 `node` 路径加脚本路径。
- `env` 只写 server 必需值；`workspace_mcp_status` 只展示环境变量键名。
- 代码内容变化需要触发重载时，把对应文件或目录放进 `watch_paths`。
