# Workspace MCP 声明

非插件 MCP 使用 `workspace/mcp/servers/*.toml` 声明；每个文件只声明一个 server，文件名必须与 `name` 一致。运行时按内容 revision 热重载，不再读取 `mcp_servers.json`，也不提供 `mcp_add`、`mcp_remove` 或 `mcp_list`。

例如真实布局：

```text
workspace/mcp/
├── servers/
│   └── fitbit.toml
└── fitbit-mcp/
    ├── run_mcp.py
    └── src/
```

`servers/fitbit.toml`：

```toml
schema_version = 1
name = "fitbit"
command = ["python", "run_mcp.py"]
cwd = "../fitbit-mcp"
watch_paths = ["../fitbit-mcp/run_mcp.py", "../fitbit-mcp/src"]

[env]
LOG_LEVEL = "INFO"
```

`cwd` 和 `watch_paths` 先相对声明文件所在目录解析，最终路径必须位于 `workspace/mcp/` 安全根内。因此 `../fitbit-mcp` 合法，越出 `mcp/` 或通过 symlink 跳出该目录会被拒绝。`watch_paths` 支持文件、目录或尚未创建的路径；新增、修改和删除都会改变 revision。任一声明无效或 server 连接失败时，整批候选被拒绝，旧 generation 继续服务；修复文件后 watcher 自动重试。删除全部 `.toml` 或整个 `servers` 目录会原子发布空 generation，并排空旧 MCP 进程。
