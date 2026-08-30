# 插件系统

Akashic 插件采用“全局只管启停，插件自己声明能力”的模型。插件仓库必须提供根目录 `plugin.py`，不再读取 `.aka-plugin/plugin.json`、`manifest.yaml`、`mcp/servers.json` 或 `registry.json`。

```text
┌─ ~/.akashic-plugin/manifest.toml
│  └─ 只记录 plugin_id 与 enabled
├─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/<version>/
│  └─ 从 Git 仓库安装的只读代码与 MCP 虚拟环境
└─ <workspace>/plugin-data/<plugin>-<marketplace>/
   ├─ config.local.toml
   └─ 数据库、Token、模型与日志等持久状态
```

## 最小插件

```python
from agent.plugins import Plugin


class DemoPlugin(Plugin):
    name = "demo"
    version = "1.0.0"
    desc = "最小插件"
```

## Web UI 插件

插件通过 `web_module` 发布自包含的 JavaScript 和同名 CSS，并在 `activate(ctx)` 中用 `ctx.ui.inject(...)` 接入另一个插件提供的 mount；完整合同、示例、样式边界与验收见 [`docs/design/web-ui-plugin-composition.md`](../docs/design/web-ui-plugin-composition.md)。

目录名、`name` 与安装后的插件身份必须一致。安装到 `github` 市场后，插件 ID 是 `demo@github`。

## 全局启停清单

`~/.akashic-plugin/manifest.toml` 只回答“插件是否启用”：

```toml
[plugins."demo@github"]
enabled = true
```

能力、命令、路径和配置 schema 都不写入全局清单。

## 插件配置

插件通过 Pydantic 模型声明配置，用户值放在插件数据目录。

```python
from pydantic import BaseModel, Field
from agent.plugins import Plugin


class ProactiveConfig(BaseModel):
    enabled: bool = True


class DemoConfig(BaseModel):
    proactive: ProactiveConfig = Field(default_factory=ProactiveConfig)


class DemoPlugin(Plugin):
    name = "demo"
    version = "1.0.0"
    ConfigModel = DemoConfig
```

对应配置：

```toml
# <workspace>/plugin-data/demo-github/config.local.toml
[proactive]
enabled = false
```

实例方法通过 `self.context.config` 读取验证后的模型，通过 `self.context.data_dir` 读写持久数据。不要向插件仓库或 cache 写运行状态。

## 声明 skills

```python
class DemoPlugin(Plugin):
    name = "demo"
    version = "1.0.0"

    @classmethod
    def skill_roots(cls) -> tuple[str, ...]:
        return ("skills",)

    @classmethod
    def drift_skill_roots(cls) -> tuple[str, ...]:
        return ("drift/skills",)
```

路径相对插件根目录解析，声明的目录必须存在。

## 声明 MCP server

```python
from agent.plugins import McpServerSpec, Plugin


class DemoPlugin(Plugin):
    name = "demo"
    version = "1.0.0"

    @classmethod
    def mcp_servers(cls) -> list[McpServerSpec]:
        return [
            McpServerSpec(
                name="demo",
                command=("python", "mcp/run_mcp.py"),
            )
        ]
```

安装器会根据 MCP 入口附近的 `requirements.txt` 创建 `.venv`。运行时自动注入 `AKA_PLUGIN_DATA_DIR`，并把 Python 命令解析到插件自己的虚拟环境。

## 声明主动信息源

同一个插件可以同时提供 alert、content 和 context：

```python
from agent.plugins import ProactiveSourceSpec


def proactive_sources(self) -> list[ProactiveSourceSpec]:
    if not self.context.config.proactive.enabled:
        return []
    return [
        ProactiveSourceSpec(
            id="alerts",
            channels=("alert",),
            server="demo",
            fetch_tool="get_proactive_events",
            ack_tool="acknowledge_events",
        ),
        ProactiveSourceSpec(
            id="state",
            channels=("context",),
            server="demo",
            fetch_tool="get_context",
        ),
    ]
```

```text
┌─ PluginManager 加载 plugin.py
│  ├─ 读取 manifest.toml 决定是否加载
│  ├─ 验证 config.local.toml
│  └─ 收集 MCP、skills、生命周期与主动信息源
├─ MCP 生命周期维护外部数据和缓存
├─ 核心 runtime 每个 tick 异步调用 fetch_tool 获取快照
└─ proactive 引擎按通道消费
   ├─ alert   ──> 快速告警与精确 ACK
   ├─ content ──> 兴趣判断、投递与 ACK
   └─ context ──> 只注入决策上下文
```

缓存刷新由 MCP 自己负责；插件只声明读取能力，不自行驱动 agent。这样切换 proactive 流程不会中断数据刷新。

## 安装与检查

```bash
python main.py plugin-install \
  --source https://github.com/example/demo.git \
  --marketplace github

python main.py plugin-doctor --plugin demo@github
```

运行中的 Runtime 会自动应用启停和卸载变化：

```bash
python main.py plugin-disable demo@github
python main.py plugin-enable demo@github
python main.py plugin-uninstall demo@github
```

`plugin-disable` 与 `plugin-enable` 只修改全局 `manifest.toml`。`plugin-uninstall` 删除插件的 manifest 条目和全部 cache 版本，但始终保留 `data/demo-github/`，重新安装后会继续复用原配置与持久数据。

安装后检查：

```text
┌─ manifest.toml 中 enabled = true
├─ cache/github/demo/<version>/plugin.py 存在
├─ data/demo-github/config.local.toml 可通过 schema 验证
├─ 声明的 skills 与 MCP 入口存在
└─ watcher 发布 committed snapshot，运行日志无加载错误
```

## 热重载

Runtime 每秒检查插件清单、源码和本地配置的元数据。发现变化后先构建完整候选代际，通过声明、资源 readiness 与插件语义检查后再一次性发布。

```text
┌─ 变化入口
│  ├─ 安装或删除插件目录
│  ├─ manifest.toml enabled 改变
│  ├─ plugin.py 或插件资源改变
│  └─ config.local.toml 改变
├─ Candidate Gate
│  ├─ 编译 lifecycle、tool、skill、MCP、job、proactive
│  ├─ 预热 Dashboard、Channel 与 managed service
│  └─ 失败时丢弃候选，旧代继续服务
├─ RuntimeSnapshot 原子发布
│  ├─ 已开始的执行继续持有旧快照
│  └─ 新执行只租用新快照
└─ Drain
   └─ 最后一个旧 lease 释放后关闭旧资源
```

## 升级边界

```text
┌─ cache
│  └─ 可替换：源码、静态资源、MCP 虚拟环境
└─ data
   └─ 必须保留：配置、数据库、Token、模型、日志
```

插件需要独占后台服务时，通过 `managed_services()` 声明；短期异步任务使用 `self.context.create_task()`。MCP bridge 只连接服务，不应再维护第二套进程所有权。

## 仍需重启的边界

热重载只替换插件代际，不替换宿主进程本身。

```text
┌─ 可热重载
│  ├─ Python 插件源码与资源
│  ├─ config.local.toml
│  └─ MCP、Skill、Job、Channel、Dashboard 与 managed service 声明
└─ 必须重启 Runtime
   ├─ CPython 或核心 Runtime ABI 变化
   ├─ 已载入的原生动态库升级
   └─ Runtime 自身依赖与启动参数变化
```
