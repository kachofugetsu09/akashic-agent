# Proactive 主动推送指南

Proactive 的判断和投递由核心 runtime 负责；插件通过 MCP 提供已经刷新的数据。不再使用 `~/.akashic/workspace/proactive_sources.json`。

```text
┌─ 插件 plugin.py
│  ├─ 声明 MCP server
│  └─ 声明一个或多个 ProactiveSourceSpec
├─ 核心 runtime
│  ├─ 每个 tick 异步调用 fetch_tool
│  └─ 将同一份快照分发给各通道
├─ MCP 生命周期
│  └─ 自行维护外部数据与本地缓存的新鲜度
├─ proactive 判断
│  ├─ alert   ──> 紧急事件
│  ├─ content ──> 候选内容
│  └─ context ──> 背景状态
└─ 投递成功
   └─ 按原 source 与 event_id 调用 ack_tool
```
## 三种通道

| 通道 | 用途 | 是否触发推送 | ACK |
|---|---|---:|---:|
| `alert` | 健康告警、日程提醒、异常 | 是 | 通常需要 |
| `content` | RSS、新闻、社区内容 | 经过兴趣判断 | 通常需要 |
| `context` | 睡眠、在线状态、环境状态 | 否，只辅助判断 | 不需要 |

一个插件可以同时声明多个 source，也可以让一个 source 提供多个通道。

## 插件声明

```python
from agent.plugins import McpServerSpec, Plugin, ProactiveSourceSpec


class SourcePlugin(Plugin):
    name = "source"
    version = "1.0.0"

    @classmethod
    def mcp_servers(cls) -> list[McpServerSpec]:
        return [McpServerSpec(name="source", command=("python", "mcp/run.py"))]

    def proactive_sources(self) -> list[ProactiveSourceSpec]:
        if not self.context.config.proactive.enabled:
            return []
        return [
            ProactiveSourceSpec(
                id="updates",
                channels=("alert", "content"),
                server="source",
                fetch_tool="get_proactive_events",
                ack_tool="acknowledge_events",
            )
        ]
```

`fetch_tool` 不负责驱动 agent。需要缓存的 MCP 应在自己的 lifespan、后台服务或按需读取路径中维护新鲜度，避免 proactive 插件切换后刷新任务消失。

## 插件内关闭主动链路

```toml
# ~/.akashic-plugin/data/source-github/config.local.toml
[proactive]
enabled = false
```

关闭主动信息源不等于关闭整个插件。插件的普通 MCP 工具、skills 与其他生命周期能力仍可继续工作。若要关闭整个插件，修改全局 `manifest.toml`。

## MCP 返回协议

alert/content 的 `fetch_tool` 返回 JSON 数组：

```json
[
  {
    "event_id": "event-1",
    "kind": "alert",
    "source_type": "calendar",
    "source_name": "work",
    "title": "会议即将开始",
    "content": "项目周会将在 10 分钟后开始",
    "severity": "medium",
    "published_at": "2026-07-10T17:00:00+08:00"
  }
]
```

context 返回一个 JSON 对象或对象数组：

```json
{
  "available": true,
  "topic": "用户睡眠状态",
  "summary": "用户当前更可能醒着",
  "hint": "这是概率判断，不能当作事实确认"
}
```

ACK 工具接收 `event_ids: list[str]`。只确认已经成功投递的原始事件，不能用标题或 URL 代替 ID。

## 缓存刷新语义

```text
┌─ MCP 启动
│  └─ 启动自己的缓存刷新生命周期
├─ proactive tick
│  └─ 并发调用各 source 的 fetch_tool 读取快照
├─ 通道筛选与决策
└─ 投递成功后精确 ACK
```

source 的稳定身份是 `<plugin_id>:<source_id>`。公共主动链只理解 source、channel 和工具名，不硬编码插件名，也不拥有外部数据刷新周期。

## 验证清单

```text
┌─ plugin.py 能被导入
├─ MCP server 能建立 stdio 连接
├─ fetch_tool 返回合法 JSON
├─ kind 与声明通道一致
├─ context 不会单独触发推送
├─ 重复 tick 不会重复投递已 ACK 事件
└─ proactive.enabled = false 时 source 列表为空
```
