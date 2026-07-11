---
name: create-proactive-source
description: 创建或更新由插件程序化声明的主动信息源 MCP server，并完成本地验证。
---

# 创建 Proactive 信息源

产出是插件中的 MCP server、`ProactiveSourceSpec`、插件本地配置与测试。不创建 `proactive_sources.json`。

## 通道选择

| 通道 | 用途 | ACK |
|---|---|---:|
| `alert` | 紧急告警、提醒、异常 | 需要 |
| `content` | RSS、新闻、候选内容 | 需要 |
| `context` | 睡眠、在线状态等决策背景 | 不需要 |

一个插件或 source 可以同时提供多种通道。

## 声明方式

```python
from agent.plugins import McpServerSpec, Plugin, ProactiveSourceSpec


class DemoPlugin(Plugin):
    name = "demo"
    version = "1.0.0"

    @classmethod
    def mcp_servers(cls) -> list[McpServerSpec]:
        return [McpServerSpec(name="demo", command=("python", "mcp/run.py"))]

    def proactive_sources(self) -> list[ProactiveSourceSpec]:
        if not self.context.config.proactive.enabled:
            return []
        return [
            ProactiveSourceSpec(
                id="events",
                channels=("alert", "content"),
                server="demo",
                fetch_tool="get_proactive_events",
                ack_tool="acknowledge_events",
                poll_tool="poll_events",
                poll_interval_seconds=300,
            )
        ]
```

## MCP 契约

alert/content 的 fetch 工具返回 JSON 数组，每项至少包含 `event_id`、`kind`、`source_type`、`source_name`、`title`、`content`。context 返回 JSON 对象或对象数组。

ACK 工具接收 `event_ids: list[str]`，并只确认成功投递的原始 ID。`poll_tool` 只刷新上游数据，不调用 agent，也不自行决定推送。

## 配置

插件用 `ConfigModel` 声明：

```toml
[proactive]
enabled = true
poll_interval_seconds = 300
```

文件位于 `~/.akashic-plugin/data/<plugin>-<marketplace>/config.local.toml`。关闭 proactive 只移除主动 source，不关闭插件其他能力。

## 验证

```text
┌─ 导入 plugin.py
├─ 校验配置模型
├─ 启动 MCP 并调用 fetch_tool
├─ 校验三种通道返回值
├─ 校验 ACK 精确匹配 event_id
├─ 校验 poll_tool 由 runtime 周期调用
└─ 校验 enabled = false 时 proactive_sources() 为空
```
