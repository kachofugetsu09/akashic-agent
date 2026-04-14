---
name: create-proactive-source
description: 创建或更新主动行为信息源 MCP server，注册到 proactive_sources.json。当用户要新增主动循环中的主动推送的数据来源时使用。
---

# 创建 Proactive 信息源

## 目标

把一个新的数据来源接入 proactive 主动推送系统。产出是一个 MCP server + 对应的配置注册。

## 何时使用

- 用户想订阅一个新的信息源（API、网站、传感器、服务等）
- 用户想给现有 MCP server 增加一条新的 proactive 通道
- 现有信息源协议不符合标准，需要适配

## 三条通道选型

根据数据性质选择正确的通道：

| 通道 | 适用场景 | 特征 |
|------|----------|------|
| **alert** | 需要立即通知用户的紧急事件（健康告警、日程提醒、传感器异常） | bypass 评分，直接发送；有 severity 等级；需要 ACK |
| **content** | 内容流（RSS、社交媒体、新闻聚合） | 参与 HyDE 兴趣评分 → mark_interesting/not_interesting；有去重和 TTL ACK；可选 poll_tool 定时拉取 |
| **context** | 背景状态信息（睡眠状态、在线状态、天气、设备状态） | 不主动触发推送；作为 fallback 或辅助决策信号；无 ACK |

一个 MCP server 可以同时提供多条通道（如 fitbit 同时提供 alert + context）。

## MCP Server 协议规范

### alert / content 通道

必须实现 `get_proactive_events` 工具（或在配置中用 `get_tool` 指定别名），返回格式：

```json
[
  {
    "kind": "alert" | "content",
    "event_id": "唯一标识，用于去重和 ACK",
    "source_type": "来源类型标识（如 rss / sensor / calendar）",
    "source_name": "人类可读来源名",
    "title": "标题",
    "content": "正文摘要或告警消息",
    "url": "可选，原文链接",
    "published_at": "可选，ISO 8601 时间戳",
    "severity": "仅 alert：high / normal / low",
    "suggested_tone": "可选，建议语气",
    "metrics": "可选，dict，附加指标"
  }
]
```

关键规则：
- `event_id` 必须稳定且唯一，engine 用它做去重和 ACK
- `kind` 字段决定事件归入哪条通道，engine 按 kind 过滤
- 无事件时返回空列表 `[]`
- alert 的 `severity: "high"` 会触发 urgent fast-path

必须实现 `acknowledge_events` 工具（或在配置中用 `ack_tool` 指定别名）：

```
输入: {"event_ids": ["id1", "id2"], "ttl_hours": 168}
```

- 收到 ACK 后，对应 event_id 在 ttl_hours 内不再被 get_proactive_events 返回
- ttl_hours=0 或缺省时视为永久 ACK

### content 通道可选 poll_tool

如果内容源需要定时刷新（如 RSS），可实现 `poll_feeds` 工具：
- 无参数调用
- 返回 "ok" 或 "error: ..." 字符串
- engine 按 `feed_poller_interval_seconds`（默认 150s）定时调用

### context 通道

必须实现 `get_context`（或在配置中用 `get_tool` 指定别名），返回格式：

```json
{
  "available": true,
  "任意字段": "值",
  "_source": "自动由 engine 填充，无需返回"
}
```

或返回 `list[dict]` 多条上下文。context 无需 ACK。

## 配置注册

### 1. MCP Server 注册（~/.akashic/workspace/mcp_servers.json）

```json
{
  "servers": {
    "新server名": {
      "command": ["python", "/path/to/run_mcp.py"],
      "env": {"API_KEY": "..."}
    }
  }
}
```

### 2. Proactive Source 注册（~/.akashic/workspace/proactive_sources.json）

每条通道一个条目：

```json
{
  "server": "server名（对应 mcp_servers.json 的 key）",
  "channel": "alert | content | context",
  "get_tool": "get_proactive_events（默认，可省略）",
  "ack_tool": "acknowledge_events（alert/content 必填）",
  "poll_tool": "poll_feeds（content 可选）",
  "enabled": true
}
```

## 创建流程

1. **确认数据源和通道类型**：明确数据来源是什么、应该走哪条通道
2. **创建 MCP server**：
   - 在 `~/.akasic/workspace/mcp/` 下创建目录
   - 用 FastMCP 实现标准协议工具
   - 创建 `run_mcp.py` 入口和虚拟环境
3. **注册配置**：
   - 在 `mcp_servers.json` 添加 server 启动命令
   - 在 `proactive_sources.json` 添加 source 条目
4. **验证**：重启 agent 后确认 `[mcp_pool] connected: xxx` 日志

## MCP Server 模板

最小 content 源示例：

```python
import json
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-source")

# 内部维护已 ACK 的 event_id 集合
_acked: dict[str, float] = {}  # event_id -> ack_until_timestamp

@mcp.tool()
def get_proactive_events() -> str:
    events = _fetch_from_upstream()  # 你的数据拉取逻辑
    result = []
    for item in events:
        if item["id"] in _acked and time.time() < _acked[item["id"]]:
            continue
        result.append({
            "kind": "content",
            "event_id": item["id"],
            "source_type": "my_source",
            "source_name": "My Source",
            "title": item["title"],
            "content": item["summary"],
            "url": item.get("url", ""),
            "published_at": item.get("date", ""),
        })
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def acknowledge_events(event_ids: list[str], ttl_hours: int = 0) -> str:
    import time
    until = time.time() + ttl_hours * 3600 if ttl_hours > 0 else float("inf")
    for eid in event_ids:
        _acked[eid] = until
    return json.dumps({"ok": True, "acked": len(event_ids)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 约束

- MCP server 文件放在 `~/.akasic/workspace/mcp/<server-name>/`，不要放到仓库内
- 一个 server 可以服务多条通道，每条通道在 `proactive_sources.json` 中独立注册
- `event_id` 必须在同一 server 内全局唯一且稳定（相同事件每次返回相同 id）
- ACK 状态建议持久化到文件，避免 server 重启后丢失
- 不要在 MCP server 内做兴趣评分或决策——这是 engine (agent_tick) 的职责
