---
name: create-proactive-source
description: 创建或更新主动行为信息源 MCP server，注册到 proactive_sources.json。当用户要新增主动循环中的主动推送的数据来源时使用。
---

# 创建 Proactive 信息源

## 目标

把一个新的数据来源接入 proactive 主动推送系统。产出是一个 MCP server + 对应的配置注册，**并通过本地验证**。

## 何时使用

- 用户想订阅一个新的信息源（API、网站、传感器、服务等）
- 用户想给现有 MCP server 增加一条新的 proactive 通道
- 现有信息源协议不符合标准，需要适配

## 三条通道选型

| 通道 | 适用场景 | 特征 |
|------|----------|------|
| **alert** | 需要立即通知用户的紧急事件（健康告警、日程提醒、传感器异常） | bypass 评分，直接发送；有 severity 等级；需要 ACK |
| **content** | 内容流（RSS、社交媒体、新闻聚合） | 参与 HyDE 兴趣评分 → mark_interesting/not_interesting；有去重和 TTL ACK；可选 poll_tool 定时拉取 |
| **context** | 背景状态信息（睡眠状态、在线状态、天气、设备状态） | 不主动触发推送；作为 fallback 或辅助决策信号；无 ACK |

一个 MCP server 可以同时提供多条通道（如 fitbit 同时提供 alert + context）。

## MCP Server 协议规范

### alert / content 通道

必须实现 `get_proactive_events` 工具（或在配置中用 `get_tool` 指定别名）。

**返回值必须是 JSON 序列化的 list[dict]**，engine 的处理链路：
1. `pool.call()` 把 MCP 工具返回的字符串 JSON 反序列化
2. `_extract_proactive_events(data, server, kind)` 检查 `isinstance(data, list)`，逐条检查 `event.get("kind") == channel`
3. 通过的事件进入 `normalize_alert()` 或 `normalize_content()` 构建 Contract

**因此返回值如果不是 list、或 kind 字段不匹配，事件会被静默丢弃。**

每条事件的必需字段：

```json
{
  "kind": "alert",
  "event_id": "humidity_low_2026-04-15",
  "source_type": "sensor",
  "source_name": "Zigbee 温湿度",
  "title": "室内湿度过低",
  "content": "当前湿度 28%，低于阈值 30%"
}
```

alert 额外字段（影响 engine 行为）：
- `severity`: `"high"` → engine 的 `is_urgent()` 返回 True，触发 fast-path bypass 评分
- `suggested_tone`: 建议语气，传入 LLM prompt
- `metrics`: dict，附加指标，会序列化注入 prompt

content 额外字段：
- `url`: 原文链接，gateway 会预取正文；engine 用 URL 做 delivery 去重
- `published_at`: ISO 8601 时间戳
- `source_name`: 会显示在 content 列表的 source 列

**关键规则：**
- `event_id` 必须稳定且唯一——相同事件每次返回相同 id，engine 用它做去重和 ACK
- `kind` 必须是 `"alert"` 或 `"content"`，不是 `"type"`，不是大写
- 无事件时**必须返回空列表 `[]`**（不是空字符串 `""`、不是 `null`、不是 `{}`）
- MCP server 应该**持续返回未被 ACK 的事件**，直到 engine 调用 `acknowledge_events`；不要在 server 侧做"只报一次"的防抖——那是 ACK 的职责

必须实现 `acknowledge_events` 工具（或在配置中用 `ack_tool` 指定）：

```
输入: {"event_ids": ["id1", "id2"], "ttl_hours": 168}
```

- 收到 ACK 后，对应 event_id 在 ttl_hours 内不再被 `get_proactive_events` 返回
- `ttl_hours=0` 或缺省时视为永久 ACK
- ACK 状态**必须持久化到文件**，MCP server 是 stdio 模式按需启动，内存状态会丢失

### content 通道可选 poll_tool

如果内容源需要定时刷新（如 RSS），可实现 `poll_feeds` 工具：
- 无参数调用，返回 `"ok"` 或 `"error: ..."` 字符串
- engine 按 `feed_poller_interval_seconds`（默认 150s）定时调用

### context 通道

实现 `get_context`（或在配置中用 `get_tool` 指定），返回 `dict` 或 `list[dict]`：

```json
{"available": true, "humidity_rh": 37.5, "temperature_c": 22.1}
```

engine 的 `_extract_context_items` 兼容单 dict 和 list[dict]，会自动补 `_source` 字段。
context 无需 ACK，无需 `event_id`/`kind`。

### 常见错误

| 错误 | 后果 | 正确做法 |
|------|------|----------|
| 返回 `{"type": "alert", "message": "..."}` | `_extract_proactive_events` 检查 `isinstance(data, list)` 失败，事件被静默丢弃 | 返回 `[{"kind": "alert", ...}]` |
| 无事件时返回空字符串 `""` | `pool.call()` 不会 JSON 解析（不以 `[{` 开头），`_extract_proactive_events` 收到字符串，`isinstance(data, list)` 为 False | 返回 `json.dumps([])` |
| 缺少 `ack_tool` 配置 | engine 无法 ACK，已发送的 alert 下次 tick 重复推送 | 在 `proactive_sources.json` 中配置 `ack_tool`，在 server 中实现 `acknowledge_events` |
| server 侧"只报一次"防抖 | 如果 engine 那个 tick 被 pre-gate 拦了，事件永远丢失 | 让 server 持续返回事件，由 engine ACK 来消除 |
| ACK 状态只存内存 | MCP server 是 stdio 按需启停，重启后 ACK 丢失，已发送事件重复推 | ACK 状态持久化到 JSON 文件 |

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
  "get_tool": "get_proactive_events（默认值，可省略）",
  "ack_tool": "acknowledge_events（alert/content 必填）",
  "poll_tool": "poll_feeds（content 可选）",
  "enabled": true
}
```

## 创建流程

### 0. 盘点现有 server

- 读取 `~/.akashic/workspace/mcp_servers.json`，列出所有已注册的 server 名称
- 读取 `~/.akashic/workspace/proactive_sources.json`，列出已接入的 source 条目
- 浏览 `~/.akashic/workspace/mcp/` 目录，了解每个 server 的用途
- 判断目标数据源能否复用或扩展现有 server（如给已有 server 加一条新通道），而不是从零新建

### 1. 确认数据源和通道类型

明确数据来源是什么、应该走哪条通道。

### 2. 创建或扩展 MCP server

- 如果复用现有 server：直接在其代码中添加新工具，在 `proactive_sources.json` 中增加新通道条目
- 如果新建 server：在 `~/.akashic/workspace/mcp/` 下创建目录，用 FastMCP 实现标准协议工具，创建 `run_mcp.py` 入口和虚拟环境

### 3. 注册配置

- 新 server 需在 `mcp_servers.json` 添加启动命令
- 在 `proactive_sources.json` 添加 source 条目

### 4. 本地验证（必须）

实现完成后，**必须执行以下验证步骤**，确认返回值能通过 engine 的提取和归一化链路。
直接在 MCP server 目录下用 python 调用 backend 函数，不需要启动完整 agent。

#### alert 通道验证

```python
import json, sys
sys.path.insert(0, "/mnt/data/coding/akasic-agent")
from src.backend import 你的函数名  # 按实际路径调整

# ① 模拟有事件时的返回
raw = 你的函数名()  # 先制造触发条件再调用
data = json.loads(raw)

# 检查点 1: 必须是 list
assert isinstance(data, list), f"FAIL: 返回不是 list，是 {type(data).__name__}"

# 检查点 2: 模拟 engine 提取
from proactive_v2.mcp_sources import _extract_proactive_events
events = _extract_proactive_events(data, server="你的server名", kind="alert")
assert len(events) > 0, "FAIL: _extract_proactive_events 过滤后为空（检查 kind 字段）"

# 检查点 3: 模拟 engine 归一化
from proactive_v2.contracts import normalize_alert
for e in events:
    contract = normalize_alert(e)
    assert contract.item_id != "?:?", f"FAIL: event_id 缺失 → item_id={contract.item_id}"
    assert contract.content, f"FAIL: content 为空"
    print(contract.to_prompt_line(1))

# ② 模拟无事件时的返回
raw_empty = 你的函数名()  # 在正常条件下调用
data_empty = json.loads(raw_empty)
assert isinstance(data_empty, list), f"FAIL: 无事件时返回不是 list，是 {type(data_empty).__name__}"
assert len(data_empty) == 0, f"FAIL: 无事件时应返回空列表，实际有 {len(data_empty)} 条"

# ③ ACK 往返验证
from src.backend import acknowledge_events  # 按实际路径调整
ack_result = json.loads(acknowledge_events([events[0]["event_id"]], ttl_hours=24))
assert "ok" in ack_result or "error" not in ack_result, f"FAIL: ACK 返回异常 {ack_result}"
# ACK 后再拉一次，被 ACK 的事件不应出现
raw_after = 你的函数名()
data_after = json.loads(raw_after)
acked_ids = {events[0]["event_id"]}
remaining = [e for e in data_after if e.get("event_id") in acked_ids]
assert len(remaining) == 0, f"FAIL: ACK 后事件仍被返回 {remaining}"

print("ALL CHECKS PASSED")
```

#### content 通道验证

```python
import json, sys
sys.path.insert(0, "/mnt/data/coding/akasic-agent")

raw = 你的get函数()
data = json.loads(raw)

assert isinstance(data, list), f"FAIL: 返回不是 list"

from proactive_v2.mcp_sources import _extract_proactive_events
events = _extract_proactive_events(data, server="你的server名", kind="content")
assert len(events) > 0, "FAIL: 提取后为空（检查 kind='content'）"

from proactive_v2.contracts import normalize_content
for e in events:
    # content 走 gateway 预取，gateway 用 content_meta 格式
    meta = {
        "id": f"{e.get('ack_server', '你的server名')}:{e['event_id']}",
        "title": e.get("title", ""),
        "source": e.get("source_name", ""),
        "url": e.get("url", ""),
    }
    contract = normalize_content(meta)
    assert contract.item_id, f"FAIL: item_id 为空"
    assert contract.title, f"FAIL: title 为空"
    print(contract.to_prompt_line(1, has_content=bool(e.get("url"))))

# ACK 往返（同 alert）
ack_result = json.loads(你的ack函数([events[0]["event_id"]], ttl_hours=168))
raw_after = 你的get函数()
data_after = json.loads(raw_after)
remaining = [e for e in data_after if e.get("event_id") == events[0]["event_id"]]
assert len(remaining) == 0, f"FAIL: ACK 后事件仍被返回"

print("ALL CHECKS PASSED")
```

#### context 通道验证

```python
import json, sys
sys.path.insert(0, "/mnt/data/coding/akasic-agent")

raw = 你的context函数()
data = json.loads(raw)

from proactive_v2.mcp_sources import _extract_context_items
items = _extract_context_items(data, server="你的server名")
assert len(items) > 0, f"FAIL: _extract_context_items 返回空（检查返回的是 dict 或 list[dict]）"

from proactive_v2.contracts import normalize_context
for item in items:
    contract = normalize_context(item)
    prompt = contract.to_prompt_item()
    print(json.dumps(prompt, ensure_ascii=False, indent=2))
    # context 没有强制字段，但至少应该有业务数据
    biz_keys = [k for k in prompt if k not in ("_source", "available")]
    assert len(biz_keys) > 0, "FAIL: context 除 _source/available 外没有业务字段"

print("ALL CHECKS PASSED")
```

### 5. 清理验证状态

验证会产生 ACK 状态文件记录，验证通过后清理：
- 删除或重置 ACK 持久化文件中的测试数据
- 如果有状态文件（如 alert 状态），重置为初始状态

## MCP Server 模板

最小 alert 源示例（含 ACK 持久化）：

```python
import json, time
from pathlib import Path
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("my-alert-source")
ACK_FILE = Path(__file__).parent / "ack_state.json"

def _load_acks() -> dict[str, float]:
    try:
        return json.loads(ACK_FILE.read_text()) if ACK_FILE.exists() else {}
    except Exception:
        return {}

def _save_acks(acks: dict[str, float]) -> None:
    ACK_FILE.write_text(json.dumps(acks))

@mcp.tool()
def get_proactive_events() -> str:
    acks = _load_acks()
    now = time.time()
    # — 你的数据拉取逻辑 —
    raw_events = _check_something()  # 返回 list[dict]
    result = []
    for item in raw_events:
        eid = item["event_id"]
        if eid in acks and now < acks[eid]:
            continue  # 已 ACK 且未过期
        result.append({
            "kind": "alert",
            "event_id": eid,
            "source_type": "sensor",
            "source_name": "My Sensor",
            "title": item["title"],
            "content": item["message"],
            "severity": item.get("severity", "normal"),
        })
    return json.dumps(result, ensure_ascii=False)

@mcp.tool()
def acknowledge_events(event_ids: list[str], ttl_hours: int = 0) -> str:
    acks = _load_acks()
    until = time.time() + ttl_hours * 3600 if ttl_hours > 0 else float("inf")
    for eid in event_ids:
        acks[eid] = until
    _save_acks(acks)
    return json.dumps({"ok": True, "acked": len(event_ids)})

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

## 约束

- MCP server 文件放在 `~/.akashic/workspace/mcp/<server-name>/`，不要放到仓库内
- 一个 server 可以服务多条通道，每条通道在 `proactive_sources.json` 中独立注册
- `event_id` 必须在同一 server 内全局唯一且稳定（相同事件每次返回相同 id）
- ACK 状态必须持久化到文件（MCP server 是 stdio 模式，按需启停）
- 不要在 MCP server 内做兴趣评分或"只报一次"防抖——持续返回未 ACK 事件，由 engine 负责 ACK 消除
- **步骤 4 的本地验证必须通过后才能完成**
