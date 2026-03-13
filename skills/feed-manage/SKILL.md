---
name: feed-manage
description: 管理和查询信息来源订阅（RSS/动态）。当用户问"你有什么信息来源"、"你在监控什么"、"你订阅了什么"、"有哪些动态"、"信息来源是什么"时触发。
---

# Feed 管理与查询

## 触发条件（以下任何一种情况都必须使用本技能）

- 用户问"你有什么信息来源"、"你都在看什么"、"你订阅了什么"
- 用户问"有哪些动态"、"在监控什么"、"信息来源是什么"
- 用户要查最新内容、最近更新、最新动态
- 用户要添加/删除/管理订阅

## 工具用法

### 查看当前所有订阅

```
mcp_feed__feed_manage(action="list")
```

返回所有订阅的名称、URL、类型、备注。

### 查询最近内容

```
mcp_feed__feed_query(action="latest", limit=10)
```

可加 `source="名称"` 限定某个来源。

### 查询订阅概况

```
mcp_feed__feed_query(action="summary")
```

每个订阅的条目数、最近更新时间。

### 关键词搜索

```
mcp_feed__feed_query(action="search", keyword="关键词")
```

### 订阅列表（含 URL）

```
mcp_feed__feed_query(action="catalog")
```

### 添加订阅

```
mcp_feed__feed_manage(action="subscribe", name="名称", url="RSS地址")
```

### 取消订阅

```
mcp_feed__feed_manage(action="unsubscribe", name="名称")
```

## 注意

- 问"你有什么信息来源"时，先调 `mcp_feed__feed_manage(action=list)` 拿完整列表，再根据列表内容回答，**不得凭记忆或推测列举**
- 如果之前会话中调过 feed 工具，本轮仍需重新调用获取最新状态，不得用历史结果代答
