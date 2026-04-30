# 插件开发教程

## 目录

1. [概述](#概述)
2. [插件目录结构](#插件目录结构)
3. [第一个插件](#第一个插件)
4. [注册工具 (`@tool`)](#注册工具-tool)
5. [拦截工具调用 (`@on_tool_pre`)](#拦截工具调用-on_tool_pre)
6. [生命周期事件钩子 (`@on_*`)](#生命周期事件钩子-on_)
7. [Before-Turn 管道模块 (`PhaseModule`)](#before-turn-管道模块-phasemodule)
8. [插件配置 (`_conf_schema.json`)](#插件配置-_conf_schemajson)
9. [持久化存储 (`PluginKVStore`)](#持久化存储-pluginkvstore)
10. [元数据 (`manifest.yaml`)](#元数据-manifestyaml)
11. [初始化与清理](#初始化与清理)
12. [多插件协作](#多插件协作)
13. [完整示例：留言板插件](#完整示例留言板插件)
14. [参考](#参考)

---

## 概述

插件系统允许以独立目录形式扩展 agent 的行为，无需修改核心代码。

每个插件是一个包含 `plugin.py` 的目录，**放在 `plugins/` 下自动发现**。插件可以：

| 能力 | 使用方式 | 运行时机 |
|---|---|---|
| 注册新工具 | `@tool(name="...")` | LLM 调用工具时 |
| 拦截工具调用 | `@on_tool_pre(tool_name="...")` | 工具执行前 |
| 钩入生命周期事件 | `@on_before_turn` 等 | 对应生命周期阶段 |
| 插入 before-turn 管道模块 | `before_turn_modules_early()` / `late()` | 每轮 turn 开始 |
| 读取配置 | `self.context.config` | 任意时机 |
| 持久化键值存储 | `self.context.kv_store` | 任意时机 |
| 初始化 / 清理 | `initialize()` / `terminate()` | 加载 / 停机 |

---

## 插件目录结构

```
plugins/
└── my_plugin/                  # 插件目录名 = 插件名
    ├── plugin.py               # 必需：入口文件，定义 Plugin 子类
    ├── manifest.yaml           # 可选：覆盖元数据
    ├── _conf_schema.json       # 可选：声明默认配置项
    └── plugin_config.json      # 可选：用户级配置覆盖
```

`plugin.py` 是最小要求。管理器扫描 `plugins/` 下所有子目录，发现 `plugin.py` 即视为插件。

---

## 第一个插件

```python
# plugins/hello/plugin.py
from agent.plugins import Plugin

class Hello(Plugin):
    name = "hello"
    desc = "一个示例插件"

    async def initialize(self) -> None:
        print(f"[hello] 插件已加载，plugin_id={self.context.plugin_id}")

    async def terminate(self) -> None:
        print("[hello] 插件已卸载")
```

继承 `Plugin`，设置 `name`、`desc` 即可。`initialize()` 在所有注册完成后被调用，`terminate()` 在停机时调用。

`self.context` 在实例化后由管理器注入，类型为 `PluginContext`：

```python
@dataclass
class PluginContext:
    event_bus: Any              # EventBus 实例
    tool_registry: Any          # ToolRegistry 实例
    plugin_id: str              # 插件名
    plugin_dir: Path            # 插件目录
    kv_store: PluginKVStore     # 基于文件的键值存储
    config: PluginConfig | None # 解析后的配置
    observe_db_path: Path | None # observe.db 路径
```

---

## 注册工具 (`@tool`)

使用 `@tool` 装饰器声明工具。管理器自动创建 `Tool` 实例并注册到 `ToolRegistry`。

```python
# plugins/weather/plugin.py
from agent.plugins import Plugin, tool

class Weather(Plugin):
    name = "weather"

    @tool(
        name="get_weather",
        risk="read-only",
        search_hint="查询天气",
    )
    async def get_weather(
        self,
        event,                              # 第二个参数固定为 event（插件工具无实际值，传 None）
        city: str,                          # 参数类型决定 JSON Schema
        date: str = "today",
    ) -> str:
        """查询指定城市的天气信息。

        Args:
            city: 城市名称，如 "北京"
            date: 日期，默认 "today"
        """
        return f"{city} 在 {date} 的天气：晴，23°C"
```

### 装饰器参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `name` | `str` | 必需。工具名，LLM 看到的就是这个名字 |
| `risk` | `str` | 默认 `"read-write"`。标记风险等级 |
| `always_on` | `bool` | 默认 `False`。是否始终在 tools 列表中 |
| `search_hint` | `str` | 搜索提示，`/help` 等场景使用 |

### 参数 Schema 自动生成

`_derive_params_schema()` 通过函数签名 + docstring（Google/NumPy 风格）自动构建 JSON Schema：

- 前两个参数 `self` 和 `event` 被跳过
- Python 类型映射：`str → string`、`int → number`、`float → number`、`bool → boolean`、`dict → object`、`list → array`
- 无默认值的参数进入 `required` 列表
- docstring 中的 `Args:` 段被解析为参数描述

### 函数签名约束

**前两个参数必须是 `self` 和 `event`**，否则装饰器抛 `TypeError`。这是为了对齐事件处理规范 — 第二个参数 `event` 在插件工具中传 `None`。

---

## 拦截工具调用 (`@on_tool_pre`)

在工具真正执行前修改参数或拒绝调用。

```python
# plugins/shell_restore/plugin.py
from agent.plugins import Plugin, on_tool_pre
from agent.lifecycle.types import PreToolCtx

class ShellRestore(Plugin):
    name = "shell_restore"

    @on_tool_pre(tool_name="shell")
    async def rewrite_rm_to_mv(self, event: PreToolCtx) -> dict | None:
        """拦截 shell 工具的 rm 命令，改写为 mv（防止误删）。"""
        command = str(event.arguments.get("command", "")).strip()

        # 解析 rm 命令，提取目标文件
        if " rm " not in f" {command} ":
            return None  # 不是 rm 命令，不做处理

        targets = self._extract_targets(command)
        if not targets:
            return None

        # 改写为 mv
        mv_cmd = f"mv -- {' '.join(targets)} /home/user/restore/"
        return dict(event.arguments, command=mv_cmd)
```

### `PreToolCtx` 结构

```python
@dataclass
class PreToolCtx:
    session_key: str         # 会话标识
    channel: str             # 通道名（telegram / cli）
    chat_id: str             # 聊天 ID
    tool_name: str           # 被调用的工具名
    arguments: dict          # 原始参数字典
```

### 返回值

| 返回值 | 效果 |
|---|---|
| `None` | 不做修改，原参数继续执行 |
| `dict` | 用返回的字典**替换** arguments |

### 装饰器参数

| 参数 | 说明 |
|---|---|
| `tool_name` | 限定目标工具名。`None`（默认）匹配所有工具 |

---

## 生命周期事件钩子 (`@on_*`)

在 agent turn 生命周期的关键阶段插入 handler。handler 通过 EventBus 以 **顺序链** 方式运行。

### 可用事件

| 装饰器 | 事件阶段 | 模式 |
|---|---|---|
| `@on_before_turn` | Turn 开始 | GATE |
| `@on_before_reasoning` | 推理前 | GATE |
| `@on_before_step` | 每步推理前 | GATE |
| `@on_after_step` | 每步推理后 | TAP |
| `@on_after_reasoning` | 推理完成后 | GATE |
| `@on_after_turn` | Turn 结束后 | TAP |
| `@on_tool_call` | 工具调用前 | TAP |
| `@on_tool_result` | 工具返回后 | TAP |

### GATE vs TAP

- **GATE**：可以修改或**阻断**事件。返回修改后的事件（或 `None` 跳过）。运行于 `EventBus.emit()` 顺序链中。
- **TAP**：只能做观察/记录，不可修改事件。运行于 `EventBus.observe()` 或 `enqueue()` 中，失败不打断主流程。

### 上下文类型

每个事件对应一个上下文类型：

| 事件 | 上下文类型 | 关键字段 |
|---|---|---|
| `before_turn` | `BeforeTurnCtx` | `content`, `skill_names`, `retrieved_memory_block`, `history_messages`, `abort`, `abort_reply` |
| `before_reasoning` | `BeforeReasoningCtx` | `messages`, `system_prompt`, `tool_schemas` |
| `before_step` | `BeforeStepCtx` | `messages`, `step_index` |
| `after_step` | `AfterStepCtx` | `tool_calls`, `assistant_content` |
| `after_reasoning` | `AfterReasoningCtx` | `tool_calls`, `final_text` |
| `after_turn` | `AfterTurnCtx` | `outbound_content`, `session_key` |
| `before_tool_call` | `BeforeToolCallCtx` | `tool_name`, `tool_args`, `tool_call_id` |
| `after_tool_result` | `AfterToolResultCtx` | `tool_name`, `result_text`, `error` |

完整定义见 `agent/lifecycle/types.py`。

### 示例

```python
# plugins/audit/plugin.py
from agent.plugins import Plugin, on_before_turn, on_after_turn
from agent.lifecycle.types import BeforeTurnCtx, AfterTurnCtx

class Audit(Plugin):
    name = "audit"

    @on_before_turn()
    async def log_start(self, event: BeforeTurnCtx) -> BeforeTurnCtx:
        """每轮 turn 开始前记录。"""
        print(f"[audit] turn 开始: session={event.session_key}, content={event.content[:50]}")
        return event    # GATE handler 必须返回事件

    @on_after_turn()
    async def log_finish(self, event: AfterTurnCtx) -> None:
        """每轮 turn 结束后记录。"""
        print(f"[audit] turn 结束: outbound={event.outbound_content[:50]}")

    @on_before_turn(priority=100)   # 高优先级先执行
    async def content_filter(self, event: BeforeTurnCtx) -> BeforeTurnCtx | None:
        """屏蔽特定关键词。"""
        if "禁止词" in (event.content or ""):
            event.abort = True
            event.abort_reply = "此消息已被过滤器拦截。"
        return event
```

### GATE 阻断机制

在 `BeforeTurnCtx` 中设置 `abort=True` + `abort_reply` 可以阻断整个 turn：

```python
@on_before_turn()
async def gate(self, event: BeforeTurnCtx) -> BeforeTurnCtx:
    if self._should_block(event.content):
        event.abort = True
        event.abort_reply = "您的请求已被拦截。"
    return event
```

`BeforeTurnCtx.abort` 被设置后，`PassiveTurnPipeline.run()` 会直接返回 `abort_reply`，跳过推理和执行。

### priority 参数

所有 `@on_*` 装饰器接受 `priority: int` 参数（默认 0）。值越大越先执行：

```python
@on_before_turn(priority=100)  # 第一个执行
async def first(self, event): ...

@on_before_turn(priority=-10)  # 最后执行
async def last(self, event): ...
```

---

## Before-Turn 管道模块 (`PhaseModule`)

Before-turn 管道是 `default_before_turn_modules()` 返回的模块链，按顺序执行。

### 管道顺序

```
1. _AcquireSessionModule         ← 加载 session，产出 SESSION_SLOT
2. ★ plugin_modules_early        ← 你的早于检索的模块
3. _PrepareContextModule         ← 记忆检索，产出 CONTEXT_BUNDLE_SLOT
4. _BuildBeforeTurnCtxModule     ← 构建 BeforeTurnCtx，产出 CTX_SLOT
5. _EmitBeforeTurnCtxModule      ← EventBus 触发 on_before_turn GATE
6. ★ plugin_modules_late         ← 你的晚于 EventBus 的模块
7. _ReturnBeforeTurnCtxModule    ← 产出最终 output
```

### 早于 vs 晚于

- **`early`**：在记忆检索和 EventBus 之前运行。适合做**命令拦截**（如 `/memory_status`），因为可以 abort 并跳过后面的检索/推理。
- **`late`**：在 EventBus 之后运行。适合在 GATE handler 处理完后做**补充处理**。

### 编写 PhaseModule

一个 `PhaseModule` 必须符合协议：

```python
class MyModule:
    requires = ("session:session",)     # 依赖哪些 slot
    produces = ("session:ctx",)         # 产出哪些 slot

    async def run(self, frame: BeforeTurnFrame) -> BeforeTurnFrame:
        state = frame.input                    # TurnState (含 msg, session_key 等)
        session = frame.slots["session:session"]
        # ... 处理逻辑 ...
        if should_abort:
            frame.slots["session:ctx"] = BeforeTurnCtx(
                session_key=state.session_key,
                channel=state.msg.channel,
                chat_id=state.msg.chat_id,
                content=state.msg.content,
                timestamp=state.msg.timestamp,
                skill_names=[],
                retrieved_memory_block="",
                retrieval_trace_raw=None,
                history_messages=(),
                abort=True,
                abort_reply="回复文本",
            )
        return frame
```

### 关键 Slot 名

| Slot | 内容 | 由谁产出 |
|---|---|---|
| `session:session` | `Session` 对象 | `_AcquireSessionModule` |
| `session:context_bundle` | `ContextBundle`（记忆、skill 等） | `_PrepareContextModule` |
| `session:ctx` | `BeforeTurnCtx`（最终输出） | `_BuildBeforeTurnCtxModule` |

### 模块间通信：slot 依赖与跳过

如果前面的模块已经设置了 `session:ctx`（abort），后面的模块应该检查并跳过：

```python
class MyLateModule:
    requires = ("session:ctx",)
    produces = ("session:ctx",)

    async def run(self, frame):
        ctx = frame.slots.get("session:ctx")
        if ctx and ctx.abort:
            return frame  # 已 abort，跳过
        # 正常处理...
        return frame
```

### 从插件暴露 PhaseModule

在 Plugin 子类上定义 `before_turn_modules_early()` 和/或 `before_turn_modules_late()`：

```python
class StatusCommands(Plugin):
    name = "status_commands"

    def before_turn_modules_early(self) -> list[object]:
        return [
            MemoryStatusCommandModule(),
            KVCacheCommandModule(self.context.observe_db_path),
        ]
```

这些方法在 `PluginManager._load_one()` 中被调用，返回的模块列表被合并到管理器的 `_before_turn_modules_early` / `_late` 列表中。

---

## 插件配置 (`_conf_schema.json`)

在插件目录放置 `_conf_schema.json` 声明可配置项默认值：

```json
{
    "max_results": {
        "default": 5,
        "description": "最大返回结果数"
    },
    "api_endpoint": {
        "default": "https://api.example.com",
        "description": "API 端点地址"
    }
}
```

用户可以用 `plugin_config.json` 覆盖：

```json
{
    "max_results": 10
}
```

在代码中读取：

```python
max_results = self.context.config.get("max_results")  # 10
endpoint = self.context.config.api_endpoint            # "https://api.example.com"
```

`PluginConfig` 同时支持 `get(key)` 和 `config.key` 两种访问方式。

---

## 持久化存储 (`PluginKVStore`)

`self.context.kv_store` 是一个基于 JSON 文件的键值存储，文件位于 `{plugin_dir}/.kv.json`。

```python
# 读写
self.context.kv_store.set("last_run", str(datetime.now()))
last = self.context.kv_store.get("last_run", default="从未运行")

# 原子递增
count = self.context.kv_store.increment("request_count")  # +1 并返回新值
count = self.context.kv_store.increment("request_count", 10)  # +10
```

每次 `set()` / `increment()` 都会写回文件。适合存储少量元数据（计数器、状态标记等），不适合高频写入或大量数据。

---

## 元数据 (`manifest.yaml`)

`manifest.yaml` 覆盖 Plugin 基类的元数据字段：

```yaml
name: my_plugin
version: "1.0.0"
desc: 我的插件描述
author: your-name
```

这些字段也可以在类上直接设置：

```python
class MyPlugin(Plugin):
    name = "my_plugin"
    version = "1.0.0"
    desc = "我的插件描述"
    author = "your-name"
```

`manifest.yaml` 的优先级更高（会覆盖类属性）。管理器加载时调用 `_apply_manifest()` 读取并设置。

---

## 初始化与清理

```python
class MyPlugin(Plugin):
    async def initialize(self) -> None:
        """所有 handler、tool、hook 注册完成后调用。"""
        # 启动后台任务
        # 连接外部服务
        # 初始化状态

    async def terminate(self) -> None:
        """停机时调用。在 CoreRuntime.stop() 中触发。"""
        # 断开连接
        # 清理资源
        # 保存状态
```

### 重要：initialize 失败会回滚

如果 `initialize()` 抛出异常，管理器会**回滚**该插件的所有注册：

- 从 `PluginRegistry` 移除（handler、class、instance）
- 从 `ToolRegistry` **注销**已注册的工具
- 移除已添加的 tool hooks
- 移除已添加的 before-turn 模块

所以 `initialize()` 是验证外部依赖的好位置 — 如果所需服务不可用，抛异常即可安全回滚。

---

## 完整示例：留言板插件

一个带完整功能的示例插件：注册留言工具 + 拦截 shell 危险命令 + after-turn 日志。

```
plugins/message_board/
├── plugin.py
├── _conf_schema.json
└── manifest.yaml
```

### `_conf_schema.json`

```json
{
    "max_messages": {
        "default": 50,
        "description": "最多保留留言条数"
    }
}
```

### `manifest.yaml`

```yaml
name: message_board
version: "1.0.0"
desc: 留言板插件，支持发布留言和查看历史
author: plugin-author
```

### `plugin.py`

```python
"""plugin.py — 留言板插件"""
from datetime import datetime
from pathlib import Path

from agent.lifecycle.types import (
    AfterTurnCtx,
    PreToolCtx,
)
from agent.plugins import Plugin, tool, on_tool_pre, on_after_turn


class MessageBoard(Plugin):
    name = "message_board"
    desc = "留言板插件"

    # ── 配置 ─────────────────────────────────────────

    @property
    def _max_messages(self) -> int:
        return self.context.config.get("max_messages", 50)

    # ── 生命周期 ─────────────────────────────────────

    async def initialize(self) -> None:
        # 确保存储文件存在
        _ = self.context.kv_store.get("messages", default=[])

    # ── 工具注册 ─────────────────────────────────────

    @tool(
        name="leave_message",
        risk="read-write",
        search_hint="发布留言",
    )
    async def leave_message(
        self,
        event,
        content: str,
        author: str = "anonymous",
    ) -> str:
        """在留言板上发布一条留言。

        Args:
            content: 留言内容
            author: 留言者，默认 anonymous
        """
        messages = self.context.kv_store.get("messages", [])
        messages.append({
            "author": author,
            "content": content,
            "time": datetime.now().isoformat(),
        })
        # 裁剪到最大条数
        messages = messages[-self._max_messages:]
        self.context.kv_store.set("messages", messages)
        return f"留言已发布。当前共 {len(messages)} 条。"

    @tool(
        name="read_messages",
        risk="read-only",
        search_hint="查看留言板",
    )
    async def read_messages(
        self,
        event,
        limit: int = 10,
    ) -> str:
        """读取最近的留言。

        Args:
            limit: 返回条数，默认 10
        """
        messages = self.context.kv_store.get("messages", [])
        recent = messages[-min(limit, len(messages)):]
        if not recent:
            return "留言板暂无内容。"
        lines = []
        for m in recent:
            lines.append(f"[{m['time'][:19]}] {m['author']}: {m['content']}")
        return "\n".join(lines)

    # ── 工具拦截 ─────────────────────────────────────

    @on_tool_pre(tool_name="shell")
    async def block_dangerous_commands(
        self,
        event: PreToolCtx,
    ) -> dict | None:
        """拦截 shell 中的危险命令。"""
        command = str(event.arguments.get("command", ""))
        dangerous = ["rm -rf /", "mkfs.", "dd if=", ":(){ :|:& };:"]
        for pattern in dangerous:
            if pattern in command:
                # 返回新命令：替换为无害的 echo
                return dict(
                    event.arguments,
                    command=f"echo '[message_board] 危险命令已拦截: {pattern}'",
                )
        return None

    # ── 生命周期钩子 ─────────────────────────────────

    @on_after_turn()
    async def log_outbound(self, event: AfterTurnCtx) -> None:
        """记录每次发送的内容长度。"""
        content_len = len(event.outbound_content or "")
        self.context.kv_store.set("last_content_length", content_len)
        self.context.kv_store.increment("total_turns")
```

---

## 多插件协作

当多个插件同时加载时，执行顺序由以下机制控制：

### EventBus 装饰器：优先级控制

`@on_before_turn`、`@on_before_reasoning` 等所有 `@on_*` 装饰器接受 `priority` 参数（默认 0）。值越大执行越早：

```python
class PluginA(Plugin):
    @on_before_reasoning(priority=100)
    async def a_early(self, event: BeforeReasoningCtx) -> BeforeReasoningCtx:
        """先执行：高优先级修改 system prompt"""
        event.system_prompt += "\n附加规则 A"
        return event

class PluginB(Plugin):
    @on_before_reasoning(priority=-10)
    async def b_late(self, event: BeforeReasoningCtx) -> BeforeReasoningCtx:
        """后执行：看到的是 A 改过的 prompt"""
        print(f"当前 prompt: {event.system_prompt}")
        return event
```

同优先级时按插件加载顺序执行。`PluginHandlerRegistry.append()` 内部按 `-priority` 排序。

### PhaseModule：加载顺序即执行顺序

`before_turn_modules_early()` 和 `late()` 返回的模块按插件发现顺序合并到统一列表中。`discover()` 按目录名排序：

```
plugins/
├── 01_auth/        # 先被发现 → 模块先执行
├── 02_filter/      # 中间
└── 03_business/    # 最后
```

PhaseModule 本身没有 priority 机制。如果需要控制跨插件的 PhaseModule 顺序，在插件目录名上编码是最直接的做法。

### ToolHook 拦截链

`@on_tool_pre` 钩子按加载顺序逐个执行。每个钩子看到的是前一个钩子修改过的参数：

```python
# PluginA 先加载
@on_tool_pre(tool_name="shell")
async def hook_a(self, event) -> dict | None:
    args = dict(event.arguments)
    args["command"] = f"echo prefix; {args['command']}"
    return args

# PluginB 后加载：看到的是加了 prefix 的命令
@on_tool_pre(tool_name="shell")
async def hook_b(self, event) -> dict | None:
    print(f"即将执行: {event.arguments['command']}")
    return None  # 不改参，只观察
```

如果某个钩子返回 `decision="deny"`，后续钩子不再执行，调用被阻断。

### 工具注册：同名后覆盖前

多个插件可以注册同名工具，但后加载的会覆盖先加载的：

```python
# ToolRegistry.register() 做 self._tools[tool.name] = tool
# 后加载的插件同名工具会覆盖先加载的
```

如果插件之间需要共享或扩展工具，建议在 `@tool` 上使用**不同的工具名**，需要时在 `@on_tool_pre` 中编排调用链。

### 使用建议

| 场景 | 推荐做法 |
|---|---|
| 跨插件 EventBus handler 排序 | 显式设 `priority` |
| 跨插件 PhaseModule 排序 | 目录名前缀编码 |
| 工具互斥（同名冲突） | 避免同名，或用不同 namespace 前缀 |
| 工具编排（A 需要先处理再交给 B） | B 的 `@on_tool_pre` 在 A 之后加载即可 |

---

## 参考

### 核心类型

| 类型 | 位置 |
|---|---|
| `Plugin` | `agent/plugins/base.py` |
| `PluginContext` | `agent/plugins/context.py` |
| `PluginConfig` | `agent/plugins/config.py` |
| `PluginKVStore` | `agent/plugins/context.py` |
| `PreToolCtx` | `agent/lifecycle/types.py` |
| `BeforeTurnCtx` | `agent/lifecycle/types.py` |
| `BeforeTurnFrame` | `agent/lifecycle/phases/before_turn.py` |
| `BeforeTurnCtx`（含 abort/abort_reply） | `agent/lifecycle/types.py` |
| `ToolHook` | `agent/tool_hooks/base.py` |
| `HookContext` / `HookOutcome` | `agent/tool_hooks/types.py` |

### 所有生命周期上下文类型

见 `agent/lifecycle/types.py`：`BeforeTurnCtx`, `BeforeReasoningCtx`, `BeforeStepCtx`, `AfterStepCtx`, `AfterReasoningCtx`, `AfterTurnCtx`, `BeforeToolCallCtx`, `AfterToolResultCtx`, `PreToolCtx`。

### 已有插件

| 插件 | 目录 | 演示的能力 |
|---|---|---|
| shell_restore | `plugins/shell_restore/` | `@on_tool_pre` 拦截 shell 工具 |
| status_commands | `plugins/status_commands/` | `before_turn_modules_early()` + `PhaseModule` 命令拦截 |
