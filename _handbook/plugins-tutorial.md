# 插件开发教程

## 目录

1. [概述](#概述)
2. [插件目录结构](#插件目录结构)
3. [第一个插件](#第一个插件)
4. [注册工具 (`@tool`)](#注册工具-tool)
5. [拦截工具调用 (`@on_tool_pre`)](#拦截工具调用-on_tool_pre)
6. [生命周期事件钩子 (`@on_*`)](#生命周期事件钩子-on_)
7. [生命周期管道模块 (`PhaseModule`)](#生命周期管道模块-phasemodule)
8. [Slot Export 机制](#slot-export-机制)
9. [插件配置 (`_conf_schema.json`)](#插件配置-_conf_schemajson)
10. [持久化存储 (`PluginKVStore`)](#持久化存储-pluginkvstore)
11. [元数据 (`manifest.yaml`)](#元数据-manifestyaml)
12. [初始化与清理](#初始化与清理)
13. [多插件协作](#多插件协作)
14. [完整示例：留言板插件](#完整示例留言板插件)
15. [参考](#参考)

---

## 概述

插件系统允许以独立目录形式扩展 agent 的行为，无需修改核心代码。

每个插件是一个包含 `plugin.py` 的目录，**放在 `plugins/` 下自动发现**。插件可以：

| 能力 | 使用方式 | 运行时机 |
|---|---|---|
| 注册新工具 | `@tool(name="...")` | LLM 调用工具时 |
| 拦截工具调用 | `@on_tool_pre(tool_name="...")` | 工具执行前 |
| 钩入生命周期事件 | `@on_before_turn` 等 | 对应生命周期阶段 |
| 插入生命周期管道模块 | `before/reasoning/step/after_*_modules_*()` 共 12 个注入点 | 对应阶段 |
| 通过 slot 导出数据 | `frame.slots["<phase>:<category>:<key>"] = value` | PhaseModule 内 |
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
    workspace: Path | None      # 工作区路径
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
|---|---|---|---|
| `before_turn` | `BeforeTurnCtx` | `content`, `skill_names`, `retrieved_memory_block`, `history_messages`, `extra_hints`, `extra_metadata`, `abort`, `abort_reply` |
| `before_reasoning` | `BeforeReasoningCtx` | `content`, `skill_names`, `retrieved_memory_block`, `extra_hints`, `abort`, `abort_reply` |
| `before_step` | `BeforeStepCtx` | `iteration`, `input_tokens_estimate`, `visible_tool_names`, `extra_hints`, `early_stop`, `early_stop_reply` |
| `after_step` | `AfterStepCtx` | `iteration`, `tools_called`, `partial_reply`, `extra_metadata` |
| `after_reasoning` | `AfterReasoningCtx` | `reply`, `thinking`, `tools_used`, `tool_chain`, `media`, `meme_tag`, `outbound_metadata` |
| `after_turn` | `AfterTurnCtx` | `reply`, `tools_used`, `thinking`, `will_dispatch`, `extra_metadata` |
| `before_tool_call` | `BeforeToolCallCtx` | `tool_name`, `arguments` |
| `after_tool_result` | `AfterToolResultCtx` | `tool_name`, `arguments`, `result`, `status` |

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
        print(f"[audit] turn 结束: reply={event.reply[:50]}")

    @on_before_turn(priority=100)   # 高优先级先执行
    async def content_filter(self, event: BeforeTurnCtx) -> BeforeTurnCtx | None:
        """屏蔽特定关键词。"""
        if "禁止词" in (event.content or ""):
            event.abort = True
            event.abort_reply = "此消息已被过滤器拦截。"
        return event
```

### GATE 阻断机制

GATE 阶段支持两种阻断方式：

**方式 1：直接修改 ctx（适用 before_emit 插件）**

```python
@on_before_turn()
async def gate(self, event: BeforeTurnCtx) -> BeforeTurnCtx:
    if self._should_block(event.content):
        event.abort = True
        event.abort_reply = "您的请求已被拦截。"
    return event
```

**方式 2：Slot export（适用 after_emit 插件）**

在 after_emit 位置不能直接改 ctx（GATE emit 已经发生），应通过 slot 触发 abort：

```python
class MyAfterEmitModule:
    async def run(self, frame):
        frame.slots["session:abort_reply"] = "由插件阻断"
        return frame
```

各阶段的 abort slot：`session:abort_reply`、`reasoning:abort_reply`、`step:abort_reply`。

`BeforeTurnCtx.abort` / `BeforeReasoningCtx.abort` 被设置后，`PassiveTurnPipeline.run()` 会直接返回 `abort_reply`，跳过推理和执行。`step:abort_reply` 映射为 `BeforeStepCtx.early_stop_reply`，只终止当前 tool loop。

### priority 参数

所有 `@on_*` 装饰器接受 `priority: int` 参数（默认 0）。值越大越先执行：

```python
@on_before_turn(priority=100)  # 第一个执行
async def first(self, event): ...

@on_before_turn(priority=-10)  # 最后执行
async def last(self, event): ...
```

---

## 生命周期管道模块 (`PhaseModule`)

每个生命周期阶段都是一条 `PhaseModule` 链，按顺序执行。插件通过定义特定方法返回模块列表来注入到各阶段的特定位置。

### 全阶段注入点一览

| 阶段 | 注入方法 | 插入位置 | 典型用途 |
|---|---|---|---|
| `before_turn` | `before_turn_modules_early()` | 记忆检索 + EventBus 之前 | 命令拦截、早期 abort |
| `before_turn` | `before_turn_modules_late()` | EventBus 之后、返回之前 | 补充 hints / metadata |
| `before_reasoning` | `before_reasoning_modules_before_emit()` | EventBus emit 之前 | 修改 ctx 字段（GATE 链） |
| `before_reasoning` | `before_reasoning_modules_after_emit()` | EventBus emit 之后、prompt warmup 之前 | 设置 slot export（hints、abort） |
| `prompt_render` | `prompt_render_modules_top()` | system sections top 拼装前 | 插入 system prompt 顶部 section |
| `prompt_render` | `prompt_render_modules_bottom()` | system sections bottom 拼装后 | 插入 system prompt 底部 section 或 extra_hints |
| `before_step` | `before_step_modules_before_emit()` | EventBus emit 之前 | 修改 BeforeStepCtx（GATE 链） |
| `before_step` | `before_step_modules_after_emit()` | EventBus emit 之后 | 设置 slot export（hints、early_stop） |
| `after_step` | `after_step_modules_before_fanout()` | fanout 之前 | 设置 telemetry slot（进入 fanout handler） |
| `after_step` | `after_step_modules_after_fanout()` | fanout 之后 | 补充 telemetry（不覆盖 fanout handler 已看到的） |
| `after_reasoning` | `after_reasoning_modules_before_emit()` | EventBus emit 之前 | 修改 ctx 字段（reply、media、outbound_metadata） |
| `after_reasoning` | `after_reasoning_modules_before_persist()` | persist 之前、emit 之后 | 设置 persist / outbound slot export |
| `after_turn` | `after_turn_modules_before_commit()` | TurnCommitted 构建之前 | 设置 `turn:extra:*` slot |
| `after_turn` | `after_turn_modules_before_fanout()` | AfterTurnCtx fanout 之前 | 设置 `turn:telemetry:*` slot |

### 阶段详解

#### before_turn 管道

```
1. _AcquireSessionModule              ← 加载 session
2. ★ plugin_modules_early             ← 适合做命令拦截（abort 跳过后续）
3. _PrepareContextModule              ← 记忆检索
4. _BuildBeforeTurnCtxModule          ← 构建 BeforeTurnCtx
5. _EmitBeforeTurnCtxModule           ← EventBus GATE 链（@on_before_turn）
6. ★ plugin_modules_late              ← 在 GATE 之后补充处理
7. _CollectBeforeTurnExportSlotsModule ← 收集 session:extra_hint:* / session:abort_reply
8. _ReturnBeforeTurnCtxModule         ← 产出最终 output
```

#### before_reasoning 管道

```
1. _SyncToolContextModule             ← 设置 tool context
2. _BuildBeforeReasoningCtxModule     ← 构建 ctx（继承 before_turn 字段）
3. ★ plugin_modules_before_emit        ← GATE 链：直接修改 ctx
4. _EmitBeforeReasoningCtxModule      ← EventBus GATE 链（@on_before_reasoning）
5. ★ plugin_modules_after_emit         ← 设置 reasoning:extra_hint:* / reasoning:abort_reply
6. _CollectBeforeReasoningExportSlotsModule ← 收集 slot 到 ctx
7. _PromptWarmupModule                ← 预热 prompt（ctx.abort 时跳过）
8. _ReturnBeforeReasoningCtxModule    ← 产出 ctx
```

#### prompt_render 管道

```
1. _BuildPromptRenderCtxModule        ← 构建 ctx
2. _EmitPromptRenderCtxModule         ← EventBus emit
3. ★ plugin_modules_top               ← 通过 system_sections_top / prompt:section_top:* 插入
4. ★ plugin_modules_bottom            ← 通过 system_sections_bottom / prompt:section_bottom:* 或 extra_hints
5. _CollectPromptExportSlotsModule    ← 收集 prompt:section_*:*/ prompt:extra_hint:*
6. _RenderPromptModule                ← ContextBuilder.render()
7. _ReturnPromptRenderResultModule    ← 产出 PromptRenderResult
```

#### before_step 管道（每轮 tool loop iteration）

```
1. _BuildBeforeStepCtxModule          ← 构建 ctx
2. ★ plugin_modules_before_emit        ← GATE 链：直接修改 ctx
3. _EmitBeforeStepCtxModule           ← EventBus GATE 链（@on_before_step）
4. ★ plugin_modules_after_emit         ← 设置 step:extra_hint:* / step:abort_reply
5. _CollectBeforeStepExportSlotsModule ← 收集 slot 到 ctx（early_stop、extra_hints）
6. _InjectHintsModule                 ← 将 extra_hints 注入 messages
7. _ReturnBeforeStepCtxModule         ← 产出 ctx
```

#### after_step 管道（每轮 tool loop iteration 后）

```
1. _CopyInputToCtxModule              ← 输入快照复制为 ctx
2. ★ plugin_modules_before_fanout      ← 设置 step:telemetry:*
3. _CollectAfterStepExportSlotsModule  ← 第 1 次收集：记入 extra_metadata + tracked keys
4. _FanoutAfterStepCtxModule          ← TAP fanout（handler 看到 collected telemetry）
5. ★ plugin_modules_after_fanout       ← 补充新的 telemetry key
6. _CollectAfterStepExportSlotsModule  ← 第 2 次收集：只补充未被 collected 的 key
7. _ReturnAfterStepCtxModule          ← 产出 ctx
```

> **注意**：after_step 的 telemetry 收集分两次。fanout 前的 telemetry 会被 handler 看到并锁定；fanout 后的补充只能新增 key，不能覆盖已看过的同名 key。

#### after_reasoning 管道

```
1. _BuildAfterReasoningCtxModule      ← 构建 ctx（含 outbound_metadata 初始值）
2. ★ plugin_modules_before_emit        ← GATE 链：修改 ctx 字段
3. _EmitAfterReasoningCtxModule       ← EventBus GATE 链（@on_after_reasoning）
4. ★ plugin_modules_before_persist     ← 设置 persist:user:*/ persist:assistant:*/ outbound:metadata:*/ outbound:media:*
5. _PersistUserMessageModule          ← 持久化 user 消息（读取 persist:user:*）
6. _PersistAssistantMessageModule     ← 持久化 assistant 消息（读取 persist:assistant:*）
7. _UpdateSessionMetadataModule       ← 更新 session 运行时 metadata
8. _AppendMessagesModule              ← 追加到 session 存储
9. _BuildOutboundMessageModule        ← 构建出站消息（读取 outbound:metadata:* / outbound:media:*）
10. _ReturnAfterReasoningResultModule ← 产出 AfterReasoningResult
```

#### after_turn 管道

```
1. _BuildTurnWorkModule               ← 构建 budget、react_stats、extra 等
2. ★ plugin_modules_before_commit      ← 设置 turn:extra:*
3. _CollectAfterTurnExtraSlotsModule   ← 收集 turn:extra:* → extra dict
4. _BuildTurnCommittedModule           ← 构建 TurnCommitted 事件（依赖 extra_collected）
5. _FanoutTurnCommittedModule          ← fanout TurnCommitted
6. _LogBudgetModule                    ← 日志记录 budget
7. _BuildAfterTurnCtxModule            ← 构建 AfterTurnCtx
8. ★ plugin_modules_before_fanout      ← 设置 turn:telemetry:*
9. _CollectAfterTurnTelemetrySlotsModule ← 收集 turn:telemetry:* → extra_metadata
10. _FanoutAfterTurnCtxModule          ← TAP fanout（@on_after_turn）
11. _DispatchOutboundModule            ← 派发出站消息
12. _ReturnOutboundMessageModule       ← 返回 OutboundMessage
```

> **注意**：after_turn 的 `_BuildTurnCommittedModule` 依赖 `_EXTRA_COLLECTED_SLOT`，保证 `_CollectAfterTurnExtraSlotsModule` 一定先于它执行。这通过模块的 `requires` / `produces` 契约实现。

### 编写 PhaseModule

`PhaseModule` 协议：

```python
class MyModule:
    requires = ("phase:slot_name",)   # 可选：依赖哪些 slot（用于顺序校验）
    produces = ("phase:slot_name",)   # 可选：产出哪些 slot

    async def run(self, frame: PhaseFrame) -> PhaseFrame:
        # 通过 frame.slots 读写中间状态
        # 通过 frame.input / frame.output 读写输入输出
        return frame
```

完整的 GATE 链 ctx 修改示例（before_reasoning）：

```python
class MyBeforeReasoningModule:
    requires = ("reasoning:ctx",)
    produces = ("reasoning:ctx",)

    async def run(self, frame):
        from typing import cast
        from agent.lifecycle.types import BeforeReasoningCtx
        ctx = cast(BeforeReasoningCtx, frame.slots["reasoning:ctx"])
        # 直接修改 ctx 字段（GATE 链允许）
        ctx.extra_hints.append("hint from my plugin")
        ctx.abort = True
        ctx.abort_reply = "由我的插件阻断"
        frame.slots["reasoning:ctx"] = ctx
        return frame
```

### 关键 Slot 名速查

| 阶段 | Slot | 内容 |
|---|---|---|
| before_turn | `session:session` | `Session` 对象 |
| before_turn | `session:context_bundle` | `ContextBundle` |
| before_turn | `session:ctx` | `BeforeTurnCtx` |
| before_reasoning | `reasoning:ctx` | `BeforeReasoningCtx` |
| prompt_render | `prompt:ctx` | `PromptRenderCtx` |
| prompt_render | `prompt:result` | `PromptRenderResult` |
| before_step | `step:ctx` | `BeforeStepCtx` |
| after_step | `step:ctx` | `AfterStepCtx` |
| after_step | `step:telemetry_collected` | `set[str]` — 已收集过的 telemetry key |
| after_reasoning | `reasoning:ctx` | `AfterReasoningCtx` |
| after_reasoning | `reasoning:outbound` | `OutboundMessage` |
| after_turn | `turn:budget` | `dict` — post-reply context budget |
| after_turn | `turn:extra` | `dict` — TurnCommitted extra |
| after_turn | `turn:extra_collected` | `True` — 标记 extra 已收集 |
| after_turn | `turn:committed` | `TurnCommitted` 事件 |
| after_turn | `turn:ctx` | `AfterTurnCtx` |

### 从插件暴露 PhaseModule

在 Plugin 子类上定义对应的方法，返回模块列表。全部可用方法：

```python
class MyPlugin(Plugin):
    name = "my_plugin"

    def before_turn_modules_early(self) -> list[object]:
        return [MyCommandModule()]

    def before_turn_modules_late(self) -> list[object]:
        return [MyHintModule()]

    def before_reasoning_modules_before_emit(self) -> list[object]:
        return [MyReasoningGateModule()]

    def before_reasoning_modules_after_emit(self) -> list[object]:
        return [MyReasoningSlotModule()]

    def prompt_render_modules_top(self) -> list[object]:
        return [MyTopSectionModule()]

    def prompt_render_modules_bottom(self) -> list[object]:
        return [MyBottomSectionModule()]

    def before_step_modules_before_emit(self) -> list[object]:
        return [MyStepGateModule()]

    def before_step_modules_after_emit(self) -> list[object]:
        return [MyStepSlotModule()]

    def after_step_modules_before_fanout(self) -> list[object]:
        return [MyStepTelemetryModule()]

    def after_step_modules_after_fanout(self) -> list[object]:
        return [MyStepLateTelemetryModule()]

    def after_reasoning_modules_before_emit(self) -> list[object]:
        return [MyAfterReasoningGateModule()]

    def after_reasoning_modules_before_persist(self) -> list[object]:
        return [MyPersistModule()]

    def after_turn_modules_before_commit(self) -> list[object]:
        return [MyTurnExtraModule()]

    def after_turn_modules_before_fanout(self) -> list[object]:
        return [MyTurnTelemetryModule()]
```

仅定义你需要的方法即可，管理器只收集已定义的方法返回的模块。

---

---

## Slot Export 机制

Slot export 是 PhaseModule 间传递数据的标准方式。插件通过向 `frame.slots` 写入带前缀的 key，由 collection 模块自动合并到下游 ctx 或持久化数据中。

### 核心 Helper

```python
from agent.lifecycle.phase import collect_prefixed_slots, append_string_exports
```

- `collect_prefixed_slots(slots, prefix, *, reserved=())` — 收集所有以 `prefix` 开头的 slot，返回 `dict[str, object]`。`reserved` 集合中的 key 会被排除（用于保护内置字段不被覆盖）。
- `append_string_exports(target, exports)` — 将 exports 中的字符串（或字符串列表）追加到 `target` 列表中。

### 可用 Slot 前缀

| 前缀 | 阶段 | 语义 | 目标 |
|---|---|---|---|
| `session:extra_hint:*` | before_turn (late) | 注入额外提示 | `BeforeTurnCtx.extra_hints` → reasoning |
| `session:abort_reply` | before_turn (late) | 阻断 turn 并返回此文本 | `BeforeTurnCtx.abort_reply` |
| `reasoning:extra_hint:*` | before_reasoning | 注入额外提示 | `BeforeReasoningCtx.extra_hints` → prompt |
| `reasoning:abort_reply` | before_reasoning | 阻断 reasoning 并返回此文本 | `BeforeReasoningCtx.abort_reply` |
| `prompt:section_top:*` | prompt_render | 插入 system prompt 顶部 | `system_sections_top` → rendered |
| `prompt:section_bottom:*` | prompt_render | 插入 system prompt 底部 | `system_sections_bottom` → rendered |
| `prompt:extra_hint:*` | prompt_render | 注入 context hint | `PromptRenderCtx.extra_hints` |
| `step:extra_hint:*` | before_step | 注入额外提示到 messages | `BeforeStepCtx.extra_hints` → `_InjectHintsModule` |
| `step:abort_reply` | before_step | 提前终止当前 tool loop | `BeforeStepCtx.early_stop_reply` |
| `step:telemetry:*` | after_step | 注入 telemetry 到 extra_metadata | `AfterStepCtx.extra_metadata` |
| `persist:user:*` | after_reasoning | 持久化到 user 消息的额外字段 | user message kwargs |
| `persist:assistant:*` | after_reasoning | 持久化到 assistant 消息的额外字段 | assistant message kwargs |
| `outbound:metadata:*` | after_reasoning | 注入出站消息 metadata | `OutboundMessage.metadata` |
| `outbound:media:*` | after_reasoning | 追加出站媒体 URL | `OutboundMessage.media` |
| `turn:extra:*` | after_turn | 注入 TurnCommitted extra 字段 | `TurnCommitted.extra` |
| `turn:telemetry:*` | after_turn | 注入 AfterTurnCtx telemetry | `AfterTurnCtx.extra_metadata` |

### 示例：在 before_reasoning 中注入 extra_hints

```python
from agent.lifecycle.phase import collect_prefixed_slots, append_string_exports
from agent.lifecycle.types import BeforeReasoningCtx

class MyHintModule:
    requires = ("reasoning:ctx",)
    produces = ("reasoning:ctx",)

    async def run(self, frame):
        # 方式 1：直接写入 slot，collection 模块会自动处理
        frame.slots["reasoning:extra_hint:weather"] = "今天北京有暴雨，建议提醒用户带伞"
        return frame
```

### 示例：在 prompt_render 中插入 system section

```python
from agent.prompting import PromptSectionRender

class MySectionModule:
    async def run(self, frame):
        # 可以用字符串
        frame.slots["prompt:section_bottom:custom"] = "## 自定义规则\n请用中文回答。"
        # 也可以用 PromptSectionRender（控制 is_static 等属性）
        frame.slots["prompt:section_bottom:rules"] = PromptSectionRender(
            name="rules",
            content="你是一个严肃、准确的助手，不可编造事实。",
            is_static=False,
        )
        return frame
```

### 示例：在 after_reasoning 中注入 persist / outbound

```python
class MyPersistModule:
    async def run(self, frame):
        # 给 user 消息添加自定义字段
        frame.slots["persist:user:plugin_tag"] = "my_plugin_v1"
        # 给 assistant 消息添加自定义字段
        frame.slots["persist:assistant:confidence"] = 0.95
        # 给出站消息添加 metadata
        frame.slots["outbound:metadata:source_plugin"] = "my_plugin"
        # 给出站消息追加媒体文件
        frame.slots["outbound:media:chart"] = "/tmp/output.png"
        return frame
```

### Slot Export 的执行顺序

每个阶段的 collection 模块在 pipeline 中的位置是固定的（见上一节各阶段管道图）。一般模式：

- **after_emit 插件** 写入 slot → **collection 模块** 读取并合并
- **after_step** 特殊：分两次 collect（fanout 前锁定 + fanout 后补充）
- **after_turn** 特殊：`turn:extra:*` 在 `_CollectAfterTurnExtraSlotsModule` 收集，`turn:telemetry:*` 在 `_CollectAfterTurnTelemetrySlotsModule` 收集

### Slot vs 直接修改 ctx

| 方式 | 适用场景 | 示例 |
|---|---|---|
| 直接修改 ctx（GATE 链） | before_emit 位置，需要 EventBus handler 也能看到变化 | `ctx.reply = "new reply"` |
| Slot export | after_emit / before_fanout / before_persist 位置，多个插件各自产出独立数据 | `frame.slots["persist:user:tag"] = "v1"` |

选择原则：如果需要 EventBus handler（`@on_*`）能看到变化 → 直接改 ctx。如果只是向下游模块传递数据 → slot export。

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
        content_len = len(event.reply or "")
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
        """先执行：高优先级追加 extra_hints"""
        event.extra_hints.append("\n附加规则 A")
        return event

class PluginB(Plugin):
    @on_before_reasoning(priority=-10)
    async def b_late(self, event: BeforeReasoningCtx) -> BeforeReasoningCtx:
        """后执行：看到的是 A 追加过的 hints"""
        print(f"当前 hints: {event.extra_hints}")
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
