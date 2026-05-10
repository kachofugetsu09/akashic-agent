# 插件系统

agent 每次收到一条用户消息，经过 6 个生命周期阶段产出一条回复。插件可以在这些阶段的任意位置插入自己的逻辑。

## 先理解它怎么运转的

插件放到 `plugins/` 目录下，启动时自动发现、加载、注册。过程：

```
PluginManager.discover()
  → 扫描 plugins/ 下每个有 plugin.py 的子目录
  → 动态 import plugin.py，Plugin.__init_subclass__() 自动注册
  → 调用 initialize()，失败则回滚该插件的所有注册
```

插件有 **4 种方式** 介入 agent 的行为。不是四种独立的系统，是同一份代码可以同时使用全部四种。

| 机制 | 适合做什么 | 在哪用 |
|------|-----------|--------|
| **PhaseModule** | 在某个 Phase 的精确位置注入逻辑 | 写一个方法返回模块列表，模块链会自动编排 |
| **EventBus 装饰器** | 在生命周期关键节点做观察或改写 | `@on_before_turn` 等 9 种 |
| **@on_tool_pre** | 工具执行前拦截、改参数、拒绝 | LLM 调用工具时 |
| **@tool** | 给 LLM 注册新工具 | 任何时候 LLM 决定调工具 |

---

## PhaseModule：插件系统最核心的机制

每个生命周期 Phase 是一条模块链。插件的模块和内置模块在链上是平等的——都声明 `slot`（身份）、`requires`（依赖哪个内置模块）、`produces`（产出什么数据 slot），框架用**拓扑排序**自动决定执行顺序。

```
BeforeTurn 管道（拓扑排序后）:
  before_turn.acquire_session  →  setup_helper.chatid  →  before_turn.prepare_context  →  before_turn.build_ctx  →  before_turn.emit  →  recall_inspector.main  →  before_turn.collect_exports  →  before_turn.return
                ↑                            ↑                               ↑                                          ↑                             ↑
           插件注入点                   requires="before_turn.acquire_session"                                     插件注入点               requires="before_turn.emit"
```

插件不再选"early 还是 late"这种固定注入点，而是**精确声明依赖哪个内置模块的 slot**。`requires=("before_turn.acquire_session",)` 就是 early，`requires=("before_turn.emit",)` 就是 late。

### 写一个 PhaseModule

```python
from agent.prompting import PromptSectionRender

class MyPromptModule:
    slot = "my_plugin.prompt"               # 模块身份，用于被依赖
    requires = ("prompt_render.emit",       # 声明依赖：在 emit 之后运行
                "prompt:ctx",)              # 声明依赖：需要 prompt:ctx 数据 slot
    produces = ("prompt:ctx",)              # 声明产出：可能改写 prompt:ctx

    async def run(self, frame):
        ctx = frame.slots["prompt:ctx"]     # PromptRenderCtx 实例
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="my_rules",
                content="## 自定义规则\n请用中文回答。",
                is_static=True,
            )
        )
        return frame
```

然后在 Plugin 子类上暴露它：

```python
class MyPlugin(Plugin):
    name = "my_plugin"

    def prompt_render_modules(self):        # 7 个统一方法之一
        return [MyPromptModule()]
```

`requires` / `produces` 在**启动时**就会被校验——如果链上某个模块要求 `reasoning:ctx` 但前面没有人产出它，启动就会报 warning。`slot` 声明为空的模块也会在拓扑排序时直接报错。

### 7 个注入方法 + 内置模块 Slot 锚点

插件覆写 7 个方法之一返回模块列表，每个模块通过 `requires` 声明依赖的内置模块 slot 来决定精确插入位置。

| Phase | 方法 | 常用锚点 slot（`requires=`） | 用途 |
|-------|------|---------------------------|------|
| BeforeTurn | `before_turn_modules()` | `before_turn.acquire_session`（early 拦截）<br>`before_turn.emit`（late 补充） | 命令拦截、记录上下文 |
| BeforeReasoning | `before_reasoning_modules()` | `before_reasoning.sync_tools`（emit 前修改）<br>`before_reasoning.emit`（emit 后传数据） | 修改 ctx、写 slot |
| PromptRender | `prompt_render_modules()` | `prompt_render.emit`（注入 section）<br>`citation.prompt`（在 citation 之后） | 注入 system prompt |
| BeforeStep | `before_step_modules()` | `before_step.build_ctx`（emit 前）<br>`before_step.emit`（emit 后） | 修改每步 ctx |
| AfterStep | `after_step_modules()` | `after_step.copy_input`（fanout 前）<br>`after_step.fanout`（fanout 后） | telemetry |
| AfterReasoning | `after_reasoning_modules()` | `after_reasoning.build_ctx`（emit 前改 reply）<br>`after_reasoning.emit`（emit 后清理）<br>`after_reasoning.persist_user`（持久化前写 slot） | 回复后处理 |
| AfterTurn | `after_turn_modules()` | `after_turn.build_work`（commit 前写 extra）<br>`after_turn.fanout_committed`（fanout 前写 telemetry） | 收尾 |

**完整的内置模块 slot 列表**可通过 `python main.py --inspect-modules` 打印 ASCII 依赖树查看。

---

## Slot：模块间的类型化数据总线

这是插件系统最巧妙的设计。每个 Phase 内部，模块不直接调下一个模块——它们只读写 `frame.slots`，管道的 collection 模块负责把 slots 翻译成下游能用的东西。

### Slot 的工作原理

```
Module A: frame.slots["persist:assistant:cited_memory_ids"] = ["m1", "m2"]
         ↓
Module B: （不需要关心 A 做了什么）
         ↓
_Collect 模块在管道末尾扫描所有 slots，按前缀规则合并到输出
         ↓
_PersistAssistantMessage 模块从合并好的数据里拿到了 cited_memory_ids
```

你不需要知道下游模块的签名，不需要 import 任何东西——**写对 slot key 就行**。

### 完整的 Slot 前缀表

| 前缀 | 阶段 | 最终去向 |
|------|------|---------|
| `session:extra_hint:*` | BeforeTurn late | 合并进 `BeforeTurnCtx.extra_hints`，最终注入 system prompt |
| `session:abort_reply` | BeforeTurn late | 设置 abort → 整个 turn 跳过推理，直接返回这段文本 |
| `reasoning:extra_hint:*` | BeforeReasoning | 合并进 `BeforeReasoningCtx.extra_hints` → prompt |
| `reasoning:abort_reply` | BeforeReasoning | 跳过 LLM 推理 |
| `prompt:section_top:*` | PromptRender | 插入 system prompt 顶部 |
| `prompt:section_bottom:*` | PromptRender | 插入 system prompt 底部 |
| `prompt:extra_hint:*` | PromptRender | context frame 额外提示 |
| `step:extra_hint:*` | BeforeStep | 注入到本轮 messages 中 |
| `step:early_stop_reason` | BeforeStep | 停止 tool loop |
| `step:telemetry:*` | AfterStep | 合并进 `AfterStepCtx.extra_metadata`（TAP handler 可见） |
| `persist:user:*` | AfterReasoning | 持久化到 user 消息的额外字段 |
| `persist:assistant:*` | AfterReasoning | 持久化到 assistant 消息的额外字段 |
| `persist:assistant:cited_memory_ids` | AfterReasoning | citation 插件专用：被引用的记忆 ID 列表 |
| `outbound:metadata:*` | AfterReasoning | 出站消息的 metadata 字典 |
| `outbound:media:*` | AfterReasoning | 追加出站媒体 URL |
| `turn:extra:*` | AfterTurn | TurnCommitted 事件的 extra 字段 |
| `turn:telemetry:*` | AfterTurn | 合并进 `AfterTurnCtx.extra_metadata` |

### 实际例子

**例 1：注入一个提示到 system prompt（最简单）**

```python
class HintModule:
    async def run(self, frame):
        frame.slots["prompt:section_bottom:weather"] = "# 天气规则\n今天北京暴雨，建议提醒用户带伞"
        return frame
```

`prompt:section_bottom:*` 前缀 + 自定义 key `weather`。管道的 `_CollectPromptExportSlotsModule` 会自动扫描所有以 `prompt:section_bottom:` 开头的 slot，合并进 system prompt。

**例 2：在 after_step 写入 telemetry 让 TAP handler 能看到**

```python
class PressureModule:
    async def run(self, frame):
        ctx = frame.slots["step:ctx"]
        if ctx.context_tokens_estimate > 800_000:
            frame.slots["step:early_stop_reason"] = "context_pressure"
            frame.slots["step:telemetry:pressure_tokens"] = ctx.context_tokens_estimate
        return frame
```

`after_step_before_fanout` 位置写入的 `step:telemetry:*` 会在 fanout 之前被收集进 `extra_metadata`，所有 `@on_after_step` TAP handler 都能读到（但 `after_step_after_fanout` 写入的 telemetry 只有第二轮收集才能拿到，fanout 的 handler 已经看不到了）。

**例 3：在 after_reasoning 中给 assistant 消息加自定义字段**

```python
class CitationModule:
    async def run(self, frame):
        # 正则扫 reply 里有没有 §cited:[id1,id2]§
        reply = frame.slots["reasoning:ctx"].reply
        cleaned, ids = extract_cited_ids(reply)
        if ids:
            # 写 slot → 持久化模块自动拿这个 key 写数据库
            frame.slots["persist:assistant:cited_memory_ids"] = ids
        # 把标签从 reply 文本里剥掉
        frame.slots["reasoning:ctx"].reply = cleaned
        return frame
```

这就是 citation 插件的实际写法。`persist:assistant:cited_memory_ids` 是一个约定好的 slot key——`_PersistAssistantMessageModule` 在持久化时专门会读这个 key，把它存到消息记录的 `extra` 字段里。**你只需要知道这个 key 名，不需要知道是谁读的、怎么存的。**

---

## EventBus 装饰器

PhaseModule 适合"在某一个精确位置做一件事"。但如果你的逻辑很简单——比如"每轮结束后记录一下回复长度"——直接挂一个装饰器就够。

```python
from agent.plugins import Plugin, on_after_turn
from agent.lifecycle.types import AfterTurnCtx

class MyPlugin(Plugin):
    name = "my_plugin"

    @on_after_turn()
    async def log_reply(self, ctx: AfterTurnCtx) -> None:
        print(f"回复长度: {len(ctx.reply)}")
```

### GATE（可修改事件）

这 5 个装饰器的 handler 可以修改 ctx，甚至可以用 `ctx.abort = True` 阻断整个 turn：

| 装饰器 | ctx 类型 | 关键可写字段 |
|--------|---------|-------------|
| `@on_before_turn()` | `BeforeTurnCtx` | `abort`, `abort_reply`, `extra_hints` |
| `@on_before_reasoning()` | `BeforeReasoningCtx` | `abort`, `abort_reply`, `extra_hints` |
| `@on_prompt_render()` | `PromptRenderCtx` | `system_sections_bottom`, `extra_hints` |
| `@on_before_step()` | `BeforeStepCtx` | `extra_hints`, `early_stop`, `early_stop_reply` |
| `@on_after_reasoning()` | `AfterReasoningCtx` | `reply`, `media`, `meme_tag`, `outbound_metadata` |

```python
@on_before_turn()
async def block_spam(self, ctx: BeforeTurnCtx) -> BeforeTurnCtx:
    if "禁止词" in ctx.content:
        ctx.abort = True
        ctx.abort_reply = "此消息已被拦截"
    return ctx  # GATE 必须返回 ctx
```

`priority` 参数控制执行顺序（越大越先执行，默认 0）：

```python
@on_before_turn(priority=100)
async def first(self, ctx): ...
```

### TAP（只观察）

这 4 个装饰器的 handler 不能修改事件，只能观察/记录：

| 装饰器 | ctx 类型 |
|--------|---------|
| `@on_after_step()` | `AfterStepCtx` |
| `@on_after_turn()` | `AfterTurnCtx` |
| `@on_tool_call()` | `BeforeToolCallCtx` |
| `@on_tool_result()` | `AfterToolResultCtx` |

---

## @on_tool_pre — LLM 调工具前拦截

PhaseModule 和 EventBus 管的是生命周期，`@on_tool_pre` 管的是**工具调用**。

```python
from agent.plugins import Plugin, on_tool_pre
from agent.lifecycle.types import PreToolCtx
from agent.tool_hooks import HookOutcome

class ShellSafety(Plugin):
    name = "shell_safety"

    @on_tool_pre(tool_name="shell")
    async def block_interactive(self, ctx: PreToolCtx) -> HookOutcome | dict | None:
        cmd = str(ctx.arguments.get("command", ""))
        if "vi " in cmd or "vim " in cmd:
            return HookOutcome(decision="deny", reason="禁止交互式命令")
        return None
```

`PreToolCtx` 关键字段：`tool_name`, `arguments`（dict）, `session_key`, `call_id`, `source`, `tool_batch`（批量调用时的完整列表）, `tool_batch_index`（自己在批量中的位置）。

返回值三种：
- `None` — 不管，继续执行
- `dict` — 替换 arguments
- `HookOutcome(decision="deny")` — 拒绝调用

**不填 `tool_name` 匹配所有工具**（tool_loop_guard 就这么做）。

---

## @tool — 注册 LLM 可调用的工具

```python
from agent.plugins import Plugin, tool

class MyPlugin(Plugin):
    name = "my_plugin"

    @tool(name="get_weather", risk="read-only", search_hint="查天气")
    async def get_weather(self, event, city: str, date: str = "today") -> str:
        """查询指定城市的天气。

        Args:
            city: 城市名称
            date: 日期，默认 today
        """
        return f"{city} {date}：晴，23°C"
```

JSON Schema 从函数签名和 docstring 的 `Args:` 段自动生成。前两个参数 `self` 和 `event` 固定（`event` 传 `None`）。

---

## 配置文件

```json
// 插件目录下 _conf_schema.json（声明默认值）
{"max_results": {"default": 5, "description": "最大返回数"}}

// 插件目录下 plugin_config.json（用户覆盖）
{"max_results": 10}
```

代码里读：`self.context.config.get("max_results")` 或 `self.context.config.max_results`。

---

## 其他

| 能力 | API |
|------|-----|
| KV 存储 | `self.context.kv_store.get/set/increment()` → `{plugin_dir}/.kv.json` |
| 直接订阅 EventBus | `self.context.event_bus.on(TurnCommitted, handler)` — 不需要装饰器 |
| 初始化 / 清理 | `async def initialize(self)`, `async def terminate(self)` |

---

## 例子

### citation — 纯 PhaseModule 的典型用法

```python
class CitationPlugin(Plugin):
    name = "citation"

    # 三个模块在一个方法里返回，topo-sort 自动按 requires 排好
    def after_reasoning_modules(self):
        return [CitationAfterReasoningModule(), ProtocolTagCleanupModule()]

    def prompt_render_modules(self):
        return [CitationPromptModule()]
```

三个模块通过 `slot` / `requires` 声明各自的插入位置：

| 模块 | slot | requires | 效果 |
|------|------|----------|------|
| `CitationPromptModule` | `citation.prompt` | `prompt_render.emit` | 在 emit 之后注入引用协议到 system prompt |
| `CitationAfterReasoningModule` | `citation.after_reasoning` | `after_reasoning.build_ctx` | 在 build 之后扫 `§cited:[]§` 提取 ID |
| `ProtocolTagCleanupModule` | `citation.protocol_cleanup` | `after_reasoning.emit` | 在 emit 之后清理残留协议标签 |

**meme 插件依赖 citation**：`MemePromptModule` 的 `requires=("citation.prompt", ...)` —— citation 产出 prompt section 后 meme 才运行。框架自动保证顺序。

### tool_loop_guard — 纯 @on_tool_pre 的典型用法

```python
class ToolLoopGuard(Plugin):
    name = "tool_loop_guard"

    @on_tool_pre()  # 通配所有工具
    async def guard(self, ctx: PreToolCtx) -> HookOutcome | None:
        sig = f"{ctx.tool_name}:{json.dumps(ctx.arguments, sort_keys=True)}"
        state = self._states.setdefault(ctx.session_key, _LoopState())
        if sig == state.signature:
            state.count += 1
        else:
            state.signature = sig
            state.count = 1
        if state.count >= 3:
            return HookOutcome(decision="deny", reason="连续重复 3 次")
        return None
```

### 命令拦截 — 用 slot 做 abort

```python
class UndoCommandModule:
    slot = "plugin_undo.undo"
    requires = ("before_turn.acquire_session",  # ← 锚定 acquire_session 之后
                "session:session",)              # ← 需要 session 数据
    produces = ("session:ctx",)

    async def run(self, frame):
        if "session:ctx" in frame.slots:
            return frame
        state = frame.input
        if state.msg.content.strip() == "/undo":
            frame.slots["session:ctx"] = BeforeTurnCtx(
                ..., abort=True, abort_reply="已回滚上一轮。"
            )
        return frame
```

`requires=("before_turn.acquire_session",)` 让它跑在 prepare_context 之前——整个 turn 不走记忆检索、不走 LLM 推理，直接返回。

---

## 所有真实插件

| 插件 | 用了什么 | 一句话 |
|------|---------|--------|
| citation | PhaseModule ×3 | 扫 `§cited:[]§` 标签提取被引用的记忆 ID |
| meme | PhaseModule + @on_after_reasoning GATE | 扫 `<meme:tag>` 替换成表情图片（依赖 citation） |
| observe | 裸 EventBus 订阅 | 全链路 trace 写入 observe.db |
| tool_loop_guard | @on_tool_pre(通配) | 连续重复调同一工具 3 次就 deny |
| shell_safety | @on_tool_pre(shell) | 拒绝 vi/vim/sudo 等交互命令 |
| shell_restore | @on_tool_pre(shell) | rm 改写成 mv |
| context_pressure | PhaseModule | token 超 80% 窗口时提前停止 tool loop |
| plugin_undo | PhaseModule | /undo 回滚 |
| status_commands | PhaseModule | /memorystatus, /kvcache |
| setup_helper | PhaseModule | /chatid, /myid |
| recall_inspector | PhaseModule + @on_tool_result TAP | recall_memory 调用追踪 |
