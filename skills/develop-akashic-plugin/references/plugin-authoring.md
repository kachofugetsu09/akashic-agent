# Akashic 插件编写参考

## 目录

1. 最小插件
2. Tool
3. Prompt 注入
4. Skill
5. MCP 与其他贡献
6. 状态与生命周期
7. Source 验证

## 1. 最小插件

当前 canonical API 来自仓库的 `agent/plugins/base.py`、`agent/plugins/decorators.py`、`agent/plugins/specs.py` 和 `agent/plugins/__init__.py`。先读这些文件和一个相邻插件；接口漂移时以代码为准。

最小 `plugin.py`：

```python
from agent.plugins import Plugin, tool


class ExamplePlugin(Plugin):
    name = "example"
    version = "0.1.0"

    @tool(
        name="example_lookup",
        risk="read-only",
        search_hint="look up an example value",
    )
    async def lookup(self, event, key: str) -> str:
        """Look up one example value.

        Args:
            key: Value key to look up.
        """
        return self.context.kv.get(key) or "not found"
```

要求：

- `plugin.py` 位于 Git 仓库根。
- `name`、`version` 是安全的非空路径片段。
- 不在 import 阶段启动进程、打开端口或写正式状态。
- 需要初始化时使用 `prepare/activate/retire/terminate` 的既有生命周期。
- 非平凡 readiness 用 `static_semantic_checks()` 或 `readiness_semantic_checks()` 返回结构化 `PluginSemanticCheck`。

## 2. Tool

`@tool` handler 的前两个参数必须是 `self, event`。其余参数由注解和 docstring 生成 schema。

- `risk="read-only"`：不改变文件、数据库、远程服务或消息。
- `risk="read-write"`：可能产生写入或外部效果；不要为了方便错标 read-only。
- `always_on=True` 只给每轮确实需要的窄能力；普通能力依赖 ToolSearch。
- `search_hint` 描述何时找这个工具，不复述实现。
- 内部契约违反直接失败；只捕获当前位置能真实恢复或转换的具体异常。

测试要同时断言 schema、真实调用结果和失败语义。仅断言装饰器注册不证明工具可用。

## 3. Prompt 注入

只需要追加 system prompt section 时，直接使用当前 `on_prompt_render` 事件，不要先搜索 phase slot、编写自定义 module factory 或复制 ContextBuilder：

```python
from agent.lifecycle.types import PromptRenderCtx
from agent.plugins import Plugin, on_prompt_render
from agent.prompting import PromptSectionRender


class PromptOnlyPlugin(Plugin):
    name = "prompt_only"
    version = "0.1.0"

    @on_prompt_render(priority=100)
    async def inject_rule(self, ctx: PromptRenderCtx) -> PromptRenderCtx:
        ctx.system_sections_bottom.append(
            PromptSectionRender(
                name="prompt_only_rule",
                content="这里写可独立断言的规则。",
                is_static=True,
            )
        )
        return ctx
```

source test 至少直接调用 handler，断言 section 的 name、content、位置和静态属性，并证明没有意外 Tool handler。真实有效性仍由安装后的 latest child 断言，source test 不能替代模型行为。

测试 `channels()`、`jobs()` 等普通实例方法时先实例化插件；只有 `skill_roots()`、`mcp_servers()`、`managed_services()` 等基类声明为 `@classmethod` 的能力才直接从类调用。不要为确认这个区别遍历 manager/EventBus 实现。

## 4. Skill

插件 Skill 放在：

```text
skills/<skill-name>/
├── SKILL.md
├── references/   可选
├── scripts/      可选
└── assets/       可选
```

插件类声明：

```python
@classmethod
def skill_roots(cls) -> tuple[str, ...]:
    return ("skills",)
```

`SKILL.md` frontmatter 至少包含 `name` 和 `description`。description 同时说明功能和触发场景；正文使用命令式步骤，保持精简。详细 schema、范例和领域规则放入一层 `references/`，并从 SKILL.md 明确路由。

验证 Skill 不止检查文件存在：

1. 用 `SkillsLoader` 或 runtime catalog 证明名称、source 和 available。
2. 调用 `load_skill` 证明正文与相对资源可读取。
3. 安装后由父 turn 创建 attached programmatic child，让 Core 自动绑定候选；不得手工选择 latest。Core 不支持因果绑定时，只能在一次性隔离 runtime 验证并报告候选隔离不可用。
4. 从 tool items、产物或领域状态证明 Skill 被正确执行。

## 5. MCP 与其他贡献

只通过 Plugin API 声明：

- `mcp_servers()` → `McpServerSpec`
- `managed_services()` → `ManagedServiceSpec`
- `proactive_sources()` → `ProactiveSourceSpec`
- `channels()` → Channel
- `jobs()` → `PluginJobSpec`
- `drift_skill_roots()` → Drift Skill roots

不要创建第二套 workspace MCP/skill owner。固定 listener 的 managed service 使用通用隔离合同：

```python
ManagedServiceSpec(
    id="api",
    command=("python", "server.py"),
    readiness_url="http://127.0.0.1:18765/ready",
    validation_port_env="PLUGIN_API_PORT",
)
```

服务进程和同插件 MCP 必须真正读取 `PLUGIN_API_PORT`；Core 在候选验证时注入临时端口和隔离 plugin-data。bot token、long-poll 或 webhook Channel 不复制正式 ownership，只在父 turn 结束后的统一切换中 stop/start。

## 6. 状态与生命周期

- canonical source：插件 Git 仓库。
- installed code：全局插件 cache；不可直接编辑。
- workspace data：`<workspace>/plugin-data/<plugin>-<marketplace>/`；卸载代码默认保留。
- runtime catalog：snapshot 投影；turn 绑定后整轮冻结。

候选 prepare/readiness 不得写正式 session、memory、plugin-data 或外部服务。行为验证若需要写入，必须使用真实事务/dry-run、隔离目标或明确副作用授权；代码 pointer 回滚不拥有这些效果。

## 7. Source 验证

先确认当前 Gateway 的 Python 解释器。仓库存在 `.venv/bin/python` 时使用它，不要默认系统 `python` 已安装 runtime/test 依赖；不确定时从 Gateway 进程的 `/proc/<pid>/exe` 核对。source test 示例：

```bash
/absolute/repository/.venv/bin/python -m pytest -q /absolute/plugin-source/tests
```

按插件仓库自己的说明执行最小测试。随后从干净 Git HEAD 安装：

```bash
python main.py plugin-install \
  --source /absolute/path/to/plugin-repo \
  --marketplace local
```

远程 source 使用其真实 URL/marketplace；先 push 安装所需 commit。安装成功返回已经说明候选身份、当前 turn 与后续动作，不要再查询 status 或运行 doctor。只有安装失败且错误明确指向结构、声明或配置时，才把 `plugin-doctor` 作为诊断工具；它不能证明新增 Tool/Skill 已经被模型真实使用。最终行为验证使用父 turn 创建的 attached programmatic child，不显式选择 runtime。Core 不支持因果候选绑定时，只能在一次性隔离环境验证，并明确报告 `safe candidate self-validation unavailable`。
