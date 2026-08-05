# Akashic 插件编写参考

## 目录

1. 最小插件
2. Tool
3. Skill
4. MCP 与其他贡献
5. 状态与生命周期
6. Source 验证

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

## 3. Skill

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
3. runtime 支持候选选择时，在 latest programmatic child 给出真实触发请求；否则只在隔离 runtime 中走 current-snapshot child，并报告候选隔离不可用。
4. 从 tool items、产物或领域状态证明 Skill 被正确执行。

## 4. MCP 与其他贡献

只通过 Plugin API 声明：

- `mcp_servers()` → `McpServerSpec`
- `managed_services()` → `ManagedServiceSpec`
- `proactive_sources()` → `ProactiveSourceSpec`
- `channels()` → Channel
- `jobs()` → `PluginJobSpec`
- `drift_skill_roots()` → Drift Skill roots

不要创建第二套 workspace MCP/skill owner。固定端口、bot token、long-poll 或 singleton daemon 属于独占 endpoint，不能与 stable generation 同时在一个 runtime 启动时，必须使用隔离验证路径。

## 5. 状态与生命周期

- canonical source：插件 Git 仓库。
- installed code：全局插件 cache；不可直接编辑。
- workspace data：`<workspace>/plugin-data/<plugin>-<marketplace>/`；卸载代码默认保留。
- runtime catalog：snapshot 投影；turn 绑定后整轮冻结。

候选 prepare/readiness 不得写正式 session、memory、plugin-data 或外部服务。行为验证若需要写入，必须使用真实事务/dry-run、隔离目标或明确副作用授权；代码 pointer 回滚不拥有这些效果。

## 6. Source 验证

按插件仓库自己的说明执行最小测试。随后从干净 Git HEAD 安装：

```bash
python main.py plugin-install \
  --source /absolute/path/to/plugin-repo \
  --marketplace local
```

远程 source 使用其真实 URL/marketplace；先 push 安装所需 commit。安装后运行：

```bash
python main.py plugin-doctor <plugin>@<marketplace>
```

doctor 只证明结构、声明与当前诊断，不证明新增 Tool/Skill 已经被模型真实使用。最终行为验证优先走 latest programmatic child；runtime 尚不支持 selector 时，只能在隔离环境走 current-snapshot child，并明确报告 `safe candidate self-validation unavailable`。
