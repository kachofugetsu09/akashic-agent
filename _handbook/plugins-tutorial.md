# 插件系统 Handbook

这份文档面向两类人：

- 学习者：想知道 Akashic 插件是什么、怎么用
- 贡献者：想做一个可以安装、可以发布的插件仓库

如果你只是想快速上手，先看前半部分。

如果你还想理解主仓 runtime 怎么把插件接进去，再看后半部分的附录。

---

## 1. 先建立直觉

把 Akashic 插件理解成一个独立 Git 仓库。

这个仓库可以给 Akashic 提供三类能力：

```text
plugin
├─ lifecycle
│  └─ 介入 agent 生命周期
├─ skills
│  └─ 给 agent 新说明书 / 工作流
└─ mcp
   └─ 给 agent 新工具
```

一句话：

```text
主仓负责运行时
插件仓库负责能力本身
```

也就是说：

- 你写插件仓库
- Akashic 负责安装、发现、加载、暴露

---

## 2. 学习顺序

不要一上来就看 runtime 源码。

推荐顺序：

```text
学习路径
├─ 1. 先分清 lifecycle / skills / mcp
├─ 2. 选一种最小插件类型
├─ 3. 抄一个最小模板
├─ 4. 本地安装
├─ 5. 重启主进程
├─ 6. 用 plugin-doctor 检查
└─ 7. 再去理解加载机制
```

---

## 3. 三种能力分别是什么

### 3.1 lifecycle

你想改的是“Akashic 自己怎么工作”，就写 lifecycle。

典型场景：

- 往 prompt 里补规则
- 在推理前后插逻辑
- 在回复后清洗输出
- 给 passive / proactive 链路加 gate

一句话：

```text
lifecycle = 改 agent 的行为
```

### 3.2 skills

你想给 agent 一份新说明书，就写 skills。

典型场景：

- GitHub 工作流
- feed 管理说明
- 某类网站的使用攻略
- 某类分析任务的固定步骤

一句话：

```text
skills = 教 agent 怎么做事
```

### 3.3 mcp

你想给 agent 新工具，就写 MCP。

典型场景：

- 接 Steam
- 接 RSS / feed
- 接外部 API
- 接浏览器、数据库、第三方平台

一句话：

```text
mcp = 给 agent 新工具
```

---

## 4. 我该写哪种插件

先别贪全。

按需求选最小类型：

```text
需求
├─ 只想加说明书         -> skills-only
├─ 只想改生命周期       -> lifecycle-only
├─ 只想加工具           -> mcp-only
└─ 三个都要             -> full plugin
```

推荐原则：

```text
先做最小能工作的那种
不要一开始就 full plugin
```

比如：

- 只是想把几个 workspace skill 搬出去 -> `skills-only`
- 只是想做一个输出清洗器 -> `lifecycle-only`
- 只是想接一个外部 API -> `mcp-only`
- 像 `feed-mcp` 这样要带 lifecycle + skill + mcp -> `full plugin`

---

## 5. 现在组织里有哪些插件

Akashic 现在走的是“一个仓库一个插件”的路线。

组织地址：

- <https://github.com/orgs/akashic-plugins/repositories>

目前公开仓库可以粗分成三类：

```text
akashic-plugins
├─ 运行类
│  ├─ observe
│  ├─ emotion
│  ├─ proactive_feedback
│  ├─ feishu
│  └─ qqbot
├─ lifecycle 类
│  ├─ citation
│  ├─ context_pressure
│  ├─ daynight_gate
│  ├─ setup_helper
│  ├─ shell_restore
│  ├─ shell_safety
│  ├─ status_commands
│  ├─ tool_loop_guard
│  ├─ plugin_undo
│  └─ meme
└─ 资源 / 工具包类
   ├─ feed-mcp
   ├─ steam-mcp
   └─ huayue-skills
```

这也是推荐给贡献者的结构：

```text
one repo
-> one plugin
-> one version stream
-> one install surface
```

---

## 6. 一个插件仓库最少长什么样

### 6.1 运行时识别条件

当前运行时把下面两种目录都当成插件根：

```text
插件根目录
├─ plugin.py
└─ 或 .aka-plugin/plugin.json
```

### 6.2 推荐给贡献者的最小结构

如果你要做一个可发布、可安装的独立插件仓库，推荐至少有：

```text
my-plugin
└─ .aka-plugin/plugin.json
```

如果它带 lifecycle，再加：

```text
my-plugin
├─ .aka-plugin/plugin.json
└─ plugin.py
```

完整推荐结构：

```text
my-plugin
├─ .aka-plugin/
│  └─ plugin.json
├─ plugin.py
├─ skills/
│  └─ <skill-name>/SKILL.md
├─ drift/
│  └─ skills/
│     └─ <drift-skill>/SKILL.md
├─ mcp/
│  ├─ servers.json
│  ├─ requirements.txt
│  └─ run_mcp.py
└─ README.md
```

能力和目录的关系：

```text
目录
├─ plugin.py      -> lifecycle
├─ skills/        -> 普通 skill
├─ drift/skills/  -> drift skill
└─ mcp/           -> MCP
```

---

## 7. 最关键的文件：`.aka-plugin/plugin.json`

当前系统真正认的是这个文件。

最常见字段：

```text
plugin.json
├─ name
├─ version
├─ description
├─ paths.skills
├─ paths.drift_skills
├─ paths.mcp_servers
└─ akashic.lifecycle.entry
```

最重要的理解：

```text
plugin.json 负责声明
runtime 负责解释声明
```

### 7.1 skills-only 模板

```json
{
  "name": "my-skills",
  "version": "0.1.0",
  "description": "A bundle of Akashic skills",
  "paths": {
    "skills": ["skills"]
  },
  "akashic": {
    "runtime": {
      "supports": ["skills"]
    },
    "skills": {
      "link_mode": "symlink"
    }
  }
}
```

### 7.2 lifecycle-only 模板

```json
{
  "name": "my-lifecycle",
  "version": "0.1.0",
  "description": "Lifecycle plugin",
  "akashic": {
    "runtime": {
      "supports": ["lifecycle"]
    },
    "lifecycle": {
      "entry": "plugin.py",
      "class": "MyPlugin",
      "restart_required": true
    }
  }
}
```

### 7.3 mcp-only 模板

```json
{
  "name": "my-mcp",
  "version": "0.1.0",
  "description": "MCP plugin",
  "paths": {
    "mcp_servers": ["mcp/servers.json"]
  },
  "akashic": {
    "runtime": {
      "supports": ["mcp"]
    }
  }
}
```

### 7.4 full plugin 模板

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "description": "Full Akashic plugin",
  "paths": {
    "skills": ["skills"],
    "drift_skills": ["drift/skills"],
    "mcp_servers": ["mcp/servers.json"]
  },
  "akashic": {
    "runtime": {
      "supports": ["lifecycle", "skills", "mcp"]
    },
    "lifecycle": {
      "entry": "plugin.py",
      "class": "MyPlugin",
      "restart_required": true
    },
    "skills": {
      "link_mode": "symlink"
    }
  }
}
```

---

## 8. 第一个插件怎么做

如果你是第一次写插件，建议这样开始：

```text
第一个插件
├─ 选 skills-only
├─ 只做一个 skill
├─ 写清楚 SKILL.md
├─ 本地安装
├─ 重启
└─ 看软链接有没有出现
```

这是学习成本最低的路线。

推荐目录：

```text
hello-skills
├─ .aka-plugin/plugin.json
└─ skills/
   └─ hello-world/
      └─ SKILL.md
```

安装：

```bash
python main.py plugin-install --source /path/to/hello-skills --marketplace local
```

重启主进程后检查：

```text
~/.akashic/workspace/skills/hello-world
```

如果这个软链接出现了，你就已经打通第一步了。

---

## 9. 本地开发循环

推荐开发循环非常朴素：

```text
本地开发循环
├─ 改插件仓库
├─ plugin-install 安装本地路径
├─ 重启主进程
├─ plugin-doctor 检查
└─ 做一次真实验证
```

对应命令：

```bash
python main.py plugin-install --source /path/to/my-plugin --marketplace local
python main.py plugin-doctor
python main.py plugin-doctor my-plugin@local
```

如果是 skills-only，验证点主要是：

- `workspace/skills` 下有没有软链接

如果是 lifecycle-only，验证点主要是：

- `registry.json` 里的 `active` 是否为 `true`

如果是 mcp-only，验证点主要是：

- `registry.json` 里有没有 `mcp_servers`
- 运行日志里有没有连上

---

## 10. 发布前检查清单

每个插件仓库都建议自己过这份清单：

```text
发布前检查
├─ 1. 仓库里有 .aka-plugin/plugin.json
├─ 2. name / version / description 合理
├─ 3. skills 目录下每个 skill 都有 SKILL.md
├─ 4. mcp 命令能在插件目录里独立跑
├─ 5. 不提交 sqlite / log / token / cache
├─ 6. plugin-install 能成功
├─ 7. 重启后能生效
└─ 8. plugin-doctor 不是 broken
```

如果只能记一句：

```text
能装
能重启加载
能真实工作
```

---

## 11. 贡献者最容易踩的坑

### 11.1 把运行态数据提交进仓库

不要提交这些：

- `.db`
- sqlite
- 日志
- 缓存
- token
- API key

运行态数据应该只放这里：

```text
~/.akashic-plugin/data/<plugin>-<marketplace>/
```

### 11.2 一上来就做 full plugin

很多时候你只需要：

- skill
- 或 MCP
- 或一个小 lifecycle

不要为了“以后可能要扩展”一开始就三件套全上。

### 11.3 把迁移逻辑写进主仓

插件自己的数据迁移，应该插件自己处理。

正确思路：

```text
公共 runtime
└─ 只提供稳定 data_dir

插件自己
└─ 在 initialize() 里迁移旧数据
```

### 11.4 skill 名字乱取

普通 skill 会以裸名暴露到 `workspace/skills`。

所以名字要：

- 稳定
- 可读
- 尽量不撞名

像下面这种就不错：

- `feed-manage`
- `rsshub-route-finder`
- `steam-inventory-analyzer`

---

## 12. 现在这套系统的能力边界

作为贡献者，最好先接受这些事实。

### 12.1 默认按“重启生效”理解

今天最稳的语义是：

```text
安装 / 改配置 / 升级
-> 重启主进程
-> 再验证
```

还没有可靠热重载：

```text
不支持可靠热重载
├─ lifecycle
├─ skills
└─ mcp
```

### 12.2 现在没有完整远端注册表协议

`akashic-plugins` 现在是组织，不是机器可消费的中心 index。

所以安装本质上还是：

```text
给一个 Git 仓库
-> clone
-> 本地物化
-> 启动时发现
```

### 12.3 现在没有正式的 uninstall CLI

今天下线插件主要还是：

- `enabled = false`
- 或移除安装目录

所以社区作者要把 README 写清楚，不要默认用户已经有图形界面或一键管理器。

---

## 13. README 至少该写什么

一个插件仓库的 README，建议至少写清楚：

```text
README 最少应包含
├─ 1. 这个插件提供什么能力
├─ 2. 它属于哪一类插件
├─ 3. 安装命令
├─ 4. 是否需要重启
├─ 5. 是否需要外部依赖 / API key
├─ 6. 数据放在哪
└─ 7. 怎么验证是否生效
```

不要只写“这是一个插件”。

让别人能装起来，才叫可贡献。

---

## 14. 附录：运行时如何运作

如果你只是来贡献插件，这一节可以先跳过。

这部分是给想理解主仓实现的人看的。

### 14.1 插件来源

运行时会同时看两类真实来源：

```text
插件来源
├─ builtin
│  └─ 主仓 plugins/ 下的内建插件
└─ installed
   └─ ~/.akashic-plugin/cache/<marketplace>/<plugin>/<version>/
```

外加一个“组织级入口”：

```text
GitHub 上的 akashic-plugins/*
```

但它现在还不是远端 registry 协议。

### 14.2 发现规则

发现逻辑：

```text
resolve_plugin_sources
├─ 扫 installed cache
└─ 扫主仓 plugins/
```

目录判定规则：

```text
一个目录满足任一条件就算插件根
├─ 有 plugin.py
└─ 有 .aka-plugin/plugin.json
```

还有一个迁移时很重要的行为：

```text
同名冲突
├─ installed 先被扫描
└─ builtin 同名项会被跳过
```

这就是外部插件能平滑覆盖旧 builtin 的原因。

### 14.3 plugin_id 规则

```text
builtin
└─ plugin_id = <name>

installed
└─ plugin_id = <name>@<marketplace>
```

例如：

- `citation`
- `default_memory`
- `feed@lab`
- `steam@github`

### 14.4 安装物化流程

`plugin-install` 做的是一次真实安装：

```text
plugin-install
├─ 临时 clone
├─ 读 plugin.json
├─ 复制到 cache/<marketplace>/<plugin>/<version>
├─ 确保 data/<plugin>-<marketplace> 存在
├─ 预处理 Python MCP 的 .venv
└─ 预写 registry.json
```

所以这套系统天然分成：

```text
~/.akashic-plugin
├─ cache
│  └─ 代码包
├─ data
│  └─ 私有运行态数据
└─ registry.json
```

更新时：

```text
update
├─ 替换 cache 里的版本
└─ 保留 data
```

### 14.5 启动加载顺序

现在的模式不是热重载，而是启动装配：

```text
进程启动
├─ discover
├─ 读 plugin policy
├─ 加载 lifecycle
├─ 收集 active plugins
├─ 同步 skill 软链接
├─ 汇总 MCP servers
├─ MCP 建连
└─ 写回 registry.json
```

### 14.6 lifecycle 如何接入

旧版 handbook 里讲 lifecycle，重点其实不是“有个 `plugin.py`”。

真正关键的是：

```text
lifecycle
├─ 可以插进被动回复主链路
├─ 可以插进 tool loop
├─ 可以插进 proactive 链路
├─ 可以注册自己的工具
└─ 现在还能和 skill / MCP 打包在同一个外部插件仓库里
```

也就是说，后来的扩展不是替代 lifecycle，而是把它变成了一个更完整的插件单元。

```text
full plugin
├─ plugin.py              -> lifecycle
├─ skills/                -> 普通 skill
├─ drift/skills/          -> drift skill
└─ mcp/servers.json       -> MCP
```

#### 14.6.1 生命周期有哪几层

当前一共可以从 4 层接入：

```text
lifecycle 接入层
├─ PhaseModule
├─ EventBus decorators
├─ @on_tool_pre
└─ @tool
```

它们不是互斥关系，同一个插件可以一起用。

#### 14.6.2 PhaseModule：最核心的生命周期能力

`Plugin` 基类支持这些 phase 挂点：

```text
Plugin methods
├─ before_turn_modules()
├─ before_reasoning_modules()
├─ prompt_render_modules()
├─ before_step_modules()
├─ after_step_modules()
├─ after_reasoning_modules()
├─ after_turn_modules()
├─ proactive_modules()
├─ jobs()
└─ channels()
```

其中前 7 个和 `proactive_modules()` 都是在“模块链”里插逻辑。

它的运行方式可以理解成：

```text
phase pipeline
├─ 内置模块
├─ 插件模块
├─ 每个模块声明 slot / requires / produces
└─ runtime 按依赖做拓扑排序
```

这也是旧版 handbook 里最重要的那部分能力。它的价值在于你不需要写死“早一点”或“晚一点”，而是精确声明自己依赖哪个阶段锚点。

```text
PromptRender
├─ prompt_render.emit
├─ citation.prompt
└─ meme.prompt
```

例如 `meme` 以前就是通过依赖 `citation.prompt`，保证自己跑在 citation 后面，而不是靠硬编码顺序。

所以一个 phase module 通常长这样：

```python
class MyPromptModule:
    slot = "my_plugin.prompt"
    requires = ("prompt_render.emit",)
    produces = ("prompt:section_bottom:my_plugin",)

    async def run(self, frame):
        frame.slots["prompt:section_bottom:my_plugin"] = "## 规则\n请用中文回答。"
        return frame


class MyPlugin(Plugin):
    name = "my_plugin"

    def prompt_render_modules(self):
        return [MyPromptModule()]
```

这类模块最适合做：

```text
PhaseModule 适用场景
├─ 注入 prompt section
├─ 早停 tool loop
├─ 回复后清洗输出
├─ 给持久化写附加字段
└─ 给 proactive 链路挂 gate / source / prompt / deliver 模块
```

#### 14.6.2.1 slot 是模块之间真正交换数据的总线

旧版 lifecycle 里最容易被低估的一点就是 `frame.slots`。

模块之间通常不是直接互调，而是靠 slot 传递上下文和产物：

```text
module A
└─ 写 frame.slots["prompt:section_bottom:foo"]
   └─ collect/export module
      └─ 识别前缀并汇总
         └─ 下游 phase ctx / 持久化 / 出站消息
```

这也是为什么插件之间可以低耦合协作。

常见前缀可以先记这几类：

```text
slot families
├─ session:*
├─ reasoning:*
├─ prompt:*
├─ step:*
├─ persist:user:*
├─ persist:assistant:*
├─ outbound:*
└─ turn:*
```

把它们粗暴理解成：

```text
slot 前缀语义
├─ session:*            -> 这一轮开始阶段
├─ reasoning:*          -> 推理前后共享数据
├─ prompt:*             -> prompt 注入
├─ step:*               -> tool loop 这一小步
├─ persist:*            -> 要写进消息持久化的数据
├─ outbound:*           -> 真正发出去的内容附加信息
└─ turn:*               -> turn 结束时的额外元数据
```

最常见的几个用法：

```text
高频 slot 用法
├─ prompt:section_bottom:*        -> 往 system prompt 末尾补规则
├─ step:early_stop_reason         -> 提前停止 tool loop
├─ persist:assistant:*            -> 给 assistant 消息写额外字段
└─ outbound:metadata:*            -> 给出站消息补 metadata
```

所以如果花月哥哥以后迁移 builtin lifecycle，最该保住的不是“类名长得一样”，而是：

```text
迁移 lifecycle 的关键
├─ slot key 语义不变
├─ requires 锚点不变
└─ phase 注入位置不退化
```

#### 14.6.3 EventBus decorators：在关键节点观察或改写

如果你的逻辑不需要自己写一个完整 module，可以直接用装饰器。

当前可用的生命周期装饰器有：

```text
GATE
├─ @on_before_turn
├─ @on_before_reasoning
├─ @on_prompt_render
├─ @on_before_step
└─ @on_after_reasoning

TAP
├─ @on_after_step
├─ @on_after_turn
├─ @on_tool_call
└─ @on_tool_result
```

经验上可以这样理解：

```text
GATE
└─ 能改 ctx，必要时能阻断流程

TAP
└─ 只观察，不改主流程
```

所以：

- 要补 prompt、改 reply、做 gate，用 GATE
- 要记日志、打点、审计工具结果，用 TAP

#### 14.6.4 `@on_tool_pre`：工具执行前拦截

这是 lifecycle 里后来很实用的一层。

它不走普通 EventBus，而是直接挂在工具执行前。

```text
LLM 想调工具
├─ 先过 @on_tool_pre
├─ 可以放行
├─ 可以改参数
└─ 可以 deny
```

典型插件：

- `shell_safety`
- `shell_restore`
- `tool_loop_guard`

这类能力很适合做：

```text
tool pre hook
├─ 禁危险命令
├─ 拦重复工具循环
├─ 自动改 shell 参数
└─ 做统一工具策略
```

#### 14.6.5 `@tool`：插件自己给 agent 注册工具

除了接外部 MCP，lifecycle 插件本身也可以直接注册本地工具。

```text
工具来源
├─ builtin tools
├─ plugin @tool
└─ plugin MCP servers
```

两者区别很简单：

```text
@tool
├─ 适合轻量本地逻辑
└─ 不需要额外 MCP 进程

MCP
├─ 适合外部服务 / 独立 server
└─ 更适合对外发布和复用
```

#### 14.6.6 `PluginContext`：现在给 lifecycle 的上下文比旧版更完整

现在插件实例拿到的上下文大致是：

```text
PluginContext
├─ plugin_id
├─ plugin_dir
├─ data_dir
├─ kv_store
├─ config
├─ workspace
├─ event_bus
├─ tool_registry
├─ session_manager
├─ memory_engine
└─ llm
```

这里有两个变化很重要：

第一，外部 `.aka-plugin` 插件现在也能稳定拿到 `data_dir`，所以插件自己的迁移逻辑应该写在插件内，而不是写进主仓。

第二，plugin 不只是“被动回复里的一个 patch”，它现在也能碰到：

```text
PluginContext 可支撑的扩展面
├─ workspace skill 软链接
├─ 独立 MCP server
├─ session / memory 协作
└─ 插件自己的持久化状态
```

#### 14.6.7 初始化、终止、job、channel

旧版大家最常用的是 phase module，但后面其实又补了几类能力：

```text
插件运行期能力
├─ initialize()
├─ terminate()
├─ jobs()
├─ channels()
└─ proactive_modules()
```

适合的职责大致是：

```text
职责划分
├─ initialize()
│  └─ 读配置、迁移旧数据、准备运行态
├─ terminate()
│  └─ 收尾
├─ jobs()
│  └─ 定时任务 / 后台任务
├─ channels()
│  └─ 增加渠道接入
└─ proactive_modules()
   └─ 给主动链路加模块
```

这里也顺手澄清一个现实语义：

```text
当前最稳的插件语义
├─ lifecycle 改动 -> 重启后加载
├─ skill 软链接 -> 随 active plugin 同步
└─ MCP server -> 随 active plugin 汇总并建连
```

也就是你前面提到的那种思路：

```text
推荐语义
├─ lifecycle 视为重启生效
└─ skill / mcp 作为同一插件声明的一部分一起装配
```

这样边界最清楚，也最不容易把热重载做坏。

### 14.7 skill 如何接入

skill 不是复制进主仓，而是做软链接：

```text
active plugins
├─ 读取 manifest 里的 skill roots
├─ 计算 link name
├─ 链接到 ~/.akashic/workspace/skills
└─ drift skill 链接到 ~/.akashic/workspace/drift/skills
```

普通 `.aka-plugin` skill 会用裸名。

drift skill 会带插件名前缀，避免撞名。

#### 14.7.1 软链接规则是后面新增的重要能力

现在 skill 不是复制进主仓，而是统一由运行时维护软链接。

```text
skill link rules
├─ 普通 .aka-plugin skills
│  └─ ~/.akashic/workspace/skills/<skill-name>
├─ drift skills
│  └─ ~/.akashic/workspace/drift/skills/<plugin>:<skill-name>
└─ 非 .aka-plugin 的旧 builtin
   └─ 仍可能带 plugin_id 前缀
```

这个设计的价值是：

```text
skill symlink 的收益
├─ skill 仍然以 workspace 视角暴露
├─ 插件仓库可以独立发布
├─ 主仓不需要复制内容
└─ 迁移 builtin -> 外部插件时更平滑
```

### 14.8 MCP 如何接入

MCP 也是声明式：

```text
plugin.json
└─ paths.mcp_servers -> servers.json
```

descriptor 读出后，运行时做两步：

```text
安装阶段
├─ 注入 AKA_PLUGIN_DATA_DIR
└─ 准备 Python MCP 的 .venv

启动阶段
├─ 汇总 active plugin 的 mcp_servers
└─ 交给 MCP registry 建连
```

#### 14.8.1 MCP 现在是插件声明的一等能力

这一点和旧版 lifecycle 时代很不一样。

现在一个插件仓库可以天然声明：

```text
one plugin repo
├─ lifecycle
├─ skills
└─ mcp
```

也就是说，像 `feed-mcp`、`steam-mcp` 这种插件，不再只是“仓库外再手工注册一个 MCP”，而是插件自己把 MCP 能力声明出来，运行时统一收口。

这也是为什么现在 handbook 必须把 lifecycle、skill、MCP 放在同一张图里讲，而不能分成三套互不相干的说明。

### 14.9 registry 是什么

`~/.akashic-plugin/registry.json` 是本机插件总表。

它不是源码，也不是远端索引。

它描述的是：

```text
当前这台机器
├─ 有哪些插件
├─ 插件是否 enabled
├─ 插件是否 active
├─ 暴露了哪些 skill
└─ 暴露了哪些 MCP server
```

最值得看的字段：

- `plugin_id`
- `enabled`
- `local_disabled`
- `active`
- `plugin_root`
- `data_dir`
- `capabilities`
- `skills`
- `drift_skills`
- `mcp_servers`

判断状态时建议永远按这个顺序：

```text
状态判断
├─ enabled=false
├─ local_disabled=true
├─ active=false
└─ capabilities.*
```

### 14.10 配置怎么控制插件

主开关在 `config.toml`：

```toml
[plugins.daynight_gate]
enabled = true
timezone = "Asia/Shanghai"
start = "00:00"
end = "06:00"
pass_probability = 0.15
reason = "quiet_hours"
```

通常可以理解成：

```text
[plugins.<plugin_id>]
├─ enabled = true/false
├─ 插件自己的业务配置
├─ capabilities.lifecycle.enabled
├─ capabilities.skills.enabled
├─ capabilities.mcp.enabled
└─ mcp_servers.<server>.enabled
```

实践上最好统一写完整 `plugin_id`。
