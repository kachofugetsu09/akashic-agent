[![欢迎加入交流群](https://img.shields.io/badge/QQ%E4%BA%A4%E6%B5%81%E7%BE%A4-%E6%AC%A2%E8%BF%8E%E5%8A%A0%E5%85%A5-2ea44f?style=for-the-badge)](./COMMUNICATION.md)

# akashic Agent

一个**会主动找你**的 AI 伙伴——不只是被动回答问题，还能根据你订阅的信息源主动判断"现在该不该发消息、发什么"，在空闲时自主执行后台任务。

**[Proactive 主动推送](./_handbook/proactive-guide.md)** — **[Drift 空闲任务](./_handbook/drift-guide.md)**

---

## Quickstart

需要 Python 3.12。

```bash
git clone <this-repo>
cd akashic-agent
uv venv && uv pip install -r requirements.txt
```

没有 uv？先 `pip install uv`。

### 1. 初始化

```bash
uv run python main.py setup    # 交互向导（推荐）
# 或
uv run python main.py init     # 非交互，CI/自动化用
```

### 2. 填写 config.toml 最少配置

```toml
[llm.main]
model = "deepseek-v4-flash"
api_key = "sk-..."

[channels.telegram]
token = "123456:ABC..."
allow_from = ["your_username"]
```

见 `config.example.toml` 完整模板。

### 3. 启动

```bash
uv run python main.py
```

给 bot 发一条消息即可开始对话。

---

## 核心功能

### 被动回复链路

收到消息 → 记忆检索 → 工具调用 → 流式回复。每轮经过 6 个生命周期 Phase（BeforeTurn → BeforeReasoning → Reasoner → AfterReasoning → AfterTurn），插件可在各 Phase 通过 EventBus 或模块插入点介入。

### Proactive 主动推送

agent 定期检查你订阅的信息源，自主决定是否推送。

```
proactive loop
  ├── 自适应轮询（你的活跃度决定频率）
  ├── 三路数据预取（alert / content / context）
  ├── LLM 逐条评分分类（interesting / not_interesting）
  └── 推送决策（去重 + 打扰检查 → 发送或跳过）
```

三类数据通道：
- **alert**：高优先级告警（心率异常、日历提醒），直接透传不评分
- **content**：内容流（RSS 新闻、社交更新），逐条评分分类
- **context**：背景上下文（睡眠状态、游戏在线），概率注入

详见 [proactive-guide.md](./_handbook/proactive-guide.md)。

### Drift 空闲任务

没有可推送内容时，agent 不空转——它会执行你定义的 drift skill：

- **audit-dirty-memories** — 随机抽检长期记忆，回溯原文验证准确性
- **explore-curiosity** — 补足用户画像空白，随口一问
- **review-drift-gaps** — Drift 自我反思，跟踪停滞方向

详见 [drift-guide.md](./_handbook/drift-guide.md)。

---

## 其他命令

```bash
uv run python main.py cli       # 连接运行中的 agent（TUI）
uv run python main.py dashboard # 打开 Dashboard（默认 :2236）
uv run python main.py --help    # 查看全部子命令

pytest tests/                   # 单元测试
akashic_RUN_SCENARIOS=1 pytest -c pytest-scenarios.ini tests_scenarios/  # 场景测试
```

---

## 配置图像能力

| 路线 | 配置 | 说明 |
|------|------|------|
| A: 主模型多模态 | `multimodal = true`，`vl.model = ""` | 图片直接进主模型 |
| B: 主模型 + VL 工具 | `multimodal = false`，`vl.model = "qwen-vl-plus"` | 推荐方案（DeepSeek + Qwen VL） |
| C: 纯文本 | `multimodal = false`，`vl.model = ""` | 图片不可理解 |

---

## 工作区

所有运行时数据在 `~/.akashic/workspace/`。
