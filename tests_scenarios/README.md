# tests_scenarios

这不是普通单元测试目录。

这里的测试是为了：

- 模拟真实用户消息进入 `AgentLoop`
- 模拟真实的 history / memory / memory2 场景
- 调用真实模型
- 注册真实工具
- 走真实 embedding / retrieve / injection 链路
- 观察 agent 在“接近真实环境”的情况下是否行为正常

一句话说，这套测试是为了模拟真实场景，不是为了测几个 isolated function。

## 1. 设计原则

### 1.1 尽量真实

这里默认追求：

- 真实 `config.json`
- 真实 `AgentLoop`
- 真实工具注册逻辑
- 真实工具执行逻辑
- 真实 embedding
- 真实向量召回
- 真实 memory injection

### 1.2 绝对隔离

虽然链路尽量真实，但数据必须是测试专用的。

每个场景都会创建自己的独立 workspace，不得污染：

- 真实 session
- 真实 memory 文件
- 真实 SQLite
- 真实工具上下文

### 1.3 可观测

场景测试失败后，应该优先看 artifact，而不是先猜。

当前 artifact 目录：

```text
.pytest_artifacts/agent-loop-mvp/<scenario_id>/latest/
```

当前 workspace 目录：

```text
.pytest_artifacts/agent-loop-mvp/<scenario_id>/latest/workspace/
```

## 2. 目录说明

- [fixtures.py](/mnt/data/coding/akasic-agent/tests_scenarios/fixtures.py)
  - 定义场景数据结构
  - 定义样例场景

- [scenario_runner.py](/mnt/data/coding/akasic-agent/tests_scenarios/scenario_runner.py)
  - 创建隔离 runtime
  - 写入测试专用 history / memory / memory2
  - 执行真实 `AgentLoop`
  - 落盘 artifact

- [judge_runner.py](/mnt/data/coding/akasic-agent/tests_scenarios/judge_runner.py)
  - 用 light model 做语义 judge
  - 只负责语义判断

- [test_agent_loop_scenarios.py](/mnt/data/coding/akasic-agent/tests_scenarios/test_agent_loop_scenarios.py)
  - pytest 用例入口
  - 每个测试方法都应写清楚：在测什么、在做什么

## 3. 当前测试套件的边界

### 3.1 这套测试负责什么

- 主循环在真实链路下是否行为正常
- route gate 是否稳定
- retrieve 是否命中
- 工具是否被正确调用
- 最终回答是否基本符合预期

### 3.2 这套测试不负责什么

- 不负责验证某个纯函数细节
- 不负责替代单元测试
- 不负责大规模 fuzz
- 不负责所有外部系统联调

如果一个问题只需要 fake provider + fake tool + 硬断言就能测清楚，优先写到普通 `tests/` 单元测试里。

如果一个问题必须依赖：

- 真实模型
- 真实工具注册
- 真实 embedding / retrieval
- 真实上下文拼装

才应该放到这里。

## 4. 如何运行

默认不会自动执行，避免误调用真实模型。

运行命令：

```bash
AKASIC_RUN_SCENARIOS=1 python -m pytest -c pytest-scenarios.ini tests_scenarios -q
```

只跑某一条：

```bash
AKASIC_RUN_SCENARIOS=1 python -m pytest -c pytest-scenarios.ini tests_scenarios/test_agent_loop_scenarios.py -k test_real_smalltalk_does_not_trigger_retrieve -q
```

## 5. 如何新增一个场景

建议流程：

1. 先明确这条测试是不是“真实场景模拟”
   - 如果只是纯逻辑，放回普通单元测试

2. 在 [fixtures.py](/mnt/data/coding/akasic-agent/tests_scenarios/fixtures.py) 里补一个 `ScenarioSpec`
   - 写清消息
   - 写清 history
   - 写清 memory
   - 写清 memory2 item
   - 写清硬断言
   - 必要时再加 judge

3. 在 [test_agent_loop_scenarios.py](/mnt/data/coding/akasic-agent/tests_scenarios/test_agent_loop_scenarios.py) 里补一个测试方法
   - 必须写步骤注释
   - 必须说明在测什么

4. 运行单条场景
   - 看是否通过
   - 如果失败，优先看 artifact

## 6. 写场景时的要求

### 6.1 先写硬断言，再考虑 judge

硬断言优先检查：

- `route_decision`
- `history_hits`
- `tools_used`
- `tool_calls`
- `final_content` 关键字

judge 只负责这些更难硬断言的内容：

- 回答是否利用了召回内容
- 回答是否完成任务
- 回答是否承认未知

judge 不能替代结构断言。

### 6.2 不要用真实用户数据

测试场景里的：

- history
- memory
- memory2 items

都必须是测试专用数据，不要把真实用户记录直接搬进来。

### 6.3 失败时先看 artifact

优先排查顺序：

1. `summary.json`
2. `memory_trace.json`
3. `llm_calls.jsonl`
4. `tool_calls.jsonl`
5. `session_before.json`
6. `session_after.json`

## 7. 当前已知问题

本套件已经帮助定位到一个真实问题：

- `memory_v2.gate_llm_timeout_ms = 800` 时
- route gate 在真实环境下容易超时
- 超时后 fail-open 成 `RETRIEVE`

当前真实配置已调整到 `1600`。

这类问题正是这套测试存在的意义：

- 它不是为了证明代码“理论上没问题”
- 而是为了在接近真实环境的条件下，把真实行为问题尽早暴露出来
