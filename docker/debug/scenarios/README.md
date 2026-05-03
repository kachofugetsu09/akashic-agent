# Context Probe Scenarios

这个目录存放可公开的 `docker/debug/context_probe.py` 输入场景。

```
scenario json
  |
  +-- name
  |
  +-- turns
  |     |
  |     +-- user turn
  |     +-- consolidate action
  |     +-- final user turn
  |
  +-- result
        |
        +-- markdown report
        +-- json report
```

运行示例：

```bash
python docker/debug/context_probe.py \
  --profile v4flash-memory-window \
  --messages docker/debug/scenarios/sleepy_study_plan.json \
  --reset-workspace \
  --start-agent \
  --stop-agent \
  --quiet-agent \
  --disable-qq
```

场景 JSON 是公开测试输入，可以提交。`docker/debug/profiles/<profile>/workspace/` 下生成的报告 JSON / Markdown 是运行产物，默认不提交。

场景只描述输入和流程，不写结果要求。运行后观察：

- 最终回答是否自然带入前文状态
- `RECENT_CONTEXT.md` 写了什么
- `memory2.db` 写了什么
- 是否调用了 `recall_memory`
- 普通 history 还剩多少

样本来源策略：

```
real session records
  |
  +-- extract conversation rhythm and task shape
  |
  +-- remove private names, concrete projects, account data, paths, commands
  |
  +-- rewrite as synthetic pure chat
  |
  +-- publish scenario json
```

公开场景只保留“话题切换、隐性状态、最终中性问题”的结构，不保留真实用户原文。
