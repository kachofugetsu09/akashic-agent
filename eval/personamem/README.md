# PersonaMem Benchmark

PersonaMem 的数据、导入、运行时和评分代码位于 `eval/personamem`：

```text
┌────────────────────┐
│ PersonaMem 数据适配 │
├────────────────────┤
│ ingest 回放         │
├────────────────────┤
│ consolidation       │
├────────────────────┤
│ QA 真实 AgentLoop   │
├────────────────────┤
│ 选项解析 / accuracy │
└────────────────────┘
```

当前实现是 MVP：

- 直接读取 `questions_*.csv`
- 直接读取 `shared_contexts_*.jsonl`
- 每个 benchmark 样本独立 workspace
- 共享向量记忆只在该样本内部生效，不会串题
- 回答格式固定为选项标签，如 `(a)`

现有 ingest/QA 入口仍调用旧 `CoreRuntime.session_manager` 和 `CoreRuntime.loop`；当前
`CoreRuntime` 已不提供这些属性。本次只解除对退役 LongMemEval 包的代码依赖，尚未
完成 PersonaMem 的运行链迁移，下方命令不代表已通过端到端验收。

## 运行

```bash
python -m eval.personamem.run \
  --config eval/personamem/config.toml \
  --questions /path/to/questions_32k.csv \
  --contexts /path/to/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_bench \
  --workers 4 \
  --resume-auto
```

只跑某一类：

```bash
python -m eval.personamem.run \
  --config eval/personamem/config.toml \
  --questions /path/to/questions_32k.csv \
  --contexts /path/to/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_recall \
  --type recall_user_shared_facts \
  --workers 2
```
