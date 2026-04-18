# PersonaMem Benchmark

这版接法复用了 `eval/longmemeval` 的主链：

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

## 运行

```bash
python -m eval.personamem.run \
  --config eval/longmemeval/config.toml \
  --questions /path/to/questions_32k.csv \
  --contexts /path/to/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_bench \
  --workers 4 \
  --resume-auto
```

只跑某一类：

```bash
python -m eval.personamem.run \
  --config eval/longmemeval/config.toml \
  --questions /path/to/questions_32k.csv \
  --contexts /path/to/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_recall \
  --type recall_user_shared_facts \
  --workers 2
```
