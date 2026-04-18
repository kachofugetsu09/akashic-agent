# PersonaMem 隐式检索案例

这三题更适合观察“问题没有直接说出记忆点，但模型需要从长期偏好里反推”的场景。

下面每条命令都会：

- 先 ingest 这一题的历史
- 再跑 QA
- 最后把完整 trace 写到对应 `workspace/trace.log`
- 运行时会实时打印 ingest / QA 进度

## Case 1

`question_id`: `f546a74f-54de-40d0-9d88-8b0e30467d7b`

`question_type`: `provide_preference_aligned_recommendations`

```bash
python -m eval.personamem.run_one_case \
  --config /mnt/data/coding/akasic-agent/eval/longmemeval/config.toml \
  --questions /mnt/data/coding/akasic-agent/eval/personamem/data/questions_32k.csv \
  --contexts /mnt/data/coding/akasic-agent/eval/personamem/data/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_cases/f546a74f-54de-40d0-9d88-8b0e30467d7b \
  --question-id f546a74f-54de-40d0-9d88-8b0e30467d7b
```

## Case 2

`question_id`: `b3588797-acdf-40d3-bcc5-951f81896f95`

`question_type`: `suggest_new_ideas`

```bash
python -m eval.personamem.run_one_case \
  --config /mnt/data/coding/akasic-agent/eval/longmemeval/config.toml \
  --questions /mnt/data/coding/akasic-agent/eval/personamem/data/questions_32k.csv \
  --contexts /mnt/data/coding/akasic-agent/eval/personamem/data/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_cases/b3588797-acdf-40d3-bcc5-951f81896f95 \
  --question-id b3588797-acdf-40d3-bcc5-951f81896f95
```

## Case 3

`question_id`: `d06f511a-0fd8-4ee0-8c35-fd3a87fd35ec`

`question_type`: `suggest_new_ideas`

```bash
python -m eval.personamem.run_one_case \
  --config /mnt/data/coding/akasic-agent/eval/longmemeval/config.toml \
  --questions /mnt/data/coding/akasic-agent/eval/personamem/data/questions_32k.csv \
  --contexts /mnt/data/coding/akasic-agent/eval/personamem/data/shared_contexts_32k.jsonl \
  --workspace /tmp/personamem_cases/d06f511a-0fd8-4ee0-8c35-fd3a87fd35ec \
  --question-id d06f511a-0fd8-4ee0-8c35-fd3a87fd35ec
```

## 看什么

- 终端里的 `predicted / pick / gold`
- `workspace` 目录下的 `trace.log`
- 是否显式调用了 `recall_memory / search_messages / fetch_messages`
- `recall_memory` 的 query 是否是从隐式需求里反推出的
