# LongMemEval Benchmark

这个目录保留一版可提交的 LongMemEval 子集 benchmark。

当前 benchmark 只测三类题：

```text
┌────────────────────────────┐
│ single-session-user        │
├────────────────────────────┤
│ single-session-preference  │
├────────────────────────────┤
│ knowledge-update           │
└────────────────────────────┘
```

对应数据文件：

- `eval/longmemeval/data/longmemeval_akashic.json`

## 它在测什么

这不是纯 retrieval benchmark。
它测的是 `akashic-agent` 这套记忆系统在真实 AgentLoop 里的端到端效果：

```text
┌────────────────────┐
│ haystack 回放        │
├────────────────────┤
│ consolidation 写记忆 │
├────────────────────┤
│ recall/search/fetch │
├────────────────────┤
│ 最终 QA + judge      │
└────────────────────┘
```

所以它更接近：

```text
┌────────────────────┐
│ memory-enabled agent│
│ benchmark           │
└────────────────────┘
```

不是：

```text
┌────────────────────┐
│ 纯 memory engine     │
│ benchmark           │
└────────────────────┘
```

## 适用边界

这版 benchmark 适合回答：

- 单 session 用户事实记忆是否可用
- 单 session 偏好记忆是否可用
- knowledge update 场景里，系统最终能否选对更新后的答案
- 改 consolidation / recall / prompt 后，端到端有没有回归

它不适合直接回答：

- 纯 retrieval engine 排名
- multi-session 能力
- 时间推理的完整上限
- “是否已经把旧记忆真正 supersede 掉”

## 运行方式

全量自动续跑：

```bash
python -m eval.longmemeval.run \
  --config eval/longmemeval/config.toml \
  --data eval/longmemeval/data/longmemeval_akashic.json \
  --workspace /tmp/lme_bench \
  --workers 4 \
  --resume-auto
```

只跑某一类：

```bash
python -m eval.longmemeval.run \
  --config eval/longmemeval/config.toml \
  --data eval/longmemeval/data/longmemeval_akashic.json \
  --workspace /tmp/lme_bench_user \
  --type single-session-user \
  --workers 4 \
  --resume-auto
```

只跑 smoke：

```bash
python -m eval.longmemeval.run \
  --config eval/longmemeval/config.toml \
  --data eval/longmemeval/data/longmemeval_akashic.json \
  --workspace /tmp/lme_bench_smoke \
  --limit 3 \
  --workers 1 \
  --resume-auto
```

单题只重跑 QA：

```bash
python -m eval.longmemeval.run_one_qa \
  --config eval/longmemeval/config.toml \
  --data eval/longmemeval/data/longmemeval_akashic.json \
  --workspace /tmp/lme_one_case \
  --question-id 94f70d80
```

## `resume-auto`

推荐始终使用 `--resume-auto`。

语义如下：

```text
┌─────────────────────────────┐
│ workspace/<qid>/result.json │
└────────────┬────────────────┘
             │ yes
             v
      ┌──────────────┐
      │ 直接复用结果  │
      └──────────────┘
             │ no
             v
┌─────────────────────────────┐
│ ingest_state.json.completed │
└────────────┬────────────────┘
             │ yes
             v
      ┌──────────────┐
      │ 只跑 QA+judge │
      └──────────────┘
             │ no
             v
      ┌──────────────┐
      │ ingest+QA+judge │
      └──────────────┘
```

每题会落盘：

- `workspace/<question_id>/result.json`
- `workspace/<question_id>/trace.log`
- `workspace/<question_id>/ingest_state.json`

## 评分解释

主看：

- `judge`

辅看：

- `F1`
- `EM`

建议按下面顺序解读：

```text
┌──────────────┐
│ 先看 judge    │
├──────────────┤
│ 再看 F1 / EM  │
└──────────────┘
```

原因很简单：

- `judge` 更接近“最终回答在语义上对没对”
- `F1 / EM` 容易受句式影响

例如：

- `Four bikes.` vs `4`
- `Above your bed in the bedroom.` vs `in my bedroom`

这种题常见 `judge=✅` 但 `F1` 不高。

## 当前实现里的关键点

### 1. judge 固定走主模型

现在 judge 使用 `llm.main`，不再走 `llm.fast`。

### 2. benchmark persona 是硬约束

`SELF.md` 会在 benchmark runtime 里覆盖为专用 prompt，要求：

- 英文短答
- 所有题答案都假定存在于记忆里
- 必须先 `recall_memory`
- 必要时继续 `search_messages`
- 具体事实题必须 `fetch_messages`
- 允许中英混合 query

### 3. ingest 收尾会分块 finalize

末端未归档 tail 不再丢失。
现在会把最后未归档消息按 chunk 分块 consolidate，避免长尾事实漏进 benchmark。

### 4. 每次 consolidate 后会跑一次 post-response invalidation

现在 benchmark ingest 在每个 session consolidate 后，都会额外跑一次 `post_response_worker`。

但要注意：

```text
┌────────────────────┐
│ 它只更偏向处理        │
│ 显式 invalidation    │
├────────────────────┤
│ 不是完整的            │
│ “自然知识更新替换器”   │
└────────────────────┘
```

## 当前观察到的结论

### single-session-user

这类题现在已经接近天花板。
主要意义更偏回归基线，而不是继续深挖。

### single-session-preference

能测到：

- 是否召回偏好相关证据
- 是否把具体事件抽象成偏好

这类题比 `single-session-user` 更容易暴露“召回到了但不会用”的问题。

### knowledge-update

这是当前最值得看的部分。

现在测出来的真实情况更像：

```text
┌────────────────────┐
│ 新旧记忆经常同时存在   │
├────────────────────┤
│ agent 通过 recall +   │
│ search + fetch 选对新值│
├────────────────────┤
│ 但旧值未必真的被退休   │
└────────────────────┘
```

所以目前 `knowledge-update` 的成功更常代表：

- 检索和取证够强
- agent 能在冲突里选对新值

不自动代表：

- supersede 机制已经稳定生效

## 当前 stance

这版 benchmark 的目标不是把 supersede 调得很激进。

当前更偏保守策略：

```text
┌────────────────────┐
│ 保留旧值             │
├────────────────────┤
│ 让新值更容易被找出来  │
├────────────────────┤
│ 由 agent 最终选新版   │
└────────────────────┘
```

这个策略对产品可接受，但要明确：

- 它保证的是最终答案正确率
- 不是记忆库内部的一致性最优

## 看 trace 的方式

单题 trace 在：

- `/tmp/.../<question_id>/trace.log`

看 trace 时建议按这个顺序：

```text
┌────────────────────┐
│ recall_memory       │
├────────────────────┤
│ fetch_messages      │
├────────────────────┤
│ search_messages     │
├────────────────────┤
│ 最终回答             │
└────────────────────┘
```

如果答对了，要再问一句：

```text
┌────────────────────┐
│ 是 recall 直接命中新值 │
├────────────────────┤
│ 还是 search/fetch    │
│ 把新旧都拉出来后选对   │
└────────────────────┘
```

这两种成功含义不一样。

## 保留的脚本

目录里只保留 benchmark 主路径需要的脚本：

- `run.py`
- `run_one_qa.py`
- `ingest.py`
- `runtime.py`
- `qa_runner.py`
- `metrics.py`
- `dataset.py`

不再保留一次性的 case runner。
