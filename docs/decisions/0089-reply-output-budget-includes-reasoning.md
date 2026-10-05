# 0089：回复输出预算包含推理，截断不作为完成

- 状态：accepted
- 日期：2026-10-05
- 范围：Reply、scheduler、subagent、Reply Program、ReAct
- 依据：维护者授权调查 hua-home 凌晨失败、参考三个 harness、自主设计并交付 PR。
- 关联：ERR-001、RUN-005、SES-001、STA-002；沿用 [0063](0063-execution-failures-have-terminal-results.md) 的失败及副作用边界。

## 证据与问题

2026-10-05 北京时间 02:55:27，hua-home 的对话追加了“空响应不是 quiet”失败。
只读查询定位到原模型调用 `58f31ebe81ad45f6ac88bdd548b82c1d`：

- 模型为 `deepseek/deepseek-v4.1-flash`，使用 OpenAI-compatible driver。
- 冻结请求的 `max_output_tokens=4096`；回执 `finish_reason=length`。
- usage 为 output 4096、reasoning 4096；正文为空、工具调用为空。
- 当次固定的模型能力中输出上限未知；上下文窗口为 1,000,000，输入 usage 为 149,618。

这是推理耗尽请求输出预算的直接证据。回执已保存，但 ReAct 未检查结束原因，
把它归入空响应；同一条路径还可能把截断正文当完成，或执行参数碰巧可解析的截断工具。
不修改原会话的业务目标或重跑原工具。当前容器未取到事故时段日志，本次以持久消息、
冻结请求和模型回执为证，不推断中间代理的内部行为。

## 参考与取舍

本地源码版本及主要入口：

| 项目 | 观察 | 本次采用 |
|---|---|---|
| deepseek-harness `477b4f4205`，`packages/llm/llm-pi-ai/src/stream.ts` | `stop` 无内容是 EMPTY_RESPONSE，`length` 独立映射 max-tokens；retry 插件可重试空响应 | 分清截断与空响应；不照搬其重试预算和传输重试策略 |
| pi-mono `4a6ed0194`，`packages/ai/src/api/simple-options.ts`、`packages/agent/src/agent-loop.ts` | 缺省预算跟随模型；推理与答案共用额度；length 工具不执行 | 用当前模型能力选择预算，拦截截断工具 |
| Codex `b741e480e2`，`codex-rs/codex-api/src/sse/responses.rs` | 显式区分 completed、incomplete；非 interrupted 的 incomplete 保留原因并报错 | 保留未完成原因，不伪装成 quiet 或正常结束 |

[DeepSeek 官方 Chat Completions 合同](https://api-docs.deepseek.com/api/create-chat-completion/)
也将 `length` 解释为输出或上下文限制导致的截断。本次中间代理是否完全遵循最新官方默认值未知，
因此不依赖省略 wire 参数时的服务端默认值，也不对模型名写特殊规则。

## 决定

```text
┌─────────────────────────────────────┐
│ 来源预算：显式正整数 / 缺省 None      │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Reply Program 取得固定模型           │
│ 缺省 → 已知输出上限；未知 → 32768     │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ 同一预算用于 Context 预留和冻结请求  │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ Models 保存真实响应和 usage          │
└─────────────────┬───────────────────┘
                  ├── length → 明确失败，原工具结果保留
                  └── 其他 → 原解码、提交和工具流程
```

聊天、scheduler 和 subagent 配置省略 `max_output_tokens` 时都传 None；策略只在
Reply Program 拥有。显式预算继续原样使用，已有调用者的整数和 0 语义不变。
32768 是能力未知时的有限默认额度，不声称所有模型都支持；能力小于它时应登记真实上限，
也可显式设置更小预算。Context 仍按同一预算检查容量，不暗中挤掉输出额度或修改持久消息。

ReAct 在内容和工具解码前检查 length，抛出不可自动重试的 `OutputLengthError`。
因此截断正文不提交成完成，截断批次的工具不执行；先前已完成的工具不回滚或重跑。
真空响应仍报原 `EmptyResponseError`。不通过加“继续”消息、关闭推理、放宽 quiet、
清空历史或新增自动重试层掩盖问题。新预算无法保证永不截断；再次截断时明确报原因。

代价是缺省可用输出增加，可能提高一次请求的费用、延迟和上下文预留量；并非每次
都会用满额度。显式预算可控制成本。若之后要增加自动续接，须先解决原回执重放、
截断工具、推理续接协议和跨重启预算，不能只围绕此次模型增加循环。

## 状态、恢复与验证

没有 schema 迁移或正式配置改写。Models 正常增加调用账，started 由原 owner 结算并保留
完整响应和实际 usage；这里的 success 仍表示 provider 调用返回，不代表业务回复完成。
来源按既有协议追加 failure Control，旧 Message 不更新、失效或减少。冻结请求仍由原
准备记录复用，旧 4096 请求不因升级改写；调整预算后以新输入开启新准备。
取消、CAS、工具回执和原有恢复策略不变，没有新删除路径。

源码恢复点为修改前 main `b13be081`。正式数据只读，交付仅 PR，不含合并和部署。
可复用验收入口为 `docker/debug/reply_output_budget.py`：真实 PluginManager、Channel Input、
Reply Program、Models、HTTP/SSE driver、Tools 和 SQLite，只有上游服务使用合成协议响应。
覆盖未知能力、已知 8K/64K、显式 4K/16K、纯推理耗尽、正文/工具截断、工具后截断和真空响应；
核对 wire 预算、失败无自动重发、合法工具只执行一次、原输入与 usage 保留、关库重读。
未宣称真实付费 provider 或生产版本已经验收。
