# ReAct 请求间隔测量

为两个已启动的 harness 使用同一模型、上游、任务输入、源码提交、工具权限和模型输出预算。
Akashic 使用新的 Project、独立 workspace/HOME/plugin home；关闭 Akasha、Markdown 记忆及外部插件。
Pi 使用独立 agent directory，关闭扩展、MCP、技能和额外上下文；用常驻 RPC，收到 `get_state`
成功响应后才提交输入，避免把 CLI 冷启动算成用户输入延迟。

## 边界与指标

```text
输入提交 → 持久接纳 → Reply/目录/材料准备 → 代理收到请求 Qn
  → 上游发送 → 首个有效 delta → [DONE] → Output 可观察
  → 工具排队/准备 → invoke → 前序等待/结果提交 → 下轮准备 → Qn+1
```

- 每行是一次物理 provider attempt，间隔是 `Qn+1 - Qn`。同轮重试不增加成功 ReAct 轮数。
- Q 是同机代理完整收到请求体的时刻，**不能证明远端 LLM 收到或开始计算的时刻**。
  `upstream.sent`、首字节和首个正文/思考/工具 delta 进一步区分本地发送与上游等待。
- 末轮截止到最终输出被客户端观察；首轮另列输入到 Q1。总计、前 10 次输出与每轮间隔都保留。
- Akashic 使用落库 API 返回和日志订阅中的最早观测；订阅可能早于 await 返回。Pi 使用原生
  `message_end`、`tool_execution_*` 事件。工具边界有此差异，不能把它们的微小差值全归因于存储。
- 并行工具的时间用区间并集和整个 batch 的 wall time，不能直接相加。缺边界保留 null，
  无法解释的时间列入 `unattributed_ms`；缺失、取消、截断或未完成不视为成功。
- 最终输出不等于任务完成。另查真实 diff、目标行为与针对性验证，记录首次修改、重复读取和失败。

## 采集

Akashic 以 `AKASHIC_TIMING=1 AKASHIC_LOG_FORMAT=json` 启动；正式安装的 Core 和插件都应来自
被测提交。默认关闭打点。日志只包含身份、阶段、计数和同机 `monotonic_ns`，不包含正文或凭据。
`runtime.timing` 通过 request key、call record ID、Output ID 和工具 key 关联。保留完整运行日志。

```sh
python scripts/react-performance/proxy.py \
  --endpoint https://provider.example/v1/chat/completions \
  --key-file /private/benchmark-key --directory /private/trace --port 2311
```

两边的 Base URL 分别设为 `http://127.0.0.1:2311/akashic-r1/v1` 和
`http://127.0.0.1:2311/pi-r1/v1`，认证用占位值；真实凭据只由代理读取。
代理不改变请求正文和响应字节。原始请求/响应仅保存在私有目录，不能直接上传到 PR。
客户端断开会取消代理的上游等待，记录 `client.disconnected`；不能让已放弃请求与重试重叠。
首次模型配置探测必须在 benchmark 输入时刻之前完成。
对齐两边的上游等待上限和重试次数；默认读超时可能不同，应记录实际配置与每次取消。

客户端逐行记录 `{"mono_ns": <time.monotonic_ns()>, "event": <原生事件>}`。
发出输入前记录 `event={"bench":"input"}`；Akashic 同时记录 `session_id` 和请求的 `message_id`。
持续排空事件，避免 stdout/WebSocket 背压造成假性慢。Pi 等到 `agent_settled` 后关闭 stdin。
订阅确认后仍可能重放旧消息；Akashic 按本次 Input 的序号排除历史，首个最终输出结束测量。

```sh
python scripts/react-performance/rounds.py --harness akashic --label akashic-r1 \
  --wire /private/trace/wire.jsonl --events /private/akashic-events.jsonl \
  --timing /private/runtime.log --output /private/akashic-result.json
python scripts/react-performance/rounds.py --harness pi --label pi-r1 \
  --wire /private/trace/wire.jsonl --events /private/pi-events.jsonl \
  --output /private/pi-result.json
```

输出 JSON 和逐 attempt CSV。先用真实短工具链确认边界完整、时间可对账，再跑原始任务；
每个候选使用新的任务目录和会话。保留超时与中断样本，不只选择最快结果。交替运行顺序并记录
cache usage、上下文规模及工具数；先对照同等阶段，再判断慢因。改变提示词的行为实验必须单列，
不能与纯计时修复混称一个基线。停止观察不删除会话、工具结果、失败记录或原始 trace。
