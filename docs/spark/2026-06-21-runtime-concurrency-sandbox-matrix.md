# Runtime 并发竞态沙盒矩阵

日期：2026-06-21
分支：feat/runtime-chat-lane

## 目标

这份文档列出当前 runtime 改造后需要用沙盒验证的并发时序。范围只覆盖会影响 agent 最后回复和可见发送顺序的路径：

- passive：用户消息触发的被动回复链路。
- scheduler soft：定时任务先生成，再发送。
- scheduler instant：定时任务直接发送。
- proactive：主动链路生成后发送。
- drift：drift 生成后发送。

当前 MVP 只保证同一 chat 内：

- passive 未完成或 passive 回复未真实发送完时，non-passive 发送等待。
- 已经进入真实 sender 的 non-passive 不抢占，不回滚。
- scheduler soft 与 passive runtime 复用通过全局 runtime lock 串行。

## 当前路径

```
┌─────────────────────────────────────────────────────────────┐
│ user inbound                                                 │
└──────────────┬──────────────────────────────────────────────┘
               │ publish_inbound
               v
┌─────────────────────────────────────────────────────────────┐
│ ChatLane                                                     │
│ passive_turns += 1                                           │
└──────────────┬──────────────────────────────────────────────┘
               │ AgentLoop.run
               v
┌─────────────────────────────────────────────────────────────┐
│ passive runtime admission lock                              │
│ passive / scheduler soft process_direct 串行                 │
└──────────────┬──────────────────────────────────────────────┘
               │ publish_outbound
               v
┌─────────────────────────────────────────────────────────────┐
│ ChatLane                                                     │
│ passive_sends += 1                                           │
└──────────────┬──────────────────────────────────────────────┘
               │ dispatch_outbound -> run_passive
               v
┌─────────────────────────────────────────────────────────────┐
│ channel sender                                               │
│ passive_sends -= 1                                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ proactive / drift / scheduler instant / scheduler soft send  │
└──────────────┬──────────────────────────────────────────────┘
               │ message_push(_commit_role="non_passive")
               v
┌─────────────────────────────────────────────────────────────┐
│ ChatLane.run_non_passive                                    │
│ waits while sending or passive_turns > 0 or passive_sends > 0│
│ same chat non-passive uses FIFO ticket                       │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ channel sender                                               │
└─────────────────────────────────────────────────────────────┘
```

## 沙盒约定

```
P   = passive 用户 turn
D   = drift
PR  = proactive
SS  = scheduler soft
SI  = scheduler instant
NP  = non-passive visible send
CL  = ChatLane
RTL = passive runtime admission lock
```

所有会等待的调用都必须包 `asyncio.wait_for(..., timeout=...)`，死锁要变成测试失败。所有 sender 用 fake sender 记录顺序，例如：

```
events = [
  "passive:start:reply",
  "passive:end:reply",
  "non_passive:drift",
]
```

需要制造精确交错时，必须在 fake generate / fake sender 中使用 `asyncio.Event` 或等价 barrier。不要只依赖 `asyncio.sleep(0)` 或调度运气来证明 t0/t1/t2 顺序。

## 已落地 Docker 探针

探针入口：

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/runtime_race_probe.py --scenario all
```

可用控制开关：

```text
AKASHIC_RACE_SCENARIO  选择单个场景，默认 all
AKASHIC_RACE_TIMEOUT   每个等待点的超时秒数，默认 2
AKASHIC_RACE_TRACE     写出 JSON 结果的路径
AKASHIC_RACE_CONFIG    指定 config.toml；不指定时生成无外部 channel 的最小配置
AKASHIC_RACE_WORKSPACE 指定临时 workspace；不指定时使用临时目录
```

探针只使用真实 runtime 排序组件，不连接真实 Telegram / QQ / LLM：

```text
┌─────────────────────────────────────────────────────────────┐
│ docker/debug/runtime_race_probe.py                          │
└──────────────┬──────────────────────────────────────────────┘
               │ fake inbound / fake sender
               v
┌─────────────────────────────────────────────────────────────┐
│ real MessageBus + ChatLane                                  │
└──────────────┬──────────────────────────────────────────────┘
               │
      ┌────────┴────────┐
      v                 v
┌──────────────┐  ┌──────────────────────┐
│ BusOutbound  │  │ PushToolOutbound     │
│ passive path │  │ non-passive path     │
└──────┬───────┘  └──────────┬───────────┘
       │                     │
       v                     v
┌─────────────────────────────────────────────────────────────┐
│ fake sender records start/end order                         │
└─────────────────────────────────────────────────────────────┘
```

其中 `agent-loop-runtime` 会额外启动真实 `AgentLoop.run()`，但仍不启动 Telegram / QQ / CLI server：

```text
┌─────────────────────────────────────────────────────────────┐
│ generated or provided config.toml                           │
│ channels.telegram disabled / channels.qq disabled            │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ real AgentLoop.run + CoreRunner + AgentCore                 │
│ fake Reasoner blocks passive turn                           │
└──────────────┬──────────────────────────────────────────────┘
               │ same MessageBus / ChatLane
       ┌───────┴──────────────────┐
       v                          v
┌──────────────┐          ┌───────────────────────────┐
│ user inbound │          │ scheduler process_direct  │
│ passive turn │          │ waits runtime lock        │
└──────┬───────┘          └──────────┬────────────────┘
       v                             v
┌─────────────────────────────────────────────────────────────┐
│ assert: passive reply -> drift send -> scheduler send        │
│ assert: reasoner max_active == 1                             │
└─────────────────────────────────────────────────────────────┘
```

当前覆盖场景：

```text
agent-loop-runtime             真实 AgentLoop.run + RTL + ChatLane 联合验证
config-runtime-llm             显式运行；真实 config.toml + 真实 LLM + fake channel
a1-drift-before-push          A1
a3-drift-sending-then-user    A3
b1-scheduler-after-user       B1 / B4
d1-fifo-passive-insert        D1 + passive 插入
c2-cross-chat-isolated        跨 chat 不互相阻塞
e1-silent-passive             passive 无回复仍释放 lane
e6-cancelled-nonpassive-ticket 取消等待中的 non-passive 不留下 ticket 洞
```

`config-runtime-llm` 不包含在默认 `--scenario all` 中，因为它会调用真实模型。它用于回答“接近线上 runtime 的真实 config 验证”：

```bash
docker compose -f docker/debug/docker-compose.yml run --rm akashic-debug \
  python docker/debug/runtime_race_probe.py \
    --scenario config-runtime-llm \
    --config config.toml \
    --timeout 120
```

```text
┌─────────────────────────────────────────────────────────────┐
│ real config.toml                                             │
│ real provider / model / memory / tools / plugins             │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ build_core_runtime                                           │
│ no Telegram / QQ / CLI server start                          │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ real AgentLoop passive + real scheduler process_direct        │
│ fake proactive/drift generation                              │
│ fake channel sender                                          │
└──────────────┬──────────────────────────────────────────────┘
               v
┌─────────────────────────────────────────────────────────────┐
│ assert passive reply before non-passive visible sends         │
└─────────────────────────────────────────────────────────────┘
```

## A. Drift / Proactive 与用户消息

### A1. drift 已经开始生成，用户消息进入，然后 drift 才 message_push

```
t0  D  开始生成
t1  P  publish_inbound
       CL.passive_turns = 1
t2  D  message_push(non_passive)
       CL.run_non_passive waits passive_turns > 0
t3  P  进入 RTL，生成回复
t4  P  publish_outbound
       CL.passive_sends = 1
t5  P  complete_inbound
       CL.passive_turns = 0
t6  P  dispatch_outbound -> run_passive -> sender
       CL.passive_sends = 0
t7  D  non_passive sender
```

期望：passive 回复先发，drift 后发。proactive 同样适用。

### A2. drift 已到 message_push，但用户消息已经先登记

```
t0  P  publish_inbound
       CL.passive_turns = 1
t1  D  message_push(non_passive)
       waits
t2  P  complete + passive sender done
t3  D  sender
```

期望：passive 先，drift 后。

### A3. drift 已经进入真实 sender，用户消息才进入

```
t0  D  message_push(non_passive)
       CL.sending = True
t1  P  publish_inbound
       CL.passive_turns = 1
t2  D  sender done
       CL.sending = False
t3  P  passive runtime + passive sender
```

期望：drift 已经开始真实发送，不能被抢占；drift 先发，passive 后发。沙盒只验证不死锁。

### A4. proactive gate 通过后，用户消息插入

```
t0  PR gate pass
t1  PR fetch / judge / resolve awaits
t2  P  publish_inbound
t3  PR message_push(non_passive)
       waits
t4  P  passive reply sent
t5  PR sender
```

期望：不作废 proactive，只延后发送。

## B. Scheduler

### B1. scheduler soft 正在生成，用户消息进入

```
t0  SS process_direct enters RTL
t1  P  publish_inbound
       CL.passive_turns = 1
t2  SS generation done, exits RTL
t3  SS message_push(non_passive)
       waits passive_turns > 0
t4  P  enters RTL, generates reply
t5  P  passive sender done
t6  SS sender
```

期望：用户回复先发，soft 结果后发。代价：P 的生成会被 SS 持有的全局 RTL 延后。

### B2. 用户 passive 先进入 runtime，scheduler soft 同时触发

```
t0  P  enters RTL
t1  SS process_direct waits RTL
t2  P  reply publish + complete + sender
t3  SS enters RTL
t4  SS generation done
t5  SS sender
```

期望：P 先完成，SS 后完成。

### B3. 两个 scheduler soft 同时到期

```
t0  SS1 process_direct enters RTL
t1  SS2 process_direct waits RTL
t2  SS1 generation done
t3  SS2 generation starts
t4  SS1 message_push(non_passive)
t5  SS2 message_push(non_passive)
```

期望：生成阶段串行。同 chat 发送按 non-passive FIFO；不同 chat 发送可并发。

### B4. scheduler instant 与 pending passive

```
t0  P  publish_inbound
       CL.passive_turns = 1
t1  SI message_push(non_passive)
       waits
t2  P  passive sender done
t3  SI sender
```

期望：passive 先发，instant 后发。

### B5. scheduler instant 已经进入真实 sender，用户消息才进入

```
t0  SI message_push(non_passive)
       CL.sending = True
t1  P  publish_inbound
t2  SI sender done
t3  P  passive sender
```

期望：不可抢占，SI 先发；之后 P 正常回复。

## C. Passive 与 Passive

### C1. 同一 chat 连续两条用户消息，期间有 non-passive 等待

```
t0  P1 publish_inbound
       CL.passive_turns = 1
t1  P2 publish_inbound
       CL.passive_turns = 2
t2  NP message_push(non_passive)
       waits
t3  P1 complete + passive sender
       CL.passive_turns = 1
t4  P2 complete + passive sender
       CL.passive_turns = 0
t5  NP sender
```

期望：同 chat passive backlog 全部清空后，NP 才发送。

### C2. 不同 chat 的 passive 与 non-passive

```
t0  chat A P publish_inbound
       lane A passive_turns = 1
t1  chat B NP message_push(non_passive)
       lane B no pending
t2  chat B NP sender
t3  chat A P sender
```

期望：ChatLane 不跨 chat 阻塞。注意 RTL 仍是全局，passive 生成本身可能排队。

### C3. passive 生成中，另一个 chat 的用户消息进入

```
t0  chat A P enters RTL
t1  chat B P publish_inbound
t2  chat B P waits RTL
t3  chat A P exits RTL
t4  chat B P enters RTL
```

期望：当前 MVP 中 passive runtime 全局串行，这是已知吞吐代价。

## D. Non-passive 与 Non-passive

### D1. 同一 chat 多个 non-passive 同时准备发送

```
t0  NP1 gets FIFO ticket 0
t1  NP2 gets FIFO ticket 1
t2  NP1 sender
t3  NP2 sender
```

期望：同 chat non-passive FIFO。

### D2. 同一 chat NP2 等待时，用户消息进入

```
t0  NP1 sender running
t1  NP2 waits ticket 1
t2  P publish_inbound
       CL.passive_turns = 1
t3  NP1 sender done
t4  P passive sender
t5  NP2 sender
```

期望：等待中的 non-passive 让位给后来进入的 passive。

### D3. 不同 chat 多个 non-passive

```
t0  chat A NP1 sender
t1  chat B NP2 sender
```

期望：不同 chat 不互相等待。

### D4. 持续 passive 高负载

```
t0  NP waits
t1  P1 pending
t2  P2 pending
t3  P3 pending
t4  P backlog drains
t5  NP sender
```

期望：只要 passive backlog 最终归零，NP 不永久饿死；如果同 chat 一直有 passive 输入，NP 会按产品优先级一直延后。

## E. Liveness

### E1. silent / gate-exit passive 没有 outbound

```
t0  P publish_inbound
       CL.passive_turns = 1
t1  NP message_push(non_passive)
       waits
t2  P complete_inbound
       CL.passive_turns = 0
t3  NP sender
```

期望：没有 passive outbound 也不会泄漏；NP 能恢复。

### E2. passive 被取消

```
t0  P publish_inbound
t1  P task cancelled
t2  AgentLoop finally complete_inbound
t3  NP sender
```

期望：不泄漏 passive_turns。若取消前已经 publish_outbound，则还要等 passive_sends 发送完。

### E3. passive 报错，错误回复经 bus 发出

```
t0  P publish_inbound
t1  P raises
t2  AgentLoop publishes error outbound
       CL.passive_sends = 1
t3  AgentLoop complete_inbound
       CL.passive_turns = 0
t4  error sender done
       CL.passive_sends = 0
t5  NP sender
```

期望：错误回复仍算 passive 可见发送，NP 在错误回复之后。

### E4. passive dispatch 首次失败并进入 2s retry

```
t0  P run_passive enters sender
       CL.sending = True
t1  sender fails, sleeps 2s
t2  NP waits sending == True
t3  retry or fallback returns
       CL.sending = False
t4  NP sender
```

期望：retry 期间后续同 chat 发送等待，retry 结束后释放。

### E5. 手工 publish_inbound 但没有 AgentLoop complete

```
t0  test/manual publish_inbound
       CL.passive_turns = 1
t1  no AgentLoop consumes it
t2  NP waits forever
```

期望：这是测试桩错误，不是生产路径；沙盒里直接打 MessageBus 时必须手工 complete_inbound。

### E6. non-passive 持票等待时被取消

```
t0  P  publish_inbound
       CL.passive_turns = 1
t1  NP1 run_non_passive
       gets ticket 0 and waits
t2  NP1 cancelled by wait_for timeout or task cancel
       ticket 0 is marked cancelled
t3  P  complete_inbound
       CL.passive_turns = 0
t4  NP2 run_non_passive
       gets ticket 1
       CL skips cancelled ticket 0
t5  NP2 sender
```

期望：被取消的 non-passive waiter 不留下 ticket 洞；同 chat 后续 non-passive 不永久阻塞。

## F. Isolation

### F1. passive tool_search 与 scheduler soft tool_search

```
t0  P enters RTL
t1  SS process_direct waits RTL
t2  P tool_search execute(excluded_names=...)
t3  P exits RTL
t4  SS enters RTL
t5  SS tool_search execute(excluded_names=...)
```

期望：不并发复用 passive runtime；tool_search 的 excluded_names 也是 per-call 参数，没有共享字段。

### F2. scheduler soft 与 scheduler soft

```
t0  SS1 enters RTL
t1  SS2 waits RTL
t2  SS1 exits RTL
t3  SS2 enters RTL
```

期望：soft 生成阶段串行，避免复用 passive runtime。

### F3. proactive / drift 生成与 passive

```
t0  PR or D generates in proactive runtime
t1  P generates in passive runtime
t2  visible send goes through ChatLane
```

期望：生成阶段不共享 passive reasoner / tool_search；可见发送顺序由 ChatLane 兜住。

### F4. passive 内部直接调用 message_push

```
t0  P tool call message_push
t1  _commit_role is empty
t2  MessagePushTool sends directly
t3  P final bus outbound may send later
```

期望：这是 passive turn 内的显式外部副作用，不按 non-passive 排队。沙盒需要确认它不会因为同一个 turn 的 passive_turns 自己等自己。

## G. TOCTOU / 相关性

### G1. proactive 入口 gate 通过后用户插话

```
t0  PR gate sees user not busy
t1  PR awaits fetch / judge / resolve
t2  P publish_inbound
t3  PR message_push(non_passive)
       waits
t4  P reply sent
t5  PR sender
```

期望：排序正确，但内容不作废。这是当前产品语义。

### G2. proactive 入口 gate 看到 busy

```
t0  P already busy
t1  PR gate checks busy
t2  PR skip
```

期望：入口产品节流仍生效；ChatLane 只处理已生成内容的提交顺序。

## H. 多 outbound / 分段发送

### H1. 一个 passive turn 发布多条 outbound

```
t0  P publish_outbound #1
       passive_sends = 1
t1  P publish_outbound #2
       passive_sends = 2
t2  P complete_inbound
       passive_turns = 0
t3  sender #1 done
       passive_sends = 1
t4  sender #2 done
       passive_sends = 0
t5  NP sender
```

期望：已入队的 passive outbound 全部真实发送后，NP 才发送。

### H2. passive complete 后才由后台补发 outbound

```
t0  P complete_inbound
       passive_turns = 0
t1  NP enters sender
t2  background publishes passive-like outbound
```

期望：当前系统不应把 passive 可见回复放到 turn 完成后异步补发；如果未来出现这种路径，需要把它接入 ChatLane 的 passive_sends 或改成 non-passive。

## 建议沙盒批次

```
batch 1  liveness
         E1 E2 E3 E4

batch 2  ordering same chat
         A1 A2 A3 B4 D1 D2 H1

batch 3  scheduler isolation
         B1 B2 B3 F1 F2

batch 4  cross chat
         C2 C3 D3

batch 5  product semantics
         G1 G2 F4 H2
```

## 已知 MVP 代价

```
┌──────────────────────────────┬────────────────────────────────────────┐
│ 边界                         │ 当前行为                               │
├──────────────────────────────┼────────────────────────────────────────┤
│ 已进入真实 sender 的 NP       │ 不抢占，不回滚                         │
│ passive runtime admission     │ 全局锁，不按 chat 拆分                 │
│ passive 内部 message_push     │ 不打 non_passive 标记，避免自己等自己  │
│ proactive 相关性              │ 不作废，只延后                         │
└──────────────────────────────┴────────────────────────────────────────┘
```
