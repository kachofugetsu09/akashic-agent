# 发消息 → 模型收到请求（TTFT 前段）

用户按下发送，到第一次模型请求离开本机，这段时间全部在本地：接纳输入、选择回复、准备记忆与上下文材料、组装请求。模型自己的首 token 时间另算（生产上是 `model_calls.first_token_ms`）。这一段越短，用户越早看到"开始回复"。

## 口径

- **生产**：`prod_gaps.py`，只读。取用户 Input 的 `ts` 到其后第一条 `model_calls.started_at` 的间隔，按会话规模（<1000、≥1000、≥5000 条消息）和日期分组取中位数。
  - 秒级精度。
  - 并发会话的调用可能落在两者之间，单条偏小，所以只看趋势。
  - 部署重启后的第一次提问单独看，它包含冷进程和冷页缓存。
- **本地**：隔离实例 + 生产库只读副本 + `fixture_model.py`。指标是 `measure.mjs` 在浏览器里按下回车，到 fixture 记录到本轮标记的墙钟。
  - 同时记录 gateway 在这段时间里的 CPU 和磁盘读量（`/proc/<pid>/stat`、`/proc/<pid>/io`）。
  - **冷**：重启实例，等 120 秒，对 workspace 下的 sqlite 文件做 `POSIX_FADV_DONTNEED`，然后发第一轮。
  - **热**：同一进程里接着发，轮间隔 20 秒。间隔必须超过共享读面的保留期（10 秒），否则热状态被高估。
- **分阶段**：实例带 `AKASHIC_TIMING=1 AKASHIC_LOG_FORMAT=json` 启动，`phases.py` 把启动日志切成逐轮阶段表：`reply.selection`、`materials`、`context.build` 等。

## 运行

```bash
# fixture：只监听本机；记录请求到达时刻与请求体里数值最大的 TTFT 标记
python scripts/ttft-performance/fixture_model.py 18472 /tmp/ttft-fixture.jsonl

# 会话模型先在选择器里切到 fixture；measure 检查到不是 fixture 就不发送
node scripts/ttft-performance/measure.mjs --origin http://127.0.0.1:4440 --storage state.json \
  --session <会话 id> --fixture-log /tmp/ttft-fixture.jsonl --runs 3 --gap 20000 \
  --gateway-pid <gateway pid> --before-first '<驱逐页缓存的命令>' --label main --out results.jsonl

python scripts/ttft-performance/phases.py <state>/startup-*.log --session <会话 id 后缀>
python3 scripts/ttft-performance/prod_gaps.py /srv/data/services/akashic/state/workspace --days 14
```

### 坑

- 实例必须断网运行（独立网络命名空间）。副本里有真实凭据。
- 回合在第一次调用 LLM 之前，会先调用 akasha 的 embedding。本地要把 embedding 请求也转给 fixture，但**不能改 embedding 连接的地址**：地址属于 embedding 空间身份，改了以后 akasha 图会判为"不能混用不同 embedding 空间"，每次通知都整套重新装载再失败，本地会测出 20 秒的假数据。转发要放在不改变连接身份的那一层做。
- 召回材料会带着更早轮次的标记，fixture 取数值最大的标记作为本轮。

## 基线与爬坡

被测会话：1.1 万条消息、正文 191MB、模型请求约 600 条 / 2MB。

### 生产（`prod_gaps.py`，hua-home）

| 时段 | ≥5000 条会话，日中位数 | <1000 条会话，日中位数 |
|---|---|---|
| 09-20 – 10-05 | 20–37s | — |
| 10-17 – 10-26 | 6.5–11.6s | 1.7–2.2s（个别 3.7–5.8s） |
| 10-27 部署重启后第一次 | 53.2s（其中事件循环停顿 7.9s） | — |

### 本地（Core Ultra 5 125H + SSD，prod 副本；生产是 Ryzen 7 2700 + 机械硬盘）

| 版本 | 冷启动首轮 | 热状态 | 冷启动读盘 |
|---|---|---|---|
| main 0ac01d9a | 4.74 / 4.98s | 2.78–3.24s | 约 415MB |
| #1732 同一轮全史只解码一次、组装移出事件循环 | 3.65 / 3.69s | 1.68–2.23s | 约 415MB |

main 热状态的阶段拆分（#1732 之前）：`reply.selection` 2.1–3.3s，`materials` 0.3–0.6s，`context.build` 0.17s（冷启动 0.8s）。

- `reply.selection` 的主体是全史解码（离线单独测约 1.07s：SQL 0.3–0.45s、JSON 0.6s、冻结 0.25s）加上 Turn 投影。
- #1732 之前，同一轮有两个读者先后各解码一次全史：改动记录和回复选择。

### 仍未解决

- **每轮仍要解码一次全史。** 模型投影折叠和 Turn 投影都按全史设计；把解码结果常驻在内存里要占 466MB。
- **冷启动随机读 sessions.db 约 200–400MB。** 同一会话的行和其他会话按写入顺序交错存放，机械硬盘上这是冷启动的主要开销。
