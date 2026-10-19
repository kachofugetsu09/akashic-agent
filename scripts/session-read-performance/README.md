# Session 读取性能基线与爬坡测量

为 `plugins/ledger` 读取面建立可复跑的性能基线，配合爬坡优化逐项对账：
每个候选改动单独测量，保留有真实收益的，回退无收益的。

## 口径

- 一次性临时目录构造真实 `sessions.db`，不触碰任何正式 workspace。
- 目标会话按 4 条小消息（约 500B 输入/输出）+ 1 条大 `ToolResult`（默认 50KB）
  混合，模拟工具密集型长会话；目录另有 `--sessions` 个普通会话（各 40 条）。
  目标会话最后灌入，使目录第一页包含它，`catalog.sessions` 的 stats 聚合成本
  计入测量。
- 冷态 = 新建 `MessageLog`（解码缓存、读连接、共享前缀全部为空），模拟进程
  重启或会话首次被读取；热态 = 同一 log 上追加 4 条消息后再次
  `committed_snapshot`（react 稳态路径，前缀由 writer 直接延展）。
- 每次测量前 `gc.collect()`；报告 `--repeats` 次的中位数与最小值。单次运行间
  存在约 10% 机器噪声，接近的对比用同机背靠背 A/B（交替跑两版代码）。
- SQL 语句数经 sqlite3 连接工厂计数；解码次数包装 `MessageLog._decode`。

## 用例与生产入口

| case | 生产入口 |
|---|---|
| `cold_snapshot` | 重启/恢复后 react 首轮 `committed_snapshot` |
| `warm_snapshot_tail4` | react 稳态每轮增量读面 |
| `read_tail_50` | `GET /api/chat/sessions/{key}/messages` 首屏 |
| `read_pages_full` | 聊天历史向前翻页直到读完全史 |
| `scan_full` | wake/scheduler/subagent 等后台全量 `snapshot()` |
| `settle_scan` | 旧 `_committed`：全前缀扫描找一条 ToolResult |
| `settle_lookup` | 新 `result_committed`：`message_call_result` 索引点查 |
| `catalog_sessions_50` | `GET /api/chat/sessions` 目录首页 |
| `snapshot_heads` | 目录订阅/活动推送的 head 快照 |

## 运行

```sh
.venv/bin/python scripts/session-read-performance/run.py \
  --messages 20000 --sessions 300 --repeats 7 --label <change-name>
```

结果追加到 `results.jsonl`（含 git sha），并打印汇总表。`results.jsonl`
是本仓库的测量档案；对比同一 `--messages/--sessions` 参数下的记录。
