# 插件运行时执行轨迹诊断

## 1. 先固定诊断身份

不要按提示词正文、插件显示名或“最近一条”猜目标。先从 programmatic/control 输出记录：

```text
execution_id
thread_id / session_key
turn_id
plugin_id = <name>@<marketplace>
reload tx_id（安装后从 journal 取得）
父 turn_id（验证调用来自父 turn 时）
```

诊断只读取正式状态。不要对在线 `sessions.db` 或 reload journal 执行 `UPDATE`、`DELETE`、`VACUUM`、迁移或复制覆盖；使用 SQLite `mode=ro`，让读取包含在线 WAL。不要对在线库使用 `immutable=1`。

## 2. 按层重建，不从结果倒推

```text
source commit
    │
    ▼
reload transaction ── candidate/generation 是否发布
    │
    ▼
turn admission ────── queued 是否进入 started
    │
    ▼
plugin module ─────── 是否执行、产生何种 section/tool
    │
    ▼
provider boundary ─── 实际输入与输出
    │
    ▼
SessionDB ─────────── final、items、messages、tool trace
```

后一层失败不能自动证明前一层失败。`plugin-doctor healthy`、reload `complete`、turn `completed` 都不是行为 oracle。

## 3. 先检查真实 schema

设置本次诊断变量；只使用 control/runtime 返回的严格 ID：

```bash
AK_DIAG_WORKSPACE=/absolute/path/to/workspace
AK_DIAG_TURN_ID='turn:<uuid>'
AK_DIAG_PLUGIN_ID='<name>@<marketplace>'
```

读取当前 schema，不凭旧文档猜列名：

```bash
sqlite3 "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" '.schema turns'
sqlite3 "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" '.schema sessions'
sqlite3 "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" '.schema messages'
sqlite3 "file:$AK_DIAG_WORKSPACE/runtime/plugin-reloads.sqlite3?mode=ro" \
  '.schema reload_transactions'
sqlite3 "file:$AK_DIAG_WORKSPACE/runtime/plugin-reloads.sqlite3?mode=ro" \
  '.schema reload_events'
```

任一表或列不同就停止使用后续固定查询，按实际 schema 调整；不要用空结果掩盖查询错误。

## 4. 一次导出 turn 轨迹

优先运行 Skill 自带的只读脚本。默认只输出状态、时间线和内容形状，不打印会话正文、工具参数或 metadata 值：

```bash
python skills/develop-akashic-plugin/scripts/inspect-runtime-trace.py \
  --workspace "$AK_DIAG_WORKSPACE" \
  --turn-id "$AK_DIAG_TURN_ID" \
  --plugin-id "$AK_DIAG_PLUGIN_ID" \
  > /tmp/akashic-plugin-runtime-trace.json
```

分页读取输出，不把大 JSON 塞进单个 control frame。报告前至少核对：

- `turn.status`、`created_at`、`started_at`、`completed_at`；
- `turn.input_summary`、`items_summary`、`final_response_summary`、`error_summary`；
- 同 session 的 `messages[].content_summary/tool_chain_summary/extra_summary`；
- 最新 reload transaction 与完整 phase events。

确实需要核对正文或工具参数时，只在可信本机终端显式加 `--include-content`。该输出可能包含 token、私密消息、文件内容和外部 API 参数，不要转存到 CI、公共日志或 agent transcript：

```bash
python skills/develop-akashic-plugin/scripts/inspect-runtime-trace.py \
  --workspace "$AK_DIAG_WORKSPACE" \
  --turn-id "$AK_DIAG_TURN_ID" \
  --plugin-id "$AK_DIAG_PLUGIN_ID" \
  --include-content
```

没有脚本时，用当前 schema 执行下列只读查询。

### 4.1 Turn 状态、结果和时间线

```bash
sqlite3 -json "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" \
  "SELECT id, session_key, status,
          created_at, started_at, completed_at,
          ROUND((julianday(started_at)-julianday(created_at))*86400, 3)
            AS queued_seconds,
          final_response, error_json, usage_json,
          input_json, items_json
     FROM turns
    WHERE id='$AK_DIAG_TURN_ID';"
```

读取完整 item，而不是只看 final response：

```bash
sqlite3 -json "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" \
  "SELECT item.key AS ordinal,
          json_extract(item.value, '$.id') AS item_id,
          json_extract(item.value, '$.type') AS item_type,
          json_extract(item.value, '$.data') AS item_data
     FROM turns AS turn_record,
          json_each(turn_record.items_json) AS item
    WHERE turn_record.id='$AK_DIAG_TURN_ID'
    ORDER BY item.key;"
```

判读：

- `queued` 且 `started_at IS NULL`：只证明已受理但未执行；检查它是否复用了父 turn 的 session/thread，以及控制面容量是否已满，不能再默认归因于全局锁。
- `started_at` 晚于父 turn `completed_at`：直接证明 child 等父 turn 释放后才启动。
- `completed` 但 final 不符合 oracle：行为验证失败，不能用 doctor/reload 成功覆盖。
- `failed/interrupted/cancelled`：读取 `error_json` 和已闭合 items，保留真实终态。

### 4.2 Session 消息、模型上下文和工具轨迹

```bash
sqlite3 -json "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" \
  "SELECT id, seq, role, content, tool_chain, extra, ts
     FROM messages
    WHERE session_key=(SELECT session_key FROM turns WHERE id='$AK_DIAG_TURN_ID')
    ORDER BY seq;"
```

单独提取模型面对的 context frame：

```bash
sqlite3 -json "file:$AK_DIAG_WORKSPACE/sessions.db?mode=ro" \
  "SELECT seq, role,
          json_extract(extra, '$.llm_context_frame') AS llm_context_frame,
          tool_chain
     FROM messages
    WHERE session_key=(SELECT session_key FROM turns WHERE id='$AK_DIAG_TURN_ID')
    ORDER BY seq;"
```

`messages.tool_chain` 保存实际工具调用参数和结果；`turns.items_json` 保存 control item 与 assistant metadata。两者都没有目标工具时，不能声称工具行为通过。

重要限制：`llm_context_frame` 只保存 context-frame user message。插件写入普通 `system_sections_top/system_sections_bottom` 的内容会进入 system prompt，当前 SessionDB 不保存完整 system prompt。因此在 `llm_context_frame` 找不到插件 section，不能证明插件未执行；需要 runtime log、插件探针或 provider payload 补证。

`items_json` 里的 assistant `thinking` 是模型输出，不是 runtime provenance。它可以帮助发现“模型自述没有看到某段提示”，但不能单独证明最终 system prompt 的字节内容。

## 5. 读取 reload generation 轨迹

```bash
sqlite3 -json \
  "file:$AK_DIAG_WORKSPACE/runtime/plugin-reloads.sqlite3?mode=ro" \
  "SELECT tx_id, plugin_id, base_snapshot_id, candidate_snapshot_id,
          generation_id, source_revision, config_revision,
          phase, started_at, updated_at, error
     FROM reload_transactions
    WHERE plugin_id='$AK_DIAG_PLUGIN_ID'
    ORDER BY started_at DESC
    LIMIT 1;"
```

取得上一步精确 `tx_id` 后读取事件：

```bash
AK_DIAG_TX_ID='<tx-id-from-previous-query>'
sqlite3 -json \
  "file:$AK_DIAG_WORKSPACE/runtime/plugin-reloads.sqlite3?mode=ro" \
  "SELECT sequence, phase, details_json, created_at
     FROM reload_events
    WHERE tx_id='$AK_DIAG_TX_ID'
    ORDER BY sequence;"
```

`complete` 证明候选 generation 完成发布与排空状态机，不证明目标 child 绑定它，也不证明插件 module 或 Tool 被执行。`aborted` 必须读取 `error` 和最后事件，不重复安装相同 source revision。

## 6. 先确认日志是否真的可回读

Python runtime 默认把日志写到 stderr。先定位当前 Gateway PID，再看真实 sink：

```bash
ps -eo pid,ppid,pgid,lstart,args | rg '[p]ython.*main.py.*gateway'
AK_DIAG_GATEWAY_PID='<gateway-pid>'
readlink "/proc/$AK_DIAG_GATEWAY_PID/fd/2"
```

按结果处理：

- 普通文件：使用该文件。
- 已知 systemd unit 的 journal pipe：从进程 owner/启动配置确认 unit 后使用 `journalctl --user -u <exact-unit>`；不要猜 unit。
- `/dev/pts/*`：日志只在启动终端，没有持久文件可供当前 Agent 回读。明确报告 `runtime log unavailable: stderr is tty`。
- 已关闭、无权限或未知 pipe：明确报告不可用，不把空 `rg` 当作“没有错误”。

有持久日志时，只用精确 identity 对齐：

```bash
AK_DIAG_RUNTIME_LOG=/absolute/path/to/runtime.log
rg -n -F \
  -e "$AK_DIAG_TURN_ID" \
  -e "$AK_DIAG_PLUGIN_ID" \
  -e "$AK_DIAG_TX_ID" \
  "$AK_DIAG_RUNTIME_LOG"
```

同时检查时间窗口，避免把旧 generation 的同名日志归给当前 turn。

## 7. 日志不够时写受控插件探针

插件能证明“自己的代码执行过”，不能单独证明“自己从未执行”。为隔离 canary 或获授权的诊断版本增加结构化 marker：

```python
import hashlib
import json
import logging

from agent.control.context import current_turn_id
from agent.plugins.snapshot import get_current_runtime_snapshot

logger = logging.getLogger(__name__)


def log_prompt_probe(*, plugin_context, prompt_ctx, section) -> None:
    snapshot = get_current_runtime_snapshot()
    marker = {
        "event": "plugin_prompt_probe",
        "plugin_id": plugin_context.plugin_id,
        "generation_id": plugin_context.generation_id,
        "snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
        "turn_id": current_turn_id.get(),
        "session_key": prompt_ctx.session_key,
        "section_name": section.name,
        "section_chars": len(section.content),
        "section_sha256": hashlib.sha256(section.content.encode()).hexdigest(),
    }
    logger.info("plugin_probe %s", json.dumps(marker, ensure_ascii=False))
```

在 section 已追加、Tool 即将返回真实结果的边界记录 marker；不要只在 `prepare()` 记录，因为它只能证明插件被构造。

默认只记录 section 名称、长度和摘要，避免把记忆、凭据或用户正文复制进诊断日志。需要取得精确动态 section 内容时，只在不含敏感信息的隔离 canary 中显式增加 `section_content`，并把这项 capture 写进任务 write set；生产插件继续使用 hash 与 canonical source 对照。

stderr 没有持久 sink 时，只对隔离 canary 或用户明确授权的插件使用 `plugin_context.kv_store.set("last_prompt_probe", marker)` 保存最后一次探针。先记录 plugin-data 基线和 write set；探针写入是持久副作用，不得在默认只读验证中偷偷加入，也不得把删除整个 plugin-data 当作清理。

判读：

| reload | turn | probe | final/tool oracle | 结论 |
|---|---|---|---|---|
| 无事务或 aborted | 任意 | 任意 | 任意 | 安装/候选阶段失败 |
| complete | queued | 无 | 无 | admission 阻塞，尚未执行插件 |
| complete | started/completed | 无 | 失败 | snapshot/module 接线可疑；候选不能自证未运行 |
| complete | completed | 有，hash 正确 | 失败 | module 已运行；继续查完整 provider payload、指令优先级或后续改写 |
| complete | completed | 有 | 通过 | 具备执行证据；再核对 child snapshot identity 和副作用 |

插件 marker 不是 provider receipt。没有完整 system prompt/provider payload 时，只能报告“module 已执行并生成指定 section”，不能升级成“模型一定收到”。

## 8. 递归诊断的当前边界

父子使用不同 session 时：

```text
T 安装 → reload event committed → 创建 V
T 仍在运行 ───────────────────────┐
V started/completed → T 读取结果 ──┘
```

因此当前实现支持 `T await V → T 根据结果继续修改`。必须用父子的 `started_at/completed_at` 证明 V 在 T 完成前终止；只看到两个 completed 记录不足以证明没有排队。V 一直 queued 时，立即检查 session/thread identity 与 admission 容量，不要再次增加 timeout。

staged install 的 reload transaction 到达 `latest_ready` 后，V 才能显式租用 latest；T 和普通 session 继续使用 stable。行为 oracle 完成后再 promote 或 discard，并确认 journal 到达对应终态。attached CLI 断开会取消该连接拥有的服务端 turn；若仍长期 queued，优先检查 thread/session identity、latest readiness 和 admission 容量。

最终报告按证据层写明：source commit、reload tx/generation、child thread/turn、时间线、items/tool trace、final oracle、日志 sink、探针 marker、未取得的 provider/snapshot 证据。没有取得的层保持未知，不用推断补齐。
