# Host Bridge Protocol V2

- 状态：accepted / implementation
- 决策：[0055](../decisions/0055-host-bridge-uses-typed-protobuf.md)
- 关联：RUN-013～RUN-015、SH-001～SH-003、[持久化状态地图](persistence-state-map.md)

## 1. 目标与范围

删除 Host Bridge wire 中的 JSON/Base64，降低编码 CPU 和传输量。所有 12 个现有 RPC
一次切换到 typed V2。维持工具公开 API、执行模型和完整诊断日志；不引入 streaming、连接池、
压缩或独立执行服务。端到端收益必须由 Local/V1/V2 对比证明，编码微测不代替 shell 性能。

```text
┌──────────────────────────────┐
│ Core 工具 / Bridge client    │
│ 既有工具合同 → typed request │
└──────────────┬───────────────┘
               │ grpc.aio UDS · Protobuf bytes
┌──────────────▼───────────────┐
│ Bridge RPC 边界              │
│ metadata 认证 / 字段校验     │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 既有 ShellProcessManager     │
│ 进程 / lease / 输出消费 / 日志│
└──────────────────────────────┘
```

Bridge service 拥有 boot admission 和 manager lease；ShellProcessManager 拥有 execution、
进程组、等待和输出。Protobuf 只是边界表示。Core 的 Turn、Session、插件 generation 和
正式 workspace owner 不变。文件 adapter 复用原工具，只把其结果转换为文本或二进制图片。

## 2. 请求字段

源文件：`agent/host_bridge/host_bridge.proto`。所有请求包含必需的 `RequestContext`：
`boot_id`、`manager_id`、`request_id`、`expected_release_commit`、`expected_toolchain_digest`
为非空字符串；`session_ref`、`turn_id` 可省略，设置时必须非空。非空字段的缺省空字符串直接拒绝；
允许零/空的必需标量使用 explicit presence。认证只接受一项 `authorization: Bearer <token>` metadata。

下表的“必需”代表缺失时 INVALID_ARGUMENT；repeated/map 无存在性，空集合按表内合同解释。

| RPC | Context 之外的请求字段与规则 |
|---|---|
| Inspect、ClaimBoot、Probe、Heartbeat、ShutdownManager、ActiveExecutions | 无 |
| Exec | 必需非空 command、owner_session_key；argv 非空且元素非空；env 为 string map，可空且 value 可空；cwd optional；必需 tty（false 合法）、yield_time_ms（零合法，等待仍按原 manager clamp）、max_output_tokens（≥0）、hard_timeout_s（>0） |
| WriteStdin | 必需 execution_id>0、非空 owner_session_key、chars（空代表等待）、yield_time_ms、max_output_tokens≥0 |
| Stop | 必需 execution_id>0、非空 owner_session_key |
| TerminateOwner | 必需非空 owner_session_key |
| FileTool | allowed_dir optional；必需 operation oneof，见下表 |
| SkillRequirements | bins、env 为 repeated string，集合可空，元素必须非空 |

`max_output_tokens` 和 `hard_timeout_s` 的限制是 V2 公开边界规则，不能描述成 manager
已有的直接 RPC 校验。cwd、allowed_dir 省略映射为 None，明确空字符串仍按原 Path 行为处理。

| FileTool operation | 参数 |
|---|---|
| read | 必需 path；optional offset≥0，省略为0；optional limit≥1，省略为None |
| write | 必需 path、content，允许空内容 |
| edit | 必需 path、old_text、new_text；optional replace_all，省略为false |
| list | 必需 path |

文件字符串允许空值，继续由原文件工具解释；missing 与显式空值不同。无法匹配 oneof 或缺少
必需字段时拒绝调用。原工具“文件不存在”“匹配多处”等业务错误仍是 OK RPC 的文本结果。

## 3. 响应字段

| RPC | 响应规则 |
|---|---|
| Inspect、Probe | 非空 release_commit/toolchain_digest；非空 capabilities 集合，元素非空 |
| ClaimBoot | 非空 owner_boot_id；previous_boot_id optional，省略表示此前无 owner；必需 cleaned_manager_count、cleaned_execution_count，可为0 |
| Heartbeat | 必需 alive，成功须为true |
| Exec、WriteStdin | 必需 output bytes（可空）、wall_time_ms/original_token_count/output_omitted_bytes≥0、非空 finish_reason；必需 result oneof：execution_id>0 或 exit_code（零与负信号值合法）；output_path optional，设置时非空 |
| Stop | 必需 stopped，false 的存在性不能丢失 |
| TerminateOwner、ShutdownManager | attempted/cleaned 为正整数集合，可空；failures 为 execution_id>0、非空 error_type/message 列表，可空 |
| ActiveExecutions | execution_ids 为正整数集合，可空 |
| FileTool | 必需 result oneof：text（可空），或 image(text、mime_type、data、detail)；图片文本可空但必须存在，data 必须存在且非空，detail 为现有high |
| SkillRequirements | 必需 available/missing，各含 bins/env 名称列表，允许空；精确覆盖请求，不泄露环境变量值 |

客户端在远端响应边界校验 presence、值域和 oneof，不用默认值伪造成功。
文件图片只接受现有单个 image_url/high/data URI 结果。未知 block、额外 metadata、坏 data URI
都 fail-loud；不丢字段、不转空结果。客户端重建已有 ToolResult，模型能力投影仍在 Core。
单条 gRPC 消息的收发限制维持 16 MiB，Shell 输出预算与有界缓冲沿用原 manager。

## 4. 错误、取消和并发

结构/范围错误为 INVALID_ARGUMENT；缺失、重复、格式错误或不匹配的 token 为 PERMISSION_DENIED；
release、toolchain、boot 或 owner 的权限错误仍为 PERMISSION_DENIED；内部未预期错误为 INTERNAL。
命令非零退出是正常 ExecutionReply。请求身份只在 RPC 入口认证一次，内部仍逐操作检查实时 lease。

取消显式传播 CancelledError，不改为 INTERNAL，不终止已登记的 execution。Exec 响应丢失、
UNAVAILABLE、deadline 或 caller 取消后，执行状态可能未知；WriteStdin 的输入也可能已经写入。
客户端不自动重发，不开启 retry/hedging，channel 显式关闭 gRPC retries。request_id 只用于诊断，
不承诺幂等、exactly-once 输出或跨 boot 恢复。ActiveExecutions/Stop 继续由同 manager 管理现存句柄。

不同 session 并发不新增全局执行锁。Shell manager 持续复用 async channel/stub；同步 Skill checker
仍每次 with 创建并关闭短 channel，不新增生命周期 owner。RPC 取消不等于外部效果已经撤销。

现有实现没有 PTY resize 入口；本次只保持现有 PTY 输入、输出、stop，不能宣称 SH-003 的 resize
已通过验收。该既有差距在 NOW 独立记录。

## 5. 生成、发布与恢复

固定生成器为 `grpcio-tools==1.78.0`。提交原始 `host_bridge_pb2.py`、`host_bridge_pb2.pyi`、
`host_bridge_pb2_grpc.py`，不手工编辑生成物。开发命令：

```bash
uv run --isolated --no-project --with grpcio-tools==1.78.0 python scripts/generate_host_bridge_protocol.py
uv run --isolated --no-project --with grpcio-tools==1.78.0 python scripts/generate_host_bridge_protocol.py --check
```

CI 在独立临时 venv 安装同一生成器后运行同一脚本的 `--check`。运行环境不依赖生成器。
最低 grpcio 1.78.0 / protobuf 6.33.0 和正式锁定版本均须通过 import 与 UDS smoke。

service package 是唯一协议 major owner：`akashic.host.v2`。V1 route 返回 UNIMPLEMENTED，
没有 V1 adapter、自动协商或 fallback。按 RUN-015 在维护窗口内成对切换同 commit Core/Bridge；
软件恢复使用上一套成对 release，保留原清理、预检和 readiness 事务。本任务不部署。

本任务不修改正式持久状态。执行诊断日志正常追加；其原位更新、终态移除、省略保留和清理 owner
均沿用持久化状态地图。本地实现恢复点为 `backup/host-bridge-v1-20260905`，测试只写一次性目录。

## 6. 验收

1. 复用已有 Bridge、Shell、File、boot cleanup 和 owner 测试；新增真实 UDS presence/二进制/认证测试。
2. 用 server admission barrier 确定性取消已登记 Exec；确认同一 manager 仅一个 execution，
   可以 WriteStdin 续接、Stop 收回，不能用 sleep 猜测 admission。
3. 固定命令、shell/login、预算、日志和并发，对 Local/V1/V2 分别测冷/热连接、空命令、4KiB、
   40KB、1MiB 输出、1/8/32 并发、PTY；记录 p50/p95，不把编码成本当端到端速度。
4. 运行生成一致性、类型检查、现有 Python/Web 回归和 change-impact Gate，再由 Terra xhigh
   对完整实现 diff 和证据做独立审查。未执行/环境失败项保持未验证。
