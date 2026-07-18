# 变更影响分析与跨仓库契约 Gate

日期：2026-07-16
分支：feature/projectneed
基线：origin/main@6a0616c8
状态：G1 公开 Gate 与 private plan selection 已实施；G2 确定性 provider 执行、Feed freshness 与外部状态待实施

## 1. 结论

Akashic Agent 使用一个统一 Gate 收口代码变更：实现者只运行一个入口，Gate 根据 Git diff 和公开能力索引选择确定性 Docker 场景，再由 private runtime 把受影响能力映射到真实外置 provider。无法解释的可执行代码改动不得静默跳过，必须回退全量场景并使影响分析失败，直到补齐映射。

Gate 启用前先对现有仓库执行一次审计式 `init`：完整盘点代码、状态、需求、测试和 provider，要求 P0 语义零缺口；非 P0 现存缺口经维护者确认后进入只减不增的 coverage baseline。后续改动一旦触碰已有缺口，就必须补 oracle，不能继续沿用基线豁免。

```text
Git diff
   │
   ▼
公开能力索引
   │
   ├── 公开确定性场景
   ├── private provider 映射
   └── 未知改动 → 全量 + fail-loud
   │
   ▼
一次性 docker/debug sandbox
   │
   ├── 全新 workspace
   ├── 全新 plugin home
   ├── 真实主程序与插件进程
   └── 声明式测试输入
   │
   ▼
统一 Gate 报告与跨仓库验证记录
```

普通 MCP `tools/call` 的完成契约固定为：调用成功返回时，承诺的操作已经完成且结果可观察。插件内部使用同步函数、异步函数、缓存或后台刷新属于实现细节。只表示“已接收”或“后台任务已启动”的操作不得伪装成普通成功；真正的延迟任务以后通过显式 MCP Tasks 契约开放。

## 2. 背景与现有缺口

仓库已经拥有三块可复用基础：

1. `.github/workflows/ci.yml` 已运行 `programmatic_control_probe.py`、`workspace_mcp_reload_probe.py` 和 `restart_probe.py`。
2. `docker/debug` 的主要 Gate 已使用唯一 `/tmp` sandbox、只读源码挂载、独立 Compose project、结果报告和残留资源审计。
3. `private_runtime` 已按“主仓库消费边界指纹 × provider commit”执行 `audit/verify`，并运行 provider import、native tests 和真实只读 MCP fetch。

当前仍有五个缺口：

1. 公开 CI Gate 与私有跨仓库 Gate 是两个独立入口，没有由同一份 diff 选择。
2. 能力组的路径定义位于 private runtime，公开项目需求、Docker 场景和 provider 映射没有共同索引。
3. `verify` 接受任意 `--config`、`--workspace` 和已安装缓存；文档示例直接使用正式 workspace，不满足测试专用隔离要求。
4. 现有 `live_mcp` 只证明真实调用返回合法 payload，不能证明刷新完成、快照新鲜或远端状态已经可观察。
5. 验证记录只保存 `checks` 名称，没有保存实际场景、Gate 环境版本和报告摘要。

Feed MCP 事故已经证明接口和 payload 可以保持合法，但后台刷新停止后仍持续返回旧 SQLite 数据。因此 Gate 必须保护时间语义和状态变化，不能只保护函数签名。

## 3. 设计目标

### 3.1 必须实现

- 实现者和 coding agent 只有一个本地收口入口。
- 测试选择由版本控制内的索引决定，不由 agent 临时猜测。
- 公开能力分组、需求条款和公开场景只有一个权威定义。
- private runtime 只保存私有 provider 清单以及 provider 到公开能力分组的映射。
- 每次 Gate 创建语义干净、测试专用的一次性 workspace 和 plugin home。
- 普通 PR 不读取正式 workspace、正式插件缓存、正式配置或正式凭据。
- Gate 报告解释改了什么、为何选择这些场景、实际观察了什么以及验证的是哪些源码。
- 任何未知可执行改动、缺失 provider、缺失场景或不完整证据都 fail-loud。
- 相关主仓库边界或 provider commit 变化后，旧验证记录立即失效。
- 项目需求中的受保护语义可以追溯到能力和场景。
- 现有仓库通过一次性 `init` 建立覆盖基线，初始化后不能自动覆盖人工合同。
- P0 语义在基线成立前必须全部拥有独立 oracle；非 P0 现存缺口只允许收窄。

### 3.2 不在第一版实现

- 不通过 LLM 推断改动影响。
- 不自动修改 `projectneed.md` 或自动生成业务 oracle。
- 不把私有 provider 清单、凭据或 live 报告提交到公开主仓库。
- 不在 PR 中向不受信任代码提供正式或 live sandbox 凭据。
- 不实现 MCP Tasks；第一版只禁止普通调用返回未完成状态。
- 不替换现有 pytest、pyright 和 Docker 探针；统一入口编排并复用它们。
- 不把插件移动到主仓库或改成 monorepo。

## 4. 关联不变量

Gate 首批关联以下长期条款：

| 条款 | Gate 责任 |
|---|---|
| `OBJ-003` | 重构不能在未批准时改变外部语义 |
| `GOV-001` | 输入 change intent；声明语义变化和受保护状态 |
| `GOV-002` | 规格变化与实现变化分别批准 |
| `GOV-003` | 以 base/candidate diff 作为影响分析输入 |
| `PLG-001` | 候选插件只在隔离 generation 和隔离状态中运行 |
| `PLG-004` | 插件发布对外观察原子 |
| `PLG-009` | Skill 和 MCP 通过插件安装发布 |
| `WSP-001` | workspace 内每项可写状态有明确 owner |
| `WSP-004` | 代码 worktree 与运行数据 workspace 严格分离 |
| `TST-001` | 高风险语义使用独立 oracle |
| `TST-002` | 核对完整状态、write set 和外部调用 |
| `TST-003` | 用已知错误证明 Gate 真能失败 |
| `TST-004` | 高风险 refactor 支持 base/candidate 差分回放 |
| `TST-005` | 恢复能力必须在隔离 workspace 实际演练 |

能力索引引用稳定条款 ID，不复制需求正文。条款语义变化仍按 `projectneed.md` 第 14 节先审批规格和决策记录。

## 5. 权威文件与所有权

```text
主仓库
├── docs/projectneed.md
│   └── 长期语义与条款 ID
├── tests_scenarios/contracts/impact.toml
│   ├── 公开能力分组
│   ├── 主仓库路径触发器
│   ├── 条款 ID
│   └── 公开场景 ID
├── tests_scenarios/contracts/state-contracts.toml
│   └── 持久状态、owner、增改减协议与受保护 write set
├── tests_scenarios/contracts/scenarios.toml
│   └── 场景 ID、oracle、环境和观察项
├── tests_scenarios/contracts/coverage-baseline.json
│   └── 经维护者确认的非 P0 现存缺口
├── tests_scenarios/contracts/scenarios/
│   └── 黑盒场景与 oracle
├── docker/debug/gate.py
│   └── 唯一本地入口、选择、编排、报告与 cleanup
└── docker/debug/reports/change-gate/
    └── 忽略版本控制的运行证据

private_runtime
├── cross_repo_contracts.toml
│   ├── provider canonical source
│   ├── provider → 公开能力分组
│   ├── provider native tests
│   └── live 能力要求
└── cross_repo_verification.json
    └── 已验证组合与 Gate 证据摘要
```

公开能力分组是路径规则的唯一真相。`private_runtime/cross_repo_contracts.toml` 不再重复 `[groups.*].paths`，只引用主仓库已声明的 group ID。这样主仓库和私有目录不能分别维护两份路径集合。

### 5.1 仓库初始化与缺口棘轮

初始化入口：

```bash
python docker/debug/gate.py init --base origin/main
```

`init` 只在尚无 coverage baseline 时运行；baseline 已存在时再次运行必须失败并提示使用 `audit`。它分离机器发现与人工合同：

```text
机器发现
├── tracked executable files
├── import / direct call edges
├── SQLite table access 与 SQL verb
├── workspace 文件读写
├── HTTP/MCP/消息/子进程边界
├── 插件与 lifecycle 动态注册
└── pytest/Docker scenario inventory
          │
          ▼
人工确认
├── requirement ID
├── capability owner
├── protected state
├── depends_on
├── semantic oracle
└── accepted non-P0 gap
```

机器输出 `inventory.json` 和建议映射，不直接写入或覆盖 `impact.toml`、`state-contracts.toml` 和 `scenarios.toml`。维护者确认后的三份 TOML 才是 Gate 权威输入。

状态合同示意：

```toml
[states.session_messages]
requirements = ["SES-002", "SES-003", "SES-005"]
owner = "session.store"
normal_change = "insert_only"
destructive_owner = "explicit_user_data_management"
writers = ["session_persistence"]
consumers = ["context_projection", "akasha_rebuild", "history_api"]
oracles = ["session_append_only", "context_retry_is_nondestructive"]
```

初始化分类固定为：

- `covered`：存在可执行、独立且通过的 oracle。
- `p0_blocked`：P0 能力缺少 oracle、需求或观察器；baseline 不得成立。
- `accepted_gap`：非 P0 现存缺口，必须有稳定 gap ID、能力、owner、理由和关联条款。
- `unmapped`：代码、状态或 provider 没有 owner；baseline 不得成立。
- `ambiguous_contract`：已有实现存在多种合理语义但项目没有决定；P0 等同阻塞，非 P0 也不能自动登记为已覆盖。

P0 判据不由扫描器猜测。第一版把以下范围作为 P0：

- 权威用户事实的减少、覆盖、序号和恢复，包括 `sessions.db/messages`、`message_embeddings`、长期记忆与 plugin-data。
- 上下文、检索、派生索引或容量控制可能反向修改权威事实的路径。
- 插件候选发布、generation 原子切换和跨仓库 MCP 完成/freshness 合同。
- 外部不可逆发送、远端写入和明确 destructive command。
- 备份、恢复和 rollback 对权威状态的可用性声明。

非 P0 缺口使用棘轮规则：

1. 新增缺口或扩大现有 gap 的 paths、state、provider、side effects 时 Gate 失败。
2. 删除测试、条款或映射不能使 gap 看起来消失；必须通过决策记录关闭或迁移。
3. diff 命中某个 `accepted_gap` 的代码、条款、状态 owner 或 provider 时，该 gap 本次升级为 `blocked`，必须补 oracle 或先取得明确规格批准。
4. 与当前 diff 无关的 accepted gaps 可以留在 baseline，但报告必须逐项列出，不能折叠成一个数量。
5. nightly 全量发现增量漏测时，新增依赖边并记录为 Gate 缺陷；不能只重跑到通过。

`coverage-baseline.json` 只保存确认结果、catalog digest 和 gap 元数据，不保存扫描器的推断过程。任何 baseline 更新都进入普通 diff 和评审。

## 6. 公开能力索引

`tests_scenarios/contracts/impact.toml` 使用显式 schema。示意：

```toml
version = 1

[defaults]
baseline_scenarios = ["workspace_bootstrap"]
executable_suffixes = [".py", ".ts", ".tsx", ".js", ".toml", ".sql", ".sh"]

[groups.mcp]
requirements = ["PLG-004", "PLG-008", "PLG-009"]
paths = [
  "agent/mcp/**/*.py",
  "agent/plugins/mcp_host.py",
  "agent/tools/registry.py",
  "bootstrap/toolsets/mcp.py",
  "proactive_v2/mcp_sources.py",
]
scenarios = [
  "mcp_call_finality",
  "mcp_failure_visibility",
]

[groups.memory]
requirements = ["SES-005", "MEM-001", "MEM-002", "WSP-004"]
paths = [
  "core/memory/**/*.py",
  "memory2/**/*.py",
  "session/**/*.py",
]
scenarios = [
  "session_append_only",
  "workspace_restart",
]
```

第一版迁移现有 private catalog 的 `plugin`、`lifecycle`、`events`、`channel`、`mcp`、`proactive`、`memory` 和 `jobs` 分组，不在迁移中重新划分架构。

### 6.1 索引校验

`gate.py plan` 在运行任何场景前校验：

1. 每个 path glob 至少匹配一个版本控制内文件。
2. 每个 requirement ID 存在于 `docs/projectneed.md`。
3. 每个 scenario ID 有唯一场景定义。
4. 每个 private provider 引用的 group ID 存在。
5. 同一 scenario ID 不得由两个互不相干实现重新定义 oracle。

任一条件不满足即退出非零，不使用默认值或模糊匹配继续。

## 7. 测试影响选择算法

入口：

```bash
python docker/debug/gate.py init --base origin/main
python docker/debug/gate.py audit
python docker/debug/gate.py plan --base origin/main
python docker/debug/gate.py run --base origin/main
```

算法固定如下：

1. 取得 `base...working-tree` 的 tracked diff，并纳入非 ignored untracked 文件。
2. 记录 HEAD、dirty status、source digest 和 base commit。
3. 用公开 path glob 计算命中的能力分组。
4. 选择 baseline scenarios 与所有命中分组的 scenarios，集合去重。
5. 公开计划只标记 `private_gate_required` 和受影响 group，不读取或输出私有 provider 清单；private companion 再按 group 选择 provider。
6. 第一版不使用影响分析缩减 pytest、pyright 和现有 required Docker jobs；它只选择新增的语义场景和 private provider。现有回归测试继续全量运行，避免迁移 Gate 时先降低覆盖面。
7. 若命中 accepted gap，选择关联测试并标记 `BASELINE_GAP_TOUCHED`；最终 Gate 失败，直到本次补齐 oracle 或完成已批准的规格变更。
8. 若可执行文件没有命中任何分组，标记 `UNMAPPED_CHANGE`，选择全部公开场景；private companion 选择全部 provider。场景仍执行以收集证据，但最终 Gate 失败，要求维护索引。
9. 文档、图片等明确非执行文件可以返回 `not_affected`，但 `projectneed.md`、能力索引、Gate、场景和 CI 文件本身变化时必须运行索引自检与全部 Gate 元测试。
10. nightly 和 release 不做增量选择，固定运行全量矩阵。

选择结果在执行前打印并写入报告，不允许只输出“已选择 N 个测试”。

```text
Changed:
  agent/mcp/client.py

Affected groups:
  mcp
  proactive

Selected public scenarios:
  workspace_bootstrap
  mcp_call_finality
  mcp_failure_visibility

Private gate required:
  true

Reason:
  agent/mcp/client.py matched groups.mcp
```

private companion 使用同一份 plan digest 生成只保存在私有环境中的 provider plan。公开 stdout、GitHub artifact 和 PR comment 不出现 provider inventory。

## 8. 语义干净的 sandbox

`gate.py run` 每次创建：

```text
/tmp/akashic-change-gate-<run-id>/
├── workspace/
├── plugin-home/
├── home/
├── config.toml
├── fixtures/
└── reports/
```

硬约束：

- 不接受调用者传入 `--workspace`、`--config` 或安装缓存路径作为 Gate 数据源。
- 不挂载宿主机 HOME、`~/.akashic/workspace`、`~/.akashic-plugin/cache` 或用户配置。
- 主仓库和 canonical provider source 只读挂载。
- root filesystem 只读；只有 `/sandbox` 和 tmpfs `/tmp` 可写。
- `sessions.db` 从空库开始。需要历史的场景通过正式公开入口写入测试事件，不复制现有 DB。
- Akasha 派生库从本次场景的 session 数据和已有 embedding 重建，不复制正式派生索引，也不调用 LLM 补齐缺失输入。
- plugin home 从空目录开始，通过正式插件安装流程安装所选 provider 的指定 commit。
- 一个场景内允许复用本次 workspace 完成 restart/recovery 验证；不同场景默认各自隔离，除非场景定义显式属于同一 ordered suite。
- cleanup 使用 Compose project label 检查容器、网络和 volume 无残留。
- 运行报告复制到仓库 ignored 报告目录后才销毁 sandbox。

若任何真实路径解析到 sandbox 之外，Gate 在启动前失败。不得用 catch-and-fallback 改用正式路径。

## 9. 场景合同

每个场景声明：

```toml
id = "feed_continuous_refresh"
requirements = ["PLG-004"]
groups = ["mcp", "proactive"]
environment = "deterministic"
timeout_seconds = 30
```

每个黑盒场景按相同阶段组织：

1. **Given**：从空 workspace 建立声明式初始状态。
2. **When**：只通过正式入口执行用户动作、MCP 调用、插件安装或生命周期变化。
3. **Then/Return**：核对返回结果、错误分类和终态。
4. **Then/State**：核对完整持久快照、write set、文件变化和事件。
5. **Then/External**：核对远端可观察结果、freshness、cursor 或 read-after-write。
6. **Then/Restart**：需要持久语义时重启并再次读取。
7. **Cleanup**：销毁测试外部状态和 sandbox；cleanup 失败使 Gate 失败。

第一版不强制所有场景都有六种观察，但必须显式声明未适用项，不能因为没有观察器而默认为通过。

## 10. MCP 完成语义场景

### 10.1 普通调用

`mcp_call_finality` 固定以下契约：

```text
await tools/call 返回 success
            │
            ▼
承诺的状态现在已经可以通过正式读取接口观察
```

仅把后台任务排队、仅刷新内存标志或仅返回 accepted 都不满足普通成功。

### 10.2 Feed freshness

确定性 PR 场景使用同一 Docker 私网中的受控真实协议数据源：

```text
发布 V1
  → 启动真实主程序和真实 Feed MCP
  → 首次读取看见 V1
  → 数据源推进 V2
  → 等待插件声明的刷新边界
  → 通过主仓库 MCP 调用读取
  → 返回、插件 SQLite、cursor 均证明 V2 可见
  → 重启后 V2 仍可见且不重复
```

受控数据源必须实现 Feed 实际使用的网络协议和可变状态，不能把预期结果直接注入 MCP 返回路径。它用于确定性验证生命周期和完成语义，不替代 live sandbox。

### 10.3 延迟任务

第一版遇到 provider 声明“普通调用返回后仍需后台完成”时直接失败，并要求：

1. 改成等待最终完成后再返回；或
2. 提交独立设计引入 MCP Tasks 能力协商、任务终态和 `tasks/result`。

## 11. 三层 Gate

### G1 公开确定性 Gate

运行环境不包含私有源码和凭据：

- pyright、pytest 和 schema check。
- 现有 programmatic control smoke/failure matrix。
- 现有 workspace MCP lifecycle 与 restart soak。
- 公开能力索引自检。
- 公开场景中的空 workspace、持久化、失败和协议合同。

G1 是主仓库 PR 的 required check。

### G2 私有跨仓库 Gate

private runtime 根据公开 group ID 选择真实 provider，并在一次性 Docker sandbox 中执行。`requires_live_mcp` 在迁移后拆成确定性场景要求与可选 `live_profile`：G2 不再把真实外部网络调用当作确定性组合成立的前提，G3 单独拥有 live 验证。

1. canonical provider 工作树、commit 和 remote 核对。
2. 当前主仓库 API 下的真实 import。
3. provider 声明的 native tests。
4. 通过正式插件安装流程安装到空 plugin home。
5. 启动真实主程序和插件进程。
6. 运行命中能力的确定性黑盒场景。
7. 写入 provider commit、consumer fingerprint、场景和报告摘要。

公开 PR 不读取私有源码。G2 由 private repository 或本地维护者环境运行，并向主仓库 PR 发布单一外部状态 `private-contract-gate`。状态包含 `passed`、`failed`、`not_affected`，不能使用含义不明的 `skipped`。

不受信任 PR 不获得 private source、live 凭据或可上传私有报告的 token。若无法建立安全的外部状态回传，第一阶段保持本地维护者 Gate，并在 PR 描述附带验证记录摘要；不得为了自动化把秘密交给 PR 代码。

### G3 Live sandbox Gate

只在受信任 main commit、nightly 或 release promotion 上运行：

- 使用专用测试账户、只读 token 或可清理 sandbox 资源。
- 不使用正式用户 workspace、正式 sessions 或正式插件缓存。
- 读取型 provider 验证 freshness、cursor 推进和最后成功时间。
- 写入型 provider 使用测试租户执行 write/read/cleanup；无法安全清理时不在自动 Gate 中写入。
- 失败时阻止发布、插件重新安装或生产 promotion，并保留明确告警。

G3 解决两个 Git 仓库都未改变但外部服务、权限、网络或协议漂移的问题；它不能替代 G1/G2。

## 12. 统一报告

公开报告目录：

```text
docker/debug/reports/change-gate/<run-id>/
├── plan.json
├── gate.json
├── public/
├── providers/
└── cleanup.json
```

`gate.json` 至少包含：

```json
{
  "version": 1,
  "status": "passed",
  "base": "<commit>",
  "head": "<commit>",
  "dirtyStatus": [],
  "sourceDigest": "<sha256>",
  "impactCatalogDigest": "<sha256>",
  "affectedGroups": ["mcp"],
  "selectedScenarios": ["mcp_call_finality"],
  "privateGateRequired": true,
  "checks": [],
  "residualResources": {
    "containers": [],
    "networks": [],
    "volumes": []
  }
}
```

允许的总状态只有：

- `passed`：所有必需观察和 cleanup 完成。
- `failed`：至少一个场景或证据失败。
- `not_affected`：只有经过索引解释的非影响改动。
- `unmapped_change`：执行全量后仍以失败结束，要求维护索引。
- `blocked`：缺少 private source、测试凭据或运行前置条件；对 PR 等同失败。
- `baseline_gap_touched`：改动触碰初始化时登记的非 P0 缺口但没有补 oracle；对 PR 等同失败。

公开报告不得包含 token、用户正文、正式 workspace 路径、私有 provider ID 或私有源码内容。provider 选择和 provider 级证据写入 private runtime 自己的报告；主仓库只接收绑定同一 plan/source digest 的外部状态。

## 13. 跨仓库验证记录

`cross_repo_verification.json` 的 provider 记录升级为：

```json
{
  "provider_commit": "<sha>",
  "consumer_sha256": "<sha256>",
  "gate_version": 1,
  "environment": "docker-debug-clean-workspace-v1",
  "scenarios": [
    "mcp_call_finality",
    "feed_continuous_refresh",
    "provider_restart"
  ],
  "checks": [
    "import",
    "provider_tests",
    "deterministic_mcp"
  ],
  "report_sha256": "<sha256>"
}
```

只有 canonical、安装结果、全部必需场景、报告写入和 cleanup 都成功后才能原子替换记录。live 结果单独记录，不把一次 live success 视为永久有效组合。

## 14. 开发、提交与 CI 流程

### 14.1 本地实现者

```text
main 基线 worktree
  → 阅读工作手册
  → 声明 change_type / semantic_delta
  → 修改代码
  → 运行最近的单元测试
  → gate.py plan
  → gate.py run
  → 检查 gate.json
  → 允许提交
```

AGENTS.md 增加两条硬规则：

> 完成代码修改后，必须运行 `python docker/debug/gate.py run --base <目标分支>`。测试场景由 Gate 选择，不得由实现者自行缩减。Gate 未通过、存在未映射改动或缺少必需私有验证时，不得声称完成。

> Gate 必须创建一次性 workspace 和 plugin home；禁止传入、挂载或复制正式 workspace、sessions.db、插件缓存、用户配置和正式凭据。

### 14.2 跨仓库记录提交

受影响的 G2 通过后：

1. 原子写入 private runtime 验证记录。
2. 提交并推送 private runtime。
3. 更新主仓库 `private_runtime` submodule 指针。
4. 提交主仓库改动并创建 PR。
5. 主仓库或 private external check 对当前 PR source digest 再次 audit。

任何后续代码变化导致 consumer fingerprint 或 source digest 变化时，旧记录失效。

### 14.3 主仓库 CI

`.github/workflows/ci.yml` 增加一个始终存在的 `change-impact-gate` job：

1. 运行 `gate.py plan` 并上传 plan。
2. 运行选择后的 G1；不得通过 GitHub job `if` 把 required check 直接标记为 skipped。
3. `not_affected` 也生成完整 gate report。
4. G2 由独立、始终返回结果的 `private-contract-gate` required check 负责；公开 job 不轮询也不持有 private token。未受影响时该 check 明确返回 `not_affected`，不能显示 skipped。
5. 上传公开报告；失败时仍上传证据。

现有 `docker-control-gate` 和 `runtime-extension-gate` 第一阶段保持独立 required jobs，由 `change-impact-gate` 复用其结果或逐步纳入统一入口，避免一次迁移同时重写成熟探针。

### 14.4 Nightly / release

- nightly 固定执行全量 G1/G2，不使用影响裁剪。
- 受信任环境执行 G3。
- 定期把增量选择结果与全量结果比较；若全量发现增量漏测，新增能力映射并保留事故记录。

## 15. 失败语义和豁免

Gate 不提供 `--force-pass`、`--ignore-unmapped` 或“失败继续”的命令。

- 纯文档改动可以由索引确定为 `not_affected`。
- 可执行代码未映射时必须补索引，不能用 PR 文字说明代替。
- diff 触碰 accepted gap 时必须补 oracle 或完成会移除该能力的已批准规格变更；不能继续沿用初始化豁免。
- private provider 缺失或 canonical 工作树不干净时状态为 `blocked`。
- Docker、依赖或测试数据源启动失败时状态为 `failed`，不能降级成只跑单测。
- 紧急合并只能使用仓库管理员已有的 branch protection 管理流程，并在决策记录中留下原因；Gate 自身不实现绕过能力。

## 16. 安全边界

- 公开 PR workflow 不使用 `pull_request_target` 运行候选代码并挂载秘密。
- private source 和 provider 报告只在私有 runner 或维护者本机出现。
- live 凭据只授予受信任 main/nightly/release 工作流。
- Docker 内无宿主 Docker socket；控制器只传入本次 Compose project 必要参数。
- 所有源码只读，临时写入只在 sandbox。
- 外部写操作仅使用测试租户并要求 read-after-write 和 cleanup 证据。
- 报告先脱敏再上传；原始外部响应默认不进入公开 artifact。

## 17. 实施顺序

### Phase 0：仓库审计式 Init

1. 实现 capability/state/scenario schema，以及 `gate.py init/audit` 的发现、校验和报告入口。
2. 盘点全部 tracked executable files、持久状态访问、外部边界、动态插件注册、测试和 Docker probes。
3. 以 `projectneed.md`、`persistence-state-map.md` 和现有 private catalog 为输入，建立 capability、state、scenario 三份人工确认索引。
4. 对全仓执行 owner/mapping audit；任何 `unmapped` 阻止 baseline 成立。
5. 先落地 Session context trim 与 MCP/Feed freshness 两个历史事故 oracle，再逐项补齐其余 P0 oracle；每个 P0 oracle 都必须有已知 mutant 或等价故障注入并稳定失败，任一缺口都阻止 baseline 成立。
6. 非 P0 缺口逐项登记 gap ID、owner、理由和关联条款。
7. 在全新 Docker workspace/plugin home 上运行全部现有 required tests 和已建立场景。
8. 维护者审阅完整报告后才原子写入 `coverage-baseline.json`。

`init` 完成后只能使用 `audit` 发现漂移；不得重新运行 init 覆盖人工合同。

### Phase 1：统一选择与干净 sandbox

1. 在 Phase 0 的 `gate.py` 增加 `plan/run`，按 diff、depends_on、state 和 baseline gap 选择场景。
2. 复用现有 Docker sandbox、source digest 和 cleanup 代码，统一运行报告 schema。
3. 把 private `audit/verify` 的 workspace/config/cache 输入改为 Gate 创建的 sandbox。
4. 加入选择算法、baseline 棘轮和 sandbox 逃逸测试。
5. 保持现有 pytest、pyright 与成熟 Docker jobs 全量运行，不在本阶段削减覆盖。

### Phase 2：跨仓库确定性合同

1. private catalog 改为引用公开 group ID。
2. verification record 增加 Gate 环境、场景和报告摘要。
3. Feed 首先增加 finality、continuous refresh 和 restart 黑盒场景。
4. Steam、Fitbit、Calendar 按各自状态模型补 freshness 或 live-read 合同。
5. 建立 private external status；无法安全自动化时先保留本地维护者 Gate。

### Phase 3：需求追踪和 live promotion

1. 逐步把非 P0 accepted gaps 转成可执行合同并收窄 coverage baseline。
2. 对 `semantic_delta: none` 高风险重构加入 base/candidate 差分回放。
3. nightly 定期复跑 P0 mutant/fault-injection 元测试，防止 oracle 自身退化。
4. 建立受信任 G3 live sandbox 与发布阻断。
5. nightly 比较增量选择和全量结果，维护影响索引。

每个 phase 独立 PR；不得在 Phase 1 同时重写所有现有 Docker 探针。

## 18. 验收标准

### 18.1 选择器

- baseline 已存在时再次运行 `init` 会失败，不覆盖人工合同。
- `audit` 能列出每个 executable file、状态 owner、场景和 provider 的映射状态。
- P0 缺少 oracle 或存在 ambiguous contract 时不能生成 baseline。
- 非 P0 accepted gap 必须逐项记录；新增或扩大 gap 会失败。
- 修改 accepted gap 覆盖的代码会得到 `baseline_gap_touched`，不能沿用旧豁免。
- 修改 `agent/mcp/client.py` 会选择 `mcp` 分组及所有依赖 MCP 的 private providers。
- 修改无关 Markdown 可以得到带理由的 `not_affected`。
- 新增未映射 `.py` 文件会执行全量并以 `unmapped_change` 失败。
- 空 glob、未知条款 ID、未知场景或未知 private group 都 fail-loud。
- nightly 强制全量，不受 diff 影响。

### 18.2 sandbox

- Gate 不接受正式 workspace/config/cache 参数。
- 容器不能读取宿主 HOME 和正式 workspace。
- 每次 run 使用不同 workspace 和 plugin home。
- 一个场景重启后能读取本场景状态，不同场景之间无状态泄漏。
- Gate 完成后容器、网络和 volume 无残留。

### 18.3 跨仓库合同

- 主仓库 MCP 边界变化会使旧 provider 验证失效。
- provider commit 变化会使旧验证失效。
- Feed 只返回合法旧 payload 时，freshness 场景失败。
- Feed 只启动后台刷新便返回 success 时，finality 场景失败。
- provider native tests、真实安装或真实 MCP 进程任一失败都不更新记录。

### 18.4 Oracle

- 已知 `DELETE FROM messages` mutant 被持久化 Gate 拦截。
- 已知“停止 Feed 持续刷新” mutant 被 freshness Gate 拦截。
- 每个进入 baseline 的 P0 oracle 都登记至少一个稳定失败的 mutant 或等价故障注入。
- Gate 报告能从条款 ID 追到能力、场景、provider commit 和观察证据。

### 18.5 流程

- AGENTS.md 只要求实现者运行统一入口，不要求记住 provider 专用命令。
- 主仓库 PR 始终显示 change impact 和 private contract 两个 required check，不出现含义不明的 skipped。
- G2 缺失时 PR 状态明确为 blocked/failed，不能只在文字中说明未验证。
- G3 失败阻止发布或正式插件更新，不伪装成普通 warning。

## 19. 迁移与回滚

迁移期间保留现有探针命令和旧 `audit` 入口，统一 Gate 先编排它们。新旧 verification schema 通过显式 `version` 区分；不兼容旧记录时要求重新 verify，不自动猜测升级。

Phase 1 回滚只移除统一入口和公开索引，不修改正式 runtime。Phase 2 回滚恢复旧 private catalog 读取，但不得恢复使用正式 workspace 的验证示例。任何已经发现的生产路径污染都作为独立安全修复保留。

## 20. 被拒绝的方案

### 20.1 只在主仓库复制插件接口测试

只能保护参数和返回形态，无法观察真实插件生命周期、缓存 freshness 和远端状态。

### 20.2 由 agent 阅读 diff 后自由选择测试

选择不可复现、不可审计，新会话会基于不同假设产生不同测试集合。

### 20.3 每次 PR 无条件运行全部 live provider

速度慢、容易受外部抖动影响，并迫使不受信任 PR 接触 private source 和凭据。

### 20.4 把插件全部迁入 monorepo

可以减少提交组合问题，但不能解决外部服务漂移，也不符合插件独立安装和发布边界。

### 20.5 继续使用长期 `workspace-test`

append-only session、派生索引、插件缓存和旧配置会让测试彼此污染，无法证明候选版本从干净现实启动。

## 21. 完成定义

只有以下条件全部满足，才能说“流程 Gate 已建立”：

1. 实现者可用一个命令获得可解释的测试计划并执行。
2. 一次性 init 完成全仓 owner/mapping 盘点，P0 零缺口，非 P0 缺口进入显式棘轮基线。
3. 公开能力索引成为路径分组唯一真相。
4. 未映射可执行改动、扩大 baseline gap 和触碰未保护 gap 稳定 fail-loud。
5. 所有 Gate 使用一次性测试 workspace/plugin home，正式状态不可达。
6. 受影响 private provider 在真实安装和真实进程中完成确定性合同。
7. 验证记录绑定 consumer fingerprint、provider commit、场景和报告摘要。
8. 主仓库 CI、private external check 和 live promotion 的职责不混淆。
9. 所有 P0 oracle 的 mutant/fault injection 稳定失败，其中包括 Session 删除历史与 Feed freshness 两个已知事故。
10. AGENTS.md、Docker debug 文档、private runtime 文档和 CI 命令与实际入口一致。
11. `NOW.md` 中相应未完成事项在全部验收落地后删除。
