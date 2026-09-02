# 0037 · 插件运行时收敛为 pure v3

- 状态：accepted / implemented
- 日期：2026-08-18
- 关联条款：PLG-001～PLG-014、WSP-001～WSP-005、ERR-001、TST-001～TST-008
- supersedes：[0008](0008-plugin-runtime-publishes-only-committed-snapshots.md) 的 API v2 与 legacy host 选择
- superseded by：无

> 2026-09-02 对账：本决策的 pure-v3 runtime 结论仍有效；下文 E1～E4 是当时的迁移验收计划，已被当前 fleet、Mobile、公共 WebUI 候选 Gate 和发布流程拥有的真实环境验收取代。E1/E2 固定的 API 已随 v2 compatibility 删除，E4 又依赖不存在的 E3 runner，不能继续作为当前合并条件。

## 背景

API v2 曾用 `prepare/activate/retire/terminate` 与固定贡献字段建立第一版原子发布。
API v3 已把组合收敛为 Root Context、Service、Inject、Fiber 和 Effect，并继续由
Core 拥有 artifact、candidate、stable/latest、lease、journal、晋升与回滚。继续保留
两套公开插件 ABI 会让同一能力拥有两个注册、清理和发布路径。

维护者已明确允许本轮产生 breaking change。目标不是兼容任意旧插件，而是在完成
已跟踪 fleet 的 v3 迁移、分组故障演练和数据不变证据后，删除可达的 v2 兼容层。

## 决定

1. 公开插件运行时只接受 `akashic.plugin.toml` 与 `api_version = 3` 模块入口。
   插件通过精确 `apply(ctx, config)` 与 typed Service 投影能力，不再声明 v2
   `Plugin` 子类、lifecycle 或固定贡献方法。
2. 每个领域只保留一个 Core owner。Tool、Channel、Command、MCP、managed process、
   Job、UI、Skill、Dashboard 和被动链路均从 committed Root snapshot 读取。最后一个
   v2 consumer 迁走后立即删除对应 legacy owner，不保留 deprecated alias 或空壳。
3. Computer Use Linux 与 Context Pressure 退出已跟踪 fleet。卸载只移除安装清单与
   能力 cache；既有 `plugin-data` 默认保留，不因代码收敛而物理删除。
4. 代码合并与 hua-home 正式替换分开。同一 clean head 必须通过 fleet source/API compatibility、
   Mobile 与公共 WebUI 候选 Gate；正式 workspace 的备份、真实环境验收、切换和回滚由发布
   流程拥有并仍需单独授权。

```text
外部插件 source
        │ static manifest + api_version=3
        ▼
┌──────────────────────┐
│ Core publication plane│ artifact → candidate → validation → stable
└──────────┬───────────┘
           ▼ committed Root
┌──────────────────────┐
│ typed capability host │ Tool / Channel / Command / MCP / Job / UI / Skill
└──────────────────────┘

```

## 理由

- breaking change 在可控的 fleet 迁移中比永久双轨更容易审计：一份声明、一张 Root、
  一个 publication owner、一套 cleanup 证据。
- 主动、调度和 Dashboard 已使用普通 V3 Service、event 和 generation host；不再保留专用
  V2 admission 或 lifecycle 岛。
- 数据安全不由 ABI 兼容保证。安全来自 candidate workspace 隔离、权威数据只追加/
  明确更新协议、外部效果三态回执、generation lease、journal 和可恢复备份。

## 影响

- 无 static manifest、`api_version != 3` 或还调用 v2 固定方法的外部插件将在 admission
  时 fail-loud，不再被自动包装或跳过。
- `manifest.toml` 只声明独立插件；旧 `[packages]` 组合与 member 展开已删除，并在边界
  fail-loud。
- 历史 v2 测试、lock、Gate、文档与 CI 入口在零 production consumer 后删除。历史决策
  保留并标记 superseded。
- 插件卸载不得级联删除 plugin-data、Session、memory、附件或外部 canonical source。
- 动态发现“不兼容就忽略”不再是降级路径；错误保留为失败证据。

## 验收

- 静态 fleet 清单中每个启用插件都有 exact source commit、manifest 与 v3 module namespace；
  清单与正式安装清单的差异必须显式。
- 扫描 production source、bootstrap、RuntimeSnapshot 和 Manager 不再存在可达 V2 Plugin
  lifecycle、固定贡献 consumer、phase module 注入口或 EventBus-to-V3 类型桥。
- 每个领域完成 candidate discard/promote、old lease drain、Effect/resource cleanup、进程内失败与
  子进程崩溃恢复；不为断电或物理停机扩张本轮范围。
- 同一 clean Core head 运行 fleet source/API compatibility、Mobile 与公共 WebUI；发布流程
  另外用真实部署输入证明 `sessions.db/messages`、memory、plugin-data、artifact 与 pointer
  不发生未授权变化。
- 完成状态必须由测试和 Gate 报告确认；还有 v2 consumer 、blocked scenario 或非同 head
  证据时不得声称 pure-v3 ready。
