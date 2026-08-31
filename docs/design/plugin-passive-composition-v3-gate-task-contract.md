# Citation + Meme 纯 v3 组合 Gate 任务合同

## 1. 目标

用一个可公开复现的跨仓 Gate 证明 Citation 与 Meme 在删除 v2 shell 后，仍能通过
Akashic stable snapshot 的 Composition Root 保持回复协议、持久化 metadata、媒体、Skill、
Dashboard 与生命周期回收能力。

本任务不删除 Core v2 分支，也不把临时验证 workspace 当成正式运行数据。Core v2 的物理
删除仍由最终迁移清单和后续独立 PR 管理。

## 2. Owner 与固定输入

- Core 是 snapshot、Root、Service、Fiber、Skill 与 Dashboard publication owner；
- Citation 拥有 `citation.protocol` Service 和引用 metadata 解释；
- Meme 通过 required `inject` 等待 Citation，拥有表情目录解释和回复媒体装饰；
- `plugin-contracts` 拥有跨仓静态 API v3 合同；
- 锁内的 Core 协议提交、contract 提交和两个插件提交全部使用 40 位 SHA。

```text
fresh exact checkout
       │
       ▼
public v3 contract ──► PluginManager.load_all ──► stable snapshot lease
                                                   │
                    ┌──────────────────────────────┼──────────────┐
                    ▼                              ▼              ▼
             lifecycle serial                Skill catalog   Dashboard host
                    │                              │              │
                    └──────────────┬───────────────┴──────────────┘
                                   ▼
                         terminate + zero residue
```

## 3. 行为合同

1. 正式 Root 的 listener 注册顺序必须为 Citation prompt、Meme prompt、Citation preprocess、
   Meme preprocess、Citation cleanup；顺序来自每个 typed event 内的注册顺序，不引入 phase DAG。
2. Citation 最后提供 `citation.protocol`，Meme required Fiber 只在该 Service 存在时 active。
3. 输入 `答复正文\n§cited:[mem_1]§ <meme:shy>` 后，最终正文为 `答复正文`，assistant
   metadata 为 `cited_memory_ids=[mem_1]`，media 指向 fixture，`meme_tag=shy`。
4. stable snapshot 必须投影 `meme-manage` Skill；Meme Dashboard 必须从同一正式 Root 的
   `workspace/memes` 读取 `shy` 类别，且 `validation=false`。
5. 插件 load、事件执行和 Dashboard 读取前后，`workspace/memes` 的内容摘要必须相同。
   Manager 允许在临时 workspace 下建立 Core-owned `plugin-data` 目录，但报告必须列出其条目。
6. `terminate_all()` 后，原正式 Root 的 listener、Effect、Service、Dashboard binding 与动态
   Dashboard module 必须全部清空。

## 4. 失败与隔离

- 浮动 ref、插件顺序变化、contract 发现 v2 类、拓扑或结果漂移均 fail-loud；
- Gate 不使用 skip、mock success 或正式 workspace；
- fresh checkout 必须证明 resolved SHA、tree 与 clean status；
- CI 以完整 Git 历史读取固定 Core protocol blob，并强制当前 Core worktree clean；
- 报告记录 Gate version、profile/hash、Core head/tree、lock hash、协议 blob/hash、跨仓 source
  identity、行为结果、workspace 摘要与 cleanup 结果。

## 5. 验收

```bash
python docker/debug/plugin_passive_composition_v3_gate.py --require-clean-core
python -m basedpyright --level error docker/debug/plugin_passive_composition_v3_gate.py
git diff --check
```

真实 Gate 的 `gate.json` 必须 `status=passed`；命令失败或报告缺失都不能称为迁移成功。
