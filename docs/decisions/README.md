# 决策记录

这个目录保存 Akashic Agent 已经作出的重要工程决策和后续勘误。新会话先按任务关键词查找相关记录，不需要一次读完全部文件。

## 索引

| ID | 状态 | 主题 | 关联条款 |
|---|---|---|---|
| [0001](0001-project-workbook-is-shared-reality.md) | accepted | 项目工作手册是协作共享现实 | WBK-001～WBK-006、COM-001～COM-004 |
| [0002](0002-context-reduction-is-a-nondestructive-projection.md) | accepted | 上下文缩减是非破坏性投影 | CTX-001～CTX-005、SES-003 |
| [0003](0003-core-capability-ownership-is-semantic.md) | accepted | 核心能力归属由权威语义决定 | MOB-001、GOV-001～GOV-005 |
| [0004](0004-cross-repository-evidence-is-an-immutable-combination.md) | accepted | 跨仓库证据绑定不可变组合 | GOV-005、MOB-002～MOB-004、TST-006～TST-008 |
| [0005](0005-git-cursor-drives-one-shot-migrations.md) | accepted | Git cursor 驱动一次性兼容迁移 | MIG-001、MIG-002、WSP-003、BAK-001 |
| [0006](0006-akasha-v2-is-the-canonical-explicit-memory-engine.md) | accepted | Akasha V2 是显式记忆的唯一算法实现 | MEM-009、SES-003、GOV-005、TST-002、TST-005 |
| [0007](0007-mobile-plugin-control-and-data-planes-are-explicit.md) | accepted | 移动插件控制面与查询数据面显式分离 | MOB-001、MOB-003、MOB-006、PLG-003、PLG-011、TST-006～TST-008 |
| [0008](0008-plugin-runtime-publishes-only-committed-snapshots.md) | accepted | 插件运行时只发布已提交快照 | PLG-001～PLG-008、GOV-005、TST-006～TST-008 |
| [0009](0009-akasha-mobile-recall-preserves-semantic-lanes.md) | accepted | Akasha 移动卡片完整保留有界召回 lane | MOB-006、PLG-011、TST-006～TST-008 |

## 新增规则

1. 使用四位递增编号和短英文文件名。
2. 写明状态、日期、背景、决定、理由、影响、验收和关联条款。
3. 旧决定被推翻时保留原文件，新记录声明 `supersedes`，旧记录补 `superseded by`。
4. 没有形成选择的讨论不进入这里；未完成动作写入 `NOW.md`。
