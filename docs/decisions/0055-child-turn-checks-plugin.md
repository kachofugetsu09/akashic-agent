# 0055 Child Turn 检查插件候选

- 状态：accepted
- 日期：2026-09-02
- 关联条款：PLG-010、PLG-013、PLG-014、RUN-007、ERR-001

## 背景

现有插件安装链已经有不可变 artifact、stable/latest、候选 Root、RuntimeSnapshot lease、attached child、父 Turn 提交和 reload journal。缺口是 Core 把“候选被检查”收窄成“候选拥有的 Tool 或 Skill 被成功使用”。没有 Tool 或 Skill 的 UI、Channel、模型和后台插件无法通过同一条自更新链。

一次重构曾为这个缺口新增 RootSwitch、ServiceHold、ServiceCall、TaskControl 和 SwitchInput。当前 Core、仓库插件、外部插件源码与 hua-home 已安装插件均没有这些接口的业务调用者。它们建立了第二套切换和保活模型，却没有替代现有的 generation、lease 和发布 owner。

## 决定

保留现有插件安装链，不增加特权插件或新的通用原子。

Core 给 attached child 绑定 parent、plugin、generation 和 source。绑定完全匹配且 child 正常完成时，Core 只记录“这个 exact candidate 已被检查”。parent 正常完成表示本次检查满足业务目标；parent 失败、child 失败、没有 child、身份漂移或显式 revert 都丢弃候选。

检查内容由 Agent 使用普通插件能力完成。Core 不识别 Tool、Skill、UI、Channel 或其他插件类别，也不检查某种业务结果。

```text
旧 stable ── parent Turn ── 安装 candidate
                    │
                    └── attached child ── exact candidate ── 正常完成
                                                     │
parent 正常完成 ──────────────────────────────────────┘
                    │
                    └── 原子切换 stable 指针
```

进程在 stable 指针切换前退出，重启后仍使用旧 stable；在指针切换后退出，重启后使用新 stable。Core 不为进程崩溃再建立“复活旧世界”的控制面。

## 理由

- generation/source binding 已经证明 child 使用哪个候选，parent 已经拥有业务判断。
- 插件类别是正交变化轴，不应进入 Core 的提交条件。
- 复用已有 snapshot、lease、pointer 和 journal，失败边界只有一套 owner。
- 与 DeepSeek Harness 的做法一致：运行壳只固定身份、生命周期和提交边界，具体检查由普通能力组合完成。

## 影响

- 没有 Tool 或 Skill 的插件也可以沿同一条在线自更新链检查。
- 删除 `candidate_child_evidence` 以及未被采用的 RootSwitch、ServiceHold、ServiceCall、TaskControl 和 SwitchInput 方案，不保留兼容壳。
- 现有 stable/latest、候选隔离、parent Turn 发布、journal 恢复和 operator trusted batch 合同不变。

## 验收

- exact attached child 正常完成且 parent 正常完成时提交候选。
- 缺少 child、child/parent 非正常终结或身份不匹配时丢弃候选。
- 候选检查不读取 TurnItem，也不要求 Tool/Skill provenance。
- 插件热重载、候选隔离、lease 排空与 journal 恢复测试通过。
