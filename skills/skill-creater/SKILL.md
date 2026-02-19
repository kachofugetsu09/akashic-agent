---
name: skill-creater
description: 创建或改写 akasic-bot 技能（SKILL.md）。当用户要求新建技能、适配现有技能到当前格式、或修改技能内容时使用。
---

# Skill 创建指南

## 技能目录结构

```
~/.akasic/workspace/skills/{skill-name}/   ← 用户自定义技能（优先）
  SKILL.md                                  ← 必须
  scripts/                                  ← 可选，放辅助脚本
  references/                               ← 可选，放参考文档

项目/skills/{skill-name}/                  ← 内置技能
  SKILL.md
```

用 `write_file` 或 `edit_file` 直接创建/修改文件，无需任何脚手架脚本。

## SKILL.md 格式

```markdown
---
name: skill-name
description: 一句话说明技能功能 + 触发场景。触发词：词1, 词2, 词3
metadata: {"akasic": {"always": false, "requires": {"bins": ["curl"], "env": ["API_KEY"]}}}
---

# 技能标题

正文指令...
```

### frontmatter 字段说明

- `name`：技能目录名（小写，连字符）
- `description`：**最重要的字段**。agent 靠它判断何时触发此技能，务必包含功能描述和触发词
- `metadata`：运行配置
  - `always`：`true` = 每轮对话都注入完整正文；`false` = 按需加载（默认）
  - `requires.bins`：所需 CLI 工具，缺失时技能标记为不可用
  - `requires.env`：所需环境变量，缺失时技能标记为不可用

## 写作原则

**简洁优先**：SKILL.md 会被注入到 context，每个 token 都有成本。

- 只写 agent 不已知的信息
- 用示例命令代替大段文字解释
- 把冗长的参考内容放到 `references/` 子文件，在 SKILL.md 里注明"详见 references/xxx.md"
- 正文控制在 100 行以内

**触发词要全**：中英文都写，覆盖用户可能说的各种表达方式。

## 创建流程

1. 确认技能名称和保存位置（workspace 还是内置）
2. 用 `write_file` 写入 SKILL.md
3. 如有辅助脚本，写入 `scripts/` 目录
4. 验证：`requires.bins` 里的命令是否正确

## 改写已有技能

改写时重点检查：
- frontmatter 是否有 `metadata` 字段（旧技能可能缺失）
- `requires.bins` 是否列全了依赖
- 正文是否有指向不存在脚本的引用（如 `init_skill.py`）
- 安装命令是否适配当前系统（Linux 用 `uv`/`pacman`/`apt`，而非 `brew`）
