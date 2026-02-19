# 内置技能

本目录包含扩展 akasic-bot 能力的内置技能。

## 技能格式

每个技能是一个目录，包含 `SKILL.md` 文件：
- YAML frontmatter（name、description、metadata）
- 供 agent 阅读的 Markdown 指令正文

### metadata 格式

```yaml
metadata: {"akasic": {"always": false, "requires": {"bins": ["curl"], "env": []}}}
```

- `always`：为 `true` 时每轮对话都注入完整正文
- `requires.bins`：运行所需的 CLI 工具（未安装则标记为不可用）
- `requires.env`：运行所需的环境变量

## 可用技能

| 技能 | 描述 |
|------|------|
| `weather` | 通过 wttr.in 查询天气和预报 |
| `summarize` | 总结 URL、文件和 YouTube 视频 |
| `skill-creater` | 创建新技能 |
