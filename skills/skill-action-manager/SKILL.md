---
name: skill-action-manager
description: 管理后台定期执行的 skill actions。当用户说"有空的时候帮我做某件事"、"空闲时自动执行"、"后台定期运行"、"闲下来帮我读/处理/更新"等表达时，调用 skill_action_register 工具注册任务。当用户说停止/取消某个后台任务时，调用 skill_action_unregister。
---

# Skill Action 管理

skill action 是在我主动消息判定为 idle（决定不发消息）时自动执行的后台任务。
与 `schedule`（定时任务）不同：skill action 没有固定时间，在我"没什么可说"的空档随机触发。

## 触发场景

| 用户说 | 正确动作 |
|---|---|
| "有空的时候帮我读一下2236" | `skill_action_register` |
| "闲下来的时候更新一下知识库" | `skill_action_register` |
| "空闲时自动推进小说阅读" | `skill_action_register` |
| "停止自动读小说" | `skill_action_unregister` |
| "现在注册了哪些后台任务？" | `skill_action_list` |

## 工具用法

```
skill_action_register(
  id="novel-read-2236",           # 唯一ID，建议格式：{类型}-{标识}
  name="小说2236增量阅读",
  command="python3 ~/.akasic/workspace/skills/novel-reader/scripts/reader_task.py read-once --kb ~/.akasic/workspace/kb/2236",
  daily_max=3,                    # 每天最多执行几次
  min_interval_minutes=90,        # 两次之间最少间隔多少分钟
  weight=1.0                      # 多个任务并存时的抽取权重
)
```

注册后无需重启，下次我 idle 时自动生效。

## 注意事项

- `command` 必须是完整可执行的 shell 命令（含绝对路径）
- novel-reader 的 KB 路径在 `~/.akasic/workspace/kb/<name>/`
- 注册前可用 `shell` 工具确认命令能正常运行
