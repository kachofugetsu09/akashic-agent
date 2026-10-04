# 0087 · 会话标题覆盖：显式管理状态，空值回到推导标题

Status: proposed

## 背景

会话目录标题一直由首条消息正文截断推导。用户需要给会话起一个可辨识的名字（行内重命名），但标题不是消息事实，不能写进 `attributes`——后者在 admission 时固定，任何原位改写都会破坏 create-once 合同。

## 决定

`sessions` 表新增可空 `title` 列（yoyo 只增加列迁移 `20261004_04`）：

- `NULL` 表示沿用首条消息推导标题；非空为用户显式覆盖。
- 写入路径唯一：`MessageLog.set_session_title`，与 `set_session_deleted` 同属用户显式数据管理操作。去首尾空白，空值清除覆盖，上限 200 字符，超出拒绝。
- 幂等：与当前值相同不产生写操作。
- 不写 `updated_at`：重命名是管理操作，不改变目录排序事实，会话不因此跳到列表顶部。
- 不改消息、attachments、attributes、`deleted_at` 或任何 owner 状态。
- 读侧：`sessions()` 目录投影与 `MessageReader.title` 只在列存在时读取；缺列的旧库回退 `NULL`，与软删列的加列兼容路径一致。
- HTTP：`POST /api/chat/sessions/{key}/rename` 经 `core.session_admin` 窄能力；非本聊天目录前缀 400、不存在 404、超长 422。

## 与其他事实的关系

- 与软删正交：已删会话的 `title` 保留，恢复后原样可见。
- 与 Akasha 无关：标题是目录显示事实，不参与消息正文、embedding、学习输入或重放，不存在一致性问题。
- 置顶行经 `session_pin_row` 解析同一列，与目录显示同一事实。
- 无物理减少协议：清除覆盖是写 `NULL`，不删除行或消息。
