---
name: meme-manage
description: 维护表情包库。当用户要添加表情图片、新建表情类别、更新 manifest、整理 memes 目录时触发。
---

# 表情包库管理

## 触发条件

- 用户要把某张图加入表情包库
- 用户要新建一个表情类别（如"无语"、"委屈"）
- 用户要更新某个类别的描述或别名
- 用户要整理 `memes/` 目录结构

## 工作流

### 第一步：读取当前状态

```
read_file("memes/manifest.json")
list_dir("memes/")
```

### 第二步：确认类别是否存在

- 若类别已存在 → 直接写入图片到对应目录
- 若类别不存在 → 先创建目录，再更新 manifest

### 第三步：写入图片

将图片保存到 `memes/<category>/`，文件名建议用三位数字序号（001.png、002.jpg 等）。

如果用户通过 Telegram 发来图片，图片已在 `media` 字段中以本地路径传入，直接复制到目标目录即可（使用 `write_file` 或 shell 命令）。

### 第四步：更新 manifest.json

manifest 结构：

```json
{
  "version": 1,
  "categories": {
    "<name>": {
      "desc": "中文语义描述，用于 agent 判断何时使用",
      "aliases": ["别名1", "别名2"],
      "enabled": true
    }
  }
}
```

规则：
- `desc` 应简洁描述该表情的情绪/场景，供 agent 在 system prompt 中参考
- `aliases` 可为空列表
- `enabled: false` 可临时禁用某类别而不删除

## 注意

- 本 skill **只负责维护库**，不负责控制 agent 回复时是否发表情
- 不要修改 `agent/memes/` 下的代码
- 操作完成后告知用户：新增了哪个类别 / 添加了哪张图 / manifest 已更新
