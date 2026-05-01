# meme 插件

表情包发送能力。把工作区的表情包目录映射成 LLM 可感知的 catalog，让模型在回复中嵌入 `<meme:tag>` 标签，插件在 AfterReasoning 阶段将标签解析为实际媒体文件路径。

---

## 接入点

| 接入方式 | 阶段 |
|---|---|
| `prompt_render_modules_bottom()` | PromptRender 阶段末尾——注入表情包目录说明 |
| `@on_after_reasoning()` | AfterReasoning GATE——解析 meme 标签，附加媒体 |

---

## 运作逻辑

### 1. 初始化（initialize）

从工作区路径（`workspace/memes/`）加载 `manifest.json`，构建 `MemeCatalog` 和 `MemeDecorator` 实例。`MemeCatalog` 按需检测 manifest 的 mtime，变动时自动热重载，不需要重启。

### 2. 注入 catalog（MemePromptModule）

每轮推理前，调用 `catalog.build_prompt_block()` 把启用的表情包类别（名称、描述、别名）拼成文本块，追加到系统 prompt 底部，告知 LLM 可以在回复中嵌入 `<meme:tag>` 标签。如果 catalog 为空则跳过注入。

### 3. 解析标签（decorate_meme）

推理完成后，从 `ctx.reply` 里用正则扫描第一个 `<meme:tag>` 标签，同时把所有 meme 标签从文本中剥除。再交给 `MemeDecorator.decorate()` 处理：

- 按 tag 查找对应类别的图片目录，随机挑选一张文件，生成文件路径。
- 返回 `DecorateResult`，包含清理后的文本、媒体路径列表、以及实际使用的 tag。

结果写回 `ctx.reply`、`ctx.media`、`ctx.meme_tag`，由下游频道适配层将图片随消息一起发出。

---

## 配置

表情包资源放在工作区 `memes/` 目录下，`manifest.json` 声明各类别：

```json
{
  "categories": {
    "happy": { "desc": "开心", "aliases": ["开心", "高兴"] },
    "sad":   { "desc": "难过", "aliases": ["难过"] }
  }
}
```

每个类别对应同名子目录，目录下放图片文件（`.png` / `.jpg` / `.gif` / `.webp`）。
