# 0054 · 模型同步刷新公共能力目录

- 状态：accepted / implemented
- 日期：2026-09-01
- 关联条款：RUN-011、ONB-001、WSP-001
- 关联设计：[模型运行时注册表与首次配置](../design/runtime-model-registry-and-onboarding.md)

## 背景

Provider 的 `/models` 是账号当前可用模型的权威目录，但 OpenCode Go 等兼容接口通常不返回图片输入等
能力。固定 LiteLLM wheel 能离线识别旧模型，却无法识别上游后来登记的新模型。要求用户为每个新模型
手填能力会让首次连接和后续维护都依赖额外知识。

## 决定

用户创建连接或再次同步时，`models` 插件先读取 Provider 目录，再从 LiteLLM 官方公开 JSON 目录精确匹配
同一个 model key，只补全 Provider 未声明的字段。公共目录不增加、删除或改名 Provider 返回的模型，也不
改变 wire transport。Provider 字段优先；公开目录不能覆盖 Provider 已声明的事实。

旧版本迁入且尚无 capability ownership payload 的同 wire identity 模型，在首次成功同步时保留原 model
ID 并转为 discovery owner，使存量模型也进入同一自动刷新链。用户后来明确手工新增且已有 ownership
payload 的模型继续受保护，不被同步覆盖或禁用。

每次显式同步都尝试刷新，因此已有连接可以吸收 LiteLLM 后来增加或修正的能力。刷新只访问固定 HTTPS
地址，使用短超时、禁止跳转、限制响应大小和条目数，并校验完整 JSON。成功快照以 schema、ETag、摘要和
抓取时间组成完整 envelope，在插件自己的 data root 中 fsync 后原子替换。
同一 data root 的同步使用跨进程文件锁串行化，后开始的刷新必须先看到前一次已发布的 ETag，避免旧请求
较晚完成时覆盖更新快照。

刷新失败时按“最近一次通过校验的远端快照 → 当前固定 wheel 的随包目录 → unknown”降级。失败不阻止
Provider 模型同步，不把已知能力改成 unknown，也不生成假能力。启动、普通目录读取和 Turn 执行均不联网。

公共目录补全 Provider 未声明的图片输入、上下文窗口、输出上限和推理强度。`context_window`
使用 LiteLLM 的 `max_input_tokens`，不与 `max_output_tokens` 相加。工具调用能力继续由 Provider
提供；Akashic 尚未实现的 audio/video 输入能力不得发布。

## 理由

Provider 和公共能力目录各自只拥有一个变化轴：前者回答账号“有哪些模型”，后者回答精确型号“已知能做
什么”。显式同步既符合用户预期，也避免后台更新在没有操作时静默改变下一次 Turn。最近可信快照让新
部署离线时仍可用，让已经识别过的连接不因短暂网络故障倒退。

## 影响

- `models` 插件增加 `litellm-capabilities.json` 派生缓存；普通卸载仍保留 plugin-data。
- OpenCode Go 的文本请求基线不再被标成 Provider 已证明“仅文本”。
- 设置页显示每个连接可看图和待识别的模型数；视觉角色只列出已确认支持图片输入的模型。
- 新 API 连接先用内存凭据检测目录，不写 connection、credential 或 revision；用户选中一个模型后才按既有候选事务保存。
- 临时检测使用固定短 deadline、零重试、响应字节和模型数硬上限；关闭弹窗或修改连接字段会取消旧请求，旧结果不得提交到新连接候选。
- LiteLLM Python 依赖继续固定版本；远端 JSON 只是经过边界校验的非执行数据。

## 验收

- [x] 当前 LiteLLM 上游新增的精确型号可在重复同步后获得图片能力。
- [x] 新模型可在重复同步后获得上下文窗口、输出上限和推理强度；成功目录先原子写入插件缓存。
- [x] 远端超时、坏 JSON、异常缩小和缓存损坏均按既定顺序降级，Provider 模型仍可提交。
- [x] 远端目录不能增加 Provider 未返回的模型，也不能覆盖 Provider 已知字段。
- [x] 新连接和已有连接都使用同一同步入口；已有连接可直接重新同步，不需要改写连接；UI 明确展示可看图与待识别。
- [x] 使用维护者的真实 OpenAI-compatible 订阅绑定 `deepseek-v4-flash-vision-exp`；Web 上传图片后，首次 Provider 调用直接读图并零工具调用完成 Turn。
