# Gemini 原生接口

`gemini` 插件通过公开的 `MODEL_DRIVERS` 注册 GenerateContent 对话驱动。默认
host-runtime profile 包含此插件；连接、凭据、模型选择与重试仍由 `models` 拥有。

在模型设置中选择「Gemini 原生 API」，填写 Base URL 和 API Key，点击
「探测可用模型」并勾选型号。探测只读取目录，不保存连接或密钥；修改地址或密钥
使旧探测结果失效，关闭编辑器取消未完成的探测并忽略迟到结果。
保存前逐个发送 `Reply OK.` 验证；首个型号失败不保存候选连接，其余型号失败
报告部分完成，不撤销已保存的连接。目录不可用时可以手动填写并明确确认型号。
编辑时 URL 和 Key 留空保持不变；已保存连接沿用 Models 的重新检测和选择入口。

| 目标 | Base URL 示例 |
|---|---|
| Google | `https://generativelanguage.googleapis.com/v1beta` |
| 提供原生路由的网关 | `https://your-gateway.example/antigravity` |

Base URL 可以指向服务根路径，也可以指向显式 API 版本目录，不包含 `/models`、
模型名称或查询参数。根路径自动追加默认版本 `/v1beta`；显式 `/v1`、`/v1beta`
等版本保持不变。目录读取和生成使用同一个版本根目录。
驱动发送 `x-goog-api-key`，调用 `models/{model}:generateContent` 或
`models/{model}:streamGenerateContent?alt=sse`。网关必须提供这些原生路由；
只有 OpenAI-compatible 路由的服务不能直接使用此驱动。

```text
┌────────────────────┐   ┌───────────────────────┐   ┌──────────────────┐
│ Models 消息投影    │ → │ Gemini 原生驱动        │ → │ Google / 网关    │
│ 与真实调用账本     │ ← │ GenerateContent / SSE │ ← │ 原生 API         │
└────────────────────┘   └───────────────────────┘   └──────────────────┘
```

支持文本、内联 base64 图片、工具调用与流式回复；不支持 embedding、远程图片 URL
或其他原生多媒体输入输出。工具 schema 使用 `parametersJsonSchema`。
普通请求开启思考摘要；可选 reasoning effort 使用 Gemini 3 的 `thinkingLevel`，
仅应选择目标模型支持的等级。验证请求关闭摘要并省略显式等级，不能保证所有
Gemini 模型停止内部思考。参见 [Google 思考文档](https://ai.google.dev/gemini-api/docs/generate-content/thinking)。

原生响应的 `Content.parts`（包括 `thoughtSignature`）保存在已有调用账本的
`response_json`，不改写既有 Message。下一次同一 binding 的请求原样重放这些
部件；放弃调用、已标记内容转换或切换 binding 时不重放对应原生响应。
此状态不在不同 binding 之间迁移，完整 runtime 重启不保证沿用原 binding。
思考摘要是展示内容，签名是 provider 的不透明协议状态，两者不能互相替代。
参见 [Google 签名文档](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)。

驱动每次生成只发送一次 HTTP，明确报告授权、限流、协议和传输错误；
不会自动切到 Chat Completions。发送给网关的签名完整，并不能证明网关继续
完整转发给其后端；这一点需独立验证网关行为。

`MALFORMED_FUNCTION_CALL` 表示上游生成的工具调用无法解析，整次候选均不提交
或执行。Models 按[现有恢复规则](../../docs/decisions/0092-model-generation-recovers-until-output.md)
用原请求退避重试，保留已完成工具结果、重试进度、取消和显式次数上限；
未配置次数上限时持续恢复，不保证下一次生成成功，也可能产生重复生成费用。
失败调用的 usage 仍记为未知。`UNEXPECTED_TOOL_CALL`、无效请求、鉴权与安全拒绝
不因此获得恢复资格。不会强制工具调用或把无法解析的内容修补成可执行调用。
参见 [Google 完成原因定义](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerateContentResponse#FinishReason)。
