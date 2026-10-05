# Gemini 原生接口

`gemini` 通过公开 `MODEL_DRIVERS` 注册 GenerateContent 驱动，默认 host-runtime
profile 加载此插件。连接、凭据、选择与重试仍由 `models` 拥有。

通过 Models 设置 API 注册 `driver_id: gemini` 的连接；Base URL 指向原生版本根目录，
例如 `https://generativelanguage.googleapis.com/v1beta` 或网关的 `/antigravity/v1beta`。
凭据使用 `driver: api_key`、`access_token: <key>`，连接和模型 `driver_config` 使用
`format_version: 1`。连接可设置 `catalog_provider_id: gemini` 以同步原生模型目录。
模型填写原生名称；保存前 Models 发送短消息验证，失败不保存候选连接。

支持文本、内联 base64 图片、工具与 SSE；不支持 embedding 或其他原生媒体输出。
普通请求开启摘要，Gemini 3 可设置目标模型支持的 thinkingLevel；验证请求关闭
摘要并省略等级，不保证停止内部思考。原 Content.parts 与 thoughtSignature 保存
在既有 response_json，同 binding 原样重放；放弃调用、内容转换或换 binding
时不重放对应原生响应，完整 runtime 重启不保证沿用旧 binding。无 schema 迁移或
既有 Message 改写。每次生成只发送一次 HTTP，无隐式 Chat Completions fallback。
签名完整到达网关不证明网关完整转发给后端。

协议：[生成](https://ai.google.dev/api/generate-content)、[签名](https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures)。
