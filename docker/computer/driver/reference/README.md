# Browser reference

来源是本地 `codex-desktop-rev/computer-use-engine/@oai-browser-desktop` 0.1.1。
原文件 SHA-256 在上层 `reference-sha256.json`；先用锁定的 Prettier 3.6.2 格式化，再应用
`akashic.patch`。保留原业务与依赖代码，维护补丁和 Akashic adapter；不把 bundle 格式化说成
恢复了原工程的类型、模块划分或构建系统。

补丁只接入源码 AX core、容器 CDP 地址与权限 owner、ZXing 公开依赖路径，以及输出通道。
`akashic-container` 模式由外层 Akashic 的工具授权拥有权限，不依赖 Codex guardian/auth broker。
需要原生客户端或审批提供者的可选功能不启用；API 清单逐项记录覆盖和差异。

参考升级顺序：核对原文件哈希 → 相同格式化 → 审查/重放补丁 → 核对 API 变化 → 运行容器
生命周期与 Cua 原题对比。`accessibility.mjs`、`cdp.mjs`、`desktop.mjs`、`runtime.mjs` 和 Rust
native 源码是本项目直接维护的部分。ZXing 使用固定 npm 依赖；不携带原 AX WASM 或 Sky ELF。
