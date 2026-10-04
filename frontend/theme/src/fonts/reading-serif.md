# 中英文阅读字体

共享阅读组合使用 Source Serif 4 英文与 Noto Serif SC 中文，保留 JetBrains Mono 技术字体。
字体均本地发布，不依赖 Google Fonts 或客户端已有字体。可变字重范围为 200–900；
英文同时包含正常体、真实斜体与 optical size 轴。

来源为 Google Fonts 官方仓库的 `ofl/sourceserif4` 和 `ofl/notoserifsc`，
许可证分别见同目录 `sourceserif4-OFL.txt` 和 `notoserifsc-OFL.txt`（SIL OFL 1.1）。

- https://github.com/google/fonts/tree/main/ofl/notoserifsc
- https://github.com/google/fonts/tree/main/ofl/sourceserif4

Noto 的四个 WOFF2 分片覆盖原始字体的完整 Unicode cmap，按 `reading-serif.css`
的 Unicode 范围加载。由 FontTools 4.66.1 subset（默认选项）与系统 woff2_compress
生成，保留可变字重。源 WOFF2 来自 NotoSerifSC[wght].ttf 的完整压缩版本；
本次生成已比较源 cmap 与四片并集，确保没有删减支持字符。

源 Noto WOFF2 SHA256：`ee1376b2e481cb117e8786b95c1d7e7aa0f3c61ccfa50392823e56101e9cd065`。
