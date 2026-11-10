# 中英文阅读字体

共享阅读组合使用 Source Serif 4 英文与 Noto Serif SC 中文，保留 JetBrains Mono 技术字体。
字体均本地发布，不依赖 Google Fonts 或客户端已有字体。可变字重范围为 200–900；
英文同时包含正常体、真实斜体与 optical size 轴。

来源为 Google Fonts 官方仓库的 `ofl/sourceserif4` 和 `ofl/notoserifsc`，
许可证分别见同目录 `sourceserif4-OFL.txt` 和 `notoserifsc-OFL.txt`（SIL OFL 1.1）。

- https://github.com/google/fonts/tree/main/ofl/notoserifsc
- https://github.com/google/fonts/tree/main/ofl/sourceserif4

Noto Serif SC 按字频分层切片（Version 2.003，源 `NotoSerifSC[wght].ttf` SHA256
`050080d9255a86808f2945bffac582b31ef32bc36411ce29563b4961670c66f9`），浏览器按
`unicode-range` 只下载页面用到的分片：

- `core`：拉丁、通用标点、箭头、CJK 标点与假名、全角形式；
- `common-1`：GB2312 一级字（3755），整片；
- `common-2-NN`：GB2312 二级字（3008）按部首顺序每 500 字一片——"阈""渚"这类现代常见字在二级字里，一页常有几个，细切后只多下载几片；
- `han-NN`、`ext-a-NN`：其余基本区汉字与扩展 A，按码位每 800 字一片；
- `symbols`、`astral`：其余符号与兼容字形、增补平面。

普通中文页面只需 `core` + `common-1`（约 1.5MB，构建时预加载），原先四段码位切片会下载全部约 11MB。
每片只声明实际含有的码位，各分片的 `unicode-range` 互不重叠。不要用连续宽区间盖住生僻区、指望
"重叠时后声明的规则先匹配"让常用字只命中常用层：Chromium 如此，但 Firefox 冷启动整页时会把与页面文字
区间相交的宽分片一并下载（实测 7–8MB）；精确区间后两者都只取 `core`、`common-1` 与用到的二级字分片
（约 1.7MB），出现一个生僻字只多一片。代价是 CSS 约 97KB（gzip 约 33KB）。

重新生成（FontTools 4.66.1）：

```bash
python scripts/split-reading-font.py /path/to/NotoSerifSC[wght].ttf
```

脚本校验源摘要、分片与 CSS 区间都互不相交且并集等于源 cmap，并按 CSS 匹配规则模拟每个码位都落到含它的分片后，
才替换仓库里的分片与 `reading-serif.css`。
