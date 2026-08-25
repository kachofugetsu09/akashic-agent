import { Check, Copy, FileText, ImageIcon } from "lucide-react";
import { useState } from "react";
import "./media-render-showcase.css";

type RecipeId = "now" | "memoh" | "waku" | "propose";

const RECIPES: { id: RecipeId; label: string; note: string }[] = [
  {
    id: "now",
    label: "现在",
    note: "附件方卡偏小、坏图占位抢戏；代码块顶栏重、全宽偏闷。",
  },
  {
    id: "memoh",
    label: "Memoh",
    note: "正文图自然比例；上传用方芯片；代码淡边 + 侧置 copy，无语言条。",
  },
  {
    id: "waku",
    label: "Waku",
    note: "统一 inset 瓷砖 96×80；文件同尺寸；代码 muted wash，圆角更大。",
  },
  {
    id: "propose",
    label: "建议",
    note: "已落地：正文自然比例；120 方芯片；代码贴宽 + 侧置 copy（对齐 Memoh）。",
  },
];

const DEMO_CODE = `fn main() {
  println!("hello, paper");
}`;

const DEMO_IMG =
  "data:image/svg+xml," +
  encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="360" viewBox="0 0 640 360">
      <defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
        <stop stop-color="#d7dde6"/><stop offset="1" stop-color="#b7c2d0"/>
      </linearGradient></defs>
      <rect width="640" height="360" fill="url(#g)"/>
      <text x="40" y="190" font-family="ui-serif,serif" font-size="42" fill="#1b365d">张力.gif</text>
    </svg>`,
  );

export function MediaRenderShowcase() {
  const [recipe, setRecipe] = useState<RecipeId>("propose");
  const [copied, setCopied] = useState(false);

  return (
    <div className="mrs" data-recipe={recipe}>
      <header className="mrs-hero">
        <p className="mrs-path">/chat/?preview=media-render</p>
        <h1>媒体与代码块 · 渲染对照</h1>
        <p>
          对照 Memoh / Waku 的图片、附件芯片与代码块配方。当前会话里坏图小方块和沉重代码壳不好用；这里只做样式展示，不改生产渲染路径。
        </p>
      </header>

      <nav className="mrs-tabs" aria-label="配方">
        {RECIPES.map((item) => (
          <button
            key={item.id}
            type="button"
            className={recipe === item.id ? "is-active" : undefined}
            onClick={() => setRecipe(item.id)}
          >
            <strong>{item.label}</strong>
            <span>{item.note}</span>
          </button>
        ))}
      </nav>

      <section className="mrs-section">
        <h2>1. 正文里的图</h2>
        <p className="mrs-lead">助手回复中的图片应保留真实比例，点进灯箱再放大——不要做成过小的破损方块。</p>
        <div className="mrs-stage">
          {recipe === "now" ? (
            <div className="mrs-now-broken" role="img" aria-label="当前坏图占位">
              <ImageIcon size={18} aria-hidden="true" />
              <span>张力.gif</span>
            </div>
          ) : (
            <button type="button" className="mrs-content-image" aria-label="预览张力.gif">
              <img src={DEMO_IMG} alt="" />
            </button>
          )}
          <p className="mrs-caption">
            {recipe === "now" && "现在：破损图标 + 文件名挤在小白块里，像上传失败而不是媒体。"}
            {recipe === "memoh" && "Memoh：max-w ≈ 28rem、max-h ≈ 20rem、object-contain、细边圆角。"}
            {recipe === "waku" && "Waku：max-h-64 + object-contain，外壳 rounded≈9px + inset 底。"}
            {recipe === "propose" && "建议：自然比例、max-w 28rem / max-h 20rem、1px 发丝边、圆角 10px。"}
          </p>
        </div>
      </section>

      <section className="mrs-section">
        <h2>2. 附件芯片 / 文件卡</h2>
        <p className="mrs-lead">上传预览与非图文件用同一套「瓷砖语言」，尺寸固定、信息不抢正文。</p>
        <div className="mrs-stage mrs-chip-row">
          <div className="mrs-chip is-media" aria-label="图片附件">
            <img src={DEMO_IMG} alt="" />
          </div>
          <div className="mrs-chip is-file" aria-label="文件附件">
            <FileText size={18} aria-hidden="true" />
            <span>notes.md</span>
          </div>
          <div className="mrs-chip is-broken" aria-label="加载失败">
            <ImageIcon size={16} aria-hidden="true" />
            <span>张力.gif</span>
            <small>无法预览</small>
          </div>
        </div>
        <p className="mrs-caption">
          {recipe === "now" && "现在：栅格偏碎，坏图与成功图没有共用尺寸语言。"}
          {recipe === "memoh" && "Memoh：方芯片 size≈120；文件/粘贴同壳，hover 只加深边。"}
          {recipe === "waku" && "Waku：统一 96×80 inset 瓷砖；非图用图标 + 微字文件名。"}
          {recipe === "propose" && "建议：88×72 瓷砖；坏图仍占同一格，文案「无法预览」代替破碎感。"}
        </p>
      </section>

      <section className="mrs-section">
        <h2>3. 代码块</h2>
        <p className="mrs-lead">代码是文档块，不是工具栏展台。贴内容宽、侧置复制、去掉沉重顶栏。</p>
        <div className="mrs-stage">
          {recipe === "now" ? (
            <div className="mrs-code is-now">
              <div className="mrs-code-chrome">
                <span>rust</span>
                <button type="button" className="mrs-code-copy">
                  <Copy size={14} aria-hidden="true" />
                  复制
                </button>
              </div>
              <pre>
                <code>{DEMO_CODE}</code>
              </pre>
            </div>
          ) : (
            <div className="mrs-code">
              <pre>
                <code>{DEMO_CODE}</code>
              </pre>
              <button
                type="button"
                className="mrs-code-copy"
                aria-label={copied ? "已复制" : "复制代码"}
                onClick={() => {
                  void navigator.clipboard?.writeText(DEMO_CODE);
                  setCopied(true);
                  window.setTimeout(() => setCopied(false), 1200);
                }}
              >
                {copied ? <Check size={14} aria-hidden="true" /> : <Copy size={14} aria-hidden="true" />}
              </button>
            </div>
          )}
          <p className="mrs-inline">
            行内代码示例：配置键 <code>AKASHIC_WORKSPACE</code> 与路径 <code>~/.akashic</code>。
          </p>
          <p className="mrs-caption">
            {recipe === "now" && "现在：顶栏占一截高度，块常拉满栏宽，视觉压过正文。"}
            {recipe === "memoh" && "Memoh：白/卡片底 + border/60，copy 图标旁置，streaming 不抖。"}
            {recipe === "waku" && "Waku Web：muted/45 + rounded-xl；原生另有 28px 语言头栏。"}
            {recipe === "propose" && "建议：w-fit、发丝边、侧置 icon copy；行内用浅 wash，不用高对比胶囊。"}
          </p>
        </div>
      </section>

      <section className="mrs-section mrs-recipe">
        <h2>落地优先级</h2>
        <ol>
          <li>正文图：自然比例 + 灯箱；坏图改文件卡，禁止碎图标小白块。</li>
          <li>附件：统一瓷砖尺寸（建议 88×72），图 cover / 非图图标+截断名。</li>
          <li>代码：去掉重顶栏；侧置 copy；块宽贴内容（max-w 栏宽）。</li>
          <li>token：边框用 outline-variant 低透明；底用 surface-container-low，避免再造一套霓虹。</li>
        </ol>
      </section>
    </div>
  );
}
