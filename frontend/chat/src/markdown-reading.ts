import type { MarkdownIt, MarkdownToken } from "stream-markdown-parser";

import { copyText } from "./copy-text";

/** 只识别显式的纯文本语言，不把未知代码语言降为说明文字。 */
export function isPlainTextLanguage(language: string) {
  return /^(?:text|txt|plaintext)?$/iu.test(language.trim());
}

/** 整列都是单个数值时才启用数值排版；版本、范围和说明保留原样。 */
export function isNumericColumn(values: readonly string[]) {
  return values.length > 0 && values.every((value) =>
    /^[+-]?(?:\d+(?:\.\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?)(?:%|ms|s|GB|MB|MHz|GHz|W|°C)?$/u.test(value.trim()),
  );
}

/** 短标签后确有说明时才分栏，避免把整段加粗误认为概念名。 */
export function isConceptLabel(label: string, description: string) {
  return label.trim().length > 0 && [...label.trim()].length <= 40 && description.trim().length > 0;
}

/** 标签与说明分栏后，原文写在两者之间的冒号不再有语义，留着会悬在说明行首。 */
export function stripConceptSeparator(text: string) {
  return text.replace(/^\s*[:：]\s*/u, "");
}

/**
 * 表格对齐在 MarkdownIt 的局部 token 阶段补齐。不要使用 postTransformTokens：
 * 它会禁用稳定节点复用，并在每次追加时克隆全部 token、重复转换整个正文。
 */
export function configureReadingMarkdown(markdown: MarkdownIt): MarkdownIt {
  markdown.core.ruler.push("akashic_numeric_tables", (state: { tokens: MarkdownToken[] }) => {
    state.tokens = alignNumericTableTokens(state.tokens);
  });
  return markdown;
}

/** parser 会把未指定的表格对齐转成 left，因此在 token 阶段补齐数值列对齐。 */
function alignNumericTableTokens(tokens: MarkdownToken[]) {
  const result = [...tokens];
  let columns: number[][] = [];
  let column = 0;
  let inTable = false;
  for (let index = 0; index < tokens.length; index++) {
    const token = tokens[index];
    if (token.type === "table_open") {
      inTable = true;
      columns = [];
    } else if (inTable && token.type === "tr_open") column = 0;
    else if (inTable && (token.type === "th_open" || token.type === "td_open")) {
      (columns[column++] ??= []).push(index);
    } else if (inTable && token.type === "table_close") {
      for (const cells of columns) {
        const values = cells.slice(1).map((cell) => String(tokens[cell + 1].content));
        if (!isNumericColumn(values)) continue;
        for (const cell of cells) {
          const attrs = tokens[cell].attrs ?? [];
          if (attrs.some(([name]) => name === "style" || name === "align")) continue;
          result[cell] = { ...tokens[cell], attrs: [...attrs, ["style", "text-align:right"]] };
        }
      }
      inTable = false;
    }
  }
  return result;
}

/** 异步边界统一报告 Clipboard API 缺失与浏览器拒绝复制。 */
export async function copyReadingCode(code: string) {
  await copyText(code);
}

/** 横滚块两侧哪边还藏着内容，写到 data-overflow（start / end），样式在对应边缘渐隐。 */
export function writeOverflowEdges(target: HTMLElement, scroller: HTMLElement | null): void {
  const start = scroller !== null && scroller.scrollLeft > 1;
  const end = scroller !== null && scroller.scrollLeft + scroller.clientWidth < scroller.scrollWidth - 1;
  const next = [start ? "start" : "", end ? "end" : ""].filter(Boolean).join(" ");
  if (target.dataset.overflow !== next) target.dataset.overflow = next;
  syncScrollStop(target, start || end ? scroller : null);
}

const scrollStops = new WeakMap<HTMLElement, HTMLElement>();

// 横滚区放不下时进 Tab 序，键盘才能用方向键看藏住的列；放得下或换了滚动元素（高亮前后）就撤掉，不多一个空停靠点。
// 只撤自己加的：markstream 或页面另给的 tabindex 不动。
function syncScrollStop(target: HTMLElement, scroller: HTMLElement | null): void {
  const previous = scrollStops.get(target);
  if (previous === scroller) return;
  if (previous?.hasAttribute("data-scroll-stop")) {
    for (const name of ["tabindex", "role", "aria-label", "data-scroll-stop"]) previous.removeAttribute(name);
  }
  if (!scroller) {
    scrollStops.delete(target);
    return;
  }
  scrollStops.set(target, scroller);
  if (scroller.hasAttribute("tabindex")) return;
  scroller.tabIndex = 0;
  scroller.setAttribute("role", "group");
  scroller.setAttribute("aria-label", scroller.querySelector("table") ? "表格，可左右滚动" : "代码，可左右滚动");
  scroller.setAttribute("data-scroll-stop", "");
}

/** 自身就是滚动容器的块（宽表）：滚动、容器或内容尺寸变化时重新判定，返回清理函数。 */
export function watchOverflowEdges(scroller: HTMLElement): () => void {
  const measure = () => writeOverflowEdges(scroller, scroller);
  const resize = new ResizeObserver(measure);
  resize.observe(scroller);
  if (scroller.firstElementChild) resize.observe(scroller.firstElementChild);
  scroller.addEventListener("scroll", measure, { passive: true });
  // 不在注册时同步测量：ResizeObserver 在 observe 后的同一帧布局之后、绘制之前必回调一次，
  // 同步读 scrollLeft 会在 effect 提交阶段强制布局（落定换手实测约 12–16ms）。
  return () => {
    resize.disconnect();
    scroller.removeEventListener("scroll", measure);
  };
}
