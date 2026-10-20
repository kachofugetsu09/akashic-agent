import type { MouseEvent as ReactMouseEvent } from "react";
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
  markdown.inline.ruler.before("math", "akashic_currency_dollar", (rawState: unknown, silent?: boolean) => {
    const state = rawState as DollarState;
    if (!isRejectedDollar(state.src, state.pos)) return false;
    if (!silent) state.push("text", "", 0).content = "$";
    state.pos += 1;
    return true;
  });
  return markdown;
}

interface DollarState {
  src: string;
  pos: number;
  push: (type: string, tag: string, nesting: number) => { content: string };
}

// 行内 $…$ 与静态路径同一规则（pandoc）：开 $ 后、闭 $ 前不是空白，闭 $ 后不紧跟数字。
// 不满足时只把开头这个 $ 当文字吐出，后面的原文照常解析：闭合的那个 $ 可能是下一个真公式的开头
// （"成本 $5 和公式 $x$"），转义（"$\{x\}$"）也保持原样交给公式规则。没有闭合的留给生成中的源码态。
function isRejectedDollar(source: string, offset: number): boolean {
  if (source[offset] !== "$" || source[offset + 1] === "$" || source[offset - 1] === "\\") return false;
  let close = offset + 1;
  while ((close = source.indexOf("$", close)) >= 0 && source[close - 1] === "\\") close += 1;
  if (close < 0) return false;
  const content = source.slice(offset + 1, close);
  return content.length === 0 || /^\s|\s$/u.test(content) || /^\d/u.test(source[close + 1] ?? "");
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

/** 一次溢出测量的读相快照：渐隐状态与滚动元素（无横向溢出时为 null）。 */
export interface OverflowEdges { overflow: string; scroller: HTMLElement | null }

/** 横滚块两侧哪边还藏着内容，写到 data-overflow（start / end），样式在对应边缘渐隐。 */
export function writeOverflowEdges(target: HTMLElement, scroller: HTMLElement | null): void {
  applyOverflowEdges(target, readOverflowEdges(scroller));
}

/** 只读几何：批处理队列的读相用（scheduleOverflowJob），读必须早于同帧的任意写。 */
export function readOverflowEdges(scroller: HTMLElement | null): OverflowEdges {
  const start = scroller !== null && scroller.scrollLeft > 1;
  const end = scroller !== null && scroller.scrollLeft + scroller.clientWidth < scroller.scrollWidth - 1;
  return { overflow: [start ? "start" : "", end ? "end" : ""].filter(Boolean).join(" "),
    scroller: start || end ? scroller : null };
}

/** 只写回：把读相算出的渐隐与 Tab 序应用到 DOM，不读任何几何。 */
export function applyOverflowEdges(target: HTMLElement, { overflow, scroller }: OverflowEdges): void {
  if (target.dataset.overflow !== overflow) target.dataset.overflow = overflow;
  syncScrollStop(target, scroller);
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
  const label = scroller.querySelector("table") ? "表格" : scroller.classList.contains("reading-math-display") ? "公式" : "代码";
  scroller.setAttribute("aria-label", `${label}，可左右滚动`);
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

// 同一帧里多个代码块的溢出测量合并成一批、读写分相：先全部读几何，再全部写回。
// 逐块各自 rAF 时，前一块写回的 data-overflow/tabindex 会让后一块的读取重新强制布局；
// 一批只付一次布局（design.md 3.18 的同一原则，批量版）。切入富会话时可见区每块各排一次，
// 合并前该测量占稳态切换 CPU 的 ~43%（probe-switch-cpu，CPU×4）。
// 成立前提（改动 [data-overflow] / [data-scroll-stop] 样式前必读）：写相不得改变任何几何——
// 目前写相只碰 dataset.overflow 与 tabindex/role/aria-label/data-scroll-stop，
// 对应样式只有 mask-image（绘制属性，不参与布局）。若将来加了影响布局的规则，
// 批处理会退化成一帧陈旧，且很难查。
// 单个 job 的读写各自包 try/catch：一个块抛错只跳过自己，不带走整批
// （job 已从 Set 摘除，失败后等下一次 mutation/resize/scroll 重新调度，自行恢复）。
export interface OverflowJob { read: () => void; write: () => void }
const overflowJobs = new Set<OverflowJob>();
let overflowFrame = 0;

export function scheduleOverflowJob(job: OverflowJob): void {
  overflowJobs.add(job);
  if (overflowFrame) return;
  overflowFrame = requestAnimationFrame(() => {
    overflowFrame = 0;
    const jobs = [...overflowJobs];
    overflowJobs.clear();
    for (const job of jobs) {
      try { job.read(); } catch (error) { console.error("溢出测量读相失败", error); }
    }
    for (const job of jobs) {
      try { job.write(); } catch (error) { console.error("溢出测量写相失败", error); }
    }
  });
}

export function cancelOverflowJob(job: OverflowJob): void {
  overflowJobs.delete(job);
}

const FOOTNOTE_LINK = "a[data-footnote-ref], a[data-footnote-backref], a.footnote-link, a.footnote-anchor";

// 每个回复容器最近一次从哪个引用跳进脚注：同一脚注被引用多次时，返回落回出发的那一处。
const footnoteOrigins = new WeakMap<HTMLElement, HTMLElement>();
const FOOTNOTE_BACK = "a[data-footnote-backref], a.footnote-anchor";

// href 里的 id 先按原样找（富渲染直接拼标签，"a%20b" 就是标签本身），找不到再按 URI 解码找（静态路径编码过中文标签）。
function hrefIds(link: Element): string[] {
  const raw = (link.getAttribute("href") ?? "").slice(1);
  let decoded = raw;
  try { decoded = decodeURIComponent(raw); } catch { /* 不是合法编码，只按原样找 */ }
  return decoded === raw ? [raw] : [raw, decoded];
}

function byId(container: HTMLElement, id: string): HTMLElement[] {
  return [...container.querySelectorAll<HTMLElement>(`[id="${CSS.escape(id)}"]`)];
}

function byHref(container: HTMLElement, link: Element): HTMLElement[] {
  for (const id of hrefIds(link)) {
    const found = byId(container, id);
    if (found.length) return found;
  }
  return [];
}

// 返回目标：出发点仍在且属于这条脚注时回到出发点；否则按 id（"标签-序号"在富渲染里是同名 id 的第几个）。
function referenceFor(container: HTMLElement, back: HTMLElement): HTMLElement | null {
  const origin = footnoteOrigins.get(container);
  if (origin?.isConnected && byHref(container, origin).some((note) => note.contains(back))) return origin;
  for (const id of hrefIds(back)) {
    const exact = byId(container, id)[0];
    if (exact) return exact;
    const repeat = /^(.*)-(\d+)$/u.exec(id);
    const nth = repeat ? byId(container, repeat[1])[Number(repeat[2]) - 1] : undefined;
    if (nth) return nth;
  }
  return null;
}

// 进脚注后落在对应出发点的返回箭头上（静态路径重复引用各有一个箭头），没有对应的就落在第一个。
function backFor(note: HTMLElement, origin: HTMLElement): HTMLElement {
  const backs = [...note.querySelectorAll<HTMLElement>(FOOTNOTE_BACK)];
  const ids = [origin.id, origin.closest("sup")?.id].filter(Boolean);
  return backs.find((back) => hrefIds(back).some((id) => ids.includes(id))) ?? backs[0] ?? note;
}

/** 脚注引用与返回链接在本条回复内跳转（挂在回复容器的 onClickCapture 上）：每条回复的脚注 id 都从 1 编起，
 *  按 id 全局找会跳到别的回复；也不往地址栏写 hash、不多一条历史记录。 */
export function jumpToFootnote(event: ReactMouseEvent<HTMLElement>): void {
  const container = event.currentTarget;
  const link = event.target instanceof Element ? event.target.closest<HTMLAnchorElement>(FOOTNOTE_LINK) : null;
  if (!link?.getAttribute("href")?.startsWith("#") || !container.contains(link)) return;
  // 1. 受管的脚注链接一律不走浏览器的 hash 导航：本条里找不到目标也只留在原处，不跳出这条回复。
  event.preventDefault();
  event.stopPropagation();
  const backward = link.matches(FOOTNOTE_BACK);
  const target = backward ? referenceFor(container, link) : byHref(container, link)[0];
  if (!target) return;
  // 2. 焦点落到对面的链接：去脚注落在返回箭头上，回正文落在引用号上，回车即可来回。
  if (!backward) footnoteOrigins.set(container, link);
  const landing = backward ? (target.matches(FOOTNOTE_LINK) ? target : target.querySelector<HTMLElement>(FOOTNOTE_LINK) ?? target)
    : backFor(target, link);
  landing.focus({ preventScroll: true });
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  target.scrollIntoView({ block: "center", behavior: reduce ? "auto" : "smooth" });
}
