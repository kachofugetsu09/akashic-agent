import { memo, useEffect, useLayoutEffect, useRef, useSyncExternalStore, type ClipboardEvent as ReactClipboardEvent } from "react";
import type { KatexOptions } from "katex";
import { watchOverflowEdges } from "./markdown-reading";

// 公式渲染（参考 Claude Desktop：KaTeX 按需加载、errorColor 跟正文、复制还原 TeX）。
// 性能要点：KaTeX 排一个公式约 0.1–0.3ms，贵的是每次提交把所有公式重排一遍——这里按"源码 + 模式"缓存 HTML，
// 闭合的公式整条消息里只排一次；末尾还在写的公式每次提交试排一次，失败就截掉写到一半的命令再试，
// 仍不行就保留上一次排成的样子，不闪回源码、不出现 Loading。

type Katex = typeof import("katex").default;
type MathMode = "inline" | "display" | "inline-display";

let katex: Katex | null = null;
let katexLoad: Promise<void> | null = null;
const listeners = new Set<() => void>();

const CACHE_LIMIT = 400;
const cache = new Map<string, string | null>();

// 只输出 HTML（不带 MathML）：DOM 少两到三成、排版快近一半，复制也不会混进 MathML 的逐字文本；读屏名用 TeX 源码。
const OPTIONS: KatexOptions = { throwOnError: true, strict: false, output: "html", trust: false };

/** 检测到公式就开始加载 KaTeX、样式和常用字体：字体晚到会让排好的公式整块重排。 */
export function preloadMath(): Promise<void> {
  katexLoad ??= Promise.all([import("katex"), import("@/katex-styles")]).then(async ([module]) => {
    katex = module.default;
    for (const listener of listeners) listener();
    await Promise.allSettled(["KaTeX_Main", "KaTeX_Math", "KaTeX_Size1", "KaTeX_Size2"].map((family) => document.fonts.load(`1em ${family}`)));
  });
  return katexLoad;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

const isReady = () => katex !== null;

// 1. 命中缓存直接返回；2. 排版失败记为 null，同一个错的公式不重复抛错。
function render(tex: string, mode: MathMode): string | null {
  if (!katex) return null;
  const key = `${mode}\u0000${tex}`;
  const hit = cache.get(key);
  if (hit !== undefined) {
    cache.delete(key);
    cache.set(key, hit);
    return hit;
  }
  let html: string | null;
  try {
    const source = mode === "inline-display" ? `\\displaystyle ${tex}` : tex;
    html = katex.renderToString(source, { ...OPTIONS, displayMode: mode === "display" });
  } catch {
    html = null;
  }
  cache.set(key, html);
  if (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value!);
  return html;
}

// 写到一半的 TeX：逐步去掉末尾不完整的部分（半截命令、悬空的 _ ^ \、未配对的 \left 与环境），补齐花括号后再试。
function partialCandidates(tex: string): string[] {
  const candidates: string[] = [];
  let current = tex.replace(/\s+$/u, "");
  for (let step = 0; step < 4 && current; step += 1) {
    candidates.push(closeOpenGroups(current));
    const trimmed = current
      .replace(/\\[a-zA-Z]*$/u, "")
      .replace(/[_^\\&]+$/u, "")
      .replace(/\\(?:left|right|begin|end)\{?[a-z*]*$/u, "")
      .replace(/\s+$/u, "");
    current = trimmed === current ? current.replace(/(?:\\[a-zA-Z]+|\{[^{}]*|[^\s{}])\s*$/u, "").replace(/\s+$/u, "") : trimmed;
  }
  return candidates;
}

function closeOpenGroups(tex: string): string {
  let depth = 0;
  for (let index = 0; index < tex.length; index += 1) {
    if (tex[index] === "\\") { index += 1; continue; }
    if (tex[index] === "{") depth += 1;
    else if (tex[index] === "}" && depth > 0) depth -= 1;
  }
  let closed = tex + "}".repeat(depth);
  const lefts = (closed.match(/\\left(?![a-zA-Z])/gu) ?? []).length - (closed.match(/\\right(?![a-zA-Z])/gu) ?? []).length;
  closed += " \\right.".repeat(Math.max(0, lefts));
  const environments = [...closed.matchAll(/\\(begin|end)\{([a-zA-Z*]+)\}/gu)];
  const open: string[] = [];
  for (const [, kind, name] of environments) {
    if (kind === "begin") open.push(name);
    else if (open.at(-1) === name) open.pop();
  }
  return closed + open.reverse().map((name) => `\\end{${name}}`).join("");
}

function renderPartial(tex: string, mode: MathMode): string | null {
  for (const candidate of partialCandidates(tex)) {
    const html = render(candidate, mode);
    if (html) return html;
  }
  return null;
}

export interface MathNode {
  content?: string;
  raw?: string;
  markup?: string;
  loading?: boolean;
}

function mathMode(node: MathNode, block: boolean): MathMode {
  if (block) return "display";
  return node.markup === "$$" ? "inline-display" : "inline";
}

function sourceText(node: MathNode, block: boolean): string {
  const tex = String(node.content ?? "");
  if (block) return `$$\n${tex}\n$$`;
  return node.markup === "\\(\\)" ? `\\(${tex}\\)` : `${node.markup ?? "$"}${tex}${node.markup ?? "$"}`;
}

/** 公式节点：闭合的直接排版（缓存命中即返回）；生成中的尾部渐进排版并保留上一次成功的结果。 */
export const ReadingMath = memo(function ReadingMath({ node, block }: { node: MathNode; block: boolean }) {
  const ready = useSyncExternalStore(subscribe, isReady, isReady);
  const tex = String(node.content ?? "");
  const mode = mathMode(node, block);
  const lastGood = useRef<string | null>(null);
  useEffect(() => { void preloadMath(); }, []);
  // 1. 闭合：完整排版；排不了（真错误）按源码显示，跟正文同色，不出红字。
  // 2. 未闭合：渐进排版，失败沿用上一次结果。
  let html: string | null = null;
  if (ready) {
    html = node.loading ? renderPartial(tex, mode) ?? lastGood.current : render(tex, mode);
    if (html) lastGood.current = html;
  }
  const pending = Boolean(node.loading);
  if (block) return <MathDisplay html={html} source={sourceText(node, true)} tex={tex} pending={pending} />;
  // 还在写、又一点都排不出来（刚写出 "\fr"）：先不显示，免得源码闪一下再变成公式；它后面还没有文字，不会错位。
  if (!html && pending && ready) return null;
  if (!html) return <span className="reading-math-source" data-pending={pending || undefined}>{sourceText(node, false)}</span>;
  return <span className="reading-math" data-tex={tex} data-mode={mode} data-pending={pending || undefined}
    role="math" aria-label={tex} dangerouslySetInnerHTML={{ __html: html }} />;
}, (previous, next) => previous.block === next.block
  && previous.node.content === next.node.content
  && previous.node.loading === next.node.loading
  && previous.node.markup === next.node.markup);

// 独立公式：生成中只长高不缩回（渐进排版时分式、上下标出现会让高度来回变）；放不下时横滚并给渐隐提示、进 Tab 序。
function MathDisplay({ html, source, tex, pending }: { html: string | null; source: string; tex: string; pending: boolean }) {
  const ref = useRef<HTMLDivElement>(null);
  const tallest = useRef(0);
  useEffect(() => (ref.current ? watchOverflowEdges(ref.current) : undefined), []);
  useLayoutEffect(() => {
    const element = ref.current;
    if (!element) return;
    if (!pending) {
      tallest.current = 0;
      element.style.minHeight = "";
      return;
    }
    tallest.current = Math.max(tallest.current, element.offsetHeight);
    element.style.minHeight = `${tallest.current}px`;
  }, [html, pending]);
  if (!html) {
    // 生成中一点都排不出来时留一行公式高的空位（不闪源码）；KaTeX 未载入或公式有错时显示源码。
    return <div ref={ref} className="reading-math-display" data-pending={pending || undefined} data-empty={pending && katex !== null ? "" : undefined}>
      {pending && katex !== null ? null : <span className="reading-math-source">{source}</span>}
    </div>;
  }
  return <div ref={ref} className="reading-math-display" data-tex={tex} data-pending={pending || undefined}
    role="math" aria-label={tex} dangerouslySetInnerHTML={{ __html: html }} />;
}

// 选区里有公式时，复制得到 $tex$ / $$tex$$ 源码（参考 Claude Desktop）：KaTeX 排出的是逐字拆开的 span，
// 按原样复制只会得到一堆散字。纯文本与 HTML 两份都替换，粘到哪里都是源码。
export function copyMathAsTex(event: ReactClipboardEvent<HTMLElement>): void {
  const selection = window.getSelection();
  if (!selection || selection.isCollapsed || selection.rangeCount === 0) return;
  const range = selection.getRangeAt(0);
  const root = event.currentTarget;
  if (!root.contains(range.commonAncestorContainer)) return;
  if (![...root.querySelectorAll("[data-tex]")].some((element) => range.intersectsNode(element))) return;
  // 1. 选区起止落在公式内部时扩到整个公式，避免复制半个公式
  const startMath = elementOf(range.startContainer)?.closest("[data-tex]");
  const endMath = elementOf(range.endContainer)?.closest("[data-tex]");
  const expanded = range.cloneRange();
  if (startMath) expanded.setStartBefore(startMath);
  if (endMath) expanded.setEndAfter(endMath);
  // 2. 克隆选区，把每个公式换成源码文字
  const fragment = expanded.cloneContents();
  for (const math of fragment.querySelectorAll<HTMLElement>("[data-tex]")) {
    const tex = math.dataset.tex ?? "";
    const display = math.classList.contains("reading-math-display");
    math.replaceWith(document.createTextNode(display ? `\n$$\n${tex}\n$$\n` : math.dataset.mode === "inline-display" ? `$$${tex}$$` : `$${tex}$`));
  }
  // 3. 挂到页外取 innerText，保留段落换行
  const holder = document.createElement("div");
  holder.style.cssText = "position:fixed;left:-99999px;top:0;white-space:pre-wrap";
  holder.append(fragment);
  document.body.append(holder);
  const text = holder.innerText;
  const html = holder.innerHTML;
  holder.remove();
  event.clipboardData.setData("text/plain", text.replace(/\n{3,}/gu, "\n\n").trim());
  event.clipboardData.setData("text/html", html);
  event.preventDefault();
}

function elementOf(node: Node): Element | null {
  return node instanceof Element ? node : node.parentElement;
}

/** 正文里可能有公式时提前加载（流式路径在第一个 $ 出现时就触发）。 */
export function mightContainMath(markdown: string): boolean {
  return /\$|\\\(|\\\[/u.test(markdown);
}
