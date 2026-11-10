import { memo, useEffect, useLayoutEffect, useRef, useSyncExternalStore } from "react";
import type { KatexOptions } from "katex";
import katexAssetUrl from "katex/dist/katex.mjs?url";
import katexStyleUrl from "katex/dist/katex.min.css?url";
import { watchOverflowEdges } from "./markdown-reading";

// 公式渲染（参考 Claude Desktop：KaTeX 按需加载、errorColor 跟正文、复制还原 TeX）。
// 性能要点：KaTeX 排一个公式约 0.1–0.3ms，贵的是每次提交把所有公式重排一遍——这里按"源码 + 模式"缓存 HTML，
// 闭合的公式整条消息里只排一次；末尾还在写的公式每次提交试排一次，失败就截掉写到一半的命令再试，
// 仍不行就保留上一次排成的样子，不闪回源码、不出现 Loading。

type Katex = typeof import("katex").default;
type MathMode = "inline" | "display" | "inline-display";

let katex: Katex | null = null;
let katexLoad: Promise<void> | null = null;
let retryDelay = 2000;
let katexAttempts = 0;
const listeners = new Set<() => void>();

const CACHE_LIMIT = 400;
const cache = new Map<string, string | null>();

// 只输出 HTML（不带 MathML）：DOM 少两到三成、排版快近一半，复制也不会混进 MathML 的逐字文本；读屏名用 TeX 源码。
const OPTIONS: KatexOptions = { throwOnError: true, strict: false, output: "html", trust: false };

/** 检测到公式就开始加载 KaTeX、样式和常用字体：字体晚到会让排好的公式整块重排。
 *  加载失败（断网、发版后旧 chunk 不在）不缓存失败结果：已挂的公式先按源码显示，退避重试，新公式挂载也会再试；成功后全部排版。 */
export function preloadMath(): Promise<void> {
  installMathCopy();
  katexLoad ??= Promise.all([importKatex(), loadKatexStyles()]).then(async ([module]) => {
    katex = module.default;
    for (const listener of listeners) listener();
    await Promise.allSettled(["KaTeX_Main", "KaTeX_Math", "KaTeX_Size1", "KaTeX_Size2"].map((family) => document.fonts.load(`1em ${family}`)));
  }, () => {
    katexLoad = null;
    if (listeners.size) window.setTimeout(() => void preloadMath(), retryDelay);
    retryDelay = Math.min(retryDelay * 2, 60_000);
  });
  return katexLoad;
}

/** 等 KaTeX 载入；这一次没载入成功返回 null（之后仍会重试）。 */
export async function loadKatex(): Promise<Katex | null> {
  await preloadMath();
  return katex;
}

// 首次走正常的分包导入（与 markstream 共用同一 chunk）。失败后浏览器把这个模块地址的失败记在模块表里，
// 同一地址再导入直接复用失败结果、不发请求；重试改从自包含的 katex.mjs 资源导入，每次换查询串重新请求。
function importKatex(): Promise<typeof import("katex")> {
  katexAttempts += 1;
  if (katexAttempts === 1) return import("katex");
  return import(/* @vite-ignore */ `${katexAssetUrl}?retry=${katexAttempts}`) as Promise<typeof import("katex")>;
}

// 样式自己挂 link，加载成功才算就绪：Vite 的 CSS 预载失败后仍把那个地址记成已加载，再次导入直接放行，
// 公式会排出来却没有样式。失败的 link 移除，重试换查询串重新请求。
function loadKatexStyles(): Promise<void> {
  const current = document.querySelector<HTMLLinkElement>("link[data-katex-styles]");
  if (current?.sheet) return Promise.resolve();
  current?.remove();
  return new Promise((resolve, reject) => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.dataset.katexStyles = "";
    link.href = katexAttempts <= 1 ? katexStyleUrl : `${katexStyleUrl}?retry=${katexAttempts}`;
    link.onload = () => resolve();
    link.onerror = () => {
      link.remove();
      reject(new Error("KaTeX 样式加载失败"));
    };
    document.head.append(link);
  });
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
  // 2. 未闭合：渐进排版，失败沿用上一次提交的结果（提交后才记下，被弃置的渲染不会污染它）。
  let html: string | null = null;
  if (ready) html = node.loading ? renderPartial(tex, mode) ?? lastGood.current : render(tex, mode);
  useLayoutEffect(() => { if (html) lastGood.current = html; }, [html]);
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
// 挂在 document 上统一处理（与图片查看器同一委托方式）：选区可以跨多条消息，单条回复上的处理器接不到。
let copyInstalled = false;
function installMathCopy(): void {
  if (copyInstalled) return;
  copyInstalled = true;
  document.addEventListener("copy", copyMathAsTex);
}

function copyMathAsTex(event: ClipboardEvent): void {
  const selection = window.getSelection();
  if (event.defaultPrevented || !event.clipboardData || !selection || selection.isCollapsed || selection.rangeCount === 0) return;
  // 1. 选区起止落在公式内部时扩到整个公式，避免复制半个公式
  const expanded = selection.getRangeAt(0).cloneRange();
  const startMath = elementOf(expanded.startContainer)?.closest("[data-tex]");
  const endMath = elementOf(expanded.endContainer)?.closest("[data-tex]");
  if (startMath) expanded.setStartBefore(startMath);
  if (endMath) expanded.setEndAfter(endMath);
  // 2. 克隆选区，把每个公式换成源码文字；选区里没有公式就交给浏览器
  const fragment = expanded.cloneContents();
  const maths = fragment.querySelectorAll<HTMLElement>("[data-tex]");
  if (!maths.length) return;
  for (const math of maths) {
    const tex = math.dataset.tex ?? "";
    const display = math.classList.contains("reading-math-display");
    math.replaceWith(document.createTextNode(display ? `\n$$\n${tex}\n$$\n` : math.dataset.mode === "inline-display" ? `$$${tex}$$` : `$${tex}$`));
  }
  // 3. 挂到页外取 innerText，保留段落、列表、表格的分隔；克隆的渲染外壳带 content-visibility:auto，
  //    页外会被跳过渲染、读不到字，由 .reading-copy-holder 规则（message-view.css）统一改回可见
  const holder = document.createElement("div");
  holder.className = "reading-copy-holder";
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
