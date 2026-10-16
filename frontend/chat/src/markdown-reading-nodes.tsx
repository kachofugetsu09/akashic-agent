import { getOrCreateWorkerPoolSingleton, type WorkerPoolManager } from "@pierre/diffs/worker";
import { CheckIcon, CopyIcon } from "lucide-react";
import { createContext, useContext, useEffect, useLayoutEffect, useRef, useState, type RefObject } from "react";
import { CodeBlockNode, FootnoteNode as FootnoteRenderer, InlineCodeNode, LinkNode as LinkRenderer, ListNode, ParagraphNode as ParagraphRenderer, PreCodeNode, languageMap, normalizeLanguageIdentifier, resolveLanguageId, type CodeBlockOptions, type NodeComponentProps } from "markstream-react";
import type { CodeBlockNode as CodeData, FootnoteAnchorNode as FootnoteAnchorData, FootnoteNode as FootnoteData, FootnoteReferenceNode as FootnoteReferenceData, LinkNode as LinkData, ListNode as ListData, ParagraphNode, ParsedNode, TableNode as TableData } from "stream-markdown-parser";
import { copyReadingCode, isConceptLabel, isNumericColumn, isPlainTextLanguage, stripConceptSeparator, watchOverflowEdges, writeOverflowEdges } from "./markdown-reading";
import { renderStaticMarkdown } from "./static-markdown";
import { loadKatex } from "./reading-math";
import { DiffLines } from "./tool-result-view";
import { FILE_INTENT, openFileInWorkbench, useWorkbenchAccepts, workspaceFileRef } from "./shell-workbench";

export const CODE_BLOCK_PROPS = {
  showLineNumbers: false,
  showCollapseButton: false,
  showFontSizeButtons: false,
  showExpandButton: false,
  showPreviewButton: false,
  showTooltips: false,
  enableFontSizeControl: false,
  isShowPreview: false,
} as const;

// 代码不折行：折行会把缩进与对齐打散，窄屏读代码更难；长行在代码块自己的区域里横向滚动。
// 流式期间的 <pre> 也不折行（message-view.css），两种状态行数一致，换手时块高不变。
// 高亮组件把行内边距写死为 1ch，代码比块头语言标签左缩约 9px；注入 shadow 样式改为 16px，与块头、生成中的普通 pre 同一左缘。
// vitesse 在纸面底色（--ak-paper-quiet）上有不少 token 低于 4.5:1（浅色：变量 3.3、函数 3.9、标点 2.6；深色：标点 3.0、注释 3.6）。
// 不换主题：按原色相在 OKLab 里只调明度到约 4.6:1；半透明的引号等弱化字符调到 3.5:1，仍比正文浅一级。
// 高亮按深浅一对主题输出，每个 span 的 inline style 同时写着两套色值（--diffs-token-light / -dark），
// 只能按"变量名:色值"匹配并改写同名变量，不能直接改 color（浅色规则会命中深色值，反之亦然）。半透明色以不透明色为前缀，必须排在后面。
const READABLE_TOKENS: Readonly<Record<"light" | "dark", ReadonlyArray<readonly [string, string]>>> = {
  light: [
    ["#B07D48", "#95642F"], ["#59873A", "#4E7B2E"], ["#999999", "#6F6F6F"], ["#2E8F82", "#127D70"],
    ["#998418", "#836E00"], ["#B56959", "#A55B4B"], ["#BDA437", "#866D00"], ["#A0ADA0", "#667266"],
    ["#E36209", "#C44500"], ["#B05A78", "#A95472"], ["#AB5E3F", "#A75A3C"], ["#2E808F", "#257988"],
    ["#22863A", "#198034"], ["#B5695977", "#9C7B71"], ["#99841877", "#8D834F"], ["#393A3490", "#83837E"],
  ],
  dark: [["#666666", "#858585"], ["#6872AB", "#7781BC"], ["#758575DD", "#798979"], ["#B8A96577", "#7A7251"], ["#C98A7D77", "#896A63"]],
};
const readableTokenCSS = (["light", "dark"] as const).flatMap((scheme) => READABLE_TOKENS[scheme].map(([from, to]) =>
  `[style*="--diffs-token-${scheme}:${from}" i] { --diffs-token-${scheme}: ${to} !important; }`)).join(" ");

// 代码高亮放进 Worker 池（参考 Claude Desktop）：分词在主线程上一块就是 50–100ms 的长任务，生成中、落定、
// 打开长会话时都会卡住滚动与输入。池子按核数取 2–4 个；Worker 起不来（CSP、不支持）时高亮器自己退回主线程。
// 池子的主题必须与 markstream 的默认主题同一对，深浅切换只换 themeType，不重建池子。
let highlightPool: WorkerPoolManager | undefined;

function codeHighlightPool(): WorkerPoolManager | undefined {
  if (typeof Worker === "undefined") return undefined;
  highlightPool ??= getOrCreateWorkerPoolSingleton({
    poolOptions: {
      workerFactory: () => new Worker(new URL("@pierre/diffs/worker/worker.js", import.meta.url), { type: "module" }),
      poolSize: Math.max(2, Math.min(4, Math.floor((navigator.hardwareConcurrency || 4) / 2))),
      workerInitializationTimeout: 30_000,
    },
    highlighterOptions: { theme: { dark: "vitesse-dark", light: "vitesse-light" } },
  });
  return highlightPool;
}

// 代码区上 4px、下 12px：44px 块头的标题下方已有留白，代码首行与标题、末行与块底边的距离都约 12–15px（与块头左缩进同一节奏）。
// 流式 pre 与高亮占位按同一几何写（message-view.css），换手时块高不变。
export const CODE_BLOCK_OPTIONS: CodeBlockOptions = {
  overflow: "scroll",
  unsafeCSS: `[data-line] { padding-inline: 16px; } [data-code] { padding-block: 4px 12px; } ${readableTokenCSS}`,
  // 第一个代码块渲染时才建池：markstream 每次渲染代码块都展开这份选项，getter 在那一刻求值，没有代码的会话不起 Worker。
  get workerManager() { return codeHighlightPool(); },
};

/** 展示判断只读可见文本，不解析 raw 或猜测内容的用途。 */
function nodeText(node: ParsedNode): string {
  if ("children" in node && Array.isArray(node.children)) return node.children.map(nodeText).join("");
  if ("content" in node && typeof node.content === "string") return node.content;
  if (node.type === "inline_code") return node.code ?? "";
  return "";
}

function isParagraph(node: ParsedNode): node is ParagraphNode {
  return node.type === "paragraph";
}

/** 独立的全加粗段落强调主题，普通段落保持原排版。 */
export function ReadingParagraph(props: NodeComponentProps<ParagraphNode>) {
  const { node, ctx, renderNode, indexKey } = props;
  const content = node.children.filter((child) => child.type !== "text" || nodeText(child).trim());
  if (content.length !== 1 || content[0].type !== "strong" || !ctx || !renderNode) return <ParagraphRenderer {...props} />;
  const className = nodeText(content[0]).length < 65 ? "reading-topic" : "reading-emphasis";
  return <p className={className} dir="auto">{node.children.map((child, index) => renderNode(child, `${String(indexKey)}-${index}`, ctx))}</p>;
}

// 链接里的行内代码不做成文件按钮：<a> 里不能嵌交互控件。
const InsideLink = createContext(false);

/** 行内代码是文件路径、且工作台有工具接受文件意图时，点一下在工作台里打开（“改动”会定位到该文件）。静态路径见 StaticMessageResponse。 */
export function ReadingInlineCode(props: NodeComponentProps<{ type: "inline_code"; code: string }>) {
  const accepts = useWorkbenchAccepts(FILE_INTENT);
  const insideLink = useContext(InsideLink);
  const file = accepts && !insideLink ? workspaceFileRef(props.node.code) : null;
  if (!file) return <InlineCodeNode {...props} />;
  return <button type="button" className="reading-file-link" title={`在工作台里打开 ${file.path}`} onClick={() => void openFileInWorkbench(file)}>
    <InlineCodeNode {...props} />
  </button>;
}

/** markstream 给链接写死英文 aria-label（"Link: 地址"），它会顶替链接文字成为读屏名：摘掉后读屏念的是正文里的链接文字。 */
export function ReadingLink(props: NodeComponentProps<LinkData>) {
  const ref = useRef<HTMLSpanElement>(null);
  // 必要 effect：属性写在 markstream 内部组件上，只能提交后摘；React 只在值变化时重写属性，每次提交后摘一次即可。
  useLayoutEffect(() => {
    const link = ref.current?.querySelector<HTMLAnchorElement>("a.link-node");
    if (!link) return;
    // 链接文字或图片 alt 能命名时摘掉英文"Link: …"；空文字链接没有别的名称，用网址。
    const named = Boolean(link.textContent?.trim()) || [...link.querySelectorAll("img")].some((image) => image.alt.trim());
    if (named) link.removeAttribute("aria-label");
    else link.setAttribute("aria-label", link.getAttribute("href") ?? "链接");
  });
  return <InsideLink.Provider value={true}><span ref={ref}><LinkRenderer {...props} /></span></InsideLink.Provider>;
}

/** 脚注引用写成真正的链接（markstream 的是带点击的 span，键盘按不到），显示与静态路径同样的上标数字；跳转见 jumpToFootnote。 */
export function ReadingFootnoteReference({ node }: NodeComponentProps<FootnoteReferenceData>) {
  return <sup className="footnote-reference" id={`fnref-${node.id}`}>
    <a className="footnote-link" href={`#fnref--${node.id}`} aria-label={`脚注 ${node.id}`}>{node.id}</a>
  </sup>;
}

/** 脚注末尾的返回箭头：读屏名与静态路径同说"返回引用 1"，不念箭头字符。 */
export function ReadingFootnoteAnchor({ node }: NodeComponentProps<FootnoteAnchorData>) {
  return <a className="footnote-anchor" href={`#fnref-${node.id}`} aria-label={`返回引用 ${node.id}`}>↩</a>;
}

/** 短加粗标签组成的无序列表分栏，其余列表沿用现有 renderer。 */
export function ReadingList(props: NodeComponentProps<ListData>) {
  const { node, ctx, renderNode, indexKey } = props;
  const paragraphs = node.items.map((item) => item.children[0]);
  const concepts = !node.ordered && node.items.length >= 2 && paragraphs.every((paragraph, index) => {
    if (!paragraph || !isParagraph(paragraph)) return false;
    const first = paragraph.children[0];
    return first?.type === "strong" && isConceptLabel(
      nodeText(first),
      paragraph.children.slice(1).map(nodeText).join("") + node.items[index].children.slice(1).map(nodeText).join(""),
    );
  });
  if (!concepts || !ctx || !renderNode) return <ListNode {...props} />;
  return <ul className="reading-concepts" role="list">{node.items.map((item, index) => {
    const paragraph = item.children[0];
    // 上面的结构判断保证每项都是以 strong 开头的 paragraph。
    if (!isParagraph(paragraph)) throw new Error("Concept item must start with a paragraph");
    const key = `${String(indexKey)}-${index}`;
    const description = conceptDescription(paragraph.children.slice(1));
    return <li key={key} dir="auto">
      <span className="reading-concept-label">{renderNode(paragraph.children[0], `${key}-label`, ctx)}</span>
      <div className="reading-concept-description">
        {description.length ? renderNode({ ...paragraph, children: description }, `${key}-description`, ctx) : null}
        {item.children.slice(1).map((child, childIndex) => renderNode(child, `${key}-child-${childIndex}`, ctx))}
      </div>
    </li>;
  })}</ul>;
}

/** 去掉说明开头的冒号；只剩冒号时说明为空，由后续嵌套内容接上。 */
function conceptDescription(children: ParsedNode[]): ParsedNode[] {
  const [first, ...rest] = children;
  if (first?.type !== "text" || typeof first.content !== "string") return children;
  const content = stripConceptSeparator(first.content);
  return content ? [{ ...first, content, raw: content }, ...rest] : rest;
}

/** 显式对齐优先，整列数值用等宽字体；表格保留行列与 loading 语义。 */
export function ReadingTable({ node, ctx, renderNode, indexKey }: NodeComponentProps<TableData & { loading?: boolean }>) {
  if (!ctx || !renderNode) throw new Error("Table renderer requires its render context");
  const numeric = node.header.cells.map((_header, index) => isNumericColumn(
    node.rows.map((row) => row.cells[index] ? row.cells[index].children.map(nodeText).join("") : ""),
  ));
  const ref = useRef<HTMLDivElement>(null);
  // 必要 effect：横滚两侧是否有藏住的列只能在布局后量到；流式追加行由 ResizeObserver 跟上。
  useEffect(() => (ref.current ? watchOverflowEdges(ref.current) : undefined), []);
  return <div ref={ref} className="reading-table">
    <table aria-busy={node.loading}>
      <thead><tr>{node.header.cells.map((cell, index) => <th key={index} dir="auto"
        className={numeric[index] ? "reading-number" : undefined} style={{ textAlign: cell.align ?? (numeric[index] ? "right" : undefined) }}>
        {cell.children.map((child, childIndex) => renderNode(child, `${String(indexKey)}-th-${index}-${childIndex}`, ctx))}
      </th>)}</tr></thead>
      <tbody>{node.rows.map((row, rowIndex) => <tr key={rowIndex}>{row.cells.map((cell, index) => <td key={index} dir="auto"
        className={numeric[index] ? "reading-number" : undefined} style={{ textAlign: cell.align ?? (numeric[index] ? "right" : undefined) }}>
        {cell.children.map((child, childIndex) => renderNode(child, `${String(indexKey)}-row-${rowIndex}-${index}-${childIndex}`, ctx))}
      </td>)}</tr>)}</tbody>
    </table>
    {node.loading ? <span className="sr-only" role="status">表格接收中</span> : null}
  </div>;
}

/** 复制原始代码，反馈与失败都落到按钮本身。 */
function PlainCodeCopy({ code }: { code: string }) {
  const [status, setStatus] = useState<"idle" | "copied" | "error">("idle");
  const timer = useRef<number>(0);
  useEffect(() => () => window.clearTimeout(timer.current), []);
  const label = status === "copied" ? "已复制" : status === "error" ? "复制失败，点击重试" : "复制代码";
  return <button type="button" className="reading-code-copy" aria-label={label} title={label}
    data-copied={status === "copied" ? "true" : undefined}
    onClick={() => {
      void copyReadingCode(code).then(() => {
        setStatus("copied");
        window.clearTimeout(timer.current);
        timer.current = window.setTimeout(() => setStatus("idle"), 1500);
      }).catch((error: unknown) => {
        setStatus("error");
        console.error("复制代码失败", error);
      });
    }}>
    {status === "copied" ? <CheckIcon aria-hidden="true" /> : <CopyIcon aria-hidden="true" />}
  </button>;
}

let shadowFocusRing: CSSStyleSheet | null = null;

// 横滚代码进 Tab 序后（见 writeOverflowEdges）焦点落在 shadow 里，外面的样式够不着、宿主也不匹配 :focus-visible：往 shadow 里补一条焦点框。
function adoptFocusRing(root: ShadowRoot): void {
  shadowFocusRing ??= (() => {
    const sheet = new CSSStyleSheet();
    sheet.replaceSync("[data-scroll-stop]:focus-visible { outline: 2px solid var(--ak-rule-focus); outline-offset: -2px; }");
    return sheet;
  })();
  if (!root.adoptedStyleSheets.includes(shadowFocusRing)) root.adoptedStyleSheets = [...root.adoptedStyleSheets, shadowFocusRing];
}

// 代码的横滚发生在高亮组件的 shadow DOM 里（生成中则是普通 pre），触屏滚动条是浮层，看不出右边还有字。
function useCodeOverflowEdges(ref: RefObject<HTMLDivElement | null>): void {
  useEffect(() => {
    const wrapper = ref.current;
    if (!wrapper) return;
    let root: ShadowRoot | null = null;
    let scroller: HTMLElement | null = null;
    let frame = 0;
    // 测量要读 scrollWidth（强制布局）：流式时每次字符变化都会触发观察者，合并到下一帧只量一次（见 design.md 3.18）。
    // 曾经同步测量，一段带代码的回复生成期间在这里累计约 130ms 强制布局。
    const schedule = () => {
      if (frame === 0) frame = requestAnimationFrame(() => { frame = 0; measure(); });
    };
    const resize = new ResizeObserver(() => schedule());
    const contents = new MutationObserver(() => schedule());
    // 1. 每次测量都重新找滚动元素：高亮完成前后 shadow 里的节点会被替换；没有高亮宿主时量自身的 pre。
    const measure = () => {
      const scope: ParentNode = root ?? wrapper;
      scroller = [...scope.querySelectorAll<HTMLElement>("pre, code")].find((el) => el.scrollWidth > el.clientWidth + 1) ?? null;
      writeOverflowEdges(wrapper, scroller);
    };
    const onScroll = () => schedule();
    // 2. 高亮组件可能晚于本组件挂载，等宿主出现再接上滚动与尺寸监听。
    const attach = () => {
      const host = wrapper.querySelector("diffs-container");
      if (!host?.shadowRoot || host.shadowRoot === root) return;
      root?.removeEventListener("scroll", onScroll, true);
      root = host.shadowRoot;
      adoptFocusRing(root);
      root.addEventListener("scroll", onScroll, { capture: true, passive: true });
      resize.observe(host);
      contents.observe(root, { childList: true, subtree: true });
      schedule();
    };
    // 3. markstream 的 pre（生成中、高亮完成前）一律 tabIndex=0、读屏名写死英文"Code block: json"：
    //    摘掉，交给 writeOverflowEdges 按是否放得下决定进不进 Tab 序、用中文名。
    const adopt = () => {
      for (const pre of wrapper.querySelectorAll<HTMLElement>("pre[aria-label^='Code block']")) {
        pre.removeAttribute("tabindex");
        pre.removeAttribute("aria-label");
      }
    };
    // 4. 流式正文增长同样走这里：子树变化后重新接宿主并测量。
    const mounts = new MutationObserver(() => { adopt(); attach(); schedule(); });
    mounts.observe(wrapper, { childList: true, subtree: true, characterData: true });
    wrapper.addEventListener("scroll", onScroll, { capture: true, passive: true });
    resize.observe(wrapper);
    adopt();
    attach();
    schedule();
    return () => {
      cancelAnimationFrame(frame);
      mounts.disconnect();
      wrapper.removeEventListener("scroll", onScroll, true);
      contents.disconnect();
      resize.disconnect();
      root?.removeEventListener("scroll", onScroll, true);
    };
  }, [ref]);
}

interface CodeSnapshot {
  html: string;
  sheets: CSSStyleSheet[];
  // 高亮宿主自身的属性（字号、配色等 CSS 变量写在它的 style 上，shadow 样式靠 :host 读取）与它在代码区里的位置
  attributes: [string, string][];
  variables: [string, string][];
  top: number;
  left: number;
}

const SNAPSHOT_LIMIT = 40;
const codeSnapshots = new Map<string, CodeSnapshot>();
const HANDOFF_TIMEOUT_MS = 3000;

// 高亮交接：生成中闭合的代码块已经高亮，落定后的消息是新挂载的组件，高亮器要再等几帧（实测约 80ms）
// 才重新上色，这段时间显示的是无色占位——同一块代码"有色 → 无色 → 有色"闪一下。
// 每块代码第一次高亮完成时记下 shadow 里的结果（HTML 与样式表），同一段代码再挂载时先盖上这份快照，
// 新的高亮画稳后撤掉。快照只作视觉占位：inert 且去掉复制来的 tabindex（横滚代码加的焦点入口），不进 Tab 序、不可点。
// 只有生成中的回复（record）才记快照：打开历史会话时没有"同一段代码再挂载"，逐块等稳定、读样式全是白费的布局读取。
function useHighlightHandoff(ref: RefObject<HTMLDivElement | null>, key: string, record: boolean): void {
  useLayoutEffect(() => {
    const wrapper = ref.current;
    const snapshot = codeSnapshots.get(key);
    if (!wrapper || (!record && !snapshot)) return;
    let overlay: HTMLDivElement | null = null;
    let frame = 0;
    const started = performance.now();
    // 高亮器会分几轮重绘（先无色的行、再上色、再按可见区补全）：shadow 里连续 150ms 没有变化才算画好。
    let watched: ShadowRoot | null = null;
    let changedAt = 0;
    const shadowChanges = new MutationObserver(() => { changedAt = performance.now(); });
    const highlighted = () => {
      const host = wrapper.querySelector<HTMLElement>("diffs-container");
      const root = host?.shadowRoot;
      if (!host || !root?.querySelector("[data-line]") || host.offsetHeight === 0) return null;
      if (watched !== root) {
        watched = root;
        changedAt = performance.now();
        shadowChanges.observe(root, { childList: true, subtree: true, characterData: true, attributes: true });
      }
      return performance.now() - changedAt > 150 ? host : null;
    };
    // 1. 还没高亮、又有同一段代码的快照：在代码区盖上快照
    const body = wrapper.querySelector<HTMLElement>(".code-block-body");
    if (!highlighted() && snapshot && body) {
      overlay = document.createElement("div");
      for (const [name, value] of snapshot.attributes) overlay.setAttribute(name, value);
      overlay.classList.add("reading-code-handoff");
      overlay.setAttribute("aria-hidden", "true");
      overlay.removeAttribute("tabindex");
      overlay.inert = true;
      for (const [name, value] of snapshot.variables) overlay.style.setProperty(name, value);
      overlay.style.setProperty("--handoff-top", `${snapshot.top}px`);
      overlay.style.setProperty("--handoff-left", `${snapshot.left}px`);
      const shadow = overlay.attachShadow({ mode: "open" });
      shadow.adoptedStyleSheets = snapshot.sheets;
      shadow.innerHTML = snapshot.html;
      for (const element of shadow.querySelectorAll("[tabindex]")) element.removeAttribute("tabindex");
      body.append(overlay);
    }
    // 2. 逐帧等高亮完成：记快照、撤掉占位
    const poll = () => {
      const host = highlighted();
      if (!host && performance.now() - started < HANDOFF_TIMEOUT_MS) {
        frame = requestAnimationFrame(poll);
        return;
      }
      if (record && host?.shadowRoot && body) {
        const hostRect = host.getBoundingClientRect();
        const bodyRect = body.getBoundingClientRect();
        codeSnapshots.delete(key);
        codeSnapshots.set(key, {
          html: host.shadowRoot.innerHTML,
          sheets: [...host.shadowRoot.adoptedStyleSheets],
          attributes: [...host.attributes].filter((attribute) => attribute.name !== "id" && attribute.name !== "style").map((attribute) => [attribute.name, attribute.value]),
          variables: localVariables(host, body),
          top: hostRect.top - bodyRect.top,
          left: hostRect.left - bodyRect.left,
        });
        if (codeSnapshots.size > SNAPSHOT_LIMIT) codeSnapshots.delete(codeSnapshots.keys().next().value!);
      }
      shadowChanges.disconnect();
      overlay?.remove();
      overlay = null;
    };
    frame = requestAnimationFrame(poll);
    return () => {
      cancelAnimationFrame(frame);
      shadowChanges.disconnect();
      overlay?.remove();
    };
  }, [ref, key, record]);
}

// 高亮样式读的 CSS 变量（字号、配色）定义在宿主与代码区之间的几层元素上，快照挂在代码区上继承不到：
// 取宿主上与代码区取值不同的自定义属性，写到快照宿主上。
function localVariables(host: HTMLElement, body: HTMLElement): [string, string][] {
  const own = getComputedStyle(host);
  const outer = getComputedStyle(body);
  const result: [string, string][] = [];
  for (let index = 0; index < own.length; index += 1) {
    const name = own[index];
    if (!name.startsWith("--")) continue;
    const value = own.getPropertyValue(name);
    if (value !== outer.getPropertyValue(name)) result.push([name, value]);
  }
  for (const name of ["color", "font-family", "font-size", "line-height", "background-color"]) result.push([name, own.getPropertyValue(name)]);
  return result;
}

function HighlightedCode(props: NodeComponentProps<CodeData>) {
  const { node, ctx, isDark } = props;
  const ref = useRef<HTMLDivElement>(null);
  useCodeOverflowEdges(ref);
  // 围栏正文自带结尾换行，高亮层会把它渲染成一整行空白。
  const settled = node.code.endsWith("\n") ? { ...node, code: node.code.replace(/\n+$/, "") } : node;
  useHighlightHandoff(ref, `${isDark ? "dark" : "light"}\u0000${node.language}\u0000${settled.code}`, Boolean(ctx?.renderCodeBlocksAsPre));
  return <div ref={ref} className="reading-code-scroll">
    <CodeBlockNode node={settled} isDark={isDark} loading={Boolean(node.loading)}
      codeBlockOptions={ctx?.codeBlockOptions} onCopy={ctx?.events.onCopy}
      {...ctx?.codeBlockThemes}
      {...ctx?.codeBlockProps}
      stream={ctx?.codeBlockStream}
      {...CODE_BLOCK_PROPS} />
  </div>;
}

// 生成中的代码不高亮，但横滚提示与落定后一致；预留的块头位先写上与落定块头同名的语言，换手时只多出复制钮。
function StreamingCode({ node }: { node: CodeData }) {
  const ref = useRef<HTMLDivElement>(null);
  useCodeOverflowEdges(ref);
  // 块头名按归一后的语言取（bash/sh/zsh → shell → "Shell"），与落定块头同一张表；resolveLanguageId 是给高亮器的
  // 语法 id（shell → zsh），拿它取名生成中会显示 "Zsh"。映射外的语言只把首字母大写（myLang → MyLang）。
  const id = resolveLanguageId(node.language);
  const label = languageMap[normalizeLanguageIdentifier(node.language)] ?? languageMap[id] ?? (id ? `${id.charAt(0).toUpperCase()}${id.slice(1)}` : "");
  return <div ref={ref} className="reading-code-scroll is-streaming">
    {label ? <span className="reading-code-stream-label code-header-title" aria-hidden="true">{label}</span> : null}
    <PreCodeNode node={node} showLineNumbers={false} />
  </div>;
}

/** 纯文本始终是无标题阅读面，带语言的代码仍使用原高亮与流式路径。 */
export function ReadingCode({ node, ctx, isDark }: NodeComponentProps<CodeData>) {
  // markstream 把 ```diff 解析成"改后代码"（code 只剩 + 与上下文行、language 置空），删除行与文件头都丢了；
  // 用原文逐行着色，与工具结果里的 diff 同一外观，复制的也是完整 diff。生成中与落定同一渲染，换手不跳。
  // 还在写的块不给复制钮：复制到的会是半截代码（参考 Claude Desktop 流式期间不渲染复制按钮）。
  if (node.diff) {
    return <div className="reading-plain-text is-diff">
      <div className="reading-diff-header"><span className="code-header-title">Diff</span></div>
      <pre><code><DiffLines lines={node.raw.replace(/\n$/u, "").split("\n")} /></code></pre>
      {node.loading ? null : <PlainCodeCopy code={node.raw} />}
    </div>;
  }
  if (isPlainTextLanguage(node.language)) {
    return <div className="reading-plain-text">
      <pre><code>{node.code}</code></pre>
      {node.loading ? null : <PlainCodeCopy code={node.code} />}
    </div>;
  }
  // 生成中只有末尾还在写的那一块保持纯文本；已闭合的块内容不会再变，直接高亮，落定时不再整批换手。
  if (ctx?.renderCodeBlocksAsPre && node.loading) return <StreamingCode node={node} />;
  return <HighlightedCode node={node} ctx={ctx} isDark={isDark} />;
}

type FootnoteLookup = (label: string) => string | null;

/** 当前这条回复的脚注正文（按标签取），由 MessageResponse 按原文提供。 */
export const FootnoteSource = createContext<FootnoteLookup | null>(null);

const FOOTNOTE_ID_PREFIX = "user-content-fn-";
const FOOTNOTE_LABEL = /\[\^([^\]\s]+)\]/gu;

// 标签按 GFM 的规则归一（空白折叠、大小写不敏感），与 micromark 生成 id 前的处理一致。
const footnoteKey = (label: string) => label.replace(/[\t\n\r ]+/gu, " ").trim().toLowerCase();

// 找标签原始写法时只看正文：围栏代码、行内代码与转义的 \[ 里的 "[^A]" 不是引用。
function proseOnly(markdown: string): string {
  let fence: string | null = null;
  const lines = markdown.split("\n").map((line) => {
    const open = /^ {0,3}(`{3,}|~{3,})/u.exec(line);
    if (fence !== null) {
      if (open && open[1][0] === fence[0] && open[1].length >= fence.length && /^ {0,3}[`~]+[ \t]*$/u.test(line)) fence = null;
      return "";
    }
    if (open) fence = open[1];
    return open ? "" : line;
  });
  return lines.join("\n").replace(/(`+)[\s\S]*?\1/gu, "").replace(/\\\[/gu, "");
}

// micromark 的 id 是小写并 URI 编码过的标签；按正文里第一次真实引用的写法还原，与富渲染的 fnref--标签 用同一个标签。
function footnoteLabels(markdown: string): Map<string, string> {
  const labels = new Map<string, string>();
  for (const match of proseOnly(markdown).matchAll(FOOTNOTE_LABEL)) {
    if (!labels.has(footnoteKey(match[1]))) labels.set(footnoteKey(match[1]), match[1]);
  }
  return labels;
}

// 先按原样找（"a%20b" 本身就是标签），找不到再按 URI 解码找（中文标签被编码成 %E6…）。
function originalLabel(id: string, labels: Map<string, string>): string {
  const raw = labels.get(footnoteKey(id));
  if (raw !== undefined) return raw;
  try { return labels.get(footnoteKey(decodeURIComponent(id))) ?? id; } catch { return id; }
}

// 静态 GFM 没有数学扩展。解析前把 \$ 与 $…$ / $$…$$ 换成私用区占位，GFM 就不会去掉转义、也不会把公式拆成强调；
// 解析后代码与属性里的占位还原成原文，正文里的换成待 KaTeX 渲染的公式节点。公式不跨反引号与 ~~~，不会吃掉代码边界；
// 原文里像 HTML 标签的片段（这条路径按文字显示）整段跳过，属性里的 $ 保持字面。
// 行内公式与 pandoc 同规则：开 $ 后、闭 $ 前不是空白，闭 $ 后不紧跟数字（"$5 和 $10"是价格不是公式）。
const MATH_SOURCE = /<\/?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>\n]*)?\/?>|\\\$|\$\$((?:(?!~~~)[^`])+?)\$\$|\$(?!\s)([^$\n`]+?)(?<!\s)\$(?!\d)/gu;

type MathSlot = { source: string; tex: string; display: boolean };
type MathMarks = { escaped: string; open: string; close: string };

// 占位字符逐条挑原文里没有出现过的私用区字符：原文自带的私用区字符原样保留，不会被当成占位。
function freeMarks(markdown: string): MathMarks {
  const free: string[] = [];
  for (let code = 0xe000; free.length < 3 && code <= 0xf8ff; code += 1) {
    const char = String.fromCharCode(code);
    if (!markdown.includes(char)) free.push(char);
  }
  return { escaped: free[0], open: free[1], close: free[2] };
}

function protectMath(markdown: string): { text: string; slots: MathSlot[]; marks: MathMarks } {
  const slots: MathSlot[] = [];
  const marks = freeMarks(markdown);
  const text = markdown.replace(MATH_SOURCE, (whole: string, display?: string, inline?: string) => {
    if (whole.startsWith("<")) return whole;
    if (whole === "\\$") return marks.escaped;
    slots.push({ source: whole, tex: display ?? inline ?? "", display: display !== undefined });
    return `${marks.open}${slots.length - 1}${marks.close}`;
  });
  return { text, slots, marks };
}

// 占位的匹配式：raw 用于文本与 title/alt，encoded 用于被 URI 编码过的 href/src。
function slotPattern(marks: MathMarks, encode: (char: string) => string): RegExp {
  return new RegExp(`${encode(marks.open)}(\\d+)${encode(marks.close)}|${encode(marks.escaped)}`, "giu");
}

// 链接地址、图片地址与 title/alt 只能回到原文（\$ 在这些位置本就解成 $）。
function restoreMathAttributes(root: DocumentFragment, slots: MathSlot[], marks: MathMarks): void {
  const raw = slotPattern(marks, (char) => char);
  const encoded = slotPattern(marks, encodeURIComponent);
  for (const element of root.querySelectorAll("[href], [src], [title], [alt]")) {
    for (const name of ["href", "src", "title", "alt"]) {
      const value = element.getAttribute(name);
      if (value === null) continue;
      const restore = (asUri: boolean) => (_whole: string, index?: string) => {
        if (index === undefined) return "$";
        return asUri ? encodeURI(slots[Number(index)].source) : slots[Number(index)].source;
      };
      const next = value.replace(encoded, restore(true)).replace(raw, restore(false));
      if (next !== value) element.setAttribute(name, next);
    }
  }
}

// 占位还原：代码里回到原文；正文里转义的 $ 成为字面 $，公式成为 data-math 节点（先显示原文，挂载后再渲染）。
function restoreMath(root: DocumentFragment, slots: MathSlot[], marks: MathMarks): void {
  restoreMathAttributes(root, slots, marks);
  const pattern = slotPattern(marks, (char) => char);
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  const texts: Text[] = [];
  for (let node = walker.nextNode(); node; node = walker.nextNode()) {
    const value = node.textContent ?? "";
    if (value.includes(marks.escaped) || value.includes(marks.open)) texts.push(node as Text);
  }
  for (const text of texts) {
    const source = text.textContent ?? "";
    if (text.parentElement?.closest("pre, code")) {
      text.textContent = source.replace(pattern, (_whole: string, index?: string) => index === undefined ? "\\$" : slots[Number(index)].source);
      continue;
    }
    const parts: (string | Node)[] = [];
    let cursor = 0;
    for (const match of source.matchAll(pattern)) {
      parts.push(source.slice(cursor, match.index));
      const slot = match[1] === undefined ? null : slots[Number(match[1])];
      if (slot === null) parts.push("$");
      else {
        const math = document.createElement("span");
        math.dataset.math = slot.display ? "display" : "inline";
        math.dataset.tex = slot.tex;
        math.textContent = slot.source;
        parts.push(math);
      }
      cursor = match.index + match[0].length;
    }
    text.replaceWith(...parts, source.slice(cursor));
  }
}

// 脚注子树换成富渲染的 id 体系：引用指向 fnref--标签、显示标签；GFM 的回链换成一个指向本条引用的回链。
function adoptFootnoteItem(item: HTMLLIElement, label: string, labels: Map<string, string>): string {
  for (const copy of item.querySelectorAll("[data-static-code-copy]")) copy.remove();
  for (const back of item.querySelectorAll("a[data-footnote-backref]")) back.remove();
  for (const ref of item.querySelectorAll<HTMLAnchorElement>("a[data-footnote-ref]")) {
    const target = originalLabel((ref.getAttribute("href") ?? "").replace(`#${FOOTNOTE_ID_PREFIX}`, ""), labels);
    ref.setAttribute("href", `#fnref--${target}`);
    ref.id = `fnref-${target}`;
    ref.textContent = target;
  }
  const back = document.createElement("a");
  back.href = `#fnref-${label}`;
  back.className = "data-footnote-backref";
  back.dataset.footnoteBackref = "";
  back.setAttribute("aria-label", `返回引用 ${label}`);
  back.textContent = "↩";
  const last = item.lastElementChild;
  if (last?.tagName !== "P") item.append(back);
  else {
    // 去掉原回链留下的空白，箭头与句末只隔一个空格。
    const tail = last.lastChild;
    if (tail?.nodeType === Node.TEXT_NODE) tail.textContent = (tail.textContent ?? "").trimEnd();
    last.append(" ", back);
  }
  return item.innerHTML;
}

// 整条原文走一次完整 GFM：围栏、链接引用声明、嵌套引用都按标准解析，不自己截段。
function parseFootnotes(markdown: string): Map<string, string> {
  const items = new Map<string, string>();
  if (!markdown.includes("[^")) return items;
  const labels = footnoteLabels(markdown);
  const { text, slots, marks } = protectMath(markdown);
  const template = document.createElement("template");
  template.innerHTML = renderStaticMarkdown(text);
  restoreMath(template.content, slots, marks);
  for (const item of template.content.querySelectorAll<HTMLLIElement>(`li[id^="${FOOTNOTE_ID_PREFIX}"]`)) {
    const label = originalLabel(item.id.slice(FOOTNOTE_ID_PREFIX.length), labels);
    items.set(footnoteKey(label), adoptFootnoteItem(item, label, labels));
  }
  return items;
}

const footnoteCache = new Map<string, Map<string, string>>();

/** 惰性：真的渲染到脚注时才解析，同一原文的结果缓存，多条脚注共用一次解析。
 *  只给落定的回复用（生成中由 markstream 自己渲染脚注），流式追加不会让长回复每次整条重算。 */
export function footnoteLookup(markdown: string): FootnoteLookup {
  return (label) => {
    let items = footnoteCache.get(markdown);
    if (!items) {
      items = parseFootnotes(markdown);
      footnoteCache.set(markdown, items);
      if (footnoteCache.size > 8) footnoteCache.delete(footnoteCache.keys().next().value!);
    }
    return items.get(footnoteKey(label)) ?? null;
  };
}

// 公式节点挂载后按需加载 KaTeX 渲染；代码里的 $ 在解析阶段已还原成原文，这里不会碰到。
async function renderFootnoteMath(root: HTMLElement): Promise<void> {
  const nodes = [...root.querySelectorAll<HTMLElement>("[data-math]")];
  if (!nodes.length) return;
  const katex = await loadKatex();
  if (!katex) return;
  for (const node of nodes) {
    if (node.isConnected) katex.render(node.dataset.tex ?? "", node, { displayMode: node.dataset.math === "display", throwOnError: false });
  }
}

type FocusedLink = { href: string; index: number; element: Element };

// 生成中脚注由 markstream 渲染，落定换成完整 GFM 的子树；焦点在脚注里的链接上时旧节点被整体换掉，焦点会退回 body。
// 记下脚注内获得焦点的链接（同 href 的第几个），换手后把焦点交还给新子树里对应的那个。
function useFootnoteFocusCarry(id: string, html: string | null): void {
  const focused = useRef<FocusedLink | null>(null);
  useEffect(() => {
    const onFocusIn = (event: FocusEvent) => {
      const target = event.target instanceof Element ? event.target.closest("a[href]") : null;
      const host = target?.closest(`[id="fnref--${CSS.escape(id)}"]`);
      if (!target || !host) { focused.current = null; return; }
      const href = target.getAttribute("href") ?? "";
      const same = [...host.querySelectorAll("a[href]")].filter((link) => link.getAttribute("href") === href);
      focused.current = { href, index: same.indexOf(target), element: target };
    };
    document.addEventListener("focusin", onFocusIn);
    return () => document.removeEventListener("focusin", onFocusIn);
  }, [id]);
  // 必要 effect：焦点只能在新子树挂上之后交还；仍连着的旧节点（没有换手）不动。
  useLayoutEffect(() => {
    const last = focused.current;
    if (!last || last.element.isConnected) return;
    const active = document.activeElement;
    if (active && active !== document.body) return;
    const host = document.querySelector(`[id="fnref--${CSS.escape(id)}"]`);
    const same = host ? [...host.querySelectorAll<HTMLElement>("a[href]")].filter((link) => link.getAttribute("href") === last.href) : [];
    same[last.index]?.focus({ preventScroll: true });
  }, [html, id]);
}

/** markstream 的脚注只收段落：列表被拍成段落，代码与其后的返回链接丢掉。脚注正文改用同一原文的完整 GFM 解析结果；
 *  原文里找不到（生成中定义还没写完、未被引用的定义）时退回 markstream 自己的渲染。 */
export function ReadingFootnote(props: NodeComponentProps<FootnoteData>) {
  const lookup = useContext(FootnoteSource);
  const html = lookup?.(props.node.id) ?? null;
  const bodyRef = useRef<HTMLDivElement>(null);
  useFootnoteFocusCarry(props.node.id, html);
  // 必要 effect：公式只能在 HTML 挂上后按需加载 KaTeX 补渲染。
  useEffect(() => {
    if (bodyRef.current) void renderFootnoteMath(bodyRef.current);
  }, [html]);
  if (html === null) return <FootnoteRenderer {...props} />;
  return <div id={`fnref--${props.node.id}`} className="footnote-node reading-footnote">
    <div ref={bodyRef} className="reading-footnote__body" dangerouslySetInnerHTML={{ __html: html }} />
  </div>;
}
