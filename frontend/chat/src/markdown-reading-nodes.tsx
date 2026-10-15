import { CheckIcon, CopyIcon } from "lucide-react";
import { useEffect, useRef, useState, type RefObject } from "react";
import { CodeBlockNode, ListNode, ParagraphNode as ParagraphRenderer, PreCodeNode, languageMap, resolveLanguageId, type NodeComponentProps } from "markstream-react";
import type { CodeBlockNode as CodeData, ListNode as ListData, ParagraphNode, ParsedNode, TableNode as TableData } from "stream-markdown-parser";
import { copyReadingCode, isConceptLabel, isNumericColumn, isPlainTextLanguage, stripConceptSeparator, watchOverflowEdges, writeOverflowEdges } from "./markdown-reading";

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
export const CODE_BLOCK_OPTIONS = { overflow: "scroll", unsafeCSS: "[data-line] { padding-inline: 16px; }" } as const;

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

// 代码的横滚发生在高亮组件的 shadow DOM 里（生成中则是普通 pre），触屏滚动条是浮层，看不出右边还有字。
function useCodeOverflowEdges(ref: RefObject<HTMLDivElement | null>): void {
  useEffect(() => {
    const wrapper = ref.current;
    if (!wrapper) return;
    let root: ShadowRoot | null = null;
    let scroller: HTMLElement | null = null;
    const resize = new ResizeObserver(() => measure());
    const contents = new MutationObserver(() => measure());
    // 1. 每次测量都重新找滚动元素：高亮完成前后 shadow 里的节点会被替换；没有高亮宿主时量自身的 pre。
    const measure = () => {
      const scope: ParentNode = root ?? wrapper;
      scroller = [...scope.querySelectorAll<HTMLElement>("pre, code")].find((el) => el.scrollWidth > el.clientWidth + 1) ?? null;
      writeOverflowEdges(wrapper, scroller);
    };
    const onScroll = () => measure();
    // 2. 高亮组件可能晚于本组件挂载，等宿主出现再接上滚动与尺寸监听。
    const attach = () => {
      const host = wrapper.querySelector("diffs-container");
      if (!host?.shadowRoot || host.shadowRoot === root) return;
      root?.removeEventListener("scroll", onScroll, true);
      root = host.shadowRoot;
      root.addEventListener("scroll", onScroll, { capture: true, passive: true });
      resize.observe(host);
      contents.observe(root, { childList: true, subtree: true });
      measure();
    };
    // 3. 流式正文增长同样走这里：子树变化后重新接宿主并测量。
    const mounts = new MutationObserver(() => { attach(); measure(); });
    mounts.observe(wrapper, { childList: true, subtree: true, characterData: true });
    wrapper.addEventListener("scroll", onScroll, { capture: true, passive: true });
    resize.observe(wrapper);
    attach();
    measure();
    return () => {
      mounts.disconnect();
      wrapper.removeEventListener("scroll", onScroll, true);
      contents.disconnect();
      resize.disconnect();
      root?.removeEventListener("scroll", onScroll, true);
    };
  }, [ref]);
}

function HighlightedCode(props: NodeComponentProps<CodeData>) {
  const { node, ctx, isDark } = props;
  const ref = useRef<HTMLDivElement>(null);
  useCodeOverflowEdges(ref);
  // 围栏正文自带结尾换行，高亮层会把它渲染成一整行空白。
  const settled = node.code.endsWith("\n") ? { ...node, code: node.code.replace(/\n+$/, "") } : node;
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
  const id = resolveLanguageId(node.language);
  // 映射外的语言按落定块头的规则只把首字母大写（myLang → MyLang）。
  const label = languageMap[id] ?? (id ? `${id.charAt(0).toUpperCase()}${id.slice(1)}` : "");
  return <div ref={ref} className="reading-code-scroll is-streaming">
    {label ? <span className="reading-code-stream-label code-header-title" aria-hidden="true">{label}</span> : null}
    <PreCodeNode node={node} showLineNumbers={false} />
  </div>;
}

/** 纯文本始终是无标题阅读面，带语言的代码仍使用原高亮与流式路径。 */
export function ReadingCode({ node, ctx, isDark }: NodeComponentProps<CodeData>) {
  if (isPlainTextLanguage(node.language)) {
    return <div className="reading-plain-text">
      <pre><code>{node.code}</code></pre>
      <PlainCodeCopy code={node.code} />
    </div>;
  }
  if (ctx?.renderCodeBlocksAsPre) return <StreamingCode node={node} />;
  return <HighlightedCode node={node} ctx={ctx} isDark={isDark} />;
}
