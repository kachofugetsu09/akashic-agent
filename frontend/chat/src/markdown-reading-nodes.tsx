import { CheckIcon, CopyIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { CodeBlockNode, ListNode, ParagraphNode as ParagraphRenderer, PreCodeNode, type NodeComponentProps } from "markstream-react";
import type { CodeBlockNode as CodeData, ListNode as ListData, ParagraphNode, ParsedNode, TableNode as TableData } from "stream-markdown-parser";
import { copyReadingCode, isConceptLabel, isNumericColumn, isPlainTextLanguage } from "./markdown-reading";

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
    return <li key={key} dir="auto">
      <span className="reading-concept-label">{renderNode(paragraph.children[0], `${key}-label`, ctx)}</span>
      <div className="reading-concept-description">
        {renderNode({ ...paragraph, children: paragraph.children.slice(1) }, `${key}-description`, ctx)}
        {item.children.slice(1).map((child, childIndex) => renderNode(child, `${key}-child-${childIndex}`, ctx))}
      </div>
    </li>;
  })}</ul>;
}

/** 显式对齐优先，整列数值用等宽字体；表格保留行列与 loading 语义。 */
export function ReadingTable({ node, ctx, renderNode, indexKey }: NodeComponentProps<TableData & { loading?: boolean }>) {
  if (!ctx || !renderNode) throw new Error("Table renderer requires its render context");
  const numeric = node.header.cells.map((_header, index) => isNumericColumn(
    node.rows.map((row) => row.cells[index] ? row.cells[index].children.map(nodeText).join("") : ""),
  ));
  return <div className="reading-table">
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

/** 纯文本始终是无标题阅读面，带语言的代码仍使用原高亮与流式路径。 */
export function ReadingCode({ node, ctx, isDark }: NodeComponentProps<CodeData>) {
  if (isPlainTextLanguage(node.language)) {
    return <div className="reading-plain-text">
      <pre><code>{node.code}</code></pre>
      <PlainCodeCopy code={node.code} />
    </div>;
  }
  if (ctx?.renderCodeBlocksAsPre) return <PreCodeNode node={node} showLineNumbers={false} />;
  return <CodeBlockNode node={node} isDark={isDark} loading={Boolean(node.loading)}
    codeBlockOptions={ctx?.codeBlockOptions} onCopy={ctx?.events.onCopy}
    {...ctx?.codeBlockThemes}
    {...ctx?.codeBlockProps}
    stream={ctx?.codeBlockStream}
    {...CODE_BLOCK_PROPS} />;
}
