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
