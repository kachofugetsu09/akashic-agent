export interface MessageRenderingFeatures {
  code: boolean;
  math: boolean;
  /** 脚注定义里有列表、代码、引用等块结构（富渲染路径表达不了，见 hasBlockFootnote）。 */
  blockFootnote: boolean;
}

const fencedCodePattern = /^\s{0,3}(`{3,}|~{3,})([^\n]*)$/gm;
const blockMathPattern = /(^|\n)\s*\$\$[\s\S]*?\$\$\s*(?=\n|$)/m;
const inlineMathPattern = /(^|[^\\$])\$[^\s$](?:[^$\n]*?[^\s$])?\$(?!\$)/m;
const footnoteDefinitionPattern = /^\s{0,3}\[\^[^\]\s]+\]:/;
const footnoteBlockPattern = /^(?: {4}|\t)\s{0,3}(?:[-+*]\s|\d+[.)]\s|```|~~~|>|#{1,6}\s|\|)/;
const markdownSyntaxPattern = /(^|\n)\s{0,3}(?:#{1,6}\s|>|[-+*]\s|\d+[.)]\s|```|~~~|(?:-{3,}|_{3,}|\*{3,})\s*$)|!?(?:\[[^\]]+\]\([^\n)]+\))|`[^`\n]+`|\*\*|__|~~|https?:\/\/|<\/?[a-z][^>]*>/im;

/** Detect rich Markdown features so expensive renderers load only when needed. */
export function detectMessageRenderingFeatures(markdown: string): MessageRenderingFeatures {
  let code = false;
  let openFence: { marker: string; length: number } | undefined;
  for (const match of markdown.matchAll(fencedCodePattern)) {
    const fence = match[1];
    const marker = fence[0];
    const info = match[2].trim();
    if (openFence) {
      if (marker === openFence.marker && fence.length >= openFence.length && info === "") {
        openFence = undefined;
      }
      continue;
    }
    if (marker === "`" && info.includes("`")) continue;
    openFence = { marker, length: fence.length };
    code = true;
  }
  return {
    code,
    math: blockMathPattern.test(markdown) || inlineMathPattern.test(markdown),
    blockFootnote: hasBlockFootnote(markdown),
  };
}

// markstream 的脚注只收段落：脚注里的列表被拍成段落，代码块和其后的返回链接直接丢掉。
// 只看脚注定义的缩进续行，正文里的同类结构不算。
function hasBlockFootnote(markdown: string): boolean {
  let inside = false;
  for (const line of markdown.split("\n")) {
    if (footnoteDefinitionPattern.test(line)) {
      inside = true;
      continue;
    }
    if (!inside || !line.trim()) continue;
    if (footnoteBlockPattern.test(line)) return true;
    if (!/^(?: {4}|\t)/.test(line)) inside = false;
  }
  return false;
}

/** Keep ordinary chat text on the zero-parser path while preserving Markdown semantics. */
export function messageNeedsMarkdown(markdown: string) {
  const features = detectMessageRenderingFeatures(markdown);
  return features.code || features.math || markdownSyntaxPattern.test(markdown);
}

/** Batch only append-only stream growth; replacements must render immediately. */
export function canBatchStreamingMarkdown(
  previous: string,
  next: string,
  batchCharacters: number,
) {
  return batchCharacters > 1
    && next.startsWith(previous)
    && next.length - previous.length < batchCharacters;
}

/** 工具值只解析完整 JSON 对象或数组；普通字符串保持字面含义。 */
export function readToolData(value: unknown): unknown {
  if (typeof value !== "string" || !/^\s*[\[{]/u.test(value)) return value;
  try { return JSON.parse(value) as unknown; }
  catch (error) { if (error instanceof SyntaxError) return value; throw error; }
}

/** 复制保留原字符串；结构化工具值使用完整 JSON，不把空值丢掉。 */
export function toolDataText(value: unknown): string {
  if (value === undefined) return "";
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}
