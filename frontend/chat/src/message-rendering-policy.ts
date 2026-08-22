export interface MessageRenderingFeatures {
  code: boolean;
  math: boolean;
  mermaid: boolean;
}

const fencedCodePattern = /^\s{0,3}(`{3,}|~{3,})([^\n]*)$/gm;
const blockMathPattern = /(^|\n)\s*\$\$[\s\S]*?\$\$\s*(?=\n|$)/m;
const inlineMathPattern = /(^|[^\\$])\$[^\s$](?:[^$\n]*?[^\s$])?\$(?!\$)/m;
const markdownSyntaxPattern = /(^|\n)\s{0,3}(?:#{1,6}\s|>|[-+*]\s|\d+[.)]\s|```|~~~|(?:-{3,}|_{3,}|\*{3,})\s*$)|!?(?:\[[^\]]+\]\([^\n)]+\))|`[^`\n]+`|\*\*|__|~~|https?:\/\/|<\/?[a-z][^>]*>/im;

/** Detect rich Markdown features so expensive renderers load only when needed. */
export function detectMessageRenderingFeatures(markdown: string): MessageRenderingFeatures {
  let code = false;
  let mermaid = false;
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
    const language = info.split(/\s+/, 1)[0]?.toLowerCase();
    if (language === "mermaid") mermaid = true;
    else code = true;
  }
  return {
    code,
    math: blockMathPattern.test(markdown) || inlineMathPattern.test(markdown),
    mermaid,
  };
}

/** Keep ordinary chat text on the zero-parser path while preserving Markdown semantics. */
export function messageNeedsMarkdown(markdown: string) {
  const features = detectMessageRenderingFeatures(markdown);
  return features.code || features.math || features.mermaid || markdownSyntaxPattern.test(markdown);
}
