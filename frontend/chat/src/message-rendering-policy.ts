export interface MessageRenderingFeatures {
  code: boolean;
  math: boolean;
  mermaid: boolean;
}

const fencedCodePattern = /^\s*```+\s*([^\s`]*)/gm;
const blockMathPattern = /(^|\n)\s*\$\$[\s\S]*?\$\$\s*(?=\n|$)/m;
const inlineMathPattern = /(^|[^\\$])\$[^\s$](?:[^$\n]*?[^\s$])?\$(?!\$)/m;
const markdownSyntaxPattern = /(^|\n)\s{0,3}(?:#{1,6}\s|>|[-+*]\s|\d+[.)]\s|```|~~~|(?:-{3,}|_{3,}|\*{3,})\s*$)|!?(?:\[[^\]]+\]\([^\n)]+\))|`[^`\n]+`|\*\*|__|~~|https?:\/\/|<\/?[a-z][^>]*>/im;

/** Detect rich Markdown features so expensive renderers load only when needed. */
export function detectMessageRenderingFeatures(markdown: string): MessageRenderingFeatures {
  let code = false;
  let mermaid = false;
  let insideFence = false;
  for (const match of markdown.matchAll(fencedCodePattern)) {
    if (insideFence) {
      insideFence = false;
      continue;
    }
    insideFence = true;
    if (match[1]?.toLowerCase() === "mermaid") mermaid = true;
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
