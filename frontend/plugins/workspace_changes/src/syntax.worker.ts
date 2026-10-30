import hljs from "highlight.js/lib/core";
import typescript from "highlight.js/lib/languages/typescript";
import javascript from "highlight.js/lib/languages/javascript";
import python from "highlight.js/lib/languages/python";
import json from "highlight.js/lib/languages/json";
import bash from "highlight.js/lib/languages/bash";
import css from "highlight.js/lib/languages/css";
import xml from "highlight.js/lib/languages/xml";
import go from "highlight.js/lib/languages/go";
import rust from "highlight.js/lib/languages/rust";

const grammars = { typescript, javascript, python, json, bash, css, xml, go, rust };
export interface SyntaxRow { kind: string; text: string }

// Each hunk starts with unknown context. Never leak old-side grammar state into the new side,
// or across omitted lines. This is fragment highlighting, not a reconstruction of the file.
self.onmessage = (event: MessageEvent<{ language: keyof typeof grammars; rows: SyntaxRow[] }>) => {
  try {
    const { language, rows } = event.data;
    // JSX/TSX grammars embed XML. No language detection and no unrelated grammars compiled.
    hljs.registerLanguage("xml", xml);
    if (language !== "xml") hljs.registerLanguage(language, grammars[language]);
    const output: { indices: number[]; html: string }[] = [];
    let start = 0;
    while (start < rows.length) {
      let end = start + 1;
      while (end < rows.length && rows[end].kind !== "hunk") end++;
      for (const side of ["old", "new"]) {
        const indices: number[] = [];
        for (let i = start; i < end; i++) {
          if (rows[i].kind === "ctx" || rows[i].kind === (side === "old" ? "del" : "add")) indices.push(i);
        }
        if (!indices.length) continue;
        const code = indices.map(i => rows[i].text.slice(1)).join("\n");
        output.push({ indices, html: hljs.highlight(code, { language, ignoreIllegals: true }).value });
      }
      start = end;
    }
    self.postMessage(output);
  } catch { self.postMessage(null); }
};
