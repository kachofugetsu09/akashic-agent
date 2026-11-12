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
import ini from "highlight.js/lib/languages/ini";
import yaml from "highlight.js/lib/languages/yaml";
import sql from "highlight.js/lib/languages/sql";
import diff from "highlight.js/lib/languages/diff";
import markdown from "highlight.js/lib/languages/markdown";

const grammars = {
  typescript, javascript, python, json, bash, css, xml, go, rust, ini, yaml, sql, diff, markdown,
};

export type GrammarName = keyof typeof grammars;

self.onmessage = (event: MessageEvent<{ language: GrammarName; code: string }>) => {
  try {
    const { language, code } = event.data;
    hljs.registerLanguage("xml", xml);
    if (language !== "xml") hljs.registerLanguage(language, grammars[language]);
    const html = hljs.highlight(code, { language, ignoreIllegals: true }).value;
    self.postMessage(html);
  } catch {
    self.postMessage(null);
  }
};
