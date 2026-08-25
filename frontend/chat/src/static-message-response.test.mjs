import assert from "node:assert/strict";
import test from "node:test";

import { renderStaticMarkdown } from "./static-markdown.ts";

test("settled desktop Markdown preserves GFM and fenced code", () => {
  const html = renderStaticMarkdown("## 标题\n\n- 一\n- 二\n\n```ts\nconst value = 1;\n```\n\n| A | B |\n| - | - |\n| 1 | 2 |");
  assert.match(html, /<h2>标题<\/h2>/);
  assert.match(html, /<ul>[\s\S]*?<li>一<\/li>/);
  assert.match(html, /class="static-code-block"/);
  assert.match(html, /data-static-code-copy/);
  assert.match(html, /<code class="language-ts">const value = 1;/);
  assert.match(html, /<table>/);
});

test("settled desktop Markdown keeps raw HTML and unsafe links inert", () => {
  const html = renderStaticMarkdown('<script>alert(1)</script>\n\n[危险](javascript:alert(1))');
  assert.doesNotMatch(html, /<script>/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /<a href="">危险<\/a>/);
});
