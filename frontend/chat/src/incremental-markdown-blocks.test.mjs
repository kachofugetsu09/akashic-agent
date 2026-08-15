import assert from "node:assert/strict";
import test from "node:test";
import { parseMarkdownIntoBlocks } from "streamdown";

import { IncrementalMarkdownBlocks } from "./incremental-markdown-blocks.ts";

function paragraphParser(calls) {
  return (source) => {
    calls.push(source);
    return source.match(/[\s\S]*?(?:\n\n|$)/gu)?.filter(Boolean) ?? [];
  };
}

test("append-only Markdown reparses a bounded block tail", () => {
  const calls = [];
  const parser = new IncrementalMarkdownBlocks(paragraphParser(calls));
  const paragraphs = Array.from({ length: 40 }, (_, index) => `第 ${index} 段包含一些稳定文字。`);
  let source = "";

  for (const paragraph of paragraphs) {
    source += `${paragraph}\n\n`;
    parser.parse(source, true);
  }

  const metrics = parser.metrics();
  assert.ok(metrics.frozenBlocks > 30);
  assert.ok(metrics.maxTailCharacters < 100);
  assert.ok(metrics.parsedCharacters < source.length * 5);
  assert.ok(calls.slice(5).every((tail) => !tail.includes("第 0 段")));
});

test("non-append replacement starts a new generation", () => {
  const parser = new IncrementalMarkdownBlocks(paragraphParser([]));
  parser.parse("alpha\n\nbeta\n\ngamma", true);
  const replaced = parser.parse("completely\n\ndifferent", true);

  assert.deepEqual(replaced, ["completely\n\n", "different"]);
  assert.equal(parser.metrics().generation, 1);
});

test("terminal mode performs one full healing parse", () => {
  const calls = [];
  const parser = new IncrementalMarkdownBlocks(paragraphParser(calls));
  const source = "alpha\n\nbeta\n\ngamma\n\ndelta";
  parser.parse(source.slice(0, 20), true);
  parser.parse(source, true);
  const settled = parser.parse(source, false);

  assert.deepEqual(settled.join(""), source);
  assert.equal(calls.at(-1), source);
  assert.equal(parser.metrics().generation, 1);
});

test("Streamdown block boundaries preserve the exact accumulated source", () => {
  const parser = new IncrementalMarkdownBlocks(parseMarkdownIntoBlocks);
  const parts = ["# 标题\n\n", "第一段。\n\n", "- one\n- two\n\n", "> 引用\n\n", "结尾"];
  let source = "";
  for (const part of parts) {
    source += part;
    assert.equal(parser.parse(source, true).join(""), source);
  }
  assert.equal(parser.parse(source, false).join(""), source);
});
