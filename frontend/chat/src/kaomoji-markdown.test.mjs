import assert from "node:assert/strict";
import test from "node:test";
import kaomojiCollection from "kaomoji-collection/kaomoji.json" with { type: "json" };
import { getMarkdown, parseMarkdownToStructure } from "stream-markdown-parser";
import { configureKaomojiMarkdown, readKaomojiLiteral } from "./kaomoji-markdown.ts";

const FORMATTING_NODES = new Set([
  "emphasis",
  "highlight",
  "inline_code",
  "insert",
  "strong",
  "strikethrough",
  "subscript",
  "superscript",
]);

test("kaomoji stay literal without swallowing real Markdown", () => {
  for (const value of [
    "(=^・・^=)",
    "(*^▽^*)",
    "(*´꒳`*)",
    "(T_T)",
    "^_^",
    "*( ᵕ̤ᴗᵕ̤ )*",
    "☆_.｡.o(≧▽≦)o.｡.:_☆",
    "꒰*´∀`*꒱",
    "ฅ^•ω•^ฅ",
  ]) {
    const markdown = configureKaomojiMarkdown(getMarkdown(`kaomoji-${value}`));
    const nodes = parseMarkdownToStructure(value, markdown, { final: false });
    assert.equal(hasFormattingNode(nodes), false, value);
    assert.equal(
      [...walkNodes(nodes)].some((node) => node.type === "kaomoji_literal" && node.content === value),
      true,
      value,
    );
  }

  assert.equal(readKaomojiLiteral("(see *important*)", 0), undefined);
  assert.equal(readKaomojiLiteral("(这是 *重点*)", 0), undefined);
  for (const markdown of [
    "A --- B",
    "x^2^",
    "[link](https://example.com)",
    "`code`",
    "$x^2$",
    "正文 (*^▽^*) **重点**",
    "这是 *重点* ☆",
    "中文 **重点** ٩",
    "前缀 ~~删除~~ ♥",
    "数学 *x* Ω",
    "☆ ツ *シ* ☆",
    "★ シ **ツ** ★",
    "☆ ノ ~~ツ~~ ☆",
    "☆_.｡.o(≧",
    "꒰*",
  ]) {
    assert.equal(readKaomojiLiteral(markdown, 0), undefined, markdown);
  }
  const prose = configureKaomojiMarkdown(getMarkdown("kaomoji-prose"));
  assert.equal(hasNodeType(parseMarkdownToStructure("(see *important*)", prose, { final: true }), "emphasis"), true);
  for (const [markdown, type] of [
    ["这是 *重点* ☆", "emphasis"],
    ["中文 **重点** ٩", "strong"],
    ["前缀 ~~删除~~ ♥", "strikethrough"],
    ["数学 *x* Ω", "emphasis"],
    ["☆ ツ *シ* ☆", "emphasis"],
    ["★ シ **ツ** ★", "strong"],
    ["☆ ノ ~~ツ~~ ☆", "strikethrough"],
  ]) {
    assert.equal(hasNodeType(parseMarkdownToStructure(markdown, prose, { final: true }), type), true, markdown);
  }
});

test("kaomoji rule leaves code spans to Markdown", () => {
  const markdown = configureKaomojiMarkdown(getMarkdown("kaomoji-code-spans"));

  for (const [source, code] of [
    ["`(*^▽^*)`", "(*^▽^*)"],
    ["`` (*^▽^*) ``", "(*^▽^*)"],
    ["prefix `(*^▽^*)` suffix", "(*^▽^*)"],
    ["**bold** `(*^▽^*)` and (*^▽^*)", "(*^▽^*)"],
  ]) {
    for (const final of [false, true]) {
      const nodes = parseMarkdownToStructure(source, markdown, { final });
      assert.deepEqual(
        [...walkNodes(nodes)].filter((node) => node.type === "inline_code").map((node) => node.code),
        [code],
        source,
      );
      assert.equal(
        [...walkNodes(nodes)].some((node) => node.type === "kaomoji_literal" && node.content.includes("`")),
        false,
        source,
      );
    }
  }

  const fenced = parseMarkdownToStructure("```text\n(*^▽^*)\n```", markdown, { final: true });
  assert.equal(hasNodeType(fenced, "code_block"), true);
  assert.equal(hasNodeType(fenced, "kaomoji_literal"), false);

  for (const final of [false, true]) {
    const mixed = parseMarkdownToStructure("前缀 **重点** 后缀 (*^▽^*)", markdown, { final });
    assert.equal(hasNodeType(mixed, "strong"), true);
    assert.equal(hasNodeType(mixed, "emphasis"), false);
    assert.equal(hasNodeType(mixed, "superscript"), false);
  }
});

test("kaomoji rule keeps Markstream append-tail parsing and stable nodes", () => {
  const markdown = configureKaomojiMarkdown(getMarkdown("kaomoji-stream"));
  const first = parseMarkdownToStructure("# stable\n\n(*", markdown, {
    final: false,
    reuseStableTopLevelNodes: true,
  });
  const second = parseMarkdownToStructure("# stable\n\n(*^▽^*)", markdown, {
    final: false,
    reuseStableTopLevelNodes: true,
  });

  assert.strictEqual(second[0], first[0]);
  assert.equal(markdown.stream?.stats?.().lastMode, "tail");
  assert.equal(hasFormattingNode(second), false);

  const decoratedMarkdown = configureKaomojiMarkdown(getMarkdown("kaomoji-decorated-stream"));
  const decoratedPartial = parseMarkdownToStructure("# stable\n\n☆_.｡.o(≧", decoratedMarkdown, {
    final: false,
    reuseStableTopLevelNodes: true,
  });
  const decoratedComplete = parseMarkdownToStructure("# stable\n\n☆_.｡.o(≧▽≦)o.｡.:_☆", decoratedMarkdown, {
    final: false,
    reuseStableTopLevelNodes: true,
  });
  const decoratedAppended = parseMarkdownToStructure("# stable\n\n☆_.｡.o(≧▽≦)o.｡.:_☆\n\nmore", decoratedMarkdown, {
    final: false,
    reuseStableTopLevelNodes: true,
  });

  assert.strictEqual(decoratedComplete[0], decoratedPartial[0]);
  assert.strictEqual(decoratedAppended[0], decoratedComplete[0]);
  assert.equal(decoratedMarkdown.stream?.stats?.().lastMode, "tail");

  for (const delimiter of ["`", "``"]) {
    const codeMarkdown = configureKaomojiMarkdown(getMarkdown(`kaomoji-code-stream-${delimiter.length}`));
    const partial = parseMarkdownToStructure(`# stable\n\n${delimiter}(*`, codeMarkdown, {
      final: false,
      reuseStableTopLevelNodes: true,
    });
    const face = parseMarkdownToStructure(`# stable\n\n${delimiter}(*^▽^*)`, codeMarkdown, {
      final: false,
      reuseStableTopLevelNodes: true,
    });
    const closed = parseMarkdownToStructure(`# stable\n\n${delimiter}(*^▽^*)${delimiter}`, codeMarkdown, {
      final: false,
      reuseStableTopLevelNodes: true,
    });

    assert.strictEqual(face[0], partial[0]);
    assert.strictEqual(closed[0], face[0]);
    assert.equal(hasNodeType(face, "kaomoji_literal"), true);
    assert.equal(hasNodeType(closed, "inline_code"), true);
    assert.equal(codeMarkdown.stream?.stats?.().lastMode, "tail");
  }
});

test("compact rule protects a broad syntax-sensitive open-corpus subset", () => {
  const values = [...new Set(Object.values(kaomojiCollection).flat())];
  const stock = getMarkdown("kaomoji-corpus-stock");
  const guarded = configureKaomojiMarkdown(getMarkdown("kaomoji-corpus-guarded"));
  let vulnerable = 0;
  let protectedCount = 0;

  for (const value of values) {
    if (!hasFormattingNode(parseMarkdownToStructure(value, stock, { final: true }))) continue;
    vulnerable += 1;
    if (!hasFormattingNode(parseMarkdownToStructure(value, guarded, { final: true }))) protectedCount += 1;
  }

  assert.ok(vulnerable > 8_000, `expected a broad syntax-sensitive corpus, got ${vulnerable}`);
  assert.ok(protectedCount >= 7_000, `${protectedCount}/${vulnerable} faces stayed literal`);
  assert.ok(protectedCount / vulnerable >= 0.84, `${protectedCount}/${vulnerable} faces stayed literal`);
});

function hasFormattingNode(nodes) {
  return [...walkNodes(nodes)].some((node) => FORMATTING_NODES.has(node.type));
}

function hasNodeType(nodes, type) {
  return [...walkNodes(nodes)].some((node) => node.type === type);
}

function* walkNodes(nodes) {
  for (const node of nodes) {
    yield node;
    if (Array.isArray(node.children)) yield* walkNodes(node.children);
  }
}
