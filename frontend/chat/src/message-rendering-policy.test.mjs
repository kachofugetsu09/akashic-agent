import assert from "node:assert/strict";
import test from "node:test";

import {
  detectMessageRenderingFeatures,
  messageNeedsMarkdown,
} from "./message-rendering-policy.ts";

test("plain chat does not load rich Markdown engines", () => {
  assert.deepEqual(detectMessageRenderingFeatures("普通消息，价格为 $5。"), {
    code: false,
    math: false,
    mermaid: false,
  });
  assert.equal(messageNeedsMarkdown("普通消息，价格为 $5。\n第二行仍是普通文本。"), false);
});

test("rich Markdown engines are selected independently", () => {
  assert.deepEqual(
    detectMessageRenderingFeatures("```ts\nconst answer = 42\n```\n\n$$x^2$$"),
    { code: true, math: true, mermaid: false },
  );
  assert.deepEqual(
    detectMessageRenderingFeatures("```mermaid\ngraph TD\nA-->B\n```"),
    { code: false, math: false, mermaid: true },
  );
  assert.equal(messageNeedsMarkdown("**重点** 和 [链接](https://example.com)"), true);
  assert.equal(messageNeedsMarkdown("- 第一项\n- 第二项"), true);
});
