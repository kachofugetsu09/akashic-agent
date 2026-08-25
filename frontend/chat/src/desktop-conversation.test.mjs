import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const conversation = await readFile(new URL("./desktop-conversation.tsx", import.meta.url), "utf8");
const messageView = await readFile(new URL("./message-view.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("./styles.css", import.meta.url), "utf8");
const conversationShell = await readFile(new URL("./components/ai-elements/conversation.tsx", import.meta.url), "utf8");
const desktopApp = await readFile(new URL("./desktop-chat-view.tsx", import.meta.url), "utf8");
const desktopAutoScroll = await readFile(new URL("./desktop-auto-scroll.tsx", import.meta.url), "utf8");

test("desktop history isolates stable rows but never the active stream", () => {
  assert.match(conversation, /message\.streaming === true \? "streaming" : "history-isolated"/);
  assert.match(styles, /\.web-message-anchor\.history-isolated\s*\{[\s\S]*?content-visibility:\s*auto;/);
  assert.doesNotMatch(styles, /\.web-message-anchor\.streaming\s*\{[\s\S]*?content-visibility/);
  assert.match(conversationShell, /initial="instant"/);
  assert.match(desktopApp, /<Conversation className="conversation" resize="instant">/);
  assert.match(desktopApp, /scrollElement\.scrollTop \+= restoredAnchor\.getBoundingClientRect\(\)\.top - anchorTop/);
});

test("desktop auto-scroll subscribes only to the tail message and preserves user escape", () => {
  assert.match(desktopApp, /<DesktopAutoScroll messages=\{messages\}/);
  assert.match(desktopAutoScroll, /streamStore\.subscribe\(baselineLastMessageId, listener\)/);
  assert.match(desktopAutoScroll, /ignoreEscapes: true/);
  assert.match(desktopAutoScroll, /isAtBottom && !escapedFromLock/);
  assert.doesNotMatch(desktopAutoScroll, /\[baselineLastMessage\?\.id, streamStore\]/);
  assert.match(conversationShell, /aria-label="滚动到底部"/);
});

test("desktop rich content upgrades near the viewport without hiding fallback text", () => {
  assert.match(conversation, /deferRichContent/);
  assert.match(conversation, /new IntersectionObserver/);
  assert.match(conversation, /rootMargin: "800px 0px"/);
  assert.match(messageView, /<StaticMessageResponse onError=\{onError\}>\{content\}<\/StaticMessageResponse>/);
  assert.match(messageView, /features\.math \|\| features\.mermaid \|\| features\.code/);
  assert.match(conversation, /enhancementSuspended=\{status !== "idle"\}/);
});

test("desktop static code fallback remains visually contained", () => {
  assert.match(styles, /\.static-code-block\s*\{[\s\S]*?width:\s*fit-content;[\s\S]*?border:[^;]+;/);
  assert.match(styles, /\.static-code-block\s*\{[\s\S]*?background:[^;]+;/);
  assert.match(styles, /\.static-code-block\s*>\s*pre\s*\{[\s\S]*?overflow-x:\s*auto;/);
});

test("desktop reply availability uses one history index", () => {
  assert.match(conversation, /new Set\(messages\.map\(\(message\) => message\.id\)\)/);
  assert.match(conversation, /!messageIds\.has\(message\.reply\.messageId\)/);
  assert.match(conversation, /stopScroll\(\);[\s\S]*?scrollIntoView\(\{ behavior: "instant", block: "center" \}\)/);
  assert.doesNotMatch(conversation, /messages\.some/);
});

test("shared message contracts no longer import the desktop entry", async () => {
  const sources = await Promise.all([
    "message-view.tsx",
    "mobile-native.tsx",
    "chat-product-variants.tsx",
    "web-stream-projection.ts",
  ].map((path) => readFile(new URL(`./${path}`, import.meta.url), "utf8")));
  for (const source of sources) assert.doesNotMatch(source, /from "\.\/main(?:\.tsx)?"/);
});
