import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const app = await readFile(new URL("./desktop-chat-view.tsx", import.meta.url), "utf8");
const controller = await readFile(new URL("./use-desktop-chat-controller.ts", import.meta.url), "utf8");
const composer = await readFile(new URL("./desktop-composer.tsx", import.meta.url), "utf8");

test("desktop composer owns transient input outside the app root", () => {
  assert.match(app, /<DesktopComposer/);
  assert.doesNotMatch(app, /const \[input, setInput\]/);
  assert.match(composer, /memo\(function DesktopComposer/);
  assert.match(composer, /const \[input, setInput\] = useState\(""\)/);
  assert.match(composer, /setInput\(\(current\) => current \|\| text\)/);
});

test("desktop stop transport has one in-flight owner", () => {
  assert.match(controller, /if \(sendRequestRef\.current\) \{[\s\S]*?sendRequestRef\.current\.abort\(\);[\s\S]*?socket\.close\(1000, "pending send cancelled"\);[\s\S]*?setStatus\("idle"\);/u);
  assert.match(controller, /if \(!activeSessionId \|\| stopRequestRef\.current\) return/);
  assert.match(controller, /stopRequestRef\.current = controller/);
  assert.match(composer, /disabled=\{disabled \|\| stopPending/);
});

test("desktop submit owns optimistic history against stale reconciliation", () => {
  assert.match(controller, /messagesRequestRef\.current\?\.abort\(\);\s*olderMessagesRequestRef\.current\?\.abort\(\);\s*sendRequestRef\.current\?\.abort\(\);/u);
  assert.match(controller, /catch \(error\) \{\s*setMessages\(\(current\) => current\.filter/u);
});

test("desktop composer preserves attachment and reply capabilities", () => {
  assert.match(composer, /PromptInputActionMenuTrigger aria-label="添加文件"/);
  assert.match(composer, /PromptInputActionAddAttachments label="上传文件"/);
  assert.match(composer, /<ComposerReply/);
  assert.match(composer, /<AttachmentHoverCard/);
});
