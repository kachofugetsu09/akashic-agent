import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const app = await readFile(new URL("./desktop-chat-app.tsx", import.meta.url), "utf8");
const composer = await readFile(new URL("./desktop-composer.tsx", import.meta.url), "utf8");

test("desktop composer owns transient input outside the app root", () => {
  assert.match(app, /<DesktopComposer/);
  assert.doesNotMatch(app, /const \[input, setInput\]/);
  assert.match(composer, /memo\(function DesktopComposer/);
  assert.match(composer, /const \[input, setInput\] = useState\(""\)/);
  assert.match(composer, /setInput\(\(current\) => current \|\| text\)/);
});

test("desktop stop transport has one in-flight owner", () => {
  assert.match(app, /if \(!activeSessionId \|\| stopRequestRef\.current\) return/);
  assert.match(app, /stopRequestRef\.current = controller/);
  assert.match(composer, /disabled=\{disabled \|\| stopPending/);
});

test("desktop composer preserves attachment and reply capabilities", () => {
  assert.match(composer, /PromptInputActionAddAttachments label="上传文件"/);
  assert.match(composer, /<ComposerReply/);
  assert.match(composer, /<AttachmentHoverCard/);
});
