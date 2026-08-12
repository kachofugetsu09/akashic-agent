import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const app = await readFile(new URL("./desktop-chat-view.tsx", import.meta.url), "utf8");
const controller = await readFile(new URL("./use-desktop-chat-controller.ts", import.meta.url), "utf8");
const mobileNavigation = await readFile(new URL("./desktop-mobile-navigation.tsx", import.meta.url), "utf8");
const sidebar = await readFile(new URL("./desktop-sidebar.tsx", import.meta.url), "utf8");
const navigation = await readFile(new URL("./conversation-navigation.tsx", import.meta.url), "utf8");
const mobile = await readFile(new URL("./mobile-native.tsx", import.meta.url), "utf8");

test("desktop entry delegates navigation presentation to one controlled sidebar", () => {
  assert.match(app, /<DesktopSidebar/);
  assert.doesNotMatch(app, /<ConversationNavigation/);
  assert.match(sidebar, /memo\(function DesktopSidebar/);
  assert.match(sidebar, /onSelectSession: \(sessionId: string\) => void/);
});

test("embedded shell preserves mobile pairing and new-chat actions", () => {
  assert.match(sidebar, /\.\.\.\(embeddedShell \? \[\] : \[\{/u);
  assert.match(sidebar, /id: "connect-mobile"[\s\S]*?onActivate: onOpenPairing/u);
  assert.match(sidebar, /id: "new-chat"[\s\S]*?onActivate: onNewChat/u);
  assert.doesNotMatch(sidebar, /actions=\{embeddedShell \?/u);
});

test("session activation is idempotent and aborts stale model requests", () => {
  assert.match(controller, /activeSessionRef\.current === sessionId\) return/);
  assert.match(controller, /modelsRequestRef\.current\?\.abort\(\)/);
  assert.match(controller, /fetchChatJson<unknown>\(`\/api\/chat\/models\$\{query\}`, \{ signal: controller\.signal \}\)/);
});

test("shared navigation reports semantic session identities to both adapters", () => {
  assert.match(navigation, /onSessionActivate\(session\.id\)/);
  assert.match(navigation, /aria-current=\{session\.active \? "true" : undefined\}/);
  assert.doesNotMatch(navigation, /session\.onActivate/);
  assert.match(mobile, /onSessionActivate=\{\(sessionId\) =>/);
});

test("narrow desktop keeps the same navigation owner behind a modal trigger", () => {
  assert.match(app, /<DesktopMobileNavigation/);
  assert.match(mobileNavigation, /<DesktopSidebar/);
  assert.match(mobileNavigation, /aria-label="打开导航"/);
  assert.match(mobileNavigation, /closeThen/);
  assert.doesNotMatch(mobileNavigation, /ConversationNavigation/);
});
