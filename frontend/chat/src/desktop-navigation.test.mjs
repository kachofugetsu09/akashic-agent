import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const entry = await readFile(new URL("./main.tsx", import.meta.url), "utf8");
const mobileNavigation = await readFile(new URL("./desktop-mobile-navigation.tsx", import.meta.url), "utf8");
const sidebar = await readFile(new URL("./desktop-sidebar.tsx", import.meta.url), "utf8");
const navigation = await readFile(new URL("./conversation-navigation.tsx", import.meta.url), "utf8");
const mobile = await readFile(new URL("./mobile-native.tsx", import.meta.url), "utf8");

test("desktop entry delegates navigation presentation to one controlled sidebar", () => {
  assert.match(entry, /<DesktopSidebar/);
  assert.doesNotMatch(entry, /<ConversationNavigation/);
  assert.match(sidebar, /memo\(function DesktopSidebar/);
  assert.match(sidebar, /onSelectSession: \(sessionId: string\) => void/);
});

test("session activation is idempotent and aborts stale model requests", () => {
  assert.match(entry, /activeSessionRef\.current === sessionId\) return/);
  assert.match(entry, /modelsRequestRef\.current\?\.abort\(\)/);
  assert.match(entry, /fetchChatJson<unknown>\(`\/api\/chat\/models\$\{query\}`, \{ signal: controller\.signal \}\)/);
});

test("shared navigation reports semantic session identities to both adapters", () => {
  assert.match(navigation, /onSessionActivate\(session\.id\)/);
  assert.match(navigation, /aria-current=\{session\.active \? "true" : undefined\}/);
  assert.doesNotMatch(navigation, /session\.onActivate/);
  assert.match(mobile, /onSessionActivate=\{\(sessionId\) =>/);
});

test("narrow desktop keeps the same navigation owner behind a modal trigger", () => {
  assert.match(entry, /<DesktopMobileNavigation/);
  assert.match(mobileNavigation, /<DesktopSidebar/);
  assert.match(mobileNavigation, /aria-label="打开导航"/);
  assert.match(mobileNavigation, /closeThen/);
  assert.doesNotMatch(mobileNavigation, /ConversationNavigation/);
});
