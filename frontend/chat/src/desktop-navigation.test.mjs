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

test("L-shape keeps product destinations on the top band; sidebar is session-only", () => {
  assert.match(app, /<ChatProductBand/);
  assert.match(sidebar, /destinations=\{\[\]\}/);
  assert.match(sidebar, /id: "connect-mobile"[\s\S]*?onActivate: onOpenPairing/u);
  assert.doesNotMatch(sidebar, /id: "models"/);
  assert.doesNotMatch(sidebar, /chat-sidebar-brand/);
});

test("session activation is idempotent and aborts stale model requests", () => {
  assert.match(controller, /activeSessionRef\.current === sessionId\) return/);
  assert.match(controller, /modelsRequestRef\.current\?\.abort\(\)/);
  assert.match(controller, /fetchChatJson<unknown>\(`\/api\/chat\/models\$\{query\}`, \{ signal: controller\.signal \}\)/);
  assert.match(controller, /activeSessionRef\.current = sessionId;\s+attachSession\(socketRef\.current, sessionId\);/u);
});

test("history cannot overwrite a newer live projection", () => {
  assert.match(controller, /const projectionRevision = liveProjectionRevisionRef\.current;/u);
  assert.match(controller, /liveProjectionRevisionRef\.current !== projectionRevision/u);
  assert.match(controller, /frame\.session_id === activeSessionRef\.current[\s\S]*liveProjectionRevisionRef\.current \+= 1/u);
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
