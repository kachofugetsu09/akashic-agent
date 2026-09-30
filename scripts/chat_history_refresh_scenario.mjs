/** Manual acceptance: actual React controller with delayed HTTP and synthetic WebSocket.
 * Run: node scripts/chat_history_refresh_scenario.mjs
 * No real backend, browser layout, user data, or new unit-test suite.
 */
import assert from "node:assert/strict";
import { mkdtemp, symlink, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import { desktopModels } from "./webui-performance/fixtures.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = await mkdtemp(resolve(tmpdir(), "akashic-history-refresh-"));
const checks = [];
let root;
let dom;
try {
  await symlink(resolve(repo, "node_modules"), resolve(temporary, "node_modules"));
  await build({
    stdin: { contents: `export { useDesktopChatController } from ${JSON.stringify(resolve(repo, "frontend/chat/src/use-desktop-chat-controller.ts"))};`, resolveDir: repo, loader: "tsx" },
    bundle: true, platform: "node", format: "esm", packages: "external",
    alias: { "@": resolve(repo, "frontend/chat/src") }, loader: { ".css": "empty" },
    define: { "import.meta.env.DEV": "false" }, outfile: resolve(temporary, "controller.mjs"),
  });
  dom = new JSDOM('<!doctype html><div id="root"></div>', { url: "http://history-refresh.local/chat", pretendToBeVisual: true });
  const { window } = dom;
  for (const name of ["window", "document", "HTMLElement", "Element", "Node", "DOMException", "sessionStorage", "localStorage", "Event", "CustomEvent", "MessageEvent"]) {
    Object.defineProperty(globalThis, name, { value: name === "window" ? window : window[name], configurable: true });
  }
  Object.defineProperty(globalThis, "navigator", { value: window.navigator, configurable: true });
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const sockets = [];
  const frames = [];
  class FixtureSocket extends window.EventTarget {
    static CONNECTING = 0; static OPEN = 1; static CLOSED = 3;
    readyState = 0;
    constructor(url) {
      super(); this.url = url; sockets.push(this);
      setTimeout(() => {
        if (this.readyState !== 0) return;
        this.readyState = 1;
        this.onopen?.(new Event("open"));
        this.dispatchEvent(new Event("open"));
      }, 0);
    }
    send(data) { frames.push(JSON.parse(data)); }
    close() { this.readyState = 3; }
  }
  globalThis.WebSocket = FixtureSocket;
  const histories = [];
  const json = (data, status = 200) => new Response(JSON.stringify(data), { status });
  globalThis.fetch = async (url, options = {}) => {
    const parsed = new URL(url, window.location.origin);
    if (parsed.pathname.endsWith("/messages")) return new Promise((resolve, reject) => {
      // Intentionally deliver after cancellation: stale guards must work even after headers/body arrive.
      histories.push({ url: parsed, signal: options.signal,
        resolve: (data, status = 200) => resolve(json(data, status)), reject });
    });
    if (parsed.pathname === "/api/shell/state") return json({ status: "ready", configured: true, chatReady: true });
    if (parsed.pathname === "/api/chat/models") return json({ ...desktopModels(2), unavailableRuntimes: [] });
    if (parsed.pathname === "/api/chat/navigation/pins") return json({ pins: [], sessions: [] });
    if (parsed.pathname === "/api/chat/sessions") return json({ items: [], next_cursor: null });
    if (parsed.pathname === "/api/chat/plugin-ui/catalog") return json({ catalog_revision: "0".repeat(64), items: [] });
    throw new Error(`Unexpected URL: ${url}`);
  };
  const React = await import("react");
  const { act } = React;
  const { createRoot } = await import("react-dom/client");
  const { useDesktopChatController } = await import(pathToFileURL(resolve(temporary, "controller.mjs")));
  let controller;
  function Harness() { controller = useDesktopChatController(); return null; }
  const settle = async (action = () => {}) => act(async () => { action(); await new Promise((resolve) => setTimeout(resolve, 10)); });
  const row = (session, seq) => ({ id: `${session}:${seq}`, seq, author: "user", source: "akashic", timestamp: "2026-09-30T00:00:00Z", session_id: session, attachments: [], body: { kind: "input", parts: [{ kind: "text", value: `message ${seq}` }] } });
  const page = (session, seqs, before = null, through = seqs.at(-1) ?? -1) => ({ version: 2, items: seqs.map((seq) => row(session, seq)), through_seq: through, before_seq: before, has_more: before !== null });
  const seqs = () => controller.timelineMessages.map((item) => item.seq);
  const append = (session, after, next) => sockets.at(-1).onmessage({ data: JSON.stringify({ type: "messages.appended", version: 2, session_id: session, after_seq: after, through_seq: next, next_after_seq: next, has_more: false, items: [row(session, next)] }) });
  root = createRoot(document.getElementById("root"));
  await settle(() => root.render(React.createElement(Harness)));
  await settle();

  await settle(() => controller.prefetchSessionTail("akashic:A"));
  const prefetchedA = histories.at(-1);
  const initialCount = histories.length;
  await settle(() => controller.activateSession("akashic:A"));
  assert.equal(histories.length, initialCount, "activation must join pending prefetch");
  await settle(() => prefetchedA.resolve(page("akashic:A", [0, 1, 2])));
  await settle();
  assert.deepEqual(seqs(), [0, 1, 2]);
  await settle(() => append("akashic:A", 2, 3));
  checks.push("Hover→activation shares one history GET; WebSocket follow continues from that history and appends normally");

  await settle(() => controller.activateSession("akashic:B"));
  const abandonedB = histories.at(-1);
  await settle(() => controller.activateSession("akashic:A"));
  assert.equal(abandonedB.signal.aborted, true);
  assert.equal(controller.historyLoading, false);
  assert.deepEqual(seqs(), [0, 1, 2, 3]);
  await settle(() => abandonedB.resolve({ detail: "obsolete B failure" }, 503));
  assert.equal(controller.error, "");
  assert.equal(controller.activeSessionId, "akashic:A");
  assert.equal(controller.historyLoading, false);
  checks.push("Cache-hit A navigation cancels B immediately; delayed B 503 does not poison A or keep its loading state");

  await settle(() => controller.activateSession("akashic:B"));
  const oldB = histories.at(-1);
  await settle(() => controller.activateSession("akashic:A"));
  await settle(() => controller.activateSession("akashic:B"));
  const newB = histories.at(-1);
  assert.notEqual(newB, oldB);
  assert.equal(oldB.signal.aborted, true);
  await settle(() => oldB.resolve(page("akashic:B", [0])));
  assert.equal(controller.historyLoading, true, "old finalizer must not finish newer loading");
  assert.deepEqual(seqs(), []);
  await settle(() => newB.resolve(page("akashic:B", [5])));
  await settle();
  assert.deepEqual(seqs(), [5]);
  checks.push("A→B→A→B creates a fresh owner; cancelled same-session success cannot render stale history or clear current loading");

  await settle(() => controller.activateSession("akashic:C"));
  await settle(() => histories.at(-1).resolve(page("akashic:C", [2], 2)));
  await settle();
  let olderDone;
  await settle(() => { olderDone = controller.loadOlderMessages(); });
  const oldOlder = histories.at(-1);
  await settle(() => controller.activateSession("akashic:A"));
  assert.equal(oldOlder.signal.aborted, true);
  assert.equal(controller.historyLoadingOlder, false);
  await settle(() => controller.activateSession("akashic:C"));
  let newOlderDone;
  await settle(() => { newOlderDone = controller.loadOlderMessages(); });
  const newOlder = histories.at(-1);
  assert.notEqual(newOlder, oldOlder);
  assert.equal(newOlder.url.searchParams.get("through_seq"), "2");
  await settle(() => oldOlder.resolve({ detail: "obsolete older failure" }, 503));
  await olderDone;
  assert.equal(controller.historyLoadingOlder, true);
  assert.equal(controller.error, "");
  await settle(() => newOlder.resolve(page("akashic:C", [0, 1], null, 2)));
  await newOlderDone;
  assert.deepEqual(seqs(), [0, 1, 2]);
  await settle(() => append("akashic:C", 2, 3));
  await settle(() => controller.activateSession("akashic:A"));
  const beforeReturn = histories.length;
  await settle(() => controller.activateSession("akashic:C"));
  assert.equal(histories.length, beforeReturn);
  assert.deepEqual(seqs(), [0, 1, 2, 3]);
  assert.equal(controller.historyHasMore, false);
  checks.push("Old pagination failure/finalizer cannot affect new pagination; merged older pages and live head survive cache-hit revisit");

  await settle(() => controller.activateSession("akashic:G"));
  await settle(() => histories.at(-1).resolve(page("akashic:G", [2], 2)));
  await settle();
  await settle(() => controller.retry());
  const staleRefresh = histories.at(-1);
  let mergedOlderDone;
  await settle(() => { mergedOlderDone = controller.loadOlderMessages(); });
  await settle(() => histories.at(-1).resolve(page("akashic:G", [0, 1], null, 2)));
  await mergedOlderDone;
  assert.deepEqual(seqs(), [0, 1, 2]);
  await settle(() => staleRefresh.resolve(page("akashic:G", [2], 2)));
  await settle();
  assert.deepEqual(seqs(), [0, 1, 2]);
  assert.equal(controller.historyHasMore, false);
  checks.push("Tail refresh cannot replace a cache enriched by concurrent older pagination at the same through_seq");

  await settle(() => controller.activateSession("akashic:H"));
  const sharedH = histories.at(-1);
  const beforeSecondReader = histories.length;
  await settle(() => controller.retry());
  await settle();
  assert.equal(histories.length, beforeSecondReader);
  const revision = controller.timelineRefresh;
  const followCount = frames.filter((frame) => frame.type === "session.follow" && frame.session_id === "akashic:H").length;
  await settle(() => sharedH.resolve(page("akashic:H", [0])));
  assert.equal(controller.timelineRefresh, revision + 1);
  assert.equal(frames.filter((frame) => frame.type === "session.follow" && frame.session_id === "akashic:H").length, followCount + 1);
  checks.push("Activation plus socket-open history readers share one GET and only one reader publishes/follows the result");

  await settle(() => controller.prefetchSessionTail("akashic:I"));
  const cancelledPrefetch = histories.at(-1);
  await settle(() => controller.activateSession("akashic:I"));
  await settle(() => controller.activateSession("akashic:A"));
  assert.equal(cancelledPrefetch.signal.aborted, true);
  await settle(() => controller.activateSession("akashic:I"));
  await settle(() => histories.at(-1).resolve(page("akashic:I", [8])));
  await settle(() => cancelledPrefetch.resolve(page("akashic:I", [0])));
  await settle(() => controller.activateSession("akashic:A"));
  await settle(() => controller.activateSession("akashic:I"));
  assert.deepEqual(seqs(), [8]);
  checks.push("Cancelled prefetch delivering after a newer read cannot overwrite its cached tail");

  await settle(() => controller.prefetchSessionTail("akashic:D"));
  await settle(() => histories.at(-1).resolve({ detail: "prefetch failed" }, 503));
  assert.equal(controller.error, "");
  const failedCount = histories.length;
  await settle(() => controller.activateSession("akashic:D"));
  assert.equal(histories.length, failedCount + 1);
  await settle(() => histories.at(-1).resolve(page("akashic:D", [0])));
  await settle();
  assert.deepEqual(seqs(), [0]);
  checks.push("Failed background prefetch is silent and removed; normal activation issues a fresh successful read");

  await settle(() => controller.prefetchSessionTail("akashic:E"));
  const pendingE = histories.at(-1);
  const pendingCount = histories.length;
  await settle(() => controller.activateSession("akashic:E"));
  assert.equal(histories.length, pendingCount);
  await settle(() => pendingE.resolve({ detail: "active history unavailable" }, 503));
  assert.equal(controller.error, "active history unavailable");
  await settle(() => controller.retry());
  await settle();
  assert.equal(histories.length, pendingCount + 1);
  await settle(() => histories.at(-1).resolve(page("akashic:E", [0])));
  assert.equal(controller.error, "");
  assert.deepEqual(seqs(), [0]);
  checks.push("An activated pending prefetch failure is reported once; explicit retry creates a new request and recovers");

  await settle(() => controller.activateSession("akashic:F"));
  const outgoing = histories.at(-1);
  await settle(() => controller.startNewChat());
  assert.equal(outgoing.signal.aborted, true);
  assert.equal(controller.historyLoading, false);
  await settle(() => outgoing.resolve({ detail: "old chat failed" }, 503));
  assert.equal(controller.activeSessionId, "");
  assert.equal(controller.error, "");
  checks.push("New-chat exit resets history loading immediately and ignores delayed outgoing failures");

  await settle(() => controller.prefetchSessionTail("akashic:unmounted"));
  const unmounted = histories.at(-1);
  await settle(() => root.unmount());
  root = null;
  assert.equal(unmounted.signal.aborted, true);
  await settle(() => unmounted.resolve(page("akashic:unmounted", [0])));
  checks.push("Unmount also cancels standalone prefetch and tolerates its late completion");
  console.log(JSON.stringify({ boundary: "Actual React controller in JSDOM; synthetic delayed HTTP/WebSocket only", passed: checks.length, checks, followFrames: frames.filter((frame) => frame.type === "session.follow").length }, null, 2));
} finally {
  if (root && dom) root.unmount();
  dom?.window.close();
  await rm(temporary, { recursive: true, force: true });
}
