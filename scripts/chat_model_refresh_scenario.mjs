/** Manual acceptance: production controller in JSDOM with delayed synthetic HTTP.
 * Run: node scripts/chat_model_refresh_scenario.mjs
 * No real backend, browser layout, credentials, or user data. Not a unit-test suite.
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
const temporary = await mkdtemp(resolve(tmpdir(), "akashic-model-refresh-"));
const checks = [];
let root;
let dom;
try {
  await symlink(resolve(repo, "node_modules"), resolve(temporary, "node_modules"));
  await build({
    stdin: {
      contents: `export { useDesktopChatController } from ${JSON.stringify(resolve(repo, "frontend/chat/src/use-desktop-chat-controller.ts"))};`,
      resolveDir: repo, loader: "tsx",
    },
    bundle: true, platform: "node", format: "esm", packages: "external",
    alias: { "@": resolve(repo, "frontend/chat/src") }, loader: { ".css": "empty" },
    define: { "import.meta.env.DEV": "false" }, outfile: resolve(temporary, "controller.mjs"),
  });
  dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: "http://model-refresh.local/chat", pretendToBeVisual: true,
  });
  const { window } = dom;
  for (const name of ["window", "document", "HTMLElement", "Element", "Node", "DOMException", "sessionStorage", "localStorage", "Event", "CustomEvent", "MessageEvent"]) {
    Object.defineProperty(globalThis, name, { value: name === "window" ? window : window[name], configurable: true });
  }
  Object.defineProperty(globalThis, "navigator", { value: window.navigator, configurable: true });
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  const frames = [];
  class FixtureSocket extends window.EventTarget {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSED = 3;
    readyState = 1;
    constructor(url) { super(); this.url = url; }
    send(data) { frames.push(JSON.parse(data)); }
    close() { this.readyState = 3; }
  }
  globalThis.WebSocket = FixtureSocket;
  const React = await import("react");
  const { act } = React;
  const { createRoot } = await import("react-dom/client");
  const { useDesktopChatController } = await import(pathToFileURL(resolve(temporary, "controller.mjs")));
  let controller;
  let shellResolve;
  let requests = [];
  const ready = { status: "ready", configured: true, chatReady: true };
  const models = (overrides = {}) => ({ ...desktopModels(2), unavailableRuntimes: [], ...overrides });
  const json = (data, status = 200) => new Response(JSON.stringify(data), { status });
  globalThis.fetch = async (url, options = {}) => {
    const parsed = new URL(url, window.location.origin);
    if (parsed.pathname === "/api/shell/state") return new Promise((resolve) => { shellResolve = (data = ready) => resolve(json(data)); });
    if (parsed.pathname === "/api/chat/models") return new Promise((resolve, reject) => {
      // Deliberately ignore abort here so stale-response guards, not just fetch cancellation, are exercised.
      requests.push({ session: parsed.searchParams.get("session_key") ?? "", signal: options.signal,
        resolve: (data = models(), status = 200) => resolve(json(data, status)), reject });
    });
    if (parsed.pathname === "/api/chat/navigation/pins") return json({ pins: [], sessions: [] });
    if (parsed.pathname === "/api/chat/sessions") return json({ items: [], next_cursor: null });
    if (parsed.pathname === "/api/chat/plugin-ui/catalog") return json({ catalog_revision: "0".repeat(64), items: [] });
    if (parsed.pathname.endsWith("/messages")) return json({ version: 2, items: [], through_seq: -1, before_seq: null, has_more: false });
    throw new Error(`Unexpected URL: ${url}`);
  };
  function Harness() { controller = useDesktopChatController(); return null; }
  const settle = async (action = () => {}) => act(async () => { action(); await new Promise((resolve) => setTimeout(resolve, 0)); });
  const focus = () => window.dispatchEvent(new Event("focus"));
  const visible = () => document.dispatchEvent(new Event("visibilitychange"));
  const changed = (source = window.parent, origin = window.location.origin) => window.dispatchEvent(new MessageEvent("message", {
    data: { type: "akashic.models.changed" }, source, origin,
  }));
  async function mount(session = "", strict = false) {
    sessionStorage.clear();
    if (session) sessionStorage.setItem("akashic.chat.active-session", session);
    requests = [];
    root = createRoot(document.getElementById("root"));
    await settle(() => root.render(strict ? React.createElement(React.StrictMode, null, React.createElement(Harness)) : React.createElement(Harness)));
  }
  async function unmount() {
    await settle(() => root.unmount());
    root = null;
  }
  await mount();
  assert.equal(requests.length, 0);
  await settle(() => { focus(); visible(); });
  assert.equal(requests.length, 0);
  await settle(() => shellResolve());
  assert.equal(requests.length, 1);
  assert.equal(controller.canSend, false);
  assert.match(controller.modelProblem, /正在核对/);
  await settle(() => { focus(); visible(); });
  assert.equal(requests.length, 1);
  assert.equal(requests[0].signal.aborted, false);
  await settle(() => requests[0].resolve());
  assert.equal(controller.canSend, true);
  checks.push("Cold startup waits for readiness and issues one read; first unresolved snapshot blocks sending; simultaneous reads join without abort");

  const draftKey = controller.draftKey;
  sessionStorage.setItem("akashic.chat.draft:" + draftKey, "unsent draft");
  await settle(() => { focus(); visible(); });
  assert.equal(requests.length, 2);
  assert.equal(controller.canSend, true);
  assert.equal(controller.modelProblem, "");
  assert.equal(controller.draftKey, draftKey);
  assert.equal(sessionStorage.getItem("akashic.chat.draft:" + draftKey), "unsent draft");
  await settle(() => controller.handleModelChange("perf/runtime-1", "high"));
  await settle(() => requests[1].resolve());
  assert.equal(controller.selectedRuntimeId, "perf/runtime-1");
  assert.equal(controller.selectedReasoningEffort, "high");
  checks.push("Focus and visible share one background read without blocking or changing draft identity/storage; late response preserves a user model edit");

  await settle(focus);
  const old = requests.at(-1);
  const count = requests.length;
  await settle(() => { changed(null); changed(window.parent, "https://untrusted.invalid"); });
  assert.equal(requests.length, count);
  await settle(() => changed());
  assert.equal(requests.length, count + 1);
  assert.equal(old.signal.aborted, true);
  const fresh = requests.at(-1);
  await settle(() => fresh.resolve(models({ generationId: 2 })));
  await settle(() => old.resolve(models({ generationId: 1, runtimes: [] })));
  assert.equal(controller.modelState.generationId, 2);
  assert.equal(controller.canSend, true);
  checks.push("Trusted settings change supersedes pre-change request; untrusted events ignored; aborted late result cannot overwrite fresh catalog");

  for (const status of [401, 403, 503]) {
    await settle(focus);
    await settle(() => requests.at(-1).resolve({ detail: "fixture" }, status));
    assert.equal(controller.canSend, false);
    assert.equal(controller.modelsPhase, "error");
    const problem = controller.modelProblem;
    await settle(() => { controller.retryModels(); focus(); visible(); });
    assert.equal(controller.modelsPhase, "error");
    assert.equal(controller.modelProblem, problem);
    await settle(() => requests.at(-1).resolve());
    assert.equal(controller.canSend, true);
  }
  checks.push("401/403/503 remain blocking and retain their reason through retry; successful revalidation restores sending");

  await settle(focus);
  await settle(() => requests.at(-1).resolve(models({ runtimes: [], unavailableRuntimes: [{
    id: "perf/runtime-1", model: "fixture", sourceName: "Fixture", availability: "disabled",
  }] })));
  assert.equal(controller.canSend, false);
  assert.match(controller.modelProblem, /已停用/);
  await settle(focus);
  assert.match(controller.modelProblem, /已停用/);
  await settle(() => requests.at(-1).resolve());
  assert.equal(controller.canSend, true);
  checks.push("Confirmed disabled model blocks and remains explained during background refresh; restored catalog recovers");

  await settle(() => controller.activateSession("akashic:A"));
  assert.equal(controller.modelState, null);
  assert.equal(controller.canSend, false);
  const a = requests.at(-1);
  await settle(() => controller.activateSession("akashic:B"));
  const b = requests.at(-1);
  assert.equal(a.signal.aborted, true);
  await settle(() => a.resolve(models({ sessionOverride: "perf/runtime-1" })));
  assert.equal(controller.modelState, null);
  assert.equal(controller.canSend, false);
  await settle(() => b.resolve(models({ sessionOverride: "perf/runtime" })));
  assert.equal(controller.canSend, true);
  assert.equal(controller.selectedRuntimeId, "perf/runtime");
  await settle(() => controller.startNewChat());
  assert.equal(controller.modelState, null);
  assert.equal(controller.canSend, false);
  await settle(() => requests.at(-1).resolve());
  checks.push("A→B and existing→new chat clear prior snapshot; delayed A cannot unblock B or replace its selection");

  await settle(() => controller.handleModelChange("perf/runtime-1", "high"));
  await settle(focus);
  const draftRead = requests.at(-1);
  assert.equal(draftRead.session, "");
  await act(async () => { await controller.sendMessage("fixture message", []); });
  assert.equal(frames.at(-1).type, "message.send");
  assert.equal(frames.at(-1).model_runtime_id, "perf/runtime-1");
  const beforeDraftRead = controller.modelState;
  await settle(() => draftRead.resolve(models({ generationId: 1, runtimes: [] })));
  assert.equal(draftRead.signal.aborted, false);
  assert.equal(controller.modelState, beforeDraftRead);
  assert.equal(controller.canSend, true);
  await settle(focus);
  assert.equal(controller.canSend, true);
  assert.equal(requests.at(-1).session, controller.activeSessionId);
  await settle(() => requests.at(-1).resolve(models({ generationId: 3, sessionOverride: "perf/runtime" })));
  assert.equal(controller.modelState.generationId, 3);
  assert.equal(controller.canSend, true);
  assert.equal(controller.selectedRuntimeId, "perf/runtime-1");
  assert.equal(controller.selectedReasoningEffort, "high");
  checks.push("Draft→first send keeps usable snapshot; stale blank-draft response cannot mutate the new session; submitted turn retains outgoing model/effort");
  await unmount();

  await mount("akashic:restored", true);
  assert.equal(requests.length, 0);
  await settle(() => shellResolve());
  assert.equal(requests.length, 1);
  assert.equal(requests[0].session, "akashic:restored");
  assert.equal(controller.canSend, false);
  await settle(() => requests[0].resolve(models({ defaultRuntime: "", sessionOverride: "perf/runtime-1" })));
  assert.equal(controller.canSend, true);
  checks.push("StrictMode restored session reads only its session snapshot once; valid session override works without system default");
  await settle(focus);
  const stuck = requests.at(-1);
  await settle(() => controller.retryModels());
  assert.equal(stuck.signal.aborted, true);
  assert.notEqual(requests.at(-1), stuck);
  const abandoned = requests.at(-1);
  await settle(() => stuck.resolve());
  checks.push("Explicit retry supersedes a stuck read instead of joining it forever");
  await unmount();
  assert.equal(abandoned.signal.aborted, true);
  await settle(() => abandoned.resolve());
  checks.push("Unmount aborts pending request and ignores late completion");
  console.log(JSON.stringify({ boundary: "Production React controller in JSDOM; synthetic HTTP and WebSocket, no browser layout or real backend acceptance", passed: checks.length, checks }, null, 2));
} finally {
  if (root && dom) root.unmount();
  dom?.window.close();
  await rm(temporary, { recursive: true, force: true });
}
