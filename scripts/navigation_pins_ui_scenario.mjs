/** Manual acceptance scenario: production React navigation in JSDOM with a synthetic HTTP owner.
 * Run: node scripts/navigation_pins_ui_scenario.mjs
 * Does not exercise browser layout, a real backend, or modify user data. Not part of the unit-test suite.
 */
import assert from "node:assert/strict";
import { mkdtemp, symlink, rm, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import postcss from "postcss";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const temporary = await mkdtemp(resolve(tmpdir(), "akashic-navigation-pins-"));
const checks = [];
let root;
let dom;
try {
  await symlink(resolve(repo, "node_modules"), resolve(temporary, "node_modules"));
  await build({
    stdin: {
      contents: `export { DesktopSidebar } from ${JSON.stringify(resolve(repo, "frontend/chat/src/desktop-sidebar.tsx"))};
        export { CompactNavigation } from ${JSON.stringify(resolve(repo, "frontend/chat/src/compact-navigation.tsx"))};
        export { useNavigationPins } from ${JSON.stringify(resolve(repo, "frontend/chat/src/use-navigation-pins.ts"))};
        export { useDesktopChatController } from ${JSON.stringify(resolve(repo, "frontend/chat/src/use-desktop-chat-controller.ts"))};
        export { receivePluginUiCatalog } from ${JSON.stringify(resolve(repo, "frontend/chat/src/plugin-ui-runtime.tsx"))};`,
      resolveDir: repo, loader: "tsx",
    },
    bundle: true, platform: "node", format: "esm", packages: "external",
    alias: { "@": resolve(repo, "frontend/chat/src") }, loader: { ".css": "empty" },
    define: { "import.meta.env.DEV": "false" }, outfile: resolve(temporary, "navigation.mjs"),
  });
  dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    url: "http://navigation.local/chat", pretendToBeVisual: true,
  });
  const { window } = dom;
  for (const name of ["window", "document", "HTMLElement", "HTMLInputElement", "Element", "Node", "NodeFilter", "DOMException", "sessionStorage", "localStorage", "MutationObserver", "Event", "CustomEvent", "MouseEvent", "getComputedStyle"]) {
    Object.defineProperty(globalThis, name, { value: name === "window" ? window : window[name], configurable: true });
  }
  Object.defineProperty(globalThis, "navigator", { value: window.navigator, configurable: true });
  globalThis.requestAnimationFrame = window.requestAnimationFrame.bind(window);
  globalThis.cancelAnimationFrame = window.cancelAnimationFrame.bind(window);
  globalThis.IS_REACT_ACT_ENVIRONMENT = true;
  window.matchMedia = () => ({ matches: false, addListener() {}, removeListener() {}, addEventListener() {}, removeEventListener() {} });
  // JSDOM does not evaluate viewport media queries. Select each real CSS branch explicitly,
  // then check the same-specificity cascade; this is not geometric browser acceptance.
  const css = postcss.parse(await readFile(resolve(repo, "frontend/chat/src/styles.css"), "utf8"));
  for (const narrow of [false, true]) {
    const selected = [];
    css.walkRules((rule) => {
      if (![".project-navigation__icon", ".navigation-pin-action"].includes(rule.selector)) return;
      if (rule.parent.type === "atrule" && (!narrow || rule.parent.params !== "(max-width: 820px), (hover: none)")) return;
      selected.push(rule.toString());
    });
    const style = document.createElement("style");
    style.textContent = selected.join("\n");
    const button = document.createElement("button");
    button.className = "project-navigation__icon navigation-pin-action";
    document.head.append(style);
    document.body.append(button);
    assert.equal(getComputedStyle(button).width, narrow ? "44px" : "40px");
    assert.equal(getComputedStyle(button).height, narrow ? "44px" : "40px");
    button.remove(); style.remove();
  }
  checks.push("Actual CSS base/icon cascade gives 40px desktop and 44px narrow/touch pin targets (media-selected JSDOM, not layout)");
  const React = await import("react");
  const { createRoot } = await import("react-dom/client");
  const { DesktopSidebar, CompactNavigation, useNavigationPins, useDesktopChatController, receivePluginUiCatalog } = await import(pathToFileURL(resolve(temporary, "navigation.mjs")));
  const projectId = "p_" + "a".repeat(32);
  const absentProjectId = "p_" + "b".repeat(32);
  const alpha = { id: projectId, name: "Alpha", archived: false, createdAt: "2026-09-30", memory: "global" };
  const ordinary = { id: "akashic:ordinary", title: "Ordinary", preview: "2 条消息", active: false };
  const child = { id: "akashic:child", title: "Needle project chat", preview: "1 条消息", active: true, projectId, projectScoped: true };
  const orphan = { id: "akashic:orphan", title: "Orphan project chat", preview: "1 条消息", active: false, projectId: absentProjectId, projectScoped: true };
  const old = { key: "akashic:old", first_message_content: "Older pinned chat", scope: {} };
  const missingPin = { kind: "project", id: "p_" + "c".repeat(32) };
  let serverPins = [{ kind: "session", id: old.key }, { kind: "project", id: projectId }];
  let failRead = false;
  let failWrite = false;
  let holdWrite;
  let holdRead;
  let postCount = 0;
  let nav;
  let projectsInstalled = true;
  let compact = false;
  let enabled = true;
  let currentSessions = [ordinary, child, orphan];
  const drafts = [];
  const selected = [];
  const snapshot = () => ({ pins: structuredClone(serverPins), sessions: serverPins.some((pin) => pin.id === old.key) ? [old] : [] });
  globalThis.fetch = async (url, options = {}) => {
    assert.equal(url, "/api/chat/navigation/pins");
    const beforeRead = snapshot();
    if (options.method === "POST") {
      postCount += 1;
      if (holdWrite) await holdWrite;
      if (failWrite) return new Response(JSON.stringify({ detail: "Synthetic write failure" }), { status: 503 });
      const { kind, id, pinned } = JSON.parse(options.body);
      const exists = serverPins.some((pin) => pin.kind === kind && pin.id === id);
      if (pinned && !exists) serverPins.push({ kind, id });
      if (!pinned) serverPins = serverPins.filter((pin) => pin.kind !== kind || pin.id !== id);
    } else if (failRead) {
      return new Response(JSON.stringify({ detail: "Synthetic read failure" }), { status: 503 });
    }
    if (options.method !== "POST") {
      if (holdRead) await holdRead;
      return new Response(JSON.stringify(beforeRead), { status: 200 });
    }
    return new Response(JSON.stringify(snapshot()), { status: 200 });
  };
  function Harness() {
    nav = useNavigationPins(enabled);
    const props = {
      embeddedShell: true, surface: "chat", sessions: currentSessions, activeSessionId: "", pendingSessionId: "",
      chatReady: true, themeLabel: "Light", navigationPins: nav, onSelectSession: (id) => selected.push(id),
      onCycleTheme() {}, onNewChat: () => drafts.push("default"),
      projects: projectsInstalled ? {
        items: [alpha], pending: [], pendingError: "", activeProjectId: projectId, memoryInstalled: true,
        onNewChat: (id) => drafts.push(id), onCreate: async () => {}, onContinue: async () => {}, onStop() {},
      } : undefined,
    };
    return React.createElement(compact ? CompactNavigation : DesktopSidebar, props);
  }
  const act = React.act;
  const render = async () => act(async () => { root.render(React.createElement(Harness)); });
  const mount = async () => {
    root = createRoot(document.getElementById("root"));
    await render();
  };
  const button = (label) => [...document.querySelectorAll("button")].find((item) => item.getAttribute("aria-label") === label);
  const click = async (item) => {
    assert(item, "Missing action button");
    await act(async () => { item.click(); });
  };
  const search = async (query) => {
    const input = document.querySelector('input[aria-label="搜索会话"]');
    await act(async () => {
      Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set.call(input, query);
      input.dispatchEvent(new window.Event("input", { bubbles: true }));
    });
  };
  const unmount = async () => act(async () => root.unmount());
  const pinOrder = () => [...document.querySelector(".pinned-navigation").children].slice(1)
    .map((row) => row.querySelector(".project-group__name")?.textContent || row.querySelector("strong")?.textContent);

  await mount();
  assert.deepEqual(pinOrder(), ["Older pinned chat", "Alpha"]);
  assert.equal(document.querySelectorAll(".project-group").length, 1);
  assert(!document.querySelector('[aria-label="Alpha 的对话"]'));
  assert(button("置顶 Ordinary"));
  assert(!button("置顶 Needle project chat"));
  assert(!button("置顶 Orphan project chat"));
  checks.push("Mixed server order, out-of-page pin resolution, no duplicates, default collapsed even active, real-scope eligibility including global memory/orphan project");

  await click(document.querySelector(".project-group__open"));
  assert(document.querySelector('[aria-label="Alpha 的对话"]'));
  assert.equal(drafts.length, 0);
  await click(button("收起 Alpha 的对话"));
  await click(button("在 Alpha 中新建对话"));
  assert.deepEqual(drafts, [projectId]);
  assert(!document.querySelector('[aria-label="Alpha 的对话"]'));
  checks.push("Project name/arrow only fold; plus alone starts project draft; starting draft does not reopen collapsed project");

  await search("Needle");
  assert(document.querySelector('[aria-label="Alpha 的对话"]')?.textContent.includes("Needle project chat"));
  assert(!button("置顶 Needle project chat"));
  await search("NoSuchConversationOrProject");
  assert(document.querySelector('[role="status"]')?.textContent.includes("没有匹配"));
  assert(!document.body.textContent.includes("还没有对话"));
  await search("");
  assert(!document.querySelector('[aria-label="Alpha 的对话"]'));
  checks.push("Search reveals matching child, clearing restores manual collapse without changing local preference");

  await click(button("展开 Alpha 的对话"));
  await click(button("取消置顶 Alpha"));
  assert(document.querySelector('.project-navigation [aria-label="Alpha 的对话"]'));
  await click(button("置顶 Alpha"));
  assert(document.querySelector('.pinned-navigation [aria-label="Alpha 的对话"]'));
  assert.deepEqual(pinOrder(), ["Older pinned chat", "Alpha"]);
  await unmount();
  await mount();
  assert(document.querySelector('[aria-label="Alpha 的对话"]'));
  checks.push("Pin/unpin preserves expanded identity; remount restores local expansion; appended stable order");

  let release;
  holdWrite = new Promise((resolve) => { release = resolve; });
  const beforeRapid = postCount;
  await act(async () => {
    const target = button("置顶 Ordinary");
    target.click();
    target.click();
  });
  assert.equal(postCount, beforeRapid + 1);
  assert(button("取消置顶 Alpha").disabled);
  await act(async () => { release(); await holdWrite; });
  holdWrite = undefined;
  assert.deepEqual(pinOrder(), ["Older pinned chat", "Alpha", "Ordinary"]);
  assert.equal([...document.querySelectorAll("strong")].filter((item) => item.textContent === "Ordinary").length, 1);
  checks.push("Rapid actions serialized, controls disabled in flight, new pin appends and disappears from recent group");

  let releaseRead;
  let staleRead;
  holdRead = new Promise((resolve) => { releaseRead = resolve; });
  await act(async () => { staleRead = nav.reload(); });
  await click(button("取消置顶 Ordinary"));
  assert(button("置顶 Ordinary"));
  await act(async () => { releaseRead(); await staleRead; });
  holdRead = undefined;
  assert(button("置顶 Ordinary"));
  await click(button("置顶 Ordinary"));
  checks.push("An older GET completing after a mutation cannot roll back the server-confirmed pin state");

  failWrite = true;
  await click(button("取消置顶 Ordinary"));
  assert(document.querySelector('[role="alert"]')?.textContent.includes("Synthetic write failure"));
  assert(button("取消置顶 Ordinary"));
  failWrite = false;
  failRead = true;
  await act(async () => { await nav.reload(); });
  assert(button("取消置顶 Ordinary"));
  assert(document.querySelector('[role="alert"]')?.textContent.includes("Synthetic read failure"));
  failRead = false;
  await act(async () => { await nav.reload(); });
  assert(!document.querySelector('[role="alert"]'));
  checks.push("Read/write failure visible without overwriting existing pins; explicit refresh recovers");

  holdWrite = new Promise((resolve) => { release = resolve; });
  await click(button("取消置顶 Ordinary"));
  enabled = false;
  await render();
  enabled = true;
  await render();
  await act(async () => { release(); await holdWrite; });
  holdWrite = undefined;
  assert(button("置顶 Ordinary"));
  await click(button("置顶 Ordinary"));
  checks.push("Readiness interrupted during POST triggers a deferred reload and recovers authoritative state without waiting for focus");

  serverPins.push(missingPin);
  projectsInstalled = false;
  await render();
  await act(async () => { await nav.reload(); });
  assert(button("取消置顶 Older pinned chat"));
  assert(button("取消置顶 Ordinary"));
  assert.equal([...document.querySelectorAll("strong")].filter((item) => item.textContent === "项目暂不可用").length, 2);
  assert(!button("置顶 Needle project chat"));
  await click(button("取消置顶 项目暂不可用"));
  assert(!serverPins.some((pin) => pin.id === projectId));
  checks.push("Pins survive missing Projects plugin/targets; unresolved references disabled but can unpin; scoped fallback chats stay ineligible");

  projectsInstalled = true;
  compact = true;
  await render();
  await click(button("打开导航"));
  assert(document.querySelector('[role="dialog"]'));
  await click(button("收起 Alpha 的对话"));
  assert(document.querySelector('[role="dialog"]'));
  await click(button("展开 Alpha 的对话"));
  await click([...document.querySelectorAll("button.project-session")].find((item) => item.textContent.includes("Needle")));
  assert.equal(selected.at(-1), child.id);
  assert(!document.querySelector('[role="dialog"]'));
  await click(button("打开导航"));
  assert(document.querySelector('[aria-label="Alpha 的对话"]'));
  checks.push("Compact drawer stays open when toggling, closes on session activation, reopen preserves expansion");

  await unmount();
  // The actual controller must honor a resolved pin before the slow recent directory finishes.
  let controller;
  let releaseDirectory;
  const directoryRead = new Promise((resolve) => { releaseDirectory = resolve; });
  const scope = { computer: "fixture-computer" };
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
  const { desktopModels } = await import(pathToFileURL(resolve(repo, "scripts/webui-performance/fixtures.mjs")));
  globalThis.fetch = async (url, options = {}) => {
    const path = new URL(url, "http://navigation.local").pathname;
    const json = (data) => new Response(JSON.stringify(data), { status: 200 });
    if (path === "/api/shell/state") return json({ status: "ready", configured: true, chatReady: true });
    if (path === "/api/chat/navigation/pins") return json({ pins: [{ kind: "session", id: old.key }], sessions: [{ ...old, scope }] });
    if (path === "/api/chat/sessions") { await directoryRead; return json({ items: [], next_cursor: null }); }
    if (path === "/api/chat/models") return json({ ...desktopModels(1), unavailableRuntimes: [] });
    if (path === "/api/chat/plugin-ui/catalog") return json({ catalog_revision: "0".repeat(64), items: [] });
    if (path === "/api/chat/plugin-ui/query") {
      assert.equal(JSON.parse(options.body).method, "project.list");
      return json({ items: [{ id: projectId, name: "Alpha", archived: false, created_at: alpha.createdAt }] });
    }
    if (path.endsWith("/messages")) return json({ version: 2, items: [], through_seq: -1, before_seq: null, has_more: false });
    throw new Error(`Unexpected controller URL: ${url}`);
  };
  sessionStorage.clear();
  function ControllerHarness() { controller = useDesktopChatController(); return null; }
  root = createRoot(document.getElementById("root"));
  await act(async () => { root.render(React.createElement(ControllerHarness)); });
  for (let tick = 0; tick < 20 && !controller.navigationPins.ready; tick++) {
    await act(async () => { await new Promise((resolve) => setTimeout(resolve, 10)); });
  }
  assert(controller.navigationPins.ready);
  await act(async () => {
    await receivePluginUiCatalog({
      catalogRevision: "1".repeat(64), updating: false,
      plugins: [{ id: "projects", revision: "1", moduleUrl: "data:text/javascript,export default {slots:{}}", slots: [] }],
    });
  });
  await act(async () => { controller.startProjectChat(projectId); });
  assert.equal(controller.activeProject?.id, projectId);
  await act(async () => { controller.activateSession(old.key); });
  assert.equal(controller.activeProject, null);
  assert.equal(controller.activeSessionId, old.key);
  assert.equal(controller.sidebarSessions.find((session) => session.id === old.key)?.title, "Older pinned chat");
  assert(controller.canSend, controller.modelProblem);
  await act(async () => { await controller.sendMessage("Continue the pinned session", []); });
  const sent = frames.findLast((frame) => frame.type === "message.send");
  assert.equal(sent.session_id, old.key);
  assert.deepEqual(sent.session_dimensions, scope);
  await act(async () => { releaseDirectory(); await directoryRead; });
  await unmount();
  checks.push("Actual controller: resolved older default pin opens before recent directory, clears prior project draft/title state, and sends the original non-project scope intact");

  console.log(JSON.stringify({ boundary: "Production React components and hook in JSDOM; synthetic HTTP. No real-browser layout or backend proof.", passed: checks.length, checks }, null, 2));
} finally {
  try { if (root && dom) root.unmount(); } catch { /* Already unmounted. */ }
  dom?.window.close();
  await rm(temporary, { recursive: true, force: true });
}
