// JSDOM production-component scenario; not real-browser, layout, screenshot, or frame evidence.
// Requires Node >=22.15 (registerHooks); dependencies come from the selected source checkout.
// Usage: node scripts/akasha_recall_ui_scenario.mjs --source . --output /tmp/recall-ui.json [--baseline] [--recall-sample /tmp/backend.json]
process.env.OTEL_SDK_DISABLED = "true";
process.env.DO_NOT_TRACK = "1";
process.env.NODE_ENV = "production";
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, symlinkSync, rmSync } from "node:fs";
import { dirname, resolve, join } from "node:path";
import { tmpdir } from "node:os";
import { pathToFileURL } from "node:url";
import { createRequire, registerHooks } from "node:module";
import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
const args = process.argv.slice(2);
const scenarioSha256 = createHash("sha256").update(readFileSync(process.argv[1])).digest("hex");
const option = (name, fallback) => {
  const i = args.indexOf(name);
  if (i < 0) return fallback;
  if (!args[i + 1] || args[i + 1].startsWith("--")) throw Error("Missing value for " + name);
  return args[i + 1];
};
const repo = resolve(option("--source", process.cwd()));
const outputPath = resolve(option("--output", "/tmp/akasha-recall-ui-report.json"));
const mode = args.includes("--baseline") ? "baseline" : "candidate";
const samplePath = option("--recall-sample", "");
const sampleBytes = samplePath ? readFileSync(resolve(samplePath)) : null;
const sampleDocument = sampleBytes ? JSON.parse(sampleBytes.toString("utf8")) : null;
const backendSample = sampleDocument?.responses?.tools_open ?? sampleDocument;
const backendContract = sampleBytes ? { executed: true, path: resolve(samplePath), sha256: createHash("sha256").update(sampleBytes).digest("hex"), source: sampleDocument.source, source_sha256: sampleDocument.source_sha256 } : { executed: false, reason: "Pass --recall-sample with a real backend for_turn response or backend scenario report" };
if (backendSample && (!Array.isArray(backendSample.items) || typeof backendSample.pending !== "boolean" || backendSample.legacy_pending || backendSample.next_offset != null)) throw Error("Contract sample must be a complete settled-page aggregate, not legacy-progress/pagination output");
const require2 = createRequire(join(repo, "package.json"));
const { build } = await import(pathToFileURL(require2.resolve("esbuild")).href);
const temp = mkdtempSync(join(tmpdir(), "akashic-recall-ui-"));
process.once("exit", () => rmSync(temp, { recursive: true, force: true }));
symlinkSync(join(repo, "node_modules"), join(temp, "node_modules"), "dir");
const files = ["frontend/chat/src/desktop-chat-view.tsx", "frontend/chat/src/desktop-conversation.tsx", "frontend/chat/src/message-view.tsx", "frontend/chat/src/message-timeline.ts", "frontend/chat/src/plugin-ui-runtime.tsx", "frontend/chat/src/use-desktop-chat-controller.ts", "frontend/plugins/akasha/src/plugin-ui.js", "plugins/akasha/message_ui.js"];
const sourceReceipt = { repo, commit: execFileSync("git", ["-C", repo, "rev-parse", "HEAD"], { encoding: "utf8" }).trim(), dirty: execFileSync("git", ["-C", repo, "status", "--porcelain"], { encoding: "utf8" }).trim(), sha256: Object.fromEntries(files.map((file) => [file, createHash("sha256").update(readFileSync(join(repo, file))).digest("hex")])) };
const componentEntry = `export { DesktopTimelineMessages } from ${JSON.stringify(repo + "/frontend/chat/src/desktop-conversation.tsx")};export { ReplyActivityView } from ${JSON.stringify(repo + "/frontend/chat/src/message-view.tsx")};export { receivePluginUiCatalog } from ${JSON.stringify(repo + "/frontend/chat/src/plugin-ui-runtime.tsx")};export * as timeline from ${JSON.stringify(repo + "/frontend/chat/src/message-timeline.ts")};`;
writeFileSync(join(temp, "entry.tsx"), componentEntry);
const pluginEntry = `import definition from ${JSON.stringify(repo + "/frontend/plugins/akasha/src/plugin-ui.js")};
const record=(kind,context)=>window.__recallTrace?.push({kind,t:performance.now(),sessionId:context.sessionId,messageId:context.messageId,source:context.block?.source});
const renderer=definition.slots['turn.before_reasoning'];
const wrapped={...renderer,mount(host,context){record('mount',context); const query=context.query; const wrappedContext={...context,query(method,payload,options){window.__recallTrace?.push({kind:'plugin-query',t:performance.now(),method,payload,options,messageId:context.messageId});return query(method,payload,options)}}; const dispose=renderer.mount(host,wrappedContext);return ()=>{record('dispose',context);dispose?.()}}};
export default {...definition,slots:{...definition.slots,'turn.before_reasoning':wrapped}};`;
writeFileSync(join(temp, "plugin-entry.js"), pluginEntry);
try {
  await build({ entryPoints: [join(temp, "entry.tsx")], bundle: true, platform: "node", format: "esm", packages: "external", alias: { "@": repo + "/frontend/chat/src", mermaid: repo + "/frontend/chat/src/mermaid-stub.ts" }, loader: { ".css": "empty" }, define: { "import.meta.env.DEV": "false", "process.env.NODE_ENV": '"production"' }, outfile: join(temp, "component.mjs") });
  await build({ entryPoints: [join(temp, "plugin-entry.js")], bundle: true, format: "esm", target: "es2022", loader: { ".css": "empty" }, outfile: join(temp, "plugin.js") });
} catch (error2) {
  rmSync(temp, { recursive: true, force: true });
  throw error2;
}
registerHooks({ load(url, context, next) {
  return url.endsWith(".css") ? { format: "module", source: "export default {};", shortCircuit: true } : next(url, context);
} });
function validateChecks(checks2, mode2) {
  const byLabel = new Map(checks2.map((c) => [c.label, c]));
  const assertions = [];
  const expect = (label, predicate) => assertions.push({ label, passed: !!byLabel.has(label) && !!predicate(byLabel.get(label)) });
  expect("120 token updates", (x) => x.queries === 0);
  expect("first tool identical snapshot", (x) => mode2 === "baseline" ? !x.hostConnected : x.hostConnected && x.groupConnected && x.detailsConnected && x.open && x.currentHostSame);
  expect("first-tool refresh loading marker", (x) => mode2 === "baseline" ? x.visible : !x.visible);
  expect("10s idle", (x) => mode2 === "baseline" ? x.queries > 0 : x.queries === 0);
  expect("later tool recall visible", (x) => x.visible);
  expect("session cleanup", (x) => x.oldContentGone && x.oldHostDisconnected && (mode2 === "baseline" || x.aborted === 1));
  if (mode2 === "baseline") return assertions;
  if (backendSample) expect("real backend return contract", (x) => x.noTopLevelSchema && x.panelCount === 1 && x.detailsCount === 2 && x.noLocalError && x.queries === 1);
  expect("inflight trailing and A late after B", (x) => x.visible && x.aQueries === 2 && x.bQueries === 1);
  expect("refresh failure keeps content and details", (x) => x.errorVisible && x.groupSame && x.detailsSame && x.open);
  expect("retry error cleared", (x) => x.errorHidden && x.groupSame && x.detailsSame);
  expect("closed history and unrelated source avoid queries", (x) => x.historyInputs === 100 && x.historyInitialQueries === 100 && x.abandonedInputs === 50 && x.closedQueries === 0 && x.abandonedQueries === 0 && x.otherQueries === 0);
  expect("final preserves existing DOM", (x) => x.hostConnected && x.groupConnected && x.detailsConnected && x.open);
  expect("hot reload revision cleanup", (x) => x.oldHostDisconnected && x.aborted === 1 && x.newMount);
  expect("uninstall cleanup", (x) => x.hostCount === 0);
  expect("history tail fallback deduplicates", (x) => x.fallbackCount === 1 && x.finalCount === 1 && x.fallbackDisconnected && x.expandedRestored);
  expect("empty successful refresh has no first-read spinner", (x) => !x.visibleStatus);
  expect("unmount cancels in-flight query", (x) => x.aborted === 1 && x.rootEmpty);
  return assertions;
}
const { JSDOM } = await import(pathToFileURL(require2.resolve("jsdom")).href);
const dom = new JSDOM('<!DOCTYPE html><html><head></head><body><div id="root"></div></body></html>', { url: "http://fixture.test/", pretendToBeVisual: true });
const { window } = dom;
for (const key of ["window", "document", "HTMLElement", "Element", "Node", "NodeFilter", "DOMException", "sessionStorage", "localStorage", "MutationObserver", "Event", "MouseEvent", "getComputedStyle"]) Object.defineProperty(globalThis, key, { value: key === "window" ? window : window[key], configurable: true, writable: true });
Object.defineProperty(globalThis, "navigator", { value: window.navigator, configurable: true });
globalThis.requestAnimationFrame = window.requestAnimationFrame.bind(window);
globalThis.cancelAnimationFrame = window.cancelAnimationFrame.bind(window);
globalThis.CSS = { escape: (x) => x };
window.matchMedia = () => ({ matches: false, addListener() {
}, removeListener() {
}, addEventListener() {
}, removeEventListener() {
} });
// JSDOM has no viewport/layout: make every fixture panel visible; do not infer visual behavior.
class IntersectionObserver {
  constructor(cb) {
    this.cb = cb;
  }
  observe(target) {
    queueMicrotask(() => this.cb([{ target, isIntersecting: true }]));
  }
  disconnect() {
  }
}
class ResizeObserver {
  observe() {
  }
  unobserve() {
  }
  disconnect() {
  }
}
globalThis.IntersectionObserver = window.IntersectionObserver = IntersectionObserver;
globalThis.ResizeObserver = window.ResizeObserver = ResizeObserver;
window.HTMLElement.prototype.scrollTo = function() {
};
window.HTMLElement.prototype.scrollIntoView = function() {
};
const React = (await import(pathToFileURL(require2.resolve("react")).href)).default;
const { createRoot } = await import(pathToFileURL(require2.resolve("react-dom/client")).href);
const { StickToBottom } = await import(pathToFileURL(require2.resolve("use-stick-to-bottom")).href);
const components = await import(pathToFileURL(join(temp, "component.mjs")).href);
const { DesktopTimelineMessages, ReplyActivityView, receivePluginUiCatalog, timeline } = components;
window.__recallTrace = [];
const started = Date.now(), queries = [], checks = [], snapshots = [], errors = [];
let stage = "initial", auto = false, tool = false, late = false, delayMs = 350, failFor = null, pending = true, refresh = 0;
const session = "s1";
const message = (id, seq, body, source = "conversation", session_id = session) => ({ id, seq, body, session_id, source, author: body.kind === "input" ? "user" : "assistant", timestamp: "2026-09-30T00:00:00Z", attachments: [], metadata: {} });
const input = (id, seq, source = "conversation", session_id = session) => message(id, seq, { kind: "input", parts: [{ kind: "text", value: id }] }, source, session_id);
const output = (id, seq, parts, finish = "continue", source = "conversation") => message(id, seq, { kind: "output", parts, finish }, source);
const call = () => ({ kind: "tool_call", binding_id: "fixture-binding", name: "akasha.recall", arguments: { query: "synthetic" } });
const item = (id, text) => ({ schema: "akasha.queries.v1", query_id: id, hits: [{ lane: "dense", score: 0.9, messages: [{ message_id: id + "-memory", preview: text, presented: true }], sources: [] }] });
// Aggregate shape mirrors InspectorQueries.for_turn: schema belongs to item detail only.
const responseFor = (p) => {
  if (stage === "backend-contract") return structuredClone(backendSample);
  const id = p.payload.message_id;
  const closed = ["closed-input", "closed-output", "abandoned-input"].includes(id) || id.startsWith("closed-history-") || id.startsWith("abandoned-history-");
  const other = id === "other-input";
  const isB = id === "input-B";
  const isC = p.session_id === "s2";
  return { input_message_id: closed ? id : other ? "other-input" : isB ? "input-B" : isC ? "input-C" : "input-A", pending: closed ? false : pending, legacy_pending: false, next_offset: null, items: closed ? [item("closed", "CLOSED_HISTORY")] : other ? [item("other", "OTHER_SOURCE")] : auto || isC ? [item(isB ? "B" : isC ? "C" : "A", isB ? "AUTOMATIC_B" : isC ? "AUTOMATIC_C" : "AUTOMATIC_A"), ...!isB && !isC && tool ? [item("tool-A", "ACTIVE_TOOL_RECALL")] : [], ...!isB && !isC && late ? [item("late-A", "LATE_A_RECALL")] : []] : [] };
};
globalThis.fetch = async (url, opts = {}) => {
  if (url !== "/api/chat/plugin-ui/query") throw Error("Unexpected URL " + url);
  const p = JSON.parse(opts.body);
  const result = responseFor(p);
  const failure = failFor === "any" || failFor === p.payload.message_id;
  if (failure) failFor = null;
  const q = { index: queries.length + 1, t: Date.now() - started, stage, method: p.method, payload: p.payload, sessionId: p.session_id, resultIds: result.items.map((i) => i.query_id), failure, delayMs };
  queries.push(q);
  return await new Promise((resolve2, reject) => {
    const onAbort = () => {
      clearTimeout(timer);
      q.abortedAt = Date.now() - started;
      reject(new DOMException("Aborted", "AbortError"));
    };
    const timer = setTimeout(() => {
      opts.signal?.removeEventListener("abort", onAbort);
      q.completedAt = Date.now() - started;
      resolve2(new Response(JSON.stringify(failure ? { detail: { code: "fixture_failure", message: "Synthetic refresh failed" } } : result), { status: failure ? 500 : 200, headers: { "content-type": "application/json" } }));
    }, delayMs);
    opts.signal?.addEventListener("abort", onAbort, { once: true });
  });
};
const catalog = (rev = "1", plugins = true) => ({ catalogRevision: rev, updating: false, plugins: plugins ? [{ id: "akasha", revision: rev, moduleUrl: pathToFileURL(join(temp, "plugin.js")).href + "?revision=" + rev, slots: ["turn.before_reasoning"], navigation: { label: "Akasha", description: "Synthetic fixture" } }] : [] });
await receivePluginUiCatalog(catalog());
let rows = [input("input-A", 20)], activities = [];
const root = createRoot(document.getElementById("root"));
const ref = { current: /* @__PURE__ */ new Map() };
const no = () => {
};
const onError = (e) => errors.push(String(e));
// Match DesktopChatView composition without its network/controller/sidebar/composer.
// Refresh increments model an already-delivered reconnect/source snapshot invalidation.
function render() {
  const groups = timeline.timelineReplyGroups(rows, activities), results = timeline.timelineToolResults(rows), committed = new Set(rows.map((m) => m.id));
  const starts = timeline.timelineInputStarts?.(rows) || /* @__PURE__ */ new Map();
  const tokens = timeline.timelineSourceRefreshTokens?.(rows, activities, refresh) || /* @__PURE__ */ new Map();
  root.render(React.createElement(StickToBottom, { initial: "instant", resize: "instant" }, React.createElement(StickToBottom.Content, null, React.createElement(DesktopTimelineMessages, { messages: rows, activities, refresh, status: activities.some((a) => a.active) ? "streaming" : "idle", messageElementsRef: ref, copiedMessageId: "", onReply: no, onCopied: no, onError }), ...activities.map((a) => React.createElement(ReplyActivityView, { key: a.handle, activity: a, committed, processMessages: groups.active.get(a.handle), toolResults: results, inputStarts: starts, refreshToken: tokens.get(timeline.timelineSourceKey?.(a)), onError })))));
}
function activity(id = "output-A", text = "", active = true) {
  activities = [{ session_id: session, source: "conversation", handle: "reply-A", active, preview: active ? { message_id: id, text, thinking: "" } : null }];
  render();
}
function append(row) {
  rows = [...rows, row];
  render();
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const wait = async (fn, label, timeout = 5e3) => {
  const start = Date.now();
  while (!fn()) {
    if (Date.now() - start > timeout) throw Error("Timeout " + label);
    await sleep(10);
  }
  return fn();
};
const panel = () => document.querySelector('.plugin-ui-host[data-plugin="akasha"]');
const visible = (text) => document.body.textContent.includes(text);
const snap = (label) => snapshots.push({ label, t: Date.now() - started, queries: queries.length, lifecycle: window.__recallTrace.slice(), panels: [...document.querySelectorAll('.plugin-ui-host[data-plugin="akasha"]')].map((p) => ({ text: p.textContent, html: p.innerHTML })) });
let initialHost, initialGroup, initialDetails;
const preserve = (label) => checks.push({ label, hostConnected: !!initialHost?.isConnected, groupConnected: !!initialGroup?.isConnected, detailsConnected: !!initialDetails?.isConnected, open: !!initialDetails?.open, currentHostSame: panel() === initialHost });
let error;
try {
  render();
  await sleep(500);
  snap("input");
  stage = "draft";
  auto = true;
  activity();
  await wait(() => document.querySelector(".akasha-plugin-ui-recall-group"), "automatic recall");
  initialHost = panel();
  initialGroup = document.querySelector(".akasha-plugin-ui-recall-group");
  initialDetails = document.querySelector('details[data-lane="dense"]');
  initialDetails.open = true;
  snap("draft-auto-open");
  stage = "token-text";
  let before = queries.length;
  for (let i = 0; i < 120; i++) {
    activity("output-A", "Token " + i);
    await sleep(2);
  }
  await sleep(50);
  checks.push({ label: "120 token updates", queries: queries.length - before });
  preserve("after tokens");
  stage = "first-tool";
  append(output("output-A", 21, [{ kind: "model.facts", value: { call_record_id: "record-A", thinking: "Fixture thinking" } }, call()]));
  await sleep(200);
  snap("first-tool-delay-200ms");
  checks.push({label:"first-tool refresh loading marker",visible:[...document.querySelectorAll(".akasha-plugin-ui-query-status")].some(n=>!n.hidden&&n.textContent.includes("正在读取"))});
  await sleep(500);
  preserve("first tool identical snapshot");
  snap("first-tool-settled");
  stage = "idle-10s";
  before = queries.length;
  await sleep(10050);
  checks.push({ label: "10s idle", queries: queries.length - before });
  preserve("after 10s idle");
  stage = "tool-result";
  tool = true;
  append(message("result-A", 22, { kind: "tool_result", call_ref: { message_id: "output-A", part_index: 1 }, outcome: "success", parts: [{ kind: "text", value: "Persisted tool Recall" }] }));
  await sleep(1500);
  checks.push({ label: "later tool recall visible", visible: visible("ACTIVE_TOOL_RECALL") });
  snap("tool-result");
  if (mode !== "baseline") {
    stage = "inflight-late-A";
    delayMs = 700;
    append(output("output-A2", 23, [call()]));
    await sleep(100);
    append(input("input-B", 24));
    late = true;
    append(message("result-A2", 25, { kind: "tool_result", call_ref: { message_id: "output-A2", part_index: 0 }, outcome: "success", parts: [{ kind: "text", value: "A result after input B" }] }));
    await sleep(1800);
    checks.push({ label: "inflight trailing and A late after B", visible: visible("LATE_A_RECALL"), aQueries: queries.filter((q) => q.stage === stage && q.payload.message_id === "input-A").length, bQueries: queries.filter((q) => q.stage === stage && q.payload.message_id === "input-B").length });
    snap("late-A");
    stage = "refresh-failure";
    delayMs = 350;
    failFor = "input-A";
    const oldContent = document.querySelector('[data-message-id="input-A"] .akasha-plugin-ui-recall-group');
    const oldDetail = oldContent.querySelector("details");
    oldDetail.open = true;
    append(output("output-B", 26, [{ kind: "text", value: "Progress B" }]));
    await sleep(650);
    checks.push({ label: "refresh failure keeps content and details", errorVisible: visible("Synthetic refresh failed"), groupSame: oldContent.isConnected, detailsSame: oldDetail.isConnected, open: oldDetail.open });
    snap("failure");
    document.querySelector('[data-message-id="input-A"] .akasha-plugin-ui-query-status button')?.click();
    await sleep(500);
    checks.push({ label: "retry error cleared", errorHidden: !visible("Synthetic refresh failed") || document.querySelector('[data-message-id="input-A"] .akasha-plugin-ui-query-status').hidden, groupSame: oldContent.isConnected, detailsSame: oldDetail.isConnected });
    stage = "history-source-fixture";
    delayMs = 10;
    const historyInputs = new Set();
    const abandonedInputs = new Set();
    const historyRows = Array.from({length:100}, (_, index) => {
      const abandoned = index % 2 === 1;
      const id = `${abandoned ? "abandoned" : "closed"}-history-${index}`;
      historyInputs.add(id);
      if (abandoned) abandonedInputs.add(id);
      const seq = index * 3;
      return [input(id,seq), output(`${id}-output`,seq+1,abandoned?[call()]:[{kind:"text",value:"Closed history reply"}],abandoned?"continue":"complete"),
        ...(abandoned?[message(`${id}-control`,seq+2,{kind:"control",action:"abandon",through_seq:seq+1,reason:"Synthetic abandoned history"})]:[])];
    }).flat();
    rows = [...historyRows, input("other-input", 301, "other-source"), ...rows.map(row=>({...row,seq:row.seq+1000}))];
    render();
    await wait(()=>queries.filter(q=>historyInputs.has(q.payload.message_id)&&q.completedAt).length===100,"100 closed and abandoned initial reads",15000);
    await sleep(100);
    const historicalBefore=queries.filter(q=>historyInputs.has(q.payload.message_id)).length;
    const abandonedBefore=queries.filter(q=>abandonedInputs.has(q.payload.message_id)).length;
    const otherBefore=queries.filter(q=>q.payload.message_id==="other-input").length;
    stage = "source-isolation";
    delayMs=350;
    append(output("same-source-progress", rows.at(-1).seq+1, [{kind:"text",value:"Next same source progress"}]));
    await sleep(650);
    checks.push({label:"closed history and unrelated source avoid queries",historyInputs:historyInputs.size,historyInitialQueries:historicalBefore,abandonedInputs:abandonedInputs.size,abandonedQueries:queries.filter(q=>abandonedInputs.has(q.payload.message_id)).length-abandonedBefore,closedQueries:queries.filter(q=>historyInputs.has(q.payload.message_id)).length-historicalBefore,otherQueries:queries.filter(q=>q.payload.message_id==="other-input").length-otherBefore});
  }
  const beforeFinalHost = document.querySelector('[data-message-id="input-A"] .plugin-ui-host') || panel();
  const beforeFinalGroup = beforeFinalHost?.querySelector(".akasha-plugin-ui-recall-group");
  const beforeFinalDetails = beforeFinalHost?.querySelector("details");
  stage = "final";
  pending = false;
  append(output("final-output", rows.at(-1).seq + 1, [{ kind: "text", value: "Final fixture response" }], "complete"));
  activity("final-preview", "", false);
  await sleep(850);
  checks.push({ label: "final preserves existing DOM", hostConnected: !!beforeFinalHost?.isConnected, groupConnected: !!beforeFinalGroup?.isConnected, detailsConnected: !!beforeFinalDetails?.isConnected, open: !!beforeFinalDetails?.open });
  snap("final");
  stage = "session-cleanup";
  pending = true;
  delayMs = 700;
  rows = [input("new-pending-input", 0, "conversation", "s-pending")];
  activities = [];
  render();
  await sleep(80);
  rows = [input("input-C", 0, "conversation", "s2")];
  render();
  await sleep(850);
  checks.push({ label: "session cleanup", oldContentGone: !visible("AUTOMATIC_A"), oldHostDisconnected: !initialHost.isConnected, aborted: queries.filter((q) => q.stage === stage && q.abortedAt).length });
  snap("session-cleanup");
  if (mode !== "baseline") {
    stage = "hot-reload";
    const old = panel();
    const mountsBeforeReload=window.__recallTrace.filter(x=>x.kind==="mount").length;
    refresh++;
    render();
    await sleep(80);
    await receivePluginUiCatalog(catalog("2"));
    await sleep(850);
    checks.push({ label: "hot reload revision cleanup", oldHostDisconnected: !old.isConnected, aborted: queries.filter((q) => q.stage === stage && q.abortedAt).length, newMount: window.__recallTrace.filter((x) => x.kind === "mount").length === mountsBeforeReload + 1 });
    await receivePluginUiCatalog(catalog("3", false));
    await sleep(50);
    checks.push({ label: "uninstall cleanup", hostCount: document.querySelectorAll('.plugin-ui-host[data-plugin="akasha"]').length });
    await receivePluginUiCatalog(catalog("4"));
    stage = "history-tail";
    delayMs = 350;
    pending = false;
    rows = [output("tail-output", 1, [{ kind: "text", value: "Tail output" }], "complete")];
    render();
    await sleep(600);
    const fallback = document.querySelector(".plugin-ui-host");
    document.querySelector('details[data-lane="dense"]').open = true;
    const fallbackCount = document.querySelectorAll(".akasha-plugin-ui-recall-group").length;
    rows = [input("input-A", 0), ...rows];
    render();
    await sleep(600);
    checks.push({ label: "history tail fallback deduplicates", fallbackCount, finalCount: document.querySelectorAll(".akasha-plugin-ui-recall-group").length, fallbackDisconnected: !fallback.isConnected, expandedRestored: document.querySelector('details[data-lane="dense"]').open });
    snap("history-tail");
    if (backendSample) {
      stage = "backend-contract";
      const countBefore = queries.length;
      rows = [input("contract-input", 0, "contract-source", "contract-session")];
      render();
      await sleep(650);
      checks.push({label:"real backend return contract", noTopLevelSchema: !Object.hasOwn(backendSample,"schema"), panelCount:document.querySelectorAll(".akasha-plugin-ui-recall-group").length, detailsCount:document.querySelectorAll("details[data-lane]").length, noLocalError:![...document.querySelectorAll(".akasha-plugin-ui-query-status")].some(n=>!n.hidden), queries:queries.length-countBefore});
      snap("backend-contract");
    }
    stage = "empty-success";
    pending = true;
    auto = false;
    tool = false;
    late = false;
    rows = [input("empty-input", 0, "empty-source", "empty-session")];
    render();
    await sleep(550);
    refresh++;
    render();
    await sleep(200);
    checks.push({ label: "empty successful refresh has no first-read spinner", visibleStatus: [...document.querySelectorAll(".akasha-plugin-ui-query-status")].some((n) => !n.hidden && n.textContent.includes("\u6B63\u5728\u8BFB\u53D6")) });
    await sleep(250);
    stage = "unmount";
    delayMs = 700;
    refresh++;
    render();
    await sleep(80);
    root.unmount();
    await sleep(800);
    checks.push({ label: "unmount cancels in-flight query", aborted: queries.filter((q) => q.stage === stage && q.abortedAt).length, rootEmpty: document.getElementById("root").childElementCount === 0 });
  }
} catch (e) {
  error = String(e);
  console.error(e);
} finally {
  try {
    root.unmount();
  } catch {
  }
  ;
  const report = { mode, nodeVersion:process.version, source: sourceReceipt, scenarioSha256, backendContract, boundary: "JSDOM only, real production React timeline/generic plugin runtime/Akasha renderer; synthetic Fetch/Recall; no real browser/layout/frame proof", checks, queries, snapshots, lifecycle: window.__recallTrace, error, errors };
  report.assertions = validateChecks(checks, mode);
  report.passed = !error && !errors.length && report.assertions.every((x) => x.passed);
  mkdirSync(dirname(outputPath), { recursive: true });
  writeFileSync(outputPath, JSON.stringify(report, null, 2) + "\n");
  if (!report.passed) process.exitCode = 1;
  console.log(JSON.stringify({ mode, checks, queries: queries.length, error, errors }, null, 2));
  dom.window.close();
  rmSync(temp, { recursive: true, force: true });
}
if (error) process.exitCode = 1;
