// 生产组件读取真实后端页面；JSDOM 不提供布局或截图证据。
// 先运行 docker/debug/content_view_scenario.py --display-sample <path>。
import { readFileSync, writeFileSync, mkdtempSync, symlinkSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { registerHooks } from "node:module";
import { build } from "esbuild";
import { JSDOM } from "jsdom";
import React from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { StickToBottom } from "use-stick-to-bottom";

const repo = process.cwd();
const sample = JSON.parse(readFileSync(resolve(process.argv[2]), "utf8"));
const temp = mkdtempSync(join(tmpdir(), "akashic-tool-display-ui-"));
symlinkSync(join(repo, "node_modules"), join(temp, "node_modules"), "dir");
registerHooks({ load(url, context, next) {
  return url.endsWith(".css") ? { format: "module", source: "export default {};", shortCircuit: true } : next(url, context);
} });
const entry = `export { DesktopTimelineMessages } from ${JSON.stringify(repo + "/frontend/chat/src/desktop-conversation.tsx")}; export { ChatMessageView, ReplyActivityView } from ${JSON.stringify(repo + "/frontend/chat/src/message-view.tsx")}; export * as timeline from ${JSON.stringify(repo + "/frontend/chat/src/message-timeline.ts")};`;
writeFileSync(join(temp, "entry.tsx"), entry);
await build({ entryPoints: [join(temp, "entry.tsx")], bundle: true, platform: "node", format: "esm", packages: "external",
  alias: { "@": repo + "/frontend/chat/src", mermaid: repo + "/frontend/chat/src/mermaid-stub.ts" }, loader: { ".css": "empty" },
  define: { "import.meta.env.DEV": "false" }, outfile: join(temp, "components.mjs") });
const dom = new JSDOM('<html><body><div id="root"></div></body></html>', { url: "http://fixture.test/", pretendToBeVisual: true });
const { window } = dom;
for (const key of ["window", "document", "HTMLElement", "Element", "Node", "NodeFilter", "DOMException", "MutationObserver", "Event", "MouseEvent", "getComputedStyle", "sessionStorage", "localStorage"]) {
  Object.defineProperty(globalThis, key, { value: key === "window" ? window : window[key], configurable: true });
}
Object.defineProperty(globalThis, "navigator", { value: window.navigator, configurable: true });
globalThis.requestAnimationFrame = window.requestAnimationFrame.bind(window);
globalThis.cancelAnimationFrame = window.cancelAnimationFrame.bind(window);
globalThis.CSS = { escape: (value) => value };
window.matchMedia = () => ({ matches: false, addEventListener() {}, removeEventListener() {} });
class ResizeObserver { observe() {} unobserve() {} disconnect() {} }
globalThis.ResizeObserver = window.ResizeObserver = ResizeObserver;
window.HTMLElement.prototype.scrollTo = function() {};
window.HTMLElement.prototype.scrollIntoView = function() {};
const { DesktopTimelineMessages, ChatMessageView, ReplyActivityView, timeline } = await import(pathToFileURL(join(temp, "components.mjs")));
const host = document.getElementById("root");
const root = createRoot(host);
const errors = [];
const render = (items) => flushSync(() => root.render(React.createElement(StickToBottom, {},
  React.createElement(DesktopTimelineMessages, { messages: items.map(timeline.readTimelineMessage), activities: [], status: "idle",
    copiedMessageId: "", messageElementsRef: { current: new Map() }, onReply() {}, onCopied() {}, onError: (error) => errors.push(String(error)) }))));
const check = (condition, label) => { if (!condition) throw Error(label); };
const openTools = () => {
  for (const button of host.querySelectorAll("button.tool-group-summary")) if (button.getAttribute("aria-expanded") !== "true") flushSync(() => button.click());
  for (const button of host.querySelectorAll("button.tool-step-summary")) if (button.getAttribute("aria-expanded") !== "true") flushSync(() => button.click());
};
try {
  render(sample.messages);
  // 实际展开 read_content 行，结果必须包含原文且没有额外占位结果行。
  for (const button of host.querySelectorAll("button.tool-group-summary")) flushSync(() => button.click());
  const readButtons = [...host.querySelectorAll("button.tool-step-summary")].filter((button) => button.textContent.includes("read_content"));
  check(readButtons.length === 2, "read_content calls are present");
  for (const button of readButtons) flushSync(() => button.click());
  check(host.querySelectorAll('[data-message-kind="tool_result"]').length === 0, "loaded calls own result display");
  check(host.textContent.includes("END_OF_RESULT"), "full read content appears in result panel");
  check(!host.textContent.includes("无法展示此内容"), "no unsupported-result placeholder");
  render(sample.tail);
  check(host.querySelectorAll('[data-message-kind="tool_result"]').length === 2, "paged results remain visible without calls");
  check(host.textContent.includes("END_OF_RESULT"), "paged result resolves source outside the page");
  render(sample.data);
  const call = host.querySelector("button.tool-step-summary");
  flushSync(() => call.click());
  check(host.textContent.includes("失败"), "failed status remains visible");
  check(host.textContent.includes("**literal** <script>literal</script>"), "failed result keeps literal body");
  check(!host.querySelector("script, .tool-result-data em, .tool-result-data strong"), "tool data does not interpret HTML or Markdown");
  const labels = [...host.querySelectorAll(".tool-result-fields dt")].map((node) => node.textContent);
  check(["zero", "false", "null", "empty"].every((label) => labels.includes(label)), "false, zero, null and empty fields survive");
  const nested = host.querySelector(".tool-result-fields details");
  nested.open = true;
  check(nested.textContent.includes("items"), "nested data can be expanded");
  // 同一生产组件的分页路径读取 JSON 字符串，终端换行是字面文本。
  const row = JSON.parse(JSON.stringify(sample.tail[0]));
  row.body.parts = [{ kind: "text", value: JSON.stringify({ output: "first\n**second**", command: "echo one\necho two" }) }];
  render([row]);
  const texts = [...host.querySelectorAll("pre.tool-result")].map((node) => node.textContent);
  check(texts.includes("first\n**second**") && texts.includes("echo one\necho two"), "JSON string fields preserve real newlines");
  check(!host.querySelector("em, strong:not(.timeline-result-heading strong)"), "terminal JSON does not become Markdown");
  // 实时进度读取与历史相同的真实消息，不能再次丢失结构化结果。
  const data = sample.data.map(timeline.readTimelineMessage);
  flushSync(() => root.render(React.createElement(ReplyActivityView, {
    activity: { session_id: data[0].session_id, source: data[0].source, handle: "active", active: true, preview: null },
    committed: new Set(), processMessages: data, toolResults: timeline.timelineToolResults(data),
  })));
  openTools();
  check(host.textContent.includes("**literal** <script>literal</script>"), "live progress keeps failed structured result");
  // 旧消息入口也使用同一结果组件；复制必须保留原始值，不能复制展示摘要。
  const copied = [];
  const rawJson = row.body.parts[0].value;
  const values = [null, false, 0, "", "{broken JSON", [], {}, [rawJson, { ok: false }], rawJson];
  flushSync(() => root.render(React.createElement(ChatMessageView, {
    message: { id: "legacy", role: "assistant", content: "", blocks: values.map((value, index) => ({
      kind: "tool", callId: String(index), name: "any_tool", input: {}, output: value, status: "output-available",
    })) }, onCopyToolDetail: (value) => copied.push(value),
  })));
  openTools();
  const results = [...host.querySelectorAll(".tool-detail-section")];
  check(results.length === values.length, "empty and scalar results have result panels");
  check(results.slice(0, 5).map((node) => node.querySelector("pre").textContent).join("|") === "null|false|0||{broken JSON", "scalar and malformed JSON values remain literal");
  check(results[7].textContent.includes("first\n**second**"), "multipart JSON and data render together");
  for (const result of results) flushSync(() => result.querySelector("button").click());
  check(copied.length === values.length && copied.at(-1) === rawJson && copied[0] === "null" && copied[1] === "false", "copy preserves full original values");
  // 重叠页允许只读展示刷新，但原始引用依然是不可变事实。
  const refreshed = JSON.parse(JSON.stringify(sample.tail[0]));
  refreshed.body.parts[0].rendered = "updated display";
  render(timeline.mergeTimelineMessages(sample.tail, [refreshed]));
  check(host.textContent.includes("updated display"), "overlapping page refreshes derived display");
  check(errors.length === 0, "no component errors: " + errors.join(", "));
  console.log(JSON.stringify({ status: "passed", checks: ["read_content panel", "paged reference", "no duplicate result rows", "unknown data fallback", "failed result body", "scalar preservation", "nested data", "literal JSON strings", "live progress", "legacy message", "multipart data", "original copy", "display refresh"] }));
} finally {
  flushSync(() => root.unmount());
  window.close();
  rmSync(temp, { recursive: true, force: true });
}
