/** 流式长回复渲染基准：真实 Chromium 经 harness 构建 + 夹具 WS 推流，测流式期间主线程占用与落定延迟。
 * 指标：streamLongTasksMs（流式窗口长任务总量/最大）、settleRowsMs（末个增量→落定行出现）、
 * settleRichMs（末个增量→落定消息全部代码块进入高亮路径）、profile（CDP self time 归因用）。
 * 用法：node scripts/webui-performance/measure-stream-reply.mjs --browser <chromium> --runs 3 --output <dir> */
import { createServer } from "node:http";
import { mkdtempSync, mkdirSync, readFileSync, existsSync, writeFileSync } from "node:fs";
import { resolve, dirname, extname } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import { parseArgs } from "node:util";
import { WebSocketServer } from "ws";
import { chromium } from "playwright-core";
import { desktopModels } from "./fixtures.mjs";

const { values } = parseArgs({ options: {
  browser: { type: "string", default: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium" },
  runs: { type: "string", default: "3" },
  output: { type: "string", default: "" },
  label: { type: "string", default: "stream" },
  source: { type: "string", default: "" },
} });

const repo = values.source ? resolve(values.source) : resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const root = mkdtempSync(resolve(tmpdir(), "akashic-stream-bench-"));
console.info("artifact: " + root);
const build = spawnSync(process.execPath, [resolve(repo, "node_modules/vite/bin/vite.js"),
  "build", "--config", "frontend/chat/vite.harness.config.ts", "--outDir", root + "/dist"],
  { cwd: repo, encoding: "utf8" });
writeFileSync(root + "/build.log", build.stdout + build.stderr);
if (build.status !== 0) throw new Error("前端构建失败：" + root + "/build.log");

// ---- 夹具：30 对历史问答；message.send 后按固定节奏推流式预览，推完落定。 ----
const sessionId = "akashic:stream-bench";
const message = (seq, author, body, id) => ({ id: id ?? `m${seq}`, seq, author, source: "akashic",
  timestamp: new Date(Date.UTC(2026, 9, 1, 8, 0, seq)).toISOString(), session_id: sessionId,
  attachments: [], metadata: {}, body });
const items = [];
for (let index = 0; index < 30; index += 1) {
  items.push(message(items.length, "user", { kind: "input", parts: [{ kind: "text", value: `历史第 ${index} 问` }] }));
  items.push(message(items.length, "assistant", { kind: "output", finish: "complete",
    parts: [{ kind: "text", value: `历史第 ${index} 条回复。\n\n\`\`\`python\ndef history_${index}():\n    return ${index}\n\`\`\`` }] }));
}

// 流式正文：8 个 python 代码块夹在段落里，共 ~6KB。
const CODE = `def quicksort(items):\n    if len(items) <= 1:\n        return items\n    pivot = items[len(items) // 2]\n    left = [x for x in items if x < pivot]\n    middle = [x for x in items if x == pivot]\n    right = [x for x in items if x > pivot]\n    return quicksort(left) + quicksort(middle) + quicksort(right)\n\nprint(quicksort([3, 6, 1, 8, 2, 9, 4]))`;
const fullText = Array.from({ length: 8 }, (_, block) =>
  `第 ${block + 1} 节先解释这段代码的分区思路与递归边界，再看返回值如何逐层拼回有序数组。\n\n\`\`\`python\n${CODE}\n\`\`\`\n`).join("\n");
console.info(`stream bytes: ${fullText.length}`);

const TICK_MS = 25;
const DELTA_CHARS = 15; // ≈600 字/秒，快模型的稳态
const followers = new Set();
const send = (frame) => { for (const socket of followers) socket.send(JSON.stringify(frame)); };
const status = (active, preview) => send({ type: "reply.status", version: 2, session_id: sessionId,
  snapshot_id: `fixture-${Date.now()}-${Math.random()}`, available: true,
  items: active ? [{ session_id: sessionId, handle: "fixture", source: "akashic", active: true, preview }] : [] });
const append = (author, body) => {
  const item = message(items.length, author, body);
  items.push(item);
  send({ type: "messages.appended", version: 2, session_id: sessionId, after_seq: item.seq - 1,
    through_seq: item.seq, next_after_seq: item.seq, has_more: false, items: [item] });
};
function startStream() {
  let sent = 0;
  status(true, { message_id: "draft-1", text: "", thinking: "" });
  const timer = setInterval(() => {
    sent = Math.min(fullText.length, sent + DELTA_CHARS);
    if (sent < fullText.length) {
      status(true, { message_id: "draft-1", text: fullText.slice(0, sent), thinking: "" });
      return;
    }
    clearInterval(timer);
    append("assistant", { kind: "output", finish: "complete", parts: [{ kind: "text", value: fullText }] });
    status(false, null);
  }, TICK_MS);
}

const json = (res, data) => { res.writeHead(200, { "content-type": "application/json" }); res.end(JSON.stringify(data)); };
const server = createServer((req, res) => {
  const url = new URL(req.url, "http://localhost");
  const path = url.pathname;
  if (path === "/api/shell/state") return json(res, { status: "ready", configured: true, chatReady: true });
  if (path === "/api/chat/models") return json(res, { ...desktopModels(1), unavailableRuntimes: [] });
  if (path === "/api/chat/sessions") return json(res, { items: [{ key: sessionId, title: "流式基准",
    first_message_content: "流式基准", updated_at: items.at(-1).timestamp, message_count: items.length }], next_cursor: null });
  const history = path.match(/^\/api\/chat\/sessions\/([^/]+)\/messages$/u);
  if (history) return json(res, { version: 2, items, through_seq: items.length - 1, has_more: false, before_seq: null });
  if (path === "/api/chat/plugin-ui/catalog") return json(res, { catalog_revision: "0".repeat(64), items: [] });
  if (path === "/api/runtime/host-bridge") return json(res, { state: "healthy", failures: 0 });
  if (path.startsWith("/api/")) return json(res, { items: [], pins: [], sessions: [] });
  const file = resolve(root + "/dist", path === "/" ? "index.html" : path.replace(/^\/assets\//, "").replace(/^\//, ""));
  if (!file.startsWith(root + "/dist/") || !existsSync(file)) return res.writeHead(404).end();
  res.writeHead(200, { "content-type": ({ ".js": "text/javascript", ".css": "text/css", ".html": "text/html",
    ".woff2": "font/woff2", ".svg": "image/svg+xml" })[extname(file)] ?? "application/octet-stream" });
  res.end(readFileSync(file));
});
const ws = new WebSocketServer({ server, path: "/ws" });
ws.on("connection", (socket) => {
  socket.on("message", (raw) => {
    const frame = JSON.parse(String(raw));
    if (frame.type === "ping") return socket.send(JSON.stringify({ type: "pong", request_id: frame.request_id }));
    if (frame.type === "session.follow") {
      followers.add(socket);
      socket.send(JSON.stringify({ type: "session.following", version: 2, session_id: frame.session_id,
        through_seq: items.length - 1, request_id: frame.request_id }));
      return;
    }
    if (frame.type !== "message.send") return;
    sendReceivedAt = Date.now();
    socket.send(JSON.stringify({ type: "message.accepted", request_id: frame.request_id, session_id: frame.session_id }));
    append("user", { kind: "input", parts: [{ kind: "text", value: frame.text }] });
    startStream();
  });
  socket.on("close", () => followers.delete(socket));
});
await new Promise((done) => server.listen(0, "127.0.0.1", done));
const origin = "http://127.0.0.1:" + server.address().port;

// ---- 测量 ----
// 发送时刻以夹具收到 message.send 的 Date.now() 为准（与页面同机同时钟），
// 排除 Playwright fill/Enter 在 CPU 节流下的 CDP 调度开销。
let sendReceivedAt = 0;
const browser = await chromium.launch({ executablePath: values.browser, headless: true, args: ["--no-sandbox"] });
const results = [];
for (let run = 0; run < Number(values.runs); run += 1) {
  const context = await browser.newContext({ viewport: { width: 1365, height: 860 }, serviceWorkers: "block" });
  const page = await context.newPage();
  page.on("console", (msg) => { if (msg.type() === "error" || msg.type() === "warning") console.info("[page]", msg.type(), msg.text().slice(0, 200)); });
  page.on("pageerror", (error) => console.info("[pageerror]", String(error).slice(0, 300)));
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 20,
    downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 });
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
  await cdp.send("Profiler.enable");
  await page.addInitScript(() => {
    window.__probe = { longtasks: [], frames: [] };
    new PerformanceObserver((list) => {
      for (const e of list.getEntries()) window.__probe.longtasks.push({ t: e.startTime, d: e.duration });
    }).observe({ entryTypes: ["longtask"] });
    let last = performance.now();
    (function loop() {
      requestAnimationFrame(loop);
      const now = performance.now();
      window.__probe.frames.push([now, now - last]);
      last = now;
    })();
  });
  await page.goto(origin + "/", { waitUntil: "commit" });
  await page.getByText("流式基准", { exact: true }).click();
  await page.locator(".web-message-anchor").last().waitFor({ timeout: 30000 });
  // 等挂载后的空闲预热链（rIC timeout 3s + 分块下载求值）彻底结束，再发送：
  // 否则预热求值与出字窗口重叠，两组都测到预热成本而不是草稿延迟。
  await page.waitForTimeout(6000);
  await cdp.send("Profiler.start");
  await page.locator(".composer textarea").fill("开始");
  await page.keyboard.press("Enter");
  const epochOffset = await page.evaluate(() => Date.now() - performance.now());
  // 草稿行出现（thinking 占位，还不要求出字）
  await page.waitForFunction(() => Boolean(document.querySelector(".reply-activity")), null, { timeout: 30000 });
  const tActivity = await page.evaluate(() => performance.now());
  // 草稿开始出字
  await page.waitForFunction(() => {
    const draft = document.querySelector(".reply-activity[data-preview-message-id]");
    return draft && (draft.textContent?.length ?? 0) > 0;
  }, null, { timeout: 30000 });
  const tDraft = await page.evaluate(() => performance.now());
  // 落定：出现含完整正文的输出行，且草稿活动行已撤下
  try {
    await page.waitForFunction(() => {
      const output = [...document.querySelectorAll(".web-message-anchor.timeline-output")]
        .find((el) => el.textContent.includes("quicksort"));
      return Boolean(output) && !document.querySelector(".reply-activity[data-preview-message-id]");
    }, null, { timeout: 60000 });
  } catch (error) {
    const dump = await page.evaluate(() => ({
      anchors: [...document.querySelectorAll(".web-message-anchor")].map((el) => ({
        id: el.getAttribute("data-message-id"), cls: el.className, text: el.textContent.slice(0, 60) })).slice(-4),
      replyActivity: [...document.querySelectorAll(".reply-activity")].map((el) => ({
        cls: el.className, preview: el.getAttribute("data-preview-message-id"), text: el.textContent.slice(0, 60) })),
    }));
    console.info("DUMP", JSON.stringify(dump, null, 1));
    throw error;
  }
  const tSettled = await page.evaluate(() => performance.now());
  // 落定消息 8 个代码块全部进入高亮路径
  await page.waitForFunction(() => {
    const output = [...document.querySelectorAll(".web-message-anchor.timeline-output")]
      .find((el) => el.textContent.includes("quicksort"));
    return output?.querySelectorAll(".reading-code-scroll").length === 8;
  }, null, { timeout: 60000 });
  const tRich = await page.evaluate(() => performance.now());
  if (!sendReceivedAt) throw new Error("夹具未收到 message.send");
  const tSend = sendReceivedAt - epochOffset;
  const { profile } = await cdp.send("Profiler.stop");
  const probe = await page.evaluate(() => ({
    longtasks: window.__probe.longtasks,
    frames: window.__probe.frames.slice(-20000),
  }));
  const inWindow = probe.longtasks.filter((e) => e.t >= tSend && e.t <= tRich);
  // 只统计流式窗口（发送→落定富渲染）内的帧间隔；启动期的帧不算。
  const framesInWindow = probe.frames.filter(([t]) => t >= tSend && t <= tRich).map(([, gap]) => gap);
  const result = {
    activityMs: Math.round(tActivity - tSend),
    draftMs: Math.round(tDraft - tSend),
    settleRowsMs: Math.round(tSettled - tSend),
    settleRichMs: Math.round(tRich - tSend),
    streamMs: Math.round(tRich - tDraft),
    longTaskCount: inWindow.length,
    longTaskTotalMs: Math.round(inWindow.reduce((sum, e) => sum + e.d, 0)),
    longTaskWorstMs: Math.round(Math.max(0, ...inWindow.map((e) => e.d))),
    frameP95Ms: Math.round(framesInWindow.sort((a, b) => a - b)[Math.floor(framesInWindow.length * 0.95)] ?? 0),
    frameWorstMs: Math.round(Math.max(0, ...framesInWindow)),
  };
  results.push(result);
  if (values.output) {
    mkdirSync(values.output, { recursive: true });
    writeFileSync(`${values.output}/${values.label}-run${run}.profile.json`, JSON.stringify(profile));
  }
  console.log(`run${run}`, JSON.stringify(result));
  await context.close();
}
await browser.close();
server.close();
const median = (list) => list.sort((a, b) => a - b)[Math.floor(list.length / 2)];
const summary = Object.fromEntries(Object.keys(results[0]).map((key) => [key, median(results.map((r) => r[key]))]));
console.log("median", JSON.stringify(summary, null, 1));
if (values.output) writeFileSync(`${values.output}/${values.label}-summary.json`, JSON.stringify({ summary, results }, null, 1));
