/** 在真实首页、同文档挂载的聊天、历史和 WebSocket 边界测量冷/热启动。 */
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({ options: {
  url: { type: "string" }, output: { type: "string" }, browser: { type: "string" },
  runs: { type: "string", default: "7" }, profiles: { type: "string", default: "lan,latency,limited" },
  label: { type: "string", default: "baseline" },
} });
if (!values.url || !values.output || !values.browser) throw new Error("需要 --url、--output 和 --browser");
const profiles = {
  lan: { latency: 20, downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 },
  latency: { latency: 600, downloadThroughput: 12_500_000, uploadThroughput: 1_250_000 },
  limited: { latency: 300, downloadThroughput: 500_000, uploadThroughput: 125_000 },
};
const selected = values.profiles.split(",");
const runCount = Number(values.runs);
if (!Number.isSafeInteger(runCount) || runCount < 1 || selected.some(name => !profiles[name])) {
  throw new Error("无效的 runs 或 profiles");
}
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const hash = value => createHash("sha256").update(value).digest("hex");
const git = (...args) => execFileSync("git", args, { encoding: "utf8" }).trim();
const bootstrap = await fetch(new URL("/api/chat/web-ui/bootstrap", values.url)).then(r => {
  if (!r.ok) throw new Error(`bootstrap ${r.status}`);
  return r.text();
});
const html = await fetch(values.url).then(r => {
  if (!r.ok) throw new Error(`首页 ${r.status}`);
  return r.text();
});
const entry = html.match(/<script[^>]*src="([^"]+)"/u)?.[1];
if (!entry) throw new Error("首页缺少生产入口脚本");
const entryBytes = await fetch(new URL(entry, values.url)).then(r => r.arrayBuffer());
// 聊天挂载模块用稳定 URL 发布（ADR 0126），内容随版本变化，按字节摘要区分构建。
const chatEntry = "/assets/chat-app.js";
const chatEntryBytes = await fetch(new URL(chatEntry, values.url)).then(r => {
  if (!r.ok) throw new Error(`聊天挂载模块 ${r.status}`);
  return r.arrayBuffer();
});
const report = {
  schemaVersion: 1, label: values.label, sourceCommit: git("rev-parse", "HEAD"),
  sourceDiffSha256: hash(git("diff", "HEAD", "--", "frontend")),
  harnessSha256: hash(readFileSync(new URL(import.meta.url))),
  lockSha256: hash(readFileSync("package-lock.json")), bootstrapSha256: hash(bootstrap),
  entry, entrySha256: hash(Buffer.from(entryBytes)),
  chatEntry, chatEntrySha256: hash(Buffer.from(chatEntryBytes)),
  moduleIds: JSON.parse(bootstrap).modules.map(m => m.pluginId),
  profiles, runCount, plannedSamples: selected.length * runCount * 2,
  viewport: { width: 412, height: 915 }, cpuRate: 4,
  timeoutMs: 45_000, startedAt: new Date().toISOString(), runs: [],
};
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
report.browserVersion = browser.version();
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
try {
  // 1. 每轮新 context；同一 context 第二次访问才算热缓存，保留真实浏览器 HTTP 缓存。
  for (let run = 0; run < runCount; run++) {
    for (let offset = 0; offset < selected.length; offset++) {
      const profile = selected[(run + offset) % selected.length];
      const context = await browser.newContext({ viewport: report.viewport, serviceWorkers: "block" });
      await context.addInitScript(() => {
        localStorage.setItem("akashic.onboarding.seen", "1");
        window.__startupSockets = new Map();
        const NativeSocket = window.WebSocket;
        window.WebSocket = class extends NativeSocket {
          constructor(...args) {
            super(...args);
            if (new URL(this.url).pathname !== "/ws") return;
            window.__startupSockets.set(this, false);
            this.addEventListener("message", event => {
              if (JSON.parse(event.data).type === "reply.status") window.__startupSockets.set(this, true);
            });
            this.addEventListener("close", () => window.__startupSockets.delete(this));
          }
        };
        window.__startupLongTasks = [];
        new PerformanceObserver(list => window.__startupLongTasks.push(...list.getEntries().map(
          entry => ({ start: entry.startTime, duration: entry.duration }),
        ))).observe({ type: "longtask", buffered: true });
      });
      try {
        for (const cache of ["cold", "warm"]) {
          let result;
          try { result = await measure(context, profile, cache); }
          catch (error) {
            // 浏览器或取证失败也是本次样本失败；保留记录后继续下一页。
            result = { success: false, readyMs: null, error: String(error) };
          }
          report.runs.push({ run, profile, cache, ...result });
          writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
          console.log(JSON.stringify({ run, profile, cache, readyMs: result.readyMs,
            fcpMs: result.fcpMs, success: result.success, error: result.error }));
        }
      } finally { await context.close(); }
    }
  }
} finally { await browser.close(); }

// 2. 失败单列且进入成功率分母；小样本报告中位数和最大值，不声称 p95。
report.summary = Object.fromEntries(selected.flatMap(profile => ["cold", "warm"].map(cache => {
  const rows = report.runs.filter(r => r.profile === profile && r.cache === cache);
  const metric = name => {
    const nums = rows.map(r => r[name]).filter(v => typeof v === "number").sort((a, b) => a - b);
    return nums.length ? { median: nums[Math.floor(nums.length / 2)], max: nums.at(-1) } : null;
  };
  return [`${profile}/${cache}`, { n: rows.length, successes: rows.filter(r => r.success).length,
    readyMs: metric("readyMs"), fcpMs: metric("fcpMs"), transferBytes: metric("transferBytes") }];
})));
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report.summary, null, 2));
if (report.runs.some(r => !r.success)) process.exitCode = 1;

async function measure(context, profile, cache) {
  /** 只有历史正文、可编辑输入框和真实 WebSocket 数据均就绪才结束计时。 */
  const page = await context.newPage();
  try { return await measurePage(page, context, profile, cache); }
  finally { await page.close(); }
}

async function measurePage(page, context, profile, cache) {
  /** 测量与取证共用一页，页面关闭由调用方保证。 */
  page.setDefaultTimeout(report.timeoutMs);
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  await cdp.send("Network.emulateNetworkConditions", { offline: false, ...profiles[profile] });
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: report.cpuRate });
  const errors = [], failures = [], sockets = [];
  const requests = {}, sentFrames = {}, receivedFrames = {};
  let liveSockets = 0, peakSockets = 0, receivedBytes = 0;
  page.on("request", request => {
    const url = new URL(request.url());
    const key = `${request.method()} ${url.pathname}${url.search}`;
    requests[key] = (requests[key] ?? 0) + 1;
  });
  let confirmFollow;
  const following = new Promise(resolveFollow => { confirmFollow = resolveFollow; });
  let websocketDataAt = null;
  page.on("pageerror", error => errors.push(error.message));
  page.on("requestfailed", request => failures.push({ url: request.url(), error: request.failure()?.errorText }));
  page.on("websocket", socket => {
    if (new URL(socket.url()).pathname !== "/ws") return;
    sockets.push(socket);
    liveSockets += 1;
    peakSockets = Math.max(peakSockets, liveSockets);
    socket.on("close", () => { liveSockets -= 1; });
    socket.on("framesent", ({ payload }) => {
      const frame = JSON.parse(String(payload));
      sentFrames[frame.type] = (sentFrames[frame.type] ?? 0) + 1;
    });
    socket.on("framereceived", ({ payload }) => {
      const frame = JSON.parse(String(payload));
      receivedBytes += Buffer.byteLength(payload);
      receivedFrames[frame.type] = (receivedFrames[frame.type] ?? 0) + 1;
      if (frame.type === "error") errors.push(`WebSocket: ${frame.message}`);
      if (frame.type === "session.following" && frame.session_id === "akashic:startup-benchmark") {
        websocketDataAt ??= performance.now();
        confirmFollow();
      }
    });
  });
  const started = performance.now();
  const remaining = () => Math.max(1, report.timeoutMs - (performance.now() - started));
  let readyMs = null, error = null;
  try {
    // query 让 conversation-ui 按正式入口恢复同一个 Session。
    await page.goto(new URL("/?session=akashic%3Astartup-benchmark", values.url).href, { waitUntil: "commit", timeout: remaining() });
    const frame = page.locator(".conversation-page-chat .akashic-chat");
    await frame.locator('[data-message-id="startup-099"] .message-row').waitFor({ timeout: remaining() });
    const editor = frame.locator('textarea[name="message"]');
    await editor.fill("启动基准草稿", { timeout: remaining() });
    if (await editor.inputValue({ timeout: remaining() }) !== "启动基准草稿") throw new Error("输入框未保留草稿");
    await editor.fill("", { timeout: remaining() });
    let timer;
    try {
      await Promise.race([following, new Promise((_resolve, reject) => {
        timer = setTimeout(() => reject(new Error("未收到当前会话的 session.following")), remaining());
      })]);
    } finally { clearTimeout(timer); }
    readyMs = performance.now() - started;
    if (errors.length) throw new Error(errors.join("; "));
  } catch (reason) {
    error = String(reason);
  }
  const frames = [];
  for (const frame of page.frames()) {
    frames.push(await frame.evaluate(() => ({
      url: location.href, timeOrigin: performance.timeOrigin,
      paint: performance.getEntriesByType("paint").map(e => ({ name: e.name, start: e.startTime })),
      resources: performance.getEntriesByType("resource").map(e => ({
        name: e.name, start: e.startTime, duration: e.duration, bytes: e.transferSize,
      })), longTasks: window.__startupLongTasks,
    })));
  }
  const paints = frames.flatMap(f => f.paint.filter(p => p.name === "first-contentful-paint")
    .map(p => f.timeOrigin + p.start - frames[0].timeOrigin));
  const fcpMs = paints.length ? Math.min(...paints) : null;
  const transferBytes = frames.reduce((sum, f) => sum + f.resources.reduce((n, r) => n + r.bytes, 0), 0);
  const socketsAtReady = sockets.length;
  // 热缓存从一次完整字体加载后的访问开始；截图不计入可用耗时。
  for (const frame of page.frames()) await frame.waitForFunction(() => document.fonts.status === "loaded");
  let health = null;
  try {
    // 聊天与外壳同一文档，探测在主 frame 里进行。
    health = await page.mainFrame().evaluate(async () => {
      const started = performance.now(), deadline = started + 10000;
      let probes = 0;
      // 在浏览器同一事件循环内选择和探测，避免 Node 检查后连接恰好被界面重新创建。
      while (performance.now() < deadline) {
        const socket = [...window.__startupSockets].find(([item, reply]) => reply && item.readyState === WebSocket.OPEN)?.[0];
        if (window.__startupSockets.size !== 1 || !socket) {
          await new Promise(done => setTimeout(done, 25));
          continue;
        }
        probes += 1;
        const alive = await new Promise(done => {
          const timer = setTimeout(() => finish(false), Math.min(5000, deadline - performance.now()));
          function finish(value) {
            clearTimeout(timer);
            socket.removeEventListener("message", receive);
            socket.removeEventListener("close", closed);
            done(value);
          }
          function receive(event) {
            const frame = JSON.parse(event.data);
            if (frame.type === "pong" && frame.request_id === "startup-health") finish(true);
          }
          function closed() { finish(false); }
          socket.addEventListener("message", receive);
          socket.addEventListener("close", closed);
          socket.send(JSON.stringify({ type: "ping", request_id: "startup-health" }));
        });
        if (alive) return { probes, ms: performance.now() - started };
      }
      throw new Error("稳定阶段未恢复单条连接、回复状态与 ping");
    });
    if (errors.length) throw new Error(errors.join("; "));
  } catch (reason) { error = String(reason); }
  const suffix = error === null ? "" : "-failed";
  await page.screenshot({ path: resolve(output, `${profile}-${cache}-${report.runs.length}${suffix}.png`) });
  const socketsBeforeClose = liveSockets;
  return { success: error === null, readyMs, fcpMs, transferBytes, error, errors, failures,
    websocketDataMs: websocketDataAt === null ? null : websocketDataAt - started,
    sockets: sockets.length, socketsAtReady, peakSockets, socketsBeforeClose,
    requests, sentFrames, receivedFrames, receivedBytes, health, frames };
}
