import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";

import {
  aggregateBrowserRuns,
  compareBrowserMetrics,
  createBrowserBudgets,
} from "./browser-metrics.mjs";
import {
  desktopMessages,
  desktopModels,
  desktopSessions,
  fixtureSessionId,
  mobileSnapshot,
  mobileStreamPatch,
  mobileTerminalPatch,
} from "./fixtures.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const baselinePath = resolve(here, "baseline.json");
const updateBaseline = process.argv.includes("--update-baseline");
const runCount = integerArgument("--runs", 5);
const buildRoot = mkdtempSync(resolve(tmpdir(), "akashic-webui-browser-"));
const results = [];
let browser;

try {
  const desktopOutput = buildTarget("frontend/chat/vite.config.ts", resolve(buildRoot, "desktop"));
  const mobileOutput = buildTarget("frontend/chat/vite.mobile.config.ts", resolve(buildRoot, "mobile"));
  const desktopServer = await startFixtureServer(desktopOutput, { stripAssetsPrefix: true });
  const mobileServer = await startFixtureServer(mobileOutput, { stripAssetsPrefix: false });
  try {
    browser = await chromium.launch({ executablePath: chromiumExecutable(), headless: true });
    for (let run = 1; run <= runCount; run += 1) {
      results.push({
        run,
        scenarios: {
          desktopHistory: await measureDesktopHistory(browser, desktopServer.origin),
          desktopStream600: await measureDesktopStream(browser, desktopServer.origin),
          mobileHistory300: await measureMobileHistory(browser, mobileServer.origin),
          mobileStream600: await measureMobileStream(browser, mobileServer.origin),
        },
      });
      console.log(`完成浏览器性能采样 ${run}/${runCount}`);
    }
    const aggregate = aggregateBrowserRuns(results);
    const report = {
      schemaVersion: 1,
      sourceCommit: gitCommit(),
      capturedAt: new Date().toISOString(),
      chromiumVersion: await browser.version(),
      runCount,
      aggregate,
      runs: results,
    };
    const reportPath = writeReport(report);
    console.log(`浏览器性能报告: ${reportPath}`);
    if (updateBaseline) updateBrowserBaseline(report);
    else compareBrowserBaseline(report);
  } finally {
    await desktopServer.close();
    await mobileServer.close();
  }
} finally {
  await browser?.close();
  rmSync(buildRoot, { recursive: true, force: true });
}

async function measureDesktopHistory(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  let socket;
  await context.routeWebSocket("**/ws", (route) => { socket = route; });
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(origin, { waitUntil: "networkidle" });
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").nth(99).waitFor();
  await page.locator(".web-message-anchor .message-row").nth(99).waitFor();
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  await context.close();
  void socket;
  return metric;
}

async function measureDesktopStream(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  let socket;
  await context.routeWebSocket("**/ws", (route) => { socket = route; });
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(origin, { waitUntil: "networkidle" });
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").nth(99).waitFor();
  if (!socket) throw new Error("桌面 WebSocket 夹具没有建立");
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  socket.send(JSON.stringify({ type: "turn.started", session_id: fixtureSessionId, turn_id: "perf-turn", content: "" }));
  for (let index = 0; index < 600; index += 1) {
    socket.send(JSON.stringify({ type: "answer.delta", session_id: fixtureSessionId, turn_id: "perf-turn", delta: "片" }));
  }
  await page.waitForFunction(() => document.querySelector(".web-message-anchor:last-child")?.textContent?.includes("片".repeat(600)), null, { timeout: 20_000 });
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  await context.close();
  return metric;
}

async function measureMobileHistory(browserInstance, origin) {
  const context = await mobileContext(browserInstance);
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(`${origin}/mobile.html`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => Boolean(window.AkashicMobile));
  const snapshot = mobileSnapshot(300);
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.evaluate((value) => window.AkashicMobile.receiveSnapshot(value), snapshot);
  await page.locator('[data-message-id="mobile-299"]').waitFor();
  const metric = await readPerformanceProbe(page, startedAt, ".mobile-message-anchor");
  metric.virtualRows = await page.locator(".mobile-virtual-row").count();
  await context.close();
  return metric;
}

async function measureMobileStream(browserInstance, origin) {
  const context = await mobileContext(browserInstance);
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(`${origin}/mobile.html`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => Boolean(window.AkashicMobile));
  const snapshot = mobileSnapshot(300, { streaming: true });
  await page.evaluate((value) => window.AkashicMobile.receiveSnapshot(value), snapshot);
  await page.locator('[data-message-id="mobile-299"]').waitFor();
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  const patches = Array.from({ length: 600 }, (_, index) => mobileStreamPatch(snapshot, index, "片"));
  const terminal = mobileTerminalPatch(snapshot, "片".repeat(600));
  await page.evaluate(({ deltas, finalPatch }) => {
    for (const patch of deltas) window.AkashicMobile.receiveStreamPatch(patch);
    window.AkashicMobile.receiveStreamPatch(finalPatch);
  }, { deltas: patches, finalPatch: terminal });
  await page.waitForFunction(() => {
    const row = document.querySelector('[data-message-id="mobile-299"]');
    return row !== null && !row.classList.contains("streaming") && row.textContent?.includes("片".repeat(600));
  });
  const metric = await readPerformanceProbe(page, startedAt, ".mobile-message-anchor");
  metric.virtualRows = await page.locator(".mobile-virtual-row").count();
  await context.close();
  return metric;
}

async function mobileContext(browserInstance) {
  const context = await browserInstance.newContext({
    viewport: { width: 412, height: 915 },
    deviceScaleFactor: 2.625,
    isMobile: true,
    hasTouch: true,
  });
  await context.addInitScript(() => {
    window.AkashicNativeTransport = { postMessage() {} };
    window.AkashicNative = new Proxy({}, { get: () => () => {} });
  });
  return context;
}

async function installPerformanceProbe(page) {
  await page.addInitScript(() => {
    const state = { longTasks: [], shifts: [], frameGaps: [], previousFrame: 0 };
    new PerformanceObserver((list) => state.longTasks.push(...list.getEntries().map((entry) => entry.duration))).observe({ type: "longtask", buffered: true });
    new PerformanceObserver((list) => state.shifts.push(...list.getEntries().filter((entry) => !entry.hadRecentInput).map((entry) => entry.value))).observe({ type: "layout-shift", buffered: true });
    const frame = (timestamp) => {
      if (state.previousFrame > 0) state.frameGaps.push(timestamp - state.previousFrame);
      state.previousFrame = timestamp;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    window.__resetAkashicPerf = () => {
      state.longTasks.length = 0;
      state.shifts.length = 0;
      state.frameGaps.length = 0;
      state.previousFrame = 0;
    };
    window.__readAkashicPerf = (startedAt, selector) => ({
      durationMs: performance.now() - startedAt,
      longTaskCount: state.longTasks.length,
      longTaskTotalMs: state.longTasks.reduce((sum, value) => sum + value, 0),
      longTaskMaxMs: Math.max(0, ...state.longTasks),
      frameGapP75Ms: percentile(state.frameGaps, 0.75),
      frameGapMaxMs: Math.max(0, ...state.frameGaps),
      layoutShift: state.shifts.reduce((sum, value) => sum + value, 0),
      domRows: document.querySelectorAll(selector).length,
      jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
    });
    function percentile(values, ratio) {
      if (values.length === 0) return 0;
      const sorted = [...values].sort((left, right) => left - right);
      return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)];
    }
  });
}

async function readPerformanceProbe(page, startedAt, selector) {
  return page.evaluate(({ start, rowSelector }) => window.__readAkashicPerf(start, rowSelector), { start: startedAt, rowSelector: selector });
}

function updateBrowserBaseline(report) {
  const baseline = JSON.parse(readFileSync(baselinePath, "utf8"));
  baseline.browser = {
    status: "measured",
    sourceCommit: report.sourceCommit,
    chromiumVersion: report.chromiumVersion,
    sampling: { runs: report.runCount, aggregate: "median", tail: "p75" },
    reference: report.aggregate,
    budgets: createBrowserBudgets(report.aggregate),
  };
  writeFileSync(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`);
  console.log(`已更新浏览器性能基线: ${baselinePath}`);
}

function compareBrowserBaseline(report) {
  const baseline = JSON.parse(readFileSync(baselinePath, "utf8"));
  if (baseline.browser.status !== "measured") {
    console.log("浏览器基线尚未采样：本次只生成报告，不作为回归门禁。使用 baseline:webui-performance:browser 显式提升。 ");
    return;
  }
  const checks = compareBrowserMetrics(report.aggregate, baseline.browser.budgets);
  for (const check of checks) {
    console.log(`${check.passed ? "PASS" : "FAIL"} ${check.scenario}.${check.metric}.p75: ${check.actual} <= ${check.maximum}`);
  }
  if (checks.some((check) => !check.passed)) process.exitCode = 1;
}

function buildTarget(config, outputDirectory) {
  const vite = resolve(repoRoot, "node_modules/vite/bin/vite.js");
  const result = spawnSync(process.execPath, [vite, "build", "--config", config, "--outDir", outputDirectory, "--emptyOutDir"], {
    cwd: repoRoot,
    encoding: "utf8",
  });
  if (result.status !== 0) throw new Error(`${config} 构建失败\n${result.stdout}\n${result.stderr}`);
  return outputDirectory;
}

async function startFixtureServer(root, { stripAssetsPrefix }) {
  const server = createServer((request, response) => {
    const url = new URL(request.url, "http://127.0.0.1");
    const api = fixtureApiResponse(url.pathname);
    if (api !== undefined) return sendJson(response, api);
    let requested = url.pathname === "/" ? "index.html" : url.pathname.replace(/^\//u, "");
    if (stripAssetsPrefix) requested = requested.replace(/^assets\//u, "");
    const file = resolve(root, requested);
    if (!file.startsWith(`${root}${sep}`) || !existsSync(file) || !statSync(file).isFile()) {
      response.writeHead(404).end("not found");
      return;
    }
    response.writeHead(200, { "content-type": contentType(file), "cache-control": "no-store" });
    createReadStream(file).pipe(response);
  });
  await new Promise((resolveListen, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", resolveListen);
  });
  const address = server.address();
  return {
    origin: `http://127.0.0.1:${address.port}`,
    close: () => new Promise((resolveClose, reject) => server.close((error) => error ? reject(error) : resolveClose())),
  };
}

function fixtureApiResponse(pathname) {
  if (pathname === "/api/shell/state") return { status: "ready", configured: true, chatReady: true, settingsPath: "/settings" };
  if (pathname === "/api/chat/sessions") return desktopSessions();
  if (pathname === `/api/chat/sessions/${fixtureSessionId}/messages`) return desktopMessages();
  if (pathname === "/api/chat/models") return desktopModels();
  if (pathname === "/api/chat/plugin-ui/catalog") return { catalog_revision: "0".repeat(64), items: [] };
  return undefined;
}

function sendJson(response, payload) {
  response.writeHead(200, { "content-type": "application/json", "cache-control": "no-store" });
  response.end(JSON.stringify(payload));
}

function contentType(file) {
  return ({
    ".css": "text/css",
    ".html": "text/html",
    ".js": "text/javascript",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
  })[extname(file)] ?? "application/octet-stream";
}

function chromiumExecutable() {
  if (process.env.AKASHIC_PERF_CHROMIUM) return process.env.AKASHIC_PERF_CHROMIUM;
  const candidate = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"].find(existsSync);
  if (!candidate) throw new Error("未找到 Chromium；请设置 AKASHIC_PERF_CHROMIUM 指向受控浏览器可执行文件");
  return candidate;
}

function integerArgument(name, fallback) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = Number(process.argv[index + 1]);
  if (!Number.isSafeInteger(value) || value < 1) throw new Error(`${name} 必须是正整数`);
  return value;
}

function writeReport(report) {
  const outputDirectory = resolve(repoRoot, "artifacts", "webui-performance");
  mkdirSync(outputDirectory, { recursive: true });
  const stamp = report.capturedAt.replaceAll(":", "-");
  const outputPath = resolve(outputDirectory, `browser-${stamp}.json`);
  writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`);
  return outputPath;
}

function gitCommit() {
  const result = spawnSync("git", ["rev-parse", "HEAD"], { cwd: repoRoot, encoding: "utf8" });
  if (result.status !== 0) throw new Error("无法读取 Git commit");
  return result.stdout.trim();
}
