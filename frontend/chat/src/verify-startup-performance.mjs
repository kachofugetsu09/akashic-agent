/** 测量 WebUI 冷启动、暖启动与会话切换的可见时延，不访问正式服务器。 */
import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { execFileSync, spawnSync } from "node:child_process";

import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "../../../scripts/webui-performance/desktop-fixture-server.mjs";

const repo = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const output = resolve(process.argv.find((value, index) => index > 1 && !value.startsWith("--")) ?? "artifacts/startup-performance");
const baseline = process.argv.includes("--baseline");
// --throttle 用接近移动网络的往返放大瀑布与字节开销，验证缓存与启动壳的真实价值。
const throttled = process.argv.includes("--throttle");
const runs = 3;
const root = mkdtempSync(resolve(tmpdir(), "akashic-startup-perf-"));
let browser;
let server;
mkdirSync(output, { recursive: true });

const percentile = (values, ratio) => {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.max(0, Math.ceil(sorted.length * ratio) - 1)] ?? 0;
};
const medianOf = (rows, key) => percentile(rows.map((row) => row[key]).filter((value) => Number.isFinite(value)), 0.5);

/** 页面加载的关键里程碑与网络瀑布：插入时间戳、资源字节与 API 串行深度。 */
async function measureLoad(context, url) {
  const page = await context.newPage();
  if (throttled) {
    const cdp = await context.newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.emulateNetworkConditions", {
      offline: false, latency: 150, downloadThroughput: 1.6 * 1024 * 1024 / 8, uploadThroughput: 750 * 1024 / 8,
    });
  }
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto(url, { waitUntil: "commit" });
  await page.waitForFunction(() => "sessionList" in (window.__startupMarks ?? {}), null, { timeout: 60_000 });
  await page.waitForFunction(() => "composer" in (window.__startupMarks ?? {}), null, { timeout: 60_000 });
  if (new URL(url).searchParams.get("session")) {
    await page.waitForFunction(() => "message" in (window.__startupMarks ?? {}), null, { timeout: 30_000 });
  }
  // 节流网络下 networkidle 可能永不满足；里程碑达标后固定等一拍收集资源计时。
  await page.waitForTimeout(500);
  const measured = await page.evaluate(() => ({
    marks: window.__startupMarks,
    paints: Object.fromEntries(performance.getEntriesByType("paint").map((entry) => [entry.name, entry.startTime])),
    navigation: performance.getEntriesByType("navigation")[0].toJSON(),
    resources: performance.getEntriesByType("resource").map((entry) => ({
      name: entry.name, startTime: entry.startTime, duration: entry.duration,
      transferSize: entry.transferSize, decodedBodySize: entry.decodedBodySize,
    })),
  }));
  await page.close();
  assert.deepEqual(errors, [], "页面加载不应抛出异常");
  const api = measured.resources
    .filter((entry) => new URL(entry.name).pathname.startsWith("/api/"))
    .toSorted((left, right) => left.startTime - right.startTime);
  let serialHops = 0;
  let previousEnd = -Infinity;
  for (const request of api) {
    if (request.startTime >= previousEnd) serialHops += 1;
    previousEnd = Math.max(previousEnd, request.startTime + request.duration);
  }
  const assets = measured.resources.filter((entry) => new URL(entry.name).pathname.startsWith("/assets/"));
  return {
    firstContentfulPaintMs: measured.paints["first-contentful-paint"] ?? null,
    domContentLoadedMs: measured.navigation.domContentLoadedEventEnd,
    shellMs: measured.marks.shell ?? null,
    sessionListMs: measured.marks.sessionList ?? null,
    composerMs: measured.marks.composer ?? null,
    messageMs: measured.marks.message ?? null,
    assetRequests: assets.length,
    assetCacheHits: assets.filter((entry) => entry.transferSize === 0 && entry.decodedBodySize > 0).length,
    assetTransferBytes: assets.reduce((sum, entry) => sum + entry.transferSize, 0),
    apiRequests: api.length,
    apiSerialHops: serialHops,
  };
}

/** 打点开关键里程碑：会话行、输入框与第一条消息的首个绘制帧。 */
async function instrumentContext(context) {
  await context.addInitScript(() => {
    const marks = {};
    const targets = {
      shell: ".boot-shell",
      sessionList: ".conversation-session",
      composer: ".composer__textarea",
      message: "[data-message-id]",
    };
    const check = () => {
      for (const [name, selector] of Object.entries(targets)) {
        if (name in marks || !document.querySelector(selector)) continue;
        requestAnimationFrame(() => requestAnimationFrame(() => {
          if (!(name in marks)) marks[name] = performance.now();
        }));
      }
    };
    new MutationObserver(check).observe(document, { subtree: true, childList: true });
    check();
    window.__startupMarks = marks;
  });
}

try {
  const build = spawnSync(process.execPath, [resolve(repo, "node_modules/vite/bin/vite.js"), "build",
    "--config", "frontend/chat/vite.config.ts", "--outDir", `${root}/dist`, "--emptyOutDir"], {
    cwd: repo, encoding: "utf8",
  });
  if (build.status !== 0) throw new Error(`vite 构建失败\n${build.stdout}\n${build.stderr}`);
  server = await startDesktopFixtureServer(`${root}/dist`, { historyCount: 100 });
  browser = await chromium.launch({ executablePath: process.env.AKASHIC_PERF_CHROMIUM ?? "/usr/bin/chromium", headless: true });
  const coldRows = [];
  const coldSessionRows = [];
  const warmRows = [];
  const revisitRows = [];

  for (let run = 1; run <= runs; run += 1) {
    // 1. 冷启动：全新 context，不带会话参数。
    const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await instrumentContext(context);
    coldRows.push(await measureLoad(context, `${server.origin}/`));
    // 2. 暖启动：同一 context 二次导航，命中 HTTP 缓存的部分应不再传输字节。
    warmRows.push(await measureLoad(context, `${server.origin}/`));
    // 3. 会话直达：冷启动直接携带 ?session= 参数。
    const sessionContext = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
    await instrumentContext(sessionContext);
    coldSessionRows.push(await measureLoad(sessionContext, `${server.origin}/?session=perf-session`));
    await sessionContext.close();

    // 4. 会话切换：A 首访 → 悬停预取 B → 回 A 回访。悬停预取与回访都应命中尾页缓存。
    const page = await context.newPage();
    const messageRequests = [];
    page.on("request", (request) => {
      const path = new URL(request.url()).pathname;
      if (/^\/api\/chat\/sessions\/[^/]+\/messages$/.test(path)) messageRequests.push(path);
    });
    await page.goto(`${server.origin}/`, { waitUntil: "commit" });
    await page.getByText("纯文本性能会话", { exact: true }).click({ timeout: 60_000 });
    await page.locator('[data-message-id="desktop-plain-99"]').waitFor();
    await page.getByText("性能基线会话", { exact: true }).hover();
    await page.waitForTimeout(400);
    messageRequests.length = 0;
    const prefetchStart = await page.evaluate(() => performance.now());
    await page.getByText("性能基线会话", { exact: true }).click();
    await page.locator('[data-message-id="desktop-rich-99"]').waitFor();
    revisitRows.push({
      kind: "prefetched",
      visibleMs: await page.evaluate((start) => performance.now() - start, prefetchStart),
      messageRequests: messageRequests.length,
    });
    messageRequests.length = 0;
    const revisitStart = await page.evaluate(() => performance.now());
    await page.getByText("纯文本性能会话", { exact: true }).click();
    await page.locator('[data-message-id="desktop-plain-99"]').waitFor();
    revisitRows.push({
      kind: "revisit",
      visibleMs: await page.evaluate((start) => performance.now() - start, revisitStart),
      messageRequests: messageRequests.length,
    });
    await context.close();
  }

  const report = {
    sourceCommit: execFileSync("git", ["rev-parse", "HEAD"], { cwd: repo, encoding: "utf8" }).trim(),
    chromium: await browser.version(),
    runs,
    cold: coldRows,
    warm: warmRows,
    coldSession: coldSessionRows,
    revisit: revisitRows,
  };
  writeFileSync(`${output}/results.json`, `${JSON.stringify(report, null, 2)}\n`);
  const summary = {
    cold: { fcp: medianOf(coldRows, "firstContentfulPaintMs"), shell: medianOf(coldRows, "shellMs"), sessionList: medianOf(coldRows, "sessionListMs"),
      composer: medianOf(coldRows, "composerMs"), assetBytes: medianOf(coldRows, "assetTransferBytes"), apiHops: medianOf(coldRows, "apiSerialHops") },
    warm: { fcp: medianOf(warmRows, "firstContentfulPaintMs"), sessionList: medianOf(warmRows, "sessionListMs"),
      assetBytes: medianOf(warmRows, "assetTransferBytes"), assetCacheHits: medianOf(warmRows, "assetCacheHits") },
    coldSession: { message: medianOf(coldSessionRows, "messageMs"), apiHops: medianOf(coldSessionRows, "apiSerialHops") },
    prefetched: { visibleMs: medianOf(revisitRows.filter((row) => row.kind === "prefetched"), "visibleMs"),
      messageRequests: medianOf(revisitRows.filter((row) => row.kind === "prefetched"), "messageRequests") },
    revisit: { visibleMs: medianOf(revisitRows.filter((row) => row.kind === "revisit"), "visibleMs"),
      messageRequests: medianOf(revisitRows.filter((row) => row.kind === "revisit"), "messageRequests") },
  };
  console.log(JSON.stringify(summary, null, 2));
  if (!baseline) {
    // 暖启动时哈希资产必须命中缓存；悬停预取与回访激活不再重复拉取尾页。
    assert.ok(summary.warm.assetBytes === 0, `暖启动仍传输 ${summary.warm.assetBytes}B 哈希资产`);
    assert.ok(summary.warm.assetCacheHits >= 1, "暖启动没有命中任何资产缓存");
    assert.ok(summary.prefetched.messageRequests === 0, `预取命中后激活仍请求 ${summary.prefetched.messageRequests} 次历史`);
    assert.ok(summary.revisit.messageRequests === 0, `回访会话仍请求 ${summary.revisit.messageRequests} 次历史`);
    // 串行深度断言只在节流模式生效：localhost 的亚毫秒 RTT 无法区分真实依赖与调度抖动，
    // 且 chatReady 后的非关键请求（插件目录、项目列表）也会被计入为新一轮。
    if (throttled) {
      assert.ok(summary.cold.apiHops <= 1, `启动 API 串行深度仍有 ${summary.cold.apiHops} 层`);
    }
  }
} finally {
  await browser?.close();
  await server?.close();
  rmSync(root, { recursive: true, force: true });
}
