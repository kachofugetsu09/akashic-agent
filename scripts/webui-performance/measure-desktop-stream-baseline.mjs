import { writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { resolve } from "node:path";

import { chromium } from "playwright-core";

const cdpEndpoint = requiredEnvironment("AKASHIC_PLAYWRIGHT_CDP");
const origin = requiredEnvironment("AKASHIC_WEBUI_FIXTURE_ORIGIN");
const charactersPerSecond = numberEnvironment("AKASHIC_WEBUI_CHARACTERS_PER_SECOND", 100);
const chunkCharacters = numberEnvironment("AKASHIC_WEBUI_CHUNK_CHARACTERS", 1);
const loadAllHistory = process.env.AKASHIC_WEBUI_LOAD_ALL_HISTORY === "1";
const browser = await chromium.connectOverCDP(cdpEndpoint);
const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });

try {
  const page = await context.newPage();
  await installProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").last().waitFor();
  if (loadAllHistory) await loadHistory(page);
  const loadedHistoryRows = await page.locator(".web-message-anchor").count();
  await page.evaluate(() => window.__resetStreamBaseline());
  const startedAt = await page.evaluate(() => performance.now());
  const response = await fetch(
    `${origin}/__fixture/stream?mode=replay&characters_per_second=${charactersPerSecond}&chunk_characters=${chunkCharacters}`,
    { method: "POST" },
  );
  if (!response.ok) throw new Error(`desktop replay fixture failed: ${response.status} ${await response.text()}`);
  const replay = await response.json();
  await page.waitForFunction(() => {
    const message = document.querySelector(".web-message-anchor:last-child");
    return message !== null && !message.classList.contains("streaming");
  }, null, { timeout: 15_000 });
  const metric = await page.evaluate((start) => window.__readStreamBaseline(start), startedAt);
  const report = {
    schemaVersion: 1,
    capturedAt: new Date().toISOString(),
    chromiumVersion: await browser.version(),
    viewport: { width: 1440, height: 1000 },
    fixture: {
      charactersPerSecond,
      chunkCharacters,
      loadAllHistory,
      loadedHistoryRows,
      stageCount: replay.stageCount,
      callCount: replay.callCount,
      deltaCount: replay.deltaCount,
    },
    metric: {
      ...metric,
      messageRows: await page.locator(".web-message-anchor").count(),
      domElements: await page.locator("*").count(),
    },
  };
  const path = resolve(tmpdir(), `akashic-desktop-stream-${Date.now()}.json`);
  writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`);
  console.log(JSON.stringify({ report: path, ...report }));
} finally {
  await context.close();
}
process.exit(0);

async function loadHistory(page) {
  const button = page.getByRole("button", { name: "加载更早消息" });
  while (await button.isVisible().catch(() => false)) {
    const before = await page.locator(".web-message-anchor").count();
    await button.click();
    await page.waitForFunction(
      (count) => document.querySelectorAll(".web-message-anchor").length > count,
      before,
    );
  }
}

async function installProbe(page) {
  await page.addInitScript(() => {
    const state = { frameGaps: [], longTasks: [], previousFrame: 0 };
    new PerformanceObserver((list) => {
      state.longTasks.push(...list.getEntries().map((entry) => entry.duration));
    }).observe({ type: "longtask", buffered: true });
    const frame = (timestamp) => {
      if (state.previousFrame > 0) state.frameGaps.push(timestamp - state.previousFrame);
      state.previousFrame = timestamp;
      requestAnimationFrame(frame);
    };
    requestAnimationFrame(frame);
    window.__resetStreamBaseline = () => {
      state.frameGaps.length = 0;
      state.longTasks.length = 0;
      state.previousFrame = 0;
    };
    window.__readStreamBaseline = (startedAt) => ({
      durationMs: performance.now() - startedAt,
      frameCount: state.frameGaps.length,
      frameGapP50Ms: percentile(state.frameGaps, 0.50),
      frameGapP95Ms: percentile(state.frameGaps, 0.95),
      frameGapP99Ms: percentile(state.frameGaps, 0.99),
      frameGapMaxMs: Math.max(0, ...state.frameGaps),
      frameGapsOver33Ms: state.frameGaps.filter((value) => value > 33).length,
      frameGapsOver50Ms: state.frameGaps.filter((value) => value > 50).length,
      longTaskCount: state.longTasks.length,
      longTaskTotalMs: state.longTasks.reduce((sum, value) => sum + value, 0),
      longTaskMaxMs: Math.max(0, ...state.longTasks),
      jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
    });
    function percentile(values, ratio) {
      if (values.length === 0) return 0;
      const sorted = [...values].sort((left, right) => left - right);
      return sorted[Math.min(sorted.length - 1, Math.ceil(sorted.length * ratio) - 1)];
    }
  });
}

function requiredEnvironment(name) {
  const value = process.env[name];
  if (!value) throw new Error(`${name} is required`);
  return value;
}

function numberEnvironment(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  const value = Number(raw);
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be a positive number`);
  return value;
}
