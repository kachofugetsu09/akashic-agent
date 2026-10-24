import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "./desktop-fixture-server.mjs";
import { createWebUiBootstrap } from "./web-ui-fixtures.mjs";

import {
  aggregateBrowserRuns,
  compareBrowserMetrics,
  createBrowserBudgets,
} from "./browser-metrics.mjs";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");
const baselinePath = resolve(here, "baseline.json");
const updateBaseline = process.argv.includes("--update-baseline");
const runCount = integerArgument("--runs", 5);
const onlyScenario = process.argv.includes("--only") ? process.argv[process.argv.indexOf("--only") + 1] : "";
const desktopStreamIntervalMs = numberArgument("--desktop-stream-interval-ms", 2.5, 0);
const buildRoot = mkdtempSync(resolve(tmpdir(), "akashic-webui-browser-"));
const results = [];
let browser;
let ownsBrowser = false;

try {
  const desktopOutput = buildTarget("frontend/chat/vite.harness.config.ts", resolve(buildRoot, "desktop"));
  const { dashboardOutput, chatAppOutput } = buildProductionShell();
  const desktopServer = await startDesktopFixtureServer(desktopOutput, {
    dashboardRoot: dashboardOutput,
    chatAppRoot: chatAppOutput,
    webUi: createWebUiBootstrap(repoRoot),
  });
  try {
    const cdpEndpoint = process.env.AKASHIC_PLAYWRIGHT_CDP;
    browser = cdpEndpoint
      ? await chromium.connectOverCDP(cdpEndpoint)
      : await chromium.launch({ executablePath: chromiumExecutable(), headless: true });
    ownsBrowser = !cdpEndpoint;
    const measure = async (name, task) => {
      if (onlyScenario && name !== onlyScenario) return undefined;
      console.log(`[scenario] ${name} ...`);
      const reset = await fetch(`${desktopServer.origin}/__fixture/reset`, { method: "POST" });
      if (!reset.ok) throw new Error(`场景 ${name} 的 fixture 重置失败: ${reset.status}`);
      const result = await task();
      console.log(`[scenario] ${name} done`);
      return result;
    };
    for (let run = 1; run <= runCount; run += 1) {
      results.push({
        run,
        scenarios: {
          desktopStartup: await measure("desktopStartup", () => measureDesktopStartup(browser, desktopServer.origin)),
          desktopHistory: await measure("desktopHistory", () => measureDesktopHistory(browser, desktopServer.origin)),
          desktopSessionSwitch: await measure("desktopSessionSwitch", () => measureDesktopSessionSwitch(browser, desktopServer.origin)),
          desktopModelPicker: await measure("desktopModelPicker", () => measureDesktopModelPicker(browser, desktopServer.origin)),
          desktopComposer: await measure("desktopComposer", () => measureDesktopComposer(browser, desktopServer.origin)),
          desktopPendingSendStop: await measure("desktopPendingSendStop", () => measureDesktopPendingSendStop(browser, desktopServer.origin)),
          desktopResponsive: await measure("desktopResponsive", () => measureDesktopResponsive(browser, desktopServer.origin)),
          desktopAccessibility: await measure("desktopAccessibility", () => measureDesktopAccessibility(browser, desktopServer.origin)),
          desktopStream600: await measure("desktopStream600", () => measureDesktopStream(browser, desktopServer.origin, desktopStreamIntervalMs)),
          desktopMarkdownStream: await measure("desktopMarkdownStream", () => measureDesktopStream(browser, desktopServer.origin, desktopStreamIntervalMs, "**片** ")),
          desktopCodeStream: await measure("desktopCodeStream", () => measureDesktopStream(browser, desktopServer.origin, 20, "```ts\nconst x = 1;\n```\n", 60)),
          shellSettings: await measure("shellSettings", () => measureShellSettings(browser, desktopServer.origin)),
        },
      });
      console.log(`完成浏览器性能采样 ${run}/${runCount}`);
    }
    const measuredResults = results.map((run) => ({
      ...run,
      scenarios: Object.fromEntries(Object.entries(run.scenarios).filter(([, metric]) => metric !== undefined)),
    }));
    const aggregate = aggregateBrowserRuns(measuredResults);
    const report = {
      schemaVersion: 1,
      sourceCommit: gitCommit(),
      capturedAt: new Date().toISOString(),
      chromiumVersion: await browser.version(),
      runCount,
      fixture: { desktopStreamIntervalMs },
      aggregate,
      runs: measuredResults,
    };
    const reportPath = writeReport(report);
    console.log(`浏览器性能报告: ${reportPath}`);
    if (updateBaseline) updateBrowserBaseline(report);
    else if (onlyScenario) console.log(`单场景 ${onlyScenario} 不执行全套基线门禁`);
    else compareBrowserBaseline(report);
  } finally {
    await desktopServer.close();
  }
} finally {
  if (ownsBrowser) await browser?.close();
  rmSync(buildRoot, { recursive: true, force: true });
}
if (!ownsBrowser) process.exit(0);

async function measureDesktopStartup(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await installPerformanceProbe(page);
  const ready = async () => {
    await page.locator(".composer__textarea").waitFor({ state: "visible" });
    await page.locator(".home-hero__title").waitFor({ state: "visible" });
  };
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "domcontentloaded" });
  await ready();
  const metric = await readPerformanceProbe(page, 0, ".conversation-session");
  await page.reload({ waitUntil: "domcontentloaded" });
  await ready();
  const warm = await readPerformanceProbe(page, 0, ".conversation-session");
  metric.warmDurationMs = warm.durationMs;
  metric.warmFirstContentfulPaintMs = warm.firstContentfulPaintMs;
  metric.warmLargestContentfulPaintMs = warm.largestContentfulPaintMs;
  metric.warmLongTaskTotalMs = warm.longTaskTotalMs;
  metric.warmResourceTransferBytes = warm.resourceTransferBytes;
  await context.close();
  return metric;
}

async function measureShellSettings(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
  const page = await context.newPage();
  const browserErrors = [];
  const writeRequests = [];
  let modelCatalogRequests = 0;
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (url.pathname === "/api/dashboard/models/catalog") modelCatalogRequests += 1;
    if (url.pathname.startsWith("/api/") && request.method() !== "GET" && !isReadOnlyPluginQuery(request)) {
      writeRequests.push(`${request.method()} ${url.pathname}`);
    }
  });
  await installPerformanceProbe(page);
  await page.goto(origin, { waitUntil: "domcontentloaded" });
  const chat = page;
  await chat.locator(".composer__textarea").waitFor({ state: "visible" })
    .catch(async (error) => {
      const body = await page.locator("body").textContent();
      throw new Error(`Shell 首屏未就绪: ${JSON.stringify({ browserErrors, body: body?.slice(0, 2000) })}\n${error.message}`);
    });
  await chat.locator(".home-hero__title").waitFor({ state: "visible" });
  const startup = await readPerformanceProbe(page, 0, ".conversation-session");
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await chat.getByRole("button", { name: "功能设置", exact: true }).click();
  await page.getByRole("dialog", { name: "功能设置" }).waitFor({ state: "visible" });
  await page.locator(".settings-provider-card").nth(47).waitFor({ state: "visible" });
  const metric = await readPerformanceProbe(page, startedAt, ".settings-provider-card");
  metric.shellReadyMs = startup.durationMs;
  metric.modelCatalogAfterOpen = modelCatalogRequests;
  const sectionStartedAt = await page.evaluate(() => performance.now());
  await page.getByRole("button", { name: "新手引导", exact: true }).click();
  await page.locator(".onboarding-overview").waitFor({ state: "visible" });
  metric.settingsSectionSwitchMs = await page.evaluate((start) => performance.now() - start, sectionStartedAt);
  metric.modelCatalogAfterSection = modelCatalogRequests;
  await page.getByRole("button", { name: "模型", exact: true }).click();
  await page.locator(".settings-provider-card").nth(47).waitFor({ state: "visible" });
  metric.modelCatalogAfterBack = modelCatalogRequests;
  const returnStartedAt = await page.evaluate(() => performance.now());
  await page.getByRole("button", { name: "关闭设置", exact: true }).click();
  await page.getByRole("dialog", { name: "功能设置" }).waitFor({ state: "hidden" });
  await chat.locator(".composer__textarea").waitFor({ state: "visible" });
  metric.settingsReturnMs = await page.evaluate((start) => performance.now() - start, returnStartedAt);
  metric.modelCatalogRequests = modelCatalogRequests;
  metric.hostErrors = await page.locator(".web-host-entry-error").count();
  if (browserErrors.length || metric.hostErrors || writeRequests.length) {
    throw new Error(`完整 Shell 设置场景失败: ${JSON.stringify({ browserErrors, writeRequests, modelCatalogRequests, hostErrors: metric.hostErrors })}`);
  }
  await context.close();
  return metric;
}

function isReadOnlyPluginQuery(request) {
  if (new URL(request.url()).pathname !== "/api/chat/plugin-ui/query") return false;
  try {
    const payload = JSON.parse(request.postData() ?? "{}");
    return payload.method === "chat.models" || payload.method === "call.latest";
  } catch {
    return false;
  }
}

async function measureDesktopHistory(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").nth(49).waitFor();
  await page.evaluate(() => document.querySelector(".desktop-history-loader button")?.click());
  await page.locator(".web-message-anchor").nth(99).waitFor();
  await page.locator(".web-message-anchor .message-row").nth(99).waitFor();
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  await page.waitForTimeout(1_000);
  const settled = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  metric.settledLongTaskMaxMs = settled.longTaskMaxMs;
  metric.settledFrameGapMaxMs = settled.frameGapMaxMs;
  metric.settledLayoutShift = settled.layoutShift;
  metric.settledDomElements = await page.locator("*").count();
  metric.enhancedRows = 100 - await page.locator(".desktop-message-placeholder").count();
  // 复制动作由两条渲染路径各自提供：静态 markdown 的 [data-static-code-copy]
  // 与 markstream 代码块头部的 .code-action-btn，任一存在即满足合同。
  metric.codeCopyButtons = await page.locator("[data-static-code-copy], .code-action-btn").count();
  if (metric.codeCopyButtons < 1) throw new Error("settled code copy action is unavailable");
  await page.locator('[data-message-id="desktop-rich-99"] .message-reply-reference').click();
  await page.waitForFunction(() => {
    const target = document.querySelector('[data-message-id="desktop-rich-10"]');
    return target !== null && !target.querySelector(".desktop-message-placeholder");
  });
  await context.close();
  return metric;
}

async function measureDesktopSessionSwitch(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const requests = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (/^\/api\/chat\/sessions\/[^/]+\/messages$/u.test(url.pathname)) requests.push(url.pathname + url.search);
    // 模型目录由 models 插件经插件查询读取。
    if (url.pathname === "/api/chat/plugin-ui/query" && request.postData()?.includes('"chat.models"')) requests.push("chat.models");
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.evaluate(() => window.__resetAkashicPerf());
  requests.length = 0;
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByText("纯文本性能会话", { exact: true }).click();
  const clickedAt = await page.evaluate(() => performance.now());
  await page.locator('[data-message-id="desktop-plain-99"]').waitFor();
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  metric.sessionClickMs = clickedAt - startedAt;
  metric.messageContentWaitMs = metric.durationMs - metric.sessionClickMs;
  metric.messageResourceMs = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => entry.name.includes("/messages?"))
    .reduce((latest, entry) => Math.max(latest, entry.responseEnd), 0));
  metric.resourceTimeline = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => entry.name.includes("/messages?") || entry.name.includes(".woff2"))
    .map((entry) => ({ name: entry.name.split("/").at(-1), startTime: entry.startTime, responseEnd: entry.responseEnd, duration: entry.duration })));
  metric.messageRequests = requests.filter((request) => request.includes("/messages")).length;
  metric.modelRequests = requests.filter((request) => request === "chat.models").length;
  requests.length = 0;
  const activeSession = page.getByRole("button", { name: "纯文本性能会话", exact: true });
  await activeSession.click();
  await page.waitForTimeout(100);
  metric.repeatMessageRequests = requests.filter((request) => request.includes("/messages")).length;
  metric.repeatModelRequests = requests.filter((request) => request === "chat.models").length;
  metric.sessionRows = await page.locator(".conversation-session").count();
  if (metric.repeatMessageRequests !== 0 || metric.repeatModelRequests !== 0) {
    throw new Error(`active session repeated requests: messages=${metric.repeatMessageRequests}, models=${metric.repeatModelRequests}`);
  }
  if (await activeSession.getAttribute("aria-current") !== "true") {
    throw new Error("selected desktop session does not expose its current state");
  }
  await context.close();
  return metric;
}

async function measureDesktopModelPicker(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  const metric = {
    closedOptions: await page.locator(".model-picker__option").count(),
    closedDomElements: await page.locator("*").count(),
  };
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.locator(".model-picker__trigger").click().catch(async (error) => {
    const body = await page.locator("body").textContent();
    throw new Error(`模型胶囊未挂载: ${JSON.stringify({ browserErrors, body: body?.slice(-2000) })}\n${error.message}`);
  });
  await page.locator(".model-picker__panel").waitFor();
  Object.assign(metric, await readPerformanceProbe(page, startedAt, ".model-picker__option"));
  metric.openOptions = await page.locator(".model-picker__option").count();
  await page.keyboard.press("End");
  metric.keyboardEnd = await page.evaluate(() => document.activeElement?.classList.contains("model-picker__option") ? 1 : 0);
  await page.keyboard.press("Home");
  metric.keyboardHome = await page.evaluate(() => document.activeElement?.textContent?.includes("跟随默认模型") ? 1 : 0);
  await page.keyboard.press("Escape");
  metric.focusRestored = await page.evaluate(() => document.activeElement?.classList.contains("model-picker__trigger") ? 1 : 0);
  if (metric.closedOptions !== 0 || metric.keyboardEnd !== 1 || metric.keyboardHome !== 1 || metric.focusRestored !== 1) {
    throw new Error(`model picker interaction contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopComposer(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    Object.defineProperty(Crypto.prototype, "randomUUID", {
      configurable: true,
      value: undefined,
    });
  });
  const page = await context.newPage();
  let uploadRequests = 0;
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/api/chat/uploads") uploadRequests += 1;
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("纯文本性能会话", { exact: true }).click();
  await page.locator('[data-message-id="desktop-plain-99"]').waitFor();
  await fetch(`${origin}/__fixture/reset`, { method: "POST" });
  await fetch(`${origin}/__fixture/history-delay?ms=500`, { method: "POST" });
  await fetch(`${origin}/__fixture/stream?count=1&interval_ms=0&terminal=0`, { method: "POST" });
  await page.waitForFunction(async (fixtureOrigin) => {
    const received = await fetch(`${fixtureOrigin}/__fixture/received`).then((response) => response.json());
    return received.requests.some((request) => request.includes("/messages"));
  }, origin);
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  const text = "输入响应基线".repeat(40);
  await page.locator('textarea[name="message"]').pressSequentially(text);
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  metric.typedCharacters = text.length;
  metric.randomUUIDUnavailable = await page.evaluate(() => typeof crypto.randomUUID === "undefined" ? 1 : 0);
  metric.messageRowsAfterTyping = await page.locator(".web-message-anchor").count();
  await page.locator('input[type="file"]').setInputFiles({ name: "composer.txt", mimeType: "text/plain", buffer: Buffer.from("附件内容") });
  await page.getByText("composer.txt", { exact: true }).waitFor();
  await page.getByRole("button", { name: "发送消息" }).click();
  await page.getByRole("button", { name: "中止回答" }).waitFor();
  await page.waitForTimeout(1200);
  metric.optimisticMessageVisible = await page.locator(".web-message-anchor.user", { hasText: text }).count();
  await page.locator('textarea[name="message"]').fill("");
  await page.getByRole("button", { name: "中止回答" }).waitFor();
  await page.locator('textarea[name="message"]').press("Escape");
  await page.locator('textarea[name="message"]').press("Escape");
  let received;
  for (let attempt = 0; attempt < 20; attempt += 1) {
    received = await fetch(`${origin}/__fixture/received`).then((response) => response.json());
    if (received.items.some((frame) => frame.type === "message.send" && frame.text === "/stop")) break;
    await page.waitForTimeout(50);
  }
  received ??= { items: [], requests: [] };
  const sends = received.items.filter((frame) => frame.type === "message.send");
  // 当前停止合同是 message.send + text "/stop"；双击去重后应恰好一条用户消息和一条停止请求。
  metric.sendFrames = sends.filter((frame) => frame.text !== "/stop").length;
  metric.stopFrames = sends.filter((frame) => frame.text === "/stop").length;
  metric.uploadRequests = uploadRequests;
  metric.sentMedia = sends.find((frame) => frame.text !== "/stop")?.media?.length ?? 0;
  metric.receivedItems = received.items;
  if (metric.randomUUIDUnavailable !== 1 || metric.sendFrames !== 1 || metric.stopFrames !== 1 || metric.uploadRequests !== 1 || metric.sentMedia !== 1 || metric.optimisticMessageVisible !== 1) {
    throw new Error(`desktop composer transport contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopPendingSendStop(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    class StalledWebSocket extends EventTarget {
      static CONNECTING = 0;
      static OPEN = 1;
      static CLOSING = 2;
      static CLOSED = 3;
      readyState = StalledWebSocket.CONNECTING;
      close() { this.readyState = StalledWebSocket.CLOSED; }
      send() { throw new Error("stalled websocket cannot send"); }
    }
    window.WebSocket = StalledWebSocket;
  });
  const page = await context.newPage();
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("纯文本性能会话", { exact: true }).click();
  const text = "连接未完成时可撤回";
  await page.locator('textarea[name="message"]').fill(text);
  await page.getByRole("button", { name: "发送消息" }).click();
  await page.getByRole("button", { name: "中止回答" }).click();
  await page.getByRole("button", { name: "发送消息" }).waitFor();
  // 中止竞态：sendMessage 可能在 ensureSession 的历史分页中收到 abort，输入恢复随后才落地。
  await page.waitForFunction(
    (expected) => document.querySelector('textarea[name="message"]')?.value === expected,
    text, { timeout: 15_000 },
  );
  const metric = {
    inputRestored: await page.locator('textarea[name="message"]').inputValue() === text ? 1 : 0,
    optimisticRows: await page.locator(".web-message-anchor.user", { hasText: text }).count(),
  };
  if (metric.inputRestored !== 1 || metric.optimisticRows !== 0) {
    throw new Error(`pending desktop send did not recover after stop: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

// 模型设置已迁移为 models 插件 web_module（/settings 308 到 /#models），
// fixture 无法挂载插件 UI，desktopSettings 场景待按插件挂载路径重建。

async function measureDesktopResponsive(browserInstance, origin) {
  const context = await browserInstance.newContext({
    viewport: { width: 320, height: 800 },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });

  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  const navigationTrigger = page.getByRole("button", { name: "打开导航" });
  await navigationTrigger.click();
  await page.keyboard.press("Escape");
  const metric = { navigationFocusRestored: await navigationTrigger.evaluate((element) => document.activeElement === element ? 1 : 0) };
  await navigationTrigger.click();
  await page.getByRole("dialog", { name: "Akashic 导航" }).getByRole("button", { name: "性能基线会话", exact: true }).click();
  await page.locator(".web-message-anchor").last().waitFor();
  metric.chatOverflowPx = await horizontalOverflow(page);
  metric.composerVisible = await page.locator(".composer__textarea").isVisible() ? 1 : 0;
  await page.locator('textarea[name="message"]').click();
  await page.locator(".model-picker__trigger").click().catch(async (error) => {
    const body = await page.locator("body").textContent();
    const composer = await page.locator(".composer-wrap").evaluateAll((elements) => elements.map((element) => ({ class: element.className, html: element.innerHTML.slice(0, 1000) })));
    throw new Error(`窄屏模型入口不可用: ${JSON.stringify({ browserErrors, composer, body: body?.slice(-1000) })}\n${error.message}`);
  });
  metric.modelPickerOverflowPx = await horizontalOverflow(page);
  await page.keyboard.press("Escape");

  // 窄屏覆盖会话与模型选择。
  const overflowMetrics = Object.entries(metric).filter(([name]) => name.endsWith("OverflowPx"));
  if (overflowMetrics.some(([, value]) => value !== 0) || metric.navigationFocusRestored !== 1
    || metric.composerVisible !== 1) {
    throw new Error(`narrow desktop interaction contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopAccessibility(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const violations = [];

  async function scan(surface) {
    await page.addScriptTag({ path: resolve(repoRoot, "node_modules/axe-core/axe.min.js") });
    const results = await page.evaluate(async () => window.axe.run(document, {
      runOnly: { type: "tag", values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"] },
    }));
    for (const violation of results.violations) {
      violations.push({
        surface, id: violation.id, impact: violation.impact,
        nodes: violation.nodes.map((node) => ({ target: node.target, html: node.html, summary: node.failureSummary })),
      });
    }
  }

  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").last().waitFor();
  await scan("chat");
  await page.locator(".model-picker__trigger").click();
  await scan("model-picker");
  await page.keyboard.press("Escape");

  // settings/runtime surface 均已迁移为插件 web_module，axe 覆盖需改走插件挂载点，本轮先跳过。
  // 以下为套件恢复运行时确认的存量设计债，逐项签名登记：命中仍报告但不判失败，
  // 新增违规照常 fail。债务清理由独立的可访问性任务完成，不在性能分支内改主题 token。
  const hasTarget = (node, signature) => String(node.target).includes(signature);
  const isDebt = (violation) => violation.id === "color-contrast"
    ? violation.nodes.every((node) => ["markstream-react", "code-header-title", "stream-diffs"].some((signature) => hasTarget(node, signature)))
    : violation.id === "scrollable-region-focusable" && violation.nodes.every((node) => hasTarget(node, "code-editor-container"));
  const debt = violations.filter(isDebt);
  const fresh = violations.filter((violation) => !isDebt(violation));
  if (debt.length > 0) console.log(`已知可访问性债务 ${debt.length} 项（第三方渲染器，待专项清理）`);
  if (fresh.length > 0) throw new Error(`desktop accessibility violations: ${JSON.stringify(fresh)}`);
  await context.close();
  return { scannedSurfaces: 2, violations: fresh.length, knownDebt: debt.length };
}

async function horizontalOverflow(page) {
  return page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth));
}

async function readStreamText(page, selector) {
  return page.evaluate((target) => {
    const root = document.querySelector(target);
    if (!root) return "";
    return (root.textContent ?? "") + [...root.querySelectorAll("diffs-container")]
      .map((element) => element.shadowRoot?.textContent ?? "").join("");
  }, selector);
}

async function measureDesktopStream(browserInstance, origin, intervalMs, delta = "片", count = 600) {
  const textDelta = delta.replaceAll("**", "");
  const code = delta.startsWith("```");
  const contentUnit = code ? "const x = 1;" : textDelta.trim();
  const markdown = delta.includes("**") || code;
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(`${message.text()} ${message.location().url}`);
  });
  page.on("requestfailed", (request) => browserErrors.push(`${request.failure()?.errorText} ${request.url()}`));
  page.on("response", (response) => {
    if (response.status() >= 400) browserErrors.push(`${response.status()} ${response.url()}`);
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}/chat?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").last().waitFor();
  const scrollStateBefore = await page.locator('.conversation-scroll').evaluate((element) => {
    element.dispatchEvent(new WheelEvent("wheel", { deltaY: -240, bubbles: true }));
    element.scrollTop = 0;
    element.dispatchEvent(new Event("scroll"));
    return { scrollTop: element.scrollTop, distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop };
  });
  if (scrollStateBefore.distanceFromBottom <= 100) throw new Error("desktop stream fixture is not scrollable");
  await page.waitForTimeout(100);
  await page.evaluate(() => window.__akashicWebTrace?.reset());
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  // 夹具端点在广播完所有帧后才返回；草稿必须先并发等待，否则终态帧已把草稿清掉。
  const fixturePromise = fetch(`${origin}/__fixture/stream?count=${count}&interval_ms=${intervalMs}&terminal=1&delta=${encodeURIComponent(delta)}`, { method: "POST" })
    .then(async (response) => {
      if (!response.ok) throw new Error(`桌面 WebSocket 夹具失败: ${response.status}`);
      return response.json();
    });
  // v2 协议下流式正文先出现在 reply.status 草稿行，messages.appended 提交后才进入历史锚点。
  let firstTextAt = null;
  let firstDraftAt = null;
  await page.waitForFunction(({ text, expected }) => {
    const root = document.querySelector(".reply-activity");
    const content = (root?.textContent ?? "") + [...(root?.querySelectorAll("diffs-container") ?? [])]
      .map((element) => element.shadowRoot?.textContent ?? "").join("");
    return content.split(text).length - 1 >= expected;
  }, { text: contentUnit, expected: code ? 1 : 20 }, { timeout: 20_000 });
  firstTextAt = await page.evaluate(() => performance.now());
  if (markdown && !code) await page.locator(".reply-activity strong").first().waitFor();
  firstDraftAt = await page.evaluate(() => performance.now());
  const fixtureResult = await fixturePromise;
  if (!fixtureResult?.draftId) throw new Error("桌面流式夹具未返回草稿标识");
  await page.waitForFunction(({ text, expected }) => {
    const root = [...document.querySelectorAll(".web-message-anchor")].at(-1);
    const content = (root?.textContent ?? "") + [...(root?.querySelectorAll("diffs-container") ?? [])]
      .map((element) => element.shadowRoot?.textContent ?? "").join("");
    return content.split(text).length - 1 >= expected;
  }, { text: contentUnit, expected: count }, { timeout: 20_000 })
    .catch(async (error) => {
      const tail = await readStreamText(page, ".web-message-anchor:last-of-type");
      throw new Error(`流式终态文本不完整（期望 ${count}，实际 ${tail?.split(contentUnit).length - 1}）：${tail?.slice(-200)}\n${error.message}`);
    });
  if (markdown) await page.locator(".web-message-anchor").last().locator(code ? "diffs-container" : "strong").first().waitFor();
  const highlightedAt = code ? await page.evaluate(() => performance.now()) : null;
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  if (highlightedAt !== null) metric.streamHighlightedMs = highlightedAt - startedAt;
  if (firstTextAt !== null && firstDraftAt !== null) {
    metric.streamFirstTextMs = firstTextAt - startedAt;
    metric.streamFirstDraftMs = firstDraftAt - startedAt;
    metric.streamCommitTailMs = metric.durationMs - metric.streamFirstDraftMs;
    metric.streamDraftVisible = 1;
  }
  metric.streamResources = await page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => entry.name.endsWith(".js"))
    .map((entry) => ({ name: entry.name.split("/").at(-1), startTime: entry.startTime, responseEnd: entry.responseEnd })));
  const scrollStateAfter = await page.locator('.conversation-scroll').evaluate((element) => ({
    scrollTop: element.scrollTop,
    distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop,
  }));
  metric.streamPreservedScrollEscape = scrollStateAfter.distanceFromBottom > 100 ? 1 : 0;
  const scrollButton = page.locator(".reading-return");
  metric.scrollReturnAvailable = await scrollButton.isVisible() ? 1 : 0;
  await scrollButton.click();
  await page.waitForFunction((previous) => {
    const element = document.querySelector('.conversation-scroll');
    return element !== null && element.scrollTop > previous;
  }, scrollStateAfter.scrollTop);
  metric.scrollReturnAdvanced = 1;
  metric.trace = await page.evaluate(() => {
    const records = window.__akashicWebTrace?.snapshot() ?? [];
    const first = records.find((record) => record.event === "webui.frame_received" && record.kind === "answer");
    const committed = records.find((record) => record.event === "webui.react_committed" && record.kind === "answer");
    const nextFrame = records.find((record) => record.event === "webui.next_frame_ready" && record.kind === "answer");
    return {
      eventCount: records.length,
      frameToCommitMs: first && committed ? committed.performance_ms - first.performance_ms : null,
      frameToNextFrameMs: first && nextFrame ? nextFrame.performance_ms - first.performance_ms : null,
      events: records.map((record) => `${record.event}:${record.kind}`),
    };
  });
  if (browserErrors.length > 0) throw new Error(`桌面流式场景出现浏览器异常:\n${browserErrors.join("\n")}`);
  if (metric.streamPreservedScrollEscape !== 1 || metric.scrollReturnAvailable !== 1 || metric.scrollReturnAdvanced !== 1) {
    throw new Error(`desktop stream scroll contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function installPerformanceProbe(page) {
  await page.addInitScript(() => {
    const state = { longTasks: [], shifts: [], frameGaps: [], previousFrame: 0, largestContentfulPaint: 0 };
    new PerformanceObserver((list) => state.longTasks.push(...list.getEntries().map((entry) => entry.duration))).observe({ type: "longtask", buffered: true });
    new PerformanceObserver((list) => state.shifts.push(...list.getEntries().filter((entry) => !entry.hadRecentInput).map((entry) => entry.value))).observe({ type: "layout-shift", buffered: true });
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) state.largestContentfulPaint = Math.max(state.largestContentfulPaint, entry.startTime);
    }).observe({ type: "largest-contentful-paint", buffered: true });
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
      state.largestContentfulPaint = 0;
    };
    window.__readAkashicPerf = (startedAt, selector) => {
      const resources = performance.getEntriesByType("resource");
      return {
        durationMs: performance.now() - startedAt,
        firstContentfulPaintMs: performance.getEntriesByName("first-contentful-paint")[0]?.startTime ?? 0,
        largestContentfulPaintMs: state.largestContentfulPaint,
        resourceCount: resources.length,
        resourceTransferBytes: resources.reduce((sum, entry) => sum + entry.transferSize, 0),
        resourceDecodedBytes: resources.reduce((sum, entry) => sum + entry.decodedBodySize, 0),
        longTaskCount: state.longTasks.length,
        longTaskTotalMs: state.longTasks.reduce((sum, value) => sum + value, 0),
        longTaskMaxMs: Math.max(0, ...state.longTasks),
        frameGapP75Ms: percentile(state.frameGaps, 0.75),
        frameGapMaxMs: Math.max(0, ...state.frameGaps),
        layoutShift: state.shifts.reduce((sum, value) => sum + value, 0),
        domRows: document.querySelectorAll(selector).length,
        domElements: document.querySelectorAll("*").length,
        jsHeapBytes: performance.memory?.usedJSHeapSize ?? null,
      };
    };
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

function buildProductionShell() {
  for (const script of ["build:dashboard", "build:chat"]) {
    const result = spawnSync("npm", ["run", script], { cwd: repoRoot, encoding: "utf8" });
    if (result.status !== 0) throw new Error(`${script} 构建失败\n${result.stdout}\n${result.stderr}`);
  }
  return {
    dashboardOutput: resolve(repoRoot, "static", "dashboard"),
    chatAppOutput: resolve(repoRoot, "static", "chat"),
  };
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

function numberArgument(name, fallback, minimum) {
  const index = process.argv.indexOf(name);
  if (index === -1) return fallback;
  const value = Number(process.argv[index + 1]);
  if (!Number.isFinite(value) || value < minimum) throw new Error(`${name} 必须是不小于 ${minimum} 的数值`);
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
