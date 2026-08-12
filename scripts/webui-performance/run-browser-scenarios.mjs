import { createReadStream, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, extname, resolve, sep } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import { chromium } from "playwright-core";
import { startDesktopFixtureServer } from "./desktop-fixture-server.mjs";

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
const desktopStreamIntervalMs = numberArgument("--desktop-stream-interval-ms", 2.5, 0);
const buildRoot = mkdtempSync(resolve(tmpdir(), "akashic-webui-browser-"));
const results = [];
let browser;

try {
  const desktopOutput = buildTarget("frontend/chat/vite.config.ts", resolve(buildRoot, "desktop"));
  const mobileOutput = buildTarget("frontend/chat/vite.mobile.config.ts", resolve(buildRoot, "mobile"));
  const desktopServer = await startDesktopFixtureServer(desktopOutput);
  const mobileServer = await startStaticFixtureServer(mobileOutput, { stripAssetsPrefix: false });
  try {
    browser = await chromium.launch({ executablePath: chromiumExecutable(), headless: true });
    for (let run = 1; run <= runCount; run += 1) {
      results.push({
        run,
        scenarios: {
          desktopHistory: await measureDesktopHistory(browser, desktopServer.origin),
          desktopSessionSwitch: await measureDesktopSessionSwitch(browser, desktopServer.origin),
          desktopModelPicker: await measureDesktopModelPicker(browser, desktopServer.origin),
          desktopComposer: await measureDesktopComposer(browser, desktopServer.origin),
          desktopPendingSendStop: await measureDesktopPendingSendStop(browser, desktopServer.origin),
          desktopPairing: await measureDesktopPairing(browser, desktopServer.origin),
          desktopSettings: await measureDesktopSettings(browser, desktopServer.origin),
          desktopMemorySettings: await measureDesktopMemorySettings(browser, desktopServer.origin),
          desktopResponsive: await measureDesktopResponsive(browser, desktopServer.origin),
          desktopLazyRecovery: await measureDesktopLazyRecovery(browser, desktopServer.origin),
          desktopAccessibility: await measureDesktopAccessibility(browser, desktopServer.origin),
          desktopStream600: await measureDesktopStream(browser, desktopServer.origin, desktopStreamIntervalMs),
          desktopRuntime: await measureDesktopRuntime(browser, desktopServer.origin),
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
      fixture: { desktopStreamIntervalMs },
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
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByText("性能基线会话", { exact: true }).click();
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
  metric.codeCopyButtons = await page.locator("[data-static-code-copy]").count();
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
    if (/^\/api\/chat\/sessions\/[^/]+\/messages$/u.test(url.pathname) || url.pathname === "/api/chat/models") {
      requests.push(url.pathname + url.search);
    }
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.evaluate(() => window.__resetAkashicPerf());
  requests.length = 0;
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByText("纯文本性能会话", { exact: true }).click();
  await page.locator('[data-message-id="desktop-plain-99"]').waitFor();
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  metric.messageRequests = requests.filter((request) => request.includes("/messages")).length;
  metric.modelRequests = requests.filter((request) => request.startsWith("/api/chat/models")).length;
  requests.length = 0;
  await page.getByText("纯文本性能会话", { exact: true }).click();
  await page.waitForTimeout(100);
  metric.repeatMessageRequests = requests.filter((request) => request.includes("/messages")).length;
  metric.repeatModelRequests = requests.filter((request) => request.startsWith("/api/chat/models")).length;
  metric.sessionRows = await page.locator(".conversation-session").count();
  if (metric.repeatMessageRequests !== 0 || metric.repeatModelRequests !== 0) {
    throw new Error(`active session repeated requests: messages=${metric.repeatMessageRequests}, models=${metric.repeatModelRequests}`);
  }
  if (await page.getByRole("button", { name: /纯文本性能会话/u }).getAttribute("aria-current") !== "true") {
    throw new Error("selected desktop session does not expose its current state");
  }
  await context.close();
  return metric;
}

async function measureDesktopModelPicker(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  const metric = {
    closedOptions: await page.locator(".model-capsule__option").count(),
    closedDomElements: await page.locator("*").count(),
  };
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByRole("button", { name: /fixture：性能夹具/u }).click();
  await page.locator(".model-capsule__panel").waitFor();
  Object.assign(metric, await readPerformanceProbe(page, startedAt, ".model-capsule__option"));
  metric.openOptions = await page.locator(".model-capsule__option").count();
  await page.keyboard.press("End");
  metric.keyboardEnd = await page.evaluate(() => document.activeElement?.classList.contains("model-capsule__effort-entry") ? 1 : 0);
  await page.keyboard.press("Home");
  metric.keyboardHome = await page.evaluate(() => document.activeElement?.classList.contains("model-capsule__option") ? 1 : 0);
  await page.keyboard.press("Escape");
  metric.focusRestored = await page.evaluate(() => document.activeElement?.classList.contains("model-capsule__trigger") ? 1 : 0);
  if (metric.closedOptions !== 0 || metric.keyboardEnd !== 1 || metric.keyboardHome !== 1 || metric.focusRestored !== 1) {
    throw new Error(`model picker interaction contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopComposer(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  let uploadRequests = 0;
  page.on("request", (request) => {
    if (new URL(request.url()).pathname === "/api/chat/uploads") uploadRequests += 1;
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("纯文本性能会话", { exact: true }).click();
  await page.locator('[data-message-id="desktop-plain-99"]').waitFor();
  await fetch(`${origin}/__fixture/reset`, { method: "POST" });
  await fetch(`${origin}/__fixture/history-delay?ms=500`, { method: "POST" });
  await fetch(`${origin}/__fixture/stream?count=1&interval_ms=0&terminal=1`, { method: "POST" });
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
  metric.messageRowsAfterTyping = await page.locator(".web-message-anchor").count();
  await page.locator('input[type="file"]').setInputFiles({ name: "composer.txt", mimeType: "text/plain", buffer: Buffer.from("附件内容") });
  await page.getByText("composer.txt", { exact: true }).waitFor();
  await page.getByRole("button", { name: "发送消息" }).click();
  await page.getByRole("button", { name: "中止回答" }).waitFor();
  await page.waitForTimeout(600);
  metric.optimisticMessageVisible = await page.locator(".web-message-anchor.user", { hasText: text }).count();
  await page.evaluate(() => {
    const button = document.querySelector('.composer-action-button[data-mode="stop"]');
    button?.click();
    button?.click();
  });
  await page.waitForTimeout(100);
  const received = await fetch(`${origin}/__fixture/received`).then((response) => response.json());
  metric.sendFrames = received.items.filter((frame) => frame.type === "message.send").length;
  metric.stopFrames = received.items.filter((frame) => frame.type === "turn.stop").length;
  metric.uploadRequests = uploadRequests;
  metric.sentMedia = received.items.find((frame) => frame.type === "message.send")?.media?.length ?? 0;
  if (metric.sendFrames !== 1 || metric.stopFrames !== 1 || metric.uploadRequests !== 1 || metric.sentMedia !== 1 || metric.optimisticMessageVisible !== 1) {
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
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("纯文本性能会话", { exact: true }).click();
  const text = "连接未完成时可撤回";
  await page.locator('textarea[name="message"]').fill(text);
  await page.getByRole("button", { name: "发送消息" }).click();
  await page.getByRole("button", { name: "中止回答" }).click();
  await page.getByRole("button", { name: "发送消息" }).waitFor();
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

async function measureDesktopPairing(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  let abortedCreates = 0;
  page.on("requestfailed", (request) => {
    if (new URL(request.url()).pathname === "/api/chat/mobile-pairing") abortedCreates += 1;
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  const metric = {
    initialScripts: await page.locator('script[src]').count(),
    initialDomElements: await page.locator("*").count(),
    initialPairingResources: await pairingResourceCount(page),
  };
  const trigger = page.getByRole("button", { name: "连接手机" });
  await Promise.all([
    page.waitForRequest((request) => new URL(request.url()).pathname === "/api/chat/mobile-pairing"),
    trigger.click(),
  ]);
  await page.getByRole("button", { name: "取消" }).click();
  await page.waitForTimeout(400);
  metric.cancelledCreateRequests = abortedCreates;
  if (metric.cancelledCreateRequests !== 1) {
    throw new Error(`closing pairing dialog did not abort its create request: ${metric.cancelledCreateRequests}`);
  }
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await trigger.click();
  await page.getByAltText("Android 手机配对二维码").waitFor();
  Object.assign(metric, await readPerformanceProbe(page, startedAt, ".mobile-pairing-dialog"));
  metric.dialogScripts = await page.locator('script[src]').count();
  metric.dialogPairingResources = await pairingResourceCount(page);
  if (metric.initialPairingResources !== 0 || metric.dialogPairingResources < 1) {
    throw new Error(`pairing code was not loaded on demand: ${JSON.stringify(metric)}`);
  }
  await page.getByText("358864", { exact: false }).waitFor({ timeout: 5_000 });
  await page.getByRole("button", { name: "确认并连接" }).click();
  await page.getByText("手机已连接", { exact: true }).waitFor();
  await page.getByRole("button", { name: "完成" }).click();
  metric.focusRestored = await page.evaluate(() => document.activeElement?.textContent?.includes("连接手机") ? 1 : 0);
  if (metric.focusRestored !== 1) throw new Error("pairing dialog did not restore focus to its trigger");
  await context.close();
  return metric;
}

async function measureDesktopSettings(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => { if (message.type() === "error") browserErrors.push(message.text()); });
  await installPerformanceProbe(page);
  await fetch(`${origin}/__fixture/reset`, { method: "POST" });
  const readyStartedAt = Date.now();
  await page.goto(`${origin}/settings?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "模型连接" }).waitFor();
  const metric = {
    initialReadyMs: Date.now() - readyStartedAt,
    initialDomElements: await page.locator("*").count(),
    connectionCards: await page.locator(".settings-connection-card").count(),
  };

  const customTrigger = page.getByRole("button", { name: /自定义 API/u });
  await customTrigger.click();
  await page.waitForTimeout(100);
  if (browserErrors.length > 0) throw new Error(`settings browser error:\n${browserErrors.join("\n")}`);
  if (await page.locator(".settings-dialog").count() === 0) {
    throw new Error(`settings dialog was not mounted: ${await customTrigger.count()} triggers, ${await page.locator("body").innerText()}`);
  }
  const dialog = page.getByRole("dialog", { name: "连接自定义 API" });
  await dialog.waitFor();
  const nameInput = dialog.getByRole("textbox", { name: "连接名称" });
  metric.initialFocus = await nameInput.evaluate((element) => document.activeElement === element ? 1 : 0);
  const closeButton = dialog.getByRole("button", { name: "关闭" });
  await closeButton.focus();
  await closeButton.press("Shift+Tab");
  metric.focusTrapped = await page.evaluate(() => document.activeElement?.textContent?.includes("保存连接") ? 1 : 0);
  await nameInput.focus();
  await page.evaluate(() => window.__resetAkashicPerf());
  const typingStartedAt = await page.evaluate(() => performance.now());
  await nameInput.pressSequentially("连接名称".repeat(30));
  Object.assign(metric, await readPerformanceProbe(page, typingStartedAt, ".settings-connection-card"));
  await dialog.getByRole("textbox", { name: "Provider ID" }).fill("fixture");
  await dialog.getByRole("textbox", { name: "Base URL" }).fill("https://api.example.com/v1");
  await dialog.getByRole("textbox", { name: "API Key" }).fill("fixture-secret");
  await page.evaluate(() => {
    const button = [...document.querySelectorAll("button")].find((item) => item.textContent?.includes("检测模型"));
    button?.click();
    button?.click();
  });
  await dialog.getByRole("combobox", { name: "模型名称" }).waitFor();
  await page.waitForTimeout(50);
  let received = await fetch(`${origin}/__fixture/received`).then((response) => response.json());
  metric.modelDiscoveryRequests = received.requests.filter((request) => request === "POST /api/settings/models").length;
  await page.keyboard.press("Escape");
  metric.focusRestored = await customTrigger.evaluate((element) => document.activeElement === element ? 1 : 0);

  const codexTrigger = page.getByRole("button", { name: /Codex ChatGPT/u });
  await codexTrigger.click();
  await page.getByRole("dialog", { name: "连接 Codex" }).waitFor();
  await page.evaluate(() => {
    const button = [...document.querySelectorAll("button")].find((item) => item.textContent?.includes("开始登录"));
    button?.click();
    button?.click();
  });
  await page.getByText("ABCD-EFGH", { exact: true }).waitFor();
  await page.getByText("Codex 已登录", { exact: true }).waitFor({ timeout: 5_000 });
  received = await fetch(`${origin}/__fixture/received`).then((response) => response.json());
  metric.codexLoginRequests = received.requests.filter((request) => request === "POST /api/settings/codex-login").length;
  metric.codexStatusRequests = received.requests.filter((request) => request === "GET /api/settings/codex-login/fixture-login").length;
  if (metric.initialFocus !== 1 || metric.focusTrapped !== 1 || metric.focusRestored !== 1) {
    throw new Error(`settings dialog focus contract failed: ${JSON.stringify(metric)}`);
  }
  if (metric.modelDiscoveryRequests !== 1 || metric.codexLoginRequests !== 1 || metric.codexStatusRequests !== 1) {
    throw new Error(`settings transport ownership failed: ${JSON.stringify(metric)}`);
  }
  await page.keyboard.press("Escape");
  await context.close();
  return metric;
}

async function measureDesktopMemorySettings(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await fetch(`${origin}/__fixture/reset`, { method: "POST" });
  await page.goto(`${origin}/settings?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "语义记忆" }).waitFor();
  await page.locator(".settings-memory-engines label").filter({ hasText: "Akasha" }).click();
  await page.getByRole("button", { name: "保存记忆设置" }).click();
  await page.getByRole("alert").getByText("启用记忆前", { exact: false }).waitFor();
  const addButton = page.getByRole("button", { name: /添加向量模型/u });
  const metric = { validationFocus: await addButton.evaluate((element) => document.activeElement === element ? 1 : 0) };
  await addButton.click();
  const dialog = page.getByRole("dialog", { name: "添加向量模型" });
  await dialog.getByRole("textbox", { name: "连接名称" }).fill("向量服务");
  await dialog.getByRole("textbox", { name: "Base URL" }).fill("https://embedding.example.com/v1");
  await dialog.getByRole("textbox", { name: "API Key" }).fill("fixture-embedding-secret");
  await dialog.getByRole("textbox", { name: "模型名称" }).fill("fixture-embedding-model");
  await page.evaluate(() => {
    const button = [...document.querySelectorAll("button")].find((item) => item.textContent?.includes("验证并保存"));
    button?.click(); button?.click();
  });
  await page.getByText("fixture-embedding-model 已验证", { exact: false }).waitFor();
  metric.dialogFocusRestored = await addButton.evaluate((element) => document.activeElement === element ? 1 : 0);
  await page.evaluate(() => {
    const button = [...document.querySelectorAll("button")].find((item) => item.textContent?.includes("保存记忆设置"));
    button?.click(); button?.click();
  });
  await page.getByText("Akasha 已启用", { exact: false }).waitFor();
  const received = await fetch(`${origin}/__fixture/received`).then((response) => response.json());
  metric.embeddingRequests = received.requests.filter((request) => request === "POST /api/settings/embedding-models").length;
  metric.memoryRequests = received.requests.filter((request) => request === "POST /api/settings/memory").length;
  if (Object.values(metric).some((value) => value !== 1)) {
    throw new Error(`memory settings interaction contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopResponsive(browserInstance, origin) {
  const context = await browserInstance.newContext({
    viewport: { width: 320, height: 800 },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();

  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  const navigationTrigger = page.getByRole("button", { name: "打开导航" });
  await navigationTrigger.click();
  await page.keyboard.press("Escape");
  const metric = { navigationFocusRestored: await navigationTrigger.evaluate((element) => document.activeElement === element ? 1 : 0) };
  await navigationTrigger.click();
  await page.getByRole("dialog", { name: "Akashic 导航" }).getByRole("button", { name: /性能基线会话/u }).click();
  await page.locator(".web-message-anchor").nth(99).waitFor();
  metric.chatOverflowPx = await horizontalOverflow(page);
  metric.composerVisible = await page.getByPlaceholder("有问题，尽管问").isVisible() ? 1 : 0;
  await page.locator(".model-capsule__trigger").click();
  metric.modelPickerOverflowPx = await horizontalOverflow(page);
  await page.keyboard.press("Escape");
  await navigationTrigger.click();
  await page.getByRole("dialog", { name: "Akashic 导航" }).getByRole("button", { name: "连接手机" }).click();
  await page.getByRole("dialog", { name: "连接 Android 手机" }).waitFor();
  metric.pairingOverflowPx = await horizontalOverflow(page);
  await page.keyboard.press("Escape");

  await page.goto(`${origin}/settings?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "模型连接" }).waitFor();
  metric.settingsOverflowPx = await horizontalOverflow(page);
  await page.getByRole("button", { name: /自定义 API/u }).click();
  await page.getByRole("dialog", { name: "连接自定义 API" }).waitFor();
  metric.settingsDialogOverflowPx = await horizontalOverflow(page);
  await page.keyboard.press("Escape");

  await page.goto(`${origin}?surface=runtime&akashic_perf=1`, { waitUntil: "networkidle" });
  await page.locator(".runtime-directory__item").first().click();
  await page.locator(".runtime-detail__markdown").waitFor();
  metric.runtimeOverflowPx = await horizontalOverflow(page);
  metric.runtimeTabsVisible = await page.locator('[role="tab"]:visible').count();
  const overflowMetrics = Object.entries(metric).filter(([name]) => name.endsWith("OverflowPx"));
  if (overflowMetrics.some(([, value]) => value !== 0) || metric.navigationFocusRestored !== 1
    || metric.composerVisible !== 1 || metric.runtimeTabsVisible < 1) {
    throw new Error(`narrow desktop interaction contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopLazyRecovery(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  let failedChunks = 0;
  await page.route(/settings-app-.*\.js/u, async (route) => {
    if (failedChunks === 0) {
      failedChunks += 1;
      await route.abort("failed");
    } else {
      await route.continue();
    }
  });
  await page.goto(`${origin}/settings?akashic_perf=1`, { waitUntil: "domcontentloaded" });
  const alert = page.getByRole("alert");
  await alert.getByRole("heading", { name: "界面加载失败" }).waitFor();
  const metric = {
    failedChunks,
    reloadActionVisible: await alert.getByRole("button", { name: "重新加载" }).isVisible() ? 1 : 0,
  };
  await alert.getByRole("button", { name: "重新加载" }).click();
  await page.getByRole("heading", { name: "模型连接" }).waitFor();
  metric.recovered = 1;
  if (Object.values(metric).some((value) => value !== 1)) {
    throw new Error(`lazy surface recovery contract failed: ${JSON.stringify(metric)}`);
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

  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").nth(99).waitFor();
  await scan("chat");
  await page.locator(".model-capsule__trigger").click();
  await scan("model-picker");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "连接手机" }).click();
  await page.getByRole("dialog", { name: "连接 Android 手机" }).waitFor();
  await page.waitForTimeout(250);
  await scan("pairing");

  await page.goto(`${origin}/settings?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByRole("heading", { name: "模型连接" }).waitFor();
  await scan("settings");
  await page.getByRole("button", { name: /自定义 API/u }).click();
  await page.getByRole("dialog", { name: "连接自定义 API" }).waitFor();
  await scan("settings-dialog");

  await page.goto(`${origin}?surface=runtime&akashic_perf=1`, { waitUntil: "networkidle" });
  await page.locator(".runtime-detail__markdown").waitFor();
  await scan("runtime");
  if (violations.length > 0) throw new Error(`desktop accessibility violations: ${JSON.stringify(violations)}`);
  await context.close();
  return { scannedSurfaces: 6, violations: 0 };
}

async function horizontalOverflow(page) {
  return page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth));
}

async function pairingResourceCount(page) {
  return page.evaluate(() => performance.getEntriesByType("resource")
    .filter((entry) => /mobile-pairing-dialog/u.test(entry.name)).length);
}

async function measureDesktopStream(browserInstance, origin, intervalMs) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const browserErrors = [];
  page.on("pageerror", (error) => browserErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") browserErrors.push(message.text());
  });
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.locator(".web-message-anchor").nth(99).waitFor();
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
  const fixtureResponse = await fetch(`${origin}/__fixture/stream?count=600&interval_ms=${intervalMs}&terminal=0`, { method: "POST" });
  if (!fixtureResponse.ok) throw new Error(`桌面 WebSocket 夹具失败: ${fixtureResponse.status}`);
  await page.waitForFunction(() => document.querySelector(".web-message-anchor:last-child")?.textContent?.includes("片".repeat(600)), null, { timeout: 20_000 });
  await page.waitForFunction(() => window.__akashicWebTrace?.snapshot().some((record) => record.event === "webui.next_frame_ready"));
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  const scrollStateAfter = await page.locator('.conversation-scroll').evaluate((element) => ({
    scrollTop: element.scrollTop,
    distanceFromBottom: element.scrollHeight - element.clientHeight - element.scrollTop,
  }));
  metric.streamPreservedScrollEscape = scrollStateAfter.distanceFromBottom > 100 ? 1 : 0;
  const scrollButton = page.getByRole("button", { name: "滚动到底部" });
  metric.scrollReturnAvailable = await scrollButton.isVisible() ? 1 : 0;
  await scrollButton.click();
  await page.waitForFunction(() => {
    const element = document.querySelector('.conversation-scroll');
    return element !== null && element.scrollHeight - element.clientHeight - element.scrollTop < 2;
  });
  metric.scrollReturnReachedBottom = 1;
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
  if (metric.streamPreservedScrollEscape !== 1 || metric.scrollReturnAvailable !== 1 || metric.scrollReturnReachedBottom !== 1) {
    throw new Error(`desktop stream scroll contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopRuntime(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  const detailRequests = [];
  page.on("request", (request) => {
    const url = new URL(request.url());
    if (/^\/api\/chat\/runtime\/(?:documents\/|jobs\/|mcp$)/u.test(url.pathname)) detailRequests.push(url.pathname + url.search);
  });
  await installPerformanceProbe(page);
  const startedAt = Date.now();
  await page.goto(`${origin}?surface=runtime&akashic_perf=1`, { waitUntil: "networkidle" });
  await page.locator(".runtime-detail__markdown").waitFor();
  const initialReadyMs = Date.now() - startedAt;
  const initialDetailRequests = detailRequests.length;
  detailRequests.length = 0;
  await page.evaluate(() => window.__resetAkashicPerf());
  const switchStartedAt = await page.evaluate(() => performance.now());
  await page.getByRole("tab", { name: "文档" }).focus();
  await page.getByRole("tab", { name: "文档" }).press("ArrowRight");
  await page.locator(".runtime-detail__markdown").getByText("filesystem", { exact: false }).waitFor();
  const metric = await readPerformanceProbe(page, switchStartedAt, ".runtime-directory__item");
  metric.initialReadyMs = initialReadyMs;
  metric.initialDetailRequests = initialDetailRequests;
  metric.tabSwitchDetailRequests = detailRequests.length;
  metric.runtimeInitialScripts = await page.locator('script[src]').count();
  if (metric.initialDetailRequests !== 1) throw new Error(`runtime initial detail requests: ${metric.initialDetailRequests}`);
  if (metric.tabSwitchDetailRequests !== 1) throw new Error(`runtime tab switch detail requests: ${metric.tabSwitchDetailRequests}`);
  await context.grantPermissions(["clipboard-read", "clipboard-write"], { origin });
  await page.getByRole("button", { name: "复制标识" }).click();
  await page.getByRole("button", { name: "标识已复制" }).waitFor();
  if (await page.evaluate(() => navigator.clipboard.readText()) !== "core/filesystem") {
    throw new Error("runtime detail copy did not preserve the selected identifier");
  }
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
      domElements: document.querySelectorAll("*").length,
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

async function startStaticFixtureServer(root, { stripAssetsPrefix }) {
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
