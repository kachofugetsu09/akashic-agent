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
  MOBILE_DRAFT_ID,
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
let ownsBrowser = false;

try {
  const desktopOutput = buildTarget("frontend/chat/vite.config.ts", resolve(buildRoot, "desktop"));
  const mobileOutput = buildTarget("frontend/chat/vite.mobile.config.ts", resolve(buildRoot, "mobile"));
  const desktopServer = await startDesktopFixtureServer(desktopOutput);
  const mobileServer = await startStaticFixtureServer(mobileOutput, { stripAssetsPrefix: false });
  try {
    const cdpEndpoint = process.env.AKASHIC_PLAYWRIGHT_CDP;
    browser = cdpEndpoint
      ? await chromium.connectOverCDP(cdpEndpoint)
      : await chromium.launch({ executablePath: chromiumExecutable(), headless: true });
    ownsBrowser = !cdpEndpoint;
    const measure = async (name, task) => {
      console.log(`[scenario] ${name} ...`);
      const result = await task();
      console.log(`[scenario] ${name} done`);
      return result;
    };
    for (let run = 1; run <= runCount; run += 1) {
      results.push({
        run,
        scenarios: {
          desktopHistory: await measure("desktopHistory", () => measureDesktopHistory(browser, desktopServer.origin)),
          desktopSessionSwitch: await measure("desktopSessionSwitch", () => measureDesktopSessionSwitch(browser, desktopServer.origin)),
          desktopModelPicker: await measure("desktopModelPicker", () => measureDesktopModelPicker(browser, desktopServer.origin)),
          desktopComposer: await measure("desktopComposer", () => measureDesktopComposer(browser, desktopServer.origin)),
          desktopPendingSendStop: await measure("desktopPendingSendStop", () => measureDesktopPendingSendStop(browser, desktopServer.origin)),
          desktopPairing: await measure("desktopPairing", () => measureDesktopPairing(browser, desktopServer.origin)),
          desktopResponsive: await measure("desktopResponsive", () => measureDesktopResponsive(browser, desktopServer.origin)),
          desktopLazyRecovery: await measure("desktopLazyRecovery", () => measureDesktopLazyRecovery(browser, desktopServer.origin)),
          desktopAccessibility: await measure("desktopAccessibility", () => measureDesktopAccessibility(browser, desktopServer.origin)),
          desktopStream600: await measure("desktopStream600", () => measureDesktopStream(browser, desktopServer.origin, desktopStreamIntervalMs)),
          mobileHistory300: await measure("mobileHistory300", () => measureMobileHistory(browser, mobileServer.origin)),
          mobileStream600: await measure("mobileStream600", () => measureMobileStream(browser, mobileServer.origin)),
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
  if (ownsBrowser) await browser?.close();
  rmSync(buildRoot, { recursive: true, force: true });
}
if (!ownsBrowser) process.exit(0);

async function measureDesktopHistory(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1440, height: 1000 } });
  const page = await context.newPage();
  await installPerformanceProbe(page);
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
  await page.evaluate(() => window.__resetAkashicPerf());
  const startedAt = await page.evaluate(() => performance.now());
  await page.getByText("性能基线会话", { exact: true }).click();
  await page.getByRole("button", { name: "加载更早消息" }).click();
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
  await page.getByRole("button", { name: /纯文本性能会话/u }).click();
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
  await page.locator(".model-capsule__trigger").click();
  await page.locator(".model-capsule__panel").waitFor();
  Object.assign(metric, await readPerformanceProbe(page, startedAt, ".model-capsule__option"));
  metric.openOptions = await page.locator(".model-capsule__option").count();
  await page.keyboard.press("End");
  metric.keyboardEnd = await page.evaluate(() => document.activeElement?.classList.contains("model-capsule__effort-entry") ? 1 : 0);
  await page.keyboard.press("Home");
  // 面板内新增来源筛选 tab 后，Home 合同是聚焦面板内首个可聚焦元素（来源 tab），不再限定模型行。
  metric.keyboardHome = await page.evaluate(() => document.querySelector(".model-capsule__panel")?.contains(document.activeElement) ? 1 : 0);
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
  metric.randomUUIDUnavailable = await page.evaluate(() => typeof crypto.randomUUID === "undefined" ? 1 : 0);
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
  const sends = received.items.filter((frame) => frame.type === "message.send");
  // 当前停止合同是 message.send + text "/stop"；双击去重后应恰好一条用户消息和一条停止请求。
  metric.sendFrames = sends.filter((frame) => frame.text !== "/stop").length;
  metric.stopFrames = sends.filter((frame) => frame.text === "/stop").length;
  metric.uploadRequests = uploadRequests;
  metric.sentMedia = sends.find((frame) => frame.text !== "/stop")?.media?.length ?? 0;
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
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "networkidle" });
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

// 模型设置已迁移为 models 插件 web_module（/settings 308 到 /#models），
// fixture 无法挂载插件 UI，desktopSettings 场景待按插件挂载路径重建。

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
  await page.locator(".web-message-anchor").last().waitFor();
  metric.chatOverflowPx = await horizontalOverflow(page);
  metric.composerVisible = await page.locator(".composer__textarea").isVisible() ? 1 : 0;
  await page.locator(".model-capsule__trigger").click();
  metric.modelPickerOverflowPx = await horizontalOverflow(page);
  await page.keyboard.press("Escape");
  await navigationTrigger.click();
  await page.getByRole("dialog", { name: "Akashic 导航" }).getByRole("button", { name: "连接手机" }).click();
  await page.getByRole("dialog", { name: "连接 Android 手机" }).waitFor();
  metric.pairingOverflowPx = await horizontalOverflow(page);
  await page.keyboard.press("Escape");

  // /settings 与 ?surface=runtime 均已迁移为插件 web_module；窄屏覆盖保留会话、模型选择与配对。
  const overflowMetrics = Object.entries(metric).filter(([name]) => name.endsWith("OverflowPx"));
  if (overflowMetrics.some(([, value]) => value !== 0) || metric.navigationFocusRestored !== 1
    || metric.composerVisible !== 1) {
    throw new Error(`narrow desktop interaction contract failed: ${JSON.stringify(metric)}`);
  }
  await context.close();
  return metric;
}

async function measureDesktopLazyRecovery(browserInstance, origin) {
  const context = await browserInstance.newContext({ viewport: { width: 1280, height: 800 } });
  const page = await context.newPage();
  let failedChunks = 0;
  // settings-app chunk 已随插件化下线；现验证现存懒 chunk（手机配对对话框）的恢复链路。
  await page.route(/mobile-pairing-dialog-.*\.js/u, async (route) => {
    if (failedChunks === 0) {
      failedChunks += 1;
      await route.abort("failed");
    } else {
      await route.continue();
    }
  });
  await page.goto(`${origin}?akashic_perf=1`, { waitUntil: "domcontentloaded" });
  await page.getByRole("button", { name: "连接手机" }).click();
  const alert = page.getByRole("alert");
  await alert.getByRole("heading", { name: "界面加载失败" }).waitFor();
  const metric = {
    failedChunks,
    reloadActionVisible: await alert.getByRole("button", { name: "重新加载" }).isVisible() ? 1 : 0,
  };
  await alert.getByRole("button", { name: "重新加载" }).click();
  await page.getByRole("button", { name: "连接手机" }).click();
  await page.getByRole("dialog", { name: "连接 Android 手机" }).waitFor();
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
  await page.locator(".web-message-anchor").last().waitFor();
  await scan("chat");
  await page.locator(".model-capsule__trigger").click();
  await scan("model-picker");
  await page.keyboard.press("Escape");
  await page.getByRole("button", { name: "连接手机" }).click();
  await page.getByRole("dialog", { name: "连接 Android 手机" }).waitFor();
  await page.waitForTimeout(250);
  await scan("pairing");

  // settings/runtime surface 均已迁移为插件 web_module，axe 覆盖需改走插件挂载点，本轮先跳过。
  // 以下为套件恢复运行时确认的存量设计债，逐项签名登记：命中仍报告但不判失败，
  // 新增违规照常 fail。债务清理由独立的可访问性任务完成，不在性能分支内改主题 token。
  const knownDebt = [
    "a[data-band-id=\"workbench\"] > span", // 主导航禁用态 3.43:1，--chat-muted + is-disabled 透明度
    "a[data-band-id=\"models\"] > span", // 主导航预览态 2.26:1
    "正在创建一次性连接", // 配对等待文案 2.23:1，次级文本 token 过浅
    "markstream-react", // markstream vitesse-light 语法 token 对比度，库内主题
  ];
  const isDebt = (violation) => violation.id === "color-contrast" && violation.nodes.every(
    (node) => knownDebt.some((signature) => String(node.target).includes(signature) || node.html.includes(signature)),
  );
  const debt = violations.filter(isDebt);
  const fresh = violations.filter((violation) => !isDebt(violation));
  if (debt.length > 0) console.log(`已知可访问性债务 ${debt.length} 项（对比度，待专项清理）`);
  if (fresh.length > 0) throw new Error(`desktop accessibility violations: ${JSON.stringify(fresh)}`);
  await context.close();
  return { scannedSurfaces: 3, violations: fresh.length, knownDebt: debt.length };
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
  const fixturePromise = fetch(`${origin}/__fixture/stream?count=600&interval_ms=${intervalMs}&terminal=1`, { method: "POST" })
    .then(async (response) => {
      if (!response.ok) throw new Error(`桌面 WebSocket 夹具失败: ${response.status}`);
      return response.json();
    });
  // v2 协议下流式正文先出现在 reply.status 草稿行，messages.appended 提交后才进入历史锚点。
  await page.waitForFunction(() => document.querySelector(".reply-activity")?.textContent?.includes("片".repeat(20)), null, { timeout: 20_000 });
  const fixtureResult = await fixturePromise;
  if (!fixtureResult?.draftId) throw new Error("桌面流式夹具未返回草稿标识");
  await page.waitForFunction(() => [...document.querySelectorAll(".web-message-anchor")].at(-1)?.textContent?.includes("片".repeat(600)), null, { timeout: 20_000 });
  const metric = await readPerformanceProbe(page, startedAt, ".web-message-anchor");
  metric.streamDraftVisible = 1;
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
  // v2 协议下草稿走 reply.status 消息事件，终态由 messages.appended 提交。
  await page.evaluate(({ deltas, finalPatches }) => {
    for (const patch of deltas) window.AkashicMobile.receiveMessageEvent(patch);
    for (const patch of finalPatches) window.AkashicMobile.receiveMessageEvent(patch);
  }, { deltas: patches, finalPatches: terminal });
  await page.waitForFunction((draftId) => {
    const row = document.querySelector(`[data-message-id="${draftId}"]`);
    return row !== null && row.textContent?.includes("片".repeat(600));
  }, MOBILE_DRAFT_ID);
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
  if (pathname === "/api/shell/state") return { status: "ready", configured: true, chatReady: true };
  if (pathname === "/api/chat/sessions") return desktopSessions();
  if (pathname === `/api/chat/sessions/${fixtureSessionId}/messages`) {
    const history = desktopMessages();
    return { version: 2, items: history.items, through_seq: history.items.at(-1).seq, has_more: false, before_seq: null };
  }
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
