/** 真实 Runtime 的热会话切换：正文可见与最后代码块高亮分别计时，失败保留。 */
import { chromium } from "playwright-core";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: {
  url: { type: "string" }, output: { type: "string" }, browser: { type: "string" }, build: { type: "string" },
  profiles: { type: "string", default: "lan,weak" }, runs: { type: "string", default: "3" },
  label: { type: "string", default: "baseline" },
} });
if (!values.url || !values.output || !values.browser || !values.build) throw new Error("需要 --url、--output、--browser、--build");
const rounds = Number(values.runs), dir = values.output;
if (!Number.isSafeInteger(rounds) || rounds < 1) throw new Error("runs 必须是正整数");
const profiles = {
  lan: { latency: 20, downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 },
  weak: { latency: 800, downloadThroughput: 125_000, uploadThroughput: 125_000 },
};
const selected = values.profiles.split(",");
if (selected.length !== 2 || new Set(selected).size !== 2 || selected.some(name => !profiles[name])) {
  throw new Error("profiles 必须包含 lan、weak 各一次");
}
mkdirSync(dir);
const state = await (await fetch(new URL("/api/shell/state", values.url))).json();
if (state.build !== values.build) throw new Error("服务构建与 --build 不符");
const browser = await chromium.launch({ executablePath: values.browser });
const report = {
  complete: false, label: values.label, build: values.build, browser: browser.version(), profiles,
  harnessSha256: createHash("sha256").update(readFileSync(import.meta.filename)).digest("hex"), samples: [], setupErrors: [],
};
const save = () => writeFileSync(`${dir}/report.json`, JSON.stringify(report, null, 2));
const targets = [
  { name: "切换基准消息", id: "switch-1999", text: "切换基准消息 1999", rich: true },
  { name: "启动基准消息", id: "startup-099", text: "启动基准消息 099", rich: false },
];

/** 真实点击后，等待目标末条可见、50 条历史到齐、输入可用，再跨过两个绘制帧。 */
async function switchSession(page, target) {
  return page.evaluate(target => new Promise((resolve, reject) => {
    const button = [...document.querySelectorAll("button.conversation-session")].find(el => el.textContent.startsWith(target.name));
    if (!button?.getClientRects().length) return reject(new Error("target button missing"));
    let watched, visible = false, done = false;
    const start = performance.now();
    window.__switchStart = start;
    const stop = () => { changes.disconnect(); view.disconnect(); clearTimeout(timer); };
    const view = new IntersectionObserver(entries => {
      visible = entries.some(entry => entry.isIntersecting && entry.intersectionRect.height > 0);
      check();
    });
    const check = () => {
      if (done) return;
      const row = document.querySelector(`[data-message-id="${target.id}"] .message-row`);
      const input = document.querySelector(".composer-wrap textarea");
      if (!row?.textContent.includes(target.text) || !input || input.disabled || document.querySelectorAll("[data-message-id]").length !== 50) return;
      if (watched !== row) { watched = row; visible = false; view.disconnect(); view.observe(row); return; }
      if (!visible) return;
      done = true;
      changes.disconnect(); view.disconnect();
      requestAnimationFrame(() => requestAnimationFrame(() => {
        stop();
        const rect = row.getBoundingClientRect();
        if (!row.isConnected || rect.bottom <= 0 || rect.top >= innerHeight) return reject(new Error("content disappeared before paint"));
        resolve(performance.now() - start);
      }));
    };
    const changes = new MutationObserver(check);
    changes.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["disabled"] });
    const timer = setTimeout(() => { stop(); reject(new Error("content timeout")); }, 15_000);
    button.click(); check();
  }), target);
}

/** 不把无色代码占位算作高亮完成。 */
async function waitForHighlight(page, target) {
  if (!target.rich) return;
  await page.waitForFunction(id => {
    const root = document.querySelector(`[data-message-id="${id}"] diffs-container`)?.shadowRoot;
    return root?.querySelector('[data-line] span[style*="--diffs-token-"]') && root.textContent.includes("1999");
  }, target.id, { timeout: 15_000 });
  await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
}

let currentPage;
try {
  for (const cpu of [1, 4]) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
    await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
    const page = await context.newPage(), cdp = await context.newCDPSession(page);
    currentPage = page;
    await cdp.send("Network.enable");
    await cdp.send("Performance.enable");
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
    const errors = [];
    page.on("pageerror", error => { errors.push(error.message); report.setupErrors.push({ cpu, error: error.message }); });
    page.on("requestfailed", request => report.setupErrors.push({ cpu, path: new URL(request.url()).pathname, error: request.failure()?.errorText }));
    page.on("response", response => { if (!response.ok()) report.setupErrors.push({ cpu, path: new URL(response.url()).pathname, status: response.status() }); });
    await page.goto(new URL("/?session=akashic%3Astartup-benchmark", values.url).href);
    await page.locator('[data-message-id="startup-099"]').waitFor();
    // 1. 留在真实富文本会话等高亮、字体与默认 20 秒启动预热完成。
    await switchSession(page, targets[0]); await waitForHighlight(page, targets[0]);
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(21_000);
    await page.waitForLoadState("networkidle");
    await switchSession(page, targets[1]);
    // 2. 网络顺序轮换；前一场景的请求必须结束，避免跨入下一条瀑布流。
    for (let round = 0; round < rounds; round++) for (const profile of (round % 2 ? [...selected].reverse() : selected)) {
      await cdp.send("Network.emulateNetworkConditions", { offline: false, ...profiles[profile] });
      for (const target of targets) {
        await page.waitForLoadState("networkidle");
        const row = { cpu, round, profile, target: target.id, success: false, requests: [] };
        report.samples.push(row);
        const requests = new Map(), start = performance.now();
        const onRequest = request => {
          const item = { method: request.method(), path: new URL(request.url()).pathname, startMs: performance.now() - start };
          row.requests.push(item); requests.set(request, item);
        };
        const onDone = async request => {
          const item = requests.get(request);
          if (item) { item.endMs = performance.now() - start; item.status = (await request.response())?.status(); }
        };
        const onFailed = request => {
          const item = requests.get(request);
          if (item) { item.endMs = performance.now() - start; item.error = request.failure()?.errorText; }
        };
        page.on("request", onRequest); page.on("requestfinished", onDone); page.on("requestfailed", onFailed);
        const before = await cdp.send("Performance.getMetrics");
        try {
          row.contentMs = await switchSession(page, target);
          await waitForHighlight(page, target);
          row.richMs = await page.evaluate(() => performance.now() - window.__switchStart);
          if (errors.length) throw new Error(errors.join("\n"));
          row.success = true;
        } catch (error) { row.error = String(error); process.exitCode = 1; }
        const after = await cdp.send("Performance.getMetrics");
        row.metrics = Object.fromEntries(after.metrics.filter(metric => ["LayoutDuration", "LayoutCount", "RecalcStyleDuration", "ScriptDuration", "TaskDuration"].includes(metric.name))
          .map(metric => [metric.name, metric.value - before.metrics.find(previous => previous.name === metric.name).value]));
        // 3. 多观察一秒后台读取；未结束请求保留为空，不伪造完成时间。
        await page.waitForTimeout(1000);
        page.off("request", onRequest); page.off("requestfinished", onDone); page.off("requestfailed", onFailed);
        row.hotAssetRequests = row.requests.filter(request => /\.(js|css|woff2)$/.test(request.path)).length;
        if (row.hotAssetRequests) { row.success = false; row.error = "热路径仍在获取静态资源"; process.exitCode = 1; }
        row.screenshot = `${cpu}-${round}-${profile}-${target.id}.png`;
        await page.screenshot({ path: `${dir}/${row.screenshot}` });
        save(); console.log(JSON.stringify(row));
      }
    }
    await context.close();
  }
  report.complete = true;
} catch (error) {
  report.error = String(error); process.exitCode = 1;
  if (currentPage && !currentPage.isClosed()) await currentPage.screenshot({ path: `${dir}/failure.png` });
}
finally { save(); await browser.close(); }
