/** 测声明式表单热重开的已读内容，并单独等待后台核对，不发送配置。 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
const { values } = parseArgs({ options: {
  baseline: { type: "string" }, candidate: { type: "string" }, browser: { type: "string" }, output: { type: "string" },
  runs: { type: "string", default: "7" },
} });
for (const key of ["baseline", "candidate", "browser", "output"]) assert.ok(values[key], `需要 --${key}`);
const rounds = Number(values.runs);
assert.ok(Number.isSafeInteger(rounds) && rounds > 0);
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const browser = await chromium.launch({ executablePath: values.browser });
const report = { browser: browser.version(), plannedSamples: rounds * 8, runs: [],
  harnessSha256: createHash("sha256").update(readFileSync(import.meta.filename)).digest("hex") };
const save = () => writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
const fields = page => page.locator(".declared-form").evaluateAll(sections => sections.map(section => ({
  title: section.querySelector("h2").textContent,
  fields: [...section.querySelectorAll("input,textarea,select")].map(field => ({
    name: field.name, value: field.value, checked: field.checked, placeholder: field.placeholder,
  })),
})));

/** 两张完整表单和两次绘制就绪；这里只度量上次已读内容，最新值另行核对。 */
async function ready(page) {
  await page.waitForFunction(() => document.querySelectorAll(".declared-form .config-form > form").length === 2, {}, { timeout: 15000 });
  await page.evaluate(() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done))));
}

try {
  for (let pair = 0; pair < rounds; pair++) for (const cpu of [1, 4]) {
    for (const profile of pair % 2 ? ["weak", "lan"] : ["lan", "weak"]) {
      for (const variant of pair % 2 ? ["candidate", "baseline"] : ["baseline", "candidate"]) {
        const row = { pair, cpu, profile, variant, success: false, requests: [] };
        report.runs.push(row);
        const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
        await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
        try {
          // 1. 用真实页面和请求读取一次配置；每个样本都有独立 context。
          const page = await context.newPage();
          const cdp = await context.newCDPSession(page);
          await cdp.send("Network.enable");
          await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
          await page.goto(values[variant]);
          await page.getByRole("button", { name: "功能设置", exact: true }).click();
          const plugins = page.locator(".shell-settings-nav").getByRole("button", { name: "插件", exact: true });
          await plugins.click();
          await page.getByRole("tab", { name: "插件配置", exact: true }).click();
          const card = page.locator(".shell-plugin-card").filter({ has: page.locator("strong", { hasText: /^插件设置$/ }) });
          await card.click();
          await ready(page);
          const original = await fields(page);
          assert.ok(original.every(section => section.fields.length === 4));
          await plugins.click();
          await card.waitFor();
          await cdp.send("Network.emulateNetworkConditions", { offline: false,
            latency: profile === "lan" ? 20 : 800,
            downloadThroughput: profile === "lan" ? 12500000 : 125000, uploadThroughput: 125000,
          });
          // 2. 请求开始/完成均入报告；网络观察不使用 route，不改变 HTTP 缓存行为。
          const started = performance.now();
          const observed = new Map();
          page.on("request", request => {
            if (!new URL(request.url()).pathname.startsWith("/api/dashboard/plugin_settings/")) return;
            const item = { method: request.method(), path: new URL(request.url()).pathname, startMs: performance.now() - started };
            row.requests.push(item); observed.set(request, item);
          });
          page.on("response", response => { const item = observed.get(response.request()); if (item) item.status = response.status(); });
          page.on("requestfinished", request => { const item = observed.get(request); if (item) item.endMs = performance.now() - started; });
          page.on("requestfailed", request => { const item = observed.get(request); if (item) item.error = request.failure()?.errorText; });
          await card.evaluate(element => element.click());
          await ready(page);
          row.readyMs = performance.now() - started;
          assert.deepEqual(await fields(page), original, "重开没有完整保留上次已读字段");
          // 3. 就绪后仍等后台读取，不能靠提前关闭页面隐藏核对失败。
          const deadline = performance.now() + 15000;
          const reads = () => row.requests.filter(request => request.path.endsWith("/config"));
          while (performance.now() < deadline && (reads().length !== 2 || reads().some(request => !request.endMs && !request.error))) {
            await page.waitForTimeout(20);
          }
          assert.equal(reads().length, 2);
          assert.ok(reads().every(request => request.endMs && request.status === 200 && !request.error), "后台核对未完成或失败");
          row.freshMs = Math.max(...reads().map(request => request.endMs));
          await ready(page);
          assert.deepEqual(await fields(page), original, "后台核对意外改写字段");
          assert.equal(await page.locator(".declared-form .config-error").count(), 0);
          await page.screenshot({ path: resolve(output, `${pair}-${cpu}-${profile}-${variant}.png`) });
          row.success = true;
        } catch (error) { row.error = String(error); }
        finally { save(); console.log(JSON.stringify(row)); await context.close(); }
      }
    }
  }
} finally { save(); await browser.close(); }
if (report.runs.length !== report.plannedSamples || report.runs.some(row => !row.success)) process.exitCode = 1;
