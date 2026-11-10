import { chromium } from "playwright-core";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";

const args = Object.fromEntries(process.argv.slice(2).reduce((pairs, value, index, all) =>
  index % 2 ? pairs : [...pairs, [value.replace(/^--/, ""), all[index + 1]]], []));
const output = resolve(args.output ?? "usage-settings-report");
const runs = Number(args.runs ?? 7);
if (!args.baseline || !args.candidate || !Number.isInteger(runs) || runs < 1) {
  throw new Error("需要 --baseline URL --candidate URL [--runs 7] [--output DIR]");
}
if (existsSync(output)) throw new Error("输出目录已存在，请选择新目录以保留旧测量");
mkdirSync(output, { recursive: true });
const hash = (value) => createHash("sha256").update(value).digest("hex");
const report = { complete: false, samples: [], browser: null,
  harnessSha256: hash(readFileSync(import.meta.filename)), lockSha256: hash(readFileSync("package-lock.json")) };
const save = () => writeFileSync(`${output}/report.json`, JSON.stringify(report, null, 2));
const browser = await chromium.launch({ executablePath: args.browser ?? "/usr/bin/chromium" });
report.browser = browser.version();

/** 内容就绪要求正确范围的实际图表与四项数值，不把外壳或旧范围算完成。 */
async function ready(page, days) {
  await page.waitForFunction((days) => {
    const root = document.querySelector(".models-usage");
    const selected = root?.querySelector(`[data-days="${days}"][aria-pressed="true"]`);
    const metrics = root?.querySelectorAll(".usage-metrics b");
    const bars = root?.querySelectorAll(".usage-chart > span");
    const count = days === 1 ? (() => {
      const start = new Date(); start.setHours(0, 0, 0, 0);
      const end = new Date(start); end.setDate(end.getDate() + 1);
      return (end - start) / 3600000;
    })() : days;
    return selected && metrics.length === 4 && [...metrics].every((e) => e.textContent.trim()) && bars.length === count;
  }, days, { timeout: 15000 });
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
}

const values = (page) => page.locator("[data-usage]").innerText();
const usageResponse = (page) => page.waitForResponse((r) => new URL(r.url()).pathname === "/api/dashboard/models/usage");

try {
  for (let pair = 0; pair < runs; pair++) for (const cpu of [1, 4]) {
    for (const profile of pair % 2 ? ["weak", "lan"] : ["lan", "weak"]) {
      for (const variant of pair % 2 ? ["candidate", "baseline"] : ["baseline", "candidate"]) {
        const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
        await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
        const page = await context.newPage();
        const setup = { pair, cpu, profile, variant, phase: "setup", success: false };
        report.samples.push(setup);
        const pageErrors = [];
        page.on("pageerror", (error) => pageErrors.push(error.message));
        try {
          const cdp = await context.newCDPSession(page);
          await cdp.send("Network.enable");
          await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
          await page.goto(args[variant]);
          await page.getByRole("button", { name: "功能设置", exact: true }).click();
          const nav = page.locator(".shell-settings-nav");
          const expected = new Map();
          for (const days of [1, 7, 30]) {
            const pending = usageResponse(page);
            if (days === 1) await nav.getByRole("button", { name: "用量", exact: true }).click();
            else await page.locator(`[data-days="${days}"]`).click();
            const response = await pending;
            if (!response.ok() || (await response.json()).complete === false) throw new Error("需要已扫描完整的真实用量基线");
            await ready(page, days);
            expected.set(days, await values(page));
          }
          await nav.getByRole("button", { name: "模型", exact: true }).click();
          setup.success = true;
          await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: profile === "lan" ? 20 : 800,
            downloadThroughput: profile === "lan" ? 12500000 : 125000, uploadThroughput: 125000 });
          for (const days of [1, 7, 30]) {
            const row = { pair, cpu, profile, variant, days, phase: days === 1 ? "reopen" : "range", success: false };
            report.samples.push(row);
            const start = performance.now();
            const pending = usageResponse(page).then(async (response) => {
              const body = await response.body();
              return { response, body, at: performance.now() - start };
            });
            if (days === 1) await nav.getByRole("button", { name: "用量", exact: true }).evaluate((e) => e.click());
            else await page.locator(`[data-days="${days}"]`).evaluate((e) => e.click());
            await ready(page, days);
            row.contentMs = performance.now() - start;
            if (await values(page) !== expected.get(days)) throw new Error("热内容与预先读到的同范围结果不一致");
            const { response, body, at } = await pending;
            row.freshMs = at;
            row.request = { url: response.url(), status: response.status(), bytes: body.length, sha256: hash(body) };
            if (!response.ok() || JSON.parse(body).complete === false) throw new Error("后台核对未成功完成");
            await ready(page, days);
            if (await values(page) !== expected.get(days) || pageErrors.length) throw new Error("后台核对改变固定输入或发生脚本错误");
            row.success = true;
            await page.screenshot({ path: `${output}/${pair}-${cpu}-${profile}-${variant}-${days}.png` });
            save();
          }
        } catch (error) {
          setup.error = String(error);
          setup.success = false;
          process.exitCode = 1;
        } finally { await context.close(); save(); }
      }
    }
  }
  report.complete = true;
} finally { await browser.close(); save(); }
