import { chromium } from "playwright-core";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import assert from "node:assert/strict";

const [origin, destination = "usage-settings-behavior"] = process.argv.slice(2);
if (!origin) throw new Error("需要隔离实例 URL 和可选输出目录");
const output = resolve(destination);
if (existsSync(output)) throw new Error("输出目录已存在，请选择新目录以保留旧验证");
mkdirSync(output, { recursive: true });
const report = { success: false, checks: [] };
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium" });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, timezoneId: "Asia/Shanghai" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
await page.clock.install({ time: new Date("2026-10-31T12:00:00+08:00") });
const endpoint = "**/api/dashboard/models/usage?*";
const status = page.locator("[data-usage-status]");
const bars = () => page.locator(".usage-chart > span").count();
const expectedBars = (days) => days === 1 ? 24 : days;

/** 暂扣真实请求，显式释放后仍访问真实后端，不生成成功回包。 */
async function gate() {
  const requested = page.waitForRequest((r) => new URL(r.url()).pathname === "/api/dashboard/models/usage")
    .then((request) => request.url());
  let release;
  const released = new Promise((done) => { release = done; });
  await page.route(endpoint, async (route) => { await released; await route.continue(); }, { times: 1 });
  return { requested, release };
}

async function ready(days) {
  await page.waitForFunction(({ days, count }) =>
    document.querySelector(`[data-days="${days}"][aria-pressed="true"]`)
      && document.querySelectorAll(".usage-chart > span").length === count,
  { days, count: expectedBars(days) });
}

async function select(days) {
  const pending = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/dashboard/models/usage");
  await page.locator(`[data-days="${days}"]`).click();
  const response = await pending;
  assert.equal(response.status(), 200);
  assert.equal((await response.json()).complete, true);
  await ready(days);
}

try {
  await page.goto(origin);
  await page.getByRole("button", { name: "功能设置", exact: true }).click();
  const nav = page.locator(".shell-settings-nav");
  // 1. 首读没有旧值；进度必须在 Token 区可见，不能误写到隐藏额度区。
  const cold = await gate();
  await nav.getByRole("button", { name: "用量", exact: true }).click();
  await cold.requested;
  assert.equal(await bars(), 0);
  assert.equal(await status.isVisible(), true);
  assert.match(await status.innerText(), /正在读取用量/);
  cold.release();
  await ready(1);
  report.checks.push("cold-read-status-visible");

  // 2. 没读过的范围不显示另一范围的数据；已有范围在真实 GET 等待时仍完整显示。
  const unseen = await gate();
  await page.locator('[data-days="7"]').click();
  await unseen.requested;
  assert.equal(await bars(), 0);
  unseen.release();
  await ready(7);
  await select(30);
  await nav.getByRole("button", { name: "模型", exact: true }).click();
  const hot = await gate();
  await nav.getByRole("button", { name: "用量", exact: true }).click();
  await hot.requested;
  await ready(1);
  assert.equal(await status.textContent(), "");
  const hotBox = await page.locator("[data-usage]").boundingBox();
  const hotResponse = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/dashboard/models/usage");
  hot.release();
  await (await hotResponse).finished();
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  assert.deepEqual(await page.locator("[data-usage]").boundingBox(), hotBox);
  report.checks.push("range-cache-and-hot-reopen");

  // 3. 迟到的 7 天回包不覆盖已经选中的 30 天。
  const late = await gate();
  await page.locator('[data-days="7"]').click();
  await late.requested;
  await ready(7);
  await select(30);
  const oldFinished = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/dashboard/models/usage");
  late.release();
  await (await oldFinished).finished();
  await page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
  await ready(30);
  report.checks.push("late-range-read-ignored");

  // 4. 故障注入只返回真实失败；原值保留、失败可见，重试访问真实后端。
  await page.route(endpoint, (route) => route.fulfill({ status: 503, body: "isolated fault" }), { times: 1 });
  await page.locator('[data-days="1"]').click();
  await page.waitForFunction(() => document.querySelector("[data-usage-status]").textContent.includes("503"));
  assert.equal(await status.isVisible(), true);
  await ready(1);
  await select(1);
  assert.equal(await status.innerText(), "");
  report.checks.push("failure-preserves-value-and-real-retry");

  // 5. 仅推进浏览器日历，不推进后台时间；旧日历范围不能当作今日数据。
  await page.clock.setSystemTime(new Date("2026-11-01T12:00:00+08:00"));
  const tomorrow = await gate();
  await page.locator('[data-days="1"]').click();
  const url = new URL(await tomorrow.requested);
  assert.equal(url.searchParams.get("since"), "2026-10-31T16:00:00.000Z");
  assert.equal(await bars(), 0);
  tomorrow.release();
  await ready(1);
  report.checks.push("calendar-boundary-invalidates-old-range");
  await page.screenshot({ path: `${output}/final.png` });
  report.success = true;
} catch (error) { report.error = String(error); process.exitCode = 1; }
finally { await browser.close(); writeFileSync(`${output}/report.json`, JSON.stringify(report, null, 2)); }
