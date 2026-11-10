// 长滚动 DOM 累积探针：持续上滚 120 步再回滚,沿途采样行数、DOM 总量、编辑器宿主数。
// 回答:渲染窗口是滚动的(远处卸载)还是只增不减(累积)。
// 用法: node probe-dom-growth.mjs <origin> [session] [steps]
import { chromium } from "playwright-core";

const [url, target = "akashic:switch-benchmark", stepsArg = "120"] = process.argv.slice(2);
const steps = Number(stepsArg);
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const context = await browser.newContext({ viewport: { width: 412, height: 915 }, serviceWorkers: "block" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
await page.goto(`${url}/?session=${encodeURIComponent(target)}`, { waitUntil: "commit" });
await page.locator("[data-message-id] .message-row").first().waitFor({ timeout: 45_000 });
await page.waitForTimeout(3000);

const sample = () => page.evaluate(() => ({
  rows: document.querySelectorAll("[data-message-id]").length,
  dom: document.querySelectorAll("*").length,
  editors: document.querySelectorAll("diffs-container").length,
  pres: document.querySelectorAll("pre").length,
}));
const samples = [{ at: "start", ...(await sample()) }];
await page.evaluate(async (total) => {
  const row = document.querySelector("[data-message-id]");
  let scroller = row?.parentElement;
  while (scroller && scroller.scrollHeight <= scroller.clientHeight) scroller = scroller.parentElement;
  window.__scroller = scroller;
  for (let i = 0; i < total; i++) { scroller.scrollTop -= scroller.clientHeight * 0.85; await new Promise((r) => setTimeout(r, 50)); }
}, steps);
samples.push({ at: `after ${steps} up`, ...(await sample()) });
await page.waitForTimeout(3000);
samples.push({ at: "settle 3s", ...(await sample()) });
// 回滚到底
await page.evaluate(async (total) => {
  const scroller = window.__scroller;
  for (let i = 0; i < total; i++) { scroller.scrollTop += scroller.clientHeight * 0.85; await new Promise((r) => setTimeout(r, 30)); }
  scroller.scrollTop = scroller.scrollHeight;
}, steps);
await page.waitForTimeout(2000);
samples.push({ at: "back at bottom", ...(await sample()) });
console.log(JSON.stringify(samples, null, 1));
await browser.close();
