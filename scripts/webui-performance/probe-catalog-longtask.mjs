// 目录全量渲染成本：从导航开始采 15s longtask,覆盖目录加载收尾的大渲染。
import { chromium } from "playwright-core";
const url = process.argv[2];
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
await page.addInitScript(() => {
  window.__lt = [];
  new PerformanceObserver((list) => { for (const e of list.getEntries()) window.__lt.push(Math.round(e.startTime) + "+" + Math.round(e.duration)); }).observe({ entryTypes: ["longtask"] });
});
await page.goto(`${url}/?session=${encodeURIComponent("akashic:startup-benchmark")}`, { waitUntil: "commit" });
await page.waitForTimeout(15000);
const lt = await page.evaluate(() => window.__lt);
const durations = lt.map((s) => Number(s.split("+")[1]));
console.log(JSON.stringify({ count: lt.length, total: durations.reduce((a, b) => a + b, 0), max: Math.max(...durations), top10: durations.sort((a, b) => b - a).slice(0, 10) }));
await browser.close();
