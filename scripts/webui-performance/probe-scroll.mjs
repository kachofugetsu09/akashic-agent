// 长会话滚动性能探针:打开 2 万条会话,向上滚动 N 屏,记录帧间隔与 longtask。
// 用法: node probe-scroll.mjs <origin> [session]
import { chromium } from "playwright-core";

const [url, target = "akashic:switch-benchmark"] = process.argv.slice(2);
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const context = await browser.newContext({ viewport: { width: 412, height: 915 }, serviceWorkers: "block" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Network.enable");
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });

await page.goto(`${url}/?session=${encodeURIComponent(target)}`, { waitUntil: "commit" });
await page.locator(".conversation-page-chat .akashic-chat [data-message-id] .message-row").last().waitFor({ timeout: 45_000 });
// 等富内容增强与空闲预热落地,模拟真实阅读后再滚动。
await page.waitForTimeout(3000);

const result = await page.evaluate(async () => {
  const row = document.querySelector("[data-message-id]");
  let scroller = row?.parentElement;
  while (scroller && scroller.scrollHeight <= scroller.clientHeight) scroller = scroller.parentElement;
  if (!scroller) return { error: "no scroller" };
  // 帧间隔采样
  const gaps = [];
  let last = 0;
  let sampling = true;
  const tick = (t) => { if (last) gaps.push(t - last); last = t; if (sampling) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  const longtasks = [];
  const po = new PerformanceObserver((list) => { for (const e of list.getEntries()) longtasks.push(Math.round(e.duration)); });
  po.observe({ entryTypes: ["longtask"] });
  const pagesBefore = document.querySelectorAll("[data-message-id]").length;
  const t0 = performance.now();
  // 每 60ms 向上滚 0.85 屏,共 30 步(覆盖 ~25 屏,触发多次翻页与批量升级)。
  for (let i = 0; i < 30; i++) {
    scroller.scrollTop -= scroller.clientHeight * 0.85;
    await new Promise(r => setTimeout(r, 60));
  }
  await new Promise(r => setTimeout(r, 1500));
  sampling = false;
  const pagesAfter = document.querySelectorAll("[data-message-id]").length;
  gaps.sort((a, b) => a - b);
  const q = (p) => Math.round(gaps[Math.floor(gaps.length * p)] ?? 0);
  return {
    scrollMs: Math.round(performance.now() - t0),
    frames: gaps.length,
    frameP50: q(0.5), frameP95: q(0.95), frameMax: Math.round(gaps.at(-1) ?? 0),
    longtasks, longtaskTotal: longtasks.reduce((a, b) => a + b, 0),
    rowsBefore: pagesBefore, rowsAfter: pagesAfter,
  };
});
console.log(JSON.stringify(result, null, 1));
await browser.close();
