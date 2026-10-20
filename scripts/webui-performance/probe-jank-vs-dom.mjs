// 累积量 vs 帧率:分阶段上滚,每阶段采帧间隔,看 P95 是否随 DOM 累积恶化。
import { chromium } from "playwright-core";
const url = process.argv[2];
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const context = await browser.newContext({ viewport: { width: 412, height: 915 }, serviceWorkers: "block" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
await page.goto(`${url}/?session=${encodeURIComponent("akashic:switch-benchmark")}`, { waitUntil: "commit" });
await page.locator("[data-message-id] .message-row").first().waitFor({ timeout: 45_000 });
await page.waitForTimeout(4000);
await page.evaluate(() => {
  const row = document.querySelector("[data-message-id]");
  let scroller = row?.parentElement;
  while (scroller && scroller.scrollHeight <= scroller.clientHeight) scroller = scroller.parentElement;
  window.__scroller = scroller;
});
const phases = [];
for (let phase = 0; phase < 4; phase++) {
  const r = await page.evaluate(async () => {
    const scroller = window.__scroller;
    const gaps = [];
    let last = 0; let sampling = true;
    const tick = (t) => { if (last) gaps.push(t - last); last = t; if (sampling) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
    for (let i = 0; i < 25; i++) { scroller.scrollTop -= scroller.clientHeight * 0.85; await new Promise((r2) => setTimeout(r2, 50)); }
    await new Promise((r2) => setTimeout(r2, 1200));
    sampling = false;
    gaps.sort((a, b) => a - b);
    return {
      rows: document.querySelectorAll("[data-message-id]").length,
      dom: document.querySelectorAll("*").length,
      editors: document.querySelectorAll("diffs-container").length,
      frameP50: Math.round(gaps[Math.floor(gaps.length * 0.5)] ?? 0),
      frameP95: Math.round(gaps[Math.floor(gaps.length * 0.95)] ?? 0),
      frameMax: Math.round(gaps.at(-1) ?? 0),
    };
  });
  phases.push({ phase, ...r });
}
console.log(JSON.stringify(phases, null, 1));
await browser.close();
