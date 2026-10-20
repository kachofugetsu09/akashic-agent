// 冷启动瀑布:HTML→各 JS 资源→ws 连接→首条 ws 数据的精确时刻。
import { chromium } from "playwright-core";

const url = process.argv[2];
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Network.enable");
await cdp.send("Network.emulateNetworkConditions", { offline: false, latency: 20, downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 });
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });

const wsEvents = [];
page.on("websocket", ws => {
  wsEvents.push({ t: performance.now(), event: "ws-open", url: ws.url().slice(-40) });
  ws.on("framereceived", () => wsEvents.push({ t: performance.now(), event: "ws-first-frame" }));
});
page.on("console", m => { if (m.text().startsWith("[perf]")) console.log("page:", m.text()); });

await page.goto(`${url}/?session=akashic%3Astartup-benchmark`, { waitUntil: "commit" });
await page.locator('[data-message-id="startup-099"] .message-row').waitFor({ timeout: 30_000 });

const report = await page.evaluate(() => {
  const nav = performance.getEntriesByType("navigation")[0];
  const resources = performance.getEntriesByType("resource")
    .filter(r => r.initiatorType !== "xmlhttprequest" || r.name.includes("bootstrap") || r.name.includes("state"))
    .map(r => ({ n: r.name.replace(location.origin, "").slice(0, 70), s: Math.round(r.startTime), e: Math.round(r.responseEnd), sz: r.transferSize }))
    .sort((a, b) => a.s - b.s);
  const longtasks = performance.getEntriesByType("longtask").map(t => [Math.round(t.startTime), Math.round(t.duration)]);
  const paint = performance.getEntriesByType("paint").map(p => [p.name, Math.round(p.startTime)]);
  return { navEnd: Math.round(nav.responseEnd), resources, longtasks, paint };
});
console.log("navEnd:", report.navEnd, "paint:", JSON.stringify(report.paint));
for (const r of report.resources) console.log(`  ${String(r.s).padStart(5)}→${String(r.e).padStart(5)}  ${String(r.sz).padStart(8)}  ${r.n}`);
console.log("longtasks:", JSON.stringify(report.longtasks));
for (const w of wsEvents.slice(0, 6)) console.log(`  ws ${Math.round(w.t)}ms ${w.event}`);
await browser.close();
