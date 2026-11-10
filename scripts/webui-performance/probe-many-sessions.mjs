// 目录规模探针：启动后观察侧栏会话行数、目录请求次数与长任务。
// 用法: node probe-many-sessions.mjs <origin>
import { chromium } from "playwright-core";

const url = process.argv[2];
const browser = await chromium.launch({ executablePath: "/usr/bin/chromium", args: ["--no-sandbox", "--disable-dev-shm-usage"] });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, serviceWorkers: "block" });
await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Network.enable");
await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
const sessionRequests = [];
cdp.on("Network.requestWillBeSent", (event) => {
  const path = new URL(event.request.url).pathname + (new URL(event.request.url).search || "");
  if (path.startsWith("/api/chat/sessions")) sessionRequests.push(path);
});
await page.goto(`${url}/?session=${encodeURIComponent("akashic:startup-benchmark")}`, { waitUntil: "commit" });
const t0 = Date.now();
await page.locator("[data-message-id] .message-row").first().waitFor({ timeout: 45_000 });
const readyMs = Date.now() - t0;
// 等目录加载稳定：连续 3 次 500ms 采样行数不变即认为加载完。
let rows = -1; let stable = 0; let catalogMs = 0;
for (let i = 0; i < 120; i++) {
  const current = await page.locator(".chat-sidebar [data-row-primary], .chat-sidebar [data-session-key]").count();
  if (current === rows) { stable += 1; } else { stable = 0; rows = current; }
  if (stable >= 3) { catalogMs = Date.now() - t0; break; }
  await page.waitForTimeout(500);
}
const longtasks = await page.evaluate(() => new Promise((resolve) => {
  const entries = [];
  try {
    const po = new PerformanceObserver((list) => { for (const e of list.getEntries()) entries.push(Math.round(e.duration)); });
    po.observe({ entryTypes: ["longtask"] });
  } catch { resolve([]); return; }
  setTimeout(() => resolve(entries), 3000);
}));
const domNodes = await page.evaluate(() => document.querySelectorAll("*").length);
console.log(JSON.stringify({
  readyMs, catalogMs, sidebarRows: rows, domNodes,
  sessionRequestCount: sessionRequests.length,
  sessionRequests: sessionRequests.slice(0, 3),
  postReadyLongtasks: longtasks, postReadyLongtaskTotal: longtasks.reduce((a, b) => a + b, 0),
}));
await browser.close();
