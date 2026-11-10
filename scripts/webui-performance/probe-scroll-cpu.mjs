// 滚动窗口 CPU 归因:打开 2 万条会话,滚动期间全程 Profiler 采样,按函数自时间排序。
// 用法: node probe-scroll-cpu.mjs <origin> [session]
import { writeFileSync } from "node:fs";
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
await page.waitForTimeout(3000);

await cdp.send("Profiler.enable");
await cdp.send("Profiler.setSamplingInterval", { interval: 500 });
await cdp.send("Profiler.start");
await page.evaluate(async () => {
  const row = document.querySelector("[data-message-id]");
  let scroller = row?.parentElement;
  while (scroller && scroller.scrollHeight <= scroller.clientHeight) scroller = scroller.parentElement;
  for (let i = 0; i < 30; i++) {
    scroller.scrollTop -= scroller.clientHeight * 0.85;
    await new Promise(r => setTimeout(r, 60));
  }
  await new Promise(r => setTimeout(r, 1500));
});
const { profile } = await cdp.send("Profiler.stop");

const selfTime = new Map();
const nodes = new Map(profile.nodes.map(n => [n.id, n]));
const total = profile.samples.length;
for (const id of profile.samples) {
  const node = nodes.get(id);
  if (!node) continue;
  const f = node.callFrame;
  const key = `${f.functionName || "(anon)"} @${(f.url || "vm").split("/").slice(-1)[0]}:${f.lineNumber}`;
  selfTime.set(key, (selfTime.get(key) ?? 0) + 1);
}
console.log(`samples: ${total}`);
for (const [key, n] of [...selfTime.entries()].sort((a, b) => b[1] - a[1]).slice(0, 25)) {
  console.log(`${String(Math.round(n / total * 100)).padStart(3)}% ${String(n).padStart(5)}  ${key}`);
}
writeFileSync("/tmp/boot-perf/scroll-profile.json", JSON.stringify(profile));
await browser.close();
