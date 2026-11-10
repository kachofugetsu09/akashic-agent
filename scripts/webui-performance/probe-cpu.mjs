// CPU profile 探针:启动全程采样,按函数自时间排序输出。
import { writeFileSync } from "node:fs";
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
await cdp.send("Profiler.enable");
await cdp.send("Profiler.setSamplingInterval", { interval: 500 });
await cdp.send("Profiler.start");

await page.goto(`${url}/?session=akashic%3Astartup-benchmark`, { waitUntil: "commit" });
await page.locator('[data-message-id="startup-099"] .message-row').waitFor({ timeout: 30_000 });
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
const ms = (n) => Math.round(n * (profile.timeDeltas[0] ?? 500) / 1000);
console.log(`samples: ${total}`);
for (const [key, n] of [...selfTime.entries()].sort((a, b) => b[1] - a[1]).slice(0, 30)) {
  console.log(`${String(Math.round(n / total * 100)).padStart(3)}% ${String(n).padStart(5)}  ${key}`);
}
writeFileSync("/tmp/boot-perf/cpu-profile-latest.json", JSON.stringify(profile));
await browser.close();
