// 三次访问探针:验证投机 URL import 的稳态收益(第二次访问建立 code cache,第三次命中)。
import { chromium } from "playwright-core";

const url = process.argv[2];
const browser = await chromium.launch({
  executablePath: "/usr/bin/chromium",
  args: ["--no-sandbox", "--disable-dev-shm-usage"],
});

for (const profile of [
  { name: "lan", latency: 20, downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 },
  { name: "limited", latency: 300, downloadThroughput: 500_000, uploadThroughput: 125_000 },
]) {
  const context = await browser.newContext({
    viewport: { width: 1440, height: 900 }, serviceWorkers: "block",
  });
  await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
  for (let visit = 1; visit <= 3; visit++) {
    const page = await context.newPage();
    const cdp = await context.newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.emulateNetworkConditions", {
      offline: false, latency: profile.latency,
      downloadThroughput: profile.downloadThroughput,
      uploadThroughput: profile.uploadThroughput,
    });
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: 4 });
    const t0 = Date.now();
    await page.goto(`${url}/?session=akashic%3Astartup-benchmark`, { waitUntil: "commit" });
    try {
      await page.locator('[data-message-id="startup-099"] .message-row').waitFor({ timeout: 30_000 });
      console.log(`${profile.name} visit${visit}: ready=${Date.now() - t0}ms`);
    } catch {
      console.log(`${profile.name} visit${visit}: TIMEOUT`);
    }
    await page.close();
  }
  await context.close();
}
await browser.close();
