// 字体请求取证：列出一次冷启动里全部 woff2 请求的路径、来源与缓存命中。
// 用途:验证阅读字体只经 /dashboard/assets 取一份(见 frontend/chat/vite.config.ts 的 shareShellFonts)。
// 用法: node scripts/webui-performance/probe-fonts.mjs --url http://127.0.0.1:PORT/ --browser /usr/bin/chromium
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string" },
} });
if (!values.url || !values.browser) throw new Error("需要 --url 和 --browser");

const browser = await chromium.launch({ executablePath: values.browser });
const context = await browser.newContext();
const page = await context.newPage();
const cdp = await context.newCDPSession(page);
await cdp.send("Network.enable");
const seen = new Map();
cdp.on("Network.requestWillBeSent", event => {
  if (!event.request.url.includes("woff2")) return;
  seen.set(event.requestId, { url: new URL(event.request.url).pathname });
});
cdp.on("Network.responseReceived", event => {
  const item = seen.get(event.requestId);
  if (item) Object.assign(item, {
    status: event.response.status,
    fromDiskCache: event.response.fromDiskCache,
  });
});
await page.goto(values.url, { waitUntil: "commit" });
await page.locator(".akashic-chat").waitFor({ timeout: 30000 });
await page.waitForFunction(() => document.fonts.status === "loaded");
// 判据:同一份字体(内容寻址文件名,与目录无关)在一次冷启动里只真实网络下载一次;
// 缓存命中(fromDiskCache)与回退失败(4xx/5xx)单独列出,回退触发即失败——那是哈希漂移的形态。
const downloads = new Map();
const failures = [];
for (const item of seen.values()) {
  console.log(JSON.stringify(item));
  if (item.status === undefined) continue;
  if (item.status >= 400) failures.push(item);
  if (item.status === 200 && item.fromDiskCache === false) {
    const name = item.url.split("/").pop();
    downloads.set(name, (downloads.get(name) ?? 0) + 1);
  }
}
const repeated = [...downloads.entries()].filter(([, count]) => count > 1);
if (repeated.length > 0) {
  console.error(`同一字体被多次网络下载: ${repeated.map(([name, count]) => `${name}×${count}`).join(", ")}`);
  process.exitCode = 1;
}
if (failures.length > 0) {
  console.error(`字体请求失败(哈希漂移回退?): ${failures.map(item => `${item.status} ${item.url}`).join(", ")}`);
  process.exitCode = 1;
}
await browser.close();
