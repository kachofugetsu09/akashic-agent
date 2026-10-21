/** 会话切换延迟基准:从启动会话切到长历史会话,分解网络、渲染与绘制耗时。

配合 serve-startup-runtime.py --switch-session-messages N 灌入的
akashic:switch-benchmark 会话使用。点击用 dispatchEvent 触发,避免 hover
预取掩盖冷切换成本;每轮新 context、CPU×4 节流,口径与 measure-startup.mjs 一致。
*/
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";

const { values } = parseArgs({ options: {
  url: { type: "string" }, output: { type: "string" }, browser: { type: "string" },
  runs: { type: "string", default: "7" }, profiles: { type: "string", default: "lan,limited" },
  label: { type: "string", default: "baseline" },
  from: { type: "string", default: "akashic:startup-benchmark" },
  target: { type: "string", default: "akashic:switch-benchmark" },
  targetRows: { type: "string", default: "50" },
  // 点击前的停留毫秒:模拟真实阅读时间,让空闲预热(requestIdleCallback)有机会落地;默认 0 保持原口径。
  dwellMs: { type: "string", default: "0" },
} });
if (!values.url || !values.output || !values.browser) throw new Error("需要 --url、--output 和 --browser");
const profiles = {
  lan: { latency: 20, downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 },
  latency: { latency: 600, downloadThroughput: 12_500_000, uploadThroughput: 1_250_000 },
  limited: { latency: 300, downloadThroughput: 500_000, uploadThroughput: 125_000 },
};
const selected = values.profiles.split(",");
const runCount = Number(values.runs);
const targetRows = Number(values.targetRows);
const dwellMs = Number(values.dwellMs);
if (!Number.isSafeInteger(runCount) || runCount < 1 || selected.some(name => !profiles[name])
  || !Number.isSafeInteger(dwellMs) || dwellMs < 0) {
  throw new Error("无效的 runs、profiles 或 dwellMs");
}
const output = resolve(values.output);
mkdirSync(output, { recursive: false });
const hash = value => createHash("sha256").update(value).digest("hex");
const git = (...args) => execFileSync("git", args, { encoding: "utf8" }).trim();
const report = {
  schemaVersion: 1, label: values.label, sourceCommit: git("rev-parse", "HEAD"),
  sourceDiffSha256: hash(git("diff", "HEAD", "--", "frontend")),
  harnessSha256: hash(readFileSync(new URL(import.meta.url))),
  from: values.from, target: values.target, targetRows, dwellMs,
  profiles, runCount, viewport: { width: 412, height: 915 }, cpuRate: 4,
  timeoutMs: 45_000, startedAt: new Date().toISOString(), runs: [],
};
const browser = await chromium.launch({ executablePath: values.browser, headless: true });
report.browserVersion = browser.version();
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));

// 切换目标最后一条消息的 DOM 锚点:tail 页必含末条;消息 id 由 fixture 确定为 switch-XXXX。
const lastMessageId = await fetch(
  new URL(`/api/chat/sessions/${encodeURIComponent(values.target)}/messages?limit=1`, values.url),
).then(r => r.json()).then(page => {
  const items = page.items ?? page.messages ?? [];
  const last = items.at(-1);
  if (!last?.id) throw new Error("目标会话没有消息,先用 --switch-session-messages 灌入");
  return last.id;
});

try {
  for (let run = 0; run < runCount; run++) {
    for (let offset = 0; offset < selected.length; offset++) {
      const profile = selected[(run + offset) % selected.length];
      const context = await browser.newContext({ viewport: report.viewport, serviceWorkers: "block" });
      await context.addInitScript(() => {
        localStorage.setItem("akashic.onboarding.seen", "1");
        window.__switchLongTasks = [];
        new PerformanceObserver(list => window.__switchLongTasks.push(...list.getEntries().map(
          entry => ({ start: entry.startTime, duration: entry.duration }),
        ))).observe({ type: "longtask", buffered: true });
      });
      let result;
      try { result = await measure(context, profile); }
      catch (error) { result = { success: false, error: String(error) }; }
      report.runs.push({ run, profile, ...result });
      writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
      console.log(JSON.stringify({ run, profile, ...result }));
      await context.close();
    }
  }
} finally { await browser.close(); }

report.summary = Object.fromEntries(selected.map(profile => {
  const rows = report.runs.filter(r => r.profile === profile);
  const metric = name => {
    const nums = rows.map(r => r[name]).filter(v => typeof v === "number").sort((a, b) => a - b);
    return nums.length ? { median: nums[Math.floor(nums.length / 2)], max: nums.at(-1) } : null;
  };
  return [profile, { n: rows.length, successes: rows.filter(r => r.success).length,
    readyMs: metric("readyMs"), messagesResponseMs: metric("messagesResponseMs"),
    rowsMs: metric("rowsMs"), richMs: metric("richMs"), paintedMs: metric("paintedMs") }];
}));
writeFileSync(resolve(output, "report.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report.summary, null, 2));
if (report.runs.some(r => !r.success)) process.exitCode = 1;

async function measure(context, profile) {
  const page = await context.newPage();
  try {
    page.setDefaultTimeout(report.timeoutMs);
    const cdp = await context.newCDPSession(page);
    await cdp.send("Network.enable");
    await cdp.send("Network.emulateNetworkConditions", { offline: false, ...profiles[profile] });
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: report.cpuRate });
    const errors = [];
    page.on("pageerror", error => errors.push(error.message));
    const messagesPath = `/api/chat/sessions/${encodeURIComponent(values.target)}/messages`;
    let messagesRequestAt = null, messagesResponseAt = null, switching = false;
    const switchRequests = [];
    page.on("request", request => {
      if (!switching) return;
      const path = new URL(request.url()).pathname;
      switchRequests.push({ path, at: Math.round((performance.now() - switchStart) * 10) / 10 });
      if (path === messagesPath) messagesRequestAt ??= performance.now();
    });
    page.on("response", response => {
      if (new URL(response.url()).pathname === messagesPath && response.ok()) {
        messagesResponseAt ??= performance.now();
      }
    });
    let followResolve;
    const following = new Promise(resolve => { followResolve = resolve; });
    page.on("websocket", socket => {
      if (new URL(socket.url()).pathname !== "/ws") return;
      socket.on("framereceived", ({ payload }) => {
        const frame = JSON.parse(String(payload));
        if (frame.type === "session.following" && frame.session_id === values.from) followResolve();
        if (frame.type === "error") errors.push(`WebSocket: ${frame.message}`);
      });
    });

    // 1. 启动会话就绪(与 measure-startup 同一就绪口径的简化版)。
    const started = performance.now();
    const remaining = () => Math.max(1, report.timeoutMs - (performance.now() - started));
    await page.goto(new URL(`/?session=${encodeURIComponent(values.from)}`, values.url).href,
      { waitUntil: "commit", timeout: remaining() });
    const frame = page.locator(".conversation-page-chat .akashic-chat");
    await frame.locator('[data-message-id="startup-099"] .message-row').waitFor({ timeout: remaining() });
    await following;
    const readyMs = performance.now() - started;
    if (dwellMs > 0) await page.waitForTimeout(dwellMs);

    // 2. 点击侧边栏目标行;dispatchEvent 不经过 hover,排除预取干扰。
    //    目标会话标题回退到首条消息正文,以"切换基准消息"开头。
    const row = page.locator('button.conversation-session:has-text("切换基准消息")').first();
    // 窄视口侧边栏可能折叠,行只需存在;dispatchEvent 不要求可见。
    await row.waitFor({ state: "attached", timeout: remaining() });
    const switchStart = performance.now();
    switching = true;
    await row.dispatchEvent("click");
    const rowLocator = frame.locator(`[data-message-id="${lastMessageId}"] .message-row`);
    await rowLocator.waitFor({ timeout: remaining() });
    const rowsMs = performance.now() - switchStart;
    // 3. 富文本门槛:目标消息含围栏代码块时,等真正的富渲染升级就位。
    //    静态首帧也产出 pre(static-code-block),只有富渲染的代码块带 .reading-code-scroll。
    await rowLocator.locator(".reading-code-scroll").first().waitFor({ timeout: remaining() });
    const richMs = performance.now() - switchStart;
    // 4. 行出现后等两帧,确保 markdown 正文真正绘制。
    await page.evaluate(() => new Promise(resolve =>
      requestAnimationFrame(() => requestAnimationFrame(resolve))));
    const paintedMs = performance.now() - switchStart;
    if (errors.length) throw new Error(errors.join("; "));
    const longTasks = await page.evaluate(() => window.__switchLongTasks);
    const renderedRows = await frame.locator("[data-message-id]").count();
    return { success: true, readyMs: Math.round(readyMs * 10) / 10,
      messagesRequestMs: messagesRequestAt === null ? null : Math.round((messagesRequestAt - switchStart) * 10) / 10,
      messagesResponseMs: messagesResponseAt === null ? null : Math.round((messagesResponseAt - switchStart) * 10) / 10,
      rowsMs: Math.round(rowsMs * 10) / 10, richMs: Math.round(richMs * 10) / 10, paintedMs: Math.round(paintedMs * 10) / 10,
      renderedRows, switchRequests,
      longTasks: longTasks.filter(task => task.start * 1 >= 0)
        .map(task => Math.round(task.duration)) };
  } finally { await page.close(); }
}
