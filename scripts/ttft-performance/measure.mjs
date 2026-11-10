// 发消息→模型收到请求（TTFT 前段）：按下回车到 fixture 记录到本轮标记的墙钟，以及期间 gateway 的磁盘读与 CPU。
// 用法见 README.md；实例必须已把会话模型切到 fixture，否则不发送。
import { chromium } from "playwright-core";
import { execSync } from "node:child_process";
import { appendFileSync, existsSync, readFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values: args } = parseArgs({ options: {
  origin: { type: "string" }, storage: { type: "string" }, session: { type: "string" },
  "fixture-log": { type: "string" }, runs: { type: "string", default: "3" }, gap: { type: "string", default: "20000" },
  "gateway-pid": { type: "string" }, "before-first": { type: "string" }, label: { type: "string", default: "" },
  out: { type: "string" }, chrome: { type: "string" },
} });
for (const key of ["origin", "storage", "session", "fixture-log"]) {
  if (!args[key]) throw new Error(`缺少 --${key}`);
}
const proc = (pid, file) => readFileSync(`/proc/${pid}/${file}`, "utf8");
const ticks = (pid) => { const f = proc(pid, "stat").split(")").pop().trim().split(/\s+/); return +f[11] + +f[12]; };
const readBytes = (pid) => +proc(pid, "io").match(/read_bytes: (\d+)/)[1];
const arrivals = () => existsSync(args["fixture-log"])
  ? readFileSync(args["fixture-log"], "utf8").split("\n").filter(Boolean).map((line) => JSON.parse(line)) : [];
const replies = (page) => page.evaluate(() => (document.body.innerText.match(/模拟回复：TTFT/g) || []).length);

const browser = await chromium.launch(args.chrome ? { executablePath: args.chrome } : {});
const page = await (await browser.newContext({ storageState: args.storage, viewport: { width: 1440, height: 900 } })).newPage();
await page.goto(`${args.origin}/chat?session=${encodeURIComponent(args.session)}`);
await page.waitForTimeout(8000);
const chosen = (await page.getByRole("button", { name: /^选择模型/ }).first().innerText()).trim();
if (!/fixture/i.test(chosen)) {
  console.error(`会话模型不是 fixture（${chosen}），不发送`);
  await browser.close();
  process.exit(1);
}
const pid = args["gateway-pid"];
const results = [];
for (let run = 0; run < Number(args.runs); run++) {
  if (run === 0 && args["before-first"]) execSync(args["before-first"], { stdio: "inherit" });
  const marker = `TTFT-${Date.now()}`;
  const seen = await replies(page);
  const before = pid ? { cpu: ticks(pid), read: readBytes(pid) } : null;
  const box = page.locator("textarea").first();
  await box.fill(marker);
  const sent = Date.now();
  await box.press("Enter");
  let hit;
  for (let i = 0; i < 1200 && !hit; i++) {
    await page.waitForTimeout(50);
    hit = arrivals().find((row) => row.marker === marker);
  }
  const row = {
    label: args.label, run: run + 1, at: new Date(sent).toISOString(),
    ttft_ms: hit ? Math.round(hit.arrived * 1000 - sent) : null,
    request_messages: hit?.messages ?? null, request_bytes: hit?.bytes ?? null,
    ...(before && { cpu_s: (ticks(pid) - before.cpu) / 100, read_mb: +((readBytes(pid) - before.read) / 1e6).toFixed(1) }),
  };
  results.push(row);
  console.log(JSON.stringify(row));
  if (args.out) appendFileSync(args.out, JSON.stringify(row) + "\n");
  await page.waitForFunction((n) => (document.body.innerText.match(/模拟回复：TTFT/g) || []).length > n, seen, { timeout: 120000 }).catch(() => {});
  await page.waitForTimeout(Number(args.gap));
}
await browser.close();
