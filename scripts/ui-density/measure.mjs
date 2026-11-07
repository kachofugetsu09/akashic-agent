// 视口信息密度测量：打开页面，按步骤操作，每个命名状态输出一行 JSON 指标。
// 用法：node scripts/ui-density/measure.mjs --url <地址> [--browser /usr/bin/chromium] [--auth <secret 文件>]
//        [--device desk|phone] [--tag before] [--out <截图目录>] [--steps 'kind|arg|name;...']
// 步骤 kind：click（CSS 选择器）、text（按文字点）、button（按读屏名点按钮）、scroll（像素，负数向上）、wait（毫秒）。
// 判定方法与阈值见 docs/design/information-density.md。
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { chromium } from "playwright-core";
import { collectInk, scoreDensity } from "./density-metrics.mjs";

const { values } = parseArgs({ options: {
  url: { type: "string" }, browser: { type: "string", default: "/usr/bin/chromium" }, auth: { type: "string" },
  device: { type: "string", default: "desk" }, tag: { type: "string", default: "run" }, out: { type: "string" },
  steps: { type: "string", default: "" }, settle: { type: "string", default: "2500" },
  primary: { type: "string", default: ".agent-content, .user-bubble" },
  question: { type: "string", default: ".user-bubble" }, answer: { type: "string", default: ".flow-part" },
} });
if (!values.url) throw new Error("缺少 --url");
const phone = values.device === "phone";
const viewport = phone ? { width: 390, height: 844 } : { width: 1440, height: 900 };
const headers = values.auth ? { authorization: `Bearer ${readFileSync(values.auth, "utf8").trim()}` } : {};
const browser = await chromium.launch({ executablePath: values.browser });
const context = await browser.newContext({ viewport, deviceScaleFactor: 1, isMobile: phone, hasTouch: phone,
  extraHTTPHeaders: headers, colorScheme: "light", locale: "zh-CN" });
const page = await context.newPage();
const settle = Number(values.settle);
if (values.out) mkdirSync(values.out, { recursive: true });

// 问到答的距离：最后一个问题气泡底边到其后第一段末条回复顶边（同一页面坐标系，不受滚动影响）。
function taskDistance([question, answer]) {
  const asks = [...document.querySelectorAll(question)];
  const last = asks.at(-1);
  if (!last) return null;
  const after = [...document.querySelectorAll(answer)].find((node) => last.compareDocumentPosition(node) & Node.DOCUMENT_POSITION_FOLLOWING);
  return after ? Math.round(after.getBoundingClientRect().top - last.getBoundingClientRect().bottom) : null;
}

// iframe 在父文档里是否露出：3×3 采样点里多数命中它自己才算可见（遮罩、模态盖住时不采它的内容）。
function frameOnTop(element) {
  const r = element.getBoundingClientRect();
  let hits = 0;
  for (const fx of [0.2, 0.5, 0.8]) for (const fy of [0.2, 0.5, 0.8]) {
    const x = Math.min(innerWidth - 1, r.left + r.width * fx), y = Math.min(innerHeight - 1, r.top + r.height * fy);
    if (document.elementFromPoint(x, y) === element) hits += 1;
  }
  return hits >= 5;
}

async function measure(name) {
  // 1. 每个 frame 各自采集，按 iframe 位置平移、裁到可见区后合并；被父文档遮住的 iframe 不计入并单独报告。
  const frames = [];
  let occludedFrames = 0;
  for (const frame of page.frames()) {
    let clip = [0, 0, viewport.width, viewport.height];
    if (frame !== page.mainFrame()) {
      const element = await frame.frameElement().catch(() => null);
      const box = await element?.boundingBox();
      if (!element || !box || box.width < 2) continue;
      if (!await element.evaluate(frameOnTop)) { occludedFrames += 1; continue; }
      clip = [box.x, box.y, Math.min(box.width, viewport.width - box.x), Math.min(box.height, viewport.height - box.y)];
    }
    const data = await frame.evaluate(collectInk, values.primary).catch(() => null);
    if (data) frames.push({ clip, data });
  }
  // 2. 结构指标 + 问答距离 + JPEG 体积（Rosenholtz 杂乱度代理）。
  const metrics = scoreDensity(frames, viewport.width, viewport.height);
  const distances = await Promise.all(page.frames().map((frame) => frame.evaluate(taskDistance, [values.question, values.answer]).catch(() => null)));
  const jpeg = await page.screenshot({ type: "jpeg", quality: 75 });
  if (values.out) writeFileSync(resolve(values.out, `${values.tag}-${values.device}-${name}.jpg`), jpeg);
  console.log(JSON.stringify({ tag: values.tag, device: values.device, state: name, occludedFrames,
    taskDistance: distances.find((value) => value !== null) ?? null, ...metrics,
    clutterBpp: +((jpeg.length * 8) / (viewport.width * viewport.height)).toFixed(3) }));
}

await page.goto(values.url, { waitUntil: "domcontentloaded" });
await page.waitForTimeout(settle * 2);
await measure("load");
for (const step of values.steps.split(";").filter(Boolean)) {
  const [kind, arg, name] = step.split("|");
  if (kind === "click") await page.locator(arg).first().click();
  else if (kind === "text") await page.getByText(arg, { exact: false }).first().click();
  else if (kind === "button") await page.getByRole("button", { name: arg }).first().click();
  else if (kind === "scroll") {
    await page.mouse.move(viewport.width * 0.6, viewport.height * 0.5);
    for (let index = 0; index < 6; index += 1) await page.mouse.wheel(0, Number(arg) / 6);
  } else if (kind === "wait") await page.waitForTimeout(Number(arg));
  else throw new Error(`未知步骤 ${kind}`);
  await page.waitForTimeout(settle);
  if (name) await measure(name);
}
await browser.close();
