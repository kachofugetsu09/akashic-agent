/**
 * 动效运行时探针：在真实页面上跑一段交互剧本，从 Chrome trace 读出两类数据，补静态检查看不到的部分。
 * - 没上合成线程的动画：Animation 事件的 compositeFailed 位（含原因），例如 SVG 上的独立 rotate、布局动画把同批动画拖回主线程；
 * - 掉帧：FrameSequenceTrackerV3 按主线程/合成线程分开的期望帧与掉帧。
 * 用法：node scripts/motion-probe.mjs --url http://127.0.0.1:4732/ --scenario ./my-scenario.mjs [--throttle 4] [--chromium /usr/bin/chromium]
 * 剧本模块导出 `prepare(page)`（可选，录制前执行，如登录、打开会话）和 `run(page)`（录制中执行的交互）。
 * 颜色类属性只重绘，按 design.md §2.3 允许留在主线程，不计入失败。
 */
import { chromium } from "playwright-core";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";

const REASONS = {
  0: "AcceleratedAnimationsDisabled", 1: "EffectSuppressedByDevtools", 2: "InvalidAnimationOrEffect",
  3: "UnsupportedTimingParameters", 4: "NonReplaceCompositeMode", 5: "TargetHasInvalidCompositingState",
  6: "TargetHasIncompatibleAnimations", 7: "TargetHasCSSOffset", 10: "TransformCannotBeAcceleratedOnTarget",
  12: "FilterMayMovePixels", 13: "UnsupportedCSSProperty", 15: "MixedKeyframeValueTypes",
  16: "TimelineSourceInvalidCompositingState", 17: "NoVisibleChange", 18: "AffectsImportantProperty",
  19: "SVGTargetHasIndependentTransformProperty", 20: "NonReplaceIterationCompositeMode", 21: "UnsupportedInheritance",
};
// 没有可见变化（17）不算问题：Chrome 只是没必要把它交给合成线程。
const BENIGN = 1 << 17;
const PAINT_ONLY = /^(color|background-color|border(-[a-z]+)*-color|outline-color|text-decoration-color|caret-color|fill|stroke)$/;

function args() {
  const out = { throttle: 4, chromium: process.env.CHROMIUM ?? "/usr/bin/chromium" };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i += 2) out[argv[i].replace(/^--/, "")] = argv[i + 1];
  if (!out.url || !out.scenario) throw new Error("需要 --url 与 --scenario");
  return out;
}

// 表外的位保留成 bit:N，不吞掉。
const decode = (mask) => Array.from({ length: 31 }, (_, bit) => bit).filter((bit) => mask & (1 << bit)).map((bit) => REASONS[bit] ?? `bit:${bit}`);

function analyze(events) {
  const names = new Map();
  for (const event of events) {
    const data = event.args?.data;
    if (event.name === "Animation" && event.ph === "b" && data?.displayName) names.set(event.id2?.local ?? event.id, data);
  }
  const failures = new Map();
  for (const event of events) {
    const mask = event.name === "Animation" ? event.args?.data?.compositeFailed : 0;
    if (!mask || mask === BENIGN) continue;
    const info = names.get(event.id2?.local ?? event.id) ?? {};
    const properties = event.args.data.unsupportedProperties?.length ? event.args.data.unsupportedProperties : [info.displayName ?? "?"];
    if (properties.every((name) => PAINT_ONLY.test(name))) continue;
    failures.set(`${properties.join("+")} @ ${info.nodeName ?? "?"}`, decode(mask));
  }
  const frames = {};
  for (const event of events) {
    const data = event.args?.args?.data;
    if (event.name !== "FrameSequenceTrackerV3" || event.ph !== "b" || !data) continue;
    const slot = (frames[event.args.name] ??= { expected: 0, dropped: 0 });
    slot.expected += data.expected ?? 0;
    slot.dropped += data.dropped_v4 ?? data.dropped ?? 0;
  }
  return { failures, frames };
}

async function main() {
  const options = args();
  const scenario = await import(pathToFileURL(resolve(options.scenario)).href);
  const browser = await chromium.launch({ executablePath: options.chromium });
  try {
    const page = await browser.newPage({ viewport: { width: Number(options.width ?? 1280), height: Number(options.height ?? 860) } });
    await page.goto(options.url);
    await scenario.prepare?.(page);
    const cdp = await page.context().newCDPSession(page);
    await cdp.send("Emulation.setCPUThrottlingRate", { rate: Number(options.throttle) });
    const events = [];
    cdp.on("Tracing.dataCollected", (message) => events.push(...message.value));
    const complete = new Promise((done) => cdp.once("Tracing.tracingComplete", done));
    await cdp.send("Tracing.start", { transferMode: "ReportEvents", traceConfig: { includedCategories: ["devtools.timeline", "blink.animations", "cc", "benchmark"] } });
    await scenario.run(page);
    await cdp.send("Tracing.end");
    await complete;
    const { failures, frames } = analyze(events);
    for (const [type, slot] of Object.entries(frames)) console.log(`${type}: 掉帧 ${slot.dropped}/${slot.expected}`);
    for (const [where, reasons] of failures) console.error(`未上合成线程：${where}（${reasons.join(", ")}）`);
    if (failures.size) process.exitCode = 1;
    else console.log("所有动画都在合成线程播放。");
  } finally {
    await browser.close();
  }
}

await main();
