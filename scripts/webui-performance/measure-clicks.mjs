#!/usr/bin/env node
// 点击交互基线：对运行中的实例逐个回放常用点击，记录首开与重开时的界面稳定时间、请求、读取态与长任务。
//
// 用法：
//   node scripts/webui-performance/measure-clicks.mjs --origin http://127.0.0.1:5420 --storage state.json \
//     [--runs 5] [--only settings] [--out report.json] [--big "9879 条消息"] [--mid "208 条消息"]
// --storage 是 Playwright storageState（已登录的 Cookie）；--origin 建议经延迟代理，网络往返才会显形。
// --big / --mid 是侧栏里大、中两个会话行可访问名称的正则片段，按被测数据替换。方法与基线见 CLICKS.md。
// 场景只读：不发消息、不保存设置、不启停插件。
import { chromium } from "playwright-core";
import { writeFileSync } from "node:fs";

const CLICKABLE = "button, a[href], [role=button], [role=tab], [role=menuitem], [role=option], [role=switch], summary";
const ASSET = /\.(js|css|woff2?|svg|png|ico|wasm)(\?|$)/;
const LOADING_TEXT = "正在读取|读取中|加载中|载入中|正在加载|请稍候";

// 可选步骤：元素在才点（如工作台按会话记住了展开状态）
const opt = (pattern) => ({ source: pattern.source, optional: true });

// 1. 场景：setup 在整页加载后执行，act 是被测点击，back 在同一页面内回到 act 之前的状态以测重开
const scenarios = (big, mid) => [
  { id: "session.big", label: "切换到大会话（近万条消息）", setup: [/^新会话$/], act: big, back: [mid] },
  { id: "session.mid", label: "切换到中等会话", setup: [big], act: mid, back: [big] },
  { id: "session.new", label: "新会话", setup: [big], act: /^新会话$/, back: [big] },
  { id: "settings.open", label: "打开功能设置", act: /^功能设置$/, back: ["Escape"] },
  { id: "settings.models", label: "设置·模型", setup: [/^功能设置$/, /^插件$/], act: /^模型$/, back: [/^插件$/] },
  { id: "settings.cache", label: "设置·缓存效率", setup: [/^功能设置$/], act: /^缓存效率$/, back: [/^模型$/] },
  { id: "settings.plugins", label: "设置·插件", setup: [/^功能设置$/], act: /^插件$/, back: [/^模型$/] },
  { id: "settings.pluginConfig", label: "设置·插件·插件配置", setup: [/^功能设置$/, /^插件$/], act: /^插件配置$/, back: [/^插件列表$/] },
  { id: "settings.pluginCard", label: "设置·插件配置·打开插件卡片", setup: [/^功能设置$/, /^插件$/, /^插件配置$/], act: /^对话与智能体/, back: [/^模型$/, /^插件$/, opt(/^插件配置$/)] },
  { id: "settings.onboarding", label: "设置·新手引导", setup: [/^功能设置$/], act: /^新手引导$/, back: [/^模型$/] },
  { id: "focus.settings", label: "设置开在插件配置时切回窗口", setup: [/^功能设置$/, /^插件$/, /^插件配置$/], act: "focus", back: [] },
  { id: "focus.session", label: "会话页切回窗口", setup: [big], act: "focus", back: [] },
  { id: "panel.open", label: "打开插件面板", setup: [big], act: /^插件面板$/, back: [/^关闭$/] },
  { id: "panel.usage", label: "插件面板·用量", setup: [big, /^插件面板$/], act: /^用量/, back: [/^返回插件面板$/] },
  { id: "panel.usageTab", label: "用量·切到上下文标签", setup: [big, /^插件面板$/, /^用量/], act: /^上下文$/, back: [/^概览$/] },
  { id: "composer.model", label: "打开模型选择器", setup: [big], act: /^选择模型/, back: ["Escape"] },
  { id: "composer.project", label: "新会话·打开项目下拉", setup: [/^新会话$/], act: /^项目：/, back: ["Escape"] },
  { id: "workbench.open", label: "打开工作台", setup: [big, opt(/^收起工具区$/)], act: /^打开工作台/, back: [/^收起工具区$/] },
  { id: "workbench.changes", label: "工作台·改动", setup: [big, opt(/^打开工作台/), opt(/^新标签页$/)], act: /^改动/, back: [/^新标签页$/] },
  { id: "workbench.sites", label: "工作台·网站", setup: [big, opt(/^打开工作台/), opt(/^新标签页$/)], act: /^网站/, back: [/^新标签页$/] },
];

function parseArgs(argv) {
  const args = { runs: 5, out: "", only: null, origin: "", storage: "", big: "9879 条消息", mid: "208 条消息" };
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i].replace(/^--/, "");
    const value = argv[i + 1];
    if (key === "runs") args.runs = Number(value);
    else if (key === "only") args.only = new RegExp(value);
    else if (key in args) args[key] = value;
    else throw new Error(`未知参数 ${argv[i]}`);
    i += 1;
  }
  if (!args.origin || !args.storage) throw new Error("需要 --origin 与 --storage");
  return args;
}

// 2. 页面内：按可访问名称找可见的可点击元素
function installFinder(selector) {
  // Radix 菜单在 pointerdown 打开，单发 click 打不开：按真实顺序派发
  window.__press = (el) => {
    const init = { bubbles: true, cancelable: true, button: 0, pointerType: "mouse", isPrimary: true };
    el.dispatchEvent(new PointerEvent("pointerdown", init));
    el.dispatchEvent(new PointerEvent("pointerup", init));
    el.click();
  };
  window.__clickTarget = (source) => {
    const pattern = new RegExp(source);
    for (const el of document.querySelectorAll(selector)) {
      if (el.disabled || el.closest("[inert], [aria-hidden=true]")) continue;
      const rect = el.getBoundingClientRect();
      if (rect.width < 2 || rect.height < 2 || rect.bottom < 0 || rect.top > innerHeight) continue;
      const label = el.getAttribute("aria-label") || el.getAttribute("title") || el.textContent || "";
      if (pattern.test(label.replace(/\s+/g, " ").trim())) return el;
    }
    return null;
  };
}

// 3. 页面内：执行一次点击，记录 DOM 变化直到静默，期间的读取态与长任务
async function clickAndWatch({ source, quietMs, maxMs, loadingText }) {
  const target = source === "focus" ? window : window.__clickTarget(source);
  if (!target) return { missing: true };
  const loading = new RegExp(loadingText);
  const busy = "[aria-busy=true], [class*=skeleton], [class*=spinner]";
  const state = { first: null, last: 0, mutations: 0, loadingAt: null, loadingUntil: null, longTasks: 0 };
  const t0 = performance.now();
  const markLoading = (now) => { state.loadingAt ??= now - t0; state.loadingUntil = now - t0; };
  const observer = new MutationObserver((all) => {
    // 同值属性写入（如每次 scroll 都 classList.remove 一个本来没有的 class）不是界面变化
    const records = all.filter((r) => r.type !== "attributes" || r.oldValue !== r.target.getAttribute(r.attributeName));
    if (!records.length) return;
    const now = performance.now();
    state.first ??= now - t0;
    state.last = now - t0;
    state.mutations += records.length;
    for (const record of records) {
      for (const node of record.addedNodes) {
        if (node.nodeType === 1 && (node.matches(busy) || node.querySelector(busy))) markLoading(now);
        const text = node.nodeType === 3 ? node.data : node.nodeType === 1 && node.childElementCount < 20 ? node.textContent : "";
        if (text && loading.test(text)) markLoading(now);
      }
      if (record.type === "characterData" && loading.test(record.target.data)) markLoading(now);
    }
  });
  observer.observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeOldValue: true });
  const longTasks = new PerformanceObserver((list) => { for (const entry of list.getEntries()) state.longTasks += entry.duration; });
  longTasks.observe({ type: "longtask" });
  if (source === "focus") {
    // 模拟从别的窗口切回：先失焦，再依次到达可见与获得焦点
    window.dispatchEvent(new Event("blur"));
    document.dispatchEvent(new Event("visibilitychange"));
    window.dispatchEvent(new Event("focus"));
  } else {
    window.__press(target);
  }
  while (performance.now() - t0 < maxMs && (performance.now() - t0 < quietMs || performance.now() - t0 - state.last < quietMs)) {
    await new Promise((resolve) => setTimeout(resolve, 50));
  }
  observer.disconnect();
  longTasks.disconnect();
  return {
    doneMs: Math.round(state.last),
    firstChangeMs: state.first === null ? null : Math.round(state.first),
    mutations: state.mutations,
    loadingMs: state.loadingAt === null ? null : Math.round(state.loadingUntil - state.loadingAt) || 1,
    longTaskMs: Math.round(state.longTasks),
  };
}

// 4. CDP 网络记录：插件查询按 plugin.method 归类，静态资源单独计数
function recordNetwork(cdp) {
  const requests = new Map();
  const frames = [];
  cdp.on("Network.requestWillBeSent", ({ requestId, request, timestamp }) => {
    let key = new URL(request.url).pathname;
    if (key.endsWith("/plugin-ui/query") && request.postData) {
      const body = JSON.parse(request.postData);
      key = `query:${body.plugin_id}.${body.method}`;
    }
    requests.set(requestId, { key, start: timestamp, asset: ASSET.test(request.url) || request.url.startsWith("data:") });
  });
  cdp.on("Network.loadingFinished", ({ requestId, timestamp, encodedDataLength }) => {
    const entry = requests.get(requestId);
    if (entry) Object.assign(entry, { end: timestamp, bytes: encodedDataLength });
  });
  cdp.on("Network.webSocketFrameSent", ({ timestamp, response }) => {
    frames.push({ t: timestamp, type: (response.payloadData.match(/"(?:type|op)"\s*:\s*"([^"]+)"/) || [])[1] || "frame" });
  });
  return {
    mark() {
      let latest = 0;
      for (const entry of requests.values()) latest = Math.max(latest, entry.start, entry.end ?? 0);
      for (const frame of frames) latest = Math.max(latest, frame.t);
      return latest + 1e-6;
    },
    since(mark) {
      const picked = [...requests.values()].filter((entry) => entry.start >= mark);
      const data = picked.filter((entry) => !entry.asset);
      return {
        requests: data.length,
        kb: Math.round(data.reduce((sum, entry) => sum + (entry.bytes ?? 0), 0) / 1024),
        assets: picked.length - data.length,
        keys: data.map((entry) => entry.key),
        wsOut: frames.filter((frame) => frame.t >= mark).map((frame) => frame.type),
      };
    },
  };
}

async function waitQuiet(page, quietMs = 800, maxMs = 15000) {
  await page.evaluate(async ({ quietMs, maxMs }) => {
    let last = performance.now();
    const observer = new MutationObserver(() => { last = performance.now(); });
    observer.observe(document.body, { subtree: true, childList: true, attributes: true, characterData: true });
    const start = performance.now();
    while (performance.now() - last < quietMs && performance.now() - start < maxMs) await new Promise((r) => setTimeout(r, 50));
    observer.disconnect();
  }, { quietMs, maxMs });
}

async function step(page, action) {
  if (action === "Escape") await page.keyboard.press("Escape");
  else {
    const found = await page.evaluate((source) => { const el = window.__clickTarget(source); if (el) window.__press(el); return !!el; }, action.source);
    if (!found && !action.optional) throw new Error(`找不到 ${action.source}`);
    if (!found) return;
  }
  await waitQuiet(page);
}

async function measure(page, network, act) {
  const mark = network.mark();
  const result = await page.evaluate(clickAndWatch, { source: act, quietMs: 800, maxMs: 15000, loadingText: LOADING_TEXT });
  if (result.missing) throw new Error(`找不到 ${act}`);
  await page.waitForTimeout(200);
  return { ...result, ...network.since(mark) };
}

const toAction = (item) => (item === "Escape" ? item : { source: item.source, optional: item.optional === true });

// 5. 一轮：整页加载→setup→首开；back 回到原状态→重开
async function runScenario(page, network, origin, scenario) {
  await page.goto(origin + "/", { waitUntil: "domcontentloaded" });
  await waitQuiet(page, 1500, 30000);
  for (const item of scenario.setup ?? []) await step(page, toAction(item));
  const act = scenario.act === "focus" ? "focus" : scenario.act.source;
  const first = await measure(page, network, act);
  for (const item of scenario.back) await step(page, toAction(item));
  const again = await measure(page, network, act);
  return { first, again };
}

function median(values) {
  const sorted = values.filter((v) => typeof v === "number").sort((a, b) => a - b);
  return sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : null;
}

function summarize(runs) {
  const pick = (phase) => {
    const list = runs.map((run) => run[phase]);
    const keys = {};
    for (const key of list.at(-1).keys) keys[key] = (keys[key] ?? 0) + 1;
    return {
      doneMs: median(list.map((r) => r.doneMs)),
      firstChangeMs: median(list.map((r) => r.firstChangeMs)),
      requests: median(list.map((r) => r.requests)),
      kb: median(list.map((r) => r.kb)),
      assets: median(list.map((r) => r.assets)),
      loadingRuns: list.filter((r) => r.loadingMs !== null).length,
      longTaskMs: median(list.map((r) => r.longTaskMs)),
      keys,
      wsOut: list.at(-1).wsOut,
    };
  };
  return { first: pick("first"), again: pick("again") };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const browser = await chromium.launch({ executablePath: process.env.AKASHIC_PERF_CHROMIUM || undefined });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, storageState: args.storage });
  await context.addInitScript(installFinder, CLICKABLE);
  const page = await context.newPage();
  const cdp = await context.newCDPSession(page);
  await cdp.send("Network.enable");
  const network = recordNetwork(cdp);
  const report = { origin: args.origin, at: new Date().toISOString(), runs: args.runs, scenarios: {} };
  for (const scenario of scenarios(new RegExp(args.big), new RegExp(args.mid))) {
    if (args.only && !args.only.test(scenario.id)) continue;
    const runs = [];
    for (let i = 0; i < args.runs; i += 1) {
      try {
        runs.push(await runScenario(page, network, args.origin, scenario));
      } catch (error) {
        console.error(`${scenario.id} 第 ${i + 1} 轮失败：${error.message}`);
      }
    }
    if (!runs.length) continue;
    const summary = summarize(runs);
    report.scenarios[scenario.id] = { label: scenario.label, ...summary, raw: runs };
    const f = summary.first, a = summary.again;
    console.log(`${scenario.id.padEnd(22)} 首开 ${String(f.doneMs).padStart(5)}ms 请求 ${f.requests}(${f.kb}KB) 读取态 ${f.loadingRuns}/${runs.length} | 重开 ${String(a.doneMs).padStart(5)}ms 请求 ${a.requests}(${a.kb}KB) 读取态 ${a.loadingRuns}/${runs.length}`);
  }
  if (args.out) writeFileSync(args.out, `${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}

await main();
