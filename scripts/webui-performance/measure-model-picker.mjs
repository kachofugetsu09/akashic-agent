/** 真实目录与选择器的交错 A/B；从输入事件计时，正文与动画完成分别记录。 */
import { chromium } from "playwright-core";
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";

const { values } = parseArgs({ options: {
  baseline: { type: "string" }, candidate: { type: "string" }, browser: { type: "string" }, output: { type: "string" },
  runs: { type: "string", default: "7" }, models: { type: "string", default: "64" },
} });
for (const name of ["baseline", "candidate", "browser", "output"]) if (!values[name]) throw new Error(`需要 --${name}`);
const rounds = Number(values.runs), count = Number(values.models);
if (![rounds, count].every(value => Number.isSafeInteger(value) && value > 0)) throw new Error("runs、models 必须是正整数");
mkdirSync(values.output);
const profiles = {
  lan: { latency: 20, downloadThroughput: 12_500_000, uploadThroughput: 12_500_000 },
  weak: { latency: 800, downloadThroughput: 125_000, uploadThroughput: 125_000 },
};
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const report = { complete: false, plannedSamples: rounds * 2 * 2 * 2 * 2 * 6, samples: [], builds: {}, profiles,
  harnessSha256: hash(readFileSync(import.meta.filename)), lockSha256: hash(readFileSync("package-lock.json")) };
const save = () => writeFileSync(`${values.output}/report.json`, JSON.stringify(report, null, 2));
const browser = await chromium.launch({ executablePath: values.browser });
report.browser = browser.version();
let currentPage, frozenNames;

/** 在浏览器内从实际 pointerdown/input/keydown 事件计时，避免把自动化等待算进响应。 */
async function arm(page, action, names, query) {
  await page.evaluate(({ action, names, query }) => {
    window.__pickerResult = new Promise((resolve, reject) => {
      let start, done = false;
      const eventName = action === "open" ? "pointerdown" : action.endsWith("search") ? "input" : "keydown";
      const begin = () => { start = performance.now(); check(); };
      const observer = new MutationObserver(check);
      const stop = () => { observer.disconnect(); document.removeEventListener(eventName, begin, true); clearTimeout(timer); };
      const timer = setTimeout(() => { stop(); reject(new Error("选择器完成条件超时")); }, 15_000);
      function check() {
        if (start === undefined || done) return;
        const panel = document.querySelector('.model-picker__panel');
        if (action === "close") {
          if (panel || document.activeElement !== document.querySelector('.model-picker__trigger')) return;
        } else {
          if (!panel || panel.querySelector('input')?.value !== query) return;
          const actual = [...panel.querySelectorAll('.model-picker__row .model-picker__copy strong')].map(el => el.textContent).sort();
          if (JSON.stringify(actual) !== JSON.stringify(names)) return;
          if (!names.length && !panel.textContent.includes("没有匹配的模型")) return;
        }
        done = true;
        observer.disconnect();
        requestAnimationFrame(() => requestAnimationFrame(async () => {
          const contentMs = performance.now() - start;
          if (panel) {
            const rect = panel.getBoundingClientRect();
            if (!panel.isConnected || rect.width <= 0 || rect.height <= 0 || rect.bottom <= 0 || rect.top >= innerHeight) {
              stop(); reject(new Error("面板没有可见正文")); return;
            }
            await Promise.all(panel.getAnimations().map(animation => animation.finished));
          }
          requestAnimationFrame(() => requestAnimationFrame(() => { stop(); resolve({ contentMs, settledMs: performance.now() - start }); }));
        }));
      }
      document.addEventListener(eventName, begin, { capture: true, once: true });
      observer.observe(document.body, { subtree: true, childList: true, characterData: true });
    });
  }, { action, names, query });
}

try {
  // 1. 每种设备独立预热，两份目录必须完全相同；真实输入事件与网络仿真在预热后开始。
  for (const width of [1440, 390]) for (const cpu of [1, 4]) {
    const pages = {};
    for (const kind of ["baseline", "candidate"]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, hasTouch: width === 390,
        isMobile: width === 390, serviceWorkers: "block" });
      await context.addInitScript(() => localStorage.setItem("akashic.onboarding.seen", "1"));
      const page = await context.newPage(), cdp = await context.newCDPSession(page), errors = [], served = [];
      currentPage = page;
      page.on("pageerror", error => errors.push(error.message));
      page.on("response", response => {
        const url = new URL(response.url());
        if (url.pathname.endsWith('/plugin-ui/asset') && url.searchParams.get('plugin_id') === 'models' && url.searchParams.get('kind') === 'module') {
          served.push(response.body().then(body => {
            if (hash(body) !== url.searchParams.get('sha256')) throw new Error("实际模型模块字节与目录摘要不符");
            if (report.builds[kind] && report.builds[kind] !== hash(body)) throw new Error("被测模块在运行中变化");
            report.builds[kind] = hash(body);
          }).catch(error => { errors.push(String(error)); }));
        }
      });
      await cdp.send("Network.enable"); await cdp.send("Performance.enable");
      await cdp.send("Emulation.setCPUThrottlingRate", { rate: cpu });
      const response = page.waitForResponse(r => r.url().endsWith('/api/chat/plugin-ui/query')
        && r.request().postDataJSON()?.method === 'chat.models' && r.status() === 200);
      await page.goto(values[kind]);
      await page.getByRole('textbox', { name: '消息', exact: true }).click();
      const catalog = await (await response).json();
      const names = catalog.runtimes.map(model => model.model).sort();
      if (names.length !== count) throw new Error("实际模型目录数量不符");
      if (frozenNames && JSON.stringify(names) !== JSON.stringify(frozenNames)) throw new Error("目录在 A/B 或设备之间变化");
      frozenNames = names;
      const trigger = page.locator('.model-picker__trigger'), panel = page.getByRole('dialog', { name: '选择模型', exact: true });
      await trigger.click(); await panel.waitFor();
      await panel.getByRole('textbox', { name: '筛选模型', exact: true }).press('Escape');
      await page.evaluate(() => document.fonts.ready); await page.waitForTimeout(21_000); await page.waitForLoadState('networkidle');
      await Promise.all(served);
      if (!report.builds[kind] || errors.length) throw new Error(errors.join('\n') || '未核对模型模块');
      pages[kind] = { page, cdp, context, trigger, panel, names, errors, catalog: catalog.runtimes };
    }
    // 2. 相邻 A/B 轮换，网络顺序轮换；保留所有动作，清空搜索不能只测收缩列表。
    for (let pair = 0; pair < rounds; pair++) for (const profile of (pair % 2 ? ['weak', 'lan'] : ['lan', 'weak'])) {
      for (const kind of (pair % 2 ? ['candidate', 'baseline'] : ['baseline', 'candidate'])) {
        const { page, cdp, trigger, panel, names, errors, catalog } = pages[kind]; currentPage = page;
        await page.bringToFront(); await page.waitForLoadState('networkidle');
        await cdp.send('Network.emulateNetworkConditions', { offline: false, ...profiles[profile] });
        const needle = names[0].toLowerCase(), search = panel.getByRole('textbox', { name: '筛选模型', exact: true });
        const matches = catalog.filter(model => `${model.model} ${model.sourceName}`.toLowerCase().includes(needle)).map(model => model.model).sort();
        const actions = [
          ['open', () => trigger.click(), names, ''], ['search', () => search.fill(needle), matches, needle],
          ['clear', () => search.press('Escape'), names, ''],
          ['empty-search', () => search.fill('no-model-match-20261031'), [], 'no-model-match-20261031'],
          ['clear-empty', () => search.press('Escape'), names, ''], ['close', () => search.press('Escape'), [], ''],
        ];
        for (const [action, run, expected, query] of actions) {
          await page.waitForLoadState('networkidle');
          const row = { width, cpu, profile, pair, kind, action, success: false, requests: [] }; report.samples.push(row);
          const requests = new Map(), start = performance.now();
          const onRequest = request => { const item = { path: new URL(request.url()).pathname, method: request.method(), startMs: performance.now() - start }; row.requests.push(item); requests.set(request, item); };
          const onDone = async request => { const item = requests.get(request); if (item) { item.endMs = performance.now() - start; item.status = (await request.response())?.status(); } };
          const onFailed = request => { const item = requests.get(request); if (item) item.error = request.failure()?.errorText; };
          page.on('request', onRequest); page.on('requestfinished', onDone); page.on('requestfailed', onFailed);
          const before = await cdp.send('Performance.getMetrics');
          try {
            await arm(page, action, expected, query); await run(); Object.assign(row, await page.evaluate(() => window.__pickerResult));
            await page.waitForLoadState('networkidle');
            if (errors.length) throw new Error(errors.join('\n'));
            if (row.requests.some(request => /\.(js|css|woff2)$/.test(request.path) || request.path.endsWith('/plugin-ui/asset'))) throw new Error('热操作仍在读取静态资源');
            if (row.requests.some(request => request.status >= 400 || request.error)) throw new Error('交互期间请求失败');
            row.success = true;
          } catch (error) { row.error = String(error); throw error; }
          finally {
            const after = await cdp.send('Performance.getMetrics');
            row.metrics = Object.fromEntries(after.metrics.filter(m => ['LayoutDuration', 'RecalcStyleDuration', 'ScriptDuration', 'LayoutCount'].includes(m.name))
              .map(m => [m.name, m.value - before.metrics.find(b => b.name === m.name).value]));
            page.off('request', onRequest); page.off('requestfinished', onDone); page.off('requestfailed', onFailed); save();
          }
          if (action === 'open') await page.screenshot({ path: `${values.output}/${width}-${cpu}-${profile}-${pair}-${kind}.png` });
        }
      }
    }
    for (const { context } of Object.values(pages)) await context.close();
  }
  report.complete = true;
} catch (error) {
  report.error = String(error); process.exitCode = 1;
  if (currentPage && !currentPage.isClosed()) await currentPage.screenshot({ path: `${values.output}/failure.png` });
} finally { save(); await browser.close(); }
