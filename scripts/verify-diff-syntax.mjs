import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { chromium } from 'playwright-core';

const evidence = resolve(process.env.EVIDENCE_DIR || '../evidence');
await mkdir(evidence, { recursive: true });
const server = createServer(async (req, res) => {
  try {
    const file = req.url === '/' ? null : resolve('.' + req.url);
    if (file && !file.startsWith(resolve('plugins/workspace_changes') + '/')) { res.writeHead(404).end(); return; }
    res.setHeader('Content-Type', file?.endsWith('.js') ? 'text/javascript' : file?.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(file ? await readFile(file) : '<link rel="stylesheet" href="/plugins/workspace_changes/web_module.css"><style>body{margin:0;--ak-paper-sheet:#fff;--ak-ink-primary:#24292e;--ak-type-technical:monospace}#host{height:90vh}html.dark{--ak-paper-sheet:#24292e;--ak-ink-primary:#e1e4e8}</style><div id="host"></div>');
  } catch { res.writeHead(404).end(); }
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
const page = await browser.newPage({ viewport: { width: 1100, height: 800 } });
const errors = [];
page.on('pageerror', e => errors.push(e.message));
try {
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.evaluate(async () => {
    window.workerStats = { live: 0, max: 0, created: 0 };
    const Native = window.Worker;
    window.Worker = class extends Native {
      constructor(...args) { super(...args); workerStats.created++; workerStats.live++; workerStats.max = Math.max(workerStats.max, workerStats.live); }
      terminate() { if (!this.stopped) { workerStats.live--; this.stopped = true; } super.terminate(); }
    };
    window.longTasks = [];
    new PerformanceObserver(list => longTasks.push(...list.getEntries().map(e => e.duration))).observe({type:'longtask', buffered:true});
    window.patches = {
      'sample.ts': '@@ -1,4 +1,4 @@\n-const old = "before";\n+const current = "after";\n /* comment\n  * still comment */\n console.log("<img src=x onerror=alert(1)>");\n',
      'sample.py': '@@ -1 +1 @@\n-print("before")\n+print("after")\n',
      'unknown.xyz': '@@ -1 +1 @@\n-old\n+new\n',
      'large.ts': '@@ -1 +1,1600 @@\n-old\n' + Array.from({length:1600}, (_,i) => '+const x'+i+' = 1;').join('\n'),
      'long.ts': '@@ -1 +1 @@\n-old\n+const x = "' + 'a'.repeat(2100) + '";',
      'sample.tsx': '@@ -1 +1 @@\n-const x = null;\n+const x = <div title="test">Hello</div>;\n',
    };
    window.quotes = [];
    const { activate } = await import('/plugins/workspace_changes/web_module.js');
    const ctx = { http: { request: async value => {
      const url = new URL(value, location.href);
      const path = url.searchParams.get('path');
      await new Promise(r => setTimeout(r, path === 'sample.py' ? 25 : 2));
      return {ok:true,json:async()=>url.pathname.endsWith('/patch') ? {status:'ok',patch:patches[path]} : {status:'ok',before:'a',after:'b',files:Object.keys(patches).map(path=>({path,status:'M',added:1,removed:1})),commits:[]}};
    } }, ui: { inject: (_, fn) => fn({register: entry => window.entry = entry}) } };
    activate(ctx);
    const view = {session:'test',active:false,onSessionChange:fn=>{window.session=fn;fn('test');return()=>{}},onActiveChange:fn=>{fn(false);return()=>{}},onIntent:()=>()=>{},quote:value=>{quotes.push(value);return 'inserted'}};
    window.mount = () => { window.dispose = entry.render(document.querySelector('#host'),null,view); };
    mount();
  });
  const select = async name => {
    await page.locator(`.changes-file[title="${name}（修改）"]`).click();
    await page.waitForFunction(name => document.querySelector('.changes-pane').dataset.path === name, name);
  };
  await select('sample.ts');
  await page.waitForFunction(() => document.querySelectorAll('.changes-token').length > 3);
  assert.equal(await page.locator('.changes-code img').count(), 0);
  assert.match(await page.locator('.changes-row[data-kind=add]').innerText(), /const current = "after"/);
  const light = await page.locator('.changes-token').first().evaluate(n => getComputedStyle(n).color);
  await page.screenshot({path:resolve(evidence,'light.png')});
  await page.evaluate(() => document.documentElement.classList.add('dark'));
  const dark = await page.locator('.changes-token').first().evaluate(n => getComputedStyle(n).color);
  assert.notEqual(light,dark);
  await page.screenshot({path:resolve(evidence,'dark.png')});
  await page.locator('.changes-row[data-kind=add]').first().click();
  await page.locator('textarea').fill('check highlighted line');
  await page.locator('textarea').press('Enter');
  assert.equal(await page.evaluate(() => quotes.length),1);
  for (const path of ['large.ts','long.ts','unknown.xyz']) {
    await select(path);
    await page.waitForTimeout(100);
    assert.equal(await page.locator('.changes-token').count(),0,path);
  }
  await select('sample.tsx');
  await page.waitForFunction(() => document.querySelectorAll('.changes-token').length > 3);
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:resolve(evidence,'narrow.png')});
  // Seeded monkey: real clicks, filtering, key navigation, resize, comments, teardown and theme changes.
  let seed = 20261030;
  const random = n => { seed = (Math.imul(seed,1664525)+1013904223)>>>0; return seed%n; };
  for(let i=0;i<160;i++) {
    switch(random(7)) {
      case 0: await page.locator('.changes-filter').fill(random(2) ? 'sample' : ''); break;
      case 1: { const items=page.locator('.changes-file'); if(await items.count()) await items.nth(random(await items.count())).click(); break; }
      case 2: await page.evaluate(()=>document.documentElement.classList.toggle('dark')); break;
      case 3: await page.setViewportSize({width:random(2)?390:1100,height:800}); break;
      case 4: await page.locator('.changes-pane').click({position:{x:5,y:5}}); await page.keyboard.press(random(2)?'j':'k'); break;
      case 5: { const row=page.locator('.changes-row[data-kind=add]').first(); if(await row.count()){await row.click();await page.locator('textarea').press('Escape');} break; }
      case 6: await page.evaluate(()=>{dispose();mount();}); await page.waitForSelector('.changes-file'); break;
    }
  }
  await page.locator('.changes-filter').fill('');
  await select('sample.py');
  await page.waitForFunction(()=>document.querySelectorAll('.changes-token').length>0);
  await page.evaluate(()=>dispose());
  assert.equal(await page.evaluate(()=>workerStats.live),0);
  assert.equal(await page.evaluate(()=>workerStats.max),1);
  assert.deepEqual(errors,[]);
  const result=await page.evaluate(()=>({workers:workerStats,longTasks,monkeyActions:160}));
  await writeFile(resolve(evidence,'results.json'),JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
} finally { await browser.close();server.close();console.log('证据目录:',evidence); }
