/** 项目内会话折叠的 monkey test：固定种子随机点项目、展开更多、选会话、搜索，每步核对可见条数与折叠行状态。 */
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { chromium } from 'playwright-core';
import { startDesktopFixtureServer } from '../../../scripts/webui-performance/desktop-fixture-server.mjs';
const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const root = mkdtempSync(resolve(tmpdir(), 'akashic-project-session-monkey-'));
const evidence = process.env.EVIDENCE_DIR ?? root;
symlinkSync(resolve(repo, 'node_modules'), root + '/node_modules', 'dir');
writeFileSync(root + '/index.html', '<html><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>');
writeFileSync(root + '/fixture.tsx', `
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, 'frontend/chat/src/desktop-sidebar.tsx'))};
import { TooltipProvider } from ${JSON.stringify(resolve(repo, 'frontend/chat/src/components/ui/tooltip.tsx'))};
import { initializeTheme } from ${JSON.stringify(resolve(repo, 'frontend/theme/src/theme-runtime.ts'))};
import ${JSON.stringify(resolve(repo, 'frontend/chat/src/styles.css'))};
initializeTheme();
const nothing = () => {}; const resolved = async () => {};
// 两个项目（9 条 / 12 条）与 5 条最近会话：两边的截断与折叠行计数都能各算各的。
const make = (prefix, group, count) => Array.from({ length: count }, (_, index) => ({
  id: prefix + (index + 1), title: group + ' 会话 ' + (index + 1), scope: { project: group }, scoped: true,
}));
const projectSessions = [...make('a', 'alpha', 9), ...make('b', 'beta', 12)];
const recentSessions = Array.from({ length: 5 }, (_, index) => ({ id: 'r' + (index + 1), title: '最近会话 ' + (index + 1) }));
function App(){
 const [selected,setSelected]=useState('');
 const rows = [...projectSessions, ...recentSessions].map(s=>({ ...s, preview:'', active: s.id===selected }));
 return <TooltipProvider><div className="akashic-chat"><div className="chat-shell-body" style={{height:'100vh'}}>
 <DesktopSidebar surface="chat" sessions={rows} activeSessionId={selected}
 pendingSessionId="" chatReady={true} onSelectSession={setSelected} onNewChat={nothing}
 navigationPins={{pins:[],sessions:[],ready:true,pending:false,error:'',reload:resolved,setPinned:resolved,expandedGroups:new Set(['project:alpha','project:beta']),toggleGroup:nothing,expandGroup:nothing}}
 groups={{sections:[{source:{key:'projects:r1',pluginId:'projects',pluginRevision:'r1',provider:{dimension:'project',label:'项目',load:async()=>[]},context:()=>({query:resolved})},groups:[{id:'alpha',label:'项目甲'},{id:'beta',label:'项目乙'}]}],revision:0,ready:true,active:null,reload:nothing,startChat:nothing}} />
 <output id="selected">{selected}</output></div></div></TooltipProvider>;
}
createRoot(document.getElementById('root')).render(<App />);
`);
await build({ configFile: resolve(repo, 'frontend/chat/vite.harness.config.ts'), root, logLevel: 'warn', css: { postcss: resolve(repo, 'frontend/chat') }, build: { outDir: root + '/dist', emptyOutDir: true } });
const server = await startDesktopFixtureServer(root + '/dist');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true });

// 固定种子的 LCG：同一个种子在任何机器上都重放同一串操作。
const random = (seed) => { let value = seed >>> 0; return () => (value = (value * 1664525 + 1013904223) >>> 0) / 2 ** 32; };
const GROUP_KEYS = ['project:alpha', 'project:beta'];
// 夹具里每个项目实际有多少条会话：折叠行文案要按这个总数反算，不能拿 DOM 里挂载的行数当总数。
const EXPECTED = { 'project:alpha': 9, 'project:beta': 12 };
const SEARCH_WORDS = ['会话', '会话 1', 'alpha', 'beta 会话', '最近'];

// 每一步之后都核对：可见条数只能是「折叠 5 条」「全部」或搜索态，折叠行只在需要时出现且计数正确。
async function audit(page) {
  return page.evaluate(() => {
    const searching = Boolean(document.querySelector('.chat-sidebar__search input')?.value.trim());
    const groups = [...document.querySelectorAll('.navigation-group')].map((group) => {
      const open = group.querySelector('.navigation-group__open')?.getAttribute('aria-expanded') === 'true';
      const rows = [...group.querySelectorAll('.navigation-group__sessions .conversation-session')];
      const visible = rows.filter((row) => row.offsetParent !== null && !row.closest('.navigation-menu-row')?.hasAttribute('hidden'));
      const more = group.querySelector('.group-session--more');
      return {
        key: group.dataset.groupKey, open, mounted: rows.length, visible: visible.length,
        moreText: more?.textContent?.trim() ?? null, moreExpanded: more?.getAttribute('aria-expanded') ?? null,
        moreCount: group.querySelectorAll('.group-session--more').length,
        activeVisible: visible.some((row) => row.classList.contains('active')),
      };
    });
    return { searching, groups };
  });
}

function check(state, label) {
  for (const group of state.groups) {
    if (!group.open) {
      assert.equal(group.visible, 0, `${label} ${group.key}: 收起时不应有可见会话`);
      assert.equal(group.moreCount, 0, `${label} ${group.key}: 收起时不应有折叠行`);
      continue;
    }
    if (state.searching) {
      assert.equal(group.moreCount, 0, `${label} ${group.key}: 搜索时不显示折叠行`);
      continue;
    }
    const total = EXPECTED[group.key];
    assert.ok(total, `${label}: 未知分组 ${group.key}`);
    assert.equal(group.moreCount, total > 5 ? 1 : 0, `${label} ${group.key}: 折叠行只在超出 5 条时出现`);
    if (group.moreExpanded === 'true') {
      assert.equal(group.visible, total, `${label} ${group.key}: 展开后全部可见`);
      assert.equal(group.mounted, total, `${label} ${group.key}: 展开后所有行都挂载`);
      assert.equal(group.moreText, '收起', `${label} ${group.key}: 展开后折叠行是收起`);
    } else {
      assert.equal(group.visible, 5, `${label} ${group.key}: 折叠时只列 5 条`);
      assert.equal(group.moreText, `展开更多（${total - 5}）`, `${label} ${group.key}: 折叠行计数`);
    }
  }
}

const results = [];
try {
  for (const seed of [7, 20261103]) {
    const ctx = await browser.newContext({ viewport: { width: 1365, height: 860 } });
    const page = await ctx.newPage(); page.setDefaultTimeout(4000);
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(server.origin + '/');
    await page.locator('.navigation-group__sessions .conversation-session').first().waitFor();
    const next = random(seed);
    const pick = (items) => items[Math.floor(next() * items.length)];
    const log = [];
    let skipped = 0;
    for (let step = 1; step <= 40; step += 1) {
      const action = pick(['group', 'more', 'session', 'search', 'clear']);
      const groupKey = pick(GROUP_KEYS);
      try {
        if (action === 'group') {
          await page.locator(`.navigation-group[data-group-key="${groupKey}"] .navigation-group__open`).click({ timeout: 1500 });
        } else if (action === 'more') {
          const more = page.locator(`.navigation-group[data-group-key="${groupKey}"] .group-session--more`);
          if (await more.count()) await more.click({ timeout: 1500 }); else skipped += 1;
        } else if (action === 'session') {
          const rows = page.locator(`.navigation-group[data-group-key="${groupKey}"] .navigation-group__sessions .conversation-session`);
          const count = await rows.count();
          if (count) {
            const row = rows.nth(Math.floor(next() * count));
            if (await row.isVisible()) await row.click({ timeout: 1500 }); else skipped += 1;
          } else skipped += 1;
        } else if (action === 'search') {
          await page.getByRole('textbox', { name: '搜索会话' }).fill(pick(SEARCH_WORDS));
        } else {
          await page.getByRole('textbox', { name: '搜索会话' }).fill('');
        }
      } catch (error) {
        // 收起瞬间行被卸载、或按钮在搜索时不可点：这属于随机操作的正常落空，记数不当失败。
        if (!/Timeout|not visible|detached|intercepts pointer events/.test(String(error))) throw error;
        skipped += 1;
      }
      const state = await audit(page);
      check(state, `seed=${seed} step=${step} ${action}`);
      log.push({ step, action, groupKey, searching: state.searching, groups: state.groups.map((group) => ({ key: group.key, open: group.open, visible: group.visible, more: group.moreText })) });
      if (step % 10 === 0) await page.screenshot({ path: `${evidence}/monkey-seed${seed}-step${step}.png`, animations: 'disabled' });
    }
    // 收尾：拉一个“甲展开全部 / 乙保持折叠”的稳定状态截图，便于人工核对两个项目各算各的。
    await page.getByRole('textbox', { name: '搜索会话' }).fill('');
    for (const key of GROUP_KEYS) {
      const open = await page.locator(`.navigation-group[data-group-key="${key}"] .navigation-group__open`).getAttribute('aria-expanded');
      if (open !== 'true') await page.locator(`.navigation-group[data-group-key="${key}"] .navigation-group__open`).click();
    }
    const moreAlpha = page.locator('.navigation-group[data-group-key="project:alpha"] .group-session--more');
    if (await moreAlpha.count() && await moreAlpha.getAttribute('aria-expanded') !== 'true') await moreAlpha.click();
    await audit(page).then((state) => check(state, `seed=${seed} final`));
    await page.screenshot({ path: `${evidence}/monkey-seed${seed}-final.png`, animations: 'disabled' });
    assert.deepEqual(errors, [], 'monkey 期间不应有页面异常');
    await ctx.close();
    results.push({ seed, skipped, errors, log });
    console.log(`PASS seed=${seed} steps=40 skipped=${skipped}`);
  }
  writeFileSync(evidence + '/results.json', JSON.stringify(results, null, 2));
} finally { await browser.close(); await server.close(); }
console.log('证据目录: ' + evidence);
