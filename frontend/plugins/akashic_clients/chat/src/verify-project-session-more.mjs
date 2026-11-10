/** 用真实侧栏组件验证项目内会话的 5 条截断、“展开更多/收起”与搜索时的全量展开。 */
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { chromium } from 'playwright-core';
import { startDesktopFixtureServer } from '../../../../../scripts/webui-performance/desktop-fixture-server.mjs';
const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), '../../../../..');
const root = mkdtempSync(resolve(tmpdir(), 'akashic-project-session-more-'));
const evidence = process.env.EVIDENCE_DIR ?? root;
symlinkSync(resolve(repo, 'node_modules'), root + '/node_modules', 'dir');
writeFileSync(root + '/index.html', '<html><body><div id="root"></div><script type="module" src="./fixture.tsx"></script></body></html>');
writeFileSync(root + '/fixture.tsx', `
import { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { DesktopSidebar } from ${JSON.stringify(resolve(repo, 'frontend/plugins/akashic_clients/chat/src/desktop-sidebar.tsx'))};
import { TooltipProvider } from ${JSON.stringify(resolve(repo, 'frontend/plugins/akashic_clients/chat/src/components/ui/tooltip.tsx'))};
import { initializeTheme } from ${JSON.stringify(resolve(repo, 'packages/akashic-theme/src/theme-runtime.ts'))};
import ${JSON.stringify(resolve(repo, 'frontend/plugins/akashic_clients/chat/src/styles.css'))};
initializeTheme();
const nothing = () => {}; const resolved = async () => {};
// 一个项目下 9 条会话，另有 2 条不属于任何项目的最近会话。
const projectSessions = Array.from({ length: 9 }, (_, index) => ({
  id: 's' + (index + 1), title: '项目会话 ' + (index + 1), scope: { project: 'project' }, scoped: true,
}));
const recentSessions = [{ id: 'r1', title: '最近会话一' }, { id: 'r2', title: '最近会话二' }];
function App(){
 const params = new URLSearchParams(location.search);
 const preset = params.get('active') ?? '';
 const [selected,setSelected]=useState(preset);
 const rows = [...projectSessions, ...recentSessions].map(s=>({ ...s, preview:'', active: s.id===selected }));
 return <TooltipProvider><div className="akashic-chat"><div className="chat-shell-body" style={{height:'100vh'}}>
 <DesktopSidebar surface="chat" sessions={rows} activeSessionId={selected}
 pendingSessionId="" chatReady={true} onSelectSession={setSelected} onNewChat={nothing}
 navigationPins={{pins:[],sessions:[],ready:true,pending:false,error:'',reload:resolved,setPinned:resolved,expandedGroups:new Set(['project:project']),toggleGroup:nothing,expandGroup:nothing}}
 groups={{sections:[{source:{key:'projects:r1',pluginId:'projects',pluginRevision:'r1',provider:{dimension:'project',label:'项目',load:async()=>[]},context:()=>({query:resolved})},groups:[{id:'project',label:'项目甲'}]}],revision:0,ready:true,active:{dimension:'project',id:'project',label:'项目甲'},reload:nothing,startChat:nothing}} />
 <output id="selected">{selected}</output></div></div></TooltipProvider>;
}
createRoot(document.getElementById('root')).render(<App />);
`);
await build({ configFile: resolve(repo, 'frontend/plugins/akashic_clients/chat/vite.harness.config.ts'), root, logLevel: 'warn', css: { postcss: resolve(repo, 'frontend/plugins/akashic_clients/chat') }, build: { outDir: root + '/dist', emptyOutDir: true } });
const server = await startDesktopFixtureServer(root + '/dist');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true });
// 分组里的会话行是宿主给的 ConversationSessionRow，class 为 conversation-session；“展开更多”是 group-session--more。
const rows = (page) => page.$$eval('.navigation-group__sessions .conversation-session', (nodes) => nodes.map((node) => ({
  title: node.getAttribute('title'), active: node.classList.contains('active'),
  hidden: node.closest('.navigation-menu-row')?.hasAttribute('hidden') ?? false,
})));
const visibleTitles = async (page) => (await rows(page)).filter((row) => !row.hidden).map((row) => row.title);
const moreButton = (page) => page.locator('.navigation-group__sessions .group-session--more');
const results = [];
try {
 {
  const ctx = await browser.newContext({ viewport: { width: 1365, height: 860 } });
  const page = await ctx.newPage(); page.setDefaultTimeout(10000); const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(server.origin + '/');
  await page.locator('.navigation-group__sessions .conversation-session').first().waitFor();
  // 1. 默认只列 5 条，末行是“展开更多（4）”。
  assert.deepEqual(await visibleTitles(page),
    ['项目会话 1', '项目会话 2', '项目会话 3', '项目会话 4', '项目会话 5'], '默认只显示 5 条会话');
  assert.equal(await page.locator('.navigation-group__sessions .conversation-session').count(), 5, '截断外的行不挂载');
  assert.equal(await moreButton(page).count(), 1, '超出上限时出现展开行');
  assert.equal((await moreButton(page).textContent()).trim(), '展开更多（4）');
  await page.screenshot({ path: evidence + '/collapsed.png', animations: 'disabled' });
  // 2. 展开后全部列出，末行改成“收起”。
  await moreButton(page).click();
  assert.equal((await visibleTitles(page)).length, 9, '展开后显示全部会话');
  assert.equal((await moreButton(page).textContent()).trim(), '收起');
  await page.screenshot({ path: evidence + '/expanded.png', animations: 'disabled' });
  // 3. 再点一次回到 5 条。
  await moreButton(page).click();
  assert.deepEqual(await visibleTitles(page),
    ['项目会话 1', '项目会话 2', '项目会话 3', '项目会话 4', '项目会话 5'], '收起后回到 5 条');
  await ctx.close();
  results.push({ scenario: 'collapse-expand', errors });
  console.log('PASS collapse-expand');
 }
 {
  // 4. 当前会话落在截断外：顶掉最后一条仍然可见，列表守住 5 条份上限。
  const ctx = await browser.newContext({ viewport: { width: 1365, height: 860 } });
  const page = await ctx.newPage(); page.setDefaultTimeout(10000); const errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(server.origin + '/?active=s8');
  await page.locator('.navigation-group__sessions .conversation-session').first().waitFor();
  const shown = await rows(page);
  assert.deepEqual(shown.map((row) => row.title),
    ['项目会话 1', '项目会话 2', '项目会话 3', '项目会话 4', '项目会话 8'], '当前会话顶掉最后一条，上限不变');
  assert.equal(shown.at(-1).active, true, '当前会话在列表中标记为激活');
  await page.screenshot({ path: evidence + '/active-outside.png', animations: 'disabled' });
  // 5. 搜索不做截断：匹配行全部出现，折叠行让位。
  const search = page.getByRole('textbox', { name: '搜索会话' });
  await search.fill('项目会话');
  await page.waitForFunction(() => document.querySelectorAll('.navigation-group__sessions .conversation-session').length >= 9);
  assert.equal((await visibleTitles(page)).length, 9, '搜索时全部匹配行可见');
  assert.equal(await moreButton(page).count(), 0, '搜索时不显示折叠行');
  assert.deepEqual(errors, []);
  await ctx.close();
  results.push({ scenario: 'active-outside-search', errors });
  console.log('PASS active-outside-search');
 }
 writeFileSync(evidence + '/results.json', JSON.stringify(results, null, 2));
} finally { await browser.close(); await server.close(); }
console.log('证据目录: ' + evidence);
