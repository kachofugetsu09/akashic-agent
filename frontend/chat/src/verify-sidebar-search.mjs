/** 用真实侧栏组件验证项目分区、置顶分区与最近会话共用搜索和键盘流程。 */
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, symlinkSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { build } from 'vite';
import { chromium } from 'playwright-core';
import { startDesktopFixtureServer } from '../../../scripts/webui-performance/desktop-fixture-server.mjs';
const repo = process.env.REVIEW_SOURCE_ROOT ?? resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
const root = mkdtempSync(resolve(tmpdir(), 'akashic-sidebar-search-'));
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
const sessions = [
 { id:'p1', title:'命中 项目一', scope:{project:'project'}, scoped:true },
 { id:'p2', title:'命中 项目二', scope:{project:'project'}, scoped:true },
 { id:'r1', title:'命中 最近会话' },
].map(s=>({...s, preview:'', active:false}));
function App(){
 const [selected,setSelected]=useState('');
 const pinned=new URLSearchParams(location.search).get('pinned')==='1';
 return <TooltipProvider><div className="akashic-chat"><div className="chat-shell-body" style={{height:'100vh'}}>
 <DesktopSidebar surface="chat" sessions={sessions} activeSessionId={selected}
 pendingSessionId="" chatReady={true} themeLabel="纸感" onSelectSession={setSelected} onNewChat={nothing} onCycleTheme={nothing}
 navigationPins={{pins:pinned?[{kind:'project',id:'project'},{kind:'session',id:'r1'}]:[],sessions:[],ready:true,pending:false,error:'',reload:resolved,setPinned:resolved,expandedGroups:new Set(),toggleGroup:nothing,expandGroup:nothing}}
 groups={{sections:[{source:{key:'projects:r1',pluginId:'projects',pluginRevision:'r1',provider:{dimension:'project',label:'项目',load:async()=>[]},context:()=>({query:resolved})},groups:[{id:'project',label:'命中 项目'}]}],revision:0,active:null,reload:nothing,startChat:nothing}} />
 <output id="selected">{selected}</output></div></div></TooltipProvider>;
}
createRoot(document.getElementById('root')).render(<App />);
`);
await build({ configFile: resolve(repo, 'frontend/chat/vite.harness.config.ts'), root, logLevel: 'warn', css: { postcss: resolve(repo, 'frontend/chat') }, build: { outDir: root + '/dist', emptyOutDir: true } });
const server = await startDesktopFixtureServer(root + '/dist');
const browser = await chromium.launch({ executablePath: process.env.CHROMIUM_PATH ?? '/usr/bin/chromium', headless: true });
const results = [];
try {
 for (const pinned of [false, true]) {
  const ctx = await browser.newContext({viewport:{width:1365,height:860}});
  const page = await ctx.newPage(); page.setDefaultTimeout(10000); const errors=[];
  page.on('pageerror', error=>errors.push(error.message));
  await page.goto(server.origin + '/?pinned=' + Number(pinned));
  const search=page.getByRole('textbox',{name:'搜索会话'});await search.fill('命中');
  await page.locator('.navigation-group__sessions .conversation-session').first().waitFor();
  // 1. 按可见顺序跨分区移动：先到分组行，再到组内会话，不能跳过。
  await search.press('ArrowDown');
  assert.equal(await page.evaluate(()=>document.activeElement?.getAttribute('title')), '收起 命中 项目');
  await page.keyboard.press('ArrowDown');
  assert.equal(await page.evaluate(()=>document.activeElement?.getAttribute('title')), '命中 项目一');
  await page.keyboard.press('ArrowDown');
  assert.equal(await page.evaluate(()=>document.activeElement?.getAttribute('title')), '命中 项目二');
  await page.keyboard.press('ArrowDown');
  assert.ok((await page.evaluate(()=>document.activeElement?.textContent)).includes('命中 最近会话'));
  for(let i=0;i<4;i++)await page.keyboard.press('ArrowUp');
  assert.equal(await search.evaluate(node=>node===document.activeElement),true);
  // 2. 高亮覆盖分组名和两种会话行；回车打开首条组内会话并清掉筛选。
  const highlight=await page.evaluate(()=>Array.from(CSS.highlights.get('sidebar-search')??[]).map(range=>range.toString()));
  assert.equal(highlight.length,4);assert.ok(highlight.every(text=>text==='命中'));
  await search.press('Enter');await page.waitForFunction(()=>document.querySelector('#selected')?.textContent==='p1');
  assert.equal(await search.inputValue(),'');
  await search.fill('项目');
  await page.waitForFunction(()=>CSS.highlights.get('sidebar-search')?.size===3);
  await search.press('Escape');assert.equal(await search.inputValue(),'');
  // 3. roving 停靠点不留在隐藏行：先停靠"命中 最近会话"，再把它搜隐藏。
  await page.locator('.conversation-session[title="命中 最近会话"]').click();
  await search.fill('项目');
  await page.waitForFunction(()=>document.querySelector('.conversation-session[title="命中 最近会话"]')?.closest('.navigation-menu-row')?.hasAttribute('hidden'));
  // 隐藏行不得持 tabindex="0"，且当前停靠点必须是可见行。
  assert.equal(await page.evaluate(()=>document.querySelector('.conversation-session[title="命中 最近会话"]')?.getAttribute('tabindex')),'-1');
  assert.equal(await page.evaluate(()=>document.querySelectorAll('.chat-sidebar__navigation [hidden] [tabindex="0"]').length),0);
  // Tab 离开搜索框后落到可见行（有输入时中间隔一枚清除钮），不能停在 [hidden] 行上。
  await search.press('Tab');
  if(await page.evaluate(()=>document.activeElement?.classList.contains('chat-sidebar__search-clear')))await page.keyboard.press('Tab');
  const stop=await page.evaluate(()=>{const el=document.activeElement;return {hidden:!(el instanceof HTMLElement)||el.offsetParent===null||el.closest('[hidden]')!==null,tab:el?.getAttribute('tabindex')};});
  assert.deepEqual(stop,{hidden:false,tab:'0'});
  await search.press('Escape');
  assert.deepEqual(errors,[]);await page.screenshot({path:root+'/pinned-'+pinned+'.png',animations:'disabled'});
  results.push({pinned,errors,highlight});console.log('PASS',pinned?'pinned-project':'project-and-recent');await ctx.close();
 }
 writeFileSync(root+'/results.json',JSON.stringify(results,null,2));
}finally{await browser.close();await server.close();}
console.log('证据目录: '+root);
