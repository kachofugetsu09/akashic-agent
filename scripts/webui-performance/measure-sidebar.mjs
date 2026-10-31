/** 真实侧栏项目开合与搜索基准；不创建、重命名、置顶或移动会话。 */
import {chromium} from 'playwright-core';
import {readFileSync,writeFileSync,mkdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {parseArgs} from 'node:util';
// 1. 使用新浏览器上下文和完整目录；认证只带 Cookie，不带旧的 UI 缓存或展开状态。
const {values}=parseArgs({options:{url:{type:'string'},browser:{type:'string'},output:{type:'string'},
 auth:{type:'string'},runs:{type:'string',default:'3'},profiles:{type:'string',default:'lan,weak'}}});
for(const key of ['url','browser','output'])if(!values[key])throw Error(`需要 --${key}`);
const rounds=Number(values.runs),profiles=values.profiles.split(',');
if(!Number.isSafeInteger(rounds)||rounds<1)throw Error('runs 必须为正整数');
if(!profiles.length||new Set(profiles).size!==profiles.length||profiles.some(p=>!['lan','weak'].includes(p)))throw Error('profiles 只接受不重复的 lan、weak');
const output=values.output;mkdirSync(output,{mode:0o700});
const auth=values.auth?JSON.parse(readFileSync(values.auth)):null;
const browser=await chromium.launch({executablePath:values.browser});
const report={complete:false,browser:browser.version(),runs:rounds,profiles,samples:[],catalogs:[],
 profileSettings:{lan:{latency:20,downloadThroughput:12500000,uploadThroughput:12500000},weak:{latency:800,downloadThroughput:125000,uploadThroughput:125000}},
 harnessSha256:createHash('sha256').update(readFileSync(import.meta.filename)).digest('hex'),
 lockSha256:createHash('sha256').update(readFileSync('package-lock.json')).digest('hex')};let current;
const label=s=>(s.title?.trim()||s.first_message_content?.trim()||'未命名对话').replace(/\s+/gu,' ').slice(0,200);
const save=()=>writeFileSync(output+'/report.json',JSON.stringify(report,null,2),{mode:0o600});
/** 从真实输入事件到完整结果，再等待按时间运行的有限动画；滚动时间线另记。 */
async function arm(page,action,expected,key,query){await page.evaluate(({action,expected,key,query})=>{
 window.__sidebarResult=new Promise((resolve,reject)=>{
  let start,done=false;const event=action.startsWith('search')?'input':'pointerdown';
  const timer=setTimeout(()=>{done=true;reject(Error('sidebar completion timeout'));},15000);
  document.addEventListener(event,()=>{start=performance.now();requestAnimationFrame(check);},{capture:true,once:true});
  function check(){if(done)return;const side=[...document.querySelectorAll('.chat-sidebar')].find(e=>e.getClientRects().length&&getComputedStyle(e).visibility!=='hidden');
   if(!side){requestAnimationFrame(check);return;}
   const group=key?[...side.querySelectorAll('[data-group-key]')].find(e=>e.dataset.groupKey===key):null;
   const root=group||side;
   const actual=[...root.querySelectorAll('.conversation-session__title,.group-session__title')].filter(e=>!e.closest('[hidden]')&&e.getClientRects().length).map(e=>e.textContent).sort();
   const opened=group?.querySelector('.navigation-group__open')?.getAttribute('aria-expanded')==='true';
   window.__sidebarDiagnostic={action,actual,expected,key,query,actualQuery:side.querySelector('.chat-sidebar__search input')?.value,opened};
   const ready=JSON.stringify(actual)===JSON.stringify(expected)&&(action==='group-open'?opened:action==='group-close'?!opened:side.querySelector('.chat-sidebar__search input')?.value===query);
   if(!ready){requestAnimationFrame(check);return;}done=true;
   requestAnimationFrame(()=>requestAnimationFrame(async()=>{try{const contentMs=performance.now()-start;const animations=root.getAnimations({subtree:true});const scrollDrivenAnimations=animations.filter(a=>a.timeline&&!(a.timeline instanceof DocumentTimeline)).map(a=>a.animationName);/* 滚动时间线取决于当前位置，不是等待时钟可完成的过渡。 */await Promise.all(animations.filter(a=>a.timeline instanceof DocumentTimeline&&a.effect?.getComputedTiming().iterations!==Infinity).map(a=>a.finished));requestAnimationFrame(()=>requestAnimationFrame(()=>{clearTimeout(timer);resolve({contentMs,settledMs:performance.now()-start,scrollDrivenAnimations});}));}catch(e){clearTimeout(timer);reject(e);}}));
  }
 });
},{action,expected,key,query});}
try{
 // 2. 正常/弱网、桌面/窄屏与 CPU 节流分开记录，不用空壳或热缓存结果充当目录真值。
 for(const width of [1440,390])for(const cpu of [1,4]){
  const context=await browser.newContext({viewport:{width,height:900},hasTouch:width===390,isMobile:width===390,...(auth?{storageState:{cookies:auth.cookies,origins:[]}}:{}),serviceWorkers:'block'});const page=await context.newPage();current=page;
  const cdp=await context.newCDPSession(page);await cdp.send('Network.enable');await cdp.send('Emulation.setCPUThrottlingRate',{rate:cpu});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
  const sessionsRead=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/chat/sessions'&&r.status()===200);
  const pinsRead=page.waitForResponse(r=>new URL(r.url()).pathname==='/api/chat/navigation/pins'&&r.status()===200);
  const projectsRead=page.waitForResponse(r=>r.url().endsWith('/api/chat/plugin-ui/query')&&r.request().postDataJSON()?.method==='project.list'&&r.status()===200);
  await page.goto(values.url);if(width===390)await page.getByRole('button',{name:'打开导航',exact:true}).click();
  const sessions=await(await sessionsRead).json(),projects=await(await projectsRead).json();assert.equal(sessions.next_cursor,null,'oracle must contain complete catalog');assert.equal(sessions.items.length,sessions.total);
  const pins=await(await pinsRead).json();const merged=new Map(pins.sessions.map(r=>[r.key,r]));for(const row of sessions.items)merged.set(row.key,row);sessions.items=[...merged.values()];
  const groups=projects.items.filter(p=>!p.archived);assert.ok(groups.length>0&&sessions.items.length>0,'需要非空的真实项目和会话目录');const known=new Set(groups.map(p=>p.id)),recent=sessions.items.filter(s=>!known.has(s.scope?.project)).map(label).sort();
  report.catalogs.push({width,cpu,sessions:sessions.total,projects:groups.length,sha256:createHash('sha256').update(JSON.stringify({sessions,projects,pins})).digest('hex'),oracle:{sessions,projects,pins}});
  const side=page.locator('.chat-sidebar').filter({visible:true}),search=side.getByRole('textbox',{name:'搜索会话',exact:true});
  await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(21000);await page.waitForLoadState('networkidle');
  for(const profile of profiles){
   await cdp.send('Network.emulateNetworkConditions',{offline:false,latency:profile==='lan'?20:800,downloadThroughput:profile==='lan'?12500000:125000,uploadThroughput:profile==='lan'?12500000:125000});
   for(let round=0;round<rounds;round++){
    for(const [index,group] of groups.entries()){
     const key='project:'+group.id;
     const selector=await page.evaluate(key=>'[data-group-key="'+CSS.escape(key)+'"] .navigation-group__open',key);
     const control=side.locator(selector);
     const expected=sessions.items.filter(s=>s.scope?.project===group.id).map(label).sort();
     await measure('group-open',()=>control.click(),expected,key,'',index);
     await measure('group-close',()=>control.click(),[],key,'',index);
    }
    const needle=label(sessions.items[0]).slice(0,6).toLowerCase();
    for(const query of ['条消息',needle,'no-session-match-20261031']){
     const expected=sessions.items.filter(s=>`${label(s)} ${s.message_count===undefined?'':s.message_count+' 条消息'}`.toLowerCase().includes(query)).map(label).sort();
     await measure('search-input',()=>search.fill(query),expected,null,query);
     await measure('search-clear',()=>search.fill(''),recent,null,'');
    }
    /** 保留每次动作和失败；网络流水与 UI 内容完成分别记录。 */
    async function measure(action,run,expected,key,query,index){
     await page.waitForLoadState('networkidle');const row={width,cpu,profile,round,action,index,queryCase:action.startsWith('search')?(query==='条消息'?'all':query===''?'clear':query==='no-session-match-20261031'?'none':'narrow'):undefined,expectedCount:expected.length,success:false,requests:[]};report.samples.push(row);const requests=new Map(),start=performance.now();
     const req=r=>{const item={path:new URL(r.url()).pathname,method:r.method(),query:r.method()==='POST'?r.postDataJSON()?.method:undefined,startMs:performance.now()-start};row.requests.push(item);requests.set(r,item);};
     const end=async r=>{const item=requests.get(r);if(item){item.endMs=performance.now()-start;item.status=(await r.response())?.status();}};const fail=r=>{const item=requests.get(r);if(item)item.error=r.failure()?.errorText;};
     page.on('request',req);page.on('requestfinished',end);page.on('requestfailed',fail);
     try{await arm(page,action,expected,key,query);await run();Object.assign(row,await page.evaluate(()=>window.__sidebarResult));await page.waitForLoadState('networkidle');assert.equal(errors.length,0,errors.join('\n'));assert.ok(!row.requests.some(r=>r.status>=400||r.error),'request failure');row.success=true;}
     catch(error){row.error=String(error);row.diagnostic=await page.evaluate(()=>window.__sidebarDiagnostic);throw error;}
     finally{page.off('request',req);page.off('requestfinished',end);page.off('requestfailed',fail);save();}
    }
   }
  }
  await page.screenshot({path:`${output}/${width}-${cpu}.png`});await context.close();
 }
 report.complete=true;
}catch(error){report.error=String(error);process.exitCode=1;if(current&&!current.isClosed())await current.screenshot({path:output+'/failure.png'});}
finally{save();await browser.close();}
