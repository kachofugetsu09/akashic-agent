import {chromium} from 'playwright-core';
import {readFileSync,writeFileSync,mkdirSync,readdirSync,utimesSync,realpathSync} from 'node:fs';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {join,relative,resolve} from 'node:path';
import {parseArgs} from 'node:util';
const {values}=parseArgs({options:{
  state:{type:'string'},baseline:{type:'string'},candidate:{type:'string'},
  browser:{type:'string'},output:{type:'string'},runs:{type:'string',default:'7'},width:{type:'string'},
}});
for(const name of ['state','baseline','candidate','browser','output'])assert.ok(values[name],'需要 --'+name);
const state=realpathSync(values.state),output=resolve(values.output),rounds=Number(values.runs);
assert.ok(Number.isSafeInteger(rounds)&&rounds>0,'runs 必须是正整数');
const widths=values.width?[Number(values.width)]:[1440,390];assert.ok(widths.every(width=>[1440,390].includes(width)),'width 只能是 1440 或 390');
const marker=JSON.parse(readFileSync(join(state,'.akashic-oneoff-fixture.json')));
assert.equal(marker.kind,'akashic-webui-oneoff');assert.equal(marker.state,state);
assert.equal(marker.workspace,join(state,'workspace'));
assert.equal(realpathSync(marker.chat),join(state,'plugins/akashic_clients/static/chat'));
assert.ok(Number.isSafeInteger(marker.pid)&&marker.pid>0);
assert.ok(Number.isSafeInteger(marker.port)&&marker.port>0&&marker.port<65536);
const url='http://127.0.0.1:'+marker.port;
const oracle=JSON.parse(readFileSync(join(state,'message-oracle.json')));
for(const id of ['interaction-long-input','interaction-long-output'])assert.equal(typeof oracle[id],'string');
mkdirSync(output,{mode:0o700});
const browser=await chromium.launch({executablePath:values.browser});
const report={complete:false,samples:[],harnessSha256:createHash('sha256').update(readFileSync(import.meta.filename)).digest('hex'),oracleSha256:createHash('sha256').update(JSON.stringify(oracle)).digest('hex')};let current;
report.browserVersion=browser.version();
report.fixtureHelperSha256=createHash('sha256').update(readFileSync(new URL('serve-startup-runtime.py',import.meta.url))).digest('hex');
report.widths=widths;report.plannedSamples=rounds*widths.reduce((count,width)=>count+(width===390?12:10)*8,0);
report.acceptance={rounds,targetActions:['quote','cancel','edit','clear'],cpu4MedianSettledGain:'>=20% and >=100ms in every network/width cell',otherMedianRegression:'<=max(25ms,10%) for content and settled',absoluteTargetMs:300};
const save=()=>writeFileSync(output+'/report.json',JSON.stringify(report,null,2),{mode:0o600});

// 只发布到仍存活的本次隔离实例；源码、workspace 和正式服务均不属于 rsync 目标。
async function install(variant){
 assert.ok(readFileSync('/proc/'+marker.pid+'/cmdline').includes(Buffer.from(state)),'隔离进程身份已改变');
 assert.equal(realpathSync(marker.chat),join(state,'plugins/akashic_clients/static/chat'));
 const source=realpathSync(values[variant]);assert.notEqual(source,marker.chat);
 assert.ok(!source.startsWith(marker.chat+'/')&&!marker.chat.startsWith(source+'/'),'制品与发布目录不能互相包含');
 assert.ok(readFileSync(join(source,'chat-app.js')).length,'缺少聊天入口制品');
 execFileSync('rsync',['-a','--checksum','--delete',source+'/',marker.chat+'/']);
 const files=[];function walk(path){for(const item of readdirSync(path,{withFileTypes:true})){const full=join(path,item.name);if(item.isDirectory())walk(full);else if(item.isFile())files.push(full);}}walk(marker.chat);
 const hashes={};for(const file of files){const hash=createHash('sha256').update(readFileSync(file)).digest('hex');hashes[relative(marker.chat,file)]=hash;const stamp=1767225600+parseInt(hash.slice(0,10),16)/1e6;utimesSync(file,stamp,stamp);}
 let verified=0;
 for(const file of files.filter(path=>/\.(js|css)$/.test(path))){
  const response=await fetch(url+'/assets/'+relative(marker.chat,file),{headers:{'Accept-Encoding':'gzip'}});
  assert.equal(response.status,200);assert.ok(Buffer.from(await response.arrayBuffer()).equals(readFileSync(file)),file);verified++;
 }
 report.servedBuilds??=[];report.servedBuilds.push({variant,verified,hashes});save();
}

// 从真实输入事件计时；正文、焦点、草稿和剪贴板由独立原文核对。
async function arm(page,action,id,text){await page.evaluate(({action,id,text})=>{
 window.__messageResult=new Promise((resolve,reject)=>{
  let start,done=false;const timer=setTimeout(()=>{done=true;reject(Error('message action completion timeout'));},15000);const event=action==='clear'?'input':'pointerdown';
  document.addEventListener(event,()=>{start=performance.now();requestAnimationFrame(check);},{capture:true,once:true});
  async function check(){if(done)return;const row=document.querySelector(`[data-message-id="${id}"]`),form=document.querySelector('form.composer'),input=form.querySelector('textarea'),reply=form.querySelector('.composer-reply');const clamp=row.querySelector('.user-text-clamp'),toggle=row.querySelector('.user-text-toggle');
   const stored=Object.keys(sessionStorage).filter(k=>k.startsWith('akashic.chat.draft:')).map(k=>sessionStorage.getItem(k));let ready=false;
   if(action==='reveal'||action==='hide')ready=row.classList.contains('is-revealed')===(action==='reveal')&&(action==='reveal'?row.querySelector('.shared-message-meta').getBoundingClientRect().height>0:row.querySelector('.shared-message-meta').getBoundingClientRect().height===0);
   if(action==='expand'||action==='collapse'){const open=action==='expand';ready=clamp.dataset.state===(open?'open':'clamped')&&toggle.getAttribute('aria-expanded')===String(open)&&(open?clamp.scrollHeight-clamp.clientHeight<=1:clamp.scrollHeight-clamp.clientHeight>1)&&clamp.textContent.replace(/\s/gu,'')===text.replace(/\s/gu,'');}
   if(action==='quote')ready=reply?.querySelector('span')?.textContent===text.replace(/\s+/gu,' ').trim().slice(0,512)&&document.activeElement===input;
   if(action==='cancel')ready=!reply;
   if(action==='edit')ready=input.value===text&&input.selectionStart===text.length&&input.selectionEnd===text.length&&document.activeElement===input&&stored.includes(text)&&form.classList.contains('is-expanded');
   if(action==='clear')ready=input.value===''&&stored.length===0&&!form.classList.contains('is-expanded');
   if(action==='copy')ready=Boolean(row.querySelector('button[aria-label="已复制"]'))&&await navigator.clipboard.readText()===text;
   window.__messageDiagnostic={action,id,clamp:clamp?.dataset.state,inputLength:input.value.length,storedLengths:stored.map(s=>s.length),reply:reply?.textContent};if(!ready){requestAnimationFrame(check);return;}done=true;const root=['quote','cancel','edit','clear'].includes(action)?form:row;
   requestAnimationFrame(()=>requestAnimationFrame(async()=>{try{const contentMs=performance.now()-start;await Promise.all(root.getAnimations({subtree:true}).filter(a=>a.timeline instanceof DocumentTimeline&&a.effect?.getComputedTiming().iterations!==Infinity).map(a=>a.finished));let previous='',stable=0;function settle(){const r=root.getBoundingClientRect(),c=form.getBoundingClientRect();const value=JSON.stringify([r.x,r.y,r.width,r.height,c.y,c.height]);stable=value===previous?stable+1:0;previous=value;if(stable<2){requestAnimationFrame(settle);return;}clearTimeout(timer);resolve({contentMs,settledMs:performance.now()-start,rootHeight:r.height,composerHeight:c.height});}settle();}catch(e){clearTimeout(timer);reject(e);}}));
  }
 });
},{action,id,text});}
try{
 for(const width of widths)for(const cpu of [1,4]){
  const pages={};for(const variant of ['baseline','candidate']){await install(variant);
  const context=await browser.newContext({viewport:{width,height:900},hasTouch:width===390,isMobile:width===390,permissions:['clipboard-read','clipboard-write'],serviceWorkers:'block'});await context.addInitScript(()=>localStorage.setItem('akashic.onboarding.seen','1'));const page=await context.newPage();current=page;const errors=[];page.on('pageerror',e=>errors.push(String(e)));const cdp=await context.newCDPSession(page);await cdp.send('Network.enable');await cdp.send('Emulation.setCPUThrottlingRate',{rate:cpu});await cdp.send('Performance.enable');
  await page.goto(url+'/?session=akashic%3Aswitch-benchmark');await page.locator('[data-message-id="interaction-long-input"] .user-text-toggle').waitFor();
  // 窄屏程序聚焦不会展开工具栏；用真实触摸打开，并等无模型提示完成，再测热操作。
  const input=page.getByRole('textbox',{name:'消息',exact:true});assert.equal(await input.inputValue(),'');if(width===390)await input.tap();else await input.focus();
  await page.locator('[data-slot="composer.toolbar"] .models-capsule-settings').waitFor({state:'visible'});
  await page.locator('.chat-send-notice').waitFor({state:'visible'});await input.evaluate(e=>e.blur());
  await page.evaluate(()=>document.fonts.ready);await page.waitForTimeout(21000);await page.waitForLoadState('networkidle');
  report.warmup??=[];report.warmup.push({width,cpu,variant,toolbarReady:true,sendNotice:await page.locator('.chat-send-notice').innerText()});save();
  pages[variant]={context,page,cdp,input,errors};}
  for(const profile of ['lan','weak']){
   for(const {cdp} of Object.values(pages))await cdp.send('Network.emulateNetworkConditions',{offline:false,latency:profile==='lan'?20:800,downloadThroughput:profile==='lan'?12500000:125000,uploadThroughput:profile==='lan'?12500000:125000});
   for(let round=0;round<rounds;round++)for(const variant of(round%2?['candidate','baseline']:['baseline','candidate'])){const {page,cdp,input,errors}=pages[variant];current=page;await page.bringToFront();for(const [action,kind] of [['reveal','input'],['expand','input'],['collapse','input'],['quote','input'],['cancel','input'],['quote','output'],['cancel','output'],['edit','input'],['clear','input'],['copy','input'],['copy','output'],['hide','input']].filter(([action])=>width===390||!['reveal','hide'].includes(action))){
    const id='interaction-long-'+kind,text=oracle[id],rowElement=page.locator(`[data-message-id="${id}"]`);let control;
    if(action==='reveal'||action==='hide')control=rowElement.locator('.user-text-clamp');else if(action==='clear')control=input;else if(action==='cancel')control=page.getByRole('button',{name:'取消引用',exact:true});else control=action==='expand'||action==='collapse'?rowElement.locator('.user-text-toggle'):rowElement.getByRole('button',{name:action==='quote'?'引用此消息':action==='edit'?'放回输入框编辑':'复制消息',exact:true});
    await control.evaluate(e=>e.scrollIntoView({block:'center',behavior:'instant'}));if(width===1440)await control.hover();await page.waitForLoadState('networkidle');
    const row={width,cpu,profile,round,variant,action,kind,success:false,requests:[]};report.samples.push(row);const requests=new Map(),start=performance.now();const req=r=>{const item={path:new URL(r.url()).pathname,method:r.method(),query:r.method()==='POST'?r.postDataJSON()?.method:undefined,startMs:performance.now()-start};row.requests.push(item);requests.set(r,item);};const end=async r=>{const item=requests.get(r);if(item){item.endMs=performance.now()-start;item.status=(await r.response())?.status();}};const fail=r=>{const item=requests.get(r);if(item)item.error=r.failure()?.errorText;};page.on('request',req);page.on('requestfinished',end);page.on('requestfailed',fail);const before=await cdp.send('Performance.getMetrics');
    try{await arm(page,action,id,text);if(action==='clear')await input.fill('');else if(width===390)await control.tap();else await control.click();Object.assign(row,await page.evaluate(()=>window.__messageResult));await page.waitForLoadState('networkidle');assert.equal(errors.length,0,errors.join('\n'));assert.ok(!row.requests.some(r=>r.status>=400||r.error));assert.ok(!row.requests.some(r=>r.method!=='GET'&&r.path!=='/api/chat/plugin-ui/query'),'unexpected write');assert.ok(!row.requests.some(r=>/\.(js|css|woff2)$/.test(r.path)),'热路径加载了制品');row.geometry=await page.evaluate(()=>Object.fromEntries(['.conversation-content','.desktop-scroll-return','.composer-wrap'].map(selector=>{
      const element=document.querySelector(selector);
      return [selector,element?{rect:element.getBoundingClientRect().toJSON(),paddingBottom:getComputedStyle(element).paddingBottom,bottom:getComputedStyle(element).bottom}:null];
    })));row.success=true;}
    catch(e){row.diagnostic=await page.evaluate(()=>window.__messageDiagnostic);throw e;}
    finally{const after=await cdp.send('Performance.getMetrics');row.metrics=Object.fromEntries(after.metrics.filter(m=>['LayoutDuration','RecalcStyleDuration','ScriptDuration'].includes(m.name)).map(m=>[m.name,m.value-before.metrics.find(b=>b.name===m.name).value]));page.off('request',req);page.off('requestfinished',end);page.off('requestfailed',fail);save();}
   }}
  }
  for(const [variant,{page,context}] of Object.entries(pages)){await page.screenshot({path:`${output}/${width}-${cpu}-${variant}.png`});await context.close();}
 }
 assert.equal(report.samples.length,report.plannedSamples);
 report.complete=true;
}catch(error){report.error=String(error);process.exitCode=1;if(current&&!current.isClosed())await current.screenshot({path:output+'/failure.png'});}
finally{await browser.close();try{await install('baseline');report.restoredBaseline=true;}catch(error){report.complete=false;report.restoreError=String(error);process.exitCode=1;}save();}
