import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import {build} from 'esbuild';
import {chromium} from 'playwright-core';
import {startDesktopFixtureServer} from '../../../scripts/webui-performance/desktop-fixture-server.mjs';
const repo=resolve(dirname(fileURLToPath(import.meta.url)),'../../..');
const out=resolve(process.env.EVIDENCE_DIR ?? 'node_modules/.cache/runner-monkey');
mkdirSync(out,{recursive:true});
writeFileSync(out+'/index.html','<html><head><link rel="stylesheet" href="/bundle.css"></head><body style="margin:20px"><div id="root"></div><div id="detail" style="max-width:26rem"></div><script type="module" src="/bundle.js"></script></body></html>');
await build({stdin:{contents:`import definition from './frontend/plugins/subagent/src/runner-ui.tsx';
let listener; window.stopCalls=[]; window.items=[]; window.publish=value=>listener?.(value);
// 宿主替身：说明项记在 window.items，展开某项就把它的 detail 挂进 #detail。
window.expand=id=>{const host=document.getElementById('detail');window.unmount?.();host.replaceChildren();const item=window.items.find(i=>i.id===id);window.unmount=item?.detail?.(host)};
const context={subscribe(fn){listener=fn;return()=>{listener=null}},async query(method,payload){window.stopCalls.push(payload);return {accepted:true}},
  meta:{publish(items){window.items=items},closePanel(){}}};
definition.slots['header.meta'].mount(document.getElementById('root'),context);`,resolveDir:repo,loader:'tsx'},bundle:true,format:'esm',jsx:'automatic',outfile:out+'/bundle.js',plugins:[{name:'public-render',setup(b){b.onResolve({filter:/^@akashic\/chat-ui-v1$/},()=>({path:'ui',namespace:'shim'}));b.onLoad({filter:/.*/,namespace:'shim'},()=>({contents:"import {createRoot} from 'react-dom/client'; export function render(host,node){const root=createRoot(host);root.render(node);return ()=>root.unmount()}",resolveDir:repo}));}}]});
const server=await startDesktopFixtureServer(out);
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH??'/usr/bin/chromium',headless:true});
let seed=20261029;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
const results=[];
try{
for(const width of [390,1440]){
 const page=await browser.newPage({viewport:{width,height:850}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(server.origin);await page.waitForFunction(()=>typeof window.publish==='function');
 await page.waitForTimeout(100);
 const run={id:'a',label:'检查源码 <script>alert(1)</script>',runner:'codex',status:'running',result:'',progress:true,activities:[{kind:'tool',text:'Read · 开始'}]};
 await page.evaluate(r=>window.publish({runs:[r]}),run);
 // 1. 在跑：说明行一项（进行中）加一个子项，子项展开后能停止。
 const items=await page.evaluate(()=>window.items.map(({detail,...rest})=>rest));
 assert.deepEqual(items.map(i=>[i.id,i.parent??null,i.tone,i.text??null]),[['runs',null,'live','代理 1 运行中'],['run:a','runs','live',null]]);
 await page.evaluate(()=>window.expand('run:a'));
 await page.getByRole('button',{name:'停止任务',exact:true}).click();
 await page.waitForFunction(()=>window.stopCalls.length===1);
 assert.deepEqual(await page.evaluate(()=>window.stopCalls),[{id:'a'}]);
 // 2. 推送乱序变化：展开的详情跟着最新一帧，标签按文本显示不当作 HTML。
 for(let i=0;i<150;i++){
  const action=Math.floor(random()*3);
  if(action===0)await page.evaluate(({run,i})=>window.publish({runs:[{...run,status:i%2?'running':'completed',result:'结果 '+i,activities:[{kind:'message',text:'长文本'.repeat(i+1)}]}]}),{run,i});
  if(action===1)await page.evaluate(()=>window.expand('run:a'));
  if(action===2)await page.keyboard.press('Tab');
 }
 await page.evaluate(r=>window.publish({runs:[{...r,status:'failed',result:'失败原因'}]}),run);
 await page.evaluate(()=>window.expand('run:a'));
 assert.match(await page.locator('#detail').innerText(),/失败原因/);
 assert.equal(await page.evaluate(()=>window.items.find(i=>i.id==='run:a').tone),'alert');
 assert.equal(await page.evaluate(()=>window.items.find(i=>i.id==='runs').text),undefined);
 assert.equal(await page.locator('script').count(),1);
 await page.screenshot({path:out+`/detail-${width}.png`});
 // 3. 没有执行时撤回全部说明项。
 await page.evaluate(()=>window.publish({runs:[]}));
 assert.deepEqual(await page.evaluate(()=>window.items),[]);
 assert.deepEqual(errors,[]);results.push({width,actions:150,errors});await page.close();
}
writeFileSync(out+'/results.json',JSON.stringify({seed:20261029,scope:'real component; simulated UI transport',results},null,2));
console.log(JSON.stringify(results));
}finally{await browser.close();await server.close();}
