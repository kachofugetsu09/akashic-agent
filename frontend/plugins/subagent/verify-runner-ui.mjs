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
writeFileSync(out+'/index.html','<html><head><link rel="stylesheet" href="/bundle.css"></head><body style="margin:20px"><div id="root"></div><script type="module" src="/bundle.js"></script></body></html>');
await build({stdin:{contents:`import definition from './frontend/plugins/subagent/src/runner-ui.tsx';
let listener; window.stopCalls=[]; window.publish=value=>listener?.(value);
const context={subscribe(fn){listener=fn;return()=>{listener=null}},async query(method,payload){window.stopCalls.push(payload);return {accepted:true}}};
definition.slots['header.end'].mount(document.getElementById('root'),context);`,resolveDir:repo,loader:'tsx'},bundle:true,format:'esm',jsx:'automatic',outfile:out+'/bundle.js',plugins:[{name:'public-render',setup(b){b.onResolve({filter:/^@akashic\/chat-ui-v1$/},()=>({path:'ui',namespace:'shim'}));b.onLoad({filter:/.*/,namespace:'shim'},()=>({contents:"import {createRoot} from 'react-dom/client'; export function render(host,node){const root=createRoot(host);root.render(node);return ()=>root.unmount()}",resolveDir:repo}));}}]});
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
 await page.getByRole('button',{name:'代理任务 1'}).click();await page.locator('summary').click();
 await page.getByRole('button',{name:'停止任务',exact:true}).click();
 assert.deepEqual(await page.evaluate(()=>window.stopCalls),[{id:'a'}]);
 for(let i=0;i<150;i++){
  const action=Math.floor(random()*5);
  if(action===0)await page.getByRole('button',{name:'代理任务 1'}).click();
  if(action===1&&await page.locator('summary').isVisible())await page.locator('summary').click();
  if(action===2)await page.evaluate(({run,i})=>window.publish({runs:[{...run,status:i%2?'running':'completed',result:'结果 '+i,activities:[{kind:'message',text:'长文本'.repeat(i+1)}]}]}),{run,i});
  if(action===3)await page.keyboard.press('Tab');
  if(action===4)await page.keyboard.press('Escape');
 }
 if(!await page.locator('summary').isVisible())await page.getByRole('button',{name:'代理任务 1'}).click();
 await page.locator('details').evaluate(e=>e.open=true);
 const box=await page.locator('.agent-runs__panel').boundingBox();assert.ok(box.x>=0&&box.x+box.width<=width,JSON.stringify(box));
 await page.screenshot({path:out+`/panel-${width}.png`});
 await page.evaluate(()=>window.publish({runs:[]}));await page.getByRole('button',{name:'代理任务 1'}).waitFor({state:'hidden'});
 assert.deepEqual(errors,[]);results.push({width,actions:150,errors});await page.close();
}
writeFileSync(out+'/results.json',JSON.stringify({seed:20261029,scope:'real component; simulated UI transport',results},null,2));
console.log(JSON.stringify(results));
}finally{await browser.close();await server.close();}
