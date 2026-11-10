import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
import {chromium} from 'playwright-core';
const out=process.env.EVIDENCE_DIR;assert.ok(out);mkdirSync(out,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.CHROMIUM_PATH??'/usr/bin/chromium',headless:true});
const results=[];let seed=20261029;const random=()=>((seed=(Math.imul(seed,1664525)+1013904223)>>>0)/2**32);
try{
for(const width of [1440,390]){
 const context=await browser.newContext({storageState:process.env.AUTH_STATE,viewport:{width,height:900}});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(process.env.APP_ORIGIN??'http://127.0.0.1:44183/');
 if(width<800)await page.getByRole('button',{name:'打开导航',exact:true}).click();
 await page.getByRole('button',{name:/^delegate read-only(?:\d+ 条消息)?$/}).click();
 // 已结束的执行只在会话信息面板里：从说明行最后一项打开面板，展开执行那一行。
 const entry=page.locator('.header-meta button').last();await entry.waitFor();
 const panel=page.locator('.header-meta-panel');
 for(let n=0;n<100;n++){
  const action=Math.floor(random()*4);
  if(action===0)await entry.click();
  if(action===1&&await panel.isVisible())await panel.locator('.header-meta-node__row').last().click();
  if(action===2)await page.keyboard.press('Tab');
  if(action===3)await page.keyboard.press('Escape');
 }
 if(!await panel.isVisible())await entry.click();
 await panel.locator('.header-meta-node__row').last().click();
 assert.match(await panel.innerText(),/已完成：中文碎包/);
 const box=await panel.boundingBox();assert.ok(box.x>=0&&box.x+box.width<=width,JSON.stringify(box));
 await page.screenshot({path:out+`/application-${width}.png`});
 assert.deepEqual(errors,[]);results.push({width,actions:100,realHttp:true,realWebSocket:true,errors});await context.close();
}
writeFileSync(out+'/results.json',JSON.stringify({seed:20261029,externalPeer:'simulated',results},null,2));console.log(JSON.stringify(results));
}finally{await browser.close()}
