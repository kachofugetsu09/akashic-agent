/** Chromium acceptance: pass the chat harness build directory as the first argument. */
import assert from 'node:assert/strict';
import {chromium} from 'playwright-core';
import {startDesktopFixtureServer} from './desktop-fixture-server.mjs';
import {desktopSessions} from './fixtures.mjs';
const fixture=await startDesktopFixtureServer(process.argv[2]);
const browser=await chromium.launch({executablePath:'/usr/bin/chromium',headless:true});
try {
 const page=await browser.newPage({viewport:{width:1365,height:860}});
 await page.addInitScript(()=>{window.heldIdle=[];window.requestIdleCallback=fn=>window.heldIdle.push(fn);window.cancelIdleCallback=()=>{};});
 const rows=desktopSessions().items;
 await page.route('**/api/chat/sessions?*',route=>{
 const last=new URL(route.request().url()).searchParams.has('after_key');
 return route.fulfill({json:{items:[rows[last?1:0]],next_cursor:last?null:{updated_at:rows[0].updated_at,session_id:rows[0].key}}});
 });
 await page.goto(fixture.origin);
 await page.getByText('纯文本性能会话',{exact:true}).first().waitFor();
 await page.evaluate(()=>{for(const fn of window.heldIdle.splice(0))fn({didTimeout:false,timeRemaining:()=>50});});
 await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
 assert.ok(await page.getByText('纯文本性能会话',{exact:true}).count(),'late first-page callback must preserve second page');
 console.log('PASS: full catalog survives delayed first-page publication');
}finally{await browser.close();await fixture.close();}
