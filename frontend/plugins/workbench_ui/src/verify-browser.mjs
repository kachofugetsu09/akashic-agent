import { build } from 'esbuild';
import { chromium } from 'playwright-core';
import {existsSync,mkdtempSync,readFileSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createServer} from 'node:http';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';

// 运行：node frontend/plugins/workbench_ui/src/verify-browser.mjs
const repoRoot = fileURLToPath(new URL('../../../../', import.meta.url));
const executablePath = process.env.AKASHIC_PERF_CHROMIUM
  || ['/usr/bin/chromium','/usr/bin/chromium-browser','/usr/bin/google-chrome'].find(existsSync);
if (!executablePath) throw new Error('请设置 AKASHIC_PERF_CHROMIUM 指向隔离测试使用的 Chromium');
const root=mkdtempSync(join(tmpdir(),'workbench-panel-check-'));
const source = `
import {activate} from './frontend/plugins/workbench_ui/src/index.tsx';
const pending=new Map();
window.fixture={calls:[],batches:[],resolve:(key,value)=>pending.get(key).resolve(value),reject:(key)=>pending.get(key).reject(new Error('late '+key))};
const wait=(key)=>new Promise((resolve,reject)=>pending.set(key,{resolve,reject}));
const rows=[{id:'slow',text:'one'},{id:'error',text:'two'},{id:'fine',text:'three'}];
const panel={id:'fixture',label:'扩展面板',rowKey:'id',pageSize:2,columns:[{key:'id',label:'记录',width:120,renderCell:v=>'<b data-cell>'+v+'</b>'},{key:'text',label:'正文',sortable:true,flex:true}],getCount:async()=>3,
fetchPage:async opts=>{window.fixture.calls.push({...opts,signal:undefined});if(opts.filters.scope==='slow')return wait('page');return {items:opts.filters.scope==='fast'?[rows[2]]:rows.slice((opts.page-1)*2,opts.page*2),total:opts.filters.scope==='fast'?1:3}},
fetchDetail:async row=>{if(row.id==='slow')return wait('detail');if(row.id==='error')throw new Error('detail failed');return row},
renderDetail:(row,host)=>{host.textContent='DETAIL '+row.id},
renderFilters:(host,dispatch)=>{host.replaceChildren();for(const value of ['slow','fast','all']){const b=document.createElement('button');b.textContent='filter '+value;b.onclick=()=>dispatch.setFilter('scope',value);host.append(b)}return()=>host.replaceChildren()},
renderNavBody:host=>{host.textContent='导航扩展'},renderTopbarAction:(host,dispatch)=>{const b=document.createElement('button');b.textContent='插件刷新';b.onclick=()=>dispatch.refresh();host.append(b);return()=>host.replaceChildren()},
batchActions:[{label:'处理选中',className:'',run:async ids=>window.fixture.batches.push(ids)}]};
activate({http:{request:async()=>new Response(JSON.stringify({items:[],total:0,next_cursor:null}))},ui:{inject:(_,register)=>register({register:entry=>entry.render(document.getElementById('root'),{child:()=>({entries:[panel],style:()=>()=>{}})})})}});
`;
await build({stdin:{contents:source,resolveDir:repoRoot,sourcefile:'fixture.js'},alias:{'@akashic/web-ui-v1':join(repoRoot,'frontend/theme/src/material-react.tsx')},bundle:true,format:'esm',outfile:join(root,'bundle.js'),define:{'process.env.NODE_ENV':'"production"'}});
const server=createServer((req,res)=>{if(req.url==='/'){res.setHeader('Content-Type','text/html');res.end('<!doctype html><html><head><link rel="stylesheet" href="/bundle.css"></head><body><div id="root"></div><script type="module" src="/bundle.js"></script></body></html>')}else if(!['/bundle.js','/bundle.css'].includes(req.url)){res.writeHead(404);res.end()}else{res.setHeader('Content-Type',req.url.endsWith('.css')?'text/css':'text/javascript');res.end(readFileSync(join(root,req.url.slice(1))))}});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
const browser=await chromium.launch({executablePath,headless:true,args:['--no-sandbox']});
const page=await browser.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
try{
await page.goto('http://127.0.0.1:'+server.address().port);
await page.getByRole('button',{name:'扩展面板',exact:true}).click();
await page.locator('b[data-cell]').filter({hasText:'slow'}).waitFor();
assert.equal(await page.getByRole('button',{name:'记录',exact:true}).count(),0);
const widths = await page.locator('tbody tr').first().locator('td').evaluateAll(cells=>cells.map(cell=>cell.getBoundingClientRect().width));
assert.equal(widths[1],120);
assert.ok(widths[2]>widths[1]);
await page.getByText('导航扩展',{exact:true}).waitFor();
await page.getByRole('checkbox',{name:'选择 slow',exact:true}).check();
await page.getByRole('button',{name:'slow',exact:true}).click();
await page.getByText('正在读取…',{exact:true}).waitFor();
await page.getByRole('button',{name:'filter fast',exact:true}).click();
await page.getByRole('button',{name:'fine',exact:true}).waitFor();
assert.equal(await page.getByRole('button',{name:'处理选中',exact:true}).isEnabled(),false);
await page.evaluate(()=>window.fixture.reject('detail'));
await page.getByRole('button',{name:'fine',exact:true}).click();
await page.getByText('DETAIL fine',{exact:true}).waitFor();
assert.equal(await page.getByRole('alert').count(),0);
await page.getByRole('button',{name:'关闭详情',exact:true}).click();
await page.getByRole('button',{name:'filter all',exact:true}).click();
await page.getByRole('button',{name:'error',exact:true}).click();
await page.getByRole('alert').filter({hasText:'detail failed'}).waitFor();
assert.equal(await page.getByText('正在读取…',{exact:true}).count(),0);
await page.getByRole('button',{name:'下一页',exact:true}).click();
await page.getByRole('button',{name:'fine',exact:true}).waitFor();
assert.equal(await page.getByRole('alert').count(),0);
await page.getByRole('button',{name:'filter slow',exact:true}).click();
await page.waitForFunction(()=>window.fixture.calls.some(x=>x.filters.scope==='slow'));
await page.getByRole('button',{name:'filter fast',exact:true}).click();
await page.waitForFunction(()=>document.querySelector('tbody')?.textContent.includes('fine'));
await page.evaluate(()=>window.fixture.reject('page'));
await page.getByRole('button',{name:'正文',exact:true}).click();
await page.waitForFunction(()=>window.fixture.calls.at(-1).sortBy==='text');
assert.equal(await page.getByRole('alert').count(),0);
await page.getByRole('checkbox',{name:'选择 fine',exact:true}).check();
page.once('dialog',d=>d.accept());
await page.getByRole('button',{name:'处理选中',exact:true}).click();
await page.waitForFunction(()=>window.fixture.batches.length===1);
assert.deepEqual(await page.evaluate(()=>window.fixture.batches),[['fine']]);
assert.equal(errors.length,0);
console.log(JSON.stringify({passed:true,calls:await page.evaluate(()=>window.fixture.calls),errors,root},null,2));
}finally{await browser.close();await new Promise(resolve=>server.close(resolve));rmSync(root,{recursive:true,force:true})}
